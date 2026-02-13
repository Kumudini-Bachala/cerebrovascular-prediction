import os
import uuid
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import random

# --- Load Environment Variables ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(APP_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_GEMINI_MODELS = [
    'gemini-1.5-pro-latest',
    'gemini-1.5-flash-latest',
    'gemini-1.0-pro',
]
AVAILABLE_GEMINI_MODELS = DEFAULT_GEMINI_MODELS.copy()

def refresh_available_gemini_models():
    """Populate AVAILABLE_GEMINI_MODELS with options supported by generateContent."""
    global AVAILABLE_GEMINI_MODELS
    if not GEMINI_API_KEY:
        return
    try:
        available_models = genai.list_models()
        compatible_models = []
        for model_info in available_models:
            supported_methods = getattr(model_info, "supported_generation_methods", []) or []
            if "generateContent" not in supported_methods:
                continue
            model_name = getattr(model_info, "name", "")
            if model_name.startswith("models/"):
                model_name = model_name.split("models/")[1]
            if model_name:
                compatible_models.append(model_name)
        if compatible_models:
            AVAILABLE_GEMINI_MODELS = compatible_models
            print(f"Detected compatible Gemini models: {AVAILABLE_GEMINI_MODELS}")
        else:
            AVAILABLE_GEMINI_MODELS = DEFAULT_GEMINI_MODELS
            print("No compatible Gemini models detected from API. Falling back to defaults.")
    except Exception as e:
        AVAILABLE_GEMINI_MODELS = DEFAULT_GEMINI_MODELS
        print(f"Unable to list Gemini models ({e}). Falling back to defaults.")

# --- Configure Gemini API ---
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
        refresh_available_gemini_models()
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY = None # Disable if configuration fails
else:
    print("Warning: GEMINI_API_KEY not found in .env file. LLM features will be disabled.")

app = Flask(__name__, static_url_path='/static')

# --- Configuration & Robust Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(APP_DIR, 'brain_stroke_final_model.h5')
TABULAR_MODEL_PATH = os.path.join(APP_DIR, 'stroke_risk_model.joblib')
UPLOAD_FOLDER = os.path.join(APP_DIR, 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_NAMES = ['Normal', 'Stroke']

# --- Model Loading ---
image_model = None
tabular_model = None

print(f"Attempting to load image model from: {IMAGE_MODEL_PATH}")
try:
    if os.path.exists(IMAGE_MODEL_PATH):
        image_model = load_model(IMAGE_MODEL_PATH)
        print("Image model loaded successfully.")
    else:
        print("Image model file not found.")
except Exception as e:
    print(f"Error loading image model: {e}")

print(f"Attempting to load tabular model from: {TABULAR_MODEL_PATH}")
try:
    if os.path.exists(TABULAR_MODEL_PATH):
        tabular_model = joblib.load(TABULAR_MODEL_PATH)
        print("Tabular model loaded successfully.")
    else:
        print("Tabular model file not found.")
except Exception as e:
    print(f"Error loading tabular model: {e}")

def get_image_prediction(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224), color_mode='rgb')
        x = img_to_array(img).astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        preds = image_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)}
        return label, probabilities
    except Exception as e:
        import traceback
        print(f"[ERROR] An exception occurred in get_image_prediction: {str(e)}")
        traceback.print_exc()
        raise

def get_llm_response(prompt):
    """Generates a response from Gemini, trying multiple models as fallbacks."""
    if not GEMINI_API_KEY:
        return "LLM features are disabled. Please configure the Gemini API key."

    models_to_try = AVAILABLE_GEMINI_MODELS
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"Attempting to use model: {model_name}")
            model = genai.GenerativeModel(model_name)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)
            print(f"Successfully generated content with {model_name}")
            return response.text
        except Exception as e:
            print(f"An error occurred with model {model_name}: {e}")
            last_error = e
            continue

    return f"Error generating response from LLM after trying all models: {last_error}"


# === THIS IS THE UPDATED FUNCTION ===
def format_llm_response_to_html(text):
    """
    Converts a simple text format (headings with colons, lists with hyphens)
    into clean, semantic HTML.
    """
    # Handles both actual newlines and escaped '\n' from the LLM
    lines = text.replace('\\n', '\n').split('\n')
    html_lines = []
    in_list = False # Flag to track if we are inside a <ul> tag

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line is a heading (e.g., "Smoking:")
        if line.endswith(':'):
            if in_list:
                html_lines.append('</ul>') # Close list before a new heading
                in_list = False
            # Use <h4> for a bold, semantic heading
            html_lines.append(f'<h4>{line}</h4>')
        
        # Check if the line is a list item (e.g., "- Quit smoking entirely.")
        elif line.startswith('-'):
            if not in_list:
                html_lines.append('<ul>') # Start a new list
                in_list = True
            # Add the list item, removing the leading '- '
            html_lines.append(f'<li>{line[1:].strip()}</li>')

        # Otherwise, treat it as a standard paragraph
        else:
            if in_list:
                html_lines.append('</ul>') # Close list before a paragraph
                in_list = False
            html_lines.append(f'<p>{line}</p>')
    
    # After the loop, close any list that might still be open
    if in_list:
        html_lines.append('</ul>')

    return "".join(html_lines)
# === END OF UPDATED FUNCTION ===


# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/patient_form', methods=['GET'])
def patient_form():
    return render_template('patient_form.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if image_model is None:
        return jsonify({'error': 'Image model not loaded. Check server logs for details.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    if file:
        try:
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + os.path.splitext(filename)[1]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            label, probabilities = get_image_prediction(file_path)
            return jsonify({'prediction': label, 'probabilities': probabilities})
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file.'}), 400

@app.route('/predict_stroke_risk', methods=['POST'])
def predict_stroke_risk():
    if tabular_model is None:
        return jsonify({'error': 'Tabular model not loaded. Check server logs for details.'}), 500
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Check .env file and server logs.'}), 500

    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction_proba = tabular_model.predict_proba(df)[0][1]

        if prediction_proba < 0.15:
            risk_category = 'Low'
        elif prediction_proba < 0.30:
            risk_category = 'Medium'
        else:
            risk_category = 'High'
        
        prompt = "Hello! I'm here to talk to you about stroke risk and how we can work together to keep you healthy. "
        prompt += f"Based on your information, you have a {risk_category.lower()} risk of stroke. "
        prompt += "Let's discuss what that means and what steps we can take.\n\n"
        prompt += f"You are a health assistant. A patient has a {risk_category.lower()} risk of stroke. Their details are: "
        prompt += f"Age: {data['age']}, Gender: {data['gender'].lower()}, BMI: {data['bmi']}, Glucose: {data['avg_glucose_level']}, "
        prompt += f"Hypertension: {'Yes' if int(data['hypertension']) == 1 else 'No'}, Heart Disease: {'Yes' if int(data['heart_disease']) == 1 else 'No'}, "
        prompt += f"Smoking: {data['smoking_status']}. "
        
        if risk_category == 'High':
            prompt += "**Crucially, advise the patient to consult a doctor immediately for a comprehensive assessment.** "
        
        prompt += "Generate a concise, easy-to-read summary of recommendations. "
        prompt += "Use plain text for headings (e.g., 'Diet Plan:') and bullet points (e.g., '- Item'). "
        prompt += "Do NOT use '###' or '*'. Keep the entire response under 150 words. Focus only on the most critical advice."

        raw_precautions = get_llm_response(prompt)
        precautions = format_llm_response_to_html(raw_precautions)

        response_data = {
            'risk_category': risk_category,
            'probability': round(prediction_proba * 100, 2),
            'precautions': precautions
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Error in get_llm_response: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)