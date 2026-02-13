CEREBROVASCULAR PREDICTION

A full-stack web application that predicts stroke risk from medical data and CT images using machine learning models with a Flask-based backend and a simple user interface.

ABOUT:

a.THIS PROJECT ENABLES:

Upload brain CT images to classify if the user is likely to have a stroke or normal.

Submit health attributes (age, BMI, glucose level, blood pressure, etc.) via API to compute a stroke risk score and categorized risk level.

Receive meaningful risk interpretation with LLM-generated health advice (when a Gemini API key is configured).

b.THE BACKEND USES:

A CNN image classification model (brain_stroke_final_model.h5)

A tabular stroke risk model (stroke_risk_model.joblib)

A Flask API to serve both prediction endpoints

FEATURES

1. Image-based stroke classification
2. Tabular stroke risk prediction
3. LLM-assisted personalized health advice
4. Simple HTML frontend + API routes
5. Uploads are stored securely with unique filenames
6. Models loaded automatically if present

PROJECT ARCHITECTURE :

Architecture Overview 
                ┌──────────────────────────┐
                │        User Interface     │
                │ (HTML / CSS / JS Frontend)│
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │      Flask Backend API    │
                │         (app.py)          │
                └─────────────┬────────────┘
                              │
      ┌───────────────────────┼────────────────────────┐
      ▼                       ▼                        ▼
┌──────────────┐     ┌──────────────────┐      ┌──────────────────┐
│ Image Model  │     │ Tabular ML Model │      │ LLM Integration  │
│ (CNN .h5)    │     │ (.joblib)        │      │ (Gemini API)     │
└──────────────┘     └──────────────────┘      └──────────────────┘
      │                       │                        │
      ▼                       ▼                        ▼
  Stroke/Normal         Risk Probability         Personalized Advice
  Prediction            & Risk Category          + Precautions

System Modules:

HTML frontend for:
a.Uploading brain CT images
b.Entering patient medical data

Displays:
a.Stroke classification results
b.Risk percentage
c.LLM generated medical advice

Backend Layer:
Handles:
Routing (/predict, /predict_stroke_risk)
File upload processing
JSON request parsing
Model inference
Response formatting

Technologies:
Flask
NumPy
Pandas
TensorFlow/Keras
Joblib

IMAGE PREDICTION MODEL 
Model:
CNN Model (brain_stroke_final_model.h5)

Workflow:
User uploads CT scan image
Image resized to 224×224
Converted to array
Normalized
Passed to CNN

Output:
Normal
Stroke
Probability scores

Tabular Stroke Risk Prediction Module
Model:
Trained ML classifier (stroke_risk_model.joblib)
Input Features:
Age
Gender
BMI
Hypertension
Heart Disease
Smoking Status
Glucose Level

Output:
Stroke probability (%)
Risk Category:
Low
Medium
High

LLM INTEGRATED MODEL:
If GEMINI_API_KEY is provided:
Sends prediction results to LLM
Generates:
Diet recommendations
Lifestyle precautions
Preventive advice
Formats output in structured HTML 

DATA FLOW ARCHITETCTURE :
Image Prediction Flow
User → Upload Image → Flask → Preprocessing → CNN Model
      → Probability → Classification → JSON Response
Risk Prediction Flow
User → Submit Health Data → Flask → Feature Encoding
      → ML Model → Risk Probability → Risk Category
      → (Optional LLM) → Personalized Advice

DATASETS USED :
Stroke Prediction Tabular Dataset - https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
Synthetic Stroke Prediction Dataset - https://data.mendeley.com/datasets/s2nh6fm925/1
Brain CT Stroke Image Datasets - https://www.kaggle.com/datasets/noshintasnia/brain-stroke-prediction-ct-scan-image-dataset


