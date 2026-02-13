import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "AIzaSyD7cyM4oOg5LBJ1mPtFqlFo27E5--LNCUo")

try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content("Hello Gemini, this is a test.")
    print(response.text)
except Exception as e:
    print("Error:", e)
