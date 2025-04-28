# src/detector.py

import joblib
import os

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Load model and vectorizer once when this file is imported
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_prompt(prompt: str) -> str:
    # Vectorize prompt
    prompt_vec = vectorizer.transform([prompt])

    # Predict
    prediction = model.predict(prompt_vec)[0]

    return prediction
