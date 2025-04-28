# src/trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Paths
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def train_and_save_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['prompt'], df['label'], test_size=0.2, random_state=42
    )

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate
    accuracy = model.score(X_test_vec, y_test)
    print(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    train_and_save_model()
