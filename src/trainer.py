# src/trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os

from src.bert_features import encode_prompts

# Paths
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"

def train_and_save_best_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    prompts = df['prompt'].tolist()
    labels = df['label'].tolist()

    # Encode prompts using BERT
    print("🔍 Encoding prompts using BERT... (this might take some time)")
    X = encode_prompts(prompts)
    y = labels

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy*100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(f"\n✅ Best Model Selected: {best_model_name} with Accuracy: {best_accuracy*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_and_save_best_model()
