import streamlit as st
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

# Set Streamlit page config
st.set_page_config(page_title="PromptShield AI 2.0", page_icon="🛡️", layout="centered")

# Load trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Load DistilBERT
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, bert_model = load_bert()

@torch.no_grad()
def encode_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return np.expand_dims(cls_embedding, axis=0)

# Predict with threshold
def predict_prompt_with_confidence(prompt: str):
    embedding = encode_prompt(prompt)
    proba = model.predict_proba(embedding)[0]

    # Get safe class index
    safe_index = model.classes_.tolist().index('safe')
    safe_score = proba[safe_index]

    # Predict class based on threshold
    predicted_class = "safe" if safe_score >= 0.8 else "injected"

    # Return both prediction and confidence
    return predicted_class, safe_score


# UI
st.title("🛡️ PromptShield AI 2.0")
prompt = st.text_area("✍️ Enter a prompt to analyze:", height=150)

if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("🔎 Analyze Prompt"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt first!")
    else:
        with st.spinner("Analyzing..."):
            result, confidence = predict_prompt_with_confidence(prompt)
        st.session_state.history.append((prompt, result, confidence))

        if result == "safe":
            st.success(f"✅ Risk Level: LOW\n\nConfidence: {confidence*100:.2f}%")
        else:
            st.error(f"🚨 Risk Level: HIGH\n\nConfidence: {100 - confidence*100:.2f}%")

st.divider()
st.header("🕘 Prompt History")
for past_prompt, past_result, past_confidence in reversed(st.session_state.history[-5:]):
    if past_result == "safe":
        st.success(f"✅ {past_prompt} ({past_confidence*100:.2f}% confidence)")
    else:
        st.error(f"🚨 {past_prompt} ({(1-past_confidence)*100:.2f}% confidence)")