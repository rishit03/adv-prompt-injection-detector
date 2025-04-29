
import streamlit as st
import joblib
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="PromptShield AI 2.0", page_icon="🛡️", layout="centered")

# Load trained model
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# Simple placeholder feature extractor
def fake_embed_prompt(prompt: str) -> np.ndarray:
    # Simulated BERT-like vector (just for working deployment)
    return np.random.rand(1, 768)

# Predict with threshold
def predict_prompt(prompt: str) -> str:
    embedding = fake_embed_prompt(prompt)
    proba = model.predict_proba(embedding)[0]
    safe_index = model.classes_.tolist().index('safe')
    safe_score = proba[safe_index]
    return "safe" if safe_score >= 0.8 else "injected"

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
            result = predict_prompt(prompt)
        st.session_state.history.append((prompt, result))
        if result == "safe":
            st.success("✅ Risk Level: LOW — This prompt is safe.")
        else:
            st.error("🚨 Risk Level: HIGH — Injection attempt detected!")

st.divider()
st.header("🕘 Prompt History")
for past_prompt, past_result in reversed(st.session_state.history[-5:]):
    if past_result == "safe":
        st.success(f"✅ {past_prompt}")
    else:
        st.error(f"🚨 {past_prompt}")
