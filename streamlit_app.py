# streamlit_app.py

import streamlit as st
import joblib
import os

# Set Streamlit page config
st.set_page_config(page_title="Prompt Injection Detector", page_icon="🛡️", layout="centered")

# Title and description
st.title("🛡️ Adversarial Prompt Injection Detection System")
st.markdown("Detect whether a user prompt is **Safe ✅**, **Medium Risk ⚠️**, or **High Risk 🚨** using a trained machine learning model.")

# Paths
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Load model and vectorizer
with st.spinner('Loading Model...'):
    model, vectorizer = load_model_and_vectorizer()

# Prediction function
def predict_prompt(prompt: str) -> str:
    prompt_vec = vectorizer.transform([prompt])
    prediction = model.predict(prompt_vec)[0]
    return prediction

# User input
st.header("🔍 Analyze a Prompt")
prompt = st.text_area("✍️ Enter a prompt to analyze:", height=150)

# Create a session state to keep track of previous prompts
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("🔎 Analyze Prompt"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt first!")
    else:
        with st.spinner('Analyzing...'):
            result = predict_prompt(prompt)
        
        # Record the result
        st.session_state.history.append((prompt, result))
        
        # Show result
        if result == "safe":
            st.success("✅ Risk Level: LOW\n\nThis prompt is safe.")
        elif result == "injected":
            st.error("🚨 Risk Level: HIGH\n\nPrompt injection attempt detected!")
        else:
            st.info("⚠️ Risk Level: MEDIUM\n\nUnable to determine with high confidence.")

# Divider
st.divider()

# Prompt History Section
st.header("🕘 Prompt Analysis History")
if st.session_state.history:
    for past_prompt, past_result in reversed(st.session_state.history[-5:]):  # Show last 5 only
        if past_result == "safe":
            st.success(f"✅ {past_prompt}")
        elif past_result == "injected":
            st.error(f"🚨 {past_prompt}")
        else:
            st.info(f"⚠️ {past_prompt}")
else:
    st.write("No prompts analyzed yet!")