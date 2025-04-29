# streamlit_app.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # 👈 Add this!
import streamlit as st
import joblib
import streamlit as st
import joblib
from bert_features import encode_prompts  # Import your BERT encoder

# Set Streamlit page config
st.set_page_config(page_title="PromptShield AI 2.0", page_icon="🛡️", layout="centered")

# Title and description
st.title("🛡️ PromptShield AI 2.0 — Adversarial Prompt Injection Detection System")
st.markdown("Detect whether a user prompt is **Safe ✅**, **Medium Risk ⚠️**, or **High Risk 🚨** using a BERT-powered machine learning model.")

# Paths
MODEL_PATH = "models/model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

# Prediction function
# Prediction function with thresholding
def predict_prompt(prompt: str) -> str:
    embedding = encode_prompts([prompt])  # Encode single prompt
    proba = model.predict_proba(embedding)[0]  # Get class probabilities
    
    # Get safe class index
    safe_index = model.classes_.tolist().index('safe')
    safe_score = proba[safe_index]

    # Apply threshold (80% confidence to call it Safe)
    if safe_score >= 0.8:
        return "safe"
    else:
        return "injected"


# Streamlit UI
st.header("🔍 Analyze a Prompt")
prompt = st.text_area("✍️ Enter a prompt to analyze:", height=150)

# Session state to keep prompt history
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
