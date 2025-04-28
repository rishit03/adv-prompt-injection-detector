# 🛡️ Adversarial Prompt Injection Detection System

A lightweight system to detect and block prompt injection attacks in LLM applications — tailored for sensitive environments like banking and finance.

## 💡 Motivation

AI agents are being integrated into high-stakes domains like banking. This system helps prevent prompt injection attacks that manipulate models into ignoring instructions or leaking sensitive info.

## 🔍 What it Does

- Detects adversarial/jailbreak inputs using rules or ML
- Flags and scores suspicious prompts
- Can sanitize or block inputs before reaching backend models

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/adv-prompt-injection-detector.git
cd adv-prompt-injection-detector
pip install -r requirements.txt
streamlit run streamlit_app.py
