# 🛡️ PromptShield AI 2.0

**Real-time BERT-powered Prompt Injection Detection Firewall**  
Stop malicious prompt manipulation attempts before they hit your AI.

![HuggingFace Spaces](https://img.shields.io/badge/Live-HuggingFace-blue?logo=huggingface)

---

## 📌 About the Project

PromptShield AI is a real-time prompt injection detection system designed for LLM-driven banking, fintech, and chatbot applications.

It uses a custom-trained model on thousands of realistic, adversarial, and gray-zone prompts to detect:
- Polite but malicious prompt injections
- Multi-step logical attack prompts
- Conversational "friendly-looking" bypass attempts
- Obvious threats like OTP bypass, admin escalation, etc.

✅ Built using BERT embeddings  
✅ Tuned with real-world thresholding logic  
✅ Live deployed on HuggingFace Spaces

---

## 🚀 Live Demo

👉 Try it here: [https://huggingface.co/spaces/rishit03/promptshield](https://huggingface.co/spaces/rishit03/promptshield)

---

## ⚙️ Features

- ✅ Real-time prediction (Safe / Injected)
- ✅ Confidence-based thresholding
- ✅ BERT embeddings via DistilBERT
- ✅ Noisy, adversarial, polite prompt detection
- ✅ Streamlit UI for public demo
- ✅ Explainability: keyword triggers shown
- ✅ HuggingFace Spaces compatible

---

## 📂 Folder Structure

```
promptshield/
├── app.py                   # Streamlit app entrypoint
├── models/
│   └── model.pkl            # Trained model
├── src/
│   ├── bert_features.py     # BERT encoder
│   ├── trainer.py           # Model trainer
│   └── __init__.py
├── data/
│   └── dataset.csv # Training data
├── requirements.txt
└── README.md
```

---

## 🧠 How to Run Locally

```bash
git clone https://github.com/rishit03/promptshield.git
cd promptshield

pip install -r requirements.txt

streamlit run app.py
```

---

## 🧪 Training

```bash
python -m src.trainer
```

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [transformers (HuggingFace)](https://huggingface.co/transformers/)
- [torch](https://pytorch.org/)
- [joblib](https://joblib.readthedocs.io/)

---

## 📜 License

MIT License

---

## 🙌 Author

Built with 🔥 by **Rishit Goel**  
(Master's Student @ CSULB)
