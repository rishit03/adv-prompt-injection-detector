# 🛡️ Adversarial Prompt Injection Detection System

Secure your AI-powered banking and financial applications from prompt injection attacks using this real-time detection system!

---

## 📋 About the Project

With the rise of AI in banking, attackers can manipulate prompts to trick AI systems into making unauthorized transactions or leaking sensitive data.  
This project detects such adversarial prompts **before** they reach the backend AI models.

Built with:
- Machine Learning (Logistic Regression)
- Natural Language Processing (TF-IDF)
- Streamlit Web Interface

---

## ✨ Features

- ✅ Detects Safe, Medium Risk, and High Risk prompts
- 🔥 Real-time analysis via a web app
- 🕘 Prompt history tracking
- 🛡️ Defends AI assistants from prompt injection attacks
- 📚 Expandable with larger datasets and advanced models

---

## 📂 Project Structure

```bash
adv-prompt-injection-detector/
│
├── src/
│   ├── main.py
│   ├── trainer.py
│   ├── detector.py
│   ├── utils.py
│   └── prompt_loader.py
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── data/
│   └── dataset.csv
├── streamlit_app.py
├── test/
├── README.md
├── requirements.txt

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.