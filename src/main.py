from src.prompt_loader import load_prompts
from src.detector import predict_prompt

def test_prompts(prompts, label):
    print(f"\n🔍 Testing {label} Prompts:")
    for prompt in prompts:
        prediction = predict_prompt(prompt)
        status = {
            "safe": "✅ Safe",
            "injected": "🚨 Injected"
        }.get(prediction, "⚠️ Unknown")

        print(f"➤ {prompt}\n   → {status}")

def main():
    safe_prompts = load_prompts("data/safe_prompts.json")
    injected_prompts = load_prompts("data/injected_prompts.json")

    test_prompts(safe_prompts, "Safe")
    test_prompts(injected_prompts, "Injected")

if __name__ == "__main__":
    main()