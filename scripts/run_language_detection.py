from app.language.detector import detect_language_state

EXAMPLES = [
    "Hi, I want to track my order. The order number is 4589.",
    "میرا اوڈر کہا ہے؟ اوڈر نمبر 4 5 8 9 ہے",
    "Mujhe apna order track karna hai. Order number 4589 hai.",
    "Mera order dispatch hua ya nahi?",  # spec's own example, section 5.4
]

def main():
    for text in EXAMPLES:
        state = detect_language_state(text)
        print(f"\nText: {text}")
        print(f"  primary_language: {state.primary_language}")
        print(f"  secondary_languages: {state.secondary_languages}")
        print(f"  script: {state.script}")
        print(f"  code_switch_score: {state.code_switch_score}")
        print(f"  confidence: {state.confidence}")

if __name__ == "__main__":
    main()
