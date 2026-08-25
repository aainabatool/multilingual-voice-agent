from app.agent.graph import run_turn
from app.agent.llm_adapter import OllamaAdapter
from app.agent.state import SessionState
from app.language.detector import detect_language_state

TURNS = [
    "Hi, I want to track my order. The order number is 4589.",
    "Mera order dispatch hua ya nahi?",
]

def main():
    llm = OllamaAdapter(model_name="llama3.2")
    session = SessionState(session_id="test-session-1")

    for text in TURNS:
        lang_state = detect_language_state(text)
        print(f"\nUser ({lang_state.primary_language}): {text}")
        reply = run_turn(session, text, lang_state, llm)
        print(f"Agent: {reply}")

if __name__ == "__main__":
    main()
