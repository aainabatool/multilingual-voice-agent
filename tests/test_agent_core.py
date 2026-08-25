from app.agent.graph import run_turn
from app.agent.llm_adapter import LLMAdapter, LLMResponse
from app.agent.state import SessionState
from app.language.detector import detect_language_state


class FakeLLM(LLMAdapter):
    """Deterministic stand-in so tests don't depend on a running Ollama instance."""

    def chat(self, messages: list[dict], temperature: float = 0.2) -> LLMResponse:
        last_user_msg = messages[-1]["content"]
        return LLMResponse(text=f"echo: {last_user_msg}", model_name="fake", inference_time_s=0.0)


def test_session_state_tracks_history_and_language():
    session = SessionState(session_id="s1")
    lang_state = detect_language_state("Hi, I want to track my order.")
    run_turn(session, "Hi, I want to track my order.", lang_state, FakeLLM())

    assert len(session.history) == 2  # user turn + assistant turn
    assert session.current_language == "en"
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"


def test_run_turn_returns_llm_text():
    session = SessionState(session_id="s2")
    lang_state = detect_language_state("Mera order dispatch hua ya nahi?")
    reply = run_turn(session, "Mera order dispatch hua ya nahi?", lang_state, FakeLLM())

    assert reply.startswith("echo:")
