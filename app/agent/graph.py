from app.agent.llm_adapter import LLMAdapter
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import SessionState
from app.language.state import LanguageState


def run_turn(
    session: SessionState,
    user_text: str,
    language_state: LanguageState,
    llm: LLMAdapter,
) -> str:
    """Run one conversation turn: update session, call the LLM with explicit
    language metadata attached, record the response, return the reply text.
    """
    session.add_user_turn(user_text, language_state.primary_language)

    # Attach explicit language metadata to this turn, per spec 5.6, so the
    # model can preserve style instead of defaulting to English.
    annotated_user_message = (
        f"(Detected language: {language_state.primary_language}, "
        f"script: {language_state.script}, "
        f"code-switch score: {language_state.code_switch_score})\n"
        f"{user_text}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.as_message_history()[:-1])  # prior turns, excluding the one we just added
    messages.append({"role": "user", "content": annotated_user_message})

    response = llm.chat(messages)
    session.add_assistant_turn(response.text, language_state.primary_language)

    return response.text
