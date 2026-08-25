from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    text: str
    language: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionState:
    """Conversation-level state, independent from any particular LLM (spec 5.5)."""

    session_id: str
    history: list[Turn] = field(default_factory=list)
    current_language: str = "unknown"
    recent_languages: list[str] = field(default_factory=list)
    code_switching_enabled: bool = True
    extracted_entities: dict = field(default_factory=dict)
    pending_action: str | None = None

    def add_user_turn(self, text: str, language: str) -> None:
        self.history.append(Turn(role="user", text=text, language=language))
        self._update_language(language)

    def add_assistant_turn(self, text: str, language: str) -> None:
        self.history.append(Turn(role="assistant", text=text, language=language))

    def _update_language(self, language: str) -> None:
        self.current_language = language
        self.recent_languages.append(language)
        # Keep only the last 5 for lightweight "recent transitions" tracking
        self.recent_languages = self.recent_languages[-5:]

    def as_message_history(self, max_turns: int = 6) -> list[dict]:
        """Return recent turns formatted for an LLM chat API."""
        recent = self.history[-max_turns:]
        return [{"role": t.role, "content": t.text} for t in recent]
