from dataclasses import dataclass, field


@dataclass
class LanguageState:
    """Result of language/script/code-switch analysis for a piece of text."""

    primary_language: str          # en | ur | ur-Latn | mixed | unknown
    secondary_languages: list[str] = field(default_factory=list)
    script: str = "unknown"        # arabic | latin | mixed | unknown
    code_switch_score: float = 0.0  # 0.0 = monolingual, 1.0 = heavily mixed
    confidence: float = 0.0
