from app.language.code_switch import detect_code_switch
from app.language.script_detector import analyze_script
from app.language.state import LanguageState


def detect_language_state(
    text: str,
    stt_language: str | None = None,
    stt_confidence: float | None = None,
) -> LanguageState:
    """Combine script analysis, lexical code-switch heuristics, and STT metadata
    into a single LanguageState, per spec 5.3: language detection is a routing
    signal, not absolute truth -- so we triangulate multiple weak signals.
    """
    script_result = analyze_script(text)
    dominant_script = script_result["dominant"]

    if dominant_script == "arabic":
        # Perso-Arabic script text -- almost certainly Urdu.
        primary = "ur"
        secondary: list[str] = []
        code_switch_score = 0.0
        confidence = 0.9

    elif dominant_script == "mixed":
        # Both scripts present in real volume (e.g. Urdu script text with English words).
        primary = "mixed"
        secondary = ["ur", "en"]
        code_switch_score = max(0.5, script_result["ratios"]["latin"])
        confidence = 0.6

    elif dominant_script == "latin":
        # Latin script -- could be English, Roman Urdu, or a code-switched mix of both.
        cs_result = detect_code_switch(text)
        code_switch_score = cs_result["code_switch_score"]

        if code_switch_score >= 0.3:
            primary = "mixed"
            secondary = ["en", "ur-Latn"]
            confidence = 0.7
        elif cs_result["ur_ratio"] > cs_result["en_ratio"]:
            primary = "ur-Latn"
            secondary = []
            confidence = 0.6
        else:
            primary = "en"
            secondary = []
            confidence = 0.7

    else:
        primary = "unknown"
        secondary = []
        code_switch_score = 0.0
        confidence = 0.0

    # If STT gave us a language guess, use it to nudge confidence -- but never
    # let it override script evidence, since Whisper is known to misdetect
    # Urdu as Hindi (see Phase 1 findings).
    if stt_language and stt_confidence:
        if stt_language == primary or (stt_language == "hi" and primary == "ur"):
            confidence = min(1.0, confidence + 0.1)

    return LanguageState(
        primary_language=primary,
        secondary_languages=secondary,
        script=dominant_script,
        code_switch_score=code_switch_score,
        confidence=round(confidence, 3),
    )
