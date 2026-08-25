from app.language.detector import detect_language_state


def test_pure_english_detected():
    state = detect_language_state("Hi, I want to track my order.")
    assert state.primary_language == "en"
    assert state.code_switch_score < 0.3


def test_native_urdu_script_detected():
    state = detect_language_state("میرا اوڈر کہا ہے؟")
    assert state.primary_language == "ur"
    assert state.script == "arabic"


def test_code_switched_roman_urdu_detected():
    state = detect_language_state("Mera order dispatch hua ya nahi?")
    assert state.primary_language == "mixed"
    assert state.code_switch_score > 0.3
    assert "ur-Latn" in state.secondary_languages
