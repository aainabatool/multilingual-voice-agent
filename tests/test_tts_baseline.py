from app.tts.piper_runner import PiperAdapter


def test_english_synthesis_native_support():
    tts = PiperAdapter()
    result = tts.synthesize("Hello there.", "en", "data/audio/test_out_en.wav")
    assert result.native_support is True
    assert result.voice_used == "en_US-lessac-medium"
    assert result.audio_duration_s > 0


def test_urdu_synthesis_native_support():
    tts = PiperAdapter()
    result = tts.synthesize("سلام", "ur", "data/audio/test_out_ur.wav")
    assert result.native_support is True
    assert result.voice_used == "ur_PK-fasih-medium"
    assert result.audio_duration_s > 0


def test_roman_urdu_falls_back_to_english_voice():
    tts = PiperAdapter()
    result = tts.synthesize("Mera order kahan hai?", "ur-Latn", "data/audio/test_out_ur_latn.wav")
    assert result.native_support is False
    assert result.voice_used == "en_US-lessac-medium"
