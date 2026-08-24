from app.stt.whisper_runner import WhisperRunner


def test_transcribe_english_sample():
    runner = WhisperRunner(model_size="tiny")
    result = runner.transcribe("data/audio/sample_en.mp3", language="en")
    assert "order" in result.text.lower()
    assert result.language == "en"
