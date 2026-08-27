from app.agent.llm_adapter import LLMAdapter, LLMResponse
from app.agent.state import SessionState
from app.pipeline import run_pipeline
from app.stt.base import STTAdapter, TranscriptionResult
from app.tts.base import SynthesisResult, TTSAdapter


class FakeSTT(STTAdapter):
    def transcribe(self, audio_path, language=None):
        return TranscriptionResult(text="Hi, I want to track my order.", language="en", confidence=0.95)

    def detect_language(self, audio_path):
        return "en"


class FakeLLM(LLMAdapter):
    def chat(self, messages, temperature=0.2):
        return LLMResponse(text="Sure, what's your order number?", model_name="fake", inference_time_s=0.0)


class FakeTTS(TTSAdapter):
    def synthesize(self, text, language, output_path):
        return SynthesisResult(
            audio_path=output_path, voice_used="fake-voice", language_requested=language,
            native_support=True, audio_duration_s=1.0, inference_time_s=0.0,
        )

    def list_voices(self):
        return {"en": "fake-voice"}


def test_run_pipeline_wires_all_stages():
    session = SessionState(session_id="test")
    result = run_pipeline("fake_audio.wav", session, FakeSTT(), FakeLLM(), FakeTTS())

    assert result.transcript == "Hi, I want to track my order."
    assert result.stt_language == "en"
    assert result.language_state.primary_language == "en"
    assert result.reply_text == "Sure, what's your order number?"
    assert result.tts_result.voice_used == "fake-voice"
    assert len(session.history) == 2
