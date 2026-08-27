from app.agent.llm_adapter import LLMAdapter, LLMResponse
from app.agent.state import SessionState
from app.pipeline import run_pipeline_from_text
from app.tts.base import SynthesisResult, TTSAdapter


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


def test_run_pipeline_from_text_skips_stt():
    session = SessionState(session_id="text-test")
    result = run_pipeline_from_text("Hi, I want to track my order.", session, FakeLLM(), FakeTTS())

    assert result.transcript == "Hi, I want to track my order."
    assert result.stt_language is None
    assert result.language_state.primary_language == "en"
    assert result.reply_text == "Sure, what's your order number?"
    assert len(session.history) == 2
