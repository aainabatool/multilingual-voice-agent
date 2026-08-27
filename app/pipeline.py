from dataclasses import dataclass

from app.agent.graph import run_turn
from app.agent.llm_adapter import LLMAdapter
from app.agent.state import SessionState
from app.language.detector import detect_language_state
from app.language.router import route_transcription
from app.language.state import LanguageState
from app.stt.whisper_runner import WhisperRunner
from app.tts.base import SynthesisResult
from app.tts.piper_runner import PiperAdapter


@dataclass
class PipelineResult:
    transcript: str
    stt_language: str | None
    stt_confidence: float | None
    language_state: LanguageState
    reply_text: str
    tts_result: SynthesisResult


def run_pipeline(
    audio_path: str,
    session: SessionState,
    stt: WhisperRunner,
    llm: LLMAdapter,
    tts: PiperAdapter,
    reply_audio_path: str = "data/audio/pipeline_reply.wav",
) -> PipelineResult:
    """One full turn: audio in -> STT+router -> language detect -> LLM -> TTS."""
    transcription = route_transcription(stt, audio_path)

    lang_state = detect_language_state(
        transcription.text,
        stt_language=transcription.language,
        stt_confidence=transcription.confidence,
    )

    reply_text = run_turn(session, transcription.text, lang_state, llm)

    reply_lang_state = detect_language_state(reply_text)
    tts_result = tts.synthesize(reply_text, reply_lang_state.primary_language, reply_audio_path)

    return PipelineResult(
        transcript=transcription.text,
        stt_language=transcription.language,
        stt_confidence=transcription.confidence,
        language_state=lang_state,
        reply_text=reply_text,
        tts_result=tts_result,
    )
