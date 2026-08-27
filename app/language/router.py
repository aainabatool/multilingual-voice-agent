import logging

from app.stt.base import TranscriptionResult
from app.stt.whisper_runner import WhisperRunner

logger = logging.getLogger(__name__)

# Per Phase 1 finding: Whisper's auto-detect frequently mislabels Urdu speech
# as Hindi (and transcribes in the wrong script) since the two languages are
# spoken nearly identically. If auto-detect reports Hindi, re-run forced to
# Urdu and keep whichever result the model itself is more confident about.
CONFUSABLE_LANGUAGE = "hi"
RETRY_LANGUAGE = "ur"


def route_transcription(runner: WhisperRunner, audio_path: str) -> TranscriptionResult:
    result = runner.transcribe(audio_path)

    if result.language == CONFUSABLE_LANGUAGE:
        logger.info(
            "STT auto-detect returned '%s' (known Urdu-confusable); retrying forced '%s'",
            CONFUSABLE_LANGUAGE, RETRY_LANGUAGE,
        )
        retry_result = runner.transcribe(audio_path, language=RETRY_LANGUAGE)

        if retry_result.confidence and (not result.confidence or retry_result.confidence >= result.confidence):
            return retry_result

    return result
