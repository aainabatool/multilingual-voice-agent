import time

from faster_whisper import WhisperModel

from app.stt.base import STTAdapter, TranscriptionResult, TranscriptSegment


class WhisperRunner(STTAdapter):
    """faster-whisper based STT adapter."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        start_time = time.perf_counter()
        segments_iter, info = self.model.transcribe(audio_path, language=language)

        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append(TranscriptSegment(text=seg.text.strip(), start=seg.start, end=seg.end))
            full_text_parts.append(seg.text.strip())

        elapsed = time.perf_counter() - start_time

        return TranscriptionResult(
            text=" ".join(full_text_parts).strip(),
            language=info.language,
            segments=segments,
            confidence=info.language_probability,
            model_name=f"faster-whisper-{self.model_size}",
            inference_time_s=elapsed,
        )

    def detect_language(self, audio_path: str) -> str:
        _, info = self.model.transcribe(audio_path)
        return info.language
