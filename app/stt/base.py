from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[TranscriptSegment] = field(default_factory=list)
    confidence: float | None = None
    model_name: str = ""
    inference_time_s: float = 0.0


class STTAdapter(ABC):
    """Common interface every STT engine must implement."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file and return the result with metadata."""
        raise NotImplementedError

    @abstractmethod
    def detect_language(self, audio_path: str) -> str:
        """Return the detected language code for an audio file."""
        raise NotImplementedError
