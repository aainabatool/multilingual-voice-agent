from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SynthesisResult:
    audio_path: str
    voice_used: str
    language_requested: str
    native_support: bool  # False if we fell back to a non-native voice
    audio_duration_s: float
    inference_time_s: float


class TTSAdapter(ABC):
    """Common interface every TTS engine must implement (spec 5.8)."""

    @abstractmethod
    def synthesize(self, text: str, language: str, output_path: str) -> SynthesisResult:
        raise NotImplementedError

    @abstractmethod
    def list_voices(self) -> dict:
        raise NotImplementedError
