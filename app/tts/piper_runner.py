import logging
import os
import subprocess
import time
import wave
from pathlib import Path

from app.tts.base import SynthesisResult, TTSAdapter

logger = logging.getLogger(__name__)

# Which voice natively covers which language. Roman Urdu (ur-Latn) and
# "mixed" have no native Piper voice yet -- per spec 16, fall back to a
# compatible voice and log it rather than silently mispronouncing or failing.
VOICE_MAP = {
    "en": "en_US-lessac-medium",
    "ur": "ur_PK-fasih-medium",
}
FALLBACK_LANGUAGE = "en"


class PiperAdapter(TTSAdapter):
    """Local Piper TTS adapter (ONNX, CPU, offline).

    Known Windows issue: Piper's CLI reads stdin using the OS console codepage
    by default, which mangles non-ASCII text (confirmed: Urdu/Arabic-script
    input was silently corrupted into mojibake, causing phonemization to fail
    with zero audio frames). Fix: force Python UTF-8 mode via PYTHONUTF8=1 and
    PYTHONIOENCODING=utf-8 in the subprocess environment.
    """

    def __init__(self, models_dir: str = "models/tts"):
        self.models_dir = Path(models_dir)

    def _resolve_voice(self, language: str) -> tuple[str, bool]:
        if language in VOICE_MAP:
            return VOICE_MAP[language], True

        logger.warning(
            "No native Piper voice for language=%s; falling back to '%s'",
            language, FALLBACK_LANGUAGE,
        )
        return VOICE_MAP[FALLBACK_LANGUAGE], False

    def synthesize(self, text: str, language: str, output_path: str) -> SynthesisResult:
        voice_name, native_support = self._resolve_voice(language)
        model_path = self.models_dir / f"{voice_name}.onnx"

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        start = time.perf_counter()
        subprocess.run(
            ["uv", "run", "piper", "--model", str(model_path), "--output_file", output_path],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            env=env,
        )
        elapsed = time.perf_counter() - start

        with wave.open(output_path, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()

        return SynthesisResult(
            audio_path=output_path,
            voice_used=voice_name,
            language_requested=language,
            native_support=native_support,
            audio_duration_s=duration,
            inference_time_s=elapsed,
        )

    def list_voices(self) -> dict:
        return dict(VOICE_MAP)
