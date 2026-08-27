import numpy as np
import soundfile as sf

from app.audio.noise import inject_white_noise
from app.audio.speed import change_speaking_rate


def _make_tone_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, tone, sample_rate)


def test_inject_white_noise_changes_audio(tmp_path):
    input_path = str(tmp_path / "clean.wav")
    output_path = str(tmp_path / "noisy.wav")
    _make_tone_wav(input_path)

    inject_white_noise(input_path, output_path, target_snr_db=5)

    clean, _ = sf.read(input_path)
    noisy, _ = sf.read(output_path)
    assert len(clean) == len(noisy)
    assert not np.allclose(clean, noisy)


def test_change_speaking_rate_shortens_audio_when_sped_up(tmp_path):
    input_path = str(tmp_path / "clean.wav")
    output_path = str(tmp_path / "fast.wav")
    _make_tone_wav(input_path, duration_s=2.0)

    change_speaking_rate(input_path, output_path, rate_factor=2.0)

    original, sr = sf.read(input_path)
    sped_up, _ = sf.read(output_path)
    assert len(sped_up) < len(original)
