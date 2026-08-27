import soundfile as sf
from scipy.signal import resample


def change_speaking_rate(input_path: str, output_path: str, rate_factor: float) -> str:
    """Speed up (rate_factor > 1.0) or slow down (rate_factor < 1.0) audio.

    Uses naive resampling (changes pitch along with speed) -- a simple,
    well-understood proxy for speaking-rate robustness testing.
    """
    audio, sample_rate = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    new_length = int(len(audio) / rate_factor)
    resampled = resample(audio, new_length)

    sf.write(output_path, resampled, sample_rate)
    return output_path
