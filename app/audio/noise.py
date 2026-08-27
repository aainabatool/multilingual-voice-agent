import numpy as np
import soundfile as sf


def inject_white_noise(input_path: str, output_path: str, target_snr_db: float) -> str:
    """Add white Gaussian noise to an audio file at a controlled SNR (dB)."""
    audio, sample_rate = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # collapse to mono

    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)

    noisy_audio = audio + noise
    peak = np.abs(noisy_audio).max()
    if peak > 1.0:
        noisy_audio = noisy_audio / peak

    sf.write(output_path, noisy_audio, sample_rate)
    return output_path
