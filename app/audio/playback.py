import soundfile as sf
import sounddevice as sd


def play_audio(audio_path: str) -> None:
    """Play a wav file through the default speaker output."""
    data, sample_rate = sf.read(audio_path)
    sd.play(data, sample_rate)
    sd.wait()
