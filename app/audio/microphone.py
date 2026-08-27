import soundfile as sf
import sounddevice as sd


def record_audio(output_path: str, duration_s: float = 5.0, sample_rate: int = 16000) -> str:
    """Record from the default microphone and save to a wav file."""
    print(f"Recording for {duration_s}s...")
    audio = sd.rec(int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(output_path, audio, sample_rate)
    print(f"Saved recording to {output_path}")
    return output_path
