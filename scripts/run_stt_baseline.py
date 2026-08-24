from app.stt.whisper_runner import WhisperRunner

SAMPLES = [
    ("data/audio/sample_en.mp3", None),
    ("data/audio/sample_ur.mp3", None),
    ("data/audio/sample_ur.mp3", "ur"),
    ("data/audio/sample_ur_latn.mp3", None),
    ("data/audio/sample_ur_latn.mp3", "ur"),
]

def main():
    print("Loading faster-whisper (small)...")
    runner = WhisperRunner(model_size="small")

    for path, forced_lang in SAMPLES:
        label = f"{path} (forced={forced_lang})" if forced_lang else f"{path} (auto-detect)"
        print(f"\n--- {label} ---")
        result = runner.transcribe(path, language=forced_lang)
        print(f"Detected language: {result.language} (confidence: {result.confidence:.2f})")
        print(f"Transcript: {result.text}")
        print(f"Inference time: {result.inference_time_s:.2f}s")

if __name__ == "__main__":
    main()
