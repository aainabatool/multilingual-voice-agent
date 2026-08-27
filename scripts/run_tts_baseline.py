from app.tts.piper_runner import PiperAdapter

CASES = [
    ("en", "Hi, I want to track my order."),
    ("ur", "میرا اوڈر کہا ہے؟"),
    ("ur-Latn", "Mera order dispatch hua ya nahi?"),
]

def main():
    tts = PiperAdapter()
    for language, text in CASES:
        out_path = f"data/audio/tts_{language.replace('-', '_')}.wav"
        result = tts.synthesize(text, language, out_path)
        print(f"\nLanguage requested: {language}")
        print(f"  Voice used: {result.voice_used} (native support: {result.native_support})")
        print(f"  Audio: {result.audio_path} ({result.audio_duration_s:.2f}s)")
        print(f"  Inference time: {result.inference_time_s:.2f}s")

if __name__ == "__main__":
    main()
