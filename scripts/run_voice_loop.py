from app.agent.llm_adapter import OllamaAdapter
from app.agent.state import SessionState
from app.audio.microphone import record_audio
from app.audio.playback import play_audio
from app.pipeline import run_pipeline
from app.stt.whisper_runner import WhisperRunner
from app.tts.piper_runner import PiperAdapter


def main() -> None:
    print("Loading models (this happens once)...")
    stt = WhisperRunner(model_size="small")
    llm = OllamaAdapter(model_name="llama3.2")
    tts = PiperAdapter()
    session = SessionState(session_id="live-session")

    print("\nMultilingual Voice Agent -- speak in English, Urdu, or Roman Urdu.")
    print("Press Ctrl+C to stop.\n")

    turn = 0
    while True:
        turn += 1
        input(f"[Turn {turn}] Press Enter, then speak for 5 seconds...")
        audio_path = f"data/audio/mic_turn_{turn}.wav"
        record_audio(audio_path, duration_s=5.0)

        result = run_pipeline(audio_path, session, stt, llm, tts)
        print(f"  You said: {result.transcript}")
        print(f"  Detected: {result.language_state.primary_language} "
              f"(code-switch: {result.language_state.code_switch_score})")
        print(f"  Agent: {result.reply_text}\n")

        play_audio(result.tts_result.audio_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession ended.")
