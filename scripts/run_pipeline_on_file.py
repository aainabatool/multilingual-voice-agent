import sys

from app.agent.llm_adapter import OllamaAdapter
from app.agent.state import SessionState
from app.audio.playback import play_audio
from app.pipeline import run_pipeline
from app.stt.whisper_runner import WhisperRunner
from app.tts.piper_runner import PiperAdapter


def main(audio_path: str) -> None:
    print("Loading models...")
    stt = WhisperRunner(model_size="small")
    llm = OllamaAdapter(model_name="llama3.2")
    tts = PiperAdapter()
    session = SessionState(session_id="pipeline-test")

    print(f"\nProcessing {audio_path}...")
    result = run_pipeline(audio_path, session, stt, llm, tts)

    print(f"  Transcript: {result.transcript}")
    print(f"  STT language: {result.stt_language} (confidence: {result.stt_confidence:.2f})")
    print(f"  Language state: primary={result.language_state.primary_language}, "
          f"script={result.language_state.script}, code_switch_score={result.language_state.code_switch_score}")
    print(f"  Reply: {result.reply_text}")
    print(f"  TTS voice: {result.tts_result.voice_used} (native: {result.tts_result.native_support})")

    print("\nPlaying reply...")
    play_audio(result.tts_result.audio_path)


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/audio/sample_en.mp3"
    main(audio_file)
