from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_stt: str = "faster-whisper"
    model_llm: str = "llama3.2"  # placeholder, local Ollama model
    model_tts: str = "kokoro"    # placeholder
    default_language: str = "auto"
    enable_code_switching: bool = True
    sample_rate: int = 16000
    temperature: float = 0.2

    class Config:
        env_file = ".env"

settings = Settings()
