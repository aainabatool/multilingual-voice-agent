from fastapi import FastAPI
from app.config import settings

app = FastAPI(title="Multilingual Voice Agent")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "config": {
            "stt_model": settings.model_stt,
            "llm_model": settings.model_llm,
            "tts_model": settings.model_tts,
            "code_switching_enabled": settings.enable_code_switching,
        },
    }
