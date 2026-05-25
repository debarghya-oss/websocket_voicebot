"""
Voicebot — FastAPI Application Entrypoint.

Usage:
    python -m voicebot.main
"""

# --- Standard Library Imports ---
import logging
import sys
import os
from contextlib import asynccontextmanager

# --- Add parent directory to sys.path to resolve voicebot package absolute imports ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# --- END LOGGING SETUP ---

# --- Third-Party Imports ---
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Module Imports ---
from voicebot.config import STATIC_DIR
from voicebot.api.router import api_router
from voicebot.services import tts_service, stt_service


# --- FastAPI Lifespan: load all models at startup ---
@asynccontextmanager
async def lifespan(app_: object):
    """Load AI models on startup; runs regardless of how uvicorn is invoked."""
    logger.info("--- [Startup] Loading AI Models ---")

    # TTS – SNAC
    tts_service.snac_model = tts_service.load_snac_model()
    if tts_service.snac_model is None:
        logger.critical("[Startup] SNAC model failed to load. TTS will be unavailable.")
    else:
        logger.info("[Startup] ✓ SNAC / TTS model loaded.")

    # STT – Whisper
    stt_service.whisper_model = stt_service.load_whisper_model()
    if stt_service.whisper_model is None:
        logger.warning("[Startup] Whisper STT model failed to load. Whisper STT will be unavailable.")
    else:
        logger.info("[Startup] ✓ Whisper STT model loaded.")

    # STT – Indic Conformer (optional – large download)
    stt_service.indic_model = stt_service.load_indic_model()
    if stt_service.indic_model is None:
        logger.warning("[Startup] Indic Conformer model failed to load. Indic STT will be unavailable.")
    else:
        logger.info("[Startup] ✓ Indic Conformer STT model loaded.")

    logger.info("--- [Startup] All models loaded. Server ready. ---")
    yield
    # (shutdown hooks can go here if needed)


# --- FastAPI App Setup ---
app = FastAPI(title="Jarvis – Orpheus TTS / STT / LLM", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static_assets")

# Include all API routes
app.include_router(api_router)


# --- Root route: serve the UI ---
@app.get("/")
async def read_root():
    """Serve index.html."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/script.js")
async def serve_script():
    """Serve script.js."""
    return FileResponse(str(STATIC_DIR / "script.js"), media_type="application/javascript")


# --- Entrypoint ---
if __name__ == "__main__":
    logger.info("Starting FastAPI Server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI Server Stopped.")
