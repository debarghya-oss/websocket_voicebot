
# --- Standard Library Imports ---
import logging
import sys
import uuid
from contextlib import asynccontextmanager

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
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx

# --- Module Imports ---
from config import (
    DEFAULT_TTS_VOICE, DEFAULT_TTS_TEMP, DEFAULT_TTS_TOP_P,
    DEFAULT_TTS_REP_PENALTY, TTS_STREAM_MIN_GROUPS, TTS_STREAM_SILENCE_MS,
    DEFAULT_MIN_DECODE_BATCH_GROUPS, TARGET_SAMPLE_RATE, SERVER_BASE_URL,
)
from models import TTSRequest, STTResponse
import tts_engine
import whisper_stt_engine
import indic_stt_engine
from tts_engine import generate_speech_stream_bytes
from whisper_stt_engine import transcribe_audio as whisper_transcribe
from indic_stt_engine import transcribe_audio as indic_transcribe
from llm_router import router as llm_api_router

# --- End Imports ---

# --- FastAPI Lifespan: load all models at startup ---
@asynccontextmanager
async def lifespan(app_: object):
    """Load AI models on startup; runs regardless of how uvicorn is invoked."""
    logger.info("--- [Startup] Loading AI Models ---")

    # TTS – SNAC
    tts_engine.snac_model = tts_engine.load_snac_model()
    if tts_engine.snac_model is None:
        logger.critical("[Startup] SNAC model failed to load. TTS will be unavailable.")
    else:
        logger.info("[Startup] ✓ SNAC / TTS model loaded.")

    # STT – Whisper
    whisper_stt_engine.whisper_model = whisper_stt_engine.load_whisper_model()
    if whisper_stt_engine.whisper_model is None:
        logger.warning("[Startup] Whisper STT model failed to load. Whisper STT will be unavailable.")
    else:
        logger.info("[Startup] ✓ Whisper STT model loaded.")

    # STT – Indic Conformer (optional – large download)
    indic_stt_engine.indic_model = indic_stt_engine.load_indic_model()
    if indic_stt_engine.indic_model is None:
        logger.warning("[Startup] Indic Conformer model failed to load. Indic STT will be unavailable.")
    else:
        logger.info("[Startup] ✓ Indic Conformer STT model loaded.")

    logger.info("--- [Startup] All models loaded. Server ready. ---")
    yield
    # (shutdown hooks can go here if needed)
# --- End Lifespan ---

# --- FastAPI App Setup ---
app = FastAPI(title="Jarvis – Orpheus TTS / STT / LLM", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static_assets")

# --- Include the LLM router ---
app.include_router(llm_api_router, prefix="/api/llm", tags=["LLM"])
# --- End Include LLM router ---


# --- Pydantic Models ---
class TTSRequestWithDefaults(TTSRequest):
    """TTSRequest with default values"""
    voice: str = DEFAULT_TTS_VOICE
    tts_temperature: float = DEFAULT_TTS_TEMP
    tts_top_p: float = DEFAULT_TTS_TOP_P
    tts_repetition_penalty: float = DEFAULT_TTS_REP_PENALTY
    buffer_groups: int = TTS_STREAM_MIN_GROUPS
    padding_ms: int = TTS_STREAM_SILENCE_MS
    min_decode_batch_groups: int = DEFAULT_MIN_DECODE_BATCH_GROUPS
# --- End Pydantic Models ---


@app.post("/api/tts/stream")
async def tts_stream_endpoint(request: TTSRequestWithDefaults):
    """Text-to-Speech streaming endpoint"""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] TTS: Received POST /api/tts/stream")
    logger.info(f"[{request_id}] TTS: Payload: {request.model_dump_json(indent=2)}")

    if tts_engine.snac_model is None:
        logger.error(f"[{request_id}] TTS: SNAC model not loaded. Returning 503.")
        raise HTTPException(status_code=503, detail="SNAC model not loaded. TTS is unavailable.")

    audio_generator = generate_speech_stream_bytes(
        text=request.text,
        voice=request.voice,
        tts_temperature=request.tts_temperature,
        tts_top_p=request.tts_top_p,
        tts_repetition_penalty=request.tts_repetition_penalty,
        buffer_groups_param=request.buffer_groups,
        padding_ms_param=request.padding_ms,
        min_decode_batch_groups_param=request.min_decode_batch_groups,
    )
    headers = {"X-Sample-Rate": str(TARGET_SAMPLE_RATE), "X-Audio-Format": "FLOAT32_PCM"}
    return StreamingResponse(audio_generator, media_type="audio/octet-stream", headers=headers)

@app.post("/api/stt/set_engine")
async def set_stt_engine(body: dict):
    engine = body.get("engine")
    # load/swap your STT model here
    return {"ok": True, "engine": engine}

@app.post("/api/stt/transcribe", response_model=STTResponse)
async def stt_transcribe_endpoint(
    audio_file: UploadFile = File(...),
    engine: str = Query(default="indic", description="STT engine: 'whisper' or 'indic'"),
    language: str = Query(default="bn", description="Language code for Indic engine, e.g. 'bn', 'hi'"),
    decode_mode: str = Query(default="ctc", description="Decode mode for Indic engine: 'ctc' or 'rnnt'"),
):
    """
    Speech-to-Text transcription endpoint.

    - **engine**: `whisper` (default) or `indic`
    - **language**: BCP-47 code used by the Indic engine (ignored for Whisper)
    - **decode_mode**: `ctc` or `rnnt` (Indic only)
    """
    request_id = str(uuid.uuid4())
    logger.info(
        f"[{request_id}] STT: POST /api/stt/transcribe | engine={engine} "
        f"| file={audio_file.filename}"
    )

    engine_lower = engine.lower().strip()

    if engine_lower == "indic":
        if indic_stt_engine.indic_model is None:
            logger.error(f"[{request_id}] STT: Indic model not loaded.")
            return STTResponse(
                text="", error="STT service unavailable: Indic Conformer model not loaded."
            )
        return await indic_transcribe(audio_file, language=language, decode_mode=decode_mode)

    else:  # default: whisper
        if whisper_stt_engine.whisper_model is None:
            logger.error(f"[{request_id}] STT: Whisper model not loaded.")
            return STTResponse(
                text="", error="STT service unavailable: Whisper model not loaded."
            )
        return await whisper_transcribe(audio_file)



@app.get("/api/llm/models")
async def list_llm_models():
    """
    Returns the list of available LLM models from the configured backend server
    (LM Studio / Ollama / OpenAI-compatible).
    """
    models_url = f"{SERVER_BASE_URL}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_url)
            resp.raise_for_status()
            data = resp.json()
            # data is typically {"object": "list", "data": [{"id": "...", ...}, ...]}
            model_ids = [m["id"] for m in data.get("data", [])]
            return JSONResponse({"models": model_ids})
    except Exception as e:
        logger.warning(f"Could not fetch models from {models_url}: {e}")
        return JSONResponse({"models": [], "error": str(e)})


@app.get("/")
async def read_root():
    """Serve index.html"""
    return FileResponse("static/index.html")


@app.get("/script.js")
async def serve_script():
    """Serve script.js"""
    return FileResponse("static/script.js", media_type="application/javascript")


# --- End FastAPI Endpoints ---

if __name__ == "__main__":
    # Models are loaded via the lifespan handler above.
    # Running `python main.py` is equivalent to `uvicorn main:app --host 0.0.0.0 --port 8000`.
    logger.info("Starting FastAPI Server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI Server Stopped.")

