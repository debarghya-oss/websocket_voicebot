
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
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect
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
from llm_router import router as llm_api_router, LLMChatRequest, generate_llm_text_stream
from audio_utils import encode_audio_to_base64, decode_audio_from_base64, encode_float32_chunk_to_base64, decode_base64_to_float32

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


@app.websocket("/ws/llm")
async def websocket_llm_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming LLM output as JSON chunks."""
    await websocket.accept()
    request_data = None
    try:
        request_data = await websocket.receive_json()
        if not isinstance(request_data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WebSocket LLM: Received request payload")
        llm_request = LLMChatRequest(**request_data)

        text_generator = generate_llm_text_stream(
            prompt=llm_request.prompt,
            history=llm_request.history,
            llm_temperature=llm_request.temperature,
            llm_top_p=llm_request.top_p,
            llm_max_tokens=llm_request.max_tokens,
            llm_repetition_penalty=llm_request.repetition_penalty,
            llm_top_k=llm_request.top_k,
            model=llm_request.model,
        )

        async def send_chunk(chunk_text: str):
            await websocket.send_json({"type": "llm_chunk", "content": chunk_text})

        for chunk in text_generator:
            await send_chunk(chunk)

        await websocket.send_json({"type": "llm_done"})

    except WebSocketDisconnect:
        logger.info("WebSocket LLM: Client disconnected")
    except Exception as exc:
        logger.exception("WebSocket LLM: Error while streaming")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS audio bytes in base64 format."""
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        if not isinstance(request_data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WebSocket TTS: Received request payload")
        tts_request = TTSRequestWithDefaults(**request_data)
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] WebSocket TTS: Text: '{tts_request.text[:100]}...'")

        await websocket.send_json({
            "type": "tts_started", 
            "sample_rate": TARGET_SAMPLE_RATE, 
            "audio_format": "FLOAT32_PCM_BASE64",
            "encoding": "base64"
        })
        logger.info(f"[{request_id}] WebSocket TTS: Sent tts_started message")

        audio_generator = generate_speech_stream_bytes(
            text=tts_request.text,
            voice=tts_request.voice,
            tts_temperature=tts_request.tts_temperature,
            tts_top_p=tts_request.tts_top_p,
            tts_repetition_penalty=tts_request.tts_repetition_penalty,
            buffer_groups_param=tts_request.buffer_groups,
            padding_ms_param=tts_request.padding_ms,
            min_decode_batch_groups_param=tts_request.min_decode_batch_groups,
        )

        sent_any_audio = False
        chunk_count = 0
        total_bytes = 0
        
        for audio_chunk in audio_generator:
            if audio_chunk:
                sent_any_audio = True
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Encode audio bytes to base64
                audio_base64 = encode_audio_to_base64(audio_chunk)
                
                # Log chunk info with base64 preview
                base64_preview = audio_base64[:80] + "..." if len(audio_base64) > 80 else audio_base64
                logger.info(
                    f"[{request_id}] TTS Chunk #{chunk_count} | "
                    f"Bytes: {len(audio_chunk)} | "
                    f"Base64 len: {len(audio_base64)} | "
                    f"Base64: {base64_preview}"
                )
                
                await websocket.send_json({
                    "type": "tts_audio_chunk",
                    "audio": audio_base64,
                    "bytes_length": len(audio_chunk)
                })

        if not sent_any_audio:
            logger.warning(f"[{request_id}] No audio frames were generated")
            await websocket.send_json({"type": "tts_warning", "message": "No audio frames were generated."})
        else:
            logger.info(
                f"[{request_id}] TTS Streaming Complete | "
                f"Total Chunks: {chunk_count} | "
                f"Total Bytes: {total_bytes}"
            )

        await websocket.send_json({"type": "tts_done"})
        logger.info(f"[{request_id}] WebSocket TTS: Sent tts_done message")

    except WebSocketDisconnect:
        logger.info("WebSocket TTS: Client disconnected")
    except Exception as exc:
        logger.exception("WebSocket TTS: Error while streaming")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/stt")
async def websocket_stt_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming STT (base64 audio upload and transcription)."""
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        if not isinstance(request_data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WebSocket STT: Received request payload")
        
        # Extract parameters from request
        engine = request_data.get("engine", "indic")
        language = request_data.get("language", "bn")
        decode_mode = request_data.get("decode_mode", "ctc")
        audio_base64 = request_data.get("audio", "")
        
        if not audio_base64:
            raise ValueError("No audio data provided")

        # Decode base64 audio
        audio_bytes = decode_audio_from_base64(audio_base64)
        if not audio_bytes:
            raise ValueError("Failed to decode audio from base64")

        logger.info(f"WebSocket STT: Received {len(audio_bytes)} bytes of audio for transcription")

        # Save to temporary file for transcription
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            await websocket.send_json({"type": "stt_processing", "message": "Transcribing audio..."})

            # Transcribe based on selected engine
            if engine == "whisper":
                transcript = whisper_transcribe(tmp_path)
            elif engine == "indic":
                transcript = indic_transcribe(tmp_path, language=language, decode_mode=decode_mode)
            else:
                transcript = indic_transcribe(tmp_path, language=language, decode_mode=decode_mode)

            await websocket.send_json({
                "type": "stt_result",
                "text": transcript,
                "engine": engine,
                "language": language
            })

        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

    except WebSocketDisconnect:
        logger.info("WebSocket STT: Client disconnected")
    except Exception as exc:
        logger.exception("WebSocket STT: Error during transcription")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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
