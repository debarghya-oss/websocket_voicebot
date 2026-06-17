# --- Standard Library Imports ---
import asyncio
import time
import logging
import sys
import uuid
import tempfile
import os
import base64
from contextlib import asynccontextmanager
from typing import Optional
import json
# --- SETUP LOGGING ---
import resampy
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Third-Party Imports ---
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
from io import BytesIO

# --- Module Imports ---
from config import (
    DEFAULT_TTS_VOICE, DEFAULT_TTS_TEMP, DEFAULT_TTS_TOP_P,
    DEFAULT_TTS_REP_PENALTY, TTS_STREAM_MIN_GROUPS, TTS_STREAM_SILENCE_MS,
    DEFAULT_MIN_DECODE_BATCH_GROUPS, TARGET_SAMPLE_RATE, SERVER_BASE_URL,
    VAD_SAMPLE_RATE, VAD_CHUNK_SAMPLES, VAD_BARGEIN_THRESHOLD,
    DEFAULT_TTS_ENGINE, PARLER_VOICE_DESCRIPTIONS, DEFAULT_PARLER_VOICE,
)
from models import TTSRequest, STTResponse
import tts_engine
import parler_tts_engine
import whisper_stt_engine
import indic_stt_engine
import vad_engine
from tts_engine import generate_speech_stream_bytes
from parler_tts_engine import generate_speech_stream_parler_tts
#from whisper_stt_engine import transcribe_audio as whisper_transcribe
from indic_stt_engine import transcribe_audio as indic_transcribe
from llm_router import router as llm_api_router, LLMChatRequest, generate_llm_text_stream
from rag.routers.ingest import router as rag_ingest_router
from audio_utils import encode_audio_to_base64, decode_audio_from_base64
from vad_engine import VADProcessor

class AsyncIterator:
    """
    Bridge to run a synchronous blocking generator in a background thread
    and yield its items asynchronously, preventing event loop blocking.
    """
    def __init__(self, sync_generator):
        self.sync_generator = sync_generator
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for item in self.sync_generator:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except Exception as e:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, e)
        finally:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        val = await self.queue.get()
        if val is None:
            raise StopAsyncIteration
        if isinstance(val, Exception):
            raise val
        return val

# --- Global TTS Engine State ---
current_tts_engine = DEFAULT_TTS_ENGINE


# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app_: object):
    global current_tts_engine
    logger.info("--- [Startup] Loading AI Models ---")

    tts_engine.snac_model = tts_engine.load_snac_model()
    if tts_engine.snac_model is None:
        logger.critical("[Startup] SNAC model failed to load. Orpheus TTS will be unavailable.")
    else:
        logger.info("[Startup] ✓ SNAC / Orpheus TTS model loaded.")

    parler_tts_engine.parler_tts_model = parler_tts_engine.load_parler_tts_model()
    if parler_tts_engine.parler_tts_model is None:
        logger.warning("[Startup] Parler TTS model failed to load. Parler TTS will be unavailable.")
    else:
        logger.info("[Startup] ✓ Parler TTS model loaded.")

    whisper_stt_engine.whisper_model = whisper_stt_engine.load_whisper_model()
    if whisper_stt_engine.whisper_model is None:
        logger.warning("[Startup] Whisper STT model failed to load.")
    else:
        logger.info("[Startup] ✓ Whisper STT model loaded.")

    indic_stt_engine.indic_model = indic_stt_engine.load_indic_model()
    if indic_stt_engine.indic_model is None:
        logger.warning("[Startup] Indic Conformer model failed to load.")
    else:
        logger.info("[Startup] ✓ Indic Conformer STT model loaded.")

    vad_engine.vad_model = vad_engine.load_vad_model()
    if vad_engine.vad_model is None:
        logger.warning("[Startup] Silero VAD model failed to load. Conversation mode will be unavailable.")
    else:
        logger.info("[Startup] ✓ Silero VAD model loaded.")

    # --- RAG: initialise Milvus collection (no-op if already exists) ---
    rag_enabled = os.getenv("RAG_ENABLED", "false").lower() == "true"
    if rag_enabled:
        try:
            from rag.vectorstore import init_collection
            init_collection()
            logger.info("[Startup] ✓ Milvus RAG collection ready.")
        except Exception as _rag_err:
            logger.warning(f"[Startup] RAG vector store init failed (RAG will be unavailable): {_rag_err}")

    # FIX: ensure temp dir exists
    os.makedirs("temp_stt_audio_files", exist_ok=True)
    logger.info("[Startup] ✓ temp_stt_audio_files directory ensured.")

    logger.info("--- [Startup] All models loaded. Server ready. ---")
    yield


# --- FastAPI App ---
app = FastAPI(title="Jarvis – Orpheus TTS / STT / LLM", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static", html=True), name="static_assets")
app.include_router(llm_api_router, prefix="/api/llm", tags=["LLM"])
app.include_router(rag_ingest_router, prefix="/api/rag", tags=["RAG"])


# --- Pydantic Models ---
class TTSRequestWithDefaults(TTSRequest):
    voice: str = DEFAULT_TTS_VOICE
    tts_temperature: float = DEFAULT_TTS_TEMP
    tts_top_p: float = DEFAULT_TTS_TOP_P
    tts_repetition_penalty: float = DEFAULT_TTS_REP_PENALTY
    buffer_groups: int = TTS_STREAM_MIN_GROUPS
    padding_ms: int = TTS_STREAM_SILENCE_MS
    min_decode_batch_groups: int = DEFAULT_MIN_DECODE_BATCH_GROUPS
    tts_model: str = DEFAULT_TTS_ENGINE  # Engine selection: "orpheus" or "parler"
    voice_description: str = PARLER_VOICE_DESCRIPTIONS.get(DEFAULT_PARLER_VOICE, "")  # For Parler TTS


# ================================================================
# HTTP ENDPOINTS
# ================================================================

@app.post("/api/tts/stream")
async def tts_stream_endpoint(request: TTSRequestWithDefaults):
    global current_tts_engine
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] TTS POST /api/tts/stream | engine={request.tts_model}")
    
    # Route to selected engine
    if request.tts_model == "parler":
        if parler_tts_engine.parler_tts_model is None:
            raise HTTPException(status_code=503, detail="Parler TTS model not loaded.")
        audio_generator = generate_speech_stream_parler_tts(
            text=request.text,
            voice_description=request.voice_description
        )
    else:  # Default to Orpheus
        if tts_engine.snac_model is None:
            raise HTTPException(status_code=503, detail="Orpheus TTS model not loaded.")
        audio_generator = generate_speech_stream_bytes(
            text=request.text, voice=request.voice,
            tts_temperature=request.tts_temperature, tts_top_p=request.tts_top_p,
            tts_repetition_penalty=request.tts_repetition_penalty,
            buffer_groups_param=request.buffer_groups, padding_ms_param=request.padding_ms,
            min_decode_batch_groups_param=request.min_decode_batch_groups,
        )
    
    headers = {"X-Sample-Rate": str(TARGET_SAMPLE_RATE), "X-Audio-Format": "FLOAT32_PCM"}
    return StreamingResponse(audio_generator, media_type="audio/octet-stream", headers=headers)


@app.post("/api/tts/set_engine")
async def set_tts_engine(body: dict):
    """Set the active TTS engine."""
    global current_tts_engine
    engine = body.get("engine", DEFAULT_TTS_ENGINE)
    if engine not in ["orpheus", "parler"]:
        return JSONResponse({"ok": False, "error": "Invalid engine"}, status_code=400)
    current_tts_engine = engine
    logger.info(f"TTS engine switched to: {engine}")
    return {"ok": True, "engine": engine}


@app.get("/api/tts/engines")
async def list_tts_engines():
    """List available TTS engines and their info."""
    engines = {
        "orpheus": {
            "available": tts_engine.snac_model is not None,
            "voices": ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"],
            "description": "Orpheus TTS with SNAC codec"
        },
        "parler": {
            "available": parler_tts_engine.parler_tts_model is not None,
            "voices": list(PARLER_VOICE_DESCRIPTIONS.keys()),
            "descriptions": PARLER_VOICE_DESCRIPTIONS,
            "description": "Parler TTS (Indic models)"
        }
    }
    return {"current": current_tts_engine, "engines": engines}


@app.post("/api/stt/set_engine")
async def set_stt_engine(body: dict):
    engine = body.get("engine")
    return {"ok": True, "engine": engine}


@app.post("/api/stt/transcribe", response_model=STTResponse)
async def stt_transcribe_endpoint(
    audio_file: UploadFile = File(...),
    engine: str = Query(default="indic"),
    language: str = Query(default="bn"),
    decode_mode: str = Query(default="ctc"),
):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] STT POST | engine={engine} | file={audio_file.filename}")
    if engine.lower().strip() == "indic":
        if indic_stt_engine.indic_model is None:
            return STTResponse(text="", error="Indic model not loaded.")
        return await indic_transcribe(audio_file, language=language, decode_mode=decode_mode)
    else:
        if whisper_stt_engine.whisper_model is None:
            return STTResponse(text="", error="Whisper model not loaded.")
        return await whisper_transcribe(audio_file)


@app.get("/api/llm/models")
async def list_llm_models():
    models_url = f"{SERVER_BASE_URL}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_url)
            resp.raise_for_status()
            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            return JSONResponse({"models": model_ids})
    except Exception as e:
        logger.warning(f"Could not fetch models from {models_url}: {e}")
        return JSONResponse({"models": [], "error": str(e)})


# ================================================================
# WEBSOCKET: LLM
# ================================================================

@app.websocket("/ws/llm")
async def websocket_llm_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        llm_request = LLMChatRequest(**request_data)
        text_generator = generate_llm_text_stream(
            prompt=llm_request.prompt, history=llm_request.history,
            llm_temperature=llm_request.temperature, llm_top_p=llm_request.top_p,
            llm_max_tokens=llm_request.max_tokens,
            llm_repetition_penalty=llm_request.repetition_penalty,
            llm_top_k=llm_request.top_k, model=llm_request.model,
        )
        for chunk in text_generator:
            await websocket.send_json({"type": "llm_chunk", "content": chunk})
        await websocket.send_json({"type": "llm_done"})
    except WebSocketDisconnect:
        logger.info("WS LLM: Client disconnected")
    except Exception as exc:
        logger.exception("WS LLM: Error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ================================================================
# WEBSOCKET: TTS
# ================================================================

@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        tts_request = TTSRequestWithDefaults(**request_data)
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] WS TTS: '{tts_request.text[:80]}'")

        await websocket.send_json({
            "type": "tts_started",
            "sample_rate": TARGET_SAMPLE_RATE,
            "audio_format": "FLOAT32_PCM_BASE64",
        })

        audio_generator = generate_speech_stream_bytes(
            text=tts_request.text, voice=tts_request.voice,
            tts_temperature=tts_request.tts_temperature, tts_top_p=tts_request.tts_top_p,
            tts_repetition_penalty=tts_request.tts_repetition_penalty,
            buffer_groups_param=tts_request.buffer_groups, padding_ms_param=tts_request.padding_ms,
            min_decode_batch_groups_param=tts_request.min_decode_batch_groups,
        )

        chunk_count = 0
        for audio_chunk in audio_generator:
            if audio_chunk:
                chunk_count += 1
                await websocket.send_json({
                    "type": "tts_audio_chunk",
                    "audio": encode_audio_to_base64(audio_chunk),
                    "bytes_length": len(audio_chunk),
                })

        if chunk_count == 0:
            await websocket.send_json({"type": "tts_warning", "message": "No audio frames generated."})

        await websocket.send_json({"type": "tts_done"})
        logger.info(f"[{request_id}] WS TTS done ({chunk_count} chunks)")

    except WebSocketDisconnect:
        logger.info("WS TTS: Client disconnected")
    except Exception as exc:
        logger.exception("WS TTS: Error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ================================================================
# WEBSOCKET: STT
# ================================================================

@app.websocket("/ws/stt")
async def websocket_stt_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        engine     = request_data.get("engine", "indic")
        language   = request_data.get("language", "bn")
        decode_mode= request_data.get("decode_mode", "ctc")
        audio_b64  = request_data.get("audio", "")

        if not audio_b64:
            raise ValueError("No audio data provided")

        audio_bytes = decode_audio_from_base64(audio_b64)
        if not audio_bytes:
            raise ValueError("Failed to decode audio")

        logger.info(f"WS STT: {len(audio_bytes)} bytes | engine={engine}")
        await websocket.send_json({"type": "stt_processing", "message": "Transcribing…"})

        # FIX: use system temp dir (always exists), no dir= argument
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            upload_file = UploadFile(filename="audio.webm", file=BytesIO(audio_bytes))
            if engine == "indic":
                result = await indic_transcribe(upload_file, language=language, decode_mode=decode_mode)
            else:
                result = await indic_transcribe(upload_file)
            transcript = result.text if hasattr(result, "text") else str(result)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        await websocket.send_json({
            "type": "stt_result",
            "text": transcript,
            "engine": engine,
            "language": language,
        })

    except WebSocketDisconnect:
        logger.info("WS STT: Client disconnected")
    except Exception as exc:
        logger.exception("WS STT: Error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ================================================================
# WEBSOCKET: CONVERSATION (VAD + STT + LLM + TTS)
# ================================================================

@app.websocket("/ws/conversation")
async def websocket_conversation_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Conversation WS: Connected")

    if vad_engine.vad_model is None:
        await websocket.send_json({"type": "error", "message": "VAD model not loaded."})
        await websocket.close()
        return

    vad_processor  = VADProcessor(vad_engine.vad_model)
    config         = {}
    chat_history   = []
    is_processing  = False
    interrupt_flag = asyncio.Event()
    audio_queue: asyncio.Queue = asyncio.Queue()

    async def _audio_receiver():
        nonlocal config
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")
                if msg_type == "config":
                    config = data
                    logger.info(f"[{session_id}] Config: engine={config.get('engine')} voice={config.get('voice')}")
                    await websocket.send_json({"type": "config_ack", "status": "ok"})
                elif msg_type == "audio_chunk":
                    await audio_queue.put(data)
                elif msg_type == "stop":
                    logger.info(f"[{session_id}] Stop received")
                    break
        except WebSocketDisconnect:
            logger.info(f"[{session_id}] Receiver: client disconnected")
        except Exception as e:
            logger.exception(f"[{session_id}] Receiver error: {e}")
        finally:
            await audio_queue.put(None)

    async def _process_pipeline(speech_audio: np.ndarray):
        nonlocal is_processing
        is_processing = True
        interrupt_flag.clear()

        try:
            # --- STT ---
            await websocket.send_json({"type": "processing", "stage": "stt"})

            engine      = config.get("engine", "indic")
            language    = config.get("language", "bn")
            decode_mode = config.get("decode_mode", "ctc")

            try:
                if engine == "indic":
                    stt_result = await indic_stt_engine.transcribe_numpy(
                        speech_audio, language=language, decode_mode=decode_mode
                    )
                elif engine == "whisper":
                    stt_result = await whisper_stt_engine.transcribe_numpy(speech_audio)
                else:
                    stt_result = await indic_stt_engine.transcribe_numpy(
                        speech_audio, language=language, decode_mode=decode_mode
                    )
                transcript = stt_result.text if hasattr(stt_result, "text") else str(stt_result)
            except Exception as e:
                logger.exception(f"[{session_id}] STT transcription error: {e}")
                transcript = ""

            if interrupt_flag.is_set():
                return

            if not transcript or not transcript.strip():
                logger.info(f"[{session_id}] Empty transcript, skipping")
                await websocket.send_json({"type": "listening"})
                return

            await websocket.send_json({"type": "transcript", "text": transcript})
            chat_history.append({"role": "user", "content": transcript})
            logger.info(f"[{session_id}] STT: '{transcript[:100]}'")

            # --- LLM ---
            if interrupt_flag.is_set():
                return

            await websocket.send_json({"type": "processing", "stage": "llm"})
            llm_text = ""

            text_generator = generate_llm_text_stream(
                prompt=transcript,
                history=chat_history[:-1],
                llm_temperature=config.get("temperature", 0.7),
                llm_top_p=config.get("top_p", 0.9),
                llm_max_tokens=config.get("max_tokens", -1),
                llm_repetition_penalty=config.get("repetition_penalty", 1.1),
                llm_top_k=config.get("top_k", 45),
                model=config.get("model", None),
            )

            async_text_generator = AsyncIterator(text_generator)
            async for chunk in async_text_generator:
                if interrupt_flag.is_set():
                    return
                llm_text += chunk
                try:
                    await websocket.send_json({"type": "llm_chunk", "content": chunk})
                except Exception:
                    return

            await websocket.send_json({"type": "llm_done"})
            chat_history.append({"role": "assistant", "content": llm_text})
            logger.info(f"[{session_id}] LLM done: '{llm_text[:100]}'")

            # --- TTS ---
            if interrupt_flag.is_set() or not llm_text.strip():
                await websocket.send_json({"type": "listening"})
                return

            await websocket.send_json({
                "type": "tts_started",
                "sample_rate": TARGET_SAMPLE_RATE,
                "audio_format": "FLOAT32_PCM_BASE64",
            })

            tts_start_time = time.time()
            total_tts_audio_bytes = 0

            audio_generator = generate_speech_stream_bytes(
                text=llm_text,
                voice=config.get("voice", "tara"),
                tts_temperature=config.get("tts_temperature", 0.9),
                tts_top_p=config.get("tts_top_p", 0.9),
                tts_repetition_penalty=config.get("tts_repetition_penalty", 1.1),
                buffer_groups_param=config.get("buffer_groups", 5),
                padding_ms_param=config.get("padding_ms", 0),
                min_decode_batch_groups_param=config.get("min_decode_batch_groups", 7),
            )

            async_audio_generator = AsyncIterator(audio_generator)
            async for audio_chunk in async_audio_generator:
                if interrupt_flag.is_set():
                    return
                if audio_chunk:
                    total_tts_audio_bytes += len(audio_chunk)
                    try:
                        await websocket.send_json({
                            "type": "tts_audio_chunk",
                            "audio": encode_audio_to_base64(audio_chunk),
                            "bytes_length": len(audio_chunk),
                        })
                    except Exception:
                        return

            await websocket.send_json({"type": "tts_done"})
            logger.info(f"[{session_id}] TTS done")

            # --- Wait for client playback to finish (echo prevention) ---
            if total_tts_audio_bytes > 0:
                # FLOAT32 PCM = 4 bytes per sample
                total_samples = total_tts_audio_bytes / 4
                playback_duration = total_samples / TARGET_SAMPLE_RATE
                elapsed = time.time() - tts_start_time
                remaining = playback_duration - elapsed + 0.5  # 0.5s safety buffer
                if remaining > 0:
                    logger.info(f"[{session_id}] Waiting {remaining:.1f}s for client playback to finish (echo prevention)")
                    try:
                        await asyncio.wait_for(interrupt_flag.wait(), timeout=remaining)
                        logger.info(f"[{session_id}] Barge-in during playback wait")
                    except asyncio.TimeoutError:
                        pass  # Normal: playback finished without interruption

        except Exception as e:
            logger.exception(f"[{session_id}] Pipeline error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            # Only flush and reset if we were NOT interrupted (i.e. normal completion)
            if not interrupt_flag.is_set():
                flushed = 0
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                        flushed += 1
                    except asyncio.QueueEmpty:
                        break
                if flushed > 0:
                    logger.info(f"[{session_id}] Flushed {flushed} backlogged audio chunks (echo prevention)")
                vad_processor.reset()
            else:
                logger.info(f"[{session_id}] Interrupted by barge-in, preserving audio queue and VAD state")
            is_processing = False
            interrupt_flag.clear()
            try:
                await websocket.send_json({"type": "listening"})
            except Exception:
                pass

    # --- Main loop ---
    residual_audio = np.array([], dtype=np.float32)
    receiver_task = asyncio.create_task(_audio_receiver())
    pipeline_task: Optional[asyncio.Task] = None  # FIX: Optional now imported
    prev_speaking = False

    try:
        while True:
            try:
                data = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if data is None:
                break

            audio_b64 = data.get("audio", "")
            if not audio_b64:
                continue

            try:
                raw_bytes      = base64.b64decode(audio_b64)
                int16_samples  = np.frombuffer(raw_bytes, dtype=np.int16)
                float32_samples = int16_samples.astype(np.float32) / 32768.0
            except Exception as e:
                logger.warning(f"[{session_id}] Audio decode error: {e}")
                continue

            # Resample if needed
            client_rate = config.get("sample_rate", 16000)
            if client_rate != VAD_SAMPLE_RATE:
                from audio_utils import resample_audio
                float32_samples = resample_audio(float32_samples, client_rate, VAD_SAMPLE_RATE)

            # Process in VAD chunk increments
            combined_samples = np.concatenate([residual_audio, float32_samples])
            offset = 0
            while offset + VAD_CHUNK_SAMPLES <= len(combined_samples):
                chunk = combined_samples[offset:offset + VAD_CHUNK_SAMPLES]
                offset += VAD_CHUNK_SAMPLES

                # Barge-in detection
                if is_processing:
                    utterance = vad_processor.process_chunk(chunk)
                    current_speaking = vad_processor.is_speech_active
                    
                    if current_speaking != prev_speaking:
                        prev_speaking = current_speaking
                        try:
                            await websocket.send_json({"type": "vad_state", "is_speaking": current_speaking})
                        except Exception:
                            break

                    if current_speaking:
                        logger.info(f"[{session_id}] BARGE-IN (speech active)")
                        interrupt_flag.set()
                        if pipeline_task and not pipeline_task.done():
                            pipeline_task.cancel()
                        try:
                            await websocket.send_json({"type": "interrupt"})
                            await websocket.send_json({"type": "listening"})
                        except Exception:
                            break
                        is_processing = False
                    continue

                # Normal VAD
                utterance = vad_processor.process_chunk(chunk)

                current_speaking = vad_processor.is_speech_active
                if current_speaking != prev_speaking:
                    prev_speaking = current_speaking
                    try:
                        await websocket.send_json({"type": "vad_state", "is_speaking": current_speaking})
                    except Exception:
                        break

                if utterance is not None:
                    logger.info(f"[{session_id}] Utterance: {len(utterance)} samples ({len(utterance)/VAD_SAMPLE_RATE:.2f}s)")
                    pipeline_task = asyncio.create_task(_process_pipeline(utterance))

            # Save remaining samples for next packet
            residual_audio = combined_samples[offset:]

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Conversation WS: Client disconnected")
    except Exception as e:
        logger.exception(f"[{session_id}] Conversation WS: Error: {e}")
    finally:
        receiver_task.cancel()
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[{session_id}] Conversation WS: Session ended")

@app.websocket("/ws/exotel/voicebot")
async def websocket_exotel_voicebot(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]

    logger.info(f"[{session_id}] Exotel Voicebot Connected")

    if vad_engine.vad_model is None:
        await websocket.close()
        return

    # =========================================================
    # EXOTEL CONSTANTS
    # =========================================================

    EXOTEL_MIN_CHUNK = 3200  # bytes
    EXOTEL_CHUNK_MULTIPLE = 320

    # =========================================================
    # SESSION STATE
    # =========================================================

    vad_processor = VADProcessor(vad_engine.vad_model)

    config = {
        "sample_rate": 8000,
        "language": "bn",
        "engine": "indic",
        "decode_mode": "ctc",
        "voice": "tara",
    }

    chat_history = []

    audio_queue: asyncio.Queue = asyncio.Queue()

    stream_sid = None

    is_processing = False

    interrupt_flag = asyncio.Event()

    pipeline_task: Optional[asyncio.Task] = None

    prev_speaking = False

    outgoing_sequence = 1

    outgoing_timestamp = 0

    # =========================================================
    # HELPERS
    # =========================================================

    def _pad_chunk(chunk: bytes) -> bytes:
        remainder = len(chunk) % EXOTEL_CHUNK_MULTIPLE

        if remainder != 0:
            padding = EXOTEL_CHUNK_MULTIPLE - remainder
            chunk += b"\x00" * padding

        return chunk

    async def send_exotel_media(
        pcm_bytes: bytes,
    ):
        nonlocal outgoing_sequence
        nonlocal outgoing_timestamp

        if not stream_sid:
            return

        if not pcm_bytes:
            return

        chunks = [
            pcm_bytes[i : i + EXOTEL_MIN_CHUNK]
            for i in range(0, len(pcm_bytes), EXOTEL_MIN_CHUNK)
        ]

        logger.info(
            f"[{session_id}] Sending {len(chunks)} TTS chunks to Exotel"
        )

        for idx, chunk in enumerate(chunks):

            if interrupt_flag.is_set():
                logger.info(
                    f"[{session_id}] TTS interrupted before chunk send"
                )

                clear_event = {
                    "event": "clear",
                    "stream_sid": stream_sid,
                }

                try:
                    await websocket.send_text(json.dumps(clear_event))
                except Exception:
                    pass

                return

            chunk = _pad_chunk(chunk)

            media_event = {
                "event": "media",
                "sequence_number": str(outgoing_sequence),
                "stream_sid": stream_sid,
                "media": {
                    "chunk": str(idx),
                    "timestamp": str(outgoing_timestamp),
                    "payload": base64.b64encode(chunk).decode(),
                },
            }

            try:
                await websocket.send_text(json.dumps(media_event))
            except Exception:
                return

            outgoing_sequence += 1

            exotel_rate = config.get("sample_rate", 8000)
            chunk_samples = len(chunk) // 2
            chunk_duration_ms = int((chunk_samples / exotel_rate) * 1000)
            chunk_duration_seconds = chunk_samples / exotel_rate

            outgoing_timestamp += chunk_duration_ms

            await asyncio.sleep(chunk_duration_seconds)

        # mark event
        mark_event = {
            "event": "mark",
            "sequence_number": str(outgoing_sequence),
            "stream_sid": stream_sid,
            "mark": {
                "name": "tts_complete",
            },
        }

        try:
            await websocket.send_text(json.dumps(mark_event))
        except Exception:
            pass

        outgoing_sequence += 1

    async def _audio_receiver():

        try:
            while True:

                raw = await websocket.receive_text()

                data = json.loads(raw)

                event = data.get("event")

                # =========================================
                # CONNECTED
                # =========================================

                if event == "connected":

                    logger.info(f"[{session_id}] Exotel Connected Event")

                # =========================================
                # START
                # =========================================

                elif event == "start":

                    nonlocal stream_sid

                    stream_sid = data.get("stream_sid")

                    start_data = data.get("start", {})

                    media_format = start_data.get("media_format", {})

                    sample_rate = int(
                        media_format.get("sample_rate", 8000)
                    )

                    config["sample_rate"] = sample_rate

                    logger.info(
                        f"[{session_id}] Start Stream="
                        f"{stream_sid} Rate={sample_rate}"
                    )

                # =========================================
                # MEDIA
                # =========================================

                elif event == "media":

                    media = data.get("media", {})

                    payload = media.get("payload")

                    if not payload:
                        continue

                    try:
                        pcm_chunk = base64.b64decode(payload)
                    except Exception:
                        continue

                    await audio_queue.put(pcm_chunk)

                # =========================================
                # DTMF
                # =========================================

                elif event == "dtmf":

                    dtmf = data.get("dtmf", {})

                    digit = dtmf.get("digit")

                    logger.info(
                        f"[{session_id}] DTMF Received: {digit}"
                    )

                # =========================================
                # MARK
                # =========================================

                elif event == "mark":

                    logger.info(
                        f"[{session_id}] Mark Event: {data}"
                    )

                # =========================================
                # STOP
                # =========================================

                elif event == "stop":

                    logger.info(f"[{session_id}] Stop Event")

                    break

        except WebSocketDisconnect:

            logger.info(
                f"[{session_id}] Client disconnected"
            )

        except Exception as e:

            logger.exception(
                f"[{session_id}] Receiver error: {e}"
            )

        finally:

            await audio_queue.put(None)

    async def _process_pipeline(
        speech_audio: np.ndarray,
    ):
        nonlocal is_processing

        is_processing = True

        interrupt_flag.clear()

        try:

            # =====================================================
            # STT
            # =====================================================

            engine = config.get("engine", "whisper")

            language = config.get("language", "en")

            decode_mode = config.get("decode_mode", "rnnt")

            try:
                if engine == "indic":
                    stt_result = await indic_stt_engine.transcribe_numpy(
                        speech_audio, language=language, decode_mode=decode_mode
                    )
                elif engine == "whisper":
                    stt_result = await whisper_stt_engine.transcribe_numpy(speech_audio)
                else:
                    stt_result = await indic_stt_engine.transcribe_numpy(
                        speech_audio, language=language, decode_mode=decode_mode
                    )
                transcript = stt_result.text if hasattr(stt_result, "text") else str(stt_result)
            except Exception as e:
                logger.exception(f"[{session_id}] STT transcription error: {e}")
                transcript = ""

            if interrupt_flag.is_set():
                return

            if not transcript or not transcript.strip():

                logger.info(
                    f"[{session_id}] Empty transcript"
                )

                return

            logger.info(
                f"[{session_id}] USER: {transcript}"
            )

            chat_history.append(
                {
                    "role": "user",
                    "content": transcript,
                }
            )

            # =====================================================
            # LLM
            # =====================================================

            llm_text = ""

            text_generator = generate_llm_text_stream(
                prompt=transcript,
                history=chat_history[:-1],
                llm_temperature=config.get(
                    "temperature",
                    0.7,
                ),
                llm_top_p=config.get("top_p", 0.9),
                llm_max_tokens=config.get(
                    "max_tokens",
                    -1,
                ),
                llm_repetition_penalty=config.get(
                    "repetition_penalty",
                    1.1,
                ),
                llm_top_k=config.get("top_k", 45),
                model=config.get("model", None),
            )

            async_text_generator = AsyncIterator(text_generator)
            async for chunk in async_text_generator:

                if interrupt_flag.is_set():
                    return

                llm_text += chunk

            llm_text = llm_text.strip()

            if not llm_text:
                return

            logger.info(
                f"[{session_id}] AI: {llm_text}"
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": llm_text,
                }
            )

            # =====================================================
            # TTS
            # =====================================================

            if interrupt_flag.is_set():
                return

            tts_start_time = time.time()
            total_tts_audio_bytes = 0

            audio_generator = generate_speech_stream_bytes(
                text=llm_text,
                voice=config.get("voice", "tara"),
                tts_temperature=config.get(
                    "tts_temperature",
                    0.9,
                ),
                tts_top_p=config.get(
                    "tts_top_p",
                    0.9,
                ),
                tts_repetition_penalty=config.get(
                    "tts_repetition_penalty",
                    1.1,
                ),
                buffer_groups_param=config.get(
                    "buffer_groups",
                    5,
                ),
                padding_ms_param=config.get(
                    "padding_ms",
                    0,
                ),
                min_decode_batch_groups_param=config.get(
                    "min_decode_batch_groups",
                    7,
                ),
            )

            async_audio_generator = AsyncIterator(audio_generator)
            async for audio_chunk in async_audio_generator:

                if interrupt_flag.is_set():

                    clear_event = {
                        "event": "clear",
                        "stream_sid": stream_sid,
                    }

                    try:
                        await websocket.send_text(
                            json.dumps(clear_event)
                        )
                    except Exception:
                        pass

                    return

                if not audio_chunk:
                    continue

                total_tts_audio_bytes += len(audio_chunk) if isinstance(audio_chunk, bytes) else audio_chunk.nbytes

                # =================================================
                # NORMALIZE INPUT AUDIO
                # =================================================

                # Orpheus returns FLOAT32 PCM audio @ 24kHz
                # We must convert correctly before resampling.

                if isinstance(audio_chunk, np.ndarray):

                    audio_float = audio_chunk.astype(np.float32)

                else:
                    # Raw bytes from generator -> FLOAT32 PCM
                    audio_float = np.frombuffer(
                        audio_chunk,
                        dtype=np.float32,
                    )

                # =================================================
                # CLIP SAFELY
                # =================================================

                audio_float = np.clip(
                    audio_float,
                    -1.0,
                    1.0,
                )

                # =================================================
                # RESAMPLE 24kHz -> 8kHz FOR EXOTEL
                # =================================================

                exotel_rate = config.get("sample_rate", 8000)
                if TARGET_SAMPLE_RATE != exotel_rate:
                    resampled = resampy.resample(
                        audio_float,
                        TARGET_SAMPLE_RATE,
                        exotel_rate,
                    )
                else:
                    resampled = audio_float

                # =================================================
                # FLOAT32 -> PCM16 LITTLE ENDIAN
                # =================================================

                resampled = np.clip(
                    resampled,
                    -1.0,
                    1.0,
                )

                pcm16 = (
                    resampled * 32767.0
                ).astype(np.int16)

                final_bytes = pcm16.tobytes()

                # =================================================
                # SEND TO EXOTEL
                # =================================================

                await send_exotel_media(final_bytes)

            logger.info(
                f"[{session_id}] TTS completed"
            )

            # --- Wait for client playback to finish (echo prevention) ---
            if total_tts_audio_bytes > 0:
                # FLOAT32 PCM = 4 bytes per sample, at TARGET_SAMPLE_RATE
                total_samples = total_tts_audio_bytes / 4
                playback_duration = total_samples / TARGET_SAMPLE_RATE
                elapsed = time.time() - tts_start_time
                remaining = playback_duration - elapsed + 0.5
                if remaining > 0:
                    logger.info(f"[{session_id}] Waiting {remaining:.1f}s for playback (echo prevention)")
                    try:
                        await asyncio.wait_for(interrupt_flag.wait(), timeout=remaining)
                        logger.info(f"[{session_id}] Barge-in during playback wait")
                    except asyncio.TimeoutError:
                        pass

        except asyncio.CancelledError:

            logger.info(
                f"[{session_id}] Pipeline cancelled"
            )

        except Exception as e:

            logger.exception(
                f"[{session_id}] Pipeline error: {e}"
            )

        finally:

            # Only flush and reset if we were NOT interrupted (i.e. normal completion)
            if not interrupt_flag.is_set():
                flushed = 0
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                        flushed += 1
                    except asyncio.QueueEmpty:
                        break
                if flushed > 0:
                    logger.info(f"[{session_id}] Flushed {flushed} backlogged audio chunks (echo prevention)")
                vad_processor.reset()
            else:
                logger.info(f"[{session_id}] Interrupted by barge-in, preserving audio queue and VAD state")

            is_processing = False

            interrupt_flag.clear()

    # =========================================================
    # START RECEIVER
    # =========================================================

    receiver_task = asyncio.create_task(
        _audio_receiver()
    )

    # =========================================================
    # MAIN AUDIO LOOP
    # =========================================================
    residual_audio = np.array([], dtype=np.float32)

    try:

        while True:

            try:

                pcm_chunk = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.5,
                )

            except asyncio.TimeoutError:
                continue

            if pcm_chunk is None:
                break

            try:

                int16_samples = np.frombuffer(
                    pcm_chunk,
                    dtype=np.int16,
                )

                float32_samples = (
                    int16_samples.astype(np.float32)
                    / 32768.0
                )

            except Exception as e:

                logger.warning(
                    f"[{session_id}] Decode error: {e}"
                )

                continue

            # =============================================
            # RESAMPLE TO VAD RATE
            # =============================================

            client_rate = config.get(
                "sample_rate",
                8000,
            )

            if client_rate != VAD_SAMPLE_RATE:

                from audio_utils import resample_audio

                float32_samples = resample_audio(
                    float32_samples,
                    client_rate,
                    VAD_SAMPLE_RATE,
                )

            # =============================================
            # PROCESS IN VAD CHUNKS
            # =============================================
            combined_samples = np.concatenate([residual_audio, float32_samples])
            offset = 0

            while offset + VAD_CHUNK_SAMPLES <= len(combined_samples):
                chunk = combined_samples[offset : offset + VAD_CHUNK_SAMPLES]
                offset += VAD_CHUNK_SAMPLES

                # =========================================
                # BARGE-IN DETECTION
                # =========================================
                if is_processing:
                    utterance = vad_processor.process_chunk(chunk)
                    current_speaking = vad_processor.is_speech_active
                    
                    if current_speaking != prev_speaking:
                        prev_speaking = current_speaking
                        logger.info(
                            f"[{session_id}] Speaking={current_speaking}"
                        )

                    if current_speaking:
                        logger.info(
                            f"[{session_id}] BARGE-IN DETECTED (speech active)"
                        )
                        interrupt_flag.set()
                        clear_event = {
                            "event": "clear",
                            "stream_sid": stream_sid,
                        }
                        try:
                            await websocket.send_text(json.dumps(clear_event))
                        except Exception:
                            pass

                        if pipeline_task and not pipeline_task.done():
                            pipeline_task.cancel()

                        is_processing = False
                    continue

                # =========================================
                # NORMAL VAD
                # =========================================
                utterance = vad_processor.process_chunk(chunk)
                current_speaking = vad_processor.is_speech_active

                if current_speaking != prev_speaking:
                    prev_speaking = current_speaking
                    logger.info(
                        f"[{session_id}] Speaking={current_speaking}"
                    )

                # =========================================
                # FINAL UTTERANCE
                # =========================================
                if utterance is not None:
                     logger.info(
                         f"[{session_id}] Utterance ready {len(utterance)/VAD_SAMPLE_RATE:.2f}s"
                     )
                     pipeline_task = asyncio.create_task(
                         _process_pipeline(utterance)
                     )

            # Save remaining samples for next packet
            residual_audio = combined_samples[offset:]

    except WebSocketDisconnect:

        logger.info(
            f"[{session_id}] WebSocket disconnected"
        )

    except Exception as e:

        logger.exception(
            f"[{session_id}] WebSocket error: {e}"
        )

    finally:

        receiver_task.cancel()

        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info(
            f"[{session_id}] Session ended"
        )



@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/script.js")
async def serve_script():
    return FileResponse("static/script.js", media_type="application/javascript")


if __name__ == "__main__":
    logger.info("Starting Jarvis server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)