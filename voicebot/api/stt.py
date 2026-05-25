"""
STT API endpoints.

Provides:
  - POST /api/stt/set_engine
  - POST /api/stt/transcribe
  - WS   /ws/stt
"""
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Query, UploadFile, WebSocket, WebSocketDisconnect

from voicebot.models import STTResponse
from voicebot.services import stt_service
from voicebot.utils.audio_utils import decode_audio_from_base64

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/stt/set_engine")
async def set_stt_engine(body: dict):
    engine = body.get("engine")
    return {"ok": True, "engine": engine}


@router.post("/api/stt/transcribe", response_model=STTResponse)
async def stt_transcribe_endpoint(
    audio_file: UploadFile = File(...),
    engine: str = Query(default="indic", description="STT engine: 'whisper' or 'indic'"),
    language: str = Query(default="bn", description="Language code for Indic engine"),
    decode_mode: str = Query(default="ctc", description="Decode mode: 'ctc' or 'rnnt'"),
):
    """Speech-to-Text transcription endpoint."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] STT: engine={engine} file={audio_file.filename}")

    eng = engine.lower().strip()
    if eng == "indic":
        if stt_service.indic_model is None:
            return STTResponse(text="", error="STT unavailable: Indic model not loaded.")
        return await stt_service.indic_transcribe(audio_file, language=language, decode_mode=decode_mode)
    else:
        if stt_service.whisper_model is None:
            return STTResponse(text="", error="STT unavailable: Whisper model not loaded.")
        return await stt_service.whisper_transcribe(audio_file)


@router.websocket("/ws/stt")
async def websocket_stt_endpoint(websocket: WebSocket):
    """WebSocket endpoint for STT (base64 audio upload and transcription)."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WS STT: Received request payload")
        engine = data.get("engine", "indic")
        language = data.get("language", "bn")
        decode_mode = data.get("decode_mode", "ctc")
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            raise ValueError("No audio data provided")

        audio_bytes = decode_audio_from_base64(audio_b64)
        if not audio_bytes:
            raise ValueError("Failed to decode audio from base64")

        logger.info(f"WS STT: {len(audio_bytes)} bytes received")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            await websocket.send_json({"type": "stt_processing", "message": "Transcribing audio..."})

            if engine == "whisper":
                transcript = stt_service.whisper_transcribe(tmp_path)
            elif engine == "indic":
                transcript = stt_service.indic_transcribe(tmp_path, language=language, decode_mode=decode_mode)
            else:
                transcript = stt_service.indic_transcribe(tmp_path, language=language, decode_mode=decode_mode)

            await websocket.send_json({
                "type": "stt_result",
                "text": transcript,
                "engine": engine,
                "language": language,
            })
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

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
