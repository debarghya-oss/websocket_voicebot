"""
TTS API endpoints.

Provides:
  - POST /api/tts/stream   (HTTP streaming)
  - WS   /ws/tts           (WebSocket streaming with base64 audio chunks)
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from voicebot.config import TARGET_SAMPLE_RATE
from voicebot.models import TTSRequest
from voicebot.services import tts_service
from voicebot.utils.audio_utils import encode_audio_to_base64

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/tts/stream")
async def tts_stream_endpoint(request: TTSRequest):
    """Text-to-Speech streaming endpoint."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] TTS: POST /api/tts/stream")

    if tts_service.snac_model is None:
        raise HTTPException(status_code=503, detail="SNAC model not loaded. TTS unavailable.")

    gen = tts_service.generate_speech_stream_bytes(
        text=request.text, voice=request.voice,
        tts_temperature=request.tts_temperature, tts_top_p=request.tts_top_p,
        tts_repetition_penalty=request.tts_repetition_penalty,
        buffer_groups_param=request.buffer_groups,
        padding_ms_param=request.padding_ms,
        min_decode_batch_groups_param=request.min_decode_batch_groups,
    )
    headers = {"X-Sample-Rate": str(TARGET_SAMPLE_RATE), "X-Audio-Format": "FLOAT32_PCM"}
    return StreamingResponse(gen, media_type="audio/octet-stream", headers=headers)


@router.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS audio bytes in base64 format."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WebSocket TTS: Received request payload")
        req = TTSRequest(**data)
        rid = str(uuid.uuid4())
        logger.info(f"[{rid}] WS TTS: Text: '{req.text[:100]}...'")

        await websocket.send_json({
            "type": "tts_started",
            "sample_rate": TARGET_SAMPLE_RATE,
            "audio_format": "FLOAT32_PCM_BASE64",
            "encoding": "base64",
        })

        gen = tts_service.generate_speech_stream_bytes(
            text=req.text, voice=req.voice,
            tts_temperature=req.tts_temperature, tts_top_p=req.tts_top_p,
            tts_repetition_penalty=req.tts_repetition_penalty,
            buffer_groups_param=req.buffer_groups,
            padding_ms_param=req.padding_ms,
            min_decode_batch_groups_param=req.min_decode_batch_groups,
        )

        sent_any = False
        chunk_n = 0
        total_bytes = 0
        for chunk in gen:
            if chunk:
                sent_any = True
                chunk_n += 1
                total_bytes += len(chunk)
                b64 = encode_audio_to_base64(chunk)
                await websocket.send_json({
                    "type": "tts_audio_chunk",
                    "audio": b64,
                    "bytes_length": len(chunk),
                })

        if not sent_any:
            await websocket.send_json({"type": "tts_warning", "message": "No audio frames generated."})
        else:
            logger.info(f"[{rid}] TTS done: {chunk_n} chunks, {total_bytes} bytes")

        await websocket.send_json({"type": "tts_done"})

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
