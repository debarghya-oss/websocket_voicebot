"""
LLM API endpoints.

Provides:
  - GET  /api/llm/models
  - POST /api/llm/chat/stream
  - WS   /ws/llm
"""
import logging

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from voicebot.config import SERVER_BASE_URL
from voicebot.models import LLMChatRequest
from voicebot.services.llm_service import generate_llm_text_stream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/llm/models")
async def list_llm_models():
    """Returns the list of available LLM models from the configured backend."""
    url = f"{SERVER_BASE_URL}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            ids = [m["id"] for m in data.get("data", [])]
            return JSONResponse({"models": ids})
    except Exception as e:
        logger.warning(f"Could not fetch models from {url}: {e}")
        return JSONResponse({"models": [], "error": str(e)})


@router.post("/api/llm/chat/stream", summary="Stream LLM Chat Completions")
async def llm_chat_stream(request_data: LLMChatRequest):
    """HTTP streaming endpoint for LLM chat completions."""
    logger.info("LLM: POST /api/llm/chat/stream")
    gen = generate_llm_text_stream(
        prompt=request_data.prompt, history=request_data.history,
        llm_temperature=request_data.temperature, llm_top_p=request_data.top_p,
        llm_max_tokens=request_data.max_tokens,
        llm_repetition_penalty=request_data.repetition_penalty,
        llm_top_k=request_data.top_k, model=request_data.model,
    )
    return StreamingResponse(gen, media_type="text/event-stream")


@router.websocket("/ws/llm")
async def websocket_llm_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming LLM output as JSON chunks."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON payload")

        logger.info("WS LLM: Received request payload")
        req = LLMChatRequest(**data)

        gen = generate_llm_text_stream(
            prompt=req.prompt, history=req.history,
            llm_temperature=req.temperature, llm_top_p=req.top_p,
            llm_max_tokens=req.max_tokens,
            llm_repetition_penalty=req.repetition_penalty,
            llm_top_k=req.top_k, model=req.model,
        )

        for chunk in gen:
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
