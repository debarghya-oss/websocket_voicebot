"""
Pydantic Models for Request/Response validation.

All API and WebSocket schemas are defined here as a single source of truth.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel

from voicebot.config import (
    DEFAULT_LMSTUDIO_MAX_TOKENS,
    DEFAULT_LMSTUDIO_REP_PENALTY,
    DEFAULT_LMSTUDIO_TEMP,
    DEFAULT_LMSTUDIO_TOP_K,
    DEFAULT_LMSTUDIO_TOP_P,
    DEFAULT_MIN_DECODE_BATCH_GROUPS,
    DEFAULT_TTS_REP_PENALTY,
    DEFAULT_TTS_TEMP,
    DEFAULT_TTS_TOP_P,
    DEFAULT_TTS_VOICE,
    TTS_STREAM_MIN_GROUPS,
    TTS_STREAM_SILENCE_MS,
)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    """Text-to-Speech request model."""

    text: str
    voice: str = DEFAULT_TTS_VOICE
    tts_temperature: float = DEFAULT_TTS_TEMP
    tts_top_p: float = DEFAULT_TTS_TOP_P
    tts_repetition_penalty: float = DEFAULT_TTS_REP_PENALTY
    buffer_groups: int = TTS_STREAM_MIN_GROUPS
    padding_ms: int = TTS_STREAM_SILENCE_MS
    min_decode_batch_groups: int = DEFAULT_MIN_DECODE_BATCH_GROUPS


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
class STTResponse(BaseModel):
    """Speech-to-Text response model."""

    text: str
    language: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLMChatRequest(BaseModel):
    """LLM chat completion request model."""

    prompt: str
    history: List[Dict[str, str]] = []
    model: Optional[str] = None  # Allow frontend to specify model
    temperature: float = DEFAULT_LMSTUDIO_TEMP
    top_p: float = DEFAULT_LMSTUDIO_TOP_P
    max_tokens: int = DEFAULT_LMSTUDIO_MAX_TOKENS
    repetition_penalty: float = DEFAULT_LMSTUDIO_REP_PENALTY
    top_k: Optional[int] = DEFAULT_LMSTUDIO_TOP_K
