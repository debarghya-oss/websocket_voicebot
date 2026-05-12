"""
Pydantic Models for Request/Response validation
"""
from pydantic import BaseModel
from typing import Optional


class TTSRequest(BaseModel):
    """Text-to-Speech request model"""
    text: str
    voice: str
    tts_temperature: float
    tts_top_p: float
    tts_repetition_penalty: float
    buffer_groups: int
    padding_ms: int
    min_decode_batch_groups: int


class STTResponse(BaseModel):
    """Speech-to-Text response model"""
    text: str
    language: Optional[str] = None
    error: Optional[str] = None
