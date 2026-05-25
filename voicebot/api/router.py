"""
Consolidated API router.

Includes all sub-routers (TTS, STT, LLM) so that main.py only needs to
include a single router.
"""
from fastapi import APIRouter

from voicebot.api.tts import router as tts_router
from voicebot.api.stt import router as stt_router
from voicebot.api.llm import router as llm_router

api_router = APIRouter()

api_router.include_router(tts_router, tags=["TTS"])
api_router.include_router(stt_router, tags=["STT"])
api_router.include_router(llm_router, tags=["LLM"])
