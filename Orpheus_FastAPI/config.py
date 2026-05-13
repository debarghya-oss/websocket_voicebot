"""
Configuration and Constants Module
Centralizes all environment variables, constants, and configuration settings
"""
import os
import torch

# --- SERVER CONFIGURATION ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")

# --- TTS CONSTANTS ---
TTS_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/completions"
TTS_MODEL = os.getenv("TTS_MODEL", "orpheus-3b-0.1-ft")
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_PROMPT_STOP_TOKENS = ["<|eot_id|>", "<|audio|>"]
DEFAULT_TTS_TEMP = 0.9
DEFAULT_TTS_TOP_P = 0.9
DEFAULT_TTS_REP_PENALTY = 1.1

# --- ORPHEUS AUDIO CODEC CONSTANTS ---
ORPHEUS_MIN_ID = 10
ORPHEUS_TOKENS_PER_LAYER = 4096
ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)

# --- AUDIO PROCESSING CONSTANTS ---
TARGET_SAMPLE_RATE = 24000
TTS_STREAM_MIN_GROUPS = 1
TTS_STREAM_SILENCE_MS = 0
DEFAULT_MIN_DECODE_BATCH_GROUPS = 10
TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS = 100

# --- VOICE CONSTANTS ---
ALL_VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
DEFAULT_TTS_VOICE = ALL_VOICES[0]
TTS_FAILED_MSG = "(TTS generation failed or produced no audio)"

# --- STT CONSTANTS ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
TEMP_AUDIO_DIR = "temp_stt_audio_files"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- STREAMING CONSTANTS ---
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"

# --- DEVICE SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
