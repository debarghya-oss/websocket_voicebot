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
# SNAC model output rate
SNAC_SAMPLE_RATE = 24000
# Target output rate for client (8000 Hz for landline compatibility, 24000 for high-quality)
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "8000"))
# Enable automatic resampling from SNAC (24kHz) to target rate
ENABLE_RESAMPLING = TARGET_SAMPLE_RATE != SNAC_SAMPLE_RATE

# --- IMPROVED CHUNKING STRATEGY FOR GLITCH PREVENTION ---
# Increased buffer to prevent underruns (3 groups = ~33ms at 24kHz before sending)
TTS_STREAM_MIN_GROUPS = int(os.getenv("TTS_STREAM_MIN_GROUPS", "3"))
# Silence padding between chunks to prevent audio overlap/glitching (50ms)
TTS_STREAM_SILENCE_MS = int(os.getenv("TTS_STREAM_SILENCE_MS", "50"))
# Increased batch decode size for smoother chunking (5 groups = ~55ms)
DEFAULT_MIN_DECODE_BATCH_GROUPS = int(os.getenv("DEFAULT_MIN_DECODE_BATCH_GROUPS", "5"))
# Increased timeout to wait for complete code groups (150ms instead of 100ms)
TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS = int(os.getenv("TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS", "150"))
# Audio fade duration for smoother chunk transitions (5ms instead of 1ms)
TTS_AUDIO_FADE_MS = int(os.getenv("TTS_AUDIO_FADE_MS", "5"))

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
