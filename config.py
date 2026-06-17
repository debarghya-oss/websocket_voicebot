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
TTS_MODEL = os.getenv("TTS_MODEL", "testfinetune")
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_PROMPT_STOP_TOKENS = ["<|eot_id|>", "<|audio|>"]
DEFAULT_TTS_TEMP = 0.9
DEFAULT_TTS_TOP_P = 0.9
DEFAULT_TTS_REP_PENALTY = 1.2  # Bumped from 1.1 to reduce TTS repetition

# --- TTS ENGINE SELECTION ---
TTS_ENGINES = ["orpheus", "parler"]  # Available TTS engines
DEFAULT_TTS_ENGINE = os.getenv("DEFAULT_TTS_ENGINE", "orpheus")
CURRENT_TTS_ENGINE = DEFAULT_TTS_ENGINE  # Will be set dynamically

# --- PARLER TTS VOICE DESCRIPTIONS ---
PARLER_VOICE_DESCRIPTIONS = {
    "female_clear": "A female speaker delivers clear speech with moderate speed and high quality.",
    "male_clear": "A male speaker delivers clear speech with moderate speed and high quality.",
    "female_expressive": "A female speaker delivers expressive and animated speech with moderate speed.",
    "male_expressive": "A male speaker delivers expressive and animated speech with moderate speed.",
    "female_slow": "A female speaker delivers clear speech at a slow pace with high quality.",
    "male_slow": "A male speaker delivers clear speech at a slow pace with high quality.",
}
DEFAULT_PARLER_VOICE = "female_clear"

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

# --- TTS ANTI-REPETITION (server-side — zero client latency) ---
# Frequency penalty: penalizes tokens proportional to occurrence count (llama.cpp param)
TTS_FREQUENCY_PENALTY = float(os.getenv("TTS_FREQUENCY_PENALTY", "0.5"))
# Presence penalty: penalizes any token that has already appeared
TTS_PRESENCE_PENALTY = float(os.getenv("TTS_PRESENCE_PENALTY", "0.3"))
# Repetition penalty lookback window (tokens); larger = catches longer loops
TTS_REPEAT_LAST_N = int(os.getenv("TTS_REPEAT_LAST_N", "256"))

# --- TTS RUNTIME REPETITION GUARD (lightweight, runs at intervals) ---
# Max generated codes = text_chars / CHARS_PER_SEC * MULTIPLIER * codes_per_sec
TTS_MAX_DURATION_MULTIPLIER = float(os.getenv("TTS_MAX_DURATION_MULTIPLIER", "6.0"))
TTS_ESTIMATED_CHARS_PER_SEC = float(os.getenv("TTS_ESTIMATED_CHARS_PER_SEC", "15.0"))
# Pattern detection: min length of repeated pattern (3 groups × 7 codes)
TTS_REPEAT_PATTERN_MIN_LEN = int(os.getenv("TTS_REPEAT_PATTERN_MIN_LEN", "21"))
# How many consecutive repeats of the pattern before we abort
TTS_REPEAT_PATTERN_THRESHOLD = int(os.getenv("TTS_REPEAT_PATTERN_THRESHOLD", "3"))
# Only run pattern check every N codes received (avoids hot-path overhead)
TTS_REPEAT_CHECK_INTERVAL = int(os.getenv("TTS_REPEAT_CHECK_INTERVAL", "70"))

# --- VOICE CONSTANTS ---
ALL_VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
DEFAULT_TTS_VOICE = ALL_VOICES[0]
TTS_FAILED_MSG = "(TTS generation failed or produced no audio)"

# --- STT CONSTANTS ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
TEMP_AUDIO_DIR = "temp_stt_audio_files"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- VAD CONSTANTS (Silero VAD) ---
VAD_SAMPLE_RATE = 16000  # Silero VAD operates at 16kHz
VAD_CHUNK_SAMPLES = 512  # 32ms chunks at 16kHz (512 / 16000 = 0.032s)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.85"))  # High = only close-mic speech triggers
VAD_BARGEIN_THRESHOLD = float(os.getenv("VAD_BARGEIN_THRESHOLD", "0.70"))  # Barge-in: still needs clear speech
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "1500"))  # Silence to trigger end-of-speech (1.5s for natural pauses)
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "30"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))  # Min speech duration to register

# --- STREAMING CONSTANTS ---
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"

# --- DEVICE SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
