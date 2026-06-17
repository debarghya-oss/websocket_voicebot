"""
Speech-to-Text Engine Module
Handles Whisper model loading and STT transcription
"""
import logging
import warnings
import time
import os
import shutil
import tempfile
import asyncio
from typing import Optional, Any

from config import TEMP_AUDIO_DIR, WHISPER_MODEL_NAME, DEVICE
from models import STTResponse

logger = logging.getLogger(__name__)

# Global Whisper model instance
whisper_model: Optional[Any] = None


def load_whisper_model() -> Optional[Any]:
    """Load the Whisper model for STT."""
    global whisper_model
    
    logger.info("=== Starting Whisper Model Loading ===")
    
    # Step 1: Import Whisper
    try:
        import whisper
        logger.info("✓ Whisper library imported successfully.")
    except ImportError as import_err:
        logger.error(f"✗ Whisper library not found. Error: {import_err}")
        logger.error("  Install with: pip install -U openai-whisper")
        return None

    # Step 2: Load the model
    try:
        logger.info(f"Loading Whisper model: '{WHISPER_MODEL_NAME}'")
        logger.info(f"Device: {DEVICE}")
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
            if DEVICE == "cpu":
                warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        
        logger.info(f"Calling whisper.load_model() with model='{WHISPER_MODEL_NAME}', device='{DEVICE}'...")
        whisper_model_instance = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
        
        if whisper_model_instance is None:
            logger.error("✗ whisper.load_model() returned None")
            return None
        
        whisper_model = whisper_model_instance
        logger.info(f"✓ Whisper model ('{WHISPER_MODEL_NAME}') loaded successfully to '{DEVICE}'.")
        logger.info("=== Whisper Model Loading Complete ===")
        return whisper_model
        
    except Exception as e:
        logger.error(f"✗ Error loading Whisper model: {type(e).__name__}: {e}")
        logger.exception(f"Full traceback for Whisper STT model ('{WHISPER_MODEL_NAME}') loading:")
        return None


async def transcribe_audio(audio_file) -> STTResponse:
    """Transcribe audio file to text using Whisper (Bengali)."""
    global whisper_model
    
    request_id = str(int(time.time() * 1000))  # Use timestamp as request ID
    tmp_audio_file_path = None
    
    if whisper_model is None:
        logger.error(f"[{request_id}] STT: Whisper model is None - not initialized.")
        logger.error(f"[{request_id}] STT: Check startup logs - model may have failed to load.")
        return STTResponse(text="", error="STT service unavailable: Whisper model not loaded. Check server logs.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_AUDIO_DIR, suffix=os.path.splitext(audio_file.filename)[1] or ".wav") as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            tmp_audio_file_path = tmp_file.name
        
        logger.info(f"Audio file '{audio_file.filename}' saved temporarily to '{tmp_audio_file_path}'")
        
        stt_start_time = time.time()
        result = whisper_model.transcribe(tmp_audio_file_path, language="bn", fp16=(DEVICE=="cuda"))
        stt_duration = time.time() - stt_start_time
        
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", "unknown") 

        logger.info(f"Transcription successful in {stt_duration:.3f}s. Language: '{detected_language}'. Text: '{transcribed_text[:100]}...'")
        return STTResponse(text=transcribed_text, language=detected_language)

    except Exception as e:
        logger.exception(f"Error during transcription for file '{audio_file.filename}'")
        return STTResponse(text="", error=f"Transcription failed: {str(e)}")
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.remove(tmp_audio_file_path)
                logger.info(f"Cleaned up temporary audio file: {tmp_audio_file_path}")
            except Exception as cleanup_e:
                logger.warning(f"Failed to clean up temporary audio file '{tmp_audio_file_path}': {cleanup_e}")
        await audio_file.close()


def _run_whisper_inference(audio_array: Any) -> dict:
    return whisper_model.transcribe(audio_array, language="bn", fp16=(DEVICE=="cuda"))


async def transcribe_numpy(audio_array: Any) -> STTResponse:
    """Transcribe in-memory NumPy float32 16kHz audio array using Whisper."""
    global whisper_model
    
    request_id = str(int(time.time() * 1000))
    if whisper_model is None:
        logger.error(f"[{request_id}] STT (numpy): Whisper model is None - not initialized.")
        return STTResponse(text="", error="STT service unavailable: Whisper model not loaded.")

    try:
        stt_start_time = time.time()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run_whisper_inference, audio_array)
        stt_duration = time.time() - stt_start_time
        
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", "bn") 

        logger.info(f"Whisper STT (numpy) successful in {stt_duration:.3f}s. Language: '{detected_language}'. Text: '{transcribed_text[:100]}...'")
        return STTResponse(text=transcribed_text, language=detected_language)

    except Exception as e:
        logger.exception("Error during Whisper numpy transcription")
        return STTResponse(text="", error=f"Transcription failed: {str(e)}")
