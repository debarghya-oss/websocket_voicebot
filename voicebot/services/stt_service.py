"""
Speech-to-Text Service.

Manages the Whisper and Indic Conformer model lifecycles and provides
transcription entry-points used by the STT API endpoints.
"""
import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from typing import Any, Optional

import torch
import torchaudio

from voicebot.config import (
    DEVICE,
    INDIC_DEFAULT_DECODE_MODE,
    INDIC_DEFAULT_LANGUAGE,
    INDIC_MODEL_NAME,
    INDIC_TARGET_SAMPLE_RATE,
    TEMP_AUDIO_DIR,
    WHISPER_MODEL_NAME,
)
from voicebot.models import STTResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model instances (populated at startup)
# ---------------------------------------------------------------------------
whisper_model: Optional[Any] = None
indic_model: Optional[Any] = None


# ================================================================
# WHISPER
# ================================================================

def load_whisper_model() -> Optional[Any]:
    """Load the Whisper model for STT."""
    global whisper_model
    logger.info("=== Starting Whisper Model Loading ===")
    try:
        import whisper
        logger.info("✓ Whisper library imported successfully.")
    except ImportError as e:
        logger.error(f"✗ Whisper library not found: {e}  |  pip install -U openai-whisper")
        return None
    try:
        logger.info(f"Loading Whisper model: '{WHISPER_MODEL_NAME}' on {DEVICE}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
            if DEVICE == "cpu":
                warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        whisper_model_instance = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
        if whisper_model_instance is None:
            logger.error("✗ whisper.load_model() returned None")
            return None
        whisper_model = whisper_model_instance
        logger.info(f"✓ Whisper model ('{WHISPER_MODEL_NAME}') loaded to '{DEVICE}'.")
        return whisper_model
    except Exception as e:
        logger.error(f"✗ Error loading Whisper model: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return None


async def whisper_transcribe(audio_file) -> STTResponse:
    """Transcribe audio file to text using Whisper (Bengali)."""
    global whisper_model
    request_id = str(int(time.time() * 1000))
    tmp_path = None

    if whisper_model is None:
        logger.error(f"[{request_id}] Whisper model not loaded.")
        return STTResponse(text="", error="STT service unavailable: Whisper model not loaded.")

    try:
        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, dir=str(TEMP_AUDIO_DIR), suffix=suffix) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"[{request_id}] Audio saved to '{tmp_path}'")
        t0 = time.time()
        result = whisper_model.transcribe(tmp_path, language="bn", fp16=(DEVICE == "cuda"))
        dt = time.time() - t0
        text = result["text"].strip()
        lang = result.get("language", "unknown")
        logger.info(f"[{request_id}] Whisper done in {dt:.3f}s. lang='{lang}' text='{text[:100]}'")
        return STTResponse(text=text, language=lang)
    except Exception as e:
        logger.exception(f"[{request_id}] Whisper transcription error")
        return STTResponse(text="", error=f"Transcription failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as ce:
                logger.warning(f"[{request_id}] Cleanup failed: {ce}")
        await audio_file.close()


# ================================================================
# INDIC CONFORMER
# ================================================================

def load_indic_model() -> Optional[Any]:
    """Load the Indic Conformer multilingual model for STT."""
    global indic_model
    logger.info("=== Starting Indic Conformer Model Loading ===")
    try:
        from transformers import AutoModel
        logger.info("✓ Transformers library imported successfully.")
    except ImportError as e:
        logger.error(f"✗ Transformers not found: {e}  |  pip install transformers")
        return None
    try:
        logger.info(f"Loading Indic Conformer model: '{INDIC_MODEL_NAME}'")
        model_instance = AutoModel.from_pretrained(INDIC_MODEL_NAME, trust_remote_code=True)
        indic_model = model_instance
        logger.info(f"✓ Indic Conformer ('{INDIC_MODEL_NAME}') loaded.")
        return indic_model
    except Exception as e:
        logger.error(f"✗ Error loading Indic Conformer: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return None


def _convert_to_wav(src_path: str, dst_path: str) -> None:
    """Convert any audio file to a 16-kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", str(INDIC_TARGET_SAMPLE_RATE), "-ac", "1", "-f", "wav", dst_path,
    ]
    logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr.decode(errors='replace')}")


def _load_wav_tensor(wav_path: str) -> torch.Tensor:
    """Load a WAV file, mix to mono, resample to INDIC_TARGET_SAMPLE_RATE. Returns (1, T)."""
    wav, sr = torchaudio.load(wav_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != INDIC_TARGET_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=INDIC_TARGET_SAMPLE_RATE)(wav)
    return wav


def _run_indic_inference(wav: torch.Tensor, language: str, decode_mode: str) -> str:
    """Run the Indic Conformer model synchronously."""
    result = indic_model(wav, language, decode_mode)
    return result.strip() if isinstance(result, str) else str(result).strip()


async def indic_transcribe(
    audio_file,
    language: str = INDIC_DEFAULT_LANGUAGE,
    decode_mode: str = INDIC_DEFAULT_DECODE_MODE,
) -> STTResponse:
    """Transcribe an uploaded audio file using the Indic Conformer model."""
    global indic_model
    request_id = str(int(time.time() * 1000))

    if indic_model is None:
        logger.error(f"[{request_id}] Indic model not loaded.")
        return STTResponse(text="", error="STT service unavailable: Indic Conformer model not loaded.")

    raw_path = None
    wav_path = None

    try:
        suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, dir=str(TEMP_AUDIO_DIR), suffix=suffix) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            raw_path = tmp.name
        logger.info(f"[{request_id}] Indic STT: '{audio_file.filename}' → '{raw_path}'")

        needs_conversion = suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}
        if needs_conversion:
            wav_path = raw_path + ".wav"
            logger.info(f"[{request_id}] Converting '{suffix}' → WAV via ffmpeg")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _convert_to_wav, raw_path, wav_path)
            audio_to_load = wav_path
        else:
            audio_to_load = raw_path

        wav = _load_wav_tensor(audio_to_load)
        logger.info(f"[{request_id}] Tensor shape={tuple(wav.shape)}, lang='{language}', mode='{decode_mode}'")

        t0 = time.time()
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _run_indic_inference, wav, language, decode_mode)
        dt = time.time() - t0
        logger.info(f"[{request_id}] Indic STT done in {dt:.3f}s. text='{text[:120]}'")
        return STTResponse(text=text, language=language)

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        logger.error(f"[{request_id}] ffmpeg failed.\n{stderr}")
        return STTResponse(text="", error=f"Audio conversion failed: {stderr[:200]}")
    except Exception as e:
        logger.exception(f"[{request_id}] Indic STT error for '{audio_file.filename}'")
        return STTResponse(text="", error=f"Transcription failed: {e}")
    finally:
        for path in (wav_path, raw_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as ce:
                    logger.warning(f"[{request_id}] Could not remove '{path}': {ce}")
        await audio_file.close()
