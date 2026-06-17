import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Any

import torch
import torchaudio

from config import TEMP_AUDIO_DIR
from models import STTResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model instance (populated by load_indic_model() at startup)
# ---------------------------------------------------------------------------
indic_model: Optional[Any] = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDIC_DEFAULT_LANGUAGE  = os.getenv("INDIC_LANGUAGE",    "bn")
INDIC_DEFAULT_DECODE_MODE = os.getenv("INDIC_DECODE_MODE", "ctc")   # "ctc" or "rnnt"
INDIC_MODEL_NAME        = os.getenv(
    "INDIC_MODEL_NAME", "ai4bharat/indic-conformer-600m-multilingual"
)
TARGET_SAMPLE_RATE = 16000   # Indic Conformer expects 16 kHz mono


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_indic_model() -> Optional[Any]:
    """Load the Indic Conformer multilingual model for STT."""
    global indic_model

    logger.info("=== Starting Indic Conformer Model Loading ===")

    # Import transformers
    try:
        from transformers import AutoModel
        logger.info("✓ Transformers library imported successfully.")
    except ImportError as e:
        logger.error(f"✗ Transformers not found: {e}  |  pip install transformers")
        return None

    # Load model
    try:
        logger.info(f"Loading Indic Conformer model: '{INDIC_MODEL_NAME}'")
        model_instance = AutoModel.from_pretrained(
            INDIC_MODEL_NAME,
            trust_remote_code=True,
        )
        indic_model = model_instance
        logger.info(f"✓ Indic Conformer ('{INDIC_MODEL_NAME}') loaded.")
        logger.info("=== Indic Conformer Model Loading Complete ===")
        return indic_model

    except Exception as e:
        logger.error(f"✗ Error loading Indic Conformer model: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _convert_to_wav(src_path: str, dst_path: str) -> None:
    """
    Convert any audio file (webm, ogg, mp4 …) to a 16-kHz mono WAV using
    ffmpeg.  Raises subprocess.CalledProcessError on failure.
    """
    cmd = [
        "ffmpeg", "-y",              # overwrite output if exists
        "-i", src_path,              # input (any format)
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",                  # mono
        "-f", "wav",
        dst_path,
    ]
    logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,         # raises CalledProcessError on non-zero exit
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )


def _load_wav_tensor(wav_path: str) -> torch.Tensor:
    """
    Load a WAV file with torchaudio, mix down to mono, and resample to
    TARGET_SAMPLE_RATE if necessary.  Returns a (1, T) float tensor.
    """
    wav, sr = torchaudio.load(wav_path)
    # Mix-down to mono
    wav = torch.mean(wav, dim=0, keepdim=True)
    # Resample (usually already 16 kHz after ffmpeg, but guard anyway)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
        )(wav)
    return wav


def _run_inference(wav: torch.Tensor, language: str, decode_mode: str) -> str:
    """
    Run the Indic Conformer model synchronously.
    Wrapped in a thread so the async endpoint stays non-blocking.
    """
    result = indic_model(wav, language, decode_mode)
    if isinstance(result, str):
        return result.strip()
    return str(result).strip()


# ---------------------------------------------------------------------------
# Public async transcription entry-point
# ---------------------------------------------------------------------------
async def transcribe_audio(
    audio_file,
    language:    str = INDIC_DEFAULT_LANGUAGE,
    decode_mode: str = INDIC_DEFAULT_DECODE_MODE,
) -> STTResponse:
    """
    Transcribe an uploaded audio file using the Indic Conformer model.

    Args:
        audio_file:  FastAPI UploadFile object (any format; webm, ogg, wav …).
        language:    BCP-47 code understood by the model (e.g. 'bn', 'hi', 'ta').
        decode_mode: 'ctc' (default) or 'rnnt'.

    Returns:
        STTResponse with .text and optional .error.
    """
    global indic_model

    request_id = str(int(time.time() * 1000))

    if indic_model is None:
        logger.error(f"[{request_id}] Indic STT: model not loaded.")
        return STTResponse(
            text="",
            error="STT service unavailable: Indic Conformer model not loaded.",
        )

    raw_path = None   # original uploaded file (may be .webm etc.)
    wav_path = None   # converted .wav (if conversion was needed)

    try:
        # ------------------------------------------------------------------ #
        # 1. Save uploaded bytes to a temp file                              #
        # ------------------------------------------------------------------ #
        suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(
            delete=False, dir=TEMP_AUDIO_DIR, suffix=suffix
        ) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            raw_path = tmp.name

        logger.info(
            f"[{request_id}] Indic STT: '{audio_file.filename}' → '{raw_path}'"
        )

        # ------------------------------------------------------------------ #
        # 2. Convert to WAV if torchaudio can't read it directly             #
        #    (i.e. anything that isn't already a wav/flac/mp3)               #
        # ------------------------------------------------------------------ #
        needs_conversion = suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}

        if needs_conversion:
            wav_path = raw_path + ".wav"
            logger.info(
                f"[{request_id}] Indic STT: Converting '{suffix}' → WAV via ffmpeg …"
            )
            # Run ffmpeg in a thread so we don't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _convert_to_wav, raw_path, wav_path)
            audio_to_load = wav_path
        else:
            audio_to_load = raw_path

        # ------------------------------------------------------------------ #
        # 3. Load & pre-process tensor                                       #
        # ------------------------------------------------------------------ #
        wav = _load_wav_tensor(audio_to_load)
        logger.info(
            f"[{request_id}] Indic STT: Tensor shape={tuple(wav.shape)}, "
            f"language='{language}', decode_mode='{decode_mode}'"
        )

        # ------------------------------------------------------------------ #
        # 4. Run inference in a thread (blocking call)                       #
        # ------------------------------------------------------------------ #
        stt_start = time.time()
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(
            None, _run_inference, wav, language, decode_mode
        )
        stt_duration = time.time() - stt_start

        logger.info(
            f"[{request_id}] Indic STT: Done in {stt_duration:.3f}s. "
            f"Language='{language}'. Text='{transcribed_text[:120]}'"
        )
        return STTResponse(text=transcribed_text, language=language)

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        logger.error(
            f"[{request_id}] Indic STT: ffmpeg conversion failed.\n{stderr}"
        )
        return STTResponse(
            text="", error=f"Audio conversion failed (ffmpeg): {stderr[:200]}"
        )

    except Exception as e:
        logger.exception(
            f"[{request_id}] Indic STT: Unexpected error for '{audio_file.filename}'"
        )
        return STTResponse(text="", error=f"Transcription failed: {str(e)}")

    finally:
        # Clean up temp files
        for path in (wav_path, raw_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[{request_id}] Indic STT: Cleaned up '{path}'")
                except Exception as ce:
                    logger.warning(
                        f"[{request_id}] Indic STT: Could not remove '{path}': {ce}"
                    )
        await audio_file.close()


async def transcribe_numpy(
    audio_array: Any,
    language:    str = INDIC_DEFAULT_LANGUAGE,
    decode_mode: str = INDIC_DEFAULT_DECODE_MODE,
) -> STTResponse:
    """
    Transcribe an in-memory 1D float32 NumPy array at 16kHz using the Indic Conformer model.
    """
    global indic_model

    request_id = str(int(time.time() * 1000))

    if indic_model is None:
        logger.error(f"[{request_id}] Indic STT (numpy): model not loaded.")
        return STTResponse(
            text="",
            error="STT service unavailable: Indic Conformer model not loaded.",
        )

    try:
        import numpy as np
        # Convert numpy array to torch Tensor and format it as shape (1, T)
        wav = torch.from_numpy(audio_array).float().unsqueeze(0)
        
        logger.info(
            f"[{request_id}] Indic STT (numpy): Tensor shape={tuple(wav.shape)}, "
            f"language='{language}', decode_mode='{decode_mode}'"
        )

        stt_start = time.time()
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(
            None, _run_inference, wav, language, decode_mode
        )
        stt_duration = time.time() - stt_start

        logger.info(
            f"[{request_id}] Indic STT (numpy): Done in {stt_duration:.3f}s. "
            f"Language='{language}'. Text='{transcribed_text[:120]}'"
        )
        return STTResponse(text=transcribed_text, language=language)

    except Exception as e:
        logger.exception(
            f"[{request_id}] Indic STT (numpy): Unexpected error"
        )
        return STTResponse(text="", error=f"Transcription failed: {str(e)}")