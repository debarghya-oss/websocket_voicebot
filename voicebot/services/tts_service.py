"""
Text-to-Speech Service.

Manages the SNAC model lifecycle and provides the streaming audio generator
that converts text → Orpheus audio tokens → SNAC decoded PCM audio bytes.
"""
import json
import logging
import time
import warnings
from typing import Any, Generator, Optional

import numpy as np
import requests

from voicebot.config import (
    DEVICE, ENABLE_RESAMPLING, ORPHEUS_N_LAYERS, SNAC_SAMPLE_RATE,
    SSE_DATA_PREFIX, SSE_DONE_MARKER, STREAM_HEADERS, STREAM_TIMEOUT_SECONDS,
    TARGET_SAMPLE_RATE, TTS_API_ENDPOINT, TTS_AUDIO_FADE_MS, TTS_MODEL,
    TTS_PROMPT_FORMAT, TTS_PROMPT_STOP_TOKENS, TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS,
)
from voicebot.utils.audio_utils import (
    apply_fade, parse_gguf_codes, redistribute_codes, resample_audio,
)

logger = logging.getLogger(__name__)

# Global SNAC model instance
snac_model: Optional[Any] = None


def load_snac_model() -> Optional[Any]:
    """Load the SNAC model for TTS."""
    global snac_model
    try:
        from snac import SNAC
        logger.info("SNAC imported successfully for TTS.")
    except ImportError:
        logger.error("SNAC not found. pip install git+https://github.com/hubertsiuzdak/snac.git")
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="snac.snac")
            try:
                snac_model_instance = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                logger.info("SNAC.from_pretrained called successfully.")
            except Exception as load_e:
                logger.error(f"SNAC.from_pretrained failed: {load_e}", exc_info=True)
                return None
        if snac_model_instance:
            snac_model = snac_model_instance.to(DEVICE).eval()
            logger.info(f"SNAC model loaded successfully to '{DEVICE}'.")
            return snac_model
        else:
            logger.error("SNAC.from_pretrained returned None. SNAC model not loaded.")
            return None
    except Exception:
        logger.exception("Fatal error during SNAC model loading process.")
        return None


def _decode_and_yield(codes, snac):
    """Decode a list of codes through SNAC and return resampled float32 bytes or None."""
    snac_start = time.time()
    audio_chunk = redistribute_codes(codes, snac)
    snac_end = time.time()
    if audio_chunk is not None and audio_chunk.size > 0:
        logger.debug(f"SNAC decoded {len(codes)} codes -> {audio_chunk.size} samples in {snac_end - snac_start:.3f}s.")
        faded = apply_fade(audio_chunk, SNAC_SAMPLE_RATE, fade_ms=TTS_AUDIO_FADE_MS)
        if ENABLE_RESAMPLING:
            faded = resample_audio(faded, SNAC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        return faded.astype(np.float32).tobytes()
    logger.warning(f"SNAC failed to decode {len(codes)} codes in {snac_end - snac_start:.3f}s.")
    return None


def generate_speech_stream_bytes(
    text: str, voice: str, tts_temperature: float, tts_top_p: float,
    tts_repetition_penalty: float, buffer_groups_param: int, padding_ms_param: int,
    min_decode_batch_groups_param: int,
) -> Generator[bytes, None, None]:
    """Generate speech audio stream bytes from text using TTS."""
    global snac_model
    if not text.strip():
        logger.warning("generate_speech_stream_bytes called with empty text.")
        return
    if snac_model is None:
        logger.error("SNAC model not loaded. Cannot generate audio bytes for TTS.")
        yield b''
        return

    cpg = ORPHEUS_N_LAYERS
    min_batch = max(1, min_decode_batch_groups_param) * cpg
    min_batch = max(min_batch, cpg)

    payload = {
        "model": TTS_MODEL,
        "prompt": TTS_PROMPT_FORMAT.format(voice=voice, text=text),
        "temperature": tts_temperature, "top_p": tts_top_p,
        "repeat_penalty": tts_repetition_penalty,
        "n_predict": -1, "stop": TTS_PROMPT_STOP_TOKENS, "stream": True,
    }

    acc = []
    t0 = time.time()
    resp = None
    ok = False
    buf_ready = False
    last_t: Optional[float] = None
    ttft_t0: Optional[float] = None
    ttft_ms: Optional[float] = None

    try:
        logger.info(">>> TTS API: Initiating stream request...")
        resp = requests.post(TTS_API_ENDPOINT, json=payload, headers=STREAM_HEADERS,
                             stream=True, timeout=STREAM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        ttft_t0 = time.time()
        logger.info(f"--- TTS API: Stream connected in {ttft_t0 - t0:.3f}s.")

        for line in resp.iter_lines():
            base = ttft_t0 or t0
            if time.time() - base > STREAM_TIMEOUT_SECONDS:
                logger.error("TTS stream timed out."); break
            if not line:
                continue
            try:
                dl = line.decode(resp.encoding or 'utf-8', errors='ignore')
            except UnicodeDecodeError:
                continue
            if not dl.startswith(SSE_DATA_PREFIX):
                continue
            js = dl[len(SSE_DATA_PREFIX):].strip()
            if js == SSE_DONE_MARKER:
                break
            if not js:
                continue
            try:
                data = json.loads(js)
            except json.JSONDecodeError:
                continue
            ct = ""
            if "content" in data:
                ct = data.get("content", "")
            elif "choices" in data and data["choices"]:
                ch = data["choices"][0]
                ct = ch.get("delta", {}).get("content", "") or ch.get("text", "")
            if ct:
                nc = parse_gguf_codes(ct)
                if nc:
                    if ttft_ms is None and ttft_t0:
                        ttft_ms = (time.time() - ttft_t0) * 1000
                        logger.info(f"--- TTS TTFT: {ttft_ms:.2f} ms ---")
                    acc.extend(nc)
                    last_t = time.time()
                    if not buf_ready and len(acc) >= cpg:
                        buf_ready = True
                    if buf_ready and last_t:
                        dt = (time.time() - last_t) * 1000
                        if dt > TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS or len(acc) >= min_batch:
                            ng = len(acc) // cpg
                            if ng > 0:
                                n = ng * cpg
                                ab = _decode_and_yield(acc[:n], snac_model)
                                acc = acc[n:]
                                if ab:
                                    yield ab; ok = True
                                last_t = time.time()
            # Stop conditions
            if "choices" in data and data["choices"] and data["choices"][0].get("finish_reason"):
                break
            if data.get("stop") is True or data.get("stopped_eos") is True:
                break

        # Flush remaining
        if len(acc) >= cpg:
            ng = len(acc) // cpg
            ab = _decode_and_yield(acc[:ng * cpg], snac_model)
            if ab:
                yield ab; ok = True
    except requests.exceptions.Timeout:
        logger.error("TTS API timed out.", exc_info=True)
    except requests.exceptions.RequestException as e:
        logger.exception(f"TTS API request failed: {e}")
    except Exception as e:
        logger.exception(f"TTS stream error: {e}")
    finally:
        if resp:
            try: resp.close()
            except Exception: pass
    if not ok:
        logger.error("--- TTS FAILED: no audio produced ---")
    else:
        s = f"(TTFT: {ttft_ms:.2f} ms)" if ttft_ms else ""
        logger.info(f"--- TTS complete {s} ---")
