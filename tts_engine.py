import logging
import warnings
import time
import json
import requests
from typing import Generator, Optional, Any
import numpy as np

from config import (
    TTS_API_ENDPOINT, TTS_MODEL, TTS_PROMPT_FORMAT, TTS_PROMPT_STOP_TOKENS,
    ORPHEUS_N_LAYERS, SNAC_SAMPLE_RATE, TARGET_SAMPLE_RATE, STREAM_TIMEOUT_SECONDS, 
    STREAM_HEADERS, SSE_DATA_PREFIX, SSE_DONE_MARKER,
    TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS, DEVICE, ENABLE_RESAMPLING, TTS_AUDIO_FADE_MS,
    TTS_FREQUENCY_PENALTY, TTS_PRESENCE_PENALTY, TTS_REPEAT_LAST_N,
    TTS_MAX_DURATION_MULTIPLIER, TTS_ESTIMATED_CHARS_PER_SEC,
    TTS_REPEAT_PATTERN_MIN_LEN, TTS_REPEAT_PATTERN_THRESHOLD, TTS_REPEAT_CHECK_INTERVAL
)
from audio_utils import parse_gguf_codes, redistribute_codes, apply_fade, resample_audio

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
        logger.error("SNAC not found. Please install it: pip install git+https://github.com/hubertsiuzdak/snac.git")
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
            logger.error("SNAC.from_pretrained returned None or failed. SNAC model not loaded.")
            return None
    except Exception as e:
        logger.exception("Fatal error during SNAC model loading process.")
        return None


def _detect_code_repetition(codes: list, min_pattern_len: int, threshold: int) -> bool:
    """Detect if tail of code sequence contains a repeating pattern.

    Scans backwards from the end of *codes* looking for a contiguous pattern
    of length *min_pattern_len* that repeats at least *threshold* times.
    Designed to be cheap: single linear scan, no allocations.
    """
    n = len(codes)
    needed = min_pattern_len * threshold
    if n < needed:
        return False

    # Only check pattern_len == min_pattern_len for speed (covers 3-group loops)
    pat_start = n - min_pattern_len
    for rep in range(1, threshold):
        cmp_start = pat_start - min_pattern_len * rep
        if cmp_start < 0:
            return False
        # Compare slice by slice
        match = True
        for j in range(min_pattern_len):
            if codes[pat_start + j] != codes[cmp_start + j]:
                match = False
                break
        if not match:
            return False
    return True


def generate_speech_stream_bytes(
    text: str, voice: str, tts_temperature: float, tts_top_p: float,
    tts_repetition_penalty: float, buffer_groups_param: int, padding_ms_param: int,
    min_decode_batch_groups_param: int
) -> Generator[bytes, None, None]:
    """Generate speech audio stream bytes from text using TTS."""
    global snac_model
    
    if not text.strip():
        logger.warning("generate_speech_stream_bytes called with empty text.")
        return
    
    if snac_model is None:
        logger.error("SNAC model not loaded. Cannot generate audio bytes for TTS. Check server logs.")
        yield b''
        return

    codes_per_group = ORPHEUS_N_LAYERS
    buffer_groups_effective = max(1, buffer_groups_param)
    min_codes_required_initial = buffer_groups_effective * codes_per_group
    min_decode_batch_groups = max(1, min_decode_batch_groups_param)
    min_codes_required_per_batch = min_decode_batch_groups * codes_per_group
    min_codes_required_per_batch = max(min_codes_required_per_batch, codes_per_group) 

    logger.debug(f"TTS Stream processing: initial buffer target={buffer_groups_effective} groups ({min_codes_required_initial} codes), padding={padding_ms_param} ms. Per-batch decode unit: {min_codes_required_per_batch} codes ({min_decode_batch_groups} groups).")

    silence_bytes = b''
    if padding_ms_param > 0:
        silence_samples = int(TARGET_SAMPLE_RATE * (padding_ms_param / 1000.0))
        if silence_samples > 0:
            logger.debug(f"Calculated silence samples per side: {silence_samples}")
            silence_bytes = np.zeros(silence_samples, dtype=np.float32).tobytes()

    payload = {
        "model": TTS_MODEL, "prompt": TTS_PROMPT_FORMAT.format(voice=voice, text=text),
        "temperature": tts_temperature, "top_p": tts_top_p, "repeat_penalty": tts_repetition_penalty,
        "frequency_penalty": TTS_FREQUENCY_PENALTY,
        "presence_penalty": TTS_PRESENCE_PENALTY,
        "repeat_last_n": TTS_REPEAT_LAST_N,
        "n_predict": -1, "stop": TTS_PROMPT_STOP_TOKENS, "stream": True
    }
    
    accumulated_codes = []
    all_codes_history: list = []  # full history for repetition detection (ints only, cheap)
    total_codes_received = 0

    # Pre-compute max codes ceiling from text length (simple int math, zero overhead)
    _est_dur = max(len(text) / TTS_ESTIMATED_CHARS_PER_SEC, 2.0)
    # ~11.7 code-groups/sec at 24kHz SNAC (24000/2048), times 7 codes per group
    _codes_per_sec = (SNAC_SAMPLE_RATE / 2048) * ORPHEUS_N_LAYERS
    max_total_codes = int(_est_dur * TTS_MAX_DURATION_MULTIPLIER * _codes_per_sec)

    request_initiation_time = time.time()
    response_obj = None
    stream_successful = False
    initial_buffer_processed = False
    last_code_received_time: Optional[float] = None

    time_stream_connected_for_ttft: Optional[float] = None
    first_token_parsed_time_for_ttft: Optional[float] = None
    calculated_ttft_ms: Optional[float] = None

    try:
        logger.info(">>> TTS API: Initiating stream request to external TTS server...")
        logger.debug(f"Sending TTS Payload: {json.dumps(payload)}")
        response_obj = requests.post(
            TTS_API_ENDPOINT, json=payload, headers=STREAM_HEADERS, stream=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        response_obj.raise_for_status()
        
        time_stream_connected_for_ttft = time.time()
        logger.info(f"--- TTS API: Stream connected successfully after {time_stream_connected_for_ttft - request_initiation_time:.3f}s. Receiving codes...")

        for line in response_obj.iter_lines():
            timeout_baseline = time_stream_connected_for_ttft if time_stream_connected_for_ttft is not None else request_initiation_time
            if time.time() - timeout_baseline > STREAM_TIMEOUT_SECONDS:
                logger.error(f" TTS API stream processing timed out after {STREAM_TIMEOUT_SECONDS} seconds while waiting for data.")
                break
            if not line: 
                continue

            try:
                decoded_line = line.decode(response_obj.encoding or 'utf-8', errors='ignore')
            except UnicodeDecodeError:
                logger.warning(f"Skipping undecodable line in TTS stream: {line[:50]}...")
                continue
            
            if decoded_line.startswith(SSE_DATA_PREFIX):
                json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                if json_str == SSE_DONE_MARKER:
                    logger.debug("Received TTS SSE_DONE_MARKER. Ending stream processing.")
                    break
                if not json_str:
                    logger.debug("Received empty data line, skipping.")
                    continue

                try:
                    data = json.loads(json_str)
                    chunk_text = ""
                    if "content" in data:
                        chunk_text = data.get("content", "")
                    elif "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        chunk_text = delta.get("content", "") or choice.get("text", "")

                    if chunk_text:
                        new_codes = parse_gguf_codes(chunk_text)
                        if new_codes:
                            if first_token_parsed_time_for_ttft is None and time_stream_connected_for_ttft is not None:
                                first_token_parsed_time_for_ttft = time.time()
                                calculated_ttft_ms = (first_token_parsed_time_for_ttft - time_stream_connected_for_ttft) * 1000
                                logger.info(f"--- TTS METRIC: Time to first audio token chunk parsed: {calculated_ttft_ms:.2f} ms (from stream connected) ---")
                            
                            accumulated_codes.extend(new_codes)
                            all_codes_history.extend(new_codes)
                            total_codes_received += len(new_codes)
                            last_code_received_time = time.time()

                            # --- Lightweight repetition guard (cheap int checks) ---
                            if total_codes_received > max_total_codes:
                                logger.warning(
                                    f"TTS: Runaway generation — {total_codes_received} codes "
                                    f"exceeds max {max_total_codes} for {len(text)}-char text. "
                                    f"Stopping early."
                                )
                                break

                            # Check for repetition every check interval AND always check after substantial codes
                            should_check_repetition = (
                                total_codes_received % TTS_REPEAT_CHECK_INTERVAL == 0
                                or total_codes_received > 500  # Always check if we have a lot of codes
                            )
                            if should_check_repetition:
                                is_repeating = _detect_code_repetition(
                                    all_codes_history,
                                    TTS_REPEAT_PATTERN_MIN_LEN,
                                    TTS_REPEAT_PATTERN_THRESHOLD
                                )
                                if is_repeating:
                                    logger.warning(
                                        f"TTS: Repeating code pattern detected after {total_codes_received} codes. "
                                        f"History length: {len(all_codes_history)}. Stopping early to prevent audio repetition."
                                    )
                                    break
                                elif total_codes_received > 500:
                                    logger.debug(f"TTS: Repetition check passed at {total_codes_received} codes (no repeating pattern detected).")

                            if not initial_buffer_processed and len(accumulated_codes) >= codes_per_group:
                                initial_buffer_processed = True
                                logger.debug(f"First codes received ({len(accumulated_codes)} codes). Entering streaming mode.")

                            if initial_buffer_processed and accumulated_codes and last_code_received_time is not None:
                                time_since_last_codes = (time.time() - last_code_received_time) * 1000
                                if time_since_last_codes > TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS or len(accumulated_codes) >= min_codes_required_per_batch:
                                    logger.debug(f"Flushing codes: {len(accumulated_codes)} accumulated (timeout: {time_since_last_codes:.1f}ms > {TTS_STREAM_PARTIAL_BATCH_TIMEOUT_MS}ms or >= batch threshold)")
                                    try:
                                        num_groups_available = len(accumulated_codes) // codes_per_group
                                        if num_groups_available > 0:
                                            codes_to_decode_count_batch = num_groups_available * codes_per_group
                                        else:
                                            codes_to_decode_count_batch = 0

                                        if codes_to_decode_count_batch > 0:
                                            codes_to_decode_batch = accumulated_codes[:codes_to_decode_count_batch]
                                            accumulated_codes = accumulated_codes[codes_to_decode_count_batch:]
                                            logger.debug(f"Flushing {len(codes_to_decode_batch)} codes ({num_groups_available} groups). Remaining: {len(accumulated_codes)}.")
                                            snac_decode_start_time = time.time()
                                            audio_chunk = redistribute_codes(codes_to_decode_batch, snac_model)
                                            snac_decode_end_time = time.time()

                                            if audio_chunk is not None and audio_chunk.size > 0:
                                                logger.debug(f"--- SNAC: Decoded flush chunk ({len(codes_to_decode_batch)} codes -> {audio_chunk.size} samples) in {snac_decode_end_time - snac_decode_start_time:.3f}s.")
                                                
                                                # Apply fade for smooth transitions
                                                faded_chunk = apply_fade(audio_chunk, SNAC_SAMPLE_RATE, fade_ms=TTS_AUDIO_FADE_MS)
                                                
                                                # Resample from SNAC rate (24kHz) to target rate if needed
                                                if ENABLE_RESAMPLING:
                                                    resampled_chunk = resample_audio(faded_chunk, SNAC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                                                    logger.debug(f"Resampled from {SNAC_SAMPLE_RATE}Hz to {TARGET_SAMPLE_RATE}Hz")
                                                else:
                                                    resampled_chunk = faded_chunk
                                                
                                                audio_bytes_to_yield = resampled_chunk.astype(np.float32).tobytes()
                                                yield audio_bytes_to_yield
                                                stream_successful = True
                                                last_code_received_time = time.time()
                                            else:
                                                logger.warning(f"--- SNAC: Failed to decode flush chunk ({len(codes_to_decode_batch)} codes) in {snac_decode_end_time - snac_decode_start_time:.3f}s, or produced no audio.")
                                        else:
                                            logger.debug("No complete groups available for flushing yet.")
                                    except Exception as decode_e:
                                        logger.exception(f"Error during flush decoding/yielding: {decode_e}.")
                    
                    stop_reason = None
                    if "choices" in data and data["choices"] and data["choices"][0].get("finish_reason"):
                        stop_reason = data["choices"][0].get("finish_reason")
                        logger.debug(f"TTS Stream stop condition met from API response: reason='{stop_reason}'")
                        break 
                    if data.get("stop") is True or data.get("stopped_eos") is True:
                        logger.debug(f"TTS Stream stop condition met from API data flags: stop={data.get('stop')}, stopped_eos={data.get('stopped_eos')}")
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in TTS stream line: '{json_str[:100]}...'", exc_info=False)
                    continue
                except Exception as e_proc:
                    logger.exception(f"Error processing TTS stream chunk data: '{json_str[:100]}...'")
                    break 

        logger.debug(f"TTS Stream ended or timed out. Processing final accumulated codes ({len(accumulated_codes)}).")
        if len(accumulated_codes) >= codes_per_group: 
            num_final_groups = len(accumulated_codes) // codes_per_group
            codes_to_decode_final = accumulated_codes[:num_final_groups * codes_per_group]
            logger.debug(f"Decoding final {len(codes_to_decode_final)} codes in {num_final_groups} groups.")
            snac_decode_start_time = time.time()
            audio_chunk = redistribute_codes(codes_to_decode_final, snac_model)
            snac_decode_end_time = time.time()
            if audio_chunk is not None and audio_chunk.size > 0:
                logger.debug(f"--- SNAC: Decoded final chunk ({len(codes_to_decode_final)} codes -> {audio_chunk.size} samples) in {snac_decode_end_time - snac_decode_start_time:.3f}s.")
                
                # Apply fade for smooth transitions
                faded_chunk = apply_fade(audio_chunk, SNAC_SAMPLE_RATE, fade_ms=TTS_AUDIO_FADE_MS)
                
                # Resample from SNAC rate (24kHz) to target rate if needed
                if ENABLE_RESAMPLING:
                    resampled_chunk = resample_audio(faded_chunk, SNAC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                    logger.debug(f"Resampled final from {SNAC_SAMPLE_RATE}Hz to {TARGET_SAMPLE_RATE}Hz")
                else:
                    resampled_chunk = faded_chunk
                
                audio_bytes_to_yield = resampled_chunk.astype(np.float32).tobytes()
                yield audio_bytes_to_yield
                stream_successful = True
            else:
                logger.warning(f"--- SNAC: Failed to decode final chunk ({len(codes_to_decode_final)} codes) in {snac_decode_end_time - snac_decode_start_time:.3f}s, or produced no audio.")
        elif accumulated_codes:
            logger.debug(f"Discarding final {len(accumulated_codes)} codes (less than {codes_per_group}) after stream end.")

    except requests.exceptions.Timeout:
        logger.error(f" TTS API stream request timed out after {STREAM_TIMEOUT_SECONDS} seconds.", exc_info=True)
    except requests.exceptions.RequestException as req_e:
        logger.exception(f" TTS API stream request failed: {req_e}")
    except Exception as e:
        logger.exception(f" TTS API stream processing loop error: {e}")
    finally:
        if response_obj:
            logger.debug("Closing TTS API response connection.")
            try:
                response_obj.close()
            except Exception as close_e:
                logger.warning(f"Error closing requests response: {close_e}")

    if not stream_successful:
        logger.error("--- TTS Stream Generation Finished - FAILED TO PRODUCE ANY AUDIO ---")
    else:
        ttft_summary_msg = f"(TTFT: {calculated_ttft_ms:.2f} ms)" if calculated_ttft_ms is not None else "(TTFT not recorded, e.g. no codes received)"
        logger.info(f"--- First Chunk TTS Stream Generation Finished Successfully {ttft_summary_msg} ---")
