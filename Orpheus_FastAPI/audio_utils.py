import re
import numpy as np
import torch
from torch import nn
import logging
from typing import List, Optional
import base64
from io import BytesIO

from config import (
    ORPHEUS_MIN_ID, ORPHEUS_MAX_ID, ORPHEUS_TOKENS_PER_LAYER, 
    ORPHEUS_N_LAYERS, TARGET_SAMPLE_RATE
)

logger = logging.getLogger(__name__)


def parse_gguf_codes(response_text: str) -> List[int]:
    """Parses GGUF-style custom audio tokens with strict validation."""
    try:
        raw_codes = [int(m) for m in re.findall(r"<custom_token_(\d+)>", response_text)]
        
        # Strict validation: ensure all codes are within valid range
        valid_codes = []
        for code in raw_codes:
            if ORPHEUS_MIN_ID <= code < ORPHEUS_MAX_ID:
                # Additional check: verify code_idx is valid
                code_idx = (code - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
                layer_idx = (code - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
                if 0 <= code_idx < ORPHEUS_TOKENS_PER_LAYER and 0 <= layer_idx < ORPHEUS_N_LAYERS:
                    valid_codes.append(code)
                else:
                    logger.warning(f"Parse: Token {code} has invalid code_idx={code_idx} or layer_idx={layer_idx}. Rejecting.")
            else:
                logger.warning(f"Parse: Token {code} outside valid range [{ORPHEUS_MIN_ID}, {ORPHEUS_MAX_ID}). Rejecting.")
        
        if len(valid_codes) < len(raw_codes):
            logger.warning(f"Parse: Filtered {len(raw_codes)} raw codes down to {len(valid_codes)} valid codes.")
        
        return valid_codes
    except Exception as e:
        logger.error(f"GGUF parse error: {e} on text: '{response_text[:200]}...'", exc_info=True)
        return []


def redistribute_codes(codes: List[int], model: nn.Module) -> Optional[np.ndarray]:
    """Redistributes parsed codes into SNAC layers and decodes using the SNAC model."""
    if not codes or model is None:
        logger.debug("Redistribute codes called with no codes or no model.")
        return None

    if len(codes) % ORPHEUS_N_LAYERS != 0:
        logger.warning(f"Redistribute codes: Received {len(codes)} codes which is not a multiple of ORPHEUS_N_LAYERS ({ORPHEUS_N_LAYERS}). Processing full groups only.")

    num_groups_in_input = len(codes) // ORPHEUS_N_LAYERS
    if num_groups_in_input == 0:
        logger.debug("Redistribute codes: Not enough codes for a complete group.")
        return None

    try:
        dev = next(model.parameters()).device
        snac_layers_codes: List[List[int]] = [[] for _ in range(3)]

        num_groups_to_process = num_groups_in_input
        codes_to_process = codes[:num_groups_to_process * ORPHEUS_N_LAYERS]
        logger.debug(f"Redistribute codes: Processing {len(codes_to_process)} codes as {num_groups_to_process} full groups.")

        valid_groups_count = 0
        for i in range(num_groups_to_process):
            idx = i * ORPHEUS_N_LAYERS
            group = codes_to_process[idx : idx + ORPHEUS_N_LAYERS]

            processed: List[Optional[int]] = [None] * ORPHEUS_N_LAYERS
            group_is_valid = True
            for j, t_id in enumerate(group):
                layer_idx_from_id = (t_id - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
                code_idx = (t_id - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
                expected_layer_for_pos = j

                if not (ORPHEUS_MIN_ID <= t_id < ORPHEUS_MAX_ID):
                    logger.warning(f"Redistribute codes: Invalid token ID {t_id} in group {i} at position {j}. Skipping group.")
                    group_is_valid = False; break
                if layer_idx_from_id != expected_layer_for_pos:
                    logger.warning(f"Redistribute codes: Code {t_id} (layer {layer_idx_from_id}) found at unexpected position {j} (expected layer {expected_layer_for_pos}) in group {i}. Skipping group.")
                    group_is_valid = False; break
                if not (0 <= code_idx < ORPHEUS_TOKENS_PER_LAYER):
                    logger.warning(f"Redistribute codes: Code index {code_idx} out of range [0, {ORPHEUS_TOKENS_PER_LAYER}) for token {t_id} in group {i}. Skipping group.")
                    group_is_valid = False; break
                processed[j] = code_idx
            
            if group_is_valid:
                try:
                    if any(p is None for p in processed):
                        logger.error(f"Redistribute codes: 'None' found in processed list for a valid group {i}. Skipping.")
                        continue
                    
                    pg_int: List[int] = [p for p in processed if p is not None] 
                    if len(pg_int) != ORPHEUS_N_LAYERS: 
                         logger.error(f"Redistribute codes: Group {i} valid, but processed list length mismatch. Skipping")
                         continue

                    snac_layers_codes[0].append(pg_int[0]) 
                    snac_layers_codes[1].append(pg_int[1]) 
                    snac_layers_codes[2].append(pg_int[2]) 
                    snac_layers_codes[2].append(pg_int[3]) 
                    snac_layers_codes[1].append(pg_int[4]) 
                    snac_layers_codes[2].append(pg_int[5]) 
                    snac_layers_codes[2].append(pg_int[6]) 
                    valid_groups_count += 1
                except IndexError as map_e:
                    logger.error(f"Redistribute codes: Code mapping error in group {i}, processed={processed}: {map_e}. Skipping group.", exc_info=True)
                    continue
                except TypeError as type_e: 
                    logger.error(f"Redistribute codes: Type error during mapping (likely None in processed) in group {i}, processed={processed}: {type_e}. Skipping group.", exc_info=True)
                    continue

        if valid_groups_count == 0:
            logger.warning("Redistribute codes: No valid Orpheus code groups could be mapped to SNAC layers.")
            return None

        expected_l0_count = valid_groups_count
        expected_l1_count = valid_groups_count * 2
        expected_l2_count = valid_groups_count * 4

        if len(snac_layers_codes[0]) != expected_l0_count or \
           len(snac_layers_codes[1]) != expected_l1_count or \
           len(snac_layers_codes[2]) != expected_l2_count:
            logger.error(f"Redistribute codes: Mismatched SNAC input tensor lengths after mapping {valid_groups_count} valid groups: L0={len(snac_layers_codes[0])}, L1={len(snac_layers_codes[1])}, L2={len(snac_layers_codes[2])}. Expected: {expected_l0_count}, {expected_l1_count}, {expected_l2_count}. Aborting decode.")
            return None
        
        tensors = [ torch.tensor(lc, device=dev, dtype=torch.long).unsqueeze(0) for lc in snac_layers_codes ]
        logger.debug(f"Redistribute codes: Decodable group count: {valid_groups_count}. Decoding...")
        with torch.no_grad():
            audio = model.decode(tensors)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        audio_np = audio.detach().squeeze().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(audio_np)):
            logger.error("!!! SNAC produced non-finite values (NaN/inf). Replacing with silence. !!!")
            return np.zeros_like(audio_np)
        logger.debug(f"Redistribute codes: Decode successful. Generated {audio_np.size} samples.")
        return audio_np
    except Exception as e:
        logger.exception(f"SNAC decode error: {e}")
        return None


def apply_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: int = 5) -> np.ndarray:
    """
    Applies a fade in/out to an audio chunk for smooth transitions.
    
    Args:
        audio_chunk: Float32 numpy array
        sample_rate: Sample rate of the audio (used to calculate fade duration)
        fade_ms: Fade duration in milliseconds (default 5ms for smooth transitions)
    
    Returns:
        Audio chunk with fade applied
    """
    if audio_chunk is None or audio_chunk.size == 0:
        return audio_chunk
    num_fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if num_fade_samples <= 0 or audio_chunk.size < 2 * num_fade_samples:
        return audio_chunk 
    fade_in = np.linspace(0., 1., num_fade_samples, dtype=audio_chunk.dtype)
    fade_out = np.linspace(1., 0., num_fade_samples, dtype=audio_chunk.dtype)
    audio_chunk[:num_fade_samples] *= fade_in
    audio_chunk[-num_fade_samples:] *= fade_out
    logger.debug(f"Applied {fade_ms}ms fade to audio chunk at {sample_rate}Hz.")
    return audio_chunk


# ================================================================
# BASE64 AUDIO ENCODING/DECODING
# ================================================================
def encode_audio_to_base64(audio_bytes: bytes) -> str:
    """Encodes raw audio bytes to base64 string."""
    if not audio_bytes:
        return ""
    try:
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode audio to base64: {e}")
        return ""


def decode_audio_from_base64(audio_base64: str) -> bytes:
    """Decodes base64 string to raw audio bytes."""
    if not audio_base64:
        return b''
    try:
        return base64.b64decode(audio_base64)
    except Exception as e:
        logger.error(f"Failed to decode audio from base64: {e}")
        return b''


def encode_float32_chunk_to_base64(float32_array: np.ndarray) -> str:
    """Converts Float32 numpy array to bytes and encodes as base64."""
    if float32_array is None or float32_array.size == 0:
        return ""
    try:
        # Ensure it's float32
        if float32_array.dtype != np.float32:
            float32_array = float32_array.astype(np.float32)
        audio_bytes = float32_array.tobytes()
        return encode_audio_to_base64(audio_bytes)
    except Exception as e:
        logger.error(f"Failed to encode float32 chunk to base64: {e}")
        return ""


def decode_base64_to_float32(audio_base64: str) -> Optional[np.ndarray]:
    """Decodes base64 string to Float32 numpy array."""
    if not audio_base64:
        return None
    try:
        audio_bytes = decode_audio_from_base64(audio_base64)
        if not audio_bytes:
            return None
        # Convert bytes back to float32 array
        float32_array = np.frombuffer(audio_bytes, dtype=np.float32)
        return float32_array
    except Exception as e:
        logger.error(f"Failed to decode base64 to float32: {e}")
        return None


# ================================================================
# AUDIO RESAMPLING
# ================================================================
def resample_audio(audio_chunk: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """
    Resample audio from original sample rate to target sample rate.
    Uses linear interpolation for fast, efficient resampling.
    
    Args:
        audio_chunk: Float32 numpy array of audio samples
        orig_sample_rate: Original sample rate (e.g., 24000)
        target_sample_rate: Target sample rate (e.g., 8000)
    
    Returns:
        Resampled audio as float32 numpy array
    """
    if audio_chunk is None or audio_chunk.size == 0:
        return audio_chunk
    
    if orig_sample_rate == target_sample_rate:
        return audio_chunk
    
    try:
        # Calculate resampling ratio
        ratio = target_sample_rate / orig_sample_rate
        orig_length = len(audio_chunk)
        target_length = int(np.round(orig_length * ratio))
        
        if target_length <= 0:
            logger.warning(f"Resample: Target length {target_length} is invalid. Returning original.")
            return audio_chunk
        
        # Use linear interpolation for smooth resampling
        x_orig = np.arange(orig_length)
        x_target = np.arange(target_length) / ratio
        
        # Linear interpolation
        resampled = np.interp(x_target, x_orig, audio_chunk)
        
        logger.debug(
            f"Resampled audio from {orig_sample_rate}Hz ({orig_length} samples) "
            f"to {target_sample_rate}Hz ({target_length} samples)"
        )
        
        return resampled.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Resampling failed: {e}. Returning original audio.")
        return audio_chunk
