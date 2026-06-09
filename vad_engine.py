"""
Voice Activity Detection Engine Module
Uses Silero VAD for real-time speech detection in streaming audio.
"""
import logging
import numpy as np
import torch
from typing import Optional, Any

from config import (
    VAD_SAMPLE_RATE, VAD_CHUNK_SAMPLES, VAD_THRESHOLD,
    VAD_BARGEIN_THRESHOLD, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS,
    VAD_MIN_SPEECH_MS
)

logger = logging.getLogger(__name__)

vad_model: Optional[Any] = None


def load_vad_model() -> Optional[Any]:
    """Load the Silero VAD model for voice activity detection."""
    global vad_model
    logger.info("=== Starting Silero VAD Model Loading ===")
    try:
        from silero_vad import load_silero_vad
        logger.info("✓ silero_vad library imported successfully.")
    except ImportError as e:
        logger.error(f"✗ silero_vad library not found: {e}. Install with: pip install silero-vad")
        return None
    try:
        vad_model_instance = load_silero_vad(onnx=False)
        if vad_model_instance is None:
            logger.error("✗ load_silero_vad() returned None")
            return None
        vad_model = vad_model_instance
        logger.info(f"✓ Silero VAD model loaded: {type(vad_model)}")
        return vad_model
    except Exception as e:
        logger.exception(f"✗ Error loading Silero VAD model: {e}")
        return None


class VADProcessor:
    """
    Per-session Voice Activity Detection processor.
    Each WebSocket session gets its own instance with isolated state.
    """

    def __init__(self, model):
        self.model = model
        self.sample_rate = VAD_SAMPLE_RATE
        self.chunk_samples = VAD_CHUNK_SAMPLES
        self.threshold = VAD_THRESHOLD
        self.min_silence_samples = int(VAD_SAMPLE_RATE * VAD_MIN_SILENCE_MS / 1000)
        self.min_speech_samples  = int(VAD_SAMPLE_RATE * VAD_MIN_SPEECH_MS / 1000)
        self.speech_pad_samples  = int(VAD_SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)

        self._is_speaking         = False
        self._silence_start: Optional[int] = None
        self._current_sample      = 0
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_sample = 0
        self._total_speech_samples = 0

        # FIX: save/restore hidden state for get_speech_prob so it doesn't
        # corrupt the VAD state machine during barge-in detection.
        # Silero stores state in model._h and model._c (LSTM hidden/cell).
        self._saved_h = None
        self._saved_c = None

        self.model.reset_states()
        logger.debug(
            f"VADProcessor initialized: threshold={self.threshold}, "
            f"min_silence={VAD_MIN_SILENCE_MS}ms, min_speech={VAD_MIN_SPEECH_MS}ms"
        )

    def _save_model_state(self):
        """Save Silero LSTM hidden state so we can restore it after a probe call."""
        try:
            self._saved_h = self.model._h.clone() if hasattr(self.model, '_h') and self.model._h is not None else None
            self._saved_c = self.model._c.clone() if hasattr(self.model, '_c') and self.model._c is not None else None
        except Exception:
            self._saved_h = None
            self._saved_c = None

    def _restore_model_state(self):
        """Restore Silero LSTM hidden state after a probe call."""
        try:
            if self._saved_h is not None and hasattr(self.model, '_h'):
                self.model._h = self._saved_h
            if self._saved_c is not None and hasattr(self.model, '_c'):
                self.model._c = self._saved_c
        except Exception:
            pass

    def _to_tensor(self, audio_chunk: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(audio_chunk).float()
        if len(t) < self.chunk_samples:
            t = torch.nn.functional.pad(t, (0, self.chunk_samples - len(t)))
        return t

    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process one audio chunk through the VAD state machine.
        Returns the complete utterance when end-of-speech is detected, else None.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        speech_prob = self.model(self._to_tensor(audio_chunk), self.sample_rate).item()
        self._current_sample += len(audio_chunk)

        neg_threshold = max(self.threshold - 0.15, 0.01)

        if speech_prob >= self.threshold:
            if not self._is_speaking:
                self._is_speaking = True
                self._silence_start = None
                self._speech_start_sample = self._current_sample
                self._speech_buffer = []
                self._total_speech_samples = 0
                logger.debug(f"VAD: Speech START (prob={speech_prob:.3f})")
            else:
                self._silence_start = None
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)
            return None

        elif speech_prob < neg_threshold and self._is_speaking:
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)

            if self._silence_start is None:
                self._silence_start = self._current_sample

            silence_duration = self._current_sample - self._silence_start

            if silence_duration >= self.min_silence_samples:
                if self._total_speech_samples >= self.min_speech_samples:
                    logger.info(
                        f"VAD: Speech END (silence={silence_duration/self.sample_rate*1000:.0f}ms, "
                        f"speech={self._total_speech_samples/self.sample_rate:.2f}s)"
                    )
                    utterance = np.concatenate(self._speech_buffer)
                    self._reset_speech_state()
                    return utterance
                else:
                    logger.debug(f"VAD: Discarding short segment ({self._total_speech_samples} samples)")
                    self._reset_speech_state()
                    return None
            return None

        elif self._is_speaking:
            # Ambiguous zone — keep accumulating
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)
            self._silence_start = None  # FIX: Reset silence timer in ambiguous zone
            return None

        return None

    @torch.no_grad()
    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability WITHOUT corrupting the VAD state machine.
        Used for barge-in detection during pipeline processing.

        FIX: saves and restores Silero's LSTM hidden state so this probe
        call doesn't affect subsequent process_chunk() calls.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0

        self._save_model_state()
        try:
            prob = self.model(self._to_tensor(audio_chunk), self.sample_rate).item()
        except Exception:
            prob = 0.0
        finally:
            self._restore_model_state()

        return prob

    def _reset_speech_state(self):
        self._is_speaking = False
        self._silence_start = None
        self._speech_buffer = []
        self._total_speech_samples = 0
        self.model.reset_states()

    def reset(self):
        """Full reset for next conversation turn."""
        self._reset_speech_state()
        self._current_sample = 0
        self._speech_start_sample = 0
        logger.debug("VADProcessor: Full reset")

    @property
    def is_speech_active(self) -> bool:
        return self._is_speaking