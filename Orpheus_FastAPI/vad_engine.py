"""
Voice Activity Detection Engine Module
Uses Silero VAD for real-time speech detection in streaming audio.

Each WebSocket session gets its own VADProcessor instance that:
- Accepts 512-sample (32ms) chunks of 16kHz audio
- Tracks speech/silence state transitions
- Accumulates speech audio and returns complete utterances
  when silence exceeds the configured threshold
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

# Global VAD model instance (loaded once at startup, shared across sessions)
vad_model: Optional[Any] = None


def load_vad_model() -> Optional[Any]:
    """Load the Silero VAD model for voice activity detection."""
    global vad_model

    logger.info("=== Starting Silero VAD Model Loading ===")

    try:
        from silero_vad import load_silero_vad
        logger.info("✓ silero_vad library imported successfully.")
    except ImportError as e:
        logger.error(f"✗ silero_vad library not found: {e}")
        logger.error("  Install with: pip install silero-vad")
        return None

    try:
        # Load the JIT model (not ONNX) for PyTorch native inference
        vad_model_instance = load_silero_vad(onnx=False)

        if vad_model_instance is None:
            logger.error("✗ load_silero_vad() returned None")
            return None

        vad_model = vad_model_instance
        logger.info(f"✓ Silero VAD model loaded successfully: {type(vad_model)}")
        logger.info("=== Silero VAD Model Loading Complete ===")
        return vad_model

    except Exception as e:
        logger.exception(f"✗ Error loading Silero VAD model: {e}")
        return None


class VADProcessor:
    """
    Per-session Voice Activity Detection processor.

    Manages the state machine for a single WebSocket conversation session:
        IDLE → SPEAKING → SILENCE_DETECTED → UTTERANCE_COMPLETE → IDLE

    Usage:
        processor = VADProcessor(model)
        for chunk in audio_stream:
            result = processor.process_chunk(chunk)
            if result is not None:
                # result is the complete speech segment as float32 numpy array
                send_to_stt(result)
    """

    def __init__(self, model):
        """
        Initialize a new VAD processor for a session.

        Args:
            model: The loaded Silero VAD model (shared across sessions).
        """
        self.model = model
        self.sample_rate = VAD_SAMPLE_RATE
        self.chunk_samples = VAD_CHUNK_SAMPLES
        self.threshold = VAD_THRESHOLD
        self.min_silence_samples = int(VAD_SAMPLE_RATE * VAD_MIN_SILENCE_MS / 1000)
        self.min_speech_samples = int(VAD_SAMPLE_RATE * VAD_MIN_SPEECH_MS / 1000)
        self.speech_pad_samples = int(VAD_SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)

        # State
        self._is_speaking = False
        self._silence_start: Optional[int] = None  # sample index where silence began
        self._current_sample = 0
        self._speech_buffer: list[np.ndarray] = []  # accumulated speech chunks
        self._speech_start_sample = 0
        self._total_speech_samples = 0

        # Each session needs its own model state — reset on init
        self.model.reset_states()

        logger.debug(
            f"VADProcessor initialized: threshold={self.threshold}, "
            f"min_silence={VAD_MIN_SILENCE_MS}ms ({self.min_silence_samples} samples), "
            f"min_speech={VAD_MIN_SPEECH_MS}ms ({self.min_speech_samples} samples)"
        )

    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single audio chunk through the VAD.

        Args:
            audio_chunk: Float32 numpy array of exactly VAD_CHUNK_SAMPLES samples
                         at VAD_SAMPLE_RATE Hz.

        Returns:
            None if still listening/accumulating, or the complete speech segment
            as a float32 numpy array when end-of-speech is detected.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        # Convert to torch tensor for Silero
        chunk_tensor = torch.from_numpy(audio_chunk).float()

        # Pad if chunk is shorter than expected (e.g., final chunk)
        if len(chunk_tensor) < self.chunk_samples:
            chunk_tensor = torch.nn.functional.pad(
                chunk_tensor, (0, self.chunk_samples - len(chunk_tensor))
            )

        # Get speech probability from Silero VAD
        speech_prob = self.model(chunk_tensor, self.sample_rate).item()
        self._current_sample += len(audio_chunk)

        # Negative threshold (hysteresis to avoid flickering)
        neg_threshold = max(self.threshold - 0.15, 0.01)

        # --- State machine ---

        if speech_prob >= self.threshold:
            # Speech detected
            if not self._is_speaking:
                # Transition: IDLE → SPEAKING
                self._is_speaking = True
                self._silence_start = None
                self._speech_start_sample = self._current_sample
                self._speech_buffer = []
                self._total_speech_samples = 0
                logger.debug(
                    f"VAD: Speech START at sample {self._current_sample} "
                    f"(prob={speech_prob:.3f})"
                )
            else:
                # Already speaking — reset any silence counter
                self._silence_start = None

            # Accumulate audio
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)
            return None

        elif speech_prob < neg_threshold and self._is_speaking:
            # Below negative threshold while in speech → potential end of speech
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)

            if self._silence_start is None:
                # Start silence timer
                self._silence_start = self._current_sample

            silence_duration = self._current_sample - self._silence_start

            if silence_duration >= self.min_silence_samples:
                # Silence long enough → end of utterance
                if self._total_speech_samples >= self.min_speech_samples:
                    logger.info(
                        f"VAD: Speech END at sample {self._current_sample} "
                        f"(silence={silence_duration} samples / "
                        f"{silence_duration / self.sample_rate * 1000:.0f}ms, "
                        f"speech_duration="
                        f"{self._total_speech_samples / self.sample_rate:.2f}s)"
                    )
                    # Concatenate all accumulated chunks
                    complete_utterance = np.concatenate(self._speech_buffer)
                    self._reset_speech_state()
                    return complete_utterance
                else:
                    # Too short — discard as noise
                    logger.debug(
                        f"VAD: Discarding short speech segment "
                        f"({self._total_speech_samples} samples < "
                        f"{self.min_speech_samples} min)"
                    )
                    self._reset_speech_state()
                    return None

            # Still within silence grace period — keep waiting
            return None

        elif self._is_speaking:
            # Between threshold and neg_threshold while speaking — ambiguous zone
            # Keep accumulating but don't reset silence timer
            self._speech_buffer.append(audio_chunk.copy())
            self._total_speech_samples += len(audio_chunk)
            return None

        # Not speaking and below threshold — idle
        return None

    def _reset_speech_state(self):
        """Reset speech tracking state after utterance is complete or discarded."""
        self._is_speaking = False
        self._silence_start = None
        self._speech_buffer = []
        self._total_speech_samples = 0
        self.model.reset_states()

    @property
    def is_speech_active(self) -> bool:
        """Whether speech is currently being detected."""
        return self._is_speaking

    @torch.no_grad()
    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Get the raw speech probability for a chunk WITHOUT modifying state.

        Used for barge-in detection during pipeline processing — we need
        to know if the user is speaking without interfering with the
        normal VAD state machine.

        Args:
            audio_chunk: Float32 numpy array of VAD_CHUNK_SAMPLES samples.

        Returns:
            Speech probability float between 0 and 1.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0

        chunk_tensor = torch.from_numpy(audio_chunk).float()

        if len(chunk_tensor) < self.chunk_samples:
            chunk_tensor = torch.nn.functional.pad(
                chunk_tensor, (0, self.chunk_samples - len(chunk_tensor))
            )

        return self.model(chunk_tensor, self.sample_rate).item()

    def reset(self):
        """Full reset of all state for next conversation turn."""
        self._reset_speech_state()
        self._current_sample = 0
        self._speech_start_sample = 0
        logger.debug("VADProcessor: Full reset")
