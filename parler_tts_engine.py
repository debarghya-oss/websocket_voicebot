import logging
import torch
from typing import Generator, Optional, Any
import soundfile as sf
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)

# Global Parler TTS model instances
parler_tts_model: Optional[Any] = None
parler_tts_tokenizer: Optional[Any] = None
parler_tts_description_tokenizer: Optional[Any] = None


def load_parler_tts_model():
    """Load Parler TTS model and tokenizers."""
    global parler_tts_model, parler_tts_tokenizer, parler_tts_description_tokenizer
    
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Parler TTS model on {device}...")
        
        parler_tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(device).eval()
        
        parler_tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        parler_tts_description_tokenizer = AutoTokenizer.from_pretrained(
            parler_tts_model.config.text_encoder._name_or_path
        )
        
        logger.info("✓ Parler TTS model loaded successfully.")
        return parler_tts_model
        
    except ImportError as e:
        logger.error(f"Parler TTS import failed: {e}. Install with: pip install parler-tts transformers")
        return None
    except Exception as e:
        logger.exception(f"Failed to load Parler TTS model: {e}")
        return None


def generate_speech_parler_tts(
    text: str,
    voice_description: str = "A male speaker delivers clear speech with moderate speed.",
    sampling_rate: int = 24000
) -> Optional[bytes]:
    """Generate speech using Parler TTS.
    
    Args:
        text: Text to convert to speech
        voice_description: Description of voice characteristics
        sampling_rate: Output sampling rate (default 24000)
    
    Returns:
        WAV audio bytes or None on error
    """
    global parler_tts_model, parler_tts_tokenizer, parler_tts_description_tokenizer
    
    if not parler_tts_model:
        logger.warning("Parler TTS model not loaded.")
        return None
    
    if not text.strip():
        logger.warning("Empty text provided to Parler TTS.")
        return None
    
    try:
        device = next(parler_tts_model.parameters()).device
        
        # Tokenize description and prompt
        description_inputs = parler_tts_description_tokenizer(
            voice_description, return_tensors="pt"
        ).to(device)
        prompt_inputs = parler_tts_tokenizer(text, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            generation = parler_tts_model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )
        
        # Convert to numpy and prepare WAV
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Write to BytesIO
        wav_buffer = BytesIO()
        sf.write(wav_buffer, audio_arr, parler_tts_model.config.sampling_rate, format='WAV')
        wav_buffer.seek(0)
        
        logger.info(f"✓ Parler TTS generated audio ({len(audio_arr)} samples).")
        return wav_buffer.getvalue()
        
    except Exception as e:
        logger.exception(f"Parler TTS generation failed: {e}")
        return None


def generate_speech_stream_parler_tts(
    text: str,
    voice_description: str = "A male speaker delivers clear speech with moderate speed.",
) -> Generator[bytes, None, None]:
    """Stream speech generation using Parler TTS.
    
    Args:
        text: Text to convert to speech
        voice_description: Description of voice characteristics
    
    Yields:
        Audio chunks as bytes
    """
    audio_bytes = generate_speech_parler_tts(text, voice_description)
    
    if audio_bytes:
        # Stream in 4KB chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]
    else:
        logger.warning("Parler TTS generation produced no audio.")
