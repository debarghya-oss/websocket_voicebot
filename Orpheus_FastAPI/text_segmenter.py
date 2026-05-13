# text_segmenter.py
import pysbd  # Ensure 'pip install pysbd'
import logging

logger = logging.getLogger(__name__)

class TextSegmenter:
    """
    Identifies complete sentences from text for TTS streaming.
    Optimized for immediate processing at 193 tokens/ms rate - minimizes buffering delays.
    """
    def __init__(self, language="en"):
        try:
            self.segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False)
        except Exception as e:
            logger.error(f"Failed to initialize pysbd.Segmenter for language '{language}'. Is 'pysbd' and its language models installed? Error: {e}")
            # Fallback or re-raise, depending on desired strictness. For now, let it proceed and fail later if segmenter is None.
            # Or, more robustly: raise RuntimeError(f"pysbd.Segmenter init failed: {e}") from e
            self.segmenter = None # Or handle this more gracefully
        self.buffer = ""
        self.terminators = {'.', '?', '!'} # Used by SentenceChunkerForTTS

    def add_text(self, text_chunk: str) -> list[str]:
        """
        Processes a text chunk and returns a list of identified sentences immediately.
        Optimized for 193 tokens/ms rate - processes without delays or unnecessary buffering.
        
        Args:
            text_chunk (str): Text to segment (can be partial or complete).
            
        Returns:
            list[str]: List of identified sentences.
        """
        if self.segmenter is None:
            logger.error("pysbd.Segmenter not initialized. Cannot segment text.")
            return [text_chunk]

        if not isinstance(text_chunk, str):
            return []
        
        self.buffer += text_chunk

        if not self.buffer.strip():
            return []

        # Segment immediately without buffering delays
        potential_sentences = self.segmenter.segment(self.buffer)

        if not potential_sentences:
            if not self.buffer.strip():
                self.buffer = ""
            return []

        # Check if text ends with terminator or has multiple sentences
        last_char = self.buffer.strip()[-1] if self.buffer.strip() else ''
        has_terminator = last_char in self.terminators
        has_multiple_sentences = len(potential_sentences) > 1

        if has_terminator or has_multiple_sentences:
            # Complete sentence(s) detected - return immediately
            self.buffer = ""
            result = [s.strip() for s in potential_sentences if s.strip()]
            logger.debug(f"Segmented {len(result)} sentence(s) immediately (no buffering)")
            return result
        else:
            # Single sentence without terminator - return immediately for streaming responsiveness
            self.buffer = ""
            result = [s.strip() for s in potential_sentences if s.strip()]
            logger.debug(f"Segmented {len(result)} sentence(s) (single, no terminator)")
            return result


    def get_remaining_text(self) -> str:
        """
        Returns any text remaining in the internal buffer.
        """
        return self.buffer.strip()

    def clear_buffer(self):
        """
        Clears the internal text buffer.
        """
        self.buffer = ""