"""
English Voice Engine using Kokoro-82M (hexgrad/Kokoro-82M).
Produces ultra-natural, conversational 24kHz speech on standard CPU hardware.
Includes graceful fallback to facebook/mms-tts-eng if Kokoro weights are downloading.
"""

import logging
from typing import Tuple, Optional
import numpy as np

try:
    from engines.base import BaseTTSEngine
except ImportError:
    from .base import BaseTTSEngine

logger = logging.getLogger(__name__)


class KokoroTTSEngine(BaseTTSEngine):
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.pipeline = None
        self.sample_rate = 24000
        self.fallback_engine = None

    def is_loaded(self) -> bool:
        return self.pipeline is not None or (self.fallback_engine is not None and self.fallback_engine.is_loaded())

    def load(self):
        """Attempts to load Kokoro-82M pipeline; falls back to Meta MMS-TTS English if needed."""
        if self.is_loaded():
            return

        try:
            logger.info("Loading Kokoro-82M English pipeline...")
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code="a")
            self.sample_rate = 24000
            logger.info("Kokoro-82M English engine successfully loaded into RAM.")
            return
        except Exception as e:
            logger.warning(f"Kokoro library not available ({e}). Trying Meta MMS-TTS English fallback.")

        try:
            try:
                from engines.mms_engine import MMSTTSEngine
            except ImportError:
                from .mms_engine import MMSTTSEngine
            self.fallback_engine = MMSTTSEngine(language="eng")
            self.fallback_engine.model_id = "facebook/mms-tts-eng"
            self.fallback_engine.load()
            self.sample_rate = self.fallback_engine.sample_rate
            logger.info("Using Meta MMS-TTS English fallback engine.")
        except Exception as ex:
            logger.error(f"Failed to load both Kokoro and MMS-TTS English: {ex}")
            raise

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesizes English text into audio.
        Returns (audio_np, sample_rate).
        """
        if not self.is_loaded():
            self.load()

        clean_text = text.strip()
        if not clean_text:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), self.sample_rate

        if self.pipeline is not None:
            try:
                chunks = []
                generator = self.pipeline(clean_text, voice=self.voice, speed=self.speed)
                for _, _, audio in generator:
                    if hasattr(audio, "numpy"):
                        chunks.append(audio.numpy())
                    elif isinstance(audio, np.ndarray):
                        chunks.append(audio)
                if chunks:
                    return np.concatenate(chunks), self.sample_rate
            except Exception as e:
                logger.error(f"Kokoro synthesis failed: {e}")

        if self.fallback_engine is not None:
            return self.fallback_engine.synthesize(clean_text)

        return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), self.sample_rate
