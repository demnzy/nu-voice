"""
Meta MMS-TTS Engine for Hausa and French.
Uses facebook/mms-tts-hau (~140MB) and facebook/mms-tts-fra (~140MB).
VITS architecture optimized for standard CPU hardware with PyTorch.
"""

import logging
from typing import Tuple, Optional
import numpy as np

try:
    from engines.base import BaseTTSEngine
except ImportError:
    from .base import BaseTTSEngine

logger = logging.getLogger(__name__)

MMS_MODEL_IDS = {
    "ha": "facebook/mms-tts-hau",
    "fr": "facebook/mms-tts-fra",
}


class MMSTTSEngine(BaseTTSEngine):
    def __init__(self, language: str = "ha"):
        self.language = language.lower().strip()
        self.model_id = MMS_MODEL_IDS.get(self.language, "facebook/mms-tts-hau")
        self.model = None
        self.tokenizer = None
        self.sample_rate = 16000

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self):
        """Loads model and tokenizer weights into RAM."""
        if self.is_loaded():
            return

        logger.info(f"Loading Meta MMS-TTS model for [{self.language}]: {self.model_id}...")
        try:
            import torch
            from transformers import VitsModel, AutoTokenizer

            if torch.get_num_threads() > 4:
                torch.set_num_threads(2)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = VitsModel.from_pretrained(self.model_id)
            self.model.eval()
            self.sample_rate = getattr(self.model.config, "sampling_rate", 16000)
            logger.info(f"Meta MMS-TTS [{self.language}] successfully loaded into RAM.")
        except Exception as e:
            logger.error(f"Failed to load Meta MMS-TTS [{self.language}]: {e}")
            raise

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Hausa or French text into audio waveform.
        Returns (audio_np, sample_rate).
        """
        if not self.is_loaded():
            self.load()

        import torch

        clean_text = text.strip()
        if not clean_text:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), self.sample_rate

        inputs = self.tokenizer(clean_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform

        audio_arr = output.squeeze().cpu().numpy()
        return audio_arr, self.sample_rate
