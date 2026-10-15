"""
Base TTS Engine Contract.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseTTSEngine(ABC):
    @abstractmethod
    def load(self):
        """Loads model weights and tokenizers into RAM."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the engine is warm and ready for inference."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesizes a single sentence or text snippet.
        Returns:
            Tuple of (audio_samples as float32 numpy array, sample_rate_hz)
        """
        pass
