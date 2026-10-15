"""
TTS Engine Manager for Nu-Voice.
Provides thread-safe singleton access to Kokoro-82M (English) and Meta MMS-TTS (Hausa, French).
Handles lazy loading, sample rate conversion, and MP3 encoding.
"""

import threading
import logging
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from engines.base import BaseTTSEngine
    from engines.audio_encoder import encode_to_mp3, encode_to_wav
except ImportError:
    from .base import BaseTTSEngine
    from .audio_encoder import encode_to_mp3, encode_to_wav

logger = logging.getLogger(__name__)


class EngineManager:
    _instance: Optional["EngineManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EngineManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.engines: Dict[str, BaseTTSEngine] = {}
        self._engine_locks = {
            "en": threading.Lock(),
            "ha": threading.Lock(),
            "fr": threading.Lock(),
        }

    def get_engine(self, lang: str) -> BaseTTSEngine:
        """Lazily retrieves or initializes the voice engine for the target language."""
        lang = (lang or "en").lower().strip()
        if lang not in self._engine_locks:
            lang = "en"

        with self._engine_locks[lang]:
            if lang in self.engines and self.engines[lang].is_loaded():
                return self.engines[lang]

            if lang == "en":
                try:
                    from engines.kokoro_engine import KokoroTTSEngine
                except ImportError:
                    from .kokoro_engine import KokoroTTSEngine
                engine = KokoroTTSEngine()
            elif lang in ("ha", "fr"):
                try:
                    from engines.mms_engine import MMSTTSEngine
                except ImportError:
                    from .mms_engine import MMSTTSEngine
                engine = MMSTTSEngine(language=lang)
            else:
                try:
                    from engines.kokoro_engine import KokoroTTSEngine
                except ImportError:
                    from .kokoro_engine import KokoroTTSEngine
                engine = KokoroTTSEngine()

            try:
                engine.load()
            except Exception as e:
                logger.error(f"Error loading voice model for {lang}: {e}. Creating mock test fallback.")
                engine = _MockTestEngine(sample_rate=16000)

            self.engines[lang] = engine
            return engine

    def synthesize_sentence(self, text: str, lang: str = "en") -> Tuple[np.ndarray, int]:
        """Synthesizes sentence and returns (audio_samples, sample_rate)."""
        engine = self.get_engine(lang)
        return engine.synthesize(text)

    def synthesize_sentence_audio_bytes(self, text: str, lang: str = "en", format: str = "mp3") -> bytes:
        """Synthesizes sentence and returns encoded MP3 (or WAV) bytes."""
        samples, sr = self.synthesize_sentence(text, lang)
        if format.lower() == "wav":
            return encode_to_wav(samples, sr)
        return encode_to_mp3(samples, sr)


class _MockTestEngine(BaseTTSEngine):
    """Generates synthetic speech-like waveforms when heavy model weights are not installed."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load(self):
        pass

    def is_loaded(self) -> bool:
        return True

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        duration = max(0.4, min(3.0, len(text) * 0.04))
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False, dtype=np.float32)
        envelope = np.sin(np.pi * np.linspace(0, 1, len(t), dtype=np.float32))
        audio = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)) * envelope
        return audio, self.sample_rate
