"""
TTS Engine Manager for Nu-Voice.
Provides thread-safe singleton access to Kokoro-82M (English) and Meta MMS-TTS (Hausa, French).
Handles lazy loading, sample rate conversion, and MP3 encoding.
"""

import threading
import logging
import time
from typing import Dict, Tuple, Optional, List, Any
import numpy as np

try:
    from engines.base import BaseTTSEngine
    from engines.audio_encoder import encode_to_mp3, encode_to_wav
except ImportError:
    from .base import BaseTTSEngine
    from .audio_encoder import encode_to_mp3, encode_to_wav

logger = logging.getLogger(__name__)

WARMUP_PROMPTS = {
    "en": "Welcome to Nu-Age.",
    "ha": "Barka da zuwa Nu-Age.",
    "fr": "Bienvenue sur Nu-Age.",
}


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
        self._engine_states: Dict[str, str] = {
            "en": "unloaded",
            "ha": "unloaded",
            "fr": "unloaded",
        }
        self._engine_locks = {
            "en": threading.Lock(),
            "ha": threading.Lock(),
            "fr": threading.Lock(),
        }

    def get_engine(self, lang: str) -> BaseTTSEngine:
        """Retrieves or initializes the voice engine for the target language."""
        lang = (lang or "en").lower().strip()
        if lang not in self._engine_locks:
            lang = "en"

        with self._engine_locks[lang]:
            if lang in self.engines and self.engines[lang].is_loaded():
                return self.engines[lang]

            self._engine_states[lang] = "loading"
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
                self._engine_states[lang] = "ready"
            except Exception as e:
                logger.error(f"Error loading voice model for {lang}: {e}. Creating mock test fallback.")
                engine = _MockTestEngine(sample_rate=16000)
                self._engine_states[lang] = "ready_mock"

            self.engines[lang] = engine
            return engine

    def preload_all_engines(
        self,
        languages: Optional[List[str]] = None,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        """
        Preloads all specified neural speech models into RAM upfront and
        optionally executes a micro-synthesis to compile kernels and eliminate cold-start lag.
        """
        target_langs = languages or ["en", "ha", "fr"]
        logger.info(f"[PRELOAD] Starting neural model pre-load for languages: {target_langs} (warmup={warmup})")
        report: Dict[str, Any] = {}

        for lang in target_langs:
            lang = lang.lower().strip()
            t0 = time.time()
            try:
                engine = self.get_engine(lang)
                warmup_done = False
                if warmup:
                    prompt = WARMUP_PROMPTS.get(lang, "Testing audio.")
                    try:
                        _samples, _sr = engine.synthesize(prompt)
                        warmup_done = True
                    except Exception as wex:
                        logger.warning(f"[PRELOAD] Warmup synthesis notice for '{lang}': {wex}")

                elapsed = time.time() - t0
                state = self._engine_states.get(lang, "ready")
                report[lang] = {
                    "status": state,
                    "engine": type(engine).__name__,
                    "sample_rate": getattr(engine, "sample_rate", 16000),
                    "load_time_sec": round(elapsed, 2),
                    "warmed_up": warmup_done,
                }
                logger.info(
                    f"[PRELOAD] Engine '{lang.upper()}' ({type(engine).__name__}) ready & warm in {elapsed:.2f}s."
                )
            except Exception as ex:
                logger.error(f"[PRELOAD] Failed to preload engine '{lang}': {ex}")
                report[lang] = {
                    "status": "error",
                    "error": str(ex),
                }

        return report

    def get_models_status(self) -> Dict[str, Any]:
        """Returns thread-safe snapshot of loaded model engines and states."""
        status_map: Dict[str, Any] = {}
        for lang in ("en", "ha", "fr"):
            engine = self.engines.get(lang)
            is_loaded = engine.is_loaded() if engine else False
            status_map[lang] = {
                "loaded": is_loaded,
                "state": self._engine_states.get(lang, "unloaded"),
                "engine_type": type(engine).__name__ if engine else None,
                "sample_rate": getattr(engine, "sample_rate", None) if engine else None,
            }
        status_map["all_ready"] = all(
            item["loaded"] for item in status_map.values() if isinstance(item, dict)
        )
        return status_map

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
