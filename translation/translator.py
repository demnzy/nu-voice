"""
Educational Translation Engine for Nu-Voice.
Translates English course material into natural, pedagogical Hausa or French
using ultra-fast LLM inference (OpenAI by default, with Groq fallback).
"""

import os
import logging
import httpx
from typing import Optional

try:
    from config import settings
except ImportError:
    from ..config import settings

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "ha": "Hausa",
    "fr": "French",
    "en": "English",
}

HAUSA_SYSTEM_PROMPT = """You are an expert native Hausa linguist and educator.
Translate the following educational course text into natural, spoken Standard Hausa suitable for an audio lecture.
Rules:
1. Ensure the rhythm and sentence structures sound natural when spoken aloud.
2. Use accurate Hausa orthography with hooked consonants (ɓ, ɗ, ƙ, ƴ).
3. Keep technical or domain terms recognizable, explaining them briefly in Hausa if necessary.
4. Output ONLY the translated Hausa text. Do not include markdown preamble, translator notes, or quotes.
"""

FRENCH_SYSTEM_PROMPT = """You are an expert French educational instructor.
Translate the following educational course text into natural, clear, and engaging spoken French suitable for an audio lecture.
Rules:
1. Ensure phrasing flows naturally for speech and listening comprehension.
2. Maintain pedagogical clarity and professional, warm tone.
3. Output ONLY the translated French text. Do not include introductory remarks or metadata.
"""


class Translator:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        default_provider: Optional[str] = None,
    ):
        self.openai_api_key = openai_api_key or getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.groq_api_key = groq_api_key or getattr(settings, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        self.default_provider = (default_provider or getattr(settings, "DEFAULT_TRANSLATION_PROVIDER", "openai") or "openai").lower().strip()

    async def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
    ) -> str:
        """
        Translates text to target_lang.
        If target_lang == source_lang, returns text unchanged.
        Defaults to OpenAI (gpt-6-luna) as the primary provider.
        """
        target_lang = (target_lang or "en").lower().strip()
        source_lang = (source_lang or "en").lower().strip()

        if target_lang == source_lang or not text.strip():
            return text

        system_prompt = HAUSA_SYSTEM_PROMPT if target_lang == "ha" else FRENCH_SYSTEM_PROMPT
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        # 1. Primary Provider: OpenAI (Default)
        if self.default_provider == "openai" and self.openai_api_key:
            try:
                translated = await self._call_openai(text, system_prompt)
                if translated and len(translated.strip()) > 0:
                    return translated.strip()
            except Exception as e:
                logger.warning(f"OpenAI translation failed: {e}. Attempting Groq fallback.")

        # 2. Secondary Provider / Groq Fallback
        if self.groq_api_key:
            try:
                translated = await self._call_groq(text, system_prompt)
                if translated and len(translated.strip()) > 0:
                    return translated.strip()
            except Exception as e:
                logger.warning(f"Groq translation failed: {e}.")

        # 3. Tertiary fallback to OpenAI if Groq was primary but failed
        if self.default_provider != "openai" and self.openai_api_key:
            try:
                translated = await self._call_openai(text, system_prompt)
                if translated and len(translated.strip()) > 0:
                    return translated.strip()
            except Exception as e:
                logger.error(f"OpenAI fallback translation failed: {e}")

        # If all providers fail or are not configured, return original text
        logger.warning(f"Translation to {target_name} unavailable; using original text.")
        return text

    async def _call_openai(self, text: str, system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        model = getattr(settings, "OPENAI_MODEL", None) or os.getenv("OPENAI_TRANSLATE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-6-luna"
        reasoning_effort = getattr(settings, "OPENAI_REASONING_EFFORT", None) or os.getenv("OPENAI_REASONING_EFFORT_TRANSLATE") or "medium"
        if str(reasoning_effort).strip().lower() in ("none", "minimal", "low"):
            reasoning_effort = "medium"

        payload = {
            "model": str(model).strip(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "reasoning_effort": str(reasoning_effort).strip().lower(),
            "max_completion_tokens": 2048,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"OpenAI error HTTP {res.status_code}: {res.text}")

    async def _call_groq(self, text: str, system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        # Try llama-3.1-8b-instant which is universally available on all Groq tiers
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
            "max_completion_tokens": 2048,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"Groq error HTTP {res.status_code}: {res.text}")
