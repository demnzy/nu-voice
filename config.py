"""
Configuration & Environment Settings for Nu-Voice.
Loads credentials for Bunny.net CDN, Groq/OpenAI translation, and server options.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent / ".env"

class Settings(BaseSettings):
    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Optional Shared Secret for API Authentication
    VOICE_API_KEY: str = ""

    # Bunny.net CDN Storage
    BUNNY_STORAGE_KEY: str = ""
    STORAGE_ZONE_NAME: str = ""
    BUNNY_REGION_URL: str = "storage.bunnycdn.com"
    PULL_ZONE_URL: str = ""

    # Translation Providers (Groq / OpenAI)
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Engine Options
    DEFAULT_SPEED: float = 1.0
    DEFAULT_ENGLISH_VOICE: str = "af_heart"
    CPU_THREADS: int = 2

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        extra="ignore"
    )

settings = Settings()
