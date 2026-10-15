"""
Bunny.net CDN Storage Adapter for Nu-Voice.
Uploads generated lesson audio files to Bunny Storage and returns
the high-speed public CDN Pull Zone URL.
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


class BunnyAudioStorage:
    def __init__(
        self,
        storage_key: Optional[str] = None,
        storage_zone: Optional[str] = None,
        region_url: Optional[str] = None,
        pull_zone_url: Optional[str] = None,
    ):
        self.storage_key = storage_key or settings.BUNNY_STORAGE_KEY or os.getenv("BUNNY_STORAGE_KEY", "")
        self.storage_zone = storage_zone or settings.STORAGE_ZONE_NAME or os.getenv("STORAGE_ZONE_NAME", "")
        self.region_url = region_url or settings.BUNNY_REGION_URL or os.getenv("BUNNY_REGION_URL", "storage.bunnycdn.com")
        self.pull_zone_url = (pull_zone_url or settings.PULL_ZONE_URL or os.getenv("PULL_ZONE_URL", "")).rstrip("/")

    async def upload_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        folder_path: str = "lessons/audio",
    ) -> str:
        """
        Uploads audio file to Bunny.net and returns the permanent CDN URL.
        """
        if not self.storage_key or not self.storage_zone:
            logger.warning("Bunny.net credentials not configured. Returning local mock path.")
            return f"/local_audio/{folder_path}/{filename}"

        clean_folder = folder_path.strip("/")
        upload_url = f"https://{self.region_url}/{self.storage_zone}/{clean_folder}/{filename}"

        headers = {
            "AccessKey": self.storage_key,
            "Content-Type": "audio/mpeg",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.put(upload_url, content=audio_bytes, headers=headers)
            if res.status_code not in (200, 201):
                logger.error(f"Bunny CDN Upload failed: HTTP {res.status_code} - {res.text}")
                raise RuntimeError(f"Bunny CDN upload failed: {res.text}")

        cdn_url = f"{self.pull_zone_url}/{clean_folder}/{filename}"
        logger.info(f"Successfully uploaded audio to Bunny CDN: {cdn_url}")
        return cdn_url
