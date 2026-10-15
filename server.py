"""
Nu-Voice FastAPI Microservice.
High-throughput, low-latency streaming TTS engine supporting English, French, and Hausa.
Features sentence-by-sentence streaming, client disconnect cancellation, and Bunny CDN caching.
"""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks, Header, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from linguistics.normalizer import prepare_text_for_speech
from translation.translator import Translator
from storage.bunny_storage import BunnyAudioStorage
from engines.engine_manager import EngineManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nu_voice")

app = FastAPI(
    title="Nu-Voice TTS Streaming Service",
    version="1.0.0",
    description="Reusable multi-language speech streaming microservice for the Nu-Age ecosystem",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_manager = EngineManager()
translator = Translator()
bunny_storage = BunnyAudioStorage()

AUDIO_CACHE: Dict[str, str] = {}


class LessonStreamRequest(BaseModel):
    lesson_id: str
    title: Optional[str] = ""
    content_text: str
    target_lang: str = Field(default="en", description="Target audio language: en, ha, fr")
    source_lang: str = Field(default="en", description="Source text language: en")
    format: str = Field(default="mp3", description="Audio format: mp3 or wav")


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@app.get("/v1/tts/stream")
async def stream_generic_text(
    request: Request,
    text: str = Query(..., description="Text to speak"),
    lang: str = Query("en", description="Language code: en, ha, fr"),
    format: str = Query("mp3", description="Audio format: mp3 or wav"),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    sentences = prepare_text_for_speech(text, lang=lang)
    if not sentences:
        raise HTTPException(status_code=400, detail="No readable sentences found.")

    async def audio_generator():
        logger.info(f"Starting generic TTS stream: lang={lang}, sentences={len(sentences)}")
        for idx, sentence in enumerate(sentences):
            if await request.is_disconnected():
                logger.info(f"Client disconnected during sentence {idx+1}/{len(sentences)}. Halting TTS.")
                return

            try:
                chunk = await asyncio.to_thread(
                    engine_manager.synthesize_sentence_audio_bytes,
                    sentence,
                    lang=lang,
                    format=format,
                )
                yield chunk
            except Exception as e:
                logger.error(f"Error synthesizing sentence '{sentence[:30]}...': {e}")
                continue

    media_type = "audio/mpeg" if format.lower() == "mp3" else "audio/wav"
    return StreamingResponse(
        audio_generator(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/v1/tts/lesson/stream")
async def stream_lesson_audio(
    payload: LessonStreamRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    lesson_id = payload.lesson_id.strip()
    target_lang = payload.target_lang.lower().strip()
    source_lang = payload.source_lang.lower().strip()
    text = payload.content_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Lesson text cannot be empty.")

    content_hash = compute_content_hash(text)
    cache_key = f"{lesson_id}_{target_lang}_{content_hash}"

    if cache_key in AUDIO_CACHE:
        cdn_url = AUDIO_CACHE[cache_key]
        logger.info(f"Cache HIT for lesson {lesson_id} [{target_lang}]: {cdn_url}")
        return JSONResponse(
            content={
                "cached": True,
                "audio_url": cdn_url,
                "lesson_id": lesson_id,
                "language": target_lang,
            }
        )

    if target_lang != source_lang:
        logger.info(f"Translating lesson {lesson_id} from {source_lang} to {target_lang}...")
        try:
            text = await translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)
        except Exception as e:
            logger.error(f"Translation failed: {e}. Proceeding with original text.")

    sentences = prepare_text_for_speech(text, lang=target_lang)
    if not sentences:
        raise HTTPException(status_code=400, detail="No readable sentences in lesson.")

    accumulated_audio = bytearray()

    async def lesson_audio_generator():
        logger.info(f"Streaming lesson {lesson_id} [{target_lang}]: {len(sentences)} sentences")
        all_completed = True

        for idx, sentence in enumerate(sentences):
            if await request.is_disconnected():
                logger.info(f"Student disconnected at sentence {idx+1}/{len(sentences)} of lesson {lesson_id}. Aborting.")
                all_completed = False
                return

            try:
                chunk = await asyncio.to_thread(
                    engine_manager.synthesize_sentence_audio_bytes,
                    sentence,
                    lang=target_lang,
                    format=payload.format,
                )
                accumulated_audio.extend(chunk)
                yield chunk
            except Exception as e:
                logger.error(f"Failed chunk {idx+1} for lesson {lesson_id}: {e}")
                continue

        if all_completed and len(accumulated_audio) > 0:
            logger.info(f"Synthesized full audio for lesson {lesson_id} ({len(accumulated_audio)} bytes). Queueing CDN upload.")
            filename = f"{lesson_id}_{target_lang}_{content_hash}.mp3"
            background_tasks.add_task(
                _upload_and_record_cache,
                cache_key=cache_key,
                filename=filename,
                audio_bytes=bytes(accumulated_audio),
            )

    media_type = "audio/mpeg" if payload.format.lower() == "mp3" else "audio/wav"
    return StreamingResponse(
        lesson_audio_generator(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
            "X-Lesson-Id": lesson_id,
            "X-Audio-Language": target_lang,
        },
    )


async def _upload_and_record_cache(cache_key: str, filename: str, audio_bytes: bytes):
    try:
        cdn_url = await bunny_storage.upload_audio_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            folder_path="lessons/audio",
        )
        AUDIO_CACHE[cache_key] = cdn_url
        logger.info(f"Registered audio cache key '{cache_key}' -> {cdn_url}")
    except Exception as e:
        logger.error(f"Failed to cache lesson audio to Bunny CDN: {e}")


@app.get("/v1/tts/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "nu-voice",
        "port": settings.PORT,
        "loaded_engines": list(engine_manager.engines.keys()),
        "cache_entries": len(AUDIO_CACHE),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.HOST, port=settings.PORT, reload=True)
