"""
Nu-Voice FastAPI Microservice.
High-throughput, low-latency streaming TTS engine supporting English, French, and Hausa.
Features decoupled background synthesis, Bunny CDN persistent caching, client disconnect resiliency,
and task deduplication.
"""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
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
logging.getLogger("phonemizer").setLevel(logging.ERROR)
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
_ACTIVE_SYNTHESES: Dict[str, asyncio.Task] = {}


class LessonStreamRequest(BaseModel):
    lesson_id: str
    title: Optional[str] = ""
    content_text: str
    target_lang: str = Field(default="en", description="Target audio language: en, ha, fr")
    source_lang: str = Field(default="en", description="Source text language: en")
    format: str = Field(default="mp3", description="Audio format: mp3 or wav")


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@app.get("/")
async def root():
    return {
        "service": "nu-voice",
        "status": "healthy",
        "version": "1.0.0",
        "cache_entries": len(AUDIO_CACHE),
        "active_syntheses": len(_ACTIVE_SYNTHESES),
    }


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


async def _synthesize_lesson_to_cdn(
    cache_key: str,
    lesson_id: str,
    target_lang: str,
    content_hash: str,
    sentences: List[str],
    audio_format: str,
    chunk_queue: Optional[asyncio.Queue] = None,
):
    """
    Decoupled background worker that synthesizes ALL sentences to completion
    and uploads the full audio to Bunny CDN.
    If chunk_queue is provided, pushes chunks as they complete for real-time streaming.
    Even if client disconnects, this task continues to completion so future requests get instant cache hits!
    """
    accumulated_audio = bytearray()
    logger.info(f"Decoupled synthesis worker started for lesson {lesson_id} [{target_lang}]: {len(sentences)} sentences")
    try:
        for idx, sentence in enumerate(sentences):
            try:
                chunk = await asyncio.to_thread(
                    engine_manager.synthesize_sentence_audio_bytes,
                    sentence,
                    lang=target_lang,
                    format=audio_format,
                )
                accumulated_audio.extend(chunk)
                if chunk_queue:
                    await chunk_queue.put(chunk)
            except Exception as e:
                logger.error(f"Failed chunk {idx+1}/{len(sentences)} for lesson {lesson_id}: {e}")

        if chunk_queue:
            await chunk_queue.put(None)  # Sentinel for completion

        if len(accumulated_audio) > 0:
            logger.info(f"Synthesized full audio for lesson {lesson_id} [{target_lang}] ({len(accumulated_audio)} bytes). Uploading to Bunny CDN.")
            filename = f"{lesson_id}_{target_lang}_{content_hash}.mp3"
            cdn_url = await bunny_storage.upload_audio_bytes(
                audio_bytes=bytes(accumulated_audio),
                filename=filename,
                folder_path="lessons/audio",
            )
            AUDIO_CACHE[cache_key] = cdn_url
            logger.info(f"Successfully cached lesson {lesson_id} [{target_lang}] to Bunny CDN -> {cdn_url}")
            return cdn_url
    except Exception as e:
        logger.error(f"Background synthesis error for lesson {lesson_id} [{target_lang}]: {e}")
        if chunk_queue:
            await chunk_queue.put(None)
    finally:
        _ACTIVE_SYNTHESES.pop(cache_key, None)


@app.post("/v1/tts/lesson/prepare")
@app.get("/v1/tts/lesson/prepare")
async def prepare_lesson_audio(
    payload: LessonStreamRequest,
):
    """
    Explicit background preparation endpoint.
    If already cached: returns CDN URL immediately.
    If in progress: returns status='generating'.
    If new: launches decoupled background synthesis and CDN upload.
    """
    lesson_id = payload.lesson_id.strip()
    target_lang = payload.target_lang.lower().strip()
    source_lang = payload.source_lang.lower().strip()
    text = payload.content_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Lesson text cannot be empty.")

    content_hash = compute_content_hash(text)
    cache_key = f"{lesson_id}_{target_lang}_{content_hash}"

    if cache_key in AUDIO_CACHE:
        return {
            "status": "ready",
            "cached": True,
            "audio_url": AUDIO_CACHE[cache_key],
            "lesson_id": lesson_id,
            "language": target_lang,
        }

    if cache_key in _ACTIVE_SYNTHESES:
        return {
            "status": "generating",
            "cached": False,
            "lesson_id": lesson_id,
            "language": target_lang,
            "message": "Synthesis in progress",
        }

    # Translate if necessary
    if target_lang != source_lang:
        try:
            text = await translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)
        except Exception as e:
            logger.error(f"Translation failed: {e}. Proceeding with original text.")

    sentences = prepare_text_for_speech(text, lang=target_lang)
    if not sentences:
        raise HTTPException(status_code=400, detail="No readable sentences in lesson.")

    # Launch decoupled background synthesis
    task = asyncio.create_task(
        _synthesize_lesson_to_cdn(
            cache_key=cache_key,
            lesson_id=lesson_id,
            target_lang=target_lang,
            content_hash=content_hash,
            sentences=sentences,
            audio_format=payload.format,
            chunk_queue=None,
        )
    )
    _ACTIVE_SYNTHESES[cache_key] = task

    return {
        "status": "generating",
        "cached": False,
        "lesson_id": lesson_id,
        "language": target_lang,
        "message": "Synthesis started in background",
    }


@app.post("/v1/tts/lesson/stream")
async def stream_lesson_audio(
    payload: LessonStreamRequest,
    request: Request,
):
    lesson_id = payload.lesson_id.strip()
    target_lang = payload.target_lang.lower().strip()
    source_lang = payload.source_lang.lower().strip()
    text = payload.content_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Lesson text cannot be empty.")

    content_hash = compute_content_hash(text)
    cache_key = f"{lesson_id}_{target_lang}_{content_hash}"

    # 1. Instant Cache Hit
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

    # 2. Translate if needed
    if target_lang != source_lang:
        logger.info(f"Translating lesson {lesson_id} from {source_lang} to {target_lang}...")
        try:
            text = await translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)
        except Exception as e:
            logger.error(f"Translation failed: {e}. Proceeding with original text.")

    sentences = prepare_text_for_speech(text, lang=target_lang)
    if not sentences:
        raise HTTPException(status_code=400, detail="No readable sentences in lesson.")

    # 3. Queue-based streaming decoupled from client drops
    chunk_queue: asyncio.Queue = asyncio.Queue()

    # Launch decoupled task if not already running
    if cache_key not in _ACTIVE_SYNTHESES:
        task = asyncio.create_task(
            _synthesize_lesson_to_cdn(
                cache_key=cache_key,
                lesson_id=lesson_id,
                target_lang=target_lang,
                content_hash=content_hash,
                sentences=sentences,
                audio_format=payload.format,
                chunk_queue=chunk_queue,
            )
        )
        _ACTIVE_SYNTHESES[cache_key] = task

    async def lesson_audio_generator():
        logger.info(f"Streaming lesson {lesson_id} [{target_lang}] to client...")
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"Client disconnected during stream of lesson {lesson_id}. Decoupled background synthesis continues.")
                    break

                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    if cache_key not in _ACTIVE_SYNTHESES:
                        break
                    continue

                if chunk is None:  # Sentinel
                    break
                yield chunk
        except Exception as e:
            logger.warning(f"Stream generator notice for lesson {lesson_id}: {e}")

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


@app.get("/v1/tts/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "nu-voice",
        "port": settings.PORT,
        "loaded_engines": list(engine_manager.engines.keys()),
        "cache_entries": len(AUDIO_CACHE),
        "active_syntheses": len(_ACTIVE_SYNTHESES),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.HOST, port=settings.PORT, reload=True)
