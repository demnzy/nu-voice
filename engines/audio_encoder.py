"""
Audio Encoding & Streaming Chunker.
Converts floating-point PCM audio arrays into standard MP3 frames or WAV chunks
for immediate streaming and client playback.
"""

import io
import struct
import numpy as np
from typing import Optional


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Clips and converts float32 [-1.0, 1.0] audio array to int16 PCM."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def encode_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encodes a single chunk into standard RIFF WAV format."""
    pcm16 = float_to_int16(audio)
    num_samples = len(pcm16)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * (bits_per_sample // 8)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt subchunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # Subchunk1Size (16 for PCM)
    buf.write(struct.pack("<H", 1))   # AudioFormat (1 for PCM)
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data subchunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm16.tobytes())

    return buf.getvalue()


def encode_to_mp3(audio: np.ndarray, sample_rate: int, bitrate_kbps: int = 128) -> bytes:
    """
    Encodes PCM audio array into standard MP3 bytes.
    Tries lameenc first; falls back to pydub / ffmpeg, or WAV if MP3 encoders are absent.
    """
    pcm16 = float_to_int16(audio)

    # 1. Try lameenc (fastest C-extension MP3 encoder)
    try:
        import lameenc
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate_kbps)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)  # High quality
        mp3_data = encoder.encode(pcm16.tobytes())
        mp3_data += encoder.flush()
        return bytes(mp3_data)
    except ImportError:
        pass

    # 2. Try pydub if installed
    try:
        from pydub import AudioSegment
        segment = AudioSegment(
            pcm16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
        out = io.BytesIO()
        segment.export(out, format="mp3", bitrate=f"{bitrate_kbps}k")
        return out.getvalue()
    except Exception:
        pass

    # 3. Fallback: Return WAV bytes (playable by all browser & desktop players)
    return encode_to_wav(audio, sample_rate)
