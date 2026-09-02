"""Audio encoding for the local TTS API.

Encodes MyVoice's native 24 kHz mono int16 audio into the three response
formats the OpenAI ``/v1/audio/speech`` surface exposes:

- ``mp3``  -> ``lameenc`` (128 kbps, pure wheels, no ffmpeg) -> ``audio/mpeg``
- ``wav``  -> ``soundfile`` (PCM_16) -> ``audio/wav``
- ``pcm``  -> raw little-endian int16 passthrough -> ``audio/L16; rate=24000``

Two entry points:

- :func:`encode_buffered` for the whole-clip (``stream:false``) path.
- :class:`StreamEncoder` for incremental (``stream:true``) mp3/pcm framing.

Native format is fixed at 24 kHz mono (see tech-spec: Native audio format).
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MyVoice native output (qwen_tts_service.py): 24 kHz, mono, int16 playback.
SAMPLE_RATE = 24000
CHANNELS = 1
MP3_BITRATE_KBPS = 128

# Pinned media types (tech-spec H5). Use these exact strings for both the
# buffered and streamed variants of each format.
MEDIA_TYPE_MP3 = "audio/mpeg"
MEDIA_TYPE_WAV = "audio/wav"
MEDIA_TYPE_PCM = "audio/L16; rate=24000"

VALID_FORMATS = ("mp3", "wav", "pcm")


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] mono array to little-endian int16 bytes.

    Mirrors the GUI playback conversion at app.py:2990
    (``np.clip(...) * 32767 -> int16 -> tobytes()``) so the API and desktop
    paths produce byte-identical PCM.
    """
    if audio is None:
        return b""
    arr = np.asarray(audio)
    if arr.dtype != np.int16:
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return arr.tobytes()


def _encode_mp3(int16_bytes: bytes) -> bytes:
    """Encode raw int16 PCM bytes to a complete MP3 payload via lameenc."""
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE_KBPS)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(CHANNELS)
    encoder.set_quality(2)  # 2 = high quality / near-best, reasonable speed
    mp3 = encoder.encode(int16_bytes)
    mp3 += encoder.flush()
    return bytes(mp3)


def _encode_wav(int16_bytes: bytes) -> bytes:
    """Encode raw int16 PCM bytes to an in-memory WAV (PCM_16) via soundfile."""
    import soundfile as sf

    samples = np.frombuffer(int16_bytes, dtype=np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def encode_buffered(audio: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """Encode a whole-clip float32 array to ``(payload, media_type)``.

    Args:
        audio: float32 mono array (``QwenTTSResponse.audio_data``).
        sample_rate: source sample rate (expected 24000; logged if not).
        fmt: one of ``"mp3"``, ``"wav"``, ``"pcm"``.

    Raises:
        ValueError: on an unsupported ``fmt``.
    """
    if fmt not in VALID_FORMATS:
        raise ValueError(f"Unsupported audio format: {fmt!r}")

    if sample_rate != SAMPLE_RATE:
        # v1 is fixed at 24 kHz; we do not resample. Warn but proceed so a
        # config drift surfaces in logs rather than silently shipping bad audio.
        logger.warning(
            "encode_buffered got sample_rate=%s, expected %s; encoding as-is",
            sample_rate,
            SAMPLE_RATE,
        )

    int16_bytes = float32_to_int16_bytes(audio)

    if fmt == "pcm":
        return int16_bytes, MEDIA_TYPE_PCM
    if fmt == "wav":
        return _encode_wav(int16_bytes), MEDIA_TYPE_WAV
    return _encode_mp3(int16_bytes), MEDIA_TYPE_MP3


class StreamEncoder:
    """Incremental encoder for the chunked streaming path.

    For ``mp3`` it holds a persistent ``lameenc.Encoder`` and frames each
    int16 chunk as it arrives, emitting a final ``flush()`` at end-of-stream.
    For ``pcm`` chunks pass through as raw int16 bytes (``flush`` is empty).

    ``wav`` is intentionally unsupported for streaming (a chunked RIFF can't
    declare its length up front) — the route rejects ``stream + wav`` with 400.
    """

    def __init__(self, fmt: str):
        if fmt not in ("mp3", "pcm"):
            raise ValueError(f"Streaming not supported for format: {fmt!r}")
        self.fmt = fmt
        self._encoder = None
        if fmt == "mp3":
            import lameenc

            enc = lameenc.Encoder()
            enc.set_bit_rate(MP3_BITRATE_KBPS)
            enc.set_in_sample_rate(SAMPLE_RATE)
            enc.set_channels(CHANNELS)
            enc.set_quality(2)
            self._encoder = enc

    @property
    def media_type(self) -> str:
        return MEDIA_TYPE_MP3 if self.fmt == "mp3" else MEDIA_TYPE_PCM

    def encode_chunk(self, int16_bytes: bytes) -> bytes:
        """Encode one chunk of raw int16 PCM bytes; may return b'' for mp3."""
        if not int16_bytes:
            return b""
        if self.fmt == "pcm":
            return int16_bytes
        return bytes(self._encoder.encode(int16_bytes))

    def flush(self) -> bytes:
        """Return any encoder tail (mp3) or empty bytes (pcm)."""
        if self.fmt == "pcm" or self._encoder is None:
            return b""
        return bytes(self._encoder.flush())
