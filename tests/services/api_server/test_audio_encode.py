"""Encoder round-trip tests for the local TTS API (tech-spec Task 12)."""

import io

import numpy as np
import pytest
import soundfile as sf

from myvoice.services.api_server import audio_encode
from myvoice.services.api_server.audio_encode import (
    MEDIA_TYPE_MP3,
    MEDIA_TYPE_PCM,
    MEDIA_TYPE_WAV,
    StreamEncoder,
    encode_buffered,
    float32_to_int16_bytes,
)


@pytest.fixture
def tone():
    """0.5 s 220 Hz mono float32 tone at 24 kHz (12000 samples)."""
    t = np.linspace(0, 0.5, 12000, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_pcm_is_raw_int16_passthrough(tone):
    payload, media_type = encode_buffered(tone, 24000, "pcm")
    assert media_type == MEDIA_TYPE_PCM
    # 2 bytes/sample, mono.
    assert len(payload) == tone.size * 2
    assert payload == float32_to_int16_bytes(tone)


def test_wav_round_trips_to_24k_mono(tone):
    payload, media_type = encode_buffered(tone, 24000, "wav")
    assert media_type == MEDIA_TYPE_WAV
    data, sr = sf.read(io.BytesIO(payload), dtype="int16")
    assert sr == 24000
    assert data.ndim == 1  # mono
    assert data.shape[0] == tone.size


def test_mp3_produces_valid_mpeg_payload(tone):
    payload, media_type = encode_buffered(tone, 24000, "mp3")
    assert media_type == MEDIA_TYPE_MP3
    assert len(payload) > 0
    # lameenc emits raw MPEG frames; first byte is the frame-sync 0xFF.
    assert payload[0] == 0xFF


def test_unsupported_format_raises(tone):
    with pytest.raises(ValueError):
        encode_buffered(tone, 24000, "ogg")


def test_clipping_handles_out_of_range_floats():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    payload = float32_to_int16_bytes(loud)
    samples = np.frombuffer(payload, dtype=np.int16)
    assert samples[0] == 32767
    assert samples[1] == -32767


def test_stream_encoder_pcm_passthrough_and_empty_flush():
    enc = StreamEncoder("pcm")
    assert enc.media_type == MEDIA_TYPE_PCM
    chunk = np.array([1, 2, 3], dtype=np.int16).tobytes()
    assert enc.encode_chunk(chunk) == chunk
    assert enc.flush() == b""


def test_stream_encoder_mp3_frames_then_flush(tone):
    enc = StreamEncoder("mp3")
    assert enc.media_type == MEDIA_TYPE_MP3
    int16 = float32_to_int16_bytes(tone)
    body = enc.encode_chunk(int16) + enc.flush()
    assert len(body) > 0
    assert body[0] == 0xFF


def test_stream_encoder_rejects_wav():
    with pytest.raises(ValueError):
        StreamEncoder("wav")
