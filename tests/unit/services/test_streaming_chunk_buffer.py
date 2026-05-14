"""Unit tests for StreamingChunkBuffer (consumer-side smoothing layer).

Covers the watermark + chunk-boundary crossfade behavior. These are pure
in-memory tests against int16 byte payloads — no PyAudio, no fixtures
beyond construction.

Originated from RTX 3060 + 0.6B small tier underrun-induced silences +
chunk-boundary clicks observed 2026-05-12 (see
``epic18_producer_bottleneck_finding`` for the producer-side regime;
this is the consumer-side fix).
"""

from __future__ import annotations

import numpy as np
import pytest

from myvoice.services.streaming_chunk_buffer import StreamingChunkBuffer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_chunk(n_samples: int, value: int = 1000) -> bytes:
    """Make an int16 mono chunk of n_samples filled with `value`."""
    return np.full(n_samples, value, dtype=np.int16).tobytes()


def _samples(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.int16)


# 24kHz mono int16 → 48000 bytes/s → 100ms = 4800 bytes = 2400 samples
_SAMPLES_PER_100MS_24K = 2400


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_rejects_negative_watermark(self):
        with pytest.raises(ValueError, match="watermark_ms"):
            StreamingChunkBuffer(watermark_ms=-1)

    def test_rejects_negative_crossfade(self):
        with pytest.raises(ValueError, match="crossfade_samples"):
            StreamingChunkBuffer(crossfade_samples=-1)

    def test_rejects_non_int16_sample_width(self):
        with pytest.raises(ValueError, match="int16"):
            StreamingChunkBuffer(sample_width=4)

    def test_rejects_zero_channels(self):
        with pytest.raises(ValueError, match="channels"):
            StreamingChunkBuffer(channels=0)

    def test_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            StreamingChunkBuffer(sample_rate=0)

    def test_zero_watermark_and_crossfade_are_legal(self):
        # All-pass-through configuration — useful for tests that want to
        # verify integration plumbing without buffering side-effects.
        buf = StreamingChunkBuffer(watermark_ms=0, crossfade_samples=0)
        assert buf.watermark_ms == 0
        assert buf.crossfade_samples == 0


# --------------------------------------------------------------------------- #
# Watermark behavior
# --------------------------------------------------------------------------- #


class TestWatermark:
    def test_chunks_below_threshold_held_back(self):
        buf = StreamingChunkBuffer(
            watermark_ms=500, crossfade_samples=0, sample_rate=24000, channels=1
        )
        # 500ms @ 24kHz = 12000 samples = 24000 bytes. Push 100ms.
        out = buf.push(_make_chunk(_SAMPLES_PER_100MS_24K))
        assert out == []
        assert not buf.is_watermark_filled

    def test_threshold_cross_flushes_combined_payload(self):
        buf = StreamingChunkBuffer(
            watermark_ms=200, crossfade_samples=0, sample_rate=24000, channels=1
        )
        # 200ms @ 24kHz = 4800 samples. Push 100ms + 150ms (crosses).
        c1 = _make_chunk(_SAMPLES_PER_100MS_24K, value=1000)
        c2 = _make_chunk(int(_SAMPLES_PER_100MS_24K * 1.5), value=2000)
        assert buf.push(c1) == []
        out = buf.push(c2)
        assert len(out) == 1
        # Combined payload should equal c1 + c2 (no crossfade configured)
        assert out[0] == c1 + c2
        assert buf.is_watermark_filled

    def test_is_final_short_chunk_flushes_immediately(self):
        buf = StreamingChunkBuffer(
            watermark_ms=500, crossfade_samples=0, sample_rate=24000, channels=1
        )
        # Short utterance — flush regardless of watermark.
        c1 = _make_chunk(_SAMPLES_PER_100MS_24K, value=500)
        out = buf.push(c1, is_final=True)
        assert len(out) == 1
        assert out[0] == c1
        assert buf.is_watermark_filled

    def test_passthrough_after_watermark_filled(self):
        buf = StreamingChunkBuffer(
            watermark_ms=100, crossfade_samples=0, sample_rate=24000, channels=1
        )
        # Cross watermark in one push.
        c1 = _make_chunk(_SAMPLES_PER_100MS_24K, value=1000)
        out1 = buf.push(c1)
        assert len(out1) == 1
        # Subsequent chunks pass through one-at-a-time.
        c2 = _make_chunk(1000, value=500)
        out2 = buf.push(c2)
        assert len(out2) == 1
        assert out2[0] == c2

    def test_zero_watermark_first_push_dispatches(self):
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=0, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(100, value=1234)
        out = buf.push(c1)
        assert out == [c1]

    def test_empty_push_without_final_returns_nothing(self):
        buf = StreamingChunkBuffer(watermark_ms=100, crossfade_samples=0)
        assert buf.push(b"") == []


# --------------------------------------------------------------------------- #
# Crossfade behavior
# --------------------------------------------------------------------------- #


class TestCrossfade:
    def test_first_chunk_unmodified(self):
        # No previous tail to blend with — first dispatched chunk is identical.
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=64, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(1000, value=5000)
        out = buf.push(c1)
        assert out[0] == c1

    def test_second_chunk_head_blended_with_first_chunk_tail(self):
        # Configure deterministic tail/head pair: chunk1 = constant 10000,
        # chunk2 = constant 20000, K=4 samples. Linear ramp 0..1 over 4
        # samples → coefs [0, 1/3, 2/3, 1].
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=4, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(100, value=10000)
        c2 = _make_chunk(100, value=20000)
        buf.push(c1)
        out = buf.push(c2)
        blended = _samples(out[0])
        # Expected first 4 samples: head*ramp + tail*(1-ramp)
        ramp = np.linspace(0.0, 1.0, 4, dtype=np.float32)
        expected_head = (20000.0 * ramp + 10000.0 * (1.0 - ramp)).astype(np.int16)
        np.testing.assert_array_equal(blended[:4], expected_head)
        # Sample 0 should be ~10000 (full tail), sample 3 should be 20000 (full head)
        assert blended[0] == 10000
        assert blended[3] == 20000
        # Tail of chunk 2 (samples 4..) should be unchanged 20000.
        assert (blended[4:] == 20000).all()

    def test_crossfade_disabled_passes_through_unchanged(self):
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=0, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(100, value=10000)
        c2 = _make_chunk(100, value=20000)
        buf.push(c1)
        out = buf.push(c2)
        assert out[0] == c2

    def test_short_chunk_blends_only_available_samples(self):
        # K=64 but chunk is 8 samples. Should blend 8 samples, not raise.
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=64, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(64, value=10000)
        c2 = _make_chunk(8, value=20000)
        buf.push(c1)
        out = buf.push(c2)
        blended = _samples(out[0])
        assert blended.size == 8
        # First sample is full tail, last sample is full head.
        assert blended[0] == 10000
        assert blended[-1] == 20000

    def test_crossfade_does_not_clip_at_int16_extremes(self):
        # Worst case: tail at +32767, head at -32768. Blend must not overflow.
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=4, sample_rate=24000, channels=1
        )
        c1 = _make_chunk(10, value=32767)
        c2 = _make_chunk(10, value=-32768)
        buf.push(c1)
        out = buf.push(c2)
        blended = _samples(out[0])
        # All values must fit in int16 — explicit bounds check.
        assert blended.min() >= -32768
        assert blended.max() <= 32767

    def test_stereo_crossfade_blends_each_channel_independently(self):
        # Channels=2, K=2 frames → K_samples = 4 (interleaved L,R,L,R).
        buf = StreamingChunkBuffer(
            watermark_ms=0,
            crossfade_samples=2,
            sample_rate=24000,
            channels=2,
        )
        # Left channel = 10000, Right channel = -5000 in chunk 1.
        c1_arr = np.array(
            [10000, -5000, 10000, -5000, 10000, -5000], dtype=np.int16
        )
        c2_arr = np.array(
            [20000, 5000, 20000, 5000, 20000, 5000], dtype=np.int16
        )
        buf.push(c1_arr.tobytes())
        out = buf.push(c2_arr.tobytes())
        blended = _samples(out[0])
        # Frame 0 ramp=0 → tail unchanged; frame 1 ramp=1 → head unchanged.
        assert blended[0] == 10000   # L tail
        assert blended[1] == -5000   # R tail
        assert blended[2] == 20000   # L head
        assert blended[3] == 5000    # R head


# --------------------------------------------------------------------------- #
# Flush + reset behavior
# --------------------------------------------------------------------------- #


class TestFlushAndReset:
    def test_flush_remaining_drains_unfilled_watermark(self):
        buf = StreamingChunkBuffer(watermark_ms=500, crossfade_samples=0)
        c1 = _make_chunk(100, value=999)
        c2 = _make_chunk(100, value=888)
        buf.push(c1)
        buf.push(c2)
        out = buf.flush_remaining()
        assert len(out) == 1
        assert out[0] == c1 + c2

    def test_flush_remaining_after_watermark_filled_is_noop(self):
        buf = StreamingChunkBuffer(watermark_ms=0, crossfade_samples=0)
        buf.push(_make_chunk(100))  # watermark already crossed
        assert buf.flush_remaining() == []

    def test_flush_remaining_on_empty_buffer_is_noop(self):
        buf = StreamingChunkBuffer(watermark_ms=500, crossfade_samples=0)
        assert buf.flush_remaining() == []

    def test_reset_clears_watermark_and_tail(self):
        buf = StreamingChunkBuffer(
            watermark_ms=0, crossfade_samples=4, sample_rate=24000, channels=1
        )
        buf.push(_make_chunk(100, value=10000))
        assert buf.is_watermark_filled
        buf.reset()
        assert not buf.is_watermark_filled
        # Next push should be the "first chunk" again — no crossfade applied.
        c2 = _make_chunk(100, value=20000)
        out = buf.push(c2)
        assert out[0] == c2  # unmodified, no tail to blend with
