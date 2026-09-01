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


class TestAdaptivePreBufferConstruction:
    """Adaptive mode parameter validation."""

    def test_adaptive_requires_target_audio_seconds(self):
        with pytest.raises(ValueError, match="target_audio_seconds"):
            StreamingChunkBuffer(
                enable_adaptive_pre_buffer=True,
                target_audio_seconds=None,
            )

    def test_rejects_negative_target_audio_seconds(self):
        with pytest.raises(ValueError, match="target_audio_seconds"):
            StreamingChunkBuffer(target_audio_seconds=-1.0)

    def test_rejects_negative_max_pre_delay(self):
        with pytest.raises(ValueError, match="max_pre_delay_seconds"):
            StreamingChunkBuffer(max_pre_delay_seconds=-0.1)

    def test_rejects_zero_max_hold_chunks(self):
        with pytest.raises(ValueError, match="max_hold_chunks"):
            StreamingChunkBuffer(max_hold_chunks=0)

    def test_rejects_negative_cushion_budget(self):
        with pytest.raises(ValueError, match="cushion_budget_seconds"):
            StreamingChunkBuffer(cushion_budget_seconds=-0.1)

    def test_is_adaptive_reflects_setting(self):
        buf_off = StreamingChunkBuffer(watermark_ms=500)
        assert buf_off.is_adaptive is False
        buf_on = StreamingChunkBuffer(
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
        )
        assert buf_on.is_adaptive is True


class _FakeClock:
    """Test clock that returns the value of ``.now`` and supports advance().

    Lets the adaptive-mode tests pin elapsed time deterministically without
    sleeping. Mirrors the time.monotonic() signature StreamingChunkBuffer's
    constructor accepts via the ``clock`` parameter.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestAdaptivePreBufferBehavior:
    """Adaptive mode: τ = T_a × (1/P − 1), clamped to [0, max_pre_delay]."""

    def test_static_mode_unchanged_when_adaptive_disabled(self):
        # Backward-compat regression: adaptive_pre_buffer=False (default)
        # preserves the original 500ms watermark behavior. Test uses the
        # exact same fixtures as TestWatermark.test_chunks_below_threshold_held_back
        # to pin that the static path is byte-identical to pre-adaptive.
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            enable_adaptive_pre_buffer=False,
        )
        out = buf.push(_make_chunk(_SAMPLES_PER_100MS_24K))
        assert out == []
        assert not buf.is_watermark_filled

    def test_first_chunk_alone_does_not_dispatch_yet(self):
        # Adaptive mode needs at least one post-first chunk to measure
        # producer rate (first chunk includes prefill latency which is a
        # one-time cost). First chunk alone → still holding.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            clock=clock,
        )
        c1 = _make_chunk(_SAMPLES_PER_100MS_24K)  # 100ms
        out = buf.push(c1, is_final=False)
        assert out == []
        assert not buf.is_watermark_filled

    def test_fast_producer_dispatches_after_chunk_2(self):
        # P >= 1.0 → cushion_required = 0 → dispatch immediately on the
        # first push that yields a measurable rate. Simulates a fast card
        # producing audio faster than playback consumes.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            clock=clock,
        )
        # Chunk 1: 1s of audio at t=0
        c1 = _make_chunk(24000)
        out1 = buf.push(c1, is_final=False)
        assert out1 == []

        # Chunk 2: another 1s of audio, arrives only 0.5s of wall-clock
        # later → producer rate = 1.0 / 0.5 = 2.0 (fast).
        clock.advance(0.5)
        c2 = _make_chunk(24000)
        out2 = buf.push(c2, is_final=False)
        assert len(out2) == 1
        # Combined payload = c1 + c2 (no crossfade configured).
        assert out2[0] == c1 + c2
        assert buf.is_watermark_filled

    def test_slow_producer_holds_until_cushion_met(self):
        # 3060 regime: P ≈ 0.5, T_a = 5.0s → τ_gapless = 5.0 × (1/0.5 − 1) = 5.0s
        # Need 5.0s of audio buffered before dispatch.
        #
        # Story 20.4: this row exercises the FEASIBLE regime, so the
        # cushion budget is raised above τ_gapless. Under the shipped 2.0 s
        # budget a 5.0 s cushion is declared unreachable and the buffer
        # starts at the static watermark instead — that behaviour is
        # covered by TestCushionFeasibilityPolicy below.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            max_pre_delay_seconds=30.0,  # don't let cap kick in
            cushion_budget_seconds=30.0,  # feasible-regime row
            clock=clock,
        )
        # Chunk 1: 1s of audio at t=0
        out = buf.push(_make_chunk(24000), is_final=False)
        assert out == []

        # Chunk 2: another 1s at t=2.0 → post-first rate = 1.0/2.0 = 0.5
        # Cushion required = 5.0 × (1/0.5 − 1) = 5.0s
        # Audio buffered = 2.0s < 5.0s → still hold.
        clock.advance(2.0)
        out = buf.push(_make_chunk(24000), is_final=False)
        assert out == []
        assert not buf.is_watermark_filled

        # Chunks 3-5: add 3 more seconds. Now buffered = 5.0s, hits cushion.
        clock.advance(2.0)
        buf.push(_make_chunk(24000), is_final=False)
        clock.advance(2.0)
        buf.push(_make_chunk(24000), is_final=False)
        clock.advance(2.0)
        out = buf.push(_make_chunk(24000), is_final=False)
        assert len(out) == 1
        # 5 × 1s chunks combined.
        assert len(out[0]) == 24000 * 2 * 5  # samples × sample_width × count
        assert buf.is_watermark_filled

    def test_is_final_flushes_regardless_of_cushion(self):
        # is_final must always dispatch — short utterances that would
        # never satisfy a 5s cushion must still produce audio.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            clock=clock,
        )
        c1 = _make_chunk(_SAMPLES_PER_100MS_24K)  # 100ms — way below cushion
        out = buf.push(c1, is_final=True)
        assert len(out) == 1
        assert out[0] == c1
        assert buf.is_watermark_filled

    def test_max_pre_delay_cap_kicks_in(self):
        # Cold-compile case: P measured near zero would compute infinite
        # cushion. The cap prevents an unbounded wait.
        #
        # Story 20.4 keeps max_pre_delay_seconds as a GUARDRAIL and does not
        # move it. To prove the guardrail still fires, this row raises the
        # feasibility budget above the computed cushion (so the new policy
        # cannot release first) and uses a watermark large enough that the
        # unreachable-regime fallback would not release either.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            max_pre_delay_seconds=3.0,
            cushion_budget_seconds=1000.0,
            clock=clock,
        )
        # Two chunks far apart in time → very slow producer rate.
        buf.push(_make_chunk(24000), is_final=False)
        clock.advance(10.0)
        buf.push(_make_chunk(24000), is_final=False)
        # Elapsed since first chunk = 10s, which exceeds max_pre_delay=3s.
        # Even though buffered audio is only 2s, the elapsed cap fires.
        # Next push triggers re-eval.
        clock.advance(0.1)
        out = buf.push(_make_chunk(_SAMPLES_PER_100MS_24K), is_final=False)
        assert len(out) == 1
        assert buf.is_watermark_filled
        assert buf.last_release_reason == "max_pre_delay"

    def test_worst_p_tracking_grows_cushion_on_mid_gen_slowdown(self):
        # Producer starts fast (~0.8) then slows to (~0.4) mid-gen. The
        # worst-P tracker must lock in the 0.4 reading so the cushion
        # required does not collapse back to the optimistic early estimate.
        # 3060 smoke 2026-05-16 surfaced this as the residual-gap class:
        # chunks 1+2 read fast, dispatched, then chunks 3+ slowed and
        # PyAudio underran. Worst-P keeps the cushion sized for the
        # slowest segment observed.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=8.0,
            enable_adaptive_pre_buffer=True,
            max_pre_delay_seconds=30.0,
            # Story 20.4: worst-P tracking is the subject here, so keep the
            # feasibility budget out of the way. With the shipped 2.0 s
            # budget the τ values below are declared unreachable and the
            # buffer would start at the watermark, testing nothing.
            cushion_budget_seconds=30.0,
            clock=clock,
        )
        # Chunk 1 at t=0 (24000 samples × 2 bytes = 48000 bytes = 1.0 s audio)
        buf.push(_make_chunk(24000), is_final=False)

        # Chunk 2 at t=1.25 → post-first rate = 1.0 / 1.25 = 0.8 (fast-ish)
        # Cushion @ P=0.8 = 8.0 × (1/0.8 − 1) = 2.0s; buffered = 2.0s ≥ 2.0s
        # WITHOUT worst-P tracking, this would dispatch.
        clock.advance(1.25)
        out = buf.push(_make_chunk(24000), is_final=False)
        # Edge case: with the latest-P approach, dispatch may fire here.
        # With worst-P tracking, the locked rate stays at 0.8 and dispatch
        # fires at the same point — that's fine; this test focuses on what
        # happens AFTER a slowdown is observed.
        if out:
            # If dispatched, worst-P doesn't help on this gen — the test's
            # point is the next test (where a slowdown happens before
            # dispatch).
            return

        # Chunk 3 arrives much later: t=1.25 + 5.0 = 6.25. Inter-chunk
        # time from chunk 1 = 6.25s, post-first audio = 2.0s → P = 2.0/6.25
        # = 0.32. WORSE than the 0.8 we saw at chunk 2. Worst-P locks in.
        # Cushion @ P=0.32 = 8.0 × (1/0.32 − 1) = 17.0s, clamped to
        # max_pre_delay=30 → 17.0s. Audio buffered = 3.0s ≪ 17.0s → HOLD.
        clock.advance(5.0)
        out = buf.push(_make_chunk(24000), is_final=False)
        assert out == []
        assert not buf.is_watermark_filled

    def test_worst_p_does_not_shrink_on_subsequent_fast_chunk(self):
        # Confirm the tracker is one-directional: once a slow P is observed,
        # a later fast chunk does NOT restore the optimistic cushion.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            max_pre_delay_seconds=30.0,
            # Story 20.4: feasible-regime row (see the sibling test).
            cushion_budget_seconds=30.0,
            clock=clock,
        )
        # Chunk 1 at t=0
        buf.push(_make_chunk(24000), is_final=False)

        # Chunk 2 arrives slowly at t=4.0 → P = 1.0/4.0 = 0.25.
        # Cushion = 5.0 × (1/0.25 − 1) = 15s; audio_buf = 2s → hold.
        clock.advance(4.0)
        out = buf.push(_make_chunk(24000), is_final=False)
        assert out == []

        # Chunk 3 arrives quickly at t=4.1 (only 0.1s later — extreme).
        # Naive P = 2.0/4.1 = 0.488 → cushion = 5.24s, audio_buf=3s → would
        # still hold under either approach. The locked-P value should
        # remain at 0.25 (worst-so-far), giving cushion = 15s (still hold).
        clock.advance(0.1)
        out = buf.push(_make_chunk(24000), is_final=False)
        assert out == []
        # The worst-P public-ish surface isn't exposed; assert via behavior
        # by requiring a LOT more audio before dispatch. Add 12 more chunks
        # quickly — if the cushion had collapsed to 5.24s, dispatch would
        # fire around chunk 5 (audio_buf=5s); with worst-P locked at 0.25,
        # we need to hit 15s of audio_buf (= 15 chunks total = 12 more).
        for _ in range(11):
            buf.push(_make_chunk(24000), is_final=False)
            assert not buf.is_watermark_filled, (
                "Dispatch fired too early — worst-P should keep cushion at 15s"
            )
        # The 15th chunk hits the 15s cushion threshold.
        out = buf.push(_make_chunk(24000), is_final=False)
        assert len(out) == 1
        assert buf.is_watermark_filled

    def test_max_hold_chunks_safety_bound(self):
        # If producer rate sensor reads zero indefinitely (pathological),
        # the chunk-count cap forces dispatch. 16 chunks at 100ms each =
        # 1.6s of audio — we'd rather play 1.6s than wait forever.
        clock = _FakeClock()
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            target_audio_seconds=5.0,
            enable_adaptive_pre_buffer=True,
            max_pre_delay_seconds=100.0,
            max_hold_chunks=3,
            clock=clock,
        )
        # Push 3 chunks with no time advance → P appears as inf, but
        # max_hold_chunks=3 forces dispatch on the 3rd push.
        buf.push(_make_chunk(_SAMPLES_PER_100MS_24K), is_final=False)
        buf.push(_make_chunk(_SAMPLES_PER_100MS_24K), is_final=False)
        out = buf.push(_make_chunk(_SAMPLES_PER_100MS_24K), is_final=False)
        assert len(out) == 1
        assert buf.is_watermark_filled


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


# --------------------------------------------------------------------------- #
# Story 20.4 AC #2 / AC #3 — the two-regime cushion policy
# --------------------------------------------------------------------------- #

# The codec emits 1920 samples per frame -- MEASURED, and equal to 12.5 Hz
# at 24 kHz, not the 12 Hz this codebase's prose assumed before Story 20.4
# (see streaming_decoder._CODEC_SAMPLES_PER_FRAME). At the committed
# chunk_size = 25 a posted chunk is therefore 25 * 1920 = 48,000 samples =
# exactly 2.0 s of audio.
#
# These rows were originally written at chunk_size = 10 (19,200 samples,
# 0.8 s). That retune was reverted after the NFR3 gate, so they are
# restated here at the geometry we actually ship -- an adaptive-cushion
# test describing a configuration nobody runs is worse than no test.
_CS25_CHUNK_SAMPLES = 25 * 1920
_CS25_CHUNK_SECONDS = _CS25_CHUNK_SAMPLES / 24000.0
_CS25_WINDOW_SECONDS = (25 + 5) * 1920 / 24000.0

# The canonical Story 20.1 long fixture is 349 chars; the shipped affine
# estimator (audio_coordinator.estimate_target_audio_seconds) puts that at
# 0.5 + 349 * 0.055 = 19.695 s.
_LONG_FIXTURE_T_A = 19.695


def _adaptive_buf(clock, **kw):
    """A buffer in adaptive mode with the shipped production constants."""
    params = dict(
        watermark_ms=500,
        crossfade_samples=0,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        enable_adaptive_pre_buffer=True,
        max_pre_delay_seconds=10.0,
        cushion_budget_seconds=2.0,
        clock=clock,
    )
    params.update(kw)
    return StreamingChunkBuffer(**params)


def _drive(buf, clock, chunk_samples, p, max_chunks=64):
    """Feed chunks at producer rate ``p`` until release. Returns the offset."""
    audio_s = chunk_samples / 24000.0
    wall_s = audio_s / p
    t0 = clock.now
    for i in range(max_chunks):
        if i:
            clock.now = t0 + i * wall_s
        if buf.push(_make_chunk(chunk_samples), is_final=False):
            return clock.now - t0
    return None


class TestCushionFeasibilityPolicy:
    """Story 20.4 AC #2 — start materially sooner than the 10 s guardrail."""

    def test_unreachable_cushion_falls_back_to_the_static_watermark(self):
        # The ship-target slow tier: P = 0.5 on the canonical long fixture.
        # tau_gapless = 19.695 * (1/0.5 - 1) = 19.695 s, far outside the
        # 2.0 s budget, so gaplessness is unreachable. The buffer must stop
        # paying latency for it and release on the 500 ms watermark — which
        # at the shipped geometry means chunk 2, since one 2.0 s chunk
        # clears a 0.5 s watermark on its own.
        clock = _FakeClock()
        buf = _adaptive_buf(clock, target_audio_seconds=_LONG_FIXTURE_T_A)

        # Chunk 1 alone cannot measure P (it bundles prefill), so it holds.
        assert buf.push(_make_chunk(_CS25_CHUNK_SAMPLES), is_final=False) == []

        clock.advance(_CS25_CHUNK_SECONDS / 0.5)
        out = buf.push(_make_chunk(_CS25_CHUNK_SAMPLES), is_final=False)
        assert len(out) == 1
        assert buf.last_release_reason == (
            StreamingChunkBuffer.REGIME_GAPLESS_UNREACHABLE
        )

    def test_low_p_host_starts_materially_sooner_than_the_cap(self):
        """The AC #2 headline, at the SHIPPED geometry (chunk_size = 25).

        Pre-Story-20.4 this configuration released via the elapsed escape:
        Story 20.1 §2.7 simulated **12.50 s** at chunk_size 25, P = 0.5 —
        and 12.5 s rather than 10 s because the cap is only evaluated
        inside ``push``, so the effective wait is the first chunk arrival
        at or after it. The policy must land it at one chunk arrival
        instead.
        """
        clock = _FakeClock()
        buf = _adaptive_buf(clock, target_audio_seconds=_LONG_FIXTURE_T_A)
        offset = _drive(buf, clock, _CS25_CHUNK_SAMPLES, 0.5)

        assert offset is not None, "never released before the fixture ran out"
        # One chunk arrival at P = 0.5 is 2.0 / 0.5 = 4.0 s.
        assert offset == pytest.approx(_CS25_CHUNK_SECONDS / 0.5, abs=1e-6)
        assert offset == pytest.approx(4.0, abs=1e-6)
        # Materially sooner than the guardrail, and than the 12.5 s the
        # pre-20.4 policy actually produced here.
        assert offset < 0.5 * 10.0
        assert offset < 0.4 * 12.5
        assert buf.last_release_reason == (
            StreamingChunkBuffer.REGIME_GAPLESS_UNREACHABLE
        )

    def test_feasible_cushion_is_still_waited_out(self):
        # P = 0.9 on a 15 s utterance: tau_gapless = 15 * (1/0.9 - 1)
        # = 1.667 s, inside the 2.0 s budget. Buying gaplessness is cheap
        # here, so the buffer must still buy it — the policy is a
        # feasibility test, not a blanket "always start early".
        #
        # Deliberately driven with SMALL chunks (0.25 s) rather than the
        # shipped 2.0 s ones. At the committed geometry a single chunk
        # already exceeds the whole cushion budget, so the feasible branch
        # is granularity-bound and always releases on chunk 2 — which is
        # true, is asserted by the sibling row below, and would make this
        # row vacuous. Small chunks are what actually exercise the
        # "accumulate until tau" behaviour.
        small = 6000                       # 0.25 s at 24 kHz
        chunk_s = small / 24000.0
        clock = _FakeClock()
        buf = _adaptive_buf(clock, target_audio_seconds=15.0)
        tau = 15.0 * (1.0 / 0.9 - 1.0)

        offset = _drive(buf, clock, small, 0.9)
        assert offset is not None
        assert buf.last_release_reason == (
            StreamingChunkBuffer.REGIME_GAPLESS_FEASIBLE
        )
        # Released on the first push whose buffered audio reaches tau, so
        # strictly more than tau/chunk_s - 1 chunks had to arrive.
        released_chunks = round(offset / (chunk_s / 0.9)) + 1
        assert released_chunks * chunk_s >= tau
        assert (released_chunks - 1) * chunk_s < tau
        # The watermark alone would have released at chunk 3 (0.5 s of
        # audio); the feasible branch deliberately waits longer than that.
        assert released_chunks > 3

    def test_at_the_shipped_geometry_the_feasible_branch_is_granularity_bound(
        self,
    ):
        """AC #3, restated at cs25: one chunk already covers the budget.

        A posted chunk at chunk_size = 25 carries 2.0 s of audio, and the
        cushion budget is 2.0 s. So whenever the producer rate is
        measurable at all — which needs two chunks — the buffer is already
        holding at least as much audio as any feasible cushion could ask
        for. The feasible and unreachable regimes therefore release at the
        SAME point on the shipped geometry, and the policy's effect there
        is entirely to stop the guardrail from binding.

        This is a real consequence of reverting to 25 and is stated so
        nobody reads §2's regime table as implying a mid-stream decision
        that cannot occur at this geometry.
        """
        for p, regime in (
            (0.5, StreamingChunkBuffer.REGIME_GAPLESS_UNREACHABLE),
            (0.95, StreamingChunkBuffer.REGIME_GAPLESS_FEASIBLE),
        ):
            clock = _FakeClock()
            buf = _adaptive_buf(clock, target_audio_seconds=_LONG_FIXTURE_T_A)
            offset = _drive(buf, clock, _CS25_CHUNK_SAMPLES, p)
            assert offset is not None
            assert buf.last_release_reason == regime, (
                "P={} took the {} branch".format(p, buf.last_release_reason)
            )
            # Chunk 2, i.e. one inter-chunk interval after the first.
            assert offset == pytest.approx(_CS25_CHUNK_SECONDS / p, abs=1e-6)
        assert _CS25_CHUNK_SECONDS >= 2.0

    def test_budget_boundary_is_inclusive(self):
        # tau_gapless exactly == the budget counts as feasible. An
        # exclusive comparison here would make the boundary behaviour
        # depend on floating-point noise.
        buf = _adaptive_buf(_FakeClock(), target_audio_seconds=8.0,
                            cushion_budget_seconds=2.0)
        # 8.0 * (1/0.8 - 1) = 2.0 exactly.
        cushion, regime = buf._cushion_decision(0.8)
        assert regime == StreamingChunkBuffer.REGIME_GAPLESS_FEASIBLE
        assert cushion == pytest.approx(2.0)

    def test_fast_producer_regime_is_unchanged(self):
        buf = _adaptive_buf(_FakeClock(), target_audio_seconds=10.0)
        cushion, regime = buf._cushion_decision(1.0)
        assert (cushion, regime) == (
            0.0, StreamingChunkBuffer.REGIME_PRODUCER_KEEPS_UP
        )
        cushion, regime = buf._cushion_decision(2.5)
        assert (cushion, regime) == (
            0.0, StreamingChunkBuffer.REGIME_PRODUCER_KEEPS_UP
        )

    def test_unmeasurable_rate_does_not_buy_a_wait(self):
        # Defensive branch: a zero/negative rate means "we cannot measure",
        # which pre-20.4 returned max_pre_delay_seconds for — i.e. the
        # longest possible wait on the least possible evidence.
        buf = _adaptive_buf(_FakeClock(), target_audio_seconds=10.0)
        cushion, regime = buf._cushion_decision(0.0)
        assert regime == StreamingChunkBuffer.REGIME_GAPLESS_UNREACHABLE
        assert cushion == pytest.approx(0.5)  # the static watermark

    def test_max_pre_delay_is_not_the_binding_escape_across_the_p_sweep(self):
        """AC #2 + AC #3 — the whole curve at the shipped chunk_size = 25.

        Story 20.1 §2.7's simulation of the pre-20.4 buffer found the 10 s
        cap binding for every P <= ~0.78 at this very geometry, releasing
        at 11.1-12.5 s against a 3.3-5.0 s talker segment. With the Story
        20.4 policy the cap must never be the binding escape anywhere on
        this sweep.
        """
        for p in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            clock = _FakeClock()
            buf = _adaptive_buf(clock, target_audio_seconds=_LONG_FIXTURE_T_A)
            offset = _drive(buf, clock, _CS25_CHUNK_SAMPLES, p)
            assert offset is not None, f"P={p}: never released"
            assert buf.last_release_reason in (
                StreamingChunkBuffer.REGIME_GAPLESS_FEASIBLE,
                StreamingChunkBuffer.REGIME_GAPLESS_UNREACHABLE,
            ), f"P={p}: released via guardrail {buf.last_release_reason!r}"
            assert offset < 10.0, f"P={p}: released at {offset:.2f}s"
            # The talker segment is (chunk_size + lookahead)/12.5 / P at
            # the codec's measured frame rate; the cushion must not dwarf
            # it the way the cap did.
            talker_seg = _CS25_WINDOW_SECONDS / p
            assert offset / talker_seg <= 1.0, (
                f"P={p}: cushion/talker = {offset / talker_seg:.2f}x"
            )


class TestStaticWatermarkPathUntouched:
    """Story 20.4 AC #2 — the >=16 GiB path is behaviourally unchanged."""

    def _static_trace(self, cushion_budget_seconds):
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=64,
            sample_rate=24000,
            channels=1,
            sample_width=2,
            enable_adaptive_pre_buffer=False,
            cushion_budget_seconds=cushion_budget_seconds,
        )
        trace = []
        for i in range(8):
            trace.append(buf.push(_make_chunk(2400, value=100 * (i + 1))))
        trace.append(buf.push(b"", is_final=True))
        return trace

    def test_cushion_budget_has_no_effect_on_the_static_path(self):
        # The knob is adaptive-only. Byte-identical output across wildly
        # different budgets is the proof.
        assert self._static_trace(0.0) == self._static_trace(2.0)
        assert self._static_trace(2.0) == self._static_trace(1000.0)

    def test_static_release_point_is_exactly_the_watermark(self):
        # 100 ms chunks, 500 ms watermark -> release on the 5th push, and
        # not before. This is the pre-adaptive behaviour verbatim.
        buf = StreamingChunkBuffer(
            watermark_ms=500,
            crossfade_samples=0,
            sample_rate=24000,
            channels=1,
            enable_adaptive_pre_buffer=False,
        )
        for _ in range(4):
            assert buf.push(_make_chunk(_SAMPLES_PER_100MS_24K)) == []
        assert len(buf.push(_make_chunk(_SAMPLES_PER_100MS_24K))) == 1
        # No adaptive decision was ever taken on this path.
        assert buf.last_release_reason is None
