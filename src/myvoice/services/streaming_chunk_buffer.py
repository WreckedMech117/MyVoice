"""Pre-play watermark + chunk-boundary crossfade for streaming TTS playback.

Sits between QwenTTSService chunk producers and the per-service PyAudio
output streams (MonitorAudioService + VirtualMicrophoneService) inside
AudioCoordinator. Two operating modes:

1. **Static watermark mode** (default; backward-compatible): chunks during
   the early stream phase are accumulated until ``watermark_ms`` of audio
   is buffered, then flushed in one write. Originally added 2026-05-12
   for RTX 3060 + 0.6B small tier producer jitter. Crossfade always
   applies on top: last K samples of chunk N blend with first K of
   chunk N+1 to mask DC discontinuities at chunk boundaries.

2. **Adaptive pre-buffer mode** (opt-in, 2026-05-15; release policy
   revised by Story 20.4, 2026-09-01): for slow-producer hardware where
   TRUE_STREAM playback fundamentally cannot sustain 1.0× realtime
   (e.g. RTX 3060 12GB, observed producer ratio ~0.5×), the static
   watermark is insufficient because inter-chunk gaps add up faster than
   the cushion drains. The gapless cushion is

       τ_gapless = T_a × (1/P − 1)

   where T_a is the estimated total audio duration and P is the observed
   producer rate (audio_seconds / wall_clock_seconds). It is the minimum
   pre-start delay that lets playback finish exactly when (or after)
   generation finishes. The math holds because the minimum buffer level
   during playback is at t_gen_complete (producer stops feeding); solving
   for buffer ≥ 0 at that point yields the formula. Verified against 3060
   smoke data 2026-05-15.

   **Story 20.4 — why the policy is two-regime.** Shipping 2026-05-15 to
   2026-09-01, the buffer chased τ_gapless unconditionally and merely
   *clamped* it to ``max_pre_delay_seconds`` (10 s). Story 20.1 §2.7
   simulated the shipped class and found that for every P ≤ ~0.78 the
   clamp made the τ_min comparison unreachable: release actually happened
   via the elapsed/held escapes, at the first chunk arrival at or after
   10 s (→ ~12.5 s at P = 0.5 with chunk_size 25). That is the worst of
   both worlds — the user waits ~12.5 s **and still gets gaps**, because
   the clamped cushion was far below what gaplessness required (19−28 s).

   So the cushion is now decided against a *feasibility budget*:

     * ``τ_gapless ≤ cushion_budget_seconds``  -> **feasible**: wait for
       the full gapless cushion. Bounded by the budget, so bounded latency.
     * ``τ_gapless  > cushion_budget_seconds``  -> **unreachable**: waiting
       longer cannot buy gaplessness, only latency. Fall back to exactly
       the static watermark this class already applies on ≥16 GiB hosts,
       and start.

   The product trade is explicit and is the one MyVoice wants: Clear Comms
   is a voice-chat interjection feature (``memory/clear_comms_purpose_framing``),
   so **starting sooner with a possible gap beats starting late with one
   anyway**. Total silence is conserved either way — a cushion second not
   spent up front reappears as a gap second later — so the choice is only
   about *where* the silence lands, and the front of an interjection is the
   worst place for it.

   ``max_pre_delay_seconds`` and ``max_hold_chunks`` are UNCHANGED and stay
   as guardrails against pathological cases (cold compile, a stuck rate
   sensor, CPU-only). They simply stop being the binding escape.

State is per-session. Caller is responsible for thread-safety;
AudioCoordinator's existing per-call locking around play_audio_chunk
is sufficient.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np


class StreamingChunkBuffer:
    """Per-session pre-play watermark + chunk-boundary crossfade buffer.

    int16 / mono / configurable sample-rate. Float32 (paFloat32) is not
    supported because app.py:2671 explicitly clips and casts every chunk
    to int16 before dispatch — the buffer mirrors that constraint.
    """

    # Release-regime labels, exposed via ``last_release_reason`` for tests,
    # simulation and evidence. Not telemetry: nothing in production reads
    # them, and no metric is emitted from this class.
    REGIME_PRODUCER_KEEPS_UP = "producer_keeps_up"
    REGIME_GAPLESS_FEASIBLE = "gapless_feasible"
    REGIME_GAPLESS_UNREACHABLE = "gapless_unreachable"

    def __init__(
        self,
        watermark_ms: int = 500,
        crossfade_samples: int = 64,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,
        target_audio_seconds: Optional[float] = None,
        enable_adaptive_pre_buffer: bool = False,
        max_pre_delay_seconds: float = 10.0,
        max_hold_chunks: int = 16,
        cushion_budget_seconds: float = 2.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if watermark_ms < 0:
            raise ValueError("watermark_ms must be >= 0")
        if crossfade_samples < 0:
            raise ValueError("crossfade_samples must be >= 0")
        if sample_width != 2:
            raise ValueError(
                f"sample_width=2 (int16) only supported, got {sample_width}"
            )
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if max_pre_delay_seconds < 0:
            raise ValueError("max_pre_delay_seconds must be >= 0")
        if max_hold_chunks < 1:
            raise ValueError("max_hold_chunks must be >= 1")
        if cushion_budget_seconds < 0:
            raise ValueError("cushion_budget_seconds must be >= 0")
        if enable_adaptive_pre_buffer and target_audio_seconds is None:
            raise ValueError(
                "enable_adaptive_pre_buffer=True requires target_audio_seconds"
            )
        if target_audio_seconds is not None and target_audio_seconds < 0:
            raise ValueError("target_audio_seconds must be >= 0")

        self._watermark_ms = watermark_ms
        self._crossfade_samples = crossfade_samples
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._target_audio_seconds = target_audio_seconds
        self._enable_adaptive_pre_buffer = enable_adaptive_pre_buffer
        self._max_pre_delay_seconds = max_pre_delay_seconds
        self._max_hold_chunks = max_hold_chunks
        self._cushion_budget_seconds = cushion_budget_seconds
        self._clock = clock or time.monotonic

        bytes_per_frame = channels * sample_width
        self._bytes_per_frame = bytes_per_frame
        self._bytes_per_second = sample_rate * bytes_per_frame
        self._watermark_bytes = int(
            watermark_ms / 1000.0 * sample_rate * bytes_per_frame
        )

        self._watermark_queue: deque = deque()
        self._watermark_buffered_bytes: int = 0
        self._watermark_filled: bool = False
        self._prev_tail: np.ndarray = np.empty(0, dtype=np.int16)

        # Adaptive-mode state. ``_t_first_chunk`` is None until the first
        # non-empty push; subsequent pushes use it as the time origin for
        # producer-rate observation. ``_min_observed_producer_rate`` tracks
        # the worst (lowest) P seen so far so a mid-generation producer
        # slowdown (e.g. GPU contention from a game starting up partway
        # through gen) grows the cushion before PyAudio underruns — never
        # shrinks it back even if a later sample reads faster.
        self._t_first_chunk: Optional[float] = None
        self._first_chunk_bytes: int = 0
        self._chunks_held: int = 0
        self._min_observed_producer_rate: Optional[float] = None
        self._last_release_reason: Optional[str] = None

    @property
    def watermark_ms(self) -> int:
        return self._watermark_ms

    @property
    def crossfade_samples(self) -> int:
        return self._crossfade_samples

    @property
    def is_watermark_filled(self) -> bool:
        return self._watermark_filled

    @property
    def is_adaptive(self) -> bool:
        return self._enable_adaptive_pre_buffer

    def _audio_seconds_from_bytes(self, n_bytes: int) -> float:
        return n_bytes / self._bytes_per_second

    def _observed_producer_rate(self) -> Optional[float]:
        """Steady-state producer rate (audio_seconds / wall_clock_seconds).

        Returns None when not enough data has been collected yet (need at
        least one chunk after the first to measure the inter-chunk producer
        rate). The first chunk is excluded from the rate calculation
        because it bundles the model's startup/prefill latency, which
        is a one-time cost rather than a steady-state property.
        """
        if self._t_first_chunk is None:
            return None
        elapsed = self._clock() - self._t_first_chunk
        if elapsed <= 0:
            return None
        post_first_bytes = self._watermark_buffered_bytes - self._first_chunk_bytes
        if post_first_bytes <= 0:
            return None
        return self._audio_seconds_from_bytes(post_first_bytes) / elapsed

    def _worst_observed_producer_rate(self) -> Optional[float]:
        """Return the lowest P observed so far in this session.

        The cushion math is monotone in P (lower P → larger cushion). If
        the producer slows down mid-generation (gaming load spike, GPU
        contention, thermal throttle), a single point-in-time observation
        from chunks 1+2 under-estimates the cushion required for chunks
        3+. Tracking the WORST rate seen so far makes the cushion grow as
        evidence of slowdown accumulates — never shrinks it back. The
        consequence: a transient slow chunk anchors the cushion at the
        slow rate for the rest of the generation. We accept that mild
        over-buffering as the cost of robustness — the alternative
        (using the latest observation) is what surfaced the residual
        gaps on 3060 smoke 2026-05-16.
        """
        current = self._observed_producer_rate()
        if current is None:
            return self._min_observed_producer_rate
        if (
            self._min_observed_producer_rate is None
            or current < self._min_observed_producer_rate
        ):
            self._min_observed_producer_rate = current
        return self._min_observed_producer_rate

    @property
    def cushion_budget_seconds(self) -> float:
        return self._cushion_budget_seconds

    @property
    def last_release_reason(self) -> Optional[str]:
        """Why the most recent dispatch decision came out the way it did.

        One of the ``REGIME_*`` constants, or ``"is_final"`` /
        ``"max_hold_chunks"`` / ``"max_pre_delay"`` for the three guardrail
        escapes, or ``None`` before any adaptive decision has been taken.
        """
        return self._last_release_reason

    def _cushion_decision(self, p_observed: float):
        """Return ``(cushion_seconds, regime)`` for an observed producer rate.

        Story 20.4 AC #2. The pre-20.4 implementation was
        ``clamp(T_a × (1/P − 1), 0, max_pre_delay_seconds)`` -- it chased
        gaplessness at any price up to the 10 s guardrail, and Story 20.1
        §2.7 showed that on the ship-target slow tier (P ≤ ~0.78) the clamp
        put the required cushion *above* anything the buffer would ever
        accumulate, so this comparison never actually bound. Release fell
        through to the elapsed/held escapes at ~12.5 s, and the generation
        gapped anyway.

        Three regimes:

          * ``P ≥ 1.0`` -- the producer keeps up. No cushion. (Unchanged.)
          * ``τ_gapless ≤ cushion_budget_seconds`` -- gaplessness is cheap
            enough to buy. Wait for the full cushion; latency is bounded by
            the budget. (Same behaviour as pre-20.4 in this band.)
          * ``τ_gapless > cushion_budget_seconds`` -- gaplessness is out of
            reach inside the budget. Waiting longer buys latency, not
            gaplessness, so fall back to the static watermark and start.

        ``max_pre_delay_seconds`` still clamps the feasible branch: it is a
        guardrail, not a policy knob, and Story 20.4 does not move it.
        """
        if p_observed >= 1.0:
            return 0.0, self.REGIME_PRODUCER_KEEPS_UP

        # Static-watermark equivalent, in seconds. In the unreachable regime
        # the adaptive branch deliberately reduces to the ≥16 GiB tier's own
        # smoothing behaviour rather than inventing a third number.
        watermark_seconds = self._watermark_bytes / self._bytes_per_second

        if p_observed <= 0.0:
            # Defensive: ``_observed_producer_rate`` never returns <= 0 (it
            # returns None instead), so this is unreachable in practice. A
            # zero/negative rate means "we cannot measure", which is exactly
            # the case where waiting cannot be justified.
            return watermark_seconds, self.REGIME_GAPLESS_UNREACHABLE

        tau_gapless = self._target_audio_seconds * (1.0 / p_observed - 1.0)
        if tau_gapless <= self._cushion_budget_seconds:
            return (
                max(0.0, min(tau_gapless, self._max_pre_delay_seconds)),
                self.REGIME_GAPLESS_FEASIBLE,
            )
        return watermark_seconds, self.REGIME_GAPLESS_UNREACHABLE

    def _required_cushion_seconds(self, p_observed: float) -> float:
        """Effective cushion in seconds for ``p_observed``.

        See ``_cushion_decision`` for the policy and its justification.
        """
        return self._cushion_decision(p_observed)[0]

    def reset(self) -> None:
        """Clear all per-session state — safe to call between sessions."""
        self._watermark_queue.clear()
        self._watermark_buffered_bytes = 0
        self._watermark_filled = False
        self._prev_tail = np.empty(0, dtype=np.int16)
        self._t_first_chunk = None
        self._first_chunk_bytes = 0
        self._chunks_held = 0
        self._min_observed_producer_rate = None
        self._last_release_reason = None

    def push(self, chunk: bytes, is_final: bool = False) -> List[bytes]:
        """Push a chunk through the buffer.

        Returns the list of chunks ready to dispatch downstream. May be
        empty while holding for the pre-buffer threshold, may contain one
        entry on normal pass-through, or one merged entry on the threshold-
        cross flush. Empty input + ``is_final=False`` returns ``[]``.

        In adaptive mode, the threshold is τ_min × producer_rate of audio
        rather than a fixed watermark — recomputed on every push as more
        timing data accumulates.
        """
        if not chunk and not is_final:
            return []

        if not self._watermark_filled:
            if chunk:
                if self._t_first_chunk is None:
                    self._t_first_chunk = self._clock()
                    self._first_chunk_bytes = len(chunk)
                self._watermark_queue.append(chunk)
                self._watermark_buffered_bytes += len(chunk)
                self._chunks_held += 1

            if self._enable_adaptive_pre_buffer:
                ready_to_dispatch = self._adaptive_ready_to_dispatch(is_final)
            else:
                ready_to_dispatch = (
                    self._watermark_buffered_bytes >= self._watermark_bytes
                    or is_final
                )

            if not ready_to_dispatch:
                return []

            flushed = b"".join(self._watermark_queue)
            self._watermark_queue.clear()
            self._watermark_buffered_bytes = 0
            self._watermark_filled = True
            ready = self._apply_crossfade_and_update_tail(flushed)
            return [ready] if ready else []

        ready = self._apply_crossfade_and_update_tail(chunk)
        return [ready] if ready else []

    def _adaptive_ready_to_dispatch(self, is_final: bool) -> bool:
        """Return True when adaptive mode says it's safe to start playback.

        Decision priority (any True -> dispatch):
          1. ``is_final`` -- the stream is ending; ship whatever we have so
             the user hears the audio rather than losing it.
          2. ``_chunks_held >= max_hold_chunks`` -- GUARDRAIL against
             pathological cases (e.g. producer rate sensor reads zero
             forever). Unchanged by Story 20.4.
          3. ``elapsed >= max_pre_delay_seconds`` -- GUARDRAIL, hard time
             cap. Unchanged by Story 20.4: it is a safety bound against
             unbounded waits (cold compile, CPU-only), not a policy knob.
             Under the Story 20.4 policy it should never be the binding
             escape on real hardware; if it fires, something pathological
             happened.
          4. ``_cushion_decision`` -- the policy. ``P ≥ 1.0`` releases
             immediately; a feasible gapless cushion is waited out; an
             unreachable one falls back to the static watermark.

        Returns False when none of the above hold -- caller continues to
        accumulate chunks.
        """
        if is_final:
            self._last_release_reason = "is_final"
            return True
        if self._chunks_held >= self._max_hold_chunks:
            self._last_release_reason = "max_hold_chunks"
            return True

        elapsed = (
            self._clock() - self._t_first_chunk
            if self._t_first_chunk is not None
            else 0.0
        )
        if elapsed >= self._max_pre_delay_seconds:
            self._last_release_reason = "max_pre_delay"
            return True

        p_observed = self._worst_observed_producer_rate()
        if p_observed is None:
            return False

        cushion_required, regime = self._cushion_decision(p_observed)
        if regime == self.REGIME_PRODUCER_KEEPS_UP:
            self._last_release_reason = regime
            return True

        # The unreachable regime's ``cushion_required`` is exactly the static
        # watermark, so compare in BYTES there -- byte-identical to the
        # ``enable_adaptive_pre_buffer=False`` predicate in ``push``, with no
        # float round-trip that could make the two branches disagree at the
        # boundary.
        if regime == self.REGIME_GAPLESS_UNREACHABLE:
            ready = self._watermark_buffered_bytes >= self._watermark_bytes
        else:
            audio_buffered_seconds = self._audio_seconds_from_bytes(
                self._watermark_buffered_bytes
            )
            ready = audio_buffered_seconds >= cushion_required
        if ready:
            self._last_release_reason = regime
        return ready

    def flush_remaining(self) -> List[bytes]:
        """Drain any audio still held in the watermark queue.

        Used on stop_streaming_session paths so audio is not lost when
        the session ends before the watermark threshold is reached
        (e.g. a very short utterance or an early abort).
        """
        if not self._watermark_queue:
            return []
        flushed = b"".join(self._watermark_queue)
        self._watermark_queue.clear()
        self._watermark_buffered_bytes = 0
        self._watermark_filled = True
        ready = self._apply_crossfade_and_update_tail(flushed)
        return [ready] if ready else []

    def _apply_crossfade_and_update_tail(self, chunk: bytes) -> bytes:
        """Blend chunk's leading K samples with previous chunk's tail; stash new tail."""
        if not chunk:
            return chunk
        if self._crossfade_samples <= 0:
            return chunk

        samples = np.frombuffer(chunk, dtype=np.int16)
        n_frames = samples.size // self._channels
        k_frames = min(self._crossfade_samples, n_frames)
        k_samples = k_frames * self._channels

        if self._prev_tail.size >= k_samples and k_samples > 0:
            ramp = np.linspace(0.0, 1.0, k_frames, dtype=np.float32)
            if self._channels > 1:
                ramp = np.repeat(ramp, self._channels)
            head = samples[:k_samples].astype(np.float32)
            tail = self._prev_tail[-k_samples:].astype(np.float32)
            blended = head * ramp + tail * (1.0 - ramp)
            blended_clipped = np.clip(blended, -32768.0, 32767.0).astype(np.int16)
            samples = samples.copy()
            samples[:k_samples] = blended_clipped

        tail_samples = min(self._crossfade_samples * self._channels, samples.size)
        if tail_samples > 0:
            self._prev_tail = samples[-tail_samples:].copy()
        else:
            self._prev_tail = np.empty(0, dtype=np.int16)

        return samples.tobytes()
