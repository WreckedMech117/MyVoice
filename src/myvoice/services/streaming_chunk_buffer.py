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

2. **Adaptive pre-buffer mode** (opt-in, 2026-05-15): for slow-producer
   hardware where TRUE_STREAM playback fundamentally cannot sustain
   1.0× realtime (e.g. RTX 3060 12GB, observed producer ratio ~0.5×),
   the static watermark is insufficient because inter-chunk gaps add
   up faster than the cushion drains. Instead, compute the minimum
   pre-start delay that lets playback finish exactly when (or after)
   generation finishes:

       τ_min = T_a × (1/P − 1)

   where T_a is the estimated total audio duration and P is the observed
   producer rate (audio_seconds / wall_clock_seconds). For P ≥ 1.0 (fast
   hardware), τ_min ≤ 0 → no extra delay → TRUE_STREAM benefit preserved.
   For P < 1.0, τ_min grows to cover the deficit; for P near zero
   (CPU-only), τ_min is clamped to ``max_pre_delay_seconds`` so we never
   wait forever.

   The math holds because the minimum buffer level during playback is at
   t_gen_complete (producer stops feeding). Solving for buffer ≥ 0 at
   that point yields the τ_min formula. Verified against 3060 smoke data
   2026-05-15.

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
        # producer-rate observation.
        self._t_first_chunk: Optional[float] = None
        self._first_chunk_bytes: int = 0
        self._chunks_held: int = 0

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

    def _required_cushion_seconds(self, p_observed: float) -> float:
        """τ_min = T_a × (1/P − 1), clamped to [0, max_pre_delay_seconds].

        Called only when adaptive mode is enabled and target_audio_seconds
        is set (checked in __init__).
        """
        if p_observed >= 1.0:
            return 0.0
        if p_observed <= 0.0:
            return self._max_pre_delay_seconds
        cushion = self._target_audio_seconds * (1.0 / p_observed - 1.0)
        return max(0.0, min(cushion, self._max_pre_delay_seconds))

    def reset(self) -> None:
        """Clear all per-session state — safe to call between sessions."""
        self._watermark_queue.clear()
        self._watermark_buffered_bytes = 0
        self._watermark_filled = False
        self._prev_tail = np.empty(0, dtype=np.int16)
        self._t_first_chunk = None
        self._first_chunk_bytes = 0
        self._chunks_held = 0

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

        Decision priority (any True → dispatch):
          1. ``is_final`` — the stream is ending; ship whatever we have so
             the user hears the audio rather than losing it.
          2. ``_chunks_held >= max_hold_chunks`` — safety bound against
             pathological cases (e.g. producer rate sensor reads zero
             forever). 16 chunks at ~2 s each ≈ 32 s of audio buffered;
             at that point we've held longer than max_pre_delay anyway.
          3. ``elapsed >= max_pre_delay_seconds`` — hard time cap. Past
             this point we play whatever we have, even if math says we
             should wait longer (gen failure or cold-compile would
             otherwise trigger an unbounded wait).
          4. ``observed rate ≥ 1.0`` — producer keeps up with playback;
             no extra delay needed (fast hardware path).
          5. ``audio_buffered_seconds ≥ τ_min`` — math says we have
             enough cushion to bridge the rest of the generation.

        Returns False when none of the above hold — caller continues to
        accumulate chunks.
        """
        if is_final:
            return True
        if self._chunks_held >= self._max_hold_chunks:
            return True

        elapsed = (
            self._clock() - self._t_first_chunk
            if self._t_first_chunk is not None
            else 0.0
        )
        if elapsed >= self._max_pre_delay_seconds:
            return True

        p_observed = self._observed_producer_rate()
        if p_observed is None:
            return False

        if p_observed >= 1.0:
            return True

        cushion_required = self._required_cushion_seconds(p_observed)
        audio_buffered_seconds = self._audio_seconds_from_bytes(
            self._watermark_buffered_bytes
        )
        return audio_buffered_seconds >= cushion_required

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
