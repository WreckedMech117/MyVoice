"""Pre-play watermark + chunk-boundary crossfade for streaming TTS playback.

Sits between QwenTTSService chunk producers and the per-service PyAudio
output streams (MonitorAudioService + VirtualMicrophoneService) inside
AudioCoordinator. Solves two consumer-side glitches that surface on
producer-near-realtime hardware (RTX 3060 + 0.6B small tier observed
2026-05-12):

1. Watermark — chunks during the early stream phase are accumulated in
   an internal queue until ``watermark_ms`` of audio is buffered, then
   flushed in one write. Gives the PyAudio output buffer headroom
   against producer jitter so brief producer pauses do not underrun
   the device callback. Story 18.3 fixed end-of-stream drain; this is
   the symmetric start-of-stream fix.

2. Crossfade — last K samples of dispatched chunk N are linearly
   blended with first K samples of chunk N+1. Masks DC discontinuities
   at chunk boundaries that present as audible clicks on raw concat.
   Default K=64 samples ≈ 2.7 ms at 24 kHz.

State is per-session. Caller is responsible for thread-safety;
AudioCoordinator's existing per-call locking around play_audio_chunk
is sufficient.
"""

from __future__ import annotations

from collections import deque
from typing import List

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

        self._watermark_ms = watermark_ms
        self._crossfade_samples = crossfade_samples
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width

        bytes_per_frame = channels * sample_width
        self._watermark_bytes = int(
            watermark_ms / 1000.0 * sample_rate * bytes_per_frame
        )

        self._watermark_queue: deque = deque()
        self._watermark_buffered_bytes: int = 0
        self._watermark_filled: bool = False
        self._prev_tail: np.ndarray = np.empty(0, dtype=np.int16)

    @property
    def watermark_ms(self) -> int:
        return self._watermark_ms

    @property
    def crossfade_samples(self) -> int:
        return self._crossfade_samples

    @property
    def is_watermark_filled(self) -> bool:
        return self._watermark_filled

    def reset(self) -> None:
        """Clear all per-session state — safe to call between sessions."""
        self._watermark_queue.clear()
        self._watermark_buffered_bytes = 0
        self._watermark_filled = False
        self._prev_tail = np.empty(0, dtype=np.int16)

    def push(self, chunk: bytes, is_final: bool = False) -> List[bytes]:
        """Push a chunk through the buffer.

        Returns the list of chunks ready to dispatch downstream. May be
        empty during initial watermark fill, may contain one entry on
        normal pass-through, or one merged entry on the watermark-cross
        flush. Empty input + ``is_final=False`` returns ``[]``.
        """
        if not chunk and not is_final:
            return []

        if not self._watermark_filled:
            if chunk:
                self._watermark_queue.append(chunk)
                self._watermark_buffered_bytes += len(chunk)
            crossed = self._watermark_buffered_bytes >= self._watermark_bytes
            if not (crossed or is_final):
                return []
            flushed = b"".join(self._watermark_queue)
            self._watermark_queue.clear()
            self._watermark_buffered_bytes = 0
            self._watermark_filled = True
            ready = self._apply_crossfade_and_update_tail(flushed)
            return [ready] if ready else []

        ready = self._apply_crossfade_and_update_tail(chunk)
        return [ready] if ready else []

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
