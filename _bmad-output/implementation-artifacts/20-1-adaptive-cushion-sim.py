"""Story 20.1 review-response A4/A5 - simulate the REAL StreamingChunkBuffer.

The first draft of Follow-up C argued from the tau_min formula alone and got
the binding constraint wrong. `_adaptive_ready_to_dispatch` has five escapes
in priority order, and the tau_min comparison is the LAST of them:

    1. is_final
    2. _chunks_held >= max_hold_chunks (16)
    3. elapsed >= max_pre_delay_seconds (10.0)
    4. observed P >= 1.0
    5. audio_buffered_seconds >= tau_min

So rather than re-deriving on paper, this drives the production class with an
injected clock and synthetic chunks arriving at a chosen producer rate P, and
reports WHICH escape released playback and at what wall-clock offset from the
first chunk. No GPU, no model - pure arithmetic against the shipped code.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-1-adaptive-cushion-sim.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from myvoice.services.streaming_chunk_buffer import StreamingChunkBuffer  # noqa: E402

SAMPLE_RATE = 24000
BYTES_PER_SEC = SAMPLE_RATE * 2  # int16 mono
FRAME_HZ = 12.0
MAX_PRE_DELAY = 10.0
MAX_HOLD_CHUNKS = 16


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def why_released(buf: StreamingChunkBuffer, elapsed: float, chunks_held: int,
                 buffered_s: float, p_obs, target_audio_s: float) -> str:
    """Reproduce the priority order in _adaptive_ready_to_dispatch."""
    if chunks_held >= MAX_HOLD_CHUNKS:
        return "escape 2: _chunks_held >= max_hold_chunks (16)"
    if elapsed >= MAX_PRE_DELAY:
        return "escape 3: elapsed >= MAX_PRE_DELAY_SECONDS (10.0 s)"
    if p_obs is None:
        return "(no rate observed yet)"
    if p_obs >= 1.0:
        return "escape 4: observed P >= 1.0"
    tau = min(max(0.0, target_audio_s * (1.0 / p_obs - 1.0)), MAX_PRE_DELAY)
    if buffered_s >= tau:
        return "escape 5: buffered >= tau_min ({:.2f} s)".format(tau)
    return "(not released)"


def simulate(p: float, chunk_size: int, target_audio_s: float,
             true_audio_s: float, label: str) -> None:
    """Feed chunks at producer rate ``p`` until the buffer releases."""
    clock = FakeClock()
    buf = StreamingChunkBuffer(
        watermark_ms=500,
        crossfade_samples=64,
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width=2,
        target_audio_seconds=target_audio_s,
        enable_adaptive_pre_buffer=True,
        max_pre_delay_seconds=MAX_PRE_DELAY,
        clock=clock,
    )
    audio_per_chunk_s = chunk_size / FRAME_HZ
    wall_per_chunk_s = audio_per_chunk_s / p
    chunk_bytes = int(audio_per_chunk_s * BYTES_PER_SEC)
    payload = b"\x01\x02" * (chunk_bytes // 2)

    n_chunks_total = max(1, int(round(true_audio_s / audio_per_chunk_s)))
    t_first = None
    released_at = None
    reason = "(never released before generation ended)"
    for i in range(n_chunks_total):
        if i == 0:
            clock.t = 0.0
            t_first = 0.0
        else:
            clock.t = i * wall_per_chunk_s
        chunks_held_before = i + 1
        elapsed = clock.t - t_first
        buffered_s = chunks_held_before * audio_per_chunk_s
        p_obs = None if i == 0 else p
        out = buf.push(payload, is_final=False)
        if out:
            released_at = clock.t
            reason = why_released(
                buf, elapsed, chunks_held_before, buffered_s, p_obs,
                target_audio_s,
            )
            break

    talker_seg_s = ((chunk_size + 5) / FRAME_HZ) / p
    print("  " + label)
    print("    P={p:.2f}  chunk_size={cs}  audio/chunk={apc:.3f}s  "
          "wall/chunk={wpc:.3f}s  T_a_est={ta:.2f}s".format(
              p=p, cs=chunk_size, apc=audio_per_chunk_s,
              wpc=wall_per_chunk_s, ta=target_audio_s))
    if released_at is None:
        print("    NOT RELEASED by chunk arrival - would wait for is_final")
    else:
        print("    segment 4 (release offset from chunk 0): {:.2f} s".format(
            released_at))
        print("    released by -> {}".format(reason))
    print("    talker segment (window/P) for comparison: {:.2f} s".format(
        talker_seg_s))
    if released_at is not None:
        print("    cushion / talker ratio: {:.2f}x".format(
            released_at / talker_seg_s if talker_seg_s else float("nan")))
    print()


def main() -> int:
    print("# Story 20.1 A4 - adaptive-cushion simulation against the shipped")
    print("# StreamingChunkBuffer (streaming_chunk_buffer.py), injected clock.")
    print("# T_a_est is what the code actually uses: text_length * 0.08 s/char")
    print("# (audio_coordinator.py:89). Measured T_a is shown for contrast.")
    print()
    LONG_CHARS, LONG_TRUE_TA = 349, 19.32
    print("## Long fixture: {} chars -> T_a_est {:.2f}s "
          "(measured T_a {:.2f}s, estimator overshoot {:+.0f}%)".format(
              LONG_CHARS, LONG_CHARS * 0.08, LONG_TRUE_TA,
              100.0 * (LONG_CHARS * 0.08 - LONG_TRUE_TA) / LONG_TRUE_TA))
    print()
    for p in (0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        simulate(p, 25, LONG_CHARS * 0.08, LONG_TRUE_TA,
                 "chunk_size=25 (committed default), P={:.2f}".format(p))
    print("## Same host, chunk_size=10 (the Follow-up B candidate)")
    print()
    for p in (0.5, 0.75, 0.9):
        simulate(p, 10, LONG_CHARS * 0.08, LONG_TRUE_TA,
                 "chunk_size=10, P={:.2f}".format(p))
    print("## Estimator sensitivity (A5): same P, T_a_est vs measured T_a")
    print()
    for ta, lbl in ((LONG_CHARS * 0.08, "T_a_est 27.92 s (what ships)"),
                    (LONG_TRUE_TA, "T_a measured 19.32 s (a perfect estimator)")):
        simulate(0.5, 25, ta, LONG_TRUE_TA, "P=0.50, " + lbl)
        simulate(0.8, 25, ta, LONG_TRUE_TA, "P=0.80, " + lbl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
