"""Story 20.4 AC #2 / AC #3 - re-derive the adaptive cushion against the
SHIPPED StreamingChunkBuffer, before and after the Story 20.4 policy, at
both chunk geometries.

This is the direct successor to ``20-1-adaptive-cushion-sim.py``. Same
technique - drive the production class with an injected clock and synthetic
chunk arrivals at a chosen producer rate P, and report WHICH escape released
playback and at what offset from the first chunk. No GPU, no model.

Three things it establishes that the story asks for:

  1. AC #2 - a sub-16 GiB host now starts materially sooner than the 10 s
     ``MAX_PRE_DELAY_SECONDS`` guardrail, and the guardrail is no longer the
     binding escape anywhere on the sweep.
  2. AC #3 - the AC #2 fix holds AT chunk_size = 10, not merely at 25. Story
     20.1 SS2.7 found the coupling ran the wrong way: at cs10 the shipped
     policy's cushion-to-talker ratio WORSENED from 2.50x to 4.00x, because
     the talker segment shrinks while the cap does not.
  3. The product trade, quantified: total silence is conserved (a cushion
     second not spent up front reappears as a gap second later), so the
     choice is only about WHERE the silence lands.

The pre-20.4 policy is reproduced by subclassing the shipped buffer and
restoring its old ``_cushion_decision`` body, so both columns come out of the
same driver. The legacy column is cross-checked against Story 20.1 SS2.7's
published table at the bottom of this file - if the reproduction drifts, the
run says so rather than quietly reporting a wrong baseline.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-adaptive-cushion-sim.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from myvoice.services.audio_coordinator import (  # noqa: E402
    _DEFAULT_STREAMING_CUSHION_BUDGET_SECONDS,
    _DEFAULT_STREAMING_MAX_PRE_DELAY_SECONDS,
    _DEFAULT_STREAMING_WATERMARK_MS,
    estimate_target_audio_seconds,
)
from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402

SAMPLE_RATE = 24000
BYTES_PER_SEC = SAMPLE_RATE * 2  # int16 mono
FRAME_HZ = 12.0
MAX_PRE_DELAY = _DEFAULT_STREAMING_MAX_PRE_DELAY_SECONDS
BUDGET = _DEFAULT_STREAMING_CUSHION_BUDGET_SECONDS
LOOKAHEAD = codec_token_streamer.DEFAULT_LOOKAHEAD

# Story 20.1 fixtures. The long one is the canonical Epic 18 paragraph.
LONG_CHARS, LONG_TRUE_TA = 349, 19.32
SHORT_CHARS, SHORT_TRUE_TA = 33, 2.30

# The pre-20.4 estimator, kept only so the legacy column uses the T_a the
# legacy code actually saw.
LEGACY_CHARS_TO_AUDIO_SECONDS = 0.08


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class LegacyPolicyBuffer(StreamingChunkBuffer):
    """The shipped buffer with its PRE-Story-20.4 cushion policy restored.

    Verbatim from ``streaming_chunk_buffer.py`` as of Story 20.3:

        if p >= 1.0:            return 0.0
        if p <= 0.0:            return max_pre_delay_seconds
        cushion = T_a * (1/p - 1)
        return max(0.0, min(cushion, max_pre_delay_seconds))

    ``_adaptive_ready_to_dispatch`` then compared buffered audio against it,
    as the last of five escapes.
    """

    def _cushion_decision(self, p_observed: float):
        if p_observed >= 1.0:
            return 0.0, self.REGIME_PRODUCER_KEEPS_UP
        if p_observed <= 0.0:
            return self._max_pre_delay_seconds, self.REGIME_GAPLESS_FEASIBLE
        cushion = self._target_audio_seconds * (1.0 / p_observed - 1.0)
        return (
            max(0.0, min(cushion, self._max_pre_delay_seconds)),
            self.REGIME_GAPLESS_FEASIBLE,
        )


def simulate(cls, p, chunk_size, target_audio_s, true_audio_s, budget=BUDGET):
    """Feed chunks at producer rate ``p`` until the buffer releases.

    Returns ``(release_offset_or_None, reason, talker_segment_seconds)``.
    """
    clock = FakeClock()
    buf = cls(
        watermark_ms=_DEFAULT_STREAMING_WATERMARK_MS,
        crossfade_samples=64,
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width=2,
        target_audio_seconds=target_audio_s,
        enable_adaptive_pre_buffer=True,
        max_pre_delay_seconds=MAX_PRE_DELAY,
        cushion_budget_seconds=budget,
        clock=clock,
    )
    audio_per_chunk_s = chunk_size / FRAME_HZ
    wall_per_chunk_s = audio_per_chunk_s / p
    chunk_bytes = int(audio_per_chunk_s * BYTES_PER_SEC)
    payload = b"\x01\x02" * (chunk_bytes // 2)

    n_chunks_total = max(1, int(round(true_audio_s / audio_per_chunk_s)))
    released_at = None
    for i in range(n_chunks_total):
        clock.t = i * wall_per_chunk_s
        if buf.push(payload, is_final=False):
            released_at = clock.t
            break
    talker_seg_s = ((chunk_size + LOOKAHEAD) / FRAME_HZ) / p
    return released_at, buf.last_release_reason, talker_seg_s


def total_gap_seconds(cushion, true_audio_s, p):
    """Accumulated mid-utterance silence for a given starting cushion.

    Playback cannot outrun the producer, so it ends at
    ``max(cushion + T_a, T_gen)`` where ``T_gen = T_a / P``. The silence that
    is not spent up front therefore reappears as gaps:

        total_gap = max(0, T_gen - T_a - cushion)
    """
    t_gen = true_audio_s / p
    return max(0.0, t_gen - true_audio_s - cushion)


def row(label, released, reason, talker_seg, true_audio_s, p):
    if released is None:
        return "  {:<12} {:>10} {:<26} {:>9} {:>9} {:>9}".format(
            label, "never", "(waited for is_final)", "-", "-", "-"
        )
    gap = total_gap_seconds(released, true_audio_s, p)
    return "  {:<12} {:>9.2f}s {:<26} {:>8.2f}s {:>8.2f}x {:>8.2f}s".format(
        label, released, reason or "-", talker_seg,
        released / talker_seg if talker_seg else float("nan"), gap,
    )


HEADER = "  {:<12} {:>10} {:<26} {:>9} {:>9} {:>9}".format(
    "policy", "segment 4", "released by", "talker", "ratio", "tot. gap"
)


def sweep(chunk_size, chars, true_ta, title):
    legacy_ta = chars * LEGACY_CHARS_TO_AUDIO_SECONDS
    new_ta = estimate_target_audio_seconds(chars)
    print("## {}".format(title))
    print("   chunk_size={cs}  window={w}  audio/chunk={apc:.3f}s".format(
        cs=chunk_size, w=chunk_size + LOOKAHEAD,
        apc=chunk_size / FRAME_HZ))
    print("   T_a: measured {m:.2f}s | pre-20.4 estimate {o:.2f}s ({oe:+.0f}%)"
          " | Story 20.4 estimate {n:.2f}s ({ne:+.0f}%)".format(
              m=true_ta, o=legacy_ta,
              oe=100.0 * (legacy_ta - true_ta) / true_ta,
              n=new_ta, ne=100.0 * (new_ta - true_ta) / true_ta))
    print("   cushion budget {b:.1f}s | guardrail {g:.1f}s (unchanged)".format(
        b=BUDGET, g=MAX_PRE_DELAY))
    print()
    for p in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        print("  P = {:.2f}".format(p))
        print(HEADER)
        r_old = simulate(LegacyPolicyBuffer, p, chunk_size, legacy_ta, true_ta)
        r_new = simulate(StreamingChunkBuffer, p, chunk_size, new_ta, true_ta)
        print(row("pre-20.4", r_old[0], r_old[1], r_old[2], true_ta, p))
        print(row("Story 20.4", r_new[0], r_new[1], r_new[2], true_ta, p))
        print()


def crosscheck():
    """Reproduce Story 20.1 SS2.7's published table with the legacy policy.

    If these do not match, the legacy reproduction above is wrong and every
    "before" number in this run is untrustworthy.
    """
    print("## Cross-check: legacy reproduction vs Story 20.1 SS2.7 published")
    print()
    expected = {
        (25, 0.50): 12.50, (25, 0.70): 11.90, (25, 0.75): 11.11,
        (25, 0.80): 7.81, (25, 0.85): 4.90, (25, 0.90): 2.31,
        (25, 0.95): 2.19,
        (10, 0.50): 10.00, (10, 0.75): 10.00, (10, 0.90): 2.78,
    }
    legacy_ta = LONG_CHARS * LEGACY_CHARS_TO_AUDIO_SECONDS
    ok = True
    for (cs, p), published in sorted(expected.items()):
        got, _, _ = simulate(LegacyPolicyBuffer, p, cs, legacy_ta, LONG_TRUE_TA)
        match = got is not None and abs(got - published) < 0.02
        ok = ok and match
        print("  cs={:<3} P={:.2f}  published {:>6.2f}s   reproduced {:>6}   {}"
              .format(cs, p, published,
                      "n/a" if got is None else "{:.2f}s".format(got),
                      "OK" if match else "MISMATCH"))
    print()
    print("  legacy reproduction: {}".format("VERIFIED" if ok else "BROKEN"))
    print()
    return ok


def main() -> int:
    print("# Story 20.4 AC #2 / AC #3 - adaptive-cushion re-derivation")
    print("# Driven against the shipped StreamingChunkBuffer with an")
    print("# injected clock. Committed streamer geometry: chunk_size={}, "
          "lookahead={}.".format(codec_token_streamer.DEFAULT_CHUNK_SIZE,
                                 codec_token_streamer.DEFAULT_LOOKAHEAD))
    print()
    ok = crosscheck()
    sweep(25, LONG_CHARS, LONG_TRUE_TA,
          "Long fixture at chunk_size = 25 (the PRE-20.4 geometry)")
    sweep(10, LONG_CHARS, LONG_TRUE_TA,
          "Long fixture at chunk_size = 10 (the COMMITTED geometry) - AC #3")
    sweep(10, SHORT_CHARS, SHORT_TRUE_TA,
          "Short / Clear Comms fixture at chunk_size = 10")
    print("## Notes")
    print("  * 'tot. gap' is accumulated MID-UTTERANCE silence, derived as")
    print("    max(0, T_a/P - T_a - cushion). Total silence is conserved:")
    print("    latency saved at the front reappears as gaps later. The")
    print("    product trade is about WHERE the silence lands, and Clear")
    print("    Comms is an interjection tool, so the front is the worst")
    print("    place for it (memory/clear_comms_purpose_framing.md).")
    print("  * 'ratio' is segment 4 / talker segment. Story 20.1 SS2.7 used")
    print("    the same denominator, ((chunk_size + lookahead)/12)/P.")
    print("  * The short fixture is only ~2.3 s of audio; at chunk_size 10")
    print("    that is 3 chunks, so several rows never reach a release")
    print("    decision before is_final would fire. Those read 'never'.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
