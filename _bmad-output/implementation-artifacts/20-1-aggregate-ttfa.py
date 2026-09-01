"""Story 20.1 - aggregate the TTFA-decomposition + chunk-sweep CSVs.

Reads the per-cell CSVs produced by ``tools/ttfa_spike_harness.py`` and prints
every table the evidence file quotes.

Review-response changes (2026-08-31):
  * A1  - real linear-interpolated quantile; ``max`` reported separately.
          The prior expression ``sorted(v)[round(0.95*(n-1))]`` returns the
          MAXIMUM for every n <= 10, so "p95" was mislabelled throughout.
  * A2  - reports the independent perf_counter bracket and its slack. That,
          not the segment sum, is the falsifiable reconciliation check.
  * A10 - one vocabulary for the two TTFA quantities, used everywhere:
            TTFA(post)    = t0 -> first PCM handed to the consumer
            TTFA(release) = TTFA(post) + segment 4 (consumer cushion)
  * A11 - values at or below the 0.5 ms ``time.time`` step print as
          "<=res" rather than as measurements.
  * A12 - per-key n printed wherever it can differ from the cell's n.
  * A16 - the two unmeasured matrix cells are named, not left to inference.
  * B1  - short-utterance chunk-size sweep table.

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

D = Path(__file__).resolve().parent / "clean2"

NEWLINE = chr(10)
FRAME_HZ = 12.0
LOOKAHEAD = 5
# Measured on this host: time.time() advances in 0.5 ms steps. Anything at or
# under that is a clock artifact, not a measurement (review A11).
CLOCK_STEP_MS = 0.5

CELLS = {
    "A - long / RTX 5090 static-watermark": "20-1-ttfa-rtx5090-long-cs25.csv",
    "B - short / RTX 5090 static-watermark": "20-1-ttfa-rtx5090-short-cs25.csv",
}

SWEEP_LONG = {
    5: "20-1-sweep-long-cs5.csv",
    10: "20-1-sweep-long-cs10.csv",
    15: "20-1-sweep-long-cs15.csv",
    25: "20-1-ttfa-rtx5090-long-cs25.csv",
}

SWEEP_SHORT = {
    5: "20-1-sweep-short-cs5.csv",
    10: "20-1-sweep-short-cs10.csv",
    15: "20-1-sweep-short-cs15.csv",
    25: "20-1-ttfa-rtx5090-short-cs25.csv",
}


def load(name: str) -> List[Dict[str, Any]]:
    p = D / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def warm(rows):
    return [r for r in rows if r.get("is_warmup") == "False"]


def col(rows, key) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v not in (None, "", "None"):
            out.append(float(v))
    return out


def quantile(vals: List[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile (numpy / R type-7). See review A1."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def med(vals):
    return statistics.median(vals) if vals else None


def fmt(x, nd=1, mark_subres=False):
    if x is None:
        return "n/a"
    if mark_subres and abs(x) <= CLOCK_STEP_MS:
        return "<=res"
    return "{:,.{nd}f}".format(x, nd=nd)


SEGS = (
    ("segment 1 - prefill / prompt-encode", "seg1_prefill_ms", False),
    ("  1a - MyVoice dispatch overhead", "seg1a_dispatch_overhead_ms", True),
    ("  1b - model prompt-encode", "seg1b_prompt_encode_ms", False),
    ("segment 2 - talker to first token chunk", "seg2_talker_to_first_chunk_ms", False),
    ("segment 3 - first decode (codec -> PCM)", "seg3_first_decode_ms", False),
    ("segment 4 - consumer cushion (harness)", "seg4_consumer_cushion_ms", True),
    ("residual - decode-complete -> post", "residual_post_ms", True),
    ("TTFA(post) - t0 -> first PCM posted", "measured_t0_to_post_ms", False),
    ("  restatement: first_chunk_latency_ms", "first_chunk_latency_ms", False),
)


def cell_report(title: str, rows: List[Dict[str, Any]]) -> None:
    w = warm(rows)
    cold = [r for r in rows if r.get("is_warmup") == "True"]
    print(NEWLINE + "### " + title)
    print("warm runs n={}; discarded leading (cold) runs n={}".format(len(w), len(cold)))
    if not w:
        print("  NO DATA")
        return
    print("{:<42}{:>6}{:>13}{:>15}{:>12}".format(
        "segment", "n", "median (ms)", "p95 interp (ms)", "max (ms)"))
    for label, key, subres in SEGS:
        vals = col(w, key)
        print("{:<42}{:>6}{:>13}{:>15}{:>12}".format(
            label, len(vals),
            fmt(med(vals), 1, subres),
            fmt(quantile(vals, 0.95), 1, subres),
            fmt(max(vals) if vals else None, 1, subres),
        ))

    post = med(col(w, "measured_t0_to_post_ms")) or 0.0
    seg2 = med(col(w, "seg2_talker_to_first_chunk_ms")) or 0.0
    seg4 = med(col(w, "seg4_consumer_cushion_ms")) or 0.0
    release = post + seg4
    print(NEWLINE + "  TTFA(release) = TTFA(post) + segment 4: {} ms".format(fmt(release)))
    print("  TALKER-BOUND FRACTION (segment 2 / TTFA(release)): {:.1f} %".format(
        100.0 * seg2 / release if release else float("nan")))

    # A2 - the falsifiable check.
    brk = col(w, "independent_ttfa_bracket_ms")
    slack = col(w, "bracket_slack_ms")
    print("  INDEPENDENT BRACKET (perf_counter, outside the metric stream):")
    print("    n={}  median={} ms  vs TTFA(post) median {} ms".format(
        len(brk), fmt(med(brk)), fmt(post)))
    if slack:
        print("    slack (bracket - TTFA(post)): median={} ms  min={} ms  max={} ms".format(
            fmt(med(slack), 3), fmt(min(slack), 3), fmt(max(slack), 3)))
        print("    |slack| <= 2 x clock step (1.0 ms) on all runs: {}".format(
            all(abs(x) <= 2 * CLOCK_STEP_MS for x in slack)))

    # A2 - and the identity, stated as an identity.
    recon = col(w, "reconcile_error_pct")
    print("  segment-sum residual vs TTFA(post): median {} % "
          "(IDENTITY - the segments telescope; see evidence 2.4)".format(
              fmt(med(recon), 4)))

    paths = [r.get("first_emit_path", "?") for r in w]
    print("  first-emit path distribution: {}".format(
        {p: paths.count(p) for p in sorted(set(paths))}))
    ta = med(col(w, "total_audio_ms"))
    gen = med(col(w, "generation_wall_ms"))
    ratio = med(col(w, "producer_ratio"))
    prate = med(col(w, "producer_rate_P"))
    print("  T_a (total audio, median): {} ms over n={}; generation wall: {} ms".format(
        fmt(ta), len(col(w, "total_audio_ms")), fmt(gen)))
    print("  RTF (T_a / generation wall): {}".format(
        fmt(ta / gen, 3) if (ta and gen) else "n/a"))
    print("  producer emit/drain ratio: {} (n={}); producer rate P: {}".format(
        fmt(ratio, 3), len(col(w, "producer_ratio")), fmt(prate, 3)))
    if cold:
        c = cold[0]
        print("  [cold first-generation-of-process run, discarded] "
              "TTFA(post)={:,.0f} ms (seg1a model-load={:,.0f}, "
              "seg1b first-forward={:,.0f}, seg2 talker={:,.0f})".format(
                  float(c["measured_t0_to_post_ms"]),
                  float(c["seg1a_dispatch_overhead_ms"]),
                  float(c["seg1b_prompt_encode_ms"]),
                  float(c["seg2_talker_to_first_chunk_ms"])))


def sweep_report(title: str, table: Dict[int, str]) -> None:
    print(NEWLINE + NEWLINE + "## " + title)
    print("{:>11}{:>8}{:>13}{:>12}{:>12}{:>11}{:>15}{:>9}{:>8}{:>9}{:>11}{:>5}".format(
        "chunk_size", "window", "audio/chunk", "seg2 (ms)", "seg4 (ms)",
        "TTFA(post)", "TTFA(release)", "ratio", "P", "chunks", "gen wall", "n"))
    for k in sorted(table):
        rows = warm(load(table[k]))
        if not rows:
            print("{:>11}   NO DATA".format(k))
            continue
        post = med(col(rows, "measured_t0_to_post_ms")) or 0.0
        seg4 = med(col(rows, "seg4_consumer_cushion_ms")) or 0.0
        print("{:>11}{:>8}{:>12.0f}m{:>12}{:>12}{:>11}{:>15}{:>9}{:>8}{:>9}{:>11}{:>5}".format(
            k, k + LOOKAHEAD, k / FRAME_HZ * 1000.0,
            fmt(med(col(rows, "seg2_talker_to_first_chunk_ms")), 0),
            fmt(seg4, 1, True),
            fmt(post, 0),
            fmt(post + seg4, 0),
            fmt(med(col(rows, "producer_ratio")), 3),
            fmt(med(col(rows, "producer_rate_P")), 3),
            fmt(med(col(rows, "chunks")), 0),
            fmt(med(col(rows, "generation_wall_ms")), 0),
            len(rows),
        ))
        paths = [r.get("first_emit_path", "?") for r in rows]
        print("{:>11}  first-emit path: {}".format(
            "", {p: paths.count(p) for p in sorted(set(paths))}))


def coldwarm_report() -> None:
    print(NEWLINE + NEWLINE + "## Cold (first generation of process) vs warm - "
          "reconciliation with the Epic 18 baseline")
    for label, name in (
        ("tts_compile=auto (Epic 18 Branch A analogue)",
         "20-1-coldwarm-long-cs25-compile.csv"),
        ("tts_compile=off  (Epic 18 Branch B analogue)",
         "20-1-coldwarm-long-cs25-eager.csv"),
    ):
        rows = load(name)
        if not rows:
            print("  " + label + ": NO DATA")
            continue
        print(NEWLINE + "  " + label)
        print("    {:>5}{:>12}{:>12}{:>14}{:>13}{:>8}{:>11}".format(
            "run", "TTFA(post)", "1a load", "1b 1st-fwd", "2 talker", "3 dec",
            "gen wall"))
        for r in rows:
            print("    {:>5}{:>12,.0f}{:>12,.0f}{:>14,.0f}{:>13,.0f}{:>8,.0f}{:>11,.0f}".format(
                r["run_index"],
                float(r["measured_t0_to_post_ms"]),
                float(r["seg1a_dispatch_overhead_ms"]),
                float(r["seg1b_prompt_encode_ms"]),
                float(r["seg2_talker_to_first_chunk_ms"]),
                float(r["seg3_first_decode_ms"]),
                float(r["generation_wall_ms"])))
        warm_rows = rows[1:]
        if warm_rows:
            # GUI-equivalent = drop segment 1a (the model is preloaded at
            # startup), keep the cold run's 1b + 2 + 3.
            c = rows[0]
            gui_equiv = (float(c["seg1b_prompt_encode_ms"])
                         + float(c["seg2_talker_to_first_chunk_ms"])
                         + float(c["seg3_first_decode_ms"]))
            print("    cold GUI-equivalent (1b+2+3, model preloaded): "
                  "{:,.0f} ms".format(gui_equiv))
            print("    warm steady-state TTFA(post) median: {} ms (n={})".format(
                fmt(med(col(warm_rows, "measured_t0_to_post_ms")), 0),
                len(warm_rows)))
            print("    warm producer ratio median: {} (n={})".format(
                fmt(med(col(warm_rows, "producer_ratio")), 3),
                len(col(warm_rows, "producer_ratio"))))


def breakeven_report() -> None:
    print(NEWLINE + NEWLINE + "## AC #2b Phase 2 - DERIVED adaptive-cushion "
          "break-even (sub-16 GiB hosts; NOT observed)")
    print("  Cells C (long / sub-16 GiB) and D (short / sub-16 GiB) of the "
          "AC #2b matrix are UNMEASURED - no such host was available "
          "(review A16).")
    for title, name, chars in (
        ("A - long", "20-1-ttfa-rtx5090-long-cs25.csv", 349),
        ("B - short", "20-1-ttfa-rtx5090-short-cs25.csv", 33),
    ):
        rows = warm(load(name))
        if not rows:
            continue
        ta_ms = med(col(rows, "total_audio_ms"))
        if ta_ms is None:
            continue
        a_c = 25 / FRAME_HZ
        window = (25 + LOOKAHEAD) / FRAME_HZ
        m = math.ceil(window / a_c) * a_c
        for lbl, ta in (("measured", ta_ms / 1000.0),
                        ("code estimate 0.08 s/char", chars * 0.08)):
            print(NEWLINE + "  cell {} [T_a {}] = {:.2f} s   A_c = {:.3f} s   "
                  "W = {:.3f} s   M = {:.3f} s".format(
                      title, lbl, ta, a_c, window, m))
            print("    cushion adds ANY delay      when P < {:.3f}".format(ta / (ta + a_c)))
            print("    cushion EXCEEDS talker seg  when P < {:.3f}".format(ta / (ta + m)))
            print("    cushion hits the 10 s clamp when P < {:.3f}".format(ta / (ta + 10.0)))


def main() -> int:
    print("# Story 20.1 - TTFA decomposition aggregate (re-captured 2026-08-31)")
    print("Source: tools/ttfa_spike_harness.py on RTX 5090 / torch 2.10.0+cu128 /")
    print("transformers 4.57.3 / qwen-tts pin 3fdb4682 / bf16 + tts_compile=auto")
    print("Quantiles are linear-interpolated (type-7); ``max`` is shown "
          "separately. Values marked <=res are at or below the 0.5 ms "
          "time.time() step on this host.")
    print(NEWLINE + "## AC #2 + AC #2b Phase 1 - four-segment decomposition")
    for title, name in CELLS.items():
        cell_report(title, load(name))
    sweep_report("AC #5 - chunk-size sweep, LONG utterance "
                 "(lookahead=5, tts_compile=auto, bf16)", SWEEP_LONG)
    sweep_report("B1 - chunk-size sweep, SHORT utterance "
                 "(the Clear Comms interjection class)", SWEEP_SHORT)
    coldwarm_report()
    breakeven_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
