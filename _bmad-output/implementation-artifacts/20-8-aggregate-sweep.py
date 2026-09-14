"""Story 20.8 Phase 1 — aggregate the re-baseline sweep into the curve.

Reads the ``20-8-sweep-{tag}-{class}-cs{N}.csv`` files the launcher writes and
reports, per point, the four quantities AC #1 asks for plus the two controls.

REUSE NOTE — why this is not ``20-6-compare-arms.py``
-----------------------------------------------------
``20-6-compare-arms.py`` (and the ``20-4-aggregate-gui.py`` it imports) parse
**GUI metric-stream captures**: one row per metric, keyed by ``session_id``,
where a single file interleaves the startup-priming generation, its
registry-suppressed post, and the operator's generations. Its two hard-won
behaviours are (a) group by ``session_id`` so priming's segments are never
spliced onto a user generation, and (b) exclude a generation whose segment 1a
exceeds 200 ms as semaphore-contaminated — the operator clicked Generate
before priming released the request semaphore.

Phase 1 is headless and spends zero operator time, so neither hazard exists in
this shape of capture: the harness writes one already-segmented row per run,
drives generations strictly sequentially with a fresh collector each time, and
has no operator to click early. Re-pointing the GUI parser at a file with a
different schema would not reuse its judgement, only its name.

What IS reused is the judgement itself:

  * ``DEFAULT_MAX_DISPATCH_MS`` — the same 200 ms bar, applied to
    ``seg1a_dispatch_overhead_ms``, with excluded rows NAMED rather than
    silently dropped. Headless dispatch measures ~1-2 ms; anything near 200 ms
    means the run was not measuring what it claims to.
  * **per-frame talker cost comes from the LONG class only.** A short
    utterance can first-emit from ``residual_flush``, where the frame count is
    the whole utterance and varies per take, so dividing segment 2 by the
    nominal threshold would divide by a number we do not have. Short rows are
    reported and excluded from ms/frame — ``20-6-compare-arms.py`` §2's rule,
    unchanged.
  * medians, not means, with min/max carried so a spread cannot hide.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-aggregate-sweep.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

ARTIFACTS = os.path.dirname(os.path.abspath(__file__))

# Reused verbatim from 20-6-compare-arms.py.
DEFAULT_MAX_DISPATCH_MS = 200.0

# Consumer constants, read off audio_coordinator.py:62 / :89-91 and mirrored by
# tools/ttfa_spike_harness.py. Static watermark on >=16 GiB hosts.
WATERMARK_MS = 500
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920          # streaming_decoder.py:201 — 12.5 Hz
EDGE_LOSS_SAMPLES = 555           # streaming_decoder.py:202, first decode only

CELL_RE = re.compile(r"20-8-sweep-([AB P])?-?", re.I)


def watermark_floor():
    """Derive, do not assume, the minimum viable chunk_size (AC #1).

    The static-watermark release rule is ``buffered_bytes >= watermark_bytes``
    (streaming_chunk_buffer.py:338-341), i.e. the FIRST chunk releases on its
    own iff it carries at least ``WATERMARK_MS`` of audio. On the shipping
    state-carrying path the first decode of a session is the only one that
    pays the edge loss, so the first chunk is ``N*1920 - 555`` samples and
    every later one is exactly ``N*1920`` (codec_state_cache.py:130-135).
    """
    wm_samples = int(WATERMARK_MS / 1000.0 * SAMPLE_RATE)
    out = {"watermark_ms": WATERMARK_MS, "watermark_samples": wm_samples}
    floor = None
    rows = []
    for n in range(1, 26):
        first = n * SAMPLES_PER_FRAME - EDGE_LOSS_SAMPLES
        held = 1
        cum = first
        while cum < wm_samples:
            held += 1
            cum += n * SAMPLES_PER_FRAME
        rows.append({
            "chunk_size": n,
            "first_chunk_ms": round(first / SAMPLE_RATE * 1000.0, 1),
            "steady_chunk_ms": round(n * SAMPLES_PER_FRAME / SAMPLE_RATE * 1000.0, 1),
            "chunks_held_by_watermark": held,
        })
        if held == 1 and floor is None:
            floor = n
    out["floor_chunk_size"] = floor
    out["exact_solve"] = (
        "N*1920 - 555 >= 12000  ->  N >= {:.4f}  ->  N >= {}".format(
            (wm_samples + EDGE_LOSS_SAMPLES) / SAMPLES_PER_FRAME, floor))
    out["table"] = rows
    return out


def load_cell(path, max_dispatch_ms):
    kept, excluded = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("is_warmup", "")).strip().lower() == "true":
                continue
            r = {}
            for k, v in row.items():
                if v is None or v == "":
                    r[k] = None
                    continue
                try:
                    r[k] = float(v)
                except ValueError:
                    r[k] = v
            r["_file"] = os.path.basename(path)
            d = r.get("seg1a_dispatch_overhead_ms")
            if d is not None and d > max_dispatch_ms:
                excluded.append(r)
            else:
                kept.append(r)
    return kept, excluded


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None
            and not isinstance(r[key], str)]
    return statistics.median(vals) if vals else None


def spread(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None
            and not isinstance(r[key], str)]
    if not vals:
        return None, None, None
    return min(vals), max(vals), max(vals) - min(vals)


def ttfa_release(r):
    """End-to-end TTFA including the consumer cushion.

    ``measured_t0_to_post_ms`` ends when the chunk is posted to the consumer;
    ``seg4_consumer_cushion_ms`` is the additional hold before the buffer
    releases. Only the sum moves when the watermark floor is crossed, which is
    precisely the effect this sweep must not misattribute, so it is the
    headline number here and TTFA(post) is carried beside it.
    """
    post = r.get("measured_t0_to_post_ms")
    s4 = r.get("seg4_consumer_cushion_ms")
    if post is None:
        return None
    return post + (s4 or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dispatch-ms", type=float,
                    default=DEFAULT_MAX_DISPATCH_MS)
    ap.add_argument("--json-out",
                    default=os.path.join(ARTIFACTS, "20-8-sweep-summary.json"))
    args = ap.parse_args()

    wf = watermark_floor()
    print("=" * 96)
    print("Story 20.8 Phase 1 — chunk-size re-baseline (one sitting, one machine)")
    print("=" * 96)
    print("\nTASK 1 — WATERMARK FLOOR, DERIVED")
    print("  static watermark {} ms = {} samples (>=16 GiB hosts; "
          "audio_coordinator.py:62)".format(wf["watermark_ms"],
                                            wf["watermark_samples"]))
    print("  first chunk = N*1920 - 555 samples; later chunks = N*1920 "
          "(state-carrying path)")
    print("  " + wf["exact_solve"])
    print("  {:<11} {:>14} {:>15} {:>22}".format(
        "chunk_size", "1st chunk ms", "steady chunk ms",
        "chunks held by w/mark"))
    for row in wf["table"]:
        if row["chunk_size"] > 16 and row["chunk_size"] != 25:
            continue
        mark = "  <- FLOOR" if row["chunk_size"] == wf["floor_chunk_size"] else ""
        print("  {:<11} {:>14} {:>15} {:>22}{}".format(
            row["chunk_size"], row["first_chunk_ms"], row["steady_chunk_ms"],
            row["chunks_held_by_watermark"], mark))
    print("  => minimum viable chunk_size = {}. Points below it hand back a "
          "cushion penalty".format(wf["floor_chunk_size"]))
    print("     and their TTFA is not a geometry result.")

    cells = {}
    for path in sorted(glob.glob(os.path.join(ARTIFACTS, "20-8-sweep-*-cs*.csv"))):
        name = os.path.basename(path)[:-4]
        parts = name.split("-")          # 20 8 sweep TAG CLASS csN
        tag, klass, cs = parts[3], parts[4], int(parts[5][2:])
        kept, excluded = load_cell(path, args.max_dispatch_ms)
        cells[(cs, klass, tag)] = (kept, excluded)

    summary = {"watermark_floor": wf, "points": {}}

    for klass in ("long", "short"):
        print("\n" + "=" * 96)
        print("TASK 2 — {} class".format(klass.upper()))
        print("=" * 96)
        hdr = ("  {:<12} {:>3} {:>7} {:>9} {:>9} {:>10} {:>10} {:>9} "
               "{:>7} {:>7} {:>8} {:>7}")
        print(hdr.format("point", "n", "frames", "seg2", "ms/frame",
                         "TTFA post", "TTFA rel", "cushion", "ratio",
                         "chunks", "dec ms", "aud ms"))
        keys = sorted([k for k in cells if k[1] == klass],
                      key=lambda k: (-k[0], k[2]))
        for cs, _k, tag in keys:
            kept, excluded = cells[(cs, klass, tag)]
            if not kept:
                print("  cs{}[{}] — NO CLEAN ROWS".format(cs, tag))
                continue
            paths = {r.get("first_emit_path") for r in kept}
            frames = {r.get("first_emit_frames") for r in kept}
            fr = med(kept, "first_emit_frames")
            seg2 = med(kept, "seg2_talker_to_first_chunk_ms")
            per_frame = (seg2 / fr) if (klass == "long" and fr) else None
            rel = [ttfa_release(r) for r in kept]
            rel = [v for v in rel if v is not None]
            row = {
                "chunk_size": cs, "tag": tag, "n": len(kept),
                "first_emit_paths": sorted(str(p) for p in paths),
                "first_emit_frames": fr,
                "seg2_talker_ms": seg2,
                "ms_per_frame": per_frame,
                "ttfa_post_ms": med(kept, "measured_t0_to_post_ms"),
                "ttfa_release_ms": statistics.median(rel) if rel else None,
                "ttfa_release_min": min(rel) if rel else None,
                "ttfa_release_max": max(rel) if rel else None,
                "seg4_cushion_ms": med(kept, "seg4_consumer_cushion_ms"),
                "consumer_chunks_held": med(kept, "consumer_chunks_held"),
                "producer_ratio": med(kept, "producer_ratio"),
                "chunks": med(kept, "chunks"),
                "median_decode_chunk_ms": med(kept, "median_decode_chunk_ms"),
                "median_chunk_audio_ms": med(kept, "median_chunk_audio_ms"),
                "gen_wall_ms": med(kept, "generation_wall_ms"),
                "excluded_dispatch": len(excluded),
            }
            summary["points"]["cs{}-{}-{}".format(cs, klass, tag)] = row
            print(hdr.format(
                "cs{}[{}]".format(cs, tag), row["n"],
                "-" if fr is None else int(fr),
                "-" if seg2 is None else round(seg2, 1),
                "-" if per_frame is None else round(per_frame, 2),
                round(row["ttfa_post_ms"], 1),
                round(row["ttfa_release_ms"], 1),
                round(row["seg4_cushion_ms"], 1)
                if row["seg4_cushion_ms"] is not None else "-",
                "-" if row["producer_ratio"] is None
                else round(row["producer_ratio"], 3),
                "-" if row["chunks"] is None else int(row["chunks"]),
                "-" if row["median_decode_chunk_ms"] is None
                else round(row["median_decode_chunk_ms"], 1),
                "-" if row["median_chunk_audio_ms"] is None
                else round(row["median_chunk_audio_ms"], 0)))
            if len(paths) > 1 or len(frames) > 1:
                print("      first-emit paths seen: {}  frames: {}".format(
                    sorted(str(p) for p in paths),
                    sorted(str(f) for f in frames)))
            if excluded:
                for r in excluded:
                    print("      EXCLUDED {} run {} dispatch={:.1f} ms "
                          "(> {:.0f} ms bar)".format(
                              r["_file"], r.get("run_index"),
                              r["seg1a_dispatch_overhead_ms"],
                              args.max_dispatch_ms))

        # ---- controls and the go/no-go arithmetic --------------------- #
        ctl = [(cs, tag) for (cs, k, tag) in cells if k == klass and tag in "AB"]
        ctl_rows = {}
        for cs, tag in ctl:
            kept, _ = cells[(cs, klass, tag)]
            rel = [ttfa_release(r) for r in kept]
            rel = [v for v in rel if v is not None]
            if rel:
                ctl_rows[tag] = rel
        if len(ctl_rows) == 2:
            a, b = ctl_rows["A"], ctl_rows["B"]
            drift = statistics.median(b) - statistics.median(a)
            pooled = a + b
            print("\n  CONTROL cs25 measured twice in the sitting:")
            print("    pass A median TTFA(release) = {:.1f} ms "
                  "[{:.1f}-{:.1f}, n={}]".format(
                      statistics.median(a), min(a), max(a), len(a)))
            print("    pass B median TTFA(release) = {:.1f} ms "
                  "[{:.1f}-{:.1f}, n={}]".format(
                      statistics.median(b), min(b), max(b), len(b)))
            print("    within-sitting drift (B - A) = {:+.1f} ms".format(drift))
            print("    pooled control within-arm spread (max-min) = {:.1f} ms "
                  "(n={})".format(max(pooled) - min(pooled), len(pooled)))
            summary.setdefault("controls", {})[klass] = {
                "A_median": statistics.median(a), "B_median": statistics.median(b),
                "drift_ms": drift,
                "pooled_median": statistics.median(pooled),
                "pooled_spread_ms": max(pooled) - min(pooled),
                "pooled_n": len(pooled),
            }

    print("\n" + "=" * 96)
    print("GO / NO-GO (thresholds fixed in the story before the work)")
    print("=" * 96)
    for klass in ("long", "short"):
        c = summary.get("controls", {}).get(klass)
        if not c:
            continue
        bar = max(c["pooled_spread_ms"], 83.0)
        print("\n  {} class — control cs25 pooled median {:.1f} ms; "
              "margin bar = {:.1f} ms".format(
                  klass.upper(), c["pooled_median"], bar))
        print("    (bar = max(this sitting's control spread, Story 20.6's "
              "~83 ms within-arm spread))")
        for key, row in sorted(summary["points"].items(),
                               key=lambda kv: -kv[1]["chunk_size"]):
            if not key.endswith("-" + klass + "-P"):
                continue
            delta = row["ttfa_release_ms"] - c["pooled_median"]
            viable = row["chunk_size"] >= wf["floor_chunk_size"]
            ratio_ok = (row["producer_ratio"] is None
                        or row["producer_ratio"] < 1.0)
            beats = delta < -bar
            verdict = ("GO-candidate" if (viable and ratio_ok and beats)
                       else "below floor" if not viable
                       else "ratio >= 1.0" if not ratio_ok
                       else "inside the noise bar")
            print("    cs{:<3} TTFA(release) {:>8.1f} ms   delta {:>+8.1f} ms   "
                  "ratio {:>6}   viable={:<5} -> {}".format(
                      row["chunk_size"], row["ttfa_release_ms"], delta,
                      "n/a" if row["producer_ratio"] is None
                      else round(row["producer_ratio"], 3),
                      str(viable), verdict))
            row.setdefault("verdict", {})[klass] = verdict
            row.setdefault("delta_vs_control", {})[klass] = delta

    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote {}".format(os.path.basename(args.json_out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
