"""Story 20.4 AC #6 - aggregate the GUI progressive-playback CSV captures.

Reads ``20-4-gui-r*.csv`` (raw metric-stream captures written by
``observability/progressive_playback_csv_capture.py``) and reconstructs the
four TTFA segments per USER generation.

**Grouping by session_id is mandatory** - Story 20.3 SS4.1a learned this the
hard way. Each capture contains three or more sessions:

  * ``-`` (blank session id)  - the startup compile-priming generation
  * ``no-registry``           - priming's registry-suppressed post
  * one real uuid per user generation

A naive first-match join splices priming's segments 1-3 onto a user segment 4
and reports a "cushion" that is really however long the operator took to
click Generate. This script therefore keys everything by ``session_id`` and
keeps only sessions that carry ``ttfa_first_playback_write_ms`` - the marker
that audio actually reached the device.

Story 20.4 captures TWO user generations per launch (long, then short), so a
capture yields two qualifying sessions; they are reported in arrival order
and labelled from ``--labels``.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-aggregate-gui.py
    python310\\python.exe ... 20-4-aggregate-gui.py --glob "20-4-gui-r*.csv" --labels long,short

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import OrderedDict
from typing import Dict, List, Optional

ARTIFACTS = os.path.dirname(os.path.abspath(__file__))

BOUNDARIES = (
    "ttfa_generation_start_ms",
    "ttfa_talker_thread_start_ms",
    "ttfa_first_decode_step_ms",
    "ttfa_first_chunk_emit_ms",
    "ttfa_first_decode_complete_ms",
    "progressive_chunk_emit_ms",
    "ttfa_first_playback_write_ms",
)


def load_sessions(path: str) -> "OrderedDict[str, Dict]":
    """Group one capture's rows by session_id, preserving arrival order."""
    sessions: "OrderedDict[str, Dict]" = OrderedDict()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("session_id") or "").strip() or "-"
            entry = sessions.setdefault(
                sid, {"boundaries": {}, "chunk_audio_ms": [], "emit_ms": []}
            )
            name = row["metric_name"]
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if name in BOUNDARIES and name not in entry["boundaries"]:
                # First writer wins: a boundary is one-shot per generation.
                entry["boundaries"][name] = value
            if name == "progressive_chunk_audio_duration_ms":
                entry["chunk_audio_ms"].append(value)
            if name == "progressive_chunk_emit_ms":
                entry["emit_ms"].append(value)
    return sessions


def segments(entry: Dict) -> Optional[Dict[str, float]]:
    b = entry["boundaries"]
    need = (
        "ttfa_generation_start_ms",
        "ttfa_first_decode_step_ms",
        "ttfa_first_chunk_emit_ms",
        "ttfa_first_decode_complete_ms",
        "progressive_chunk_emit_ms",
        "ttfa_first_playback_write_ms",
    )
    if any(k not in b for k in need):
        return None
    t0 = b["ttfa_generation_start_ms"]
    out = {
        "seg1_prefill_ms": b["ttfa_first_decode_step_ms"] - t0,
        "seg2_talker_ms": (
            b["ttfa_first_chunk_emit_ms"] - b["ttfa_first_decode_step_ms"]
        ),
        "seg3_decode_ms": (
            b["ttfa_first_decode_complete_ms"] - b["ttfa_first_chunk_emit_ms"]
        ),
        "seg4_cushion_ms": (
            b["ttfa_first_playback_write_ms"] - b["progressive_chunk_emit_ms"]
        ),
        "total_ms": b["ttfa_first_playback_write_ms"] - t0,
    }
    if "ttfa_talker_thread_start_ms" in b:
        out["seg1a_dispatch_ms"] = b["ttfa_talker_thread_start_ms"] - t0
        out["seg1b_prefill_ms"] = (
            b["ttfa_first_decode_step_ms"] - b["ttfa_talker_thread_start_ms"]
        )
    out["chunks"] = len(entry["chunk_audio_ms"])

    # Steady-state producer emit/drain ratio (the OFR-E gate, Story 18.1
    # SS4.4): median inter-chunk wall time / median chunk audio duration.
    # < 1.0 means the producer outruns playback.
    emits = sorted(entry["emit_ms"])
    durations = entry["chunk_audio_ms"][:-1] or entry["chunk_audio_ms"]
    if len(emits) >= 3 and durations:
        deltas = [b - a for a, b in zip(emits, emits[1:])]
        med_delta = statistics.median(deltas)
        med_audio = statistics.median(durations)
        if med_audio:
            out["producer_ratio"] = med_delta / med_audio
    return out


FIELDS = (
    ("seg1b_prefill_ms", "1b prefill"),
    ("seg2_talker_ms", "2 talker"),
    ("seg3_decode_ms", "3 decode"),
    ("seg4_cushion_ms", "4 cushion"),
    ("total_ms", "TOTAL"),
    ("producer_ratio", "ratio"),
    ("chunks", "chunks"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="20-4-gui-r*.csv")
    ap.add_argument(
        "--labels", default="long,short",
        help="comma-separated labels for the user generations, in order",
    )
    ap.add_argument(
        "--skip-first-launch", action="store_true",
        help="drop r01 (the cold-compile-key warm-up launch)",
    )
    args = ap.parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    paths = sorted(glob.glob(os.path.join(ARTIFACTS, args.glob)))
    if args.skip_first_launch and paths:
        print("Skipping {} (cold-key warm-up launch)\n".format(
            os.path.basename(paths[0])))
        paths = paths[1:]
    if not paths:
        print("No captures matched {!r}".format(args.glob))
        return 1

    by_label: Dict[str, List[Dict[str, float]]] = {lab: [] for lab in labels}
    for path in paths:
        name = os.path.basename(path)
        sessions = load_sessions(path)
        user = [
            (sid, e) for sid, e in sessions.items()
            if "ttfa_first_playback_write_ms" in e["boundaries"]
            and sid not in ("-", "no-registry")
        ]
        print("{}: {} session(s) total, {} user generation(s)".format(
            name, len(sessions), len(user)))
        if len(user) != len(labels):
            print("  !! expected {} user generations ({}), got {} - "
                  "check the operator notes for this launch".format(
                      len(labels), ",".join(labels), len(user)))
        for i, (sid, entry) in enumerate(user):
            lab = labels[i] if i < len(labels) else "extra{}".format(i)
            seg = segments(entry)
            if seg is None:
                print("  {:<6} {} incomplete boundary set: {}".format(
                    lab, sid[:8], sorted(entry["boundaries"])))
                continue
            by_label.setdefault(lab, []).append(seg)
            print("  {:<6} {}  1b={:>7.1f}  2={:>8.1f}  3={:>6.1f}  "
                  "4={:>6.1f}  TOTAL={:>8.1f}  ratio={}  chunks={}".format(
                      lab, sid[:8],
                      seg.get("seg1b_prefill_ms", float("nan")),
                      seg["seg2_talker_ms"], seg["seg3_decode_ms"],
                      seg["seg4_cushion_ms"], seg["total_ms"],
                      "n/a" if "producer_ratio" not in seg
                      else "{:.3f}".format(seg["producer_ratio"]),
                      seg["chunks"]))

    for lab in labels:
        rows = by_label.get(lab) or []
        if not rows:
            continue
        print("\n== {} (n={}) ==".format(lab, len(rows)))
        for key, title in FIELDS:
            vals = [r[key] for r in rows if key in r]
            if not vals:
                continue
            print("  {:<12} median={:>9.3f}  min={:>9.3f}  max={:>9.3f}".format(
                title, statistics.median(vals), min(vals), max(vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
