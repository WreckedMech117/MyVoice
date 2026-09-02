"""Story 20.6 follow-up — the kill-switch A/B, aggregated segment by segment.

WHY THIS EXISTS
---------------
Story 20.6's GUI capture measured TOTAL 1,364 ms against Story 20.3's 1,353 ms
baseline and found segment 2 flat where retiring the lookahead should have cut
it by roughly a sixth. That comparison cannot be attributed, because **there is
no post-20.5, pre-20.6 GUI baseline**: Story 20.5 verified headless, and the
last GUI capture predates codec state caching. 1,353 -> 1,364 spans two
stories and an unknown amount of driver, OS and model-pin drift.

``MYVOICE_CODEC_STATE_CACHE=0`` gives the pre-20.5 geometry — stateless decode,
lookahead 5, the post-decode trim and the Story 20.4 seam blend — on **today's
code, today's machine, today's driver**. That is the clean control the 20.3
comparison cannot be, and this script scores the two arms against each other.

WHAT IT REPORTS THAT ``20-4-aggregate-gui.py`` DOES NOT
-------------------------------------------------------
1. **Segment 1a.** The 20.4 aggregator computes ``seg1a_dispatch_ms`` and then
   never prints it, so a dispatch stall lands in TOTAL with nothing naming it.
   That is exactly how two of the ten generations in the 20.6 capture came to
   carry 840 ms and 1,383 ms of pre-talker time: the operator generated before
   compile priming released the request semaphore. Here 1a is a first-class
   column, contaminated rows are EXCLUDED with the reason printed, and every
   summary also carries ``TOTAL-1a`` so a stall cannot hide in the headline.

2. **Per-frame talker cost** — the quantity actually in dispute. Segment 2 ends
   when the streamer's first-emit threshold is reached, and that threshold is
   ``chunk_size + lookahead``: **30** frames in the kill-switch arm, **25** in
   the shipping arm. Dividing by the wrong one silently mis-attributes the
   whole experiment, so each arm's lookahead is declared on the command line,
   echoed loudly, and cross-checked against the capture manifest the launcher
   writes.

   Only the LONG class yields this cleanly. A short utterance never reaches
   either threshold and first-emits from ``residual_flush``, where the frame
   count is the whole utterance and varies per take. Short rows are reported
   but excluded from the per-frame figure, and this file says so rather than
   dividing by a number it does not have.

3. **Segment-by-segment deltas**, not just TOTAL. Segments 1b and 3 both moved
   in the 20.6 capture; they are attributed here rather than absorbed.

Usage:
    # per-launch contamination check (the launcher runs this after each launch)
    python310\\python.exe 20-6-compare-arms.py --check 20-6-killswitch-r03.csv

    # the comparison
    python310\\python.exe 20-6-compare-arms.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

ARTIFACTS = os.path.dirname(os.path.abspath(__file__))

# Reuse the Story 20.4 aggregator's session grouping and segment arithmetic
# rather than restating them. That file carries the Story 20.3 SS4.1a fix (key
# everything by session_id; priming emits its own boundaries first), and a
# second copy of it is exactly the kind of drift this epic keeps paying for.
_AGG_PATH = os.path.join(ARTIFACTS, "20-4-aggregate-gui.py")
_spec = importlib.util.spec_from_file_location("_gui_agg", _AGG_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover (defensive)
    raise SystemExit("cannot load {}".format(_AGG_PATH))
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)

CHUNK_SIZE = 25  # AC #5: unchanged by Story 20.6; both launchers assert it.

# A user generation whose segment 1a exceeds this was almost certainly started
# before compile priming released the request semaphore. Clean dispatches
# measure ~2 ms; the two contaminated generations in the 20.6 capture measured
# 840 ms and 1,383 ms. Anything between is ambiguous, which is why the
# threshold is loose and the row is reported rather than silently dropped.
DEFAULT_MAX_DISPATCH_MS = 200.0


def _load_arm(pattern, labels, skip_first, max_dispatch_ms):
    """Return ``(kept_by_label, excluded)`` for one arm."""
    paths = sorted(glob.glob(os.path.join(ARTIFACTS, pattern)))
    if skip_first and paths:
        print("  skipping {} (declared cold-compile-key throwaway launch)"
              .format(os.path.basename(paths[0])))
        paths = paths[1:]
    if not paths:
        raise SystemExit("no captures matched {!r}".format(pattern))

    kept = {lab: [] for lab in labels}
    excluded = []
    for path in paths:
        name = os.path.basename(path)
        sessions = _agg.load_sessions(path)
        user = [
            (sid, e) for sid, e in sessions.items()
            if "ttfa_first_playback_write_ms" in e["boundaries"]
            and sid not in ("-", "no-registry")
        ]
        if len(user) != len(labels):
            print("  !! {}: expected {} user generations ({}), got {}".format(
                name, len(labels), ",".join(labels), len(user)))
        for i, (sid, entry) in enumerate(user):
            lab = labels[i] if i < len(labels) else "extra{}".format(i)
            seg = _agg.segments(entry)
            if seg is None:
                print("  !! {} {:<6} {} incomplete boundary set".format(
                    name, lab, sid[:8]))
                continue
            seg["_capture"] = name
            seg["_label"] = lab
            seg["_sid"] = sid[:8]
            seg["total_minus_1a_ms"] = (
                seg["total_ms"] - seg.get("seg1a_dispatch_ms", 0.0))
            if seg.get("seg1a_dispatch_ms", 0.0) > max_dispatch_ms:
                seg["_excluded"] = True
                excluded.append(seg)
            else:
                seg["_excluded"] = False
                kept.setdefault(lab, []).append(seg)
    return kept, excluded


ROWS = (
    ("seg1a_dispatch_ms", "1a dispatch"),
    ("seg1b_prefill_ms", "1b prefill"),
    ("seg2_talker_ms", "2 talker"),
    ("seg3_decode_ms", "3 decode"),
    ("seg4_cushion_ms", "4 cushion"),
    ("total_minus_1a_ms", "TOTAL-1a"),
    ("total_ms", "TOTAL"),
    ("producer_ratio", "ratio"),
    ("chunks", "chunks"),
)


def _median(rows, key):
    vals = [r[key] for r in rows if key in r]
    return statistics.median(vals) if vals else None


def _print_rows(kept, excluded):
    everything = [s for rows in kept.values() for s in rows] + excluded
    for seg in sorted(everything, key=lambda s: (s["_capture"], s["_label"])):
        flag = ("  EXCLUDED (generated before priming released the semaphore)"
                if seg.get("_excluded") else "")
        print("    {:<22} {:<6} {}  1a={:>7.1f}  1b={:>6.1f}  2={:>8.1f}  "
              "3={:>6.1f}  4={:>5.1f}  TOTAL-1a={:>8.1f}{}".format(
                  seg["_capture"], seg["_label"], seg["_sid"],
                  seg.get("seg1a_dispatch_ms", float("nan")),
                  seg.get("seg1b_prefill_ms", float("nan")),
                  seg["seg2_talker_ms"], seg["seg3_decode_ms"],
                  seg["seg4_cushion_ms"], seg["total_minus_1a_ms"], flag))


def _manifest_lookahead(pattern):
    """Read the capture manifest the launcher writes next to its CSVs, if any.

    Provenance beats a command-line flag: the manifest records the geometry
    actually resolved in the capturing process, so a comparison cannot be run
    against a mis-declared arm.
    """
    stem = pattern.split("r*")[0].rstrip("-")
    path = os.path.join(ARTIFACTS, "{}-manifest.json".format(stem))
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return int(json.load(fh)["resolved_lookahead"])
    except Exception as exc:  # noqa: BLE001
        print("  !! {} unreadable ({}); falling back to the declared flag"
              .format(os.path.basename(path), exc))
        return None


def _check_one(csv_name, max_dispatch_ms):
    """Per-launch contamination check, run by the launcher immediately after
    each launch so the operator learns DURING the six launches rather than
    after all six are spent."""
    path = os.path.join(ARTIFACTS, csv_name)
    if not os.path.exists(path):
        print("  [CHECK] no CSV at {} - nothing captured this launch".format(
            csv_name))
        return 1
    sessions = _agg.load_sessions(path)
    user = [
        (sid, e) for sid, e in sessions.items()
        if "ttfa_first_playback_write_ms" in e["boundaries"]
        and sid not in ("-", "no-registry")
    ]
    if not user:
        print("  [CHECK] {}: NO user generation reached playback.".format(
            csv_name))
        return 1
    bad = []
    for sid, entry in user:
        seg = _agg.segments(entry)
        if seg is None:
            print("  [CHECK] {} {} incomplete boundary set".format(
                csv_name, sid[:8]))
            continue
        dispatch = seg.get("seg1a_dispatch_ms", 0.0)
        status = "ok" if dispatch <= max_dispatch_ms else "CONTAMINATED"
        print("  [CHECK] {} {}  dispatch={:>8.1f} ms  talker={:>8.1f} ms  {}"
              .format(csv_name, sid[:8], dispatch, seg["seg2_talker_ms"],
                      status))
        if dispatch > max_dispatch_ms:
            bad.append(dispatch)
    if bad:
        print()
        print("  ***********************************************************")
        print("  ** THIS LAUNCH IS SPOILED.")
        print("  ** You generated before the 'Preparing TTS engine'")
        print("  ** indicator cleared, so priming still held the request")
        print("  ** semaphore and {:.0f} ms of that wait is inside the".format(
            max(bad)))
        print("  ** measurement. The generation measured queueing, not")
        print("  ** first-forward.")
        print("  ** WAIT FOR THE INDICATOR ON THE NEXT LAUNCH.")
        print("  ***********************************************************")
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="CSV",
                    help="per-launch contamination check on one capture")
    ap.add_argument("--a-glob", default="20-6-killswitch-r*.csv")
    ap.add_argument("--a-label",
                    default="kill-switch: pre-20.5 geometry")
    ap.add_argument("--a-lookahead", type=int, default=5)
    ap.add_argument("--b-glob", default="20-6-gui-r*.csv")
    ap.add_argument("--b-label",
                    default="shipping: 20.5 state cache + 20.6 retirement")
    ap.add_argument("--b-lookahead", type=int, default=0)
    ap.add_argument("--labels", default="long,short")
    ap.add_argument("--skip-first-launch", action="store_true", default=True)
    ap.add_argument("--no-skip-first-launch", dest="skip_first_launch",
                    action="store_false")
    ap.add_argument("--max-dispatch-ms", type=float,
                    default=DEFAULT_MAX_DISPATCH_MS)
    args = ap.parse_args()

    if args.check:
        return _check_one(args.check, args.max_dispatch_ms)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    arms = []
    for tag, pattern, label, lookahead in (
        ("A", args.a_glob, args.a_label, args.a_lookahead),
        ("B", args.b_glob, args.b_label, args.b_lookahead),
    ):
        from_manifest = _manifest_lookahead(pattern)
        if from_manifest is not None and from_manifest != lookahead:
            print("FATAL: arm {} declared lookahead={} but its capture "
                  "manifest records {}. One of them is wrong, and dividing "
                  "segment 2 by the wrong first-emit threshold would "
                  "mis-attribute the whole experiment."
                  .format(tag, lookahead, from_manifest), file=sys.stderr)
            return 2
        arms.append({
            "tag": tag, "glob": pattern, "label": label,
            "lookahead": lookahead, "threshold": CHUNK_SIZE + lookahead,
            "manifest": from_manifest,
        })

    if arms[0]["lookahead"] == arms[1]["lookahead"]:
        print("FATAL: both arms declare lookahead={}. This would compare a "
              "run against itself. Arm A must be the kill-switch capture "
              "(lookahead 5) and arm B the shipping one (lookahead 0)."
              .format(arms[0]["lookahead"]), file=sys.stderr)
        return 2

    print("=" * 78)
    print("Story 20.6 follow-up - kill-switch A/B, same code, same machine")
    print("=" * 78)
    for arm in arms:
        print("  arm {}: {}".format(arm["tag"], arm["label"]))
        print("         glob={}  lookahead={}  first-emit threshold={} frames"
              .format(arm["glob"], arm["lookahead"], arm["threshold"]))
        print("         provenance={}".format(
            "capture manifest" if arm["manifest"] is not None
            else "DECLARED ON THE COMMAND LINE (no manifest found)"))
    print("  chunk_size={} on both arms. A generation whose segment 1a "
          "exceeds {:.0f} ms".format(CHUNK_SIZE, args.max_dispatch_ms))
    print("  is excluded as semaphore-contaminated and named below.")
    print()

    for arm in arms:
        print("-" * 78)
        print("arm {} - {}".format(arm["tag"], arm["label"]))
        print("-" * 78)
        kept, excluded = _load_arm(
            arm["glob"], labels, args.skip_first_launch, args.max_dispatch_ms)
        arm["kept"], arm["excluded"] = kept, excluded
        _print_rows(kept, excluded)
        if excluded:
            print("  {} generation(s) excluded for semaphore contamination."
                  .format(len(excluded)))
        for lab in labels:
            rows = kept.get(lab) or []
            if not rows:
                continue
            print("  == {} (n={}) ==".format(lab, len(rows)))
            for key, title in ROWS:
                vals = [r[key] for r in rows if key in r]
                if not vals:
                    continue
                print("     {:<12} median={:>9.3f}  min={:>9.3f}  "
                      "max={:>9.3f}".format(
                          title, statistics.median(vals), min(vals), max(vals)))
        print()

    # ---- segment-by-segment delta, per label -------------------------- #
    print("=" * 78)
    print("SEGMENT BY SEGMENT: arm B (shipping) minus arm A (kill-switch)")
    print("=" * 78)
    for lab in labels:
        a_rows = arms[0]["kept"].get(lab) or []
        b_rows = arms[1]["kept"].get(lab) or []
        if not a_rows or not b_rows:
            print("  {}: not enough data (A n={}, B n={})".format(
                lab, len(a_rows), len(b_rows)))
            continue
        print("\n  -- {} (A n={}, B n={}) --".format(
            lab, len(a_rows), len(b_rows)))
        print("     {:<12} {:>12} {:>12} {:>12}".format(
            "segment", "A median", "B median", "B - A"))
        for key, title in ROWS:
            a_med, b_med = _median(a_rows, key), _median(b_rows, key)
            if a_med is None or b_med is None:
                continue
            print("     {:<12} {:>12.3f} {:>12.3f} {:>+12.3f}".format(
                title, a_med, b_med, b_med - a_med))

    # ---- the headline: per-frame talker cost -------------------------- #
    print()
    print("=" * 78)
    print("PER-FRAME TALKER COST - the quantity in dispute")
    print("=" * 78)
    print("  Segment 2 ends when the streamer's first-emit threshold is")
    print("  reached, so ms/frame = segment 2 / (chunk_size + lookahead).")
    print("  LONG class only: a short utterance never reaches either")
    print("  threshold and first-emits from residual_flush, where the frame")
    print("  count is the whole utterance and varies per take.")
    print()
    per_frame = {}
    for arm in arms:
        rows = arm["kept"].get("long") or []
        if not rows:
            print("  arm {}: no clean long generations".format(arm["tag"]))
            continue
        vals = [r["seg2_talker_ms"] / arm["threshold"] for r in rows]
        per_frame[arm["tag"]] = statistics.median(vals)
        print("  arm {} threshold {:>2} frames   {:>6.2f} ms/frame   "
              "[{:.2f}-{:.2f}, n={}]".format(
                  arm["tag"], arm["threshold"], per_frame[arm["tag"]],
                  min(vals), max(vals), len(vals)))
    print("  (Story 20.3's GUI baseline, for reference only: 1,147.5 / 30 =")
    print("   38.25 ms/frame. It is NOT one of these arms - different")
    print("   session, different month, pre-state-caching.)")

    if "A" in per_frame and "B" in per_frame:
        a, b = per_frame["A"], per_frame["B"]
        delta_pct = 100.0 * (b - a) / a if a else float("nan")
        print()
        print("  B - A = {:+.2f} ms/frame ({:+.1f} %)".format(b - a, delta_pct))
        print()
        print("  Against the prediction recorded before the run:")
        if abs(delta_pct) <= 5.0:
            print("    -> P1 HOLDS. THE ARMS AGREE. Per-frame talker cost is")
            print("       the same with and without 20.5 + 20.6, so neither")
            print("       story caused the 38.2 -> 45.2 ms shift against")
            print("       Story 20.3. That shift is cross-session drift, the")
            print("       20.3 baseline is NOT comparable, and it should stop")
            print("       being used as one.")
            print("    -> 20.5 + 20.6 are then TTFA-NEUTRAL through the GUI:")
            print("       the five frames saved are real and worth {:.0f} ms at".format(
                5 * b))
            print("       this per-frame cost, but they did not show against")
            print("       a baseline that had already moved underneath them.")
        elif b > a:
            print("    -> P2 HOLDS. THE REGRESSION IS REAL. Per-frame talker")
            print("       cost is {:+.1f} % higher WITH 20.5 + 20.6, on the".format(
                delta_pct))
            print("       same code and the same machine. Something in state")
            print("       caching or the retirement is slowing the talker and")
            print("       it absorbed the five frames the retirement saved.")
            print("    -> F2 (the chunk-size reopen) is BLOCKED until this is")
            print("       understood: its premise (cs10 at 829 ms vs cs25 at")
            print("       1,491 ms) was measured pre-state-caching and needs")
            print("       re-establishing on current code first.")
        else:
            print("    -> P3. The shipping arm is FASTER per frame ({:+.1f} %),".format(
                delta_pct))
            print("       which no hypothesis predicted. Before believing it,")
            print("       confirm the two globs really are the two arms - the")
            print("       manifest provenance line above is the check.")
        long_a = _median(arms[0]["kept"].get("long") or [], "seg2_talker_ms")
        long_b = _median(arms[1]["kept"].get("long") or [], "seg2_talker_ms")
        print()
        print("  Cross-check. If the talker is otherwise unchanged, segment 2")
        print("  should fall by exactly the five frames the retirement saves:")
        print("    predicted B - A on segment 2: {:+.0f} ms  (-5 frames x {:.2f})"
              .format(-5 * a, a))
        if long_a is not None and long_b is not None:
            print("    observed  B - A on segment 2: {:+.1f} ms".format(
                long_b - long_a))
            residual = (long_b - long_a) - (-5 * a)
            print("    residual (observed - predicted): {:+.1f} ms".format(
                residual))
            print("    A residual near zero means the ONLY thing that changed")
            print("    is the frame count. A large positive residual is the")
            print("    per-frame regression, restated in milliseconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
