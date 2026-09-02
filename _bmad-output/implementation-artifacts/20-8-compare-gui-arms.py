"""Story 20.8 AC #3 — the GUI A/B, cs25 vs cs10, scored segment by segment.

WHY A SIBLING RATHER THAN A FLAG ON ``20-6-compare-arms.py``
------------------------------------------------------------
That file scores an A/B whose arms differ in **lookahead**: it holds
``CHUNK_SIZE = 25``, derives each arm's first-emit threshold as
``CHUNK_SIZE + lookahead``, and **refuses to run if both arms declare the same
lookahead** — a guard that exists so a run cannot be compared against itself.
Here both arms have lookahead 0 and differ in *chunk size*, so that guard would
fire on a perfectly valid experiment and the threshold arithmetic would divide
segment 2 by the wrong number, which is the one mistake that mis-attributes the
whole capture.

Everything that is not the threshold is imported from it rather than restated:
``_load_arm`` (which in turn uses ``20-4-aggregate-gui.py``'s mandatory
session_id grouping — Story 20.3 §4.1a), ``_print_rows``, ``ROWS``, and the
200 ms ``seg1a`` semaphore-contamination bar with its named exclusions.

PROVENANCE
----------
The reference arm's launcher writes ``20-8-cs25-manifest.json`` recording the
geometry the capturing process actually resolved. If it disagrees with the
declared chunk size this script refuses to score, exactly as
``20-6-compare-arms.py`` refuses on a lookahead mismatch: a command-line flag
is not provenance.

Usage:
    python310\\python.exe 20-8-compare-gui-arms.py
    python310\\python.exe 20-8-compare-gui-arms.py --check 20-8-gui-r03.csv

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

ARTIFACTS = os.path.dirname(os.path.abspath(__file__))

_CMP_PATH = os.path.join(ARTIFACTS, "20-6-compare-arms.py")
_spec = importlib.util.spec_from_file_location("_cmp20_6", _CMP_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover (defensive)
    raise SystemExit("cannot load {}".format(_CMP_PATH))
_cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cmp)

REFERENCE_CS = 25
CANDIDATE_CS = 10
LOOKAHEAD = 0          # retired on both arms (Story 20.6)


def _manifest_chunk_size(stem):
    path = os.path.join(ARTIFACTS, "{}-manifest.json".format(stem))
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return int(json.load(fh)["resolved_chunk_size"])
    except Exception as exc:  # noqa: BLE001
        print("  !! {} unreadable ({}); falling back to the declared value"
              .format(os.path.basename(path), exc))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="CSV",
                    help="per-launch contamination check on one capture")
    ap.add_argument("--a-glob", default="20-8-cs25-r*.csv")
    ap.add_argument("--a-label", default="reference: chunk_size 25 (ships today)")
    ap.add_argument("--a-chunk-size", type=int, default=REFERENCE_CS)
    ap.add_argument("--a-manifest-stem", default="20-8-cs25")
    ap.add_argument("--b-glob", default="20-8-gui-r*.csv")
    ap.add_argument("--b-label", default="candidate: chunk_size 10 (committed)")
    ap.add_argument("--b-chunk-size", type=int, default=CANDIDATE_CS)
    ap.add_argument("--b-manifest-stem", default="20-8-gui")
    ap.add_argument("--labels", default="long,short")
    ap.add_argument("--skip-first-launch", action="store_true", default=True)
    ap.add_argument("--no-skip-first-launch", dest="skip_first_launch",
                    action="store_false")
    ap.add_argument("--max-dispatch-ms", type=float,
                    default=_cmp.DEFAULT_MAX_DISPATCH_MS)
    args = ap.parse_args()

    if args.check:
        return _cmp._check_one(args.check, args.max_dispatch_ms)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    arms = []
    for tag, glob_, label, cs, stem in (
        ("A", args.a_glob, args.a_label, args.a_chunk_size, args.a_manifest_stem),
        ("B", args.b_glob, args.b_label, args.b_chunk_size, args.b_manifest_stem),
    ):
        from_manifest = _manifest_chunk_size(stem)
        if from_manifest is not None and from_manifest != cs:
            print("FATAL: arm {} declared chunk_size={} but its capture "
                  "manifest records {}. One of them is wrong, and dividing "
                  "segment 2 by the wrong first-emit threshold would "
                  "mis-attribute the whole experiment."
                  .format(tag, cs, from_manifest), file=sys.stderr)
            return 2
        arms.append({"tag": tag, "glob": glob_, "label": label,
                     "chunk_size": cs, "threshold": cs + LOOKAHEAD,
                     "manifest": from_manifest})

    if arms[0]["chunk_size"] == arms[1]["chunk_size"]:
        print("FATAL: both arms declare chunk_size={}. This would compare a "
              "run against itself.".format(arms[0]["chunk_size"]),
              file=sys.stderr)
        return 2

    print("=" * 78)
    print("Story 20.8 AC #3 — GUI A/B, same code, same machine, same sitting")
    print("=" * 78)
    for arm in arms:
        print("  arm {}: {}".format(arm["tag"], arm["label"]))
        print("         glob={}  chunk_size={}  first-emit threshold={} frames"
              .format(arm["glob"], arm["chunk_size"], arm["threshold"]))
        print("         provenance={}".format(
            "capture manifest" if arm["manifest"] is not None
            else "DECLARED ON THE COMMAND LINE (no manifest found)"))
    print("  lookahead is 0 on BOTH arms (Story 20.6, retired). A generation "
          "whose\n  segment 1a exceeds {:.0f} ms is excluded as "
          "semaphore-contaminated and named."
          .format(args.max_dispatch_ms))
    print()

    for arm in arms:
        print("-" * 78)
        print("arm {} — {}".format(arm["tag"], arm["label"]))
        print("-" * 78)
        kept, excluded = _cmp._load_arm(
            arm["glob"], labels, args.skip_first_launch, args.max_dispatch_ms)
        arm["kept"], arm["excluded"] = kept, excluded
        _cmp._print_rows(kept, excluded)
        if excluded:
            print("  {} generation(s) excluded for semaphore contamination."
                  .format(len(excluded)))
        for lab in labels:
            rows = kept.get(lab) or []
            if not rows:
                continue
            print("  == {} (n={}) ==".format(lab, len(rows)))
            for key, title in _cmp.ROWS:
                vals = [r[key] for r in rows if key in r]
                if not vals:
                    continue
                print("     {:<12} median={:>9.3f}  min={:>9.3f}  "
                      "max={:>9.3f}".format(
                          title, statistics.median(vals), min(vals), max(vals)))
            # Per-frame talker cost — LONG only. A short utterance can
            # first-emit from residual_flush, where the frame count is the
            # whole utterance and varies per take, so dividing by the nominal
            # threshold would divide by a number we do not have.
            if lab == "long":
                seg2 = [r["seg2_talker_ms"] for r in rows if "seg2_talker_ms" in r]
                if seg2:
                    print("     {:<12} {:.2f} ms/frame  (segment 2 median "
                          "{:.1f} / {} frames)".format(
                              "per-frame", statistics.median(seg2) / arm["threshold"],
                              statistics.median(seg2), arm["threshold"]))
        print()

    print("=" * 78)
    print("SEGMENT BY SEGMENT: arm B (cs{}) minus arm A (cs{})".format(
        arms[1]["chunk_size"], arms[0]["chunk_size"]))
    print("=" * 78)
    for lab in labels:
        a_rows = arms[0]["kept"].get(lab) or []
        b_rows = arms[1]["kept"].get(lab) or []
        if not a_rows or not b_rows:
            print("  {}: insufficient rows (A={} B={})".format(
                lab, len(a_rows), len(b_rows)))
            continue
        print("\n  {} class (A n={}, B n={})".format(lab, len(a_rows), len(b_rows)))
        for key, title in _cmp.ROWS:
            a = _cmp._median(a_rows, key)
            b = _cmp._median(b_rows, key)
            if a is None or b is None:
                continue
            delta = b - a
            pct = (100.0 * delta / a) if a else float("nan")
            print("    {:<12} A={:>9.1f}  B={:>9.1f}  delta={:>+9.1f}  "
                  "({:+.1f} %)".format(title, a, b, delta, pct))
        # The cross-check the headless sweep predicts: segment 2 should fall by
        # the frame difference times the per-frame talker cost. A large
        # residual means first emit is NOT gated on the threshold, which would
        # invalidate the whole geometry argument.
        if lab == "long":
            a2 = _cmp._median(a_rows, "seg2_talker_ms")
            b2 = _cmp._median(b_rows, "seg2_talker_ms")
            if a2 and b2:
                per_frame = a2 / arms[0]["threshold"]
                d_frames = arms[0]["threshold"] - arms[1]["threshold"]
                predicted = -per_frame * d_frames
                observed = b2 - a2
                print("\n    cross-check (P3 falsifier):")
                print("      arm A per-frame talker cost      = {:.2f} ms"
                      .format(per_frame))
                print("      predicted segment-2 saving       = {:+.1f} ms "
                      "({} fewer frames)".format(predicted, d_frames))
                print("      observed segment-2 saving        = {:+.1f} ms"
                      .format(observed))
                print("      residual                         = {:+.1f} ms"
                      .format(observed - predicted))
                print("      A large residual would mean first emit is not "
                      "gated on the\n      threshold at all — the geometry "
                      "argument would not hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
