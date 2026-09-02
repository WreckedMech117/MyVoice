"""Story 20.8 Phase 1 — the chunk-size re-baseline sweep launcher.

ONE SITTING, ONE MACHINE. Drives ``tools/ttfa_spike_harness.py`` once per
(chunk_size, utterance-class) cell, sequentially, in a single invocation of
this script, so every point in the resulting curve shares a driver state, a
thermal state and a model pin. Story 20.6 §12 is the reason that is not
optional: the same code measured in two sessions two months apart differed by
46.66 vs 38.25 ms/frame, and a cross-session comparison already produced one
false conclusion in this epic.

Why ``cs25`` appears TWICE
-------------------------
``cs25`` is the control (Story 20.8 AC #1 forbids carrying Story 20.6's
capture over). It is measured first AND last. The difference between the two
control passes is the *within-sitting* drift, which is the only honest
yardstick for "is this point's delta real" once every point has already been
put in the same sitting. Without it the sitting itself is an unbounded
confound — the smaller version of exactly the mistake §12 caught.

Why LONG runs before SHORT within a point
-----------------------------------------
Each sweep point is now a distinct ``decode_window_frames``, hence a distinct
compile-cache key (Story 20.4 threaded the geometry through;
Story 20.6 made the lookahead half conditional; ``resolve_streamer_geometry()``
returns ``(N, 0)`` so the window IS ``N``). So the FIRST process to touch a
new point pays the cold compile. Running LONG first puts that cost in a place
we can price it: the long process's startup-priming generation is cold, the
short process's is warm on the same key, and the difference is the
cold-compile cost for that point. The measured runs in both processes are
warm either way (``--prime`` + ``--warmup 1``).

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-sweep.py
    ... 20-8-sweep.py --points 25,15,10,7 --runs 10

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ARTIFACTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ARTIFACTS))
PY = os.path.join(REPO, "python310", "python.exe")
HARNESS = os.path.join(REPO, "tools", "ttfa_spike_harness.py")


def cache_keys():
    root = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "MyVoice", "torch_compile_cache")
    if not os.path.isdir(root):
        return set()
    return {d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))}


def run_cell(chunk_size, utterance, runs, tag):
    stem = "20-8-sweep-{}-{}-cs{}".format(tag, utterance, chunk_size)
    out_csv = os.path.join(ARTIFACTS, stem + ".csv")
    log_path = os.path.join(ARTIFACTS, stem + ".log")
    if os.path.exists(out_csv):
        # Guard added mid-sitting after a retry invocation silently
        # overwrote the A-pass control: the tag is derived from --control,
        # so re-running one point with --no-second-control re-labels it "A".
        # A measurement is not a thing to overwrite by accident.
        raise SystemExit(
            "REFUSING to overwrite an existing capture: {} — move or "
            "delete it first, or pick a different tag.".format(out_csv))
    cmd = [PY, HARNESS,
           "--utterance", utterance,
           "--chunk-size", str(chunk_size),
           "--runs", str(runs),
           "--warmup", "1",
           "--prime",
           "--out", out_csv]
    before = cache_keys()
    t0 = time.time()
    print("\n>>> {}  ({})".format(stem, " ".join(cmd[2:])), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = time.time() - t0
    after = cache_keys()
    new_keys = sorted(after - before)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n== stdout ==\n")
        fh.write(proc.stdout or "")
        fh.write("\n== stderr ==\n")
        fh.write(proc.stderr or "")
    print(proc.stdout or "", flush=True)
    if proc.returncode != 0:
        print("!! rc={}  see {}".format(proc.returncode, log_path),
              file=sys.stderr, flush=True)
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        for line in tail:
            print("   " + line, file=sys.stderr, flush=True)
    print("<<< {}  rc={}  wall={:.1f}s  new_cache_keys={}".format(
        stem, proc.returncode, wall, new_keys or "none"), flush=True)
    return {
        "cell": stem, "chunk_size": chunk_size, "utterance": utterance,
        "tag": tag, "rc": proc.returncode, "process_wall_s": round(wall, 1),
        "new_cache_keys": new_keys, "csv": os.path.basename(out_csv),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="25,15,10,7")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--classes", default="long,short")
    ap.add_argument("--control", type=int, default=25)
    ap.add_argument("--no-second-control", dest="second_control",
                    action="store_false", default=True)
    ap.add_argument("--manifest",
                    default=os.path.join(ARTIFACTS, "20-8-sweep-manifest.json"))
    args = ap.parse_args()

    points = [int(p) for p in args.points.split(",") if p.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    plan = [(p, "A" if p == args.control else "") for p in points]
    if args.second_control:
        plan.append((args.control, "B"))
    # tag disambiguates the two control passes; every other point gets "P"
    plan = [(p, t or "P") for p, t in plan]

    print("=" * 78)
    print("Story 20.8 Phase 1 — chunk-size re-baseline, one sitting")
    print("  started {}".format(datetime.datetime.now().isoformat(
        timespec="seconds")))
    print("  plan: " + ", ".join("cs{}[{}]".format(p, t) for p, t in plan))
    print("  classes: {}   runs/cell: {} (+1 discarded warmup, --prime)"
          .format(",".join(classes), args.runs))
    print("  cache keys before the sitting: {}".format(sorted(cache_keys())))
    print("=" * 78, flush=True)

    results = []
    t0 = time.time()
    for chunk_size, tag in plan:
        for utterance in classes:      # LONG first: it pays the cold key
            results.append(run_cell(chunk_size, utterance, args.runs, tag))
            with open(args.manifest, "w", encoding="utf-8") as fh:
                json.dump({"cells": results,
                           "cache_keys_now": sorted(cache_keys())},
                          fh, indent=2)
    print("\nSITTING COMPLETE in {:.1f} min. cache keys after: {}".format(
        (time.time() - t0) / 60.0, sorted(cache_keys())))
    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
