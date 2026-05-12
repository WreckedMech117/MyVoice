"""Story 18.4 Task 9.5 - compute verdict from the joint audition CSV.

Cross-references the per-listener truth-table with the audition CSV at
``_bmad-output/implementation-artifacts/18-4-bf16-compile-pinbump-audition.csv``
to compute:

  - The verdict-gate result: PASS iff zero listeners flag `audible_seam`
    on any actual bf16+compile trial across all listeners.
  - Per-system defect-flag counts (the canonical Story 17.1 evidence
    table).
  - Per-listener + per-utterance subtotals.

The truth-table at
``_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/_perlistener_truthtable.json``
randomizes which actual mode (``fp32_eager`` vs ``bf16_compile``) lands
as ``trial_A`` vs ``trial_B`` per listener. So a listener's
``b_defects_observed`` column may map to fp32_eager OR bf16_compile —
this script does the mapping correctly.

Re-runnable: read-only against the CSV + truth-table.

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


SCRIPT_DIR = Path(__file__).resolve().parent
TRUTH_TABLE_PATH = SCRIPT_DIR / "18-4-perceptual-fixtures" / "_perlistener_truthtable.json"
CANONICAL_CSV = SCRIPT_DIR / "18-4-bf16-compile-pinbump-audition.csv"

ACTUAL_MODES = ("fp32_eager", "bf16_compile")
DEFECT_VOCAB = (
    "none",
    "audible_seam",
    "clipping",
    "phase_artifact",
    "tonal_distortion",
    "other_describe_in_notes",
)


def _load_truth_table() -> dict:
    if not TRUTH_TABLE_PATH.exists():
        raise SystemExit(f"FATAL: truth-table not found at {TRUTH_TABLE_PATH}")
    return json.loads(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))


def _load_audition_rows() -> list[dict]:
    if not CANONICAL_CSV.exists():
        raise SystemExit(f"FATAL: audition CSV not found at {CANONICAL_CSV}")
    with CANONICAL_CSV.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _map_row_to_actual_modes(row: dict, truth: dict) -> tuple[str, str, str]:
    """Return (a_actual_mode, b_actual_mode, listener_id) for a CSV row."""
    listener = row["listener_id"]
    utt = row["utterance_id"]
    entry = truth[listener][utt]
    return entry["trial_A_actual_mode"], entry["trial_B_actual_mode"], listener


def main() -> int:
    truth = _load_truth_table()
    rows = _load_audition_rows()

    print(f"Loaded {len(rows)} audition rows from {CANONICAL_CSV.name}")
    listeners_in_csv = sorted({r["listener_id"] for r in rows})
    print(f"Listeners with data: {listeners_in_csv}")
    listeners_in_truth = sorted(truth.keys())
    print(f"Listeners in truth-table: {listeners_in_truth}")
    missing = [L for L in listeners_in_truth if L not in listeners_in_csv]
    if missing:
        print(f"Listeners NOT YET AUDITIONED: {missing}")
        print(f"(Partial verdict will run on the {len(listeners_in_csv)} available)")
    print()

    # -- Defect-flag table per actual mode --------------------------------- #
    defects_by_mode: dict[str, Counter] = {m: Counter() for m in ACTUAL_MODES}
    audible_seam_flags: list[tuple[str, str, str]] = []  # (listener, utt, mode)

    for row in rows:
        a_mode, b_mode, listener = _map_row_to_actual_modes(row, truth)
        a_def = row["a_defects_observed"]
        b_def = row["b_defects_observed"]
        defects_by_mode[a_mode][a_def] += 1
        defects_by_mode[b_mode][b_def] += 1
        if a_def == "audible_seam":
            audible_seam_flags.append((listener, row["utterance_id"], a_mode))
        if b_def == "audible_seam":
            audible_seam_flags.append((listener, row["utterance_id"], b_mode))

    print("=" * 72)
    print("Per-actual-mode defect-flag counts")
    print("=" * 72)
    # Pretty print as table.
    col_widths = (24, 14, 14)
    header = ("defect", "fp32_eager", "bf16_compile")
    print(f"{header[0]:<{col_widths[0]}}{header[1]:>{col_widths[1]}}{header[2]:>{col_widths[2]}}")
    print("-" * sum(col_widths))
    for defect in DEFECT_VOCAB:
        fp_count = defects_by_mode["fp32_eager"][defect]
        bf_count = defects_by_mode["bf16_compile"][defect]
        marker = " <-- VERDICT GATE" if defect == "audible_seam" else ""
        print(
            f"{defect:<{col_widths[0]}}{fp_count:>{col_widths[1]}}{bf_count:>{col_widths[2]}}"
            f"{marker}"
        )

    print()
    print("=" * 72)
    print("Per-listener preference counts")
    print("=" * 72)
    pref_by_listener: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        pref_by_listener[row["listener_id"]][row["a_or_b_preferred"]] += 1
    for L in sorted(pref_by_listener.keys()):
        c = pref_by_listener[L]
        print(
            f"  {L}: A={c.get('A', 0)}, B={c.get('B', 0)}, "
            f"equivalent={c.get('equivalent', 0)}"
        )

    # Map listener preferences back to actual-mode preferences.
    print()
    print("Per-listener ACTUAL-MODE preference (after un-blinding via truth-table)")
    actual_pref_by_listener: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        a_mode, b_mode, listener = _map_row_to_actual_modes(row, truth)
        pref = row["a_or_b_preferred"]
        if pref == "A":
            actual_pref_by_listener[listener][a_mode] += 1
        elif pref == "B":
            actual_pref_by_listener[listener][b_mode] += 1
        else:
            actual_pref_by_listener[listener]["equivalent"] += 1
    for L in sorted(actual_pref_by_listener.keys()):
        c = actual_pref_by_listener[L]
        print(
            f"  {L}: fp32_eager={c.get('fp32_eager', 0)}, "
            f"bf16_compile={c.get('bf16_compile', 0)}, "
            f"equivalent={c.get('equivalent', 0)}"
        )

    print()
    print("=" * 72)
    print("VERDICT GATE: audible_seam flags on bf16_compile trials")
    print("=" * 72)
    bf16_compile_audible_seam = [
        f for f in audible_seam_flags if f[2] == "bf16_compile"
    ]
    fp32_eager_audible_seam = [
        f for f in audible_seam_flags if f[2] == "fp32_eager"
    ]
    print(f"  bf16_compile audible_seam flags: {len(bf16_compile_audible_seam)}")
    for L, utt, mode in bf16_compile_audible_seam:
        print(f"    - {L} on {utt}")
    print(f"  fp32_eager audible_seam flags:   {len(fp32_eager_audible_seam)}")
    for L, utt, mode in fp32_eager_audible_seam:
        print(f"    - {L} on {utt}")

    print()
    print("=" * 72)
    verdict_complete = (sorted(listeners_in_csv) == sorted(listeners_in_truth))
    if len(bf16_compile_audible_seam) == 0:
        if verdict_complete:
            print("VERDICT: PASS  (all listeners auditioned; zero audible_seam on bf16_compile)")
        else:
            print(
                f"PARTIAL VERDICT: PASS so far  "
                f"({len(listeners_in_csv)}/{len(listeners_in_truth)} listeners auditioned; "
                f"zero audible_seam on bf16_compile in available data)"
            )
    else:
        print(
            f"VERDICT: FAIL  "
            f"({len(bf16_compile_audible_seam)} audible_seam flag(s) on bf16_compile trials)"
        )
        print("Route to Open Question #2 per Story 18.4 Task 9.6.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
