"""
Story 17.1 audition helper (Windows-only).

Walks a listener through the perceptual A/B audition for all 10 utterances,
plays each pair via winsound, validates inputs against the controlled defect
vocabulary, and appends rows to 17-1-perceptual-ab-results.csv.

Defaults to LISTENER_ID=L1 (Commander). Pass a different listener id as a CLI
arg (e.g., `... 17-1-l1-audition-helper.py L2`) for an in-person audition with
another listener at the same machine. Remote listeners use the template CSV +
cover note instead.

Re-running is safe: rows already present in the canonical CSV for the same
listener are skipped automatically. To restart fresh, delete the existing rows
manually first.

Blinding: the script intentionally does NOT print the underlying WAV filename,
because the canonical disk naming (`*-A-true_stream.wav` / `*-B-sentence_stream.wav`)
embeds the actual mode and would defeat the per-listener truth-table
randomization. The listener only sees abstract `trial A` / `trial B` labels.

Working file — gitignored under `_bmad-output/`; NOT committed.
"""

from __future__ import annotations

import csv
import json
import sys
import winsound
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent
# Use the post-Story-16.8 regenerated fixture (audible TRUE_STREAM speech).
# The 16-7 dir's sample-A WAVs are 44-byte silent stubs from the pre-talker-fix
# bug (s-014-A 16-7=44 bytes, 16-8=106 KB after regeneration); the 16-8 dir is
# the canonical post-fix fixture. The story's `>` blockquote header at line 13
# implied an in-place regeneration but the actual disk layout has them as
# separate directories.
FIXTURE_DIR = ARTIFACTS_DIR / "16-8-perceptual-fixtures"
TRUTH_TABLE_PATH = FIXTURE_DIR / "_perlistener_truthtable.json"
CANONICAL_CSV = ARTIFACTS_DIR / "17-1-perceptual-ab-results.csv"

DEFECT_VOCAB = (
    "none",
    "audible_seam",
    "clipping",
    "phase_artifact",
    "tonal_distortion",
    "other_describe_in_notes",
)
PREFERENCE_VOCAB = ("A", "B", "equivalent")

UTTERANCE_ORDER = (
    "s-014", "s-015", "s-016", "s-017",
    "m-011", "m-012", "m-013", "m-014",
    "l-013", "l-014",
)

CSV_HEADER = (
    "utterance_id",
    "listener_id",
    "a_or_b_preferred",
    "a_defects_observed",
    "b_defects_observed",
    "free_text_notes",
)


def _play(path: Path, label: str) -> None:
    print(f"  >> Playing {label} ...")
    winsound.PlaySound(str(path), winsound.SND_FILENAME)
    print(f"  -- {label} done.")


def _prompt_choice(label: str, valid: tuple[str, ...]) -> str:
    while True:
        raw = input(f"  {label} (one of: {', '.join(valid)}): ").strip()
        if raw in valid:
            return raw
        print(f"    INVALID -- must be exactly one of: {', '.join(valid)}")


def _prompt_replay_or_continue() -> bool:
    """Return True if listener wants to replay; False to continue to entry."""
    while True:
        raw = input("  [r] replay both, [Enter] continue to entry, [q] quit: ").strip().lower()
        if raw == "q":
            print()
            print("  Aborted by user. Rows already recorded are saved.")
            sys.exit(0)
        if raw == "r":
            return True
        if raw == "":
            return False
        print("    Use [r], [Enter], or [q].")


def _existing_rows_for_listener(listener_id: str) -> set[str]:
    if not CANONICAL_CSV.exists():
        return set()
    seen: set[str] = set()
    with CANONICAL_CSV.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("listener_id") == listener_id:
                seen.add(row["utterance_id"])
    return seen


def main(listener_id: str) -> int:
    if not TRUTH_TABLE_PATH.exists():
        print(f"FATAL: truth-table not found at {TRUTH_TABLE_PATH}", file=sys.stderr)
        return 2
    truth = json.loads(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))
    if listener_id not in truth:
        print(
            f"FATAL: listener_id {listener_id!r} not in truth-table. Known keys: {sorted(truth.keys())}",
            file=sys.stderr,
        )
        print(
            "If you are a new organic listener (L4+), append a new top-level entry to "
            "_perlistener_truthtable.json mirroring the L1/L2/L3 shape (per AC #1 second-half "
            "append-only allowance) before re-running.",
            file=sys.stderr,
        )
        return 2

    listener_block = truth[listener_id]
    already_done = _existing_rows_for_listener(listener_id)

    print()
    print(f"=== Story 17.1 audition session ===")
    print(f"Listener id: {listener_id}")
    print(f"Total utterances: {len(UTTERANCE_ORDER)}")
    if already_done:
        print(f"Already recorded for {listener_id}: {sorted(already_done)} -- will skip.")
    print()
    print("Reminder (from LISTENING-INSTRUCTIONS.md):")
    print("  - Use headphones if you have them.")
    print("  - Comfortable Discord-call volume; do not crank past your normal level.")
    print("  - Listen to trial A end-to-end; then trial B end-to-end.")
    print("  - You can replay both before recording your answer.")
    print("  - The defect vocabulary is fixed -- pick exactly one per trial.")
    print()
    input("Press Enter when ready to start...")

    write_header = not CANONICAL_CSV.exists()
    with CANONICAL_CSV.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(CSV_HEADER)
            fp.flush()

        recorded_this_session = 0
        for idx, utt in enumerate(UTTERANCE_ORDER, start=1):
            if utt in already_done:
                print(f"[{idx}/{len(UTTERANCE_ORDER)}] {utt} -- already recorded, skipping.")
                continue
            entry = listener_block[utt]
            a_path = FIXTURE_DIR / entry["trial_A_filename"]
            b_path = FIXTURE_DIR / entry["trial_B_filename"]
            if not a_path.exists() or not b_path.exists():
                print(
                    f"FATAL: missing WAV(s) for {utt}: "
                    f"trial_A={a_path.name} (exists={a_path.exists()}), "
                    f"trial_B={b_path.name} (exists={b_path.exists()})",
                    file=sys.stderr,
                )
                return 2

            print()
            print(f"[{idx}/{len(UTTERANCE_ORDER)}] Utterance {utt}")
            while True:
                _play(a_path, "trial A")
                _play(b_path, "trial B")
                if not _prompt_replay_or_continue():
                    break

            pref = _prompt_choice("a_or_b_preferred", PREFERENCE_VOCAB)
            a_def = _prompt_choice("a_defects_observed", DEFECT_VOCAB)
            b_def = _prompt_choice("b_defects_observed", DEFECT_VOCAB)
            notes = input("  free_text_notes (optional, press Enter to skip): ").strip()

            writer.writerow([utt, listener_id, pref, a_def, b_def, notes])
            fp.flush()
            recorded_this_session += 1
            print(f"  OK -- row recorded for {utt}.")

    print()
    print("=== Session complete ===")
    print(f"Rows recorded this session: {recorded_this_session}")
    print(f"Canonical CSV: {CANONICAL_CSV}")
    print()
    print("Re-run the script if any utterances are still missing for this listener.")
    return 0


if __name__ == "__main__":
    listener_arg = sys.argv[1] if len(sys.argv) > 1 else "L1"
    raise SystemExit(main(listener_arg))
