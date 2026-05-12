"""Story 18.4 Task 9.2 helper — generate the per-listener truth-table for the joint audition fixture.

Reads the 20 WAVs in ``_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/``
(10 utterances × 2 renditions), verifies they all exist, and writes
``_perlistener_truthtable.json`` with L1 / L2 / L3 blocks. The truth-table
randomizes per-listener which actual mode (``fp32_eager`` or ``bf16_compile``)
lands as ``trial_A`` vs ``trial_B`` so each listener hears the A/B pair
in a different order — the canonical Story 17.1 blinding pattern.

Expected disk-naming for the 20 WAVs:
  - {utterance_id}-fp32_eager.wav    (the strict baseline; tts_precision=fp32, tts_compile=off)
  - {utterance_id}-bf16_compile.wav  (the production target; tts_precision=bf16, tts_compile=on)

For each of the 10 utterances (s-014 ... l-014), both files must exist.

Randomization seed: ``"Story 18.4 joint audition"`` + listener_id. This makes
the assignment reproducible — if the truth-table is lost or corrupted, this
script can regenerate it bit-identical.

Verdict gate (verbatim Story 17.1):
  PASS iff zero listeners flag ``audible_seam`` on any B (bf16+compile)
  trial across all 30 trials.

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ARTIFACTS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = ARTIFACTS_DIR / "18-4-perceptual-fixtures"
TRUTH_TABLE_PATH = FIXTURE_DIR / "_perlistener_truthtable.json"

UTTERANCE_IDS = (
    "s-014", "s-015", "s-016", "s-017",
    "m-011", "m-012", "m-013", "m-014",
    "l-013", "l-014",
)

ACTUAL_MODES = ("fp32_eager", "bf16_compile")
LISTENER_IDS = ("L1", "L2", "L3")
SEED_PREFIX = "Story 18.4 joint audition"


def _filename_for(utt: str, mode: str) -> str:
    return f"{utt}-{mode}.wav"


def _check_fixture_files() -> tuple[list[str], list[str]]:
    """Return (missing, found) WAV filenames relative to FIXTURE_DIR."""
    missing: list[str] = []
    found: list[str] = []
    for utt in UTTERANCE_IDS:
        for mode in ACTUAL_MODES:
            name = _filename_for(utt, mode)
            path = FIXTURE_DIR / name
            if path.exists() and path.stat().st_size > 0:
                found.append(name)
            else:
                missing.append(name)
    return missing, found


def _build_listener_block(listener_id: str) -> dict:
    """Build one listener's per-utterance block with deterministic A/B
    randomization seeded by listener_id."""
    rng = random.Random(f"{SEED_PREFIX}:{listener_id}")
    block: dict = {}
    for utt in UTTERANCE_IDS:
        # Coin-flip per utterance: True = A is fp32_eager, B is bf16_compile.
        # False = swap (A is bf16_compile, B is fp32_eager).
        a_is_fp32_eager = rng.random() < 0.5
        if a_is_fp32_eager:
            trial_a_mode = "fp32_eager"
            trial_b_mode = "bf16_compile"
        else:
            trial_a_mode = "bf16_compile"
            trial_b_mode = "fp32_eager"
        block[utt] = {
            "trial_A_filename": _filename_for(utt, trial_a_mode),
            "trial_A_actual_mode": trial_a_mode,
            "trial_B_filename": _filename_for(utt, trial_b_mode),
            "trial_B_actual_mode": trial_b_mode,
        }
    return block


def main() -> int:
    if not FIXTURE_DIR.exists():
        print(f"FATAL: fixture directory does not exist: {FIXTURE_DIR}", file=sys.stderr)
        print(
            "Create it first and populate with the 20 WAVs per "
            "Story 18.4 Task 9.2.",
            file=sys.stderr,
        )
        return 2

    missing, found = _check_fixture_files()
    print(f"Fixture WAVs present: {len(found)} / 20")
    if missing:
        print(f"Fixture WAVs MISSING ({len(missing)}):", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nGenerate the missing WAVs before re-running this script. "
            "Each utterance needs both renditions (fp32_eager + bf16_compile).",
            file=sys.stderr,
        )
        return 2

    print("All 20 WAVs present. Building truth-table...")
    truth: dict = {}
    for listener_id in LISTENER_IDS:
        truth[listener_id] = _build_listener_block(listener_id)
        # Quick summary per listener.
        a_modes = [v["trial_A_actual_mode"] for v in truth[listener_id].values()]
        fp32_count = a_modes.count("fp32_eager")
        bf16_count = a_modes.count("bf16_compile")
        print(
            f"  {listener_id}: trial_A is fp32_eager for {fp32_count} utterances, "
            f"bf16_compile for {bf16_count} utterances"
        )

    # Pretty-print so a maintainer can audit the truth-table by eye.
    TRUTH_TABLE_PATH.write_text(
        json.dumps(truth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {TRUTH_TABLE_PATH}")
    print("\nNext: run the audition.")
    print(f"  L1 (Commander): python310/python.exe {ARTIFACTS_DIR}/18-4-l1-audition-helper.py L1")
    print(f"  L2 (organic):   python310/python.exe {ARTIFACTS_DIR}/18-4-l1-audition-helper.py L2")
    print(f"  L3 (organic):   python310/python.exe {ARTIFACTS_DIR}/18-4-l1-audition-helper.py L3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
