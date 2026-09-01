"""Story 20.4 AC #5 - NFR3 perceptual audition helper (Windows-only).

Walks Commander through a blinded A/B audition of the chunk-size retune.
Seven utterances, three short / two medium / two long, all on the CLONED
Sarira-F voice, all rendered through the production TRUE_STREAM path so the
streamer's chunk boundaries, the decoder's overlap-add and the consumer's
crossfade are all in the signal.

ROUNDS
------
Round 1 (``20-4-perceptual-fixtures/``) FAILED: m-020 was clean at
chunk_size=25 and carried ``tonal_distortion`` at chunk_size=10. Its
fixture, truth table and results CSV are preserved unchanged - that round
is a recorded result, and round 2 reuses its cs25 files verbatim as the
reference arm so the two rounds share a calibration anchor.

Round 2 (``20-4-perceptual-fixtures-r2/``) auditioned the seam fix at
chunk_size=10 and FAILED worse than round 1 - three blocking rows, and the
defect class changed from tonal_distortion to click_or_discontinuity. It
also conflated two variables (geometry AND stitching), so its clicks could
not be attributed.

Round 3 (``20-4-perceptual-fixtures-r3/``, the default) isolates. BOTH arms
are chunk_size=25; the only difference is the stitching:
    reference = shipped pre-fix   candidate = with the seam fix
The seam fix is geometry-independent and changes the SHIPPING cs25 path, so
this round answers whether it helps or harms what users hear today.

Which side of each pair is which is recorded in the truth table ``_meta``
block, and the helper only consults it when unblinding at the end.

Pass a round as the second CLI argument (``r1``/``r2``) to re-run an
earlier round.

Commander solo is the protocol AC #5 specifies - it mirrors the Story
18.1 / 18.2 discipline, not Story 17.1's three-listener packet. This
change alters chunk stitching on every generation, which is a defect class
a single trained listener can adjudicate; it does not alter timbre or
dynamics, which is what needed multiple ears in 17.1.

VERDICT GATE (AC #5, and it is BLOCKING, not advisory):
    FAIL if any chunk-boundary artefact is flagged on a B/cs10 trial -
    clicks, discontinuities, or altered prosody at stitch points. A defect
    flagged on BOTH renditions is a pre-existing property of the pipeline,
    not a Story 20.4 regression, and is recorded as such rather than
    blocking.

Blinding: the script does not print which geometry is playing. The
truth-table in the fixture directory randomises A/B per utterance from a
fixed seed, so the mapping is reproducible from the generator but not
inferable from listening order.

Re-running is safe: rows already recorded for the same listener are
skipped. To restart, delete the rows from the CSV first.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-l1-audition-helper.py [L1]

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import csv
import json
import sys
import winsound
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent

# Round -> (fixture dir, results CSV). Round 1's entries are frozen: its
# result is recorded in the Story 20.4 evidence file and must stay
# reproducible.
ROUNDS = {
    "r1": ("20-4-perceptual-fixtures", "20-4-chunk-retune-audition.csv"),
    "r2": ("20-4-perceptual-fixtures-r2", "20-4-chunk-retune-audition-r2.csv"),
    "r3": ("20-4-perceptual-fixtures-r3", "20-4-chunk-retune-audition-r3.csv"),
}
DEFAULT_ROUND = "r3"

# Fallback for round 1, whose truth table predates the ``_meta`` block.
_LEGACY_META = {
    "round": 1,
    "candidate": "cs10",
    "reference": "cs25",
    "candidate_desc": "chunk_size=10 (pre-fix stitching)",
    "reference_desc": "chunk_size=25 (pre-fix stitching)",
}

# Story 17.1's controlled vocabulary, with the two entries this story's
# defect class needs made explicit. ``audible_seam`` remains the primary
# blocking finding.
DEFECT_VOCAB = (
    "none",
    "audible_seam",
    "click_or_discontinuity",
    "prosody_break_at_stitch",
    "clipping",
    "phase_artifact",
    "tonal_distortion",
    "other_describe_in_notes",
)
BLOCKING_DEFECTS = (
    "audible_seam",
    "click_or_discontinuity",
    "prosody_break_at_stitch",
)
PREFERENCE_VOCAB = ("A", "B", "equivalent")

CSV_HEADER = (
    "utterance_id",
    "listener_id",
    "a_or_b_preferred",
    "a_defects_observed",
    "b_defects_observed",
    "free_text_notes",
)


def _play(path: Path, label: str) -> None:
    print("  >> Playing {} ...".format(label))
    winsound.PlaySound(str(path), winsound.SND_FILENAME)
    print("  -- {} done.".format(label))


def _prompt_choice(label: str, valid) -> str:
    while True:
        raw = input("  {} (one of: {}): ".format(label, ", ".join(valid))).strip()
        if raw in valid:
            return raw
        print("    INVALID -- must be exactly one of: {}".format(", ".join(valid)))


def _prompt_replay_or_continue() -> bool:
    while True:
        raw = input(
            "  [r] replay both, [Enter] continue to entry, [q] quit: "
        ).strip().lower()
        if raw == "q":
            print("\n  Aborted by user. Rows already recorded are saved.")
            sys.exit(0)
        if raw == "r":
            return True
        if raw == "":
            return False
        print("    Use [r], [Enter], or [q].")


def _existing_rows_for_listener(listener_id: str):
    if not CANONICAL_CSV.exists():
        return set()
    seen = set()
    with CANONICAL_CSV.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("listener_id") == listener_id:
                seen.add(row["utterance_id"])
    return seen


def _verdict(listener_id: str, truth, meta) -> None:
    """Unblind and print the verdict once every utterance is recorded."""
    if not CANONICAL_CSV.exists():
        return
    rows = [
        r for r in csv.DictReader(CANONICAL_CSV.open(newline="", encoding="utf-8"))
        if r.get("listener_id") == listener_id
    ]
    if not rows:
        return
    cand, ref = meta["candidate"], meta["reference"]
    print("\n=== UNBLINDED VERDICT ===")
    print("  candidate = {} ({})".format(cand, meta.get("candidate_desc", "")))
    print("  reference = {} ({})".format(ref, meta.get("reference_desc", "")))
    print()
    blocking, shared = [], []
    for row in rows:
        utt = row["utterance_id"]
        entry = truth[listener_id].get(utt)
        if not entry:
            continue
        by_geom = {
            entry["trial_A_geometry"]: row["a_defects_observed"],
            entry["trial_B_geometry"]: row["b_defects_observed"],
        }
        c, r_ = by_geom.get(cand, "none"), by_geom.get(ref, "none")
        print("  {:<7} reference={:<26} candidate={}".format(utt, r_, c))
        if c in BLOCKING_DEFECTS:
            (shared if r_ in BLOCKING_DEFECTS else blocking).append((utt, c, r_))
    print()
    if blocking:
        print("  VERDICT: FAIL - chunk-boundary defect on the CANDIDATE that")
        print("  the reference does not carry:")
        for utt, c, r_ in blocking:
            print("    {}: candidate={} vs reference={}".format(utt, c, r_))
        print("  AC #5 makes this BLOCKING. Do not close the story.")
    elif shared:
        print("  VERDICT: PASS with a pre-existing finding.")
        print("  These utterances carry the same defect class on BOTH arms,")
        print("  so it is not something this change introduced - record it,")
        print("  raise it separately, do not block on it:")
        for utt, c, r_ in shared:
            print("    {}: candidate={} reference={}".format(utt, c, r_))
    else:
        print("  VERDICT: PASS - no chunk-boundary defect flagged on the")
        print("  candidate across {} utterances.".format(len(rows)))
    prefs = [r["a_or_b_preferred"] for r in rows]
    print("\n  Preference tally (blinded labels): " + ", ".join(
        "{}={}".format(p, prefs.count(p)) for p in PREFERENCE_VOCAB))
    print("  (Preference is informational. The gate is the defect column.)")


def main(listener_id: str, round_id: str = DEFAULT_ROUND) -> int:
    global FIXTURE_DIR, CANONICAL_CSV
    if round_id not in ROUNDS:
        print("FATAL: unknown round {!r}; known: {}".format(
            round_id, sorted(ROUNDS)), file=sys.stderr)
        return 2
    fixture_name, csv_name = ROUNDS[round_id]
    FIXTURE_DIR = ARTIFACTS_DIR / fixture_name
    CANONICAL_CSV = ARTIFACTS_DIR / csv_name
    truth_path = FIXTURE_DIR / "_perlistener_truthtable.json"

    if not truth_path.exists():
        print("FATAL: truth-table not found at {}".format(truth_path),
              file=sys.stderr)
        print("Run 20-4-regen-audition-fixture-r2.py first.", file=sys.stderr)
        return 2
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    meta = truth.get("_meta", _LEGACY_META)
    if listener_id not in truth:
        print("FATAL: listener_id {!r} not in truth-table. Known: {}".format(
            listener_id, [k for k in truth if k != "_meta"]), file=sys.stderr)
        return 2

    listener_block = truth[listener_id]
    order = sorted(listener_block)
    already_done = _existing_rows_for_listener(listener_id)

    print()
    print("=== Story 20.4 chunk-size retune audition - round {} ===".format(
        meta.get("round", "?")))
    print("Listener id: {}".format(listener_id))
    print("Total utterances: {}".format(len(order)))
    if already_done:
        print("Already recorded: {} -- will skip.".format(sorted(already_done)))
    print()
    print("What you are listening FOR:")
    print("  Both arms are the SAME chunk size this time. The only thing")
    print("  that differs is how consecutive chunks are joined: one arm is")
    print("  what ships today, the other has the seam fix. Round 2 could not")
    print("  tell us which change caused its clicks because it moved two")
    print("  things at once; this round moves one.")
    print("  The defect class is still at the SEAMS:")
    print("    - a click or tick partway through a word")
    print("    - a momentary discontinuity or 'stutter' in a held vowel")
    print("    - prosody that resets mid-phrase, as if two takes were cut")
    print("      together")
    print("  Timbre, loudness and accent are NOT under test - the two takes")
    print("  are different samples, not the same waveform, so they will")
    print("  differ in wording rhythm. Judge the SEAMS.")
    print()
    print("Protocol:")
    print("  - Headphones if you have them; normal Discord-call volume.")
    print("  - Trial A end-to-end, then trial B end-to-end. [r] replays both.")
    print("  - Pick exactly one defect value per trial.")
    print("  - Any seam defect on either trial: describe WHERE in the notes.")
    print()
    input("Press Enter when ready to start...")

    write_header = not CANONICAL_CSV.exists()
    with CANONICAL_CSV.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(CSV_HEADER)
            fp.flush()

        for idx, utt in enumerate(order, start=1):
            if utt in already_done:
                print("[{}/{}] {} -- already recorded, skipping.".format(
                    idx, len(order), utt))
                continue
            entry = listener_block[utt]
            a_path = FIXTURE_DIR / entry["trial_A_filename"]
            b_path = FIXTURE_DIR / entry["trial_B_filename"]
            if not a_path.exists() or not b_path.exists():
                print("FATAL: missing WAV(s) for {}: A={} (exists={}), "
                      "B={} (exists={})".format(
                          utt, a_path.name, a_path.exists(),
                          b_path.name, b_path.exists()), file=sys.stderr)
                return 2

            print("\n[{}/{}] Utterance {}".format(idx, len(order), utt))
            while True:
                _play(a_path, "trial A")
                _play(b_path, "trial B")
                if not _prompt_replay_or_continue():
                    break

            pref = _prompt_choice("a_or_b_preferred", PREFERENCE_VOCAB)
            a_def = _prompt_choice("a_defects_observed", DEFECT_VOCAB)
            b_def = _prompt_choice("b_defects_observed", DEFECT_VOCAB)
            notes = input("  free_text_notes (optional, Enter to skip): ").strip()
            writer.writerow((utt, listener_id, pref, a_def, b_def, notes))
            fp.flush()
            print("  recorded.")

    print("\nAll rows recorded -> {}".format(CANONICAL_CSV))
    _verdict(listener_id, truth, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(
        sys.argv[1] if len(sys.argv) > 1 else "L1",
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROUND,
    ))
