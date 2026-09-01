"""Story 20.4 AC #5 - NFR3 perceptual audition helper (Windows-only).

Walks Commander through a blinded A/B audition of the chunk-size retune, on
the fixture ``20-4-regen-audition-fixture.py`` produces. Seven utterances,
three short / two medium / two long, all on the CLONED Sarira-F voice, all
rendered through the production TRUE_STREAM path so the streamer's chunk
boundaries, the decoder's overlap-add and the consumer's crossfade are all
in the signal.

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
FIXTURE_DIR = ARTIFACTS_DIR / "20-4-perceptual-fixtures"
TRUTH_TABLE_PATH = FIXTURE_DIR / "_perlistener_truthtable.json"
CANONICAL_CSV = ARTIFACTS_DIR / "20-4-chunk-retune-audition.csv"

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


def _verdict(listener_id: str, truth) -> None:
    """Unblind and print the verdict once every utterance is recorded."""
    if not CANONICAL_CSV.exists():
        return
    rows = [
        r for r in csv.DictReader(CANONICAL_CSV.open(newline="", encoding="utf-8"))
        if r.get("listener_id") == listener_id
    ]
    if not rows:
        return
    print("\n=== UNBLINDED VERDICT ===")
    blocking = []
    shared = []
    for row in rows:
        utt = row["utterance_id"]
        entry = truth[listener_id].get(utt)
        if not entry:
            continue
        by_geom = {
            entry["trial_A_geometry"]: row["a_defects_observed"],
            entry["trial_B_geometry"]: row["b_defects_observed"],
        }
        cs10, cs25 = by_geom.get("cs10", "none"), by_geom.get("cs25", "none")
        print("  {:<7} cs25(pre-20.4)={:<24} cs10(committed)={}".format(
            utt, cs25, cs10))
        if cs10 in BLOCKING_DEFECTS:
            (shared if cs25 in BLOCKING_DEFECTS else blocking).append(
                (utt, cs10, cs25)
            )
    print()
    if blocking:
        print("  VERDICT: FAIL - chunk-boundary defect on the COMMITTED")
        print("  geometry that the pre-20.4 geometry does not carry:")
        for utt, cs10, cs25 in blocking:
            print("    {}: cs10={} vs cs25={}".format(utt, cs10, cs25))
        print("  AC #5 makes this BLOCKING. Do not close the story.")
    elif shared:
        print("  VERDICT: PASS with a pre-existing finding.")
        print("  These utterances carry the same defect class on BOTH")
        print("  geometries, so it predates Story 20.4 - record it, raise it")
        print("  separately, do not block on it:")
        for utt, cs10, cs25 in shared:
            print("    {}: cs10={} cs25={}".format(utt, cs10, cs25))
    else:
        print("  VERDICT: PASS - no chunk-boundary defect flagged on the")
        print("  committed geometry across {} utterances.".format(len(rows)))
    prefs = [r["a_or_b_preferred"] for r in rows]
    print("\n  Preference tally (blinded labels): " + ", ".join(
        "{}={}".format(p, prefs.count(p)) for p in PREFERENCE_VOCAB))
    print("  (Preference is informational. The gate is the defect column.)")


def main(listener_id: str) -> int:
    if not TRUTH_TABLE_PATH.exists():
        print("FATAL: truth-table not found at {}".format(TRUTH_TABLE_PATH),
              file=sys.stderr)
        print("Run 20-4-regen-audition-fixture.py first.", file=sys.stderr)
        return 2
    truth = json.loads(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))
    if listener_id not in truth:
        print("FATAL: listener_id {!r} not in truth-table. Known: {}".format(
            listener_id, sorted(truth)), file=sys.stderr)
        return 2

    listener_block = truth[listener_id]
    order = sorted(listener_block)
    already_done = _existing_rows_for_listener(listener_id)

    print()
    print("=== Story 20.4 chunk-size retune audition ===")
    print("Listener id: {}".format(listener_id))
    print("Total utterances: {}".format(len(order)))
    if already_done:
        print("Already recorded: {} -- will skip.".format(sorted(already_done)))
    print()
    print("What you are listening FOR:")
    print("  This change moves the streamer's chunk boundary from 30 frames")
    print("  to 15, so every generation is stitched together from twice as")
    print("  many pieces. The defect class is therefore at the SEAMS:")
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
    _verdict(listener_id, truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "L1"))
