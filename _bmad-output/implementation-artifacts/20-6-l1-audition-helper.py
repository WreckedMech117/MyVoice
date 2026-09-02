"""Story 20.6 AC #4 - NFR3 perceptual audition helper (Windows-only).

Walks Commander through a blinded A/B audition of retiring the 5-frame
lookahead.

    reference = what ships today  = codec state caching + the gated (0-sample)
                                    consumer crossfade + lookahead 5
    candidate = this change       = the same, with the lookahead RETIRED

Sixteen trials: seven utterances (the same seven as Story 20.4 rounds 1-4 and
Story 20.5 rounds 1-2, so every round in the epic stays comparable) plus a
zero-seam control, x two independent takes.

ONE VARIABLE, AND THE BLEND IS NOT A SECOND ONE
------------------------------------------------
Both arms carry codec state caching, chunk_size 25, and the same gated
consumer crossfade. The only difference is whether the streamer emits 30-frame
windows at stride 25 - which the worker then trims to the splice and
cross-fades over 1024 samples - or 25-frame windows at stride 25, which it
posts whole.

The Story 20.4 seam blend is a DEPENDENT of the lookahead, not a second
variable: it cross-fades the retained lookahead tail into the next chunk's
head, and with no lookahead there is no tail. Story 20.5 had already measured
it as inert under carried state (the two sides of the blend were bit-identical
on the reference decoder), so this round is not trading it away for anything.

TOKEN REUSE IS STILL AVAILABLE - VERIFIED, NOT ASSUMED
-------------------------------------------------------
Both files in every pair are decoded from ONE talker run. The chunking change
is downstream of generation: the talker's token sequence does not depend on
how the streamer slices it, so the fixture captures one run per pair, recovers
the flat frame sequence, and re-slices it at each arm's geometry. Wording,
prosody, pauses and total duration are identical to the sample. There is no
take-to-take variance to average over WITHIN a pair; the second take samples
the CONTENT lottery - whether a held vowel or a plosive lands on a boundary.

If the two files sound identical, say ``equivalent``. Here that is the
PREDICTED answer, not a cop-out.

THE PREDICTION, RECORDED BEFORE THE ROUND (AC #4)
--------------------------------------------------
Written into ``20-6-retire-the-lookahead-evidence.md`` before this round was
run, and repeated here so the helper and the evidence cannot drift.

  P1 (MAGNITUDE). ``equivalent`` is modal and near-total, >= 12 of 16. The
     trim removed exactly the lookahead's worth of PCM and the blend was an
     identity, so the candidate should post the audio the reference does.
     Measured offline: identical length on 16/16 pairs, worst level delta
     0.005 dB, median waveform difference -70 dB. FALSIFIED if <= 8 are
     equivalent - a far larger effect than -70 dB can produce, which would
     mean something else moved.
  P2 (NO NEW HARM, BLOCKING). No chunk-boundary defect on any candidate trial
     that its paired reference does not also carry. FALSIFIED by one.
  P3 (LOCATION). Where the two arms differ at all it is because the residual
     flush splits differently, so any audible difference should be at the END
     of an utterance. The measured outlier is ``s-021-t2`` (-42.8 dB, the one
     pair where the reference decoded the whole utterance as a single residual
     and the candidate split it 25 + 4). FALSIFIED if differences are reported
     at INTERIOR seams instead.
  P4 (THE EMBARRASSING ONE). If the candidate is heard as WORSE AT THE
     INTERIOR SEAMS, the diagnosis is wrong: it would mean the Story 20.4
     blend was doing real perceptual work under carried state after all, that
     Story 20.5's "the two sides are bit-identical" measurement does not
     describe what ships, and that the retirement has to be reverted rather
     than tuned. Stated first-class because it is the outcome that costs most.
  P5. Latency is NOT under test. These are rendered files; nothing about TTFA
     is auditionable here. The GUI capture is the only evidence for that.

  The zero-seam control (``ctl-020``, both takes) is BYTE-IDENTICAL between
  arms. A preference or a defect reported there is a property of the
  listening, and it calibrates everything else in the round.

VERDICT GATE (blocking, not advisory):
    FAIL if any chunk-boundary artefact is flagged on a candidate trial that
    the paired reference does not also carry. A defect flagged on BOTH
    renditions is a pre-existing property of the pipeline - and here it
    genuinely is, because both files come from the same take - so it is
    recorded rather than blocking.

Blinding: the script never prints which arm is playing. The truth table
randomises A/B per trial from a fixed seed, reproducible from the generator
but not inferable from listening order.

Re-running is safe: trials already recorded for the same listener are skipped.
To restart, delete those rows from the CSV first.

Usage:
    python310\python.exe _bmad-output\implementation-artifacts\20-6-l1-audition-helper.py [L1]

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
# result is recorded in 20-5-phase2-evidence.md and must stay reproducible.
ROUNDS = {
    "r1": ("20-6-perceptual-fixtures", "20-6-lookahead-audition.csv"),
}
DEFAULT_ROUND = "r1"

FIXTURE_DIR = ARTIFACTS_DIR / ROUNDS[DEFAULT_ROUND][0]
CANONICAL_CSV = ARTIFACTS_DIR / ROUNDS[DEFAULT_ROUND][1]

# Per-round listening guidance. Round 1 asked "does carrying codec state
# change what you hear"; round 2 asks "does removing the consumer crossfade
# fix the two rows where it did".
ROUND_BRIEF = {
    "r1": [
        "What is different between the two files:",
        "  ONLY the 5-frame lookahead. BOTH arms carry codec state caching",
        "  -- that is settled -- the same chunk size (25) and the same gated",
        "  consumer crossfade. One arm decodes 30 frames per chunk and throws",
        "  the last 5 away after cross-fading them into the next chunk; the",
        "  other decodes 25 and posts all of it.",
        "",
        "  Why that should be inaudible: the trim removed exactly what the",
        "  extra 5 frames added, and with codec state carried across the",
        "  boundary the cross-fade was blending a signal with a bit-identical",
        "  copy of itself. Offline the two arms measure about -70 dB apart,",
        "  with identical length on every pair.",
        "",
        "  So the expected answer on most trials is 'equivalent'. What this",
        "  round is really asking is whether that offline measurement is",
        "  right about what you can HEAR -- twice in this epic it was not.",
        "",
        "  Two trials (ctl-020) are byte-identical on purpose. They are not a",
        "  trick; they calibrate the rest of the round.",
    ],
}

# Story 17.1's controlled vocabulary, unchanged from Story 20.4 so the two
# stories' results are directly comparable.
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
    "trial_id",
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
                seen.add(row["trial_id"])
    return seen


def _verdict(listener_id: str, truth, meta) -> None:
    if not CANONICAL_CSV.exists():
        return
    with CANONICAL_CSV.open(newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp)
                if r.get("listener_id") == listener_id]
    if not rows:
        return
    cand, ref = meta["candidate"], meta["reference"]
    print("\n=== UNBLINDED VERDICT ===")
    print("  candidate = {} ({})".format(cand, meta.get("candidate_desc", "")))
    print("  reference = {} ({})".format(ref, meta.get("reference_desc", "")))
    print()

    blocking, shared = [], []
    cand_pref = ref_pref = equal = 0
    long_diff = short_diff = 0
    for row in rows:
        trial = row["trial_id"]
        entry = truth[listener_id].get(trial)
        if not entry:
            continue
        by_arm = {
            entry["trial_A_arm"]: row["a_defects_observed"],
            entry["trial_B_arm"]: row["b_defects_observed"],
        }
        c, r_ = by_arm.get(cand, "none"), by_arm.get(ref, "none")
        print("  {:<12} reference={:<26} candidate={}".format(trial, r_, c))
        if c in BLOCKING_DEFECTS:
            (shared if r_ in BLOCKING_DEFECTS else blocking).append((trial, c, r_))

        pref = row["a_or_b_preferred"]
        if pref == "equivalent":
            equal += 1
        else:
            arm = entry["trial_A_arm"] if pref == "A" else entry["trial_B_arm"]
            if arm == cand:
                cand_pref += 1
            else:
                ref_pref += 1
            if trial.startswith("l-"):
                long_diff += 1
            else:
                short_diff += 1

    print()
    if blocking:
        print("  VERDICT: FAIL — a chunk-boundary defect on the CANDIDATE that")
        print("  the paired reference does not carry. Both files in a pair are")
        print("  the SAME take, so this is caused by the decode:")
        for trial, c, r_ in blocking:
            print("    {}: candidate={} vs reference={}".format(trial, c, r_))
        print("  AC #4 makes this BLOCKING. Do not close the story.")
    elif shared:
        print("  VERDICT: PASS with a pre-existing finding.")
        print("  These trials carry the same defect class on BOTH arms. They")
        print("  are the same take, so the defect is upstream of the decode —")
        print("  record it, raise it separately, do not block on it:")
        for trial, c, r_ in shared:
            print("    {}: candidate={} reference={}".format(trial, c, r_))
    else:
        print("  VERDICT: PASS — no chunk-boundary defect flagged on the")
        print("  candidate across {} trials.".format(len(rows)))

    total = cand_pref + ref_pref + equal
    print("\n  Preference (unblinded): candidate={} reference={} equivalent={}"
          .format(cand_pref, ref_pref, equal))
    control = [t for t in (r["trial_id"] for r in rows)
               if t.startswith("ctl-")]
    control_nonequiv = [
        r["trial_id"] for r in rows
        if r["trial_id"].startswith("ctl-")
        and r["a_or_b_preferred"] != "equivalent"
    ]
    control_defects = [
        r["trial_id"] for r in rows
        if r["trial_id"].startswith("ctl-")
        and (r["a_defects_observed"] != "none"
             or r["b_defects_observed"] != "none")
    ]
    interior = [t for t, _, _ in blocking if not t.startswith("ctl-")]

    print("\n  Against the prediction recorded BEFORE the round:")
    print("    P1 equivalent >= 12 of {:<2}               : {}".format(
        total,
        "HELD" if equal >= 12 else
        "FALSIFIED ({} equivalent, {} candidate-preferred, {} "
        "reference-preferred)".format(equal, cand_pref, ref_pref)))
    print("    P2 no candidate-only defect (BLOCKING)  : {}".format(
        "HELD" if not blocking else
        "FALSIFIED - {}".format(sorted(t for t, _, _ in blocking))))
    print("    P4 no candidate-only INTERIOR seam harm : {}".format(
        "HELD" if not interior else
        "FALSIFIED - {} — this is the outcome that says the diagnosis is "
        "wrong. The Story 20.4 blend was doing real work under carried "
        "state; REVERT the retirement rather than tune it, and re-open "
        "Story 20.5's Phase 4 conclusion.".format(sorted(interior))))
    if control:
        print("\n  Control calibration (ctl-020 is byte-identical between arms):")
        print("    preference recorded on the control      : {}".format(
            "equivalent on both, as expected" if not control_nonequiv else
            "A PREFERENCE WAS EXPRESSED on {} — the two files are the same "
            "bytes, so read every other preference in this round in that "
            "light".format(sorted(control_nonequiv))))
        print("    defects recorded on the control         : {}".format(
            "none" if not control_defects else
            "flagged on {} — a defect heard on identical files sets the "
            "round's noise floor".format(sorted(control_defects))))
    print("\n  P3 (location) and P5 (latency) are read by hand from the notes:")
    print("    P3 expects any difference at the END of an utterance, where")
    print("       the residual flush splits differently — s-021-t2 is the")
    print("       measured outlier at -42.8 dB. A difference reported at an")
    print("       INTERIOR seam instead falsifies it.")
    print("    P5 latency is not auditionable in rendered files; the GUI")
    print("       capture is the only evidence for TTFA.")
    return


def main(listener_id: str = "L1", round_id: str = DEFAULT_ROUND) -> int:
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
        print("FATAL: truth table not found at {}".format(truth_path),
              file=sys.stderr)
        print("Run 20-6-regen-audition-fixture.py first.", file=sys.stderr)
        return 2
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    meta = truth["_meta"]
    if listener_id not in truth:
        print("FATAL: listener_id {!r} not in truth table. Known: {}".format(
            listener_id, [k for k in truth if k != "_meta"]), file=sys.stderr)
        return 2

    block = truth[listener_id]
    order = sorted(block)
    already = _existing_rows_for_listener(listener_id)

    print()
    print("=== Story 20.6 lookahead-retirement audition — round {} ===".format(
        meta.get("round", 1)))
    print("Listener id: {}".format(listener_id))
    print("Trials: {} ({} utterances x {} takes)".format(
        len(order), len(order) // meta.get("takes_per_utterance", 1),
        meta.get("takes_per_utterance", 1)))
    if already:
        print("Already recorded: {} -- will skip.".format(sorted(already)))
    print()
    for line in ROUND_BRIEF.get(round_id, ROUND_BRIEF["r1"]):
        print(line)
    print()
    print("  Both files in a pair come from the SAME generation. The words,")
    print("  the timing and the delivery are identical to the sample. If they")
    print("  sound the same, they may genuinely BE the same to within your")
    print("  ear — 'equivalent' is an expected answer here, not a cop-out.")
    print()
    print("What you are listening FOR — at the seams, roughly every 2 s:")
    print("  - a click or tick partway through a word")
    print("  - a momentary discontinuity or 'stutter' in a held vowel")
    print("  - prosody that resets mid-phrase, as if two takes were cut")
    print("    together")
    print("  - a smeared or 'phasey' consonant at a boundary")
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

        for idx, trial in enumerate(order, start=1):
            if trial in already:
                print("[{}/{}] {} -- already recorded, skipping.".format(
                    idx, len(order), trial))
                continue
            entry = block[trial]
            a_path = FIXTURE_DIR / entry["trial_A_filename"]
            b_path = FIXTURE_DIR / entry["trial_B_filename"]
            if not a_path.exists() or not b_path.exists():
                print("FATAL: missing WAV(s) for {}: A={} (exists={}), "
                      "B={} (exists={})".format(
                          trial, a_path.name, a_path.exists(),
                          b_path.name, b_path.exists()), file=sys.stderr)
                return 2

            print("\n[{}/{}] Trial {}".format(idx, len(order), trial))
            while True:
                _play(a_path, "trial A")
                _play(b_path, "trial B")
                if not _prompt_replay_or_continue():
                    break

            pref = _prompt_choice("a_or_b_preferred", PREFERENCE_VOCAB)
            a_def = _prompt_choice("a_defects_observed", DEFECT_VOCAB)
            b_def = _prompt_choice("b_defects_observed", DEFECT_VOCAB)
            notes = input("  free_text_notes (optional, Enter to skip): ").strip()
            writer.writerow((trial, listener_id, pref, a_def, b_def, notes))
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
