"""Story 20.8 AC #3 - NFR3 perceptual audition helper (Windows-only).

Walks Commander through a blinded A/B audition of the chunk-size retune.

    reference = what ships today = chunk_size 25
    candidate = this change      = chunk_size 10

Everything else is identical on both arms: codec state caching (Story 20.5),
the retired lookahead (Story 20.6), the gated 0-sample consumer crossfade.
THE ONLY VARIABLE IS HOW MANY PIECES THE STREAM IS CUT INTO.

Sixteen trials: seven utterances - the same seven as Story 20.4 rounds 1-4 and
Stories 20.5/20.6, so every round in the epic stays comparable - x two
independent takes, plus two byte-identical control trials.

WHY THIS ROUND CAN REUSE ONE TALKER RUN, WHEN STORY 20.4's COULD NOT
--------------------------------------------------------------------
Story 20.4 SS17 recorded that a chunk-size story could not reuse tokens
because "chunk size perturbs the talker", and its arms were therefore
different takes - which is precisely why its round 4 was unresolvable: the
SAME configuration flagged differently across two takes.

Story 20.8 SS7 tested that claim instead of repeating it, and it is false in
the form that matters. At a fixed seed, live cs25 / cs10 / cs7 runs emit
BIT-IDENTICAL token streams (4/4 comparisons, two seeds), and an offline
re-chunk rendered through the real decoder worker and the real consumer buffer
is bit-for-bit what the live run at that geometry produces. So both files in
every pair here come from ONE talker run: same words, same prosody, same
pauses, same duration, to the sample.

What SS7 also found - and what shapes this fixture - is that a build at a
different chunk size reads a different compile-cache directory and draws a
different (though equally valid) stream. So the capture was taken in a process
running AT THE CANDIDATE geometry. The candidate arm is therefore bit-for-bit
what a shipped cs10 build emits; the reference arm is cs25 geometry over that
same content, which is the correct content-constant control.

If the two files sound identical, say ``equivalent``. Here that is the
PREDICTED answer, not a cop-out.

THE PREDICTION, RECORDED BEFORE THE ROUND (AC #3)
--------------------------------------------------
Written into ``20-8-chunk-size-reopen-evidence.md`` SS8.4 before this round was
generated, and repeated here so the helper and the evidence cannot drift.

  P1 (MAGNITUDE). ``equivalent`` is modal, >= 10 of 16. Story 20.5 removed the
     seam residual at the CAUSE - head NRMSE 0.406 -> 0.0078, lag jitter 0 on
     every seam, edge loss 0 - and Story 20.6's retirement removed the trim and
     the blend by construction, so each seam is now a state-continuous join
     rather than a splice between two independent renderings. The COUNT rises
     2.5x; the per-seam defect is the thing that was removed.
     FALSIFIED if <= 6 are equivalent.
  P2 (NO NEW HARM - BLOCKING, and it is the gate). No chunk-boundary defect on
     a candidate trial that its paired reference does not also carry. Both
     files in a pair are ONE take rendered two ways, so such a defect is caused
     by the geometry and nothing else. FALSIFIED by one.
  P3 (LOCATION). Anything audible is on the LONG fixtures, which carry the most
     seams. FALSIFIED if a difference is heard on a short fixture but not on a
     long one - that would say seam count is not the mechanism. Story 20.4 SS17
     falsified exactly this sub-prediction once already, which is why its
     "crossover at chunk_size ~ 20" estimate must not be quoted as measured.
  P4 (THE EMBARRASSING ONE). Story 20.4's four rounds found cs10 perceptually
     WORSE - blocking in rounds 2 and 3. The entire reopening rests on "Story
     20.5 removed the cause". If cs10 flags a blocking seam defect again here,
     that mechanism argument is WRONG: state caching did not remove what
     actually made cs10 worse, Story 20.5 SS2's offline numbers do not describe
     what the ear responds to, and the geometry question is CLOSED FOR GOOD -
     not retuned to cs15, closed. Stated first-class because it is the outcome
     that costs most.
  P5 (THE OTHER DIRECTION, also embarrassing). If the CANDIDATE is preferred on
     >= 4 trials, P1 is falsified in the exciting direction - and that needs
     explaining, not celebrating. Nothing in this epic predicts that smaller
     chunks IMPROVE audio; the likeliest reading would be that the reference's
     longer decode windows accumulate codec drift state caching does not fully
     remove, which reopens Story 20.5's conclusion rather than confirming it.
  P6. Latency is NOT under test. These are rendered files; nothing about TTFA
     is auditionable here. The GUI capture is the only evidence for that.

  The two control trials (``ctl-020``, both takes) are BYTE-IDENTICAL: each is
  one arm rendered twice, asserted identical at generation time. A preference
  or a defect reported there is a property of the listening, and it calibrates
  everything else in the round.

VERDICT GATE (blocking, not advisory):
    FAIL if any chunk-boundary artefact is flagged on a candidate trial that
    the paired reference does not also carry. A defect flagged on BOTH
    renditions is upstream of the geometry - and here it demonstrably is,
    because both files come from the same take - so it is recorded rather than
    blocking.

WHAT HAPPENS NEXT DEPENDS ON THE SCORE, AND THAT IS DECIDED ALREADY
-------------------------------------------------------------------
Recorded in evidence SS8.3 before the round so the decision is not made by
drift:
  * clean pass (0 blocking, reference preferred on <= 1 of 14) -> cs7 is worth
    ONE more round;
  * pass but not clean (reference preferred on >= 2)          -> ship cs10,
    test nothing further;
  * FAIL                                                       -> fall back to
    cs15, one round, then stop regardless;
  * a control trial flags                                      -> discard the
    round and re-run rested.

Blinding: the script never prints which arm is playing. The truth table
assigns A/B balanced 8/8 from a fixed seed, reproducible from the generator
but not inferable from listening order.

Re-running is safe: trials already recorded for the same listener are skipped.
To restart, delete those rows from the CSV first.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-l1-audition-helper.py [L1]

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
    "r1": ("20-8-perceptual-fixtures", "20-8-chunksize-audition.csv"),
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
        "  ONLY the chunk size. One arm cuts the stream into 25-frame pieces",
        "  (2.0 s each) and the other into 10-frame pieces (0.8 s each), so",
        "  the second has about 2.5x as many joins. BOTH arms carry codec",
        "  state caching, the retired lookahead and the same gated consumer",
        "  crossfade.",
        "",
        "  Why more joins should NOT be worse now: before Story 20.5 each",
        "  chunk was decoded from a COLD codec state and the seam was a",
        "  splice between two independent renderings, masked by a blend.",
        "  That is what made chunk_size 10 fail two auditions in Story 20.4.",
        "  The decoder now carries its real state across the boundary, so a",
        "  join is a continuation rather than a splice - measured at 0.0078",
        "  NRMSE and zero lag jitter, against 0.406 before.",
        "",
        "  So the expected answer on most trials is 'equivalent'. What this",
        "  round is really asking is whether that measurement is right about",
        "  what you can HEAR - twice in this epic it was not, and the last",
        "  time this exact geometry was auditioned, the ear said no.",
        "",
        "  Both files in a pair are the SAME generation, re-cut two ways.",
        "  The words, the timing and the delivery are identical to the",
        "  sample. Any difference you hear is the cutting.",
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
    print("    P1 equivalent >= 10 of {:<2}               : {}".format(
        total,
        "HELD" if equal >= 10 else
        "FALSIFIED ({} equivalent, {} candidate-preferred, {} "
        "reference-preferred)".format(equal, cand_pref, ref_pref)))
    print("    P2 no candidate-only defect (BLOCKING)  : {}".format(
        "HELD" if not blocking else
        "FALSIFIED - {}".format(sorted(t for t, _, _ in blocking))))
    print("    P4 cs10 does not flag a blocking seam   : {}".format(
        "HELD" if not blocking else
        "FALSIFIED - {}. This is the outcome that says the DIAGNOSIS is "
        "wrong: Story 20.5 did not remove what made cs10 worse, its offline "
        "seam numbers do not describe what the ear responds to, and the "
        "geometry question is CLOSED FOR GOOD - do not retune to cs15, "
        "close it.".format(sorted(t for t, _, _ in blocking))))
    print("    P5 candidate NOT preferred on >= 4      : {}".format(
        "HELD" if cand_pref < 4 else
        "FALSIFIED - candidate preferred on {} trials. Smaller chunks "
        "IMPROVING audio is not predicted by anything in this epic; it "
        "needs explaining (likeliest: the reference's longer decode "
        "windows accumulate drift state caching does not fully remove, "
        "which reopens Story 20.5's conclusion) before shipping."
        .format(cand_pref)))
    if control:
        print("\n  Control calibration (ctl-020 is BYTE-IDENTICAL - one arm "
              "rendered twice):")
        print("    preference recorded on the control      : {}".format(
            "equivalent on both, as expected" if not control_nonequiv else
            "A PREFERENCE WAS EXPRESSED on {} - the two files are the same "
            "bytes, so the round's noise floor is above its signal. Evidence "
            "SS8.3 says DISCARD the round and re-run rested; do not re-scope "
            "it.".format(sorted(control_nonequiv))))
        print("    defects recorded on the control         : {}".format(
            "none" if not control_defects else
            "flagged on {} - a defect heard on identical files sets the "
            "round's noise floor".format(sorted(control_defects))))
    print("\n  P3 (location) and P6 (latency) are read by hand from the notes:")
    print("    P3 expects any difference on the LONG fixtures, which carry")
    print("       the most seams. A difference on a SHORT fixture but not a")
    print("       long one falsifies it - seam count would not be the")
    print("       mechanism, and Story 20.4 SS17 already falsified this same")
    print("       sub-prediction once.")
    print("    P6 latency is not auditionable in rendered files; the GUI")
    print("       capture is the only evidence for TTFA.")
    print("\n  WHAT HAPPENS NEXT (decided in evidence SS8.3, before the round):")
    if blocking:
        print("    FAIL -> cs10 does not ship. Fall back to cs15: one round,")
        print("    then stop regardless of its outcome.")
    elif ref_pref >= 2:
        print("    PASS but not clean ({} reference-preferred) -> ship cs10 "
              "and".format(ref_pref))
        print("    test nothing further. cs7 has MORE seams; a round on it")
        print("    would be spent confirming a worse point.")
    else:
        print("    CLEAN PASS -> the seam lever is not binding at 24 seams,")
        print("    so cs7 (34 seams, a further -137 ms) is worth ONE more")
        print("    round. That is the only condition under which it is.")
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
        print("Run 20-8-regen-audition-fixture.py first.", file=sys.stderr)
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
    print("=== Story 20.8 chunk-size audition (cs25 vs cs10) — round {} ===".format(
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
