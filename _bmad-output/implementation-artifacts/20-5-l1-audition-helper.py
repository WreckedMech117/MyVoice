"""Story 20.5 AC #4 — NFR3 perceptual audition helper (Windows-only).

Walks Commander through a blinded A/B audition of codec state caching.

    reference = cs25 + the Story 20.4 seam fix                <- ships today
    candidate = cs25 + the Story 20.4 seam fix + state caching

Fourteen trials: seven utterances (the same seven as Story 20.4 rounds 1-4,
so every round stays comparable) x two independent takes.

WHY THIS ROUND IS SHARPER THAN STORY 20.4'S FOUR
------------------------------------------------
Both files in every pair are decoded from **one talker run**. The wording,
the prosody, the pauses and the total duration are identical to the sample;
the only difference is whether the decoder carried codec state across the
chunk boundary. Story 20.4's arms could not do that — a chunk-size change
perturbs what the streamer emits and therefore what the talker samples, so
its two arms were necessarily different takes, and SS17 recorded the
consequence: the same configuration flagged differently across takes.

Here there is no take-to-take variance to average over *within* a pair. The
second take exists to sample the CONTENT lottery — whether a held vowel or a
plosive happens to land on a boundary — not to average arm variance.

So: judge the two files as two renderings of the same performance, because
that is exactly what they are. If they sound identical, say ``equivalent``;
that is a legitimate and expected answer here in a way it was not in round 4.

ROUND 2'S PREDICTION, RECORDED BEFORE LISTENING (AC #4)
--------------------------------------------------------
Written into ``20-5-phase2-evidence.md`` before this round's fixture was
generated, and repeated here so the helper and the evidence cannot drift.

Round 2 arms: reference = state caching + the 64-sample crossfade (round 1's
candidate, the configuration that produced the two blocking rows); candidate =
the same, crossfade neutralised to 0.

  Q1 (BLOCKING, the whole point). The two rows that blocked round 1 --
      ``m-020`` and ``s-020`` -- come back clean on the candidate. Neither
      carries a blocking seam defect that its paired reference does not.
      **Falsified if either still flags candidate-only.**
  Q2 (NO NEW HARM). No blocking seam defect on any candidate trial anywhere
      in the round. **Falsified by one.**
  Q3 (DIRECTION). The candidate is preferred at least as often as the
      reference. **Falsified if the reference is preferred on >= 4 of 14.**
  Q4 (MAGNITUDE). ``equivalent`` is modal, 7-12 of 14 -- slightly higher than
      round 1's 6, because the crossfade only touches 64 samples at each of
      0-9 boundaries (0.1 % of the timeline), so most trials should be
      indistinguishable. **Falsified either way**: >= 10 candidate-preferred
      is a larger effect than 0.1 % of samples should be able to produce and
      would mean something else moved; <= 4 equivalent means the arms are far
      more separable than a 2.7 ms window can explain.
  Q5 (LOCATION, the risky one). Round 1's blocking rows were SINGLE-SEAM
      trials, which is the opposite of where seam-density reasoning would put
      them. If Q1 holds, the improvement should show on the low-seam rows
      (m-020, s-020, m-021) at least as much as on the 8-9-seam long ones.
      **Falsified if the long fixtures improve and the short ones do not** --
      which would mean the blocking rows had a different cause than the
      crossfade and this round fixed the wrong thing.

Q5 is the one that can embarrass this diagnosis, and it is stated first-class
for that reason. The Phase 2 measurement said the crossfade is a per-boundary
artefact, so it should be *more* audible where it is not buried under
neighbouring seams -- which is exactly the single-seam rows. If instead only
the long fixtures move, the two blocking rows were something else and the
crossfade removal is a coincidence.

VERDICT GATE (blocking, not advisory):
    FAIL if any chunk-boundary artefact is flagged on a candidate trial that
    the paired reference does not also carry. A defect flagged on BOTH
    renditions is a pre-existing property of the pipeline — and here it
    genuinely is, because both files come from the same take — so it is
    recorded rather than blocking.

Blinding: the script never prints which arm is playing. The truth table
randomises A/B per trial from a fixed seed, reproducible from the generator
but not inferable from listening order.

Re-running is safe: trials already recorded for the same listener are
skipped. To restart, delete those rows from the CSV first.

Pass a round as the second CLI argument (``r1``) to replay round 1.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-l1-audition-helper.py [L1] [r2]

Working file — gitignored under ``_bmad-output/``; force-add per
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
    "r1": ("20-5-perceptual-fixtures", "20-5-state-cache-audition.csv"),
    "r2": ("20-5-perceptual-fixtures-r2", "20-5-state-cache-audition-r2.csv"),
}
DEFAULT_ROUND = "r2"

FIXTURE_DIR = ARTIFACTS_DIR / ROUNDS[DEFAULT_ROUND][0]
CANONICAL_CSV = ARTIFACTS_DIR / ROUNDS[DEFAULT_ROUND][1]

# Per-round listening guidance. Round 1 asked "does carrying codec state
# change what you hear"; round 2 asks "does removing the consumer crossfade
# fix the two rows where it did".
ROUND_BRIEF = {
    "r1": [
        "What is different between the two files:",
        "  ONLY the decoder. Both arms use the same chunk size (25), the",
        "  same 1024-sample seam blend and the same 64-sample consumer",
        "  crossfade. One decodes every chunk from a cold codec state, the",
        "  way the shipped build does; the other carries the codec's real",
        "  state across each boundary.",
    ],
    "r2": [
        "What is different between the two files:",
        "  ONLY the 64-sample consumer crossfade. BOTH arms carry codec",
        "  state caching -- that is settled, and round 1 preferred it 5-1",
        "  wherever the seam was exposed. One arm still cross-dissolves 2.7",
        "  ms across every chunk boundary; the other does not.",
        "",
        "  Why that matters: the crossfade blends the END of one chunk with",
        "  the START of the next -- two different moments -- so on audio",
        "  that is already continuous it smears rather than repairs. Round 1",
        "  flagged exactly that on m-020 and s-020: a click the shipped",
        "  build did not have. This round is whether removing it removes",
        "  the click without costing anything else.",
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
    if meta.get("round") == 2:
        blockers = {t for t, _, _ in blocking}
        r1_rows = {t for t in (r["trial_id"] for r in rows)
                   if t.startswith("m-020") or t.startswith("s-020")}
        print("\n  Against round 2's prediction, recorded before the round:")
        print("    Q1 m-020 / s-020 come back clean  : {}".format(
            "HELD" if not (blockers & r1_rows) else
            "FALSIFIED — still blocking: {}".format(sorted(blockers & r1_rows))))
        print("    Q2 no candidate-only defect at all: {}".format(
            "HELD" if not blocking else
            "FALSIFIED — {}".format(sorted(blockers))))
        print("    Q3 reference not preferred >= 4/14: {}".format(
            "HELD" if ref_pref < 4 else
            "FALSIFIED ({} of {})".format(ref_pref, total)))
        print("    Q4 equivalent modal, 7-12 of 14   : {}".format(
            "HELD" if 7 <= equal <= 12 else
            "FALSIFIED ({} equivalent, {} candidate-preferred)".format(
                equal, cand_pref)))
        print("    Q5 low-seam rows improve too      : {}".format(
            "HELD" if short_diff >= 0 and (short_diff > 0 or long_diff == 0)
            else "CHECK BY HAND (short={} long={})".format(
                short_diff, long_diff)))
        if long_diff > 0 and short_diff == 0:
            print("      ^ only the long fixtures moved. Round 1's blocking")
            print("        rows were SINGLE-SEAM. If they are still clean this")
            print("        is fine, but if they are not, the crossfade was not")
            print("        their cause and the diagnosis needs redoing.")
        return
    # Round 1's scoring, kept so `r1` can be replayed and re-scored.
    print("\n  Against the prediction recorded before the round:")
    print("    P1 no blocking candidate-only defect : {}".format(
        "HELD" if not blocking else "FALSIFIED"))
    print("    P2 reference not preferred >= 4/14   : {}".format(
        "HELD" if ref_pref < 4 else
        "FALSIFIED ({} of {})".format(ref_pref, total)))
    p3 = "HELD" if 6 <= equal <= 11 else (
        "FALSIFIED - large effect ({} candidate-preferred)".format(cand_pref)
        if cand_pref >= 10 else
        "FALSIFIED - arms far more distinguishable than predicted "
        "({} equivalent)".format(equal))
    print("    P3 equivalent modal, 6-11 of 14      : {}".format(p3))
    print("    P4 differences on long fixtures only : {}".format(
        "HELD" if short_diff == 0 or long_diff >= short_diff
        else "FALSIFIED (short={} long={})".format(short_diff, long_diff)))
    if cand_pref >= 10:
        print("\n  NOTE: a large audible win means the Story 20.4 seam blend")
        print("  was masking less well at cs25 than round 3 suggested. That")
        print("  strengthens AC #5's case for reopening the chunk size — as")
        print("  its own story, with its own audition.")


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
        print("Run 20-5-regen-audition-fixture{}.py first.".format(
            "" if round_id == "r1" else "-" + round_id), file=sys.stderr)
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
    print("=== Story 20.5 codec state-caching audition — round {} ===".format(
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
