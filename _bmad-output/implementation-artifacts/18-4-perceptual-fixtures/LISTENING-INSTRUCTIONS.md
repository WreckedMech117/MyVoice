# Story 16.7 — Perceptual A/B Audition Instructions

Thank you for participating in the streaming-TTS perceptual audition. The goal
is to detect any audible defects in the new chunked-streaming TTS path versus
the existing sentence-streaming baseline. Your judgment is the architecture's
gate (NFR3 — no audio stuttering / no audible overlap-add seams); the data
informs whether MyVoice flips the streaming default for GPU users.

## Your packet

This directory contains paired WAV files. For each utterance you will hear
TWO renditions of the same text — labelled `A` and `B` in the per-listener
manifest at `_perlistener_truthtable.json`. The two renditions come from
different code paths; you do NOT need to know which is which (and the labels
are randomized per-listener so guessing won't help).

## Protocol — for each utterance

1. Open `_perlistener_truthtable.json` and look up your listener id (L1, L2,
   L3, ...). Note the `trial_A_filename` and `trial_B_filename` for the
   utterance you're auditioning.
2. Listen to **trial A** end to end. Then listen to **trial B** end to end.
   Use headphones if you have them. Listen at a comfortable volume; do not
   crank the volume past your normal Discord-call level.
3. Record the following in the audition CSV (one row per utterance):
   - `utterance_id`: from the file name
   - `listener_id`: your assigned listener id (L1, L2, L3)
   - `a_or_b_preferred`: which rendition sounded better overall to you (`A`,
     `B`, or `equivalent`)
   - `a_defects_observed`: any defects you noticed in trial A — pick from
     the controlled vocabulary below; `none` if you heard nothing
   - `b_defects_observed`: same vocabulary, for trial B
   - `free_text_notes`: anything else worth flagging (one or two sentences)

## Controlled defect vocabulary

Pick exactly one. If you heard a defect not on this list, choose
`other_describe_in_notes` and add a sentence in `free_text_notes`.

| Value | What it means |
|---|---|
| `none` | No audible defects |
| `audible_seam` | Audible click, gap, or discontinuity between phrases |
| `clipping` | Distortion as if the audio is too loud / clipped |
| `phase_artifact` | Unnatural phasing or comb-filter sound |
| `tonal_distortion` | Pitch wandering or unnatural intonation |
| `other_describe_in_notes` | Some other defect — describe in notes column |

## What the gate is

Per Story 16.7 AC #2, the perceptual gate is **PASS if and only if zero
listeners flagged `audible_seam` for any TRUE_STREAM pair**. Preference is
informational at N=3; defect detection is the architectural concern.

## Submitting your results

Write your audition rows to a CSV at
`_bmad-output/implementation-artifacts/16-7-perceptual-ab-results.csv` with
the header:

```
utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes
```

If you prefer, send the maintainer a plain-text or spreadsheet copy and they
will fold the results into the canonical CSV. Do not edit the manifest file
or the WAV files.

Thank you for your time and attention.
