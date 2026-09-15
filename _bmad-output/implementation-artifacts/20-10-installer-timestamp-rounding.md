# Story 20.10: Installer Timestamp Rounding Kills the Bundled Prompt Caches

Status: done - 2026-09-14. Root cause proven with a one-file Inno test installer; `TimeStampRounding=0` + 2 s fingerprint tolerance; 2 tests mirror the bug class.

<!-- Found in the RTX 3060 log of build 2.2.0.57 (Epic 20 F6 confirmation run). -->
<!-- Risk: LOW. One installer directive, one tolerance constant, no behaviour change on a hit. -->

## Story

As **a user installing MyVoice on a fresh PC**,
I want **the 12 precomputed default-voice prompts the installer ships to actually be used**,
so that **compile priming runs at startup and the cold compile does not land on my first generation**.

## Context

The 3060 log (`installer_output/logs/myvoice.log`, session 19:16:45) shows
`Voice clone prompt cache: hydrated 0/12 CLONED voices for tier small from
disk`, then `Compile warmup priming skipped (no_priming_prompt)`, then a
first generation that took **50.2 s** request-to-audio: 13 s prompt
computation + 4 s model reload + ~30 s cold compile. The June install showed
the same `0/12` for the bundled voices (the single hit was one computed at
runtime on that PC). Every end-user install since Story 17.2 has been in
this state; the RTX 5090 dev box never was, because the dev tree's
`voice_files/*.pt` are written in place by the app, not by the installer.

**Root cause.** `_voice_clone_prompt_meta_is_valid` compares the meta's
`ref_audio_mtime` / `txt_mtime` against the installed `.wav` / `.txt` with a
**1 ms** tolerance. Inno Setup's default `TimeStampRounding=2` rounds every
installed file's mtime **down to an even second**. Proven with a one-file
test installer on the build host:

| file | source mtime | installed (default) | installed (`TimeStampRounding=0`) |
|---|---|---|---|
| Sarira-F.wav | 1761007842.8872526 | 1761007842.0 (−0.887 s) | 1761007842.8872526 (exact) |
| Sarira-F.txt | 1761007843.0022511 | 1761007842.0 (−1.002 s) | 1761007843.0022511 (exact) |

## Acceptance Criteria

### AC #1 — The installer preserves source timestamps
**Then** `installer.iss` sets `TimeStampRounding=0`, with the reason in a comment.

### AC #2 — The fingerprint check tolerates the rounding class
**Then** the meta and in-memory mtime comparisons use one shared tolerance
of 2.0 s, and a test that shifts the installed files back by 1.999 s (the
worst case the rounding can produce) still hydrates
**And** a shift of 2.5 s is still a miss, so stale detection is not weakened
in practice — size, sidecar and pin still gate independently.

### AC #3 — No regressions
**Then** the full suite is unchanged in identity.

## Out of scope, reported

* The release reason of the adaptive pre-buffer (`last_release_reason`) is
  test-only; the 3060 log cannot say which regime released. One INFO line
  per session would close that observability gap.
* Cold compile at `decode_window_frames=10` on the 3060 took ~30 s the first
  time the new inductor cache key was seen; later launches hit the disk cache.
  That is what priming exists to absorb, which is why this story matters.
