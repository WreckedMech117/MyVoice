# Story 20.7: Don't Let Generate Silently Queue Behind Priming (Phase ⊥-Polish-3)

Status: ready-for-dev

<!-- Phase tag: Phase ⊥-Polish-3. Seventh story of Epic 20. -->
<!-- Source: measured twice during Story 20.6's operator captures. Not on any follow-up list — found by telemetry, not by design review. -->
<!-- Risk: LOW-MEDIUM. Small change. The one failure that matters is a Generate button that never comes back; AC #2 exists for it. -->

## Story

As **a MyVoice user who opens the app and immediately wants to speak**,
I want **the app to tell me it isn't ready yet**,
so that **my first utterance isn't mysteriously slow with no explanation**.

## Context — this was found by measurement, not by review

Story 20.2/20.3 added compile priming at startup. It is a **real generation** and
it takes `_request_semaphore` (`qwen_tts_service.py:3031`, `:3085`, `:3740`), so a
user who presses Generate during that window **queues silently behind it**.

Measured twice in Story 20.6's captures, in the segment-1a dispatch interval:

| | segment 1a |
|---|---:|
| clean generations (16 of 18) | ~1.5–3.0 ms |
| contaminated (2 of 18) | **840.2 ms** and **1,382.6 ms** |

Three orders of magnitude, with nothing on screen to explain it. Priming runs
~4.4–4.9 s on the RTX 5090 (Story 20.2 §3), so the exposure window is real on
every launch, and slower hardware widens it.

**There is already an indicator and it is advisory.** `_emit_preparing_voice`
("Preparing TTS engine…") reaches the UI via
`app.py:2407 _on_tts_preparing_voice_message` and sets `preparing_voice_message`
on the TTS status. It informs; it gates nothing.

The button already has an enablement path —
`main_window.py:1557 set_generation_status(..., is_generating)` drives
`generate_button.setEnabled(not is_generating)`. What's missing is that **priming
does not count as busy**.

**This is a user-facing defect, not a measurement nuisance.** It cost two of
eighteen captured generations, but the person it actually hurts is a user who
launches MyVoice mid-conversation, presses Speak, and waits over a second for
reasons the UI never mentions — at exactly the moment Clear Comms exists to serve.

## Acceptance Criteria

### AC #1 — Generate is gated while priming holds the semaphore

**Given** compile priming is running and holding `_request_semaphore`
**When** the user looks at the main window
**Then** the Generate control is disabled, and the reason is visible — reuse the
existing "Preparing TTS engine…" message rather than inventing a second channel
**And** it re-enables as soon as priming releases, with no user action needed
**And** the gate keys off **priming actually holding the semaphore**, not off a
timer, a sleep, or an assumed duration

### AC #2 — The button must always come back (load-bearing)

**Given** priming is explicitly non-fatal: it can raise, be skipped by any of its
gates, or fail and land `priming_failed` telemetry
**When** any of those happen
**Then** the Generate control is re-enabled **regardless** — a failed prime must
never leave the app permanently unable to speak
**And** the release path cannot be skipped by an exception, a cancellation, or an
early return on any gate
**And** a test drives priming to **raise** and asserts the button is re-enabled;
another asserts it for each early-exit gate

> A dead Generate button is strictly worse than the silent queue this story
> exists to fix. If the implementation cannot guarantee re-enablement, say so and
> stop rather than shipping a maybe.

### AC #3 — Don't gate what isn't blocking

**Given** Story 17.2's voice-clone precompute emits the *same* indicator
("Preparing voice for streaming…") but serialises on a **per-voice lock**
(`qwen_tts_service.py:1859-1861`), not on `_request_semaphore`
**When** the gate is implemented
**Then** it applies to **compile priming only**, and the precompute path's
behaviour is unchanged
**And** if the implementation finds the precompute *can* also block a user
generation, that is reported as a separate finding with its measured cost — not
folded into this story

### AC #4 — The normal generating path is untouched

**Given** `set_generation_status(..., is_generating=True)` already disables
Generate during a user's own generation
**When** this gate is added
**Then** the two cannot fight: a user generation starting as priming ends, or
priming ending as a user generation starts, must not leave the button in the
wrong state
**And** the existing behaviour during a normal generation is unchanged
**And** a test covers the overlap in both orders

### AC #5 — No regressions

**Then** the accumulated suites pass with the pre-existing failure set unchanged
in count and identity
**And** `MYVOICE_DISABLE_COMPILE_WARMUP=1`, `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1`
and `tts_compile="off"` all still skip priming entirely, in which case the gate
never engages and the button is never disabled

## Tasks / Subtasks

- [ ] **Task 1 — Gate** (AC: #1, #3) — key off priming holding the semaphore; reuse the existing indicator channel.
- [ ] **Task 2 — Guaranteed release** (AC: #2) — release on every path including raise, cancel and each early-exit gate; tests for each.
- [ ] **Task 3 — Overlap with normal generation** (AC: #4) — tests in both orders.
- [ ] **Task 4 — Regression sweep** (AC: #5).
- [ ] **Task 5 — Verify by observation** — confirm the gate engages and releases on a real launch, ideally from `myvoice.log` rather than requiring an operator sitting. If it needs an operator, keep it to a single short check.

## Dev Notes

### Why not just make the indicator louder

That was considered and rejected before drafting: the indicator is *already*
present and was ignored twice by an operator who knew exactly what it meant and
had been explicitly warned about it in the launcher banner. An advisory that
sophisticated users miss under instruction will not save a user who has never
heard of compile priming.

### Scope discipline

This is a small story and should stay one. It is **not** a rework of the TTS
status indicator, **not** a queueing/feedback system for requests that arrive
during priming, and **not** a change to priming itself. If a friendlier design
(accept the press, show "queued behind engine warm-up…", proceed) is worth
building, that is a follow-up with its own justification — the measured harm here
is the *silence*, and disabling with a visible reason removes it.

### What this story is NOT

- Not F2 (the chunk-size reopen), which follows and needs its premise
  re-established first — `cs10` at 829 ms vs `cs25` at 1,491 ms were both measured
  in the session-drifted regime Story 20.6 §12 invalidated.
- Not F7 (the qasync call-site audit) or F6 (the RTX 3060 confirmation).

## References

- `_bmad-output/implementation-artifacts/20-6-retire-the-lookahead-evidence.md` §11–§12 — the two contaminated generations and their 840 ms / 1,383 ms cost
- `_bmad-output/implementation-artifacts/20-2-warm-path-compile-priming-evidence.md` §3 — priming duration ~4.4–4.9 s
- `src/myvoice/services/qwen_tts_service.py:3031,3085,3740` (`_request_semaphore`), `:1374` (`set_preparing_voice_callback`), `:1859-1861` (the 17.2 precompute's per-voice lock)
- `src/myvoice/app.py:512`, `:2407` (the indicator's path to the UI)
- `src/myvoice/ui/main_window.py:1557` (`set_generation_status` → `generate_button.setEnabled`)

## Dev Agent Record

_(to be filled by dev agent)_

## Change Log

- 2026-09-02 — Drafted by Winston. Found by Story 20.6's telemetry rather than by design review: two of eighteen captured generations carried 840 ms and 1,383 ms of silent semaphore wait against ~2 ms clean. Scoped deliberately small, with AC #2 as the load-bearing constraint — a Generate button that never returns is worse than the defect being fixed.
