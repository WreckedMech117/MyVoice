# Story 20.7: Don't Let Generate Silently Queue Behind Priming (Phase ⊥-Polish-3)

Status: ready-for-review

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

- [x] **Task 1 — Gate** (AC: #1, #3) — key off priming holding the semaphore; reuse the existing indicator channel.
- [x] **Task 2 — Guaranteed release** (AC: #2) — release on every path including raise, cancel and each early-exit gate; tests for each.
- [x] **Task 3 — Overlap with normal generation** (AC: #4) — tests in both orders.
- [x] **Task 4 — Regression sweep** (AC: #5).
- [x] **Task 5 — Verify by observation** — confirm the gate engages and releases on a real launch, ideally from `myvoice.log` rather than requiring an operator sitting. If it needs an operator, keep it to a single short check.

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

### What was built

**Producer (`qwen_tts_service.py`).** `_set_compile_priming_active(bool)` - a
total, non-raising declaration - plus `set_compile_priming_callback()` and a
`compile_priming_active` property. Called with `True` as the **first statement
inside** each of the two priming `try` blocks (warm path, cold path) and with
`False` as the **first statement of** each block's existing `finally`. This is
the Story 20.5 producer-declares / consumer-acts shape: the service is the only
component that knows when priming owns the slot; it does not reason about
widgets.

**Consumer (`app.py` -> `main_window.py`).** `_on_tts_compile_priming_changed`
forwards to `MainWindow.set_engine_priming`, which records `_is_priming` and
calls the new `_refresh_generate_enabled()`. `set_generation_status` was rewired
from `setEnabled(not is_generating)` to record-then-derive through that same
helper, so `_refresh_generate_enabled` is the **single owner** of the button's
enabled state. Every other call site in the app already routes through
`set_generation_status`, so they all inherit the derivation.

### AC #2 - why re-enablement is guaranteed, not hoped for

Three independent mechanisms, in order of how much would have to fail:

1. **The gate is only ever engaged inside a `try` with a releasing `finally`.**
   Every early-exit gate - `MYVOICE_DISABLE_COMPILE_WARMUP`, `tts_compile="off"`,
   the D-9 hardware probe and its raise, no `model_registry`, no model loaded,
   the cache-key computation failure, and Story 20.2's
   `MYVOICE_DISABLE_WARM_COMPILE_PRIMING` - returns *before* the flag is ever
   set. There is nothing to release, which is stronger than releasing correctly.
2. **The release is the `finally`'s first statement and cannot raise.** It
   covers the success return, `CompilePrimingSkipped`, any other exception,
   `CancelledError`, and the cold path's early `return` from inside the `try`
   (Story 20.3's key/model coherence veto). Ordering is load-bearing: the same
   `finally` also clears the preparing-voice indicator, so a release placed
   after that clear could be skipped by a raise from it.
3. **A second, independent release at the task boundary.**
   `_compile_warmup_entrypoint` now wraps `await coro_factory()` in a
   `try/finally` that calls `_on_tts_compile_priming_changed(False)`, covering
   anything outside the service's own blocks. The consumer hop is total in both
   directions - no window yet, a half-built app, or a raising window all
   fail-safe toward *enabled*.

Two AST source invariants pin the structure those rows sample (the Story
20.5/20.6 device): every `_run_compile_priming` call site must sit in a `try`
whose `finally` releases, and that release must be the `finally`'s first
statement. A third invariant pins that the Generate button's enabled state is
assigned from exactly one place.

### AC #3 - the precompute is untouched, and one finding

Story 17.2's `prepare_voice_clone_prompt` is unchanged and drives only the
advisory message channel. A source invariant asserts `_set_compile_priming_active`
is written from `warmup_compile_async` alone.

**Finding, not folded in (AC #3's escape hatch):** the 17.2 precompute *can*
serialise against a user generation - `generate_voice_clone`'s cache-miss branch
takes the same per-voice `asyncio.Lock` (`qwen_tts_service.py:1859-1861` vs the
dispatch-path acquisition around `:2918`). It differs from this story's defect
in two ways: the waiting generation is waiting for work it would otherwise have
to do itself, so the wait is not pure loss; and `app.py` already defers presses
behind that path via `_voice_pipeline_in_flight` / `_fire_pending_generation`.
**No measured cost** - it did not appear in Story 20.6's captures. Recorded here
for whoever wants to measure it.

### Scope

The three warned-against expansions were not done: the TTS status indicator is
untouched, no queueing/feedback system was built, and priming itself is
unchanged.

**One judgment call:** the Generate button's tooltip is left as "Generate speech
(Enter)" while gated, rather than restated to the priming reason. AC #1's "the
reason is visible" is served by the pre-existing "Preparing TTS engine..."
indicator the same priming region already emits, and the AC explicitly says to
reuse it rather than invent a second channel. If a reviewer wants the tooltip
too, it is a two-line follow-up.

### Tests

- `tests/unit/services/test_compile_priming_generate_gate.py` (25 rows) -
  engage/release on both priming paths, the not-a-timer property, raise, skip,
  cancel, early-return, a raising consumer, no consumer wired, all seven skip
  gates, AC #3, and the three source invariants.
- `tests/ui/test_generate_gate_during_priming.py` (12 rows) - disable/re-enable,
  idempotence, the AC #4 overlap in **both** orders, the unchanged normal
  generation path, the single-owner invariant, the app wiring, an end-to-end
  producer -> orchestrator -> button hop with no mocks in the middle, and the
  task-boundary safety net under both raise and cancel.

### Regressions (AC #5)

Full suite run twice on this machine - stashed `main` vs this branch, same
command, same ordering (`-p no:randomly -v`): **49 failed / 4 errors / 2,883
passed** before, **49 failed / 4 errors / 2,920 passed** after (+37 = the 25 +
12 new rows). The pre-existing failure set is unchanged in **count and
identity** - the sorted diff of the two FAILED+ERROR id lists (53 entries each)
is empty. The `main` numbers reproduce Story 20.6's recorded baseline exactly.

**Pre-existing flake found while doing this, unrelated to the story.** Both
runs above deselect
`tests/settings/test_reset_to_defaults.py::TestResetQuickSpeak::test_reset_quick_speak_entries`,
which **intermittently hangs the whole run** - observed on stashed `main` as
well as on this branch, and not reproducible when that file is run alone. It
touches quick-speak persistence and nothing this story changed. Flagged, not
fixed.

### Verification by observation (Task 5)

The **defect** is now pinned in `logs/myvoice.log` from an existing Story 20.6
capture launch, harder than the story's own numbers: `09:42:45.216` priming
dispatched; `09:42:52.491` the operator pressed Generate; priming did not finish
until `09:43:02.893`. That press waited **~10.4 s** behind the semaphore with
nothing but the advisory indicator on screen.

`_set_compile_priming_active` now logs two INFO lines, so the fix is verifiable
from the same file without watching the button.

**Operator check - one launch, about 30 seconds.** Run `01_Run_MyVoice.bat`,
wait for the window, then close it. In `logs/myvoice.log`, confirm the newest
launch contains **both** lines:

    QwenTTSService - INFO - Compile-priming Generate gate: ENGAGED ...
    QwenTTSService - INFO - Compile-priming Generate gate: RELEASED (Generate re-enabled)

`ENGAGED` should sit immediately before `Compile priming: dispatching against
the resident model ...`, and `RELEASED` immediately after `warm-path priming
completed (duration=...)`. **`RELEASED` is the one that matters** - if it is
absent, the Generate button is dead for that session and this must not ship.
Optionally, press Generate during the ~5 s window: the button should be greyed
out with "Preparing TTS engine..." on the TTS indicator.

## Change Log

- 2026-09-02 — Drafted by Winston. Found by Story 20.6's telemetry rather than by design review: two of eighteen captured generations carried 840 ms and 1,383 ms of silent semaphore wait against ~2 ms clean. Scoped deliberately small, with AC #2 as the load-bearing constraint — a Generate button that never returns is worse than the defect being fixed.
- 2026-09-02 - Implemented. The gate is keyed to the two priming `try` blocks rather than to any duration. AC #2 is discharged by three layers (every skip gate returns before the flag is set; a first-statement, non-raising release in each `finally`; a second release at the warmup task boundary) plus two AST invariants so the structure cannot drift. AC #3's escape hatch used once: the 17.2 precompute's per-voice lock can block a same-voice generation, recorded as an unmeasured finding rather than folded in. One deliberate omission recorded under Scope (the gated button's tooltip). Task 5's headless half found the defect itself in `myvoice.log` at ~10.4 s of hidden wait; one short operator launch remains, to confirm the RELEASED line.
