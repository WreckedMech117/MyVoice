# Story 20.11: Re-prime After a Tier Change, and Prime Through TRUE_STREAM

Status: ready-for-dev - 2026-09-14

<!-- Source: Story 20.10 evidence §5, the two follow-ups raised from the RTX 3060 build-58 log. -->
<!-- Risk: MEDIUM. Touches the priming dispatch and adds a scheduling site in app.py. -->

## Story

As **a user who switches model quality tier in Settings**,
I want **the new tier's model loaded and its compile cache primed in the background right after the switch**,
so that **my next Generate does not pay a 17 s model load plus a cold compile (19.2 s to first audio on an RTX 3060, observed 2026-09-14 20:24:35)**.

And as **a user who has sentence-stream mode selected**,
I want **startup priming to prime the TRUE_STREAM path anyway**,
so that **the first true-stream generation after switching back does not pay the codec self-test and compile that priming exists to absorb (3.08 s vs 1.86 s observed)**.

## Context

`warmup_compile_async` (qwen_tts_service.py) is scheduled once from the startup
path in `app.py` (`_initialize_services`, via `_run_async_task_when_loop_is_idle`
→ `_compile_warmup_entrypoint`). It has no once-per-process guard of its own —
the "once" is the call site. It computes the compile-cache key from the *loaded*
model, so calling it again after a different tier's model is resident primes
that model's key.

Tier change today: `_on_settings_changed` → `_tts_service.set_quality_tier(new)`
→ `ModelRegistry` unloads the model ("will take effect on next generation").
Nothing reloads or primes; the next user generation does both inside the request.

Priming today dispatches with `self._resolve_streaming_mode()` — the user's
Settings override. With the sentence-stream override on, priming primed the
batch decode path and the first TRUE_STREAM generation still paid the
`CodecStateCache` self-test (~1.5 s on a 3060) and the streaming-geometry
compile. TRUE_STREAM is the shipping path and the one whose geometry is a
compile-cache-key dimension (`decode_window_frames`); it is what priming must
exercise. `effective_streaming_mode(None)` resolves the hardware default.

## Acceptance Criteria

### AC #1 — Tier change triggers preload + priming
**Given** `set_quality_tier` returns `changed=True`
**Then** app.py schedules, from that success callback, a coroutine that (1)
preloads the model for the active profile (same choice logic as startup:
`_voice_manager.get_active_profile_model_type()` else CUSTOM_VOICE), then (2)
runs `warmup_compile_async` through `_compile_warmup_entrypoint` so the Story
20.7 Generate-gate release-on-exit guarantee holds
**And** the log shows the same priming lines as startup (`Compile-priming
Generate gate: ENGAGED` … `primed cache successfully` / `warm-path priming
completed` … `RELEASED`)
**And** a tier change back to the original tier re-primes too (the key differs
per model_id, so the warm/cold decision is per tier)
**And** if `changed` is False nothing is scheduled.

### AC #2 — The preload failure path does not strand the gate
**Given** the preload fails or raises
**Then** priming is not attempted, a WARNING names why, and the Generate gate
is not engaged (it was never engaged) — verify by test that
`_on_tts_compile_priming_changed(False)` is the terminal state.

### AC #3 — Priming dispatches TRUE_STREAM regardless of the user's override
**Then** `_run_compile_priming` resolves its mode from the hardware default
(`effective_streaming_mode(None)`), not `_resolve_streaming_mode()`
**And** the user's own generations are unaffected (they still read the override
— pin with a test that sets the override to SENTENCE_STREAM, primes, and
asserts the priming dispatch saw TRUE_STREAM while a user generation saw
SENTENCE_STREAM)
**And** on a CPU-only host the hardware default is what it always was (priming
never runs there anyway — the Ampere gate exits first).

### AC #4 — Scheduling safety
**Then** the new site is classified per Story 20.9's scheme in the evidence
file: `_on_settings_changed` is a Qt signal handler (class (a)), so the
`on_success` continuation is created from a task-done callback. State which
task is current when the continuation schedules, and if it is not None, use
`_schedule_when_loop_is_idle` and prove it under the out-of-process qasync
driver (`tests/unit/_qasync_call_site_driver.py` pattern).

### AC #5 — No regressions
**Then** the full suite passes; startup priming order and log lines are
byte-identical (Story 20.3's evidence quotes them and a test pins them).

## Dev Notes

Do not touch `warmup_compile_async`'s gates or telemetry reasons. The
"Preparing TTS engine…" indicator on the cold path should appear during the
tier-change prime exactly as it does at startup — that is the UX the user is
trading the on-click wait for. Run pytest with
`I:\MyVoiceV2\python310\python.exe -m pytest` (the worktree has no interpreter
of its own; never modify or copy `python310`).
