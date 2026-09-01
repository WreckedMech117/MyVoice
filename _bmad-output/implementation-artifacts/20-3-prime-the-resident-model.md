# Story 20.3: Prime the Resident Model (Phase ⊥-Polish-3)

Status: in-review — AC #1/#2/#3/#5 complete; **AC #4 (GUI measurement) pending an operator run**

<!-- Phase tag: Phase ⊥-Polish-3. Third story of Epic 20 (First-Audio Latency). -->
<!-- Source: Story 20.2 evidence §6 Follow-up A′. Activates the win Story 20.2 measured but could not reach. -->
<!-- Risk: MEDIUM-HIGH. Touches startup sequencing and the model-residency invariant. A wrong move here costs a model reload on the user's first generation — the exact cost Epic 20 exists to remove. -->

## Story

As **a MyVoice user whose cloned voice is loaded at startup**,
I want **the compile priming to actually run, and to prime the model I am about to use**,
so that **the first-generation speedup Story 20.2 measured is one I actually get**.

## Context

Story 20.2 measured a large, real win — first-forward 3,593 → 86 ms, first-generation
TTFA 5,316 → 1,526 ms — and then found that **no user can reach it**. Two defects sit
between the code and the user, both pre-existing:

**Defect 1 — the warmup never runs.** `warmup_compile_async` is scheduled at
`app.py:594`, but the first `await` in that same startup coroutine is the model preload
at `app.py:613`. The task therefore begins executing *at* that await, calls
`get_loaded_model()` before the preload has finished, and exits at
`reason="no_model_loaded"`. Deterministically, every launch. Story 18.4's cold priming
has never run in the shipped app either.

**Defect 2 — the priming targets the wrong model.** `_run_compile_priming` hard-codes
`model_type=QwenModelType.CUSTOM_VOICE`, while `warmup_compile_async` computes the
compile-cache key from whichever model is **loaded** (`qwen_tts_service.py:2078`,
via `name_or_path`). `compile_cache.compute_key` includes `model_id`, so these are
different keys.

Consequence on the cold path: for a cloned-voice user with BASE resident, priming
compiles **CUSTOM_VOICE** and then `mark_warm()` marks **BASE's** key. The cache can be
marked warm for a model that was never compiled. Defect 1 has been masking this.

**Domain fact establishing the target (Commander, 2026-08-31):** BASE is the default
resident model. Most users load and use whichever cloned voice they use most often;
a model switch happens only when creating a new voice or switching to one that needs a
different model. So the common startup state is **BASE resident with a cloned voice
active**, and that is the path priming must warm.

## Acceptance Criteria

### AC #1 — Priming runs after the model is actually loaded

**Given** the startup sequence schedules `hydrate_voice_clone_prompt_cache` (`app.py:569`)
and `warmup_compile_async` (`app.py:594`) as fire-and-forget tasks, and awaits
`preload_model` at `app.py:613`
**When** the app starts on an Ampere+ CUDA host with `tts_compile="auto"`
**Then** priming runs only after the preload has completed, and after prompt hydration
has completed if the resident model needs a clone prompt (AC #2)
**And** `reason="no_model_loaded"` is no longer the terminal outcome of a normal launch
**And** the Qt main thread is not blocked by the reordering — startup remains responsive,
and time-to-interactive does not regress measurably against the Story 20.2 baseline

### AC #2 — Prime the resident model, never switch models to prime (load-bearing)

**Given** `ModelRegistry` keeps exactly one model resident (`model_registry.py:5`) and
`_load_model` unloads the current model before loading another (`:364-365`)
**When** priming builds its synthetic request
**Then** it primes **the model that is already resident**, determined from the registry —
never a hard-coded model type
**And** priming **never** causes a model load, unload, or switch. This is the story's
central invariant: a priming pass that evicts the user's model has made first-generation
latency *worse*, which is the opposite of Epic 20's purpose
**And** per resident model type:
  - **BASE** (the common case) — prime with the active profile's cached
    `voice_clone_prompt`, sourced from the in-memory cache Story 17.2's
    `hydrate_voice_clone_prompt_cache` populates (`qwen_tts_service.py:1835`).
    This primes the exact model *and* conditioning regime the user's first generation
    will take.
  - **CUSTOM_VOICE** — prime with the canonical default speaker, as today.
  - **VOICE_DESIGN** — prime with a synthetic instruct, or skip with a distinct reason
    if that cannot be done without side effects. State the choice in Dev Notes.
**And** if the resident model cannot be primed (e.g. BASE resident but no cached prompt
for the active profile), priming is **skipped** with a distinct telemetry reason — it does
**not** fall back to priming a different model
**And** a test asserts the invariant directly: run priming with a model resident, assert
the registry performed zero load/unload operations and `current_model_type` is unchanged

### AC #3 — The cache key and the primed model can never diverge

**Given** `compile_cache.compute_key` includes `model_id`, and `mark_warm(key)` currently
runs on the cold path regardless of which model priming actually compiled
**When** priming completes on the cold path
**Then** `mark_warm(key)` is called **only** when the key was computed from the same model
that priming actually exercised
**And** if they would diverge, `mark_warm` is not called and the mismatch is logged and
recorded in telemetry — a cold cache that retries next launch is strictly better than a
warm marker for a model that was never compiled
**And** a regression test covers the exact defect class: key computed from model A while
priming targets model B ⇒ `mark_warm` not called

### AC #4 — Measured through the shipped GUI

**Given** Story 20.2's measurement used the harness because the GUI path was dead
**When** this story makes the GUI path live
**Then** the win is confirmed **through the real app**, using the shipped env-var capture
(`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`) that Story 20.1 put in place for exactly this
purpose — ≥5 launches, warm cache, `tts_compile="auto"`, a CLONED voice active (BASE
resident), measuring the user's genuine first generation
**And** the result is compared against Story 20.2's harness numbers, with any divergence
between harness and GUI explained rather than averaged away
**And** the telemetry reason observed on a normal launch is reported (expected:
`primed_warm`), confirming Defect 1 is closed in the shipped path
**And** results land at
`_bmad-output/implementation-artifacts/20-3-prime-the-resident-model-evidence.md`

### AC #5 — Audio suppression and existing gates preserved

**Given** Story 20.2's request-scoped `suppress_audio_output` mechanism
**When** priming is rebuilt around the resident model — including the BASE path, which
carries a real voice-clone prompt and would otherwise speak in the user's own cloned voice
**Then** every priming request still carries `suppress_audio_output=True` and resolves
through `_audio_chunk_sink(request)`
**And** the Story 20.2 hardware check still holds: a consumer wired **before** priming
receives zero chunks
**And** all existing fast-exits fire unchanged: `MYVOICE_DISABLE_COMPILE_WARMUP=1`,
`MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1`, non-Ampere / CPU, `tts_compile="off"`,
no model registry
**And** the regression surface passes with zero new failures (Story 20.2 closed at 896
across services/observability/models; the tree's other pre-existing failures are
documented in `20-2-warm-path-compile-priming-evidence.md` and are not this story's)

## Tasks / Subtasks

- [x] **Task 1 — Sequence priming after model load** (AC: #1)
  - [x] 1.1 Restructure the `app.py` startup so priming begins after `preload_model` completes, and after hydration if the resident model needs a prompt. Prefer explicit sequencing over sleeps or retry loops.
  - [x] 1.2 Keep it off the Qt main thread; confirm startup responsiveness is unchanged.
  - [x] 1.3 Test that priming is reached with a model loaded, and that `no_model_loaded` is no longer the normal-launch outcome.

- [x] **Task 2 — Prime the resident model** (AC: #2, #5)
  - [x] 2.1 Resolve the resident model type from the registry; build the priming request to match it.
  - [x] 2.2 BASE path: source the active profile's cached clone prompt from the Story 17.2 in-memory cache; skip with a distinct reason if absent.
  - [x] 2.3 CUSTOM_VOICE path: preserve today's behavior. VOICE_DESIGN: implement or skip per AC #2; record the choice.
  - [x] 2.4 Assert the no-switch invariant in a test (zero load/unload, `current_model_type` unchanged).
  - [x] 2.5 Confirm `suppress_audio_output=True` on every priming path, BASE included.

- [x] **Task 3 — Key/priming coherence** (AC: #3)
  - [x] 3.1 Guard `mark_warm` so it cannot mark a key for a model priming did not compile.
  - [x] 3.2 Regression test for the exact defect class (key model A, priming model B ⇒ no `mark_warm`).

- [ ] **Task 4 — Measure through the GUI** (AC: #4) — **BLOCKED on an operator run**
  - [ ] 4.1 ≥5 launches with a CLONED voice active, warm cache, via `MYVOICE_PROGRESSIVE_PLAYBACK_CSV`. Procedure written up in evidence §4.1; requires interactive launches on the Ampere+ host.
  - [ ] 4.2 Report first-generation TTFA and segment 1b; compare to Story 20.2's harness numbers and explain divergence. Comparison baseline + the divergences to expect are pre-written in evidence §4.3.
  - [x] 4.3 Report the observed telemetry reason. Write the evidence file. — evidence file written; the reason table (§4.2) is empty pending 4.1.

- [x] **Task 5 — Regression sweep** (AC: #5)

## Dev Notes

### The invariant that matters most

**Priming must never move the model.** Every other requirement in this story is
negotiable in implementation detail; this one is not. `ModelRegistry` holds one model
(~3.4 GB) and evicts to load another. A priming pass that switches models would unload
what the user is about to use, forcing a reload plus a fresh `engage` on their very
first generation — turning a −3.9 s win into a multi-second loss. If in doubt about
whether a path can switch models, skip priming and record a reason.

### Why BASE conditioning is better, not just different

Priming BASE with the active profile's real cached prompt exercises the same model and
the same conditioning shape as the user's first generation. The previous CUSTOM_VOICE
priming warmed a different compile-cache key entirely. This story is therefore not only
a sequencing fix — it makes the priming *relevant* for the first time.

### What this story is NOT

- **Not Follow-up B or C.** `chunk_size` and the adaptive cushion stay untouched; they
  are coupled to each other and ship together later. Do not touch `DEFAULT_CHUNK_SIZE`
  or `streaming_chunk_buffer.py`.
- **Not a change to Story 20.2's suppression mechanism.** Reuse it; do not redesign it.
- **Not a fix for the hard-coded `decode_window_frames=30` in the cache key.** Story 20.1
  §5.4 found it disconnected from real streamer geometry; it is correct today and belongs
  to Follow-up B. Changing it invalidates every warm cache directory for no benefit here.
- **Not a port.** PORT-b is Follow-up E, to be re-measured against the post-20.3 baseline.

## References

- `_bmad-output/implementation-artifacts/20-2-warm-path-compile-priming-evidence.md` §6 (Follow-up A′, with the §3 numbers as the business case)
- `_bmad-output/implementation-artifacts/20-1-ttfa-spike-faster-qwen3-tts-evidence.md` §2.5 (the cold/warm split this activates)
- `src/myvoice/app.py:569` (hydration), `:594` (warmup), `:613` (preload) — the three-way ordering
- `src/myvoice/services/qwen_tts_service.py:2078` (key computed from loaded model), `_run_compile_priming` (hard-coded CUSTOM_VOICE), `:1835` (`hydrate_voice_clone_prompt_cache`)
- `src/myvoice/services/model_registry.py:5` (single-residency), `:364-365` (unload-before-load)

## Dev Agent Record

### Agent Model Used

claude-opus-5[1m] (Claude Code)

### Completion Notes List

Full evidence: `_bmad-output/implementation-artifacts/20-3-prime-the-resident-model-evidence.md`.

**AC #1 — done.** `warmup_compile_async` is scheduled below the `preload_model`
await, through a new `MyVoiceApp._warmup_compile_after_preload` wrapper that
first waits (bounded 120 s, `asyncio.shield`-ed) for Story 17.2's hydration
task. `_run_async_task` now returns its future so the wrapper has a handle.
Still fire-and-forget: startup does not block. Guarded at the call site by AST
invariants, because the defect *was* a call-site ordering bug.

**AC #2 — done.** `_build_compile_priming_request()` resolves
`ModelRegistry.current_model_type`; the hard-coded `CUSTOM_VOICE` is gone. BASE
primes with the active profile's cached `voice_clone_prompt` (in-memory lookup
only — no compute, no Whisper); CUSTOM_VOICE is unchanged; VOICE_DESIGN is
**implemented** with a synthetic instruct (rationale in evidence §1.5). Any
un-primeable resident model raises `CompilePrimingSkipped` with a distinct
telemetry reason — never a fallback to a different model. `checkpoint_path` is
carried across verbatim, which is load-bearing: the registry's already-loaded
fast path checks it, so dropping or `Path`-normalising it would unload and
reload a fine-tuned resident model. The no-switch invariant is asserted through
the **real** `ModelRegistry` with a non-vacuity control showing the pre-20.3
hard-coded type *does* register `unload:BASE` + `load:CUSTOM_VOICE` on the same
rig.

**AC #3 — done.** `_priming_matches_cache_key()` vetoes `mark_warm` on two
independent signals: the resident model's identity re-read after priming, and
the model type priming recorded as its target. On a veto the cache stays cold,
a WARNING is logged, and `reason="key_model_mismatch"` is recorded.

**AC #5 — done.** Every priming request shape carries
`suppress_audio_output=True` (asserted per model type); Story 20.2's F6
trip-wire, `_audio_chunk_sink` assertion, four-channel wall and every fast-exit
are untouched. Regression sweep: **926 passed / 0 failed** on
`tests/unit/services + observability + models`; every failure elsewhere in the
tree is byte-identical to the pre-existing set, verified by stashing this
story's changes and re-running. **Zero new failures.**

**Mutation testing: 12/12 caught** (M4 was missed on the first pass and two
rows were added to isolate the model-identity signal).

**AC #4 — NOT DONE.** It requires ≥5 interactive launches of the shipped GUI
with a cloned voice active and a human pressing Generate; that is an operator
task. Evidence §4 carries the exact procedure, the empty results table, the
Story 20.2 comparison baseline, and the harness↔GUI divergences to expect
(1a differs by construction; the GUI now primes BASE-with-real-prompt rather
than the harness's CUSTOM_VOICE; priming holds `_request_semaphore`). **The
story cannot be signed off until that table is filled.**

Standing risks are listed in evidence §5 — most notably that a user's first
launch after creating a brand-new cloned voice will skip priming with
`no_priming_prompt` (by design: the alternative is switching models, which is
strictly worse).

### File List

**Source**

- `src/myvoice/app.py` — `_voice_clone_prompt_hydration_task` attribute;
  hydration handle retained; warmup scheduling moved below the preload;
  `_warmup_compile_after_preload` + `_HYDRATION_WAIT_TIMEOUT_S` (new);
  `_run_async_task` returns its future
- `src/myvoice/services/qwen_tts_service.py` — `CompilePrimingSkipped` (new);
  `_COMPILE_PRIMING_VOICE_DESCRIPTION` (new); `_last_priming_model_type` state;
  `_priming_matches_cache_key`, `_compile_cache_model_id`,
  `_active_profile_voice_clone_prompt`, `_build_compile_priming_request` (new);
  `_run_compile_priming` rewritten; `warmup_compile_async` skip handling +
  `mark_warm` coherence guard + docstring

**Tests**

- `tests/unit/services/test_compile_priming_resident_model.py` (new, 26 rows)
- `tests/unit/test_app_compile_warmup_sequencing.py` (new, 8 rows)
- `tests/unit/services/test_compile_priming_audio_suppression.py` (rig line
  only: the registry now declares a resident model type)

**Evidence**

- `_bmad-output/implementation-artifacts/20-3-prime-the-resident-model-evidence.md` (new)

## Change Log

- 2026-08-31 — Implemented (AC #1/#2/#3/#5). Both defects closed; 12/12 mutations caught; zero new regression failures. AC #4's GUI measurement is left open as an operator task, with the procedure and comparison baseline written up in the evidence file.
- 2026-08-31 — Drafted by Winston from Story 20.2's Follow-up A′, incorporating Commander's domain fact that BASE is the default resident model for typical cloned-voice usage. Scope covers both defects that keep 20.2's measured win unreachable: the startup-ordering defect and the primed-model/cache-key mismatch.
