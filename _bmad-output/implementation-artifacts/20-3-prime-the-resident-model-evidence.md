# Story 20.3 — Prime the Resident Model: evidence

Phase ⊥-Polish-3. Third story of Epic 20 (First-Audio Latency).
Implemented 2026-08-31.

---

## 0. Headline

Both defects that kept Story 20.2's measured win unreachable are closed in code
and covered by tests that catch the exact bug classes.

* **Defect 1 (ordering) — closed.** `warmup_compile_async` is now scheduled
  *after* the model preload completes, and after Story 17.2's
  `voice_clone_prompt` hydration. `reason="no_model_loaded"` is no longer the
  terminal outcome of a normal launch.
* **Defect 2 (wrong model) — closed.** Priming resolves the resident model
  type from `ModelRegistry.current_model_type` and builds the request to match
  it. The hard-coded `QwenModelType.CUSTOM_VOICE` is gone. Priming can no
  longer cause a model load, unload, or switch.
* **AC #3 (key/priming coherence) — closed.** `mark_warm(key)` is now vetoed
  whenever the key and the model priming actually exercised disagree.

**AC #4 (the through-the-GUI measurement) is NOT complete.** It requires ≥5
interactive launches on the Ampere+ host with a cloned voice active, which is
an operator task. §4 below is the exact procedure and the numbers to compare
against; the story cannot be signed off until that table is filled.

---

## 1. What changed

### 1.1 AC #1 — sequencing (Task 1)

`src/myvoice/app.py::_initialize_services_async`:

| before | after |
|---|---|
| `:568` hydration scheduled fire-and-forget, handle dropped | handle retained as `self._voice_clone_prompt_hydration_task` |
| `:593` `warmup_compile_async()` scheduled **above** the preload | block deleted |
| `:613` `await preload_model(...)` | unchanged |
| — | `self._warmup_compile_after_preload()` scheduled fire-and-forget **below** the preload block |

`MyVoiceApp._run_async_task` now **returns** the `asyncio.Future` it schedules
(previously it returned `None`). Existing callers ignore the value; the new
wrapper needs a handle to wait on. The wrapper swallows exceptions, so awaiting
it means "the task finished", never "the task succeeded".

New coroutine `MyVoiceApp._warmup_compile_after_preload`:

1. waits for the hydration task, bounded by `_HYDRATION_WAIT_TIMEOUT_S = 120.0`
   and wrapped in `asyncio.shield` — a timeout abandons the *wait*, not the
   hydration;
2. falls through to the warmup on timeout, cancellation, or hydration failure
   (never leaving priming unreached);
3. `await self._tts_service.warmup_compile_async()`.

**Why explicit sequencing and not a sleep/retry loop** (Task 1.1): the two
preconditions are events the startup coroutine already owns — the awaited
`preload_model` and the retained hydration future. A poll would reintroduce a
race for no benefit.

**Qt main thread** (AC #1 last clause, Task 1.2): the hand-off is still
`asyncio.ensure_future`, so `_initialize_services_async` returns immediately
and UI construction proceeds exactly as before. The scheduling point moved
~30 lines down inside the *same* coroutine and adds no synchronous work; the
statements between the old and new positions were already awaited on the
startup path. `test_warmup_is_not_awaited_inline_on_the_startup_path` is the
regression guard against "fixing" this by awaiting the warmup inline.

### 1.2 AC #2 — prime the resident model (Task 2)

`src/myvoice/services/qwen_tts_service.py`:

* **new** `CompilePrimingSkipped(Exception)` (module level) — carries a
  `.reason` used verbatim as the telemetry reason. Deliberately an exception
  rather than a sentinel return: every existing test that monkeypatches
  `_run_compile_priming` returns `None` or a `MagicMock`, and a truthy mock
  return would have been misread as a skip reason.
* **new** `_build_compile_priming_request()` — resolves
  `ModelRegistry.current_model_type` and builds one of three request shapes.
* **new** `_active_profile_voice_clone_prompt()` — **in-memory lookup only**
  against the Story 17.2 cache, via `_cache_lookup_validated` (so a replaced
  ref-audio or an edited `.txt` sidecar invalidates it exactly as the user
  path does). It never computes, never transcribes, never calls Whisper.
* **rewritten** `_run_compile_priming()` — builds the request from the
  resident model, keeps Story 20.2's F6 trip-wire and the `_audio_chunk_sink`
  assertion, records `_last_priming_model_type` **before** dispatch.

Per resident model:

| resident | priming request | rationale |
|---|---|---|
| **BASE** (common case) | `voice_clone_prompt=[<active profile's cached prompt>]` | same model *and* same conditioning regime as the user's first generation |
| **CUSTOM_VOICE** | `speaker="Ryan"` | unchanged from Story 18.4 |
| **VOICE_DESIGN** | `instruct` **and** `voice_description` = `_COMPILE_PRIMING_VOICE_DESCRIPTION` | see §1.5 |
| anything else | `CompilePrimingSkipped("unsupported_priming_model")` | never guess |
| BASE, no cached prompt | `CompilePrimingSkipped("no_priming_prompt")` | skip, never switch |
| no resident type / no registry | `CompilePrimingSkipped("no_model_loaded" / "no_model_registry")` | skip |

**`checkpoint_path` is carried across, verbatim.** This is not cosmetic:
`ModelRegistry.ensure_model_loaded`'s already-loaded fast path requires
`same_checkpoint` as well as a matching model type
(`model_registry.py:345-358`). A fine-tuned CUSTOM_VOICE resident primed with
`checkpoint_path=None` would fail that check and be **unloaded and reloaded** —
the exact cost this story removes. The raw string from
`current_checkpoint_path` is passed through **unnormalised**, because the
dispatch chain hands it back as `str(request.checkpoint_path)` and the registry
compares by equality; a `Path` round-trip rewrites separators on Windows and
would silently defeat the match.

### 1.3 AC #3 — key/priming coherence (Task 3)

* **new** `_compile_cache_model_id(loaded_model)` — the single place the
  `model_id` key dimension is derived. Two call sites deriving "which model is
  this" independently is precisely how the key and the primed model drifted
  apart.
* **new** `_priming_matches_cache_key(key_model_id, key_model_type)` — two
  independent vetoes:
  1. **model identity, re-read after priming.** This is the end-to-end
     signal: priming a model other than the resident one goes through
     `ensure_model_loaded`, which evicts and replaces the resident model, so a
     different `model_id` is resident when priming returns. It also vetoes when
     *nothing* is resident afterwards.
  2. **the recorded priming target.** `_last_priming_model_type` vs. the key's
     model type. Skipped when the target is `None` (a stubbed priming surface
     records none), so it adds signal without breaking the existing warmup
     suite.
* `warmup_compile_async` resets `_last_priming_model_type = None` before
  priming on **both** paths, and on the cold path calls the guard before
  `mark_warm`. On a veto: `mark_warm` is **not** called, a WARNING is logged,
  and `reason="key_model_mismatch"` (value 0.0, plus a `detail` tag) is
  recorded. The cache stays cold and the next launch retries.

### 1.4 New telemetry reasons

`tts_compile_warmup_priming` gains three reasons on top of the nine Story 20.2
documented:

| reason | value | meaning | `mark_warm`? |
|---|---:|---|---|
| `no_priming_prompt` | 0.0 | BASE resident, no cached prompt for the active profile — skipped | no |
| `unsupported_priming_model` | 0.0 | no priming request shape for the resident model type | no |
| `key_model_mismatch` | 0.0 | cold-path veto: key and primed model disagree | **no** |

Expected reason on a normal warm launch after this story: **`primed_warm`**
(cold cache: `primed_cold`).

### 1.5 VOICE_DESIGN decision (AC #2 requires this to be stated)

**Implemented, not skipped.** VoiceDesign generation is a pure function of
`(text, instruct)` — no reference audio, no profile state, no file writes — so
priming it is exactly as side-effect-free as the CUSTOM_VOICE path, and it goes
through the same suppressed dispatch. Skipping would leave a VoiceDesign user
permanently unprimed for no safety gain.

Both `instruct` and `voice_description` are set because the two dispatch paths
read different fields: TRUE_STREAM calls
`model.generate_voice_design(..., instruct=request.instruct, ...)`
(`qwen_tts_service.py`, TRUE_STREAM talker fork) while the BATCH fallback uses
`instruct=request.voice_description`. Setting only one would leave the fallback
priming with an empty description.

### 1.6 AC #5 — suppression and gates preserved

* Every priming request shape carries `suppress_audio_output=True`, asserted
  per model type in
  `test_every_resident_request_shape_is_suppressed`. This matters *more* after
  20.3 than before it: the BASE path carries the user's own cloned-voice
  prompt, so an unsuppressed prime would say "Hello world." **in the user's own
  voice** through their speakers and virtual microphone.
* Story 20.2's F6 trip-wire and the `_audio_chunk_sink(request) is None`
  assertion are unchanged.
* Story 20.2's "consumer wired **before** priming receives zero chunks" row
  (`test_cold_path_warmup_reaches_no_user_facing_channel`) still passes,
  now with a resident CUSTOM_VOICE model — see §3.1 on the one rig change.
* Every fast-exit is untouched: `MYVOICE_DISABLE_COMPILE_WARMUP=1`,
  `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1`, non-Ampere / CPU,
  `tts_compile="off"`, no model registry. All still covered by
  `test_qwen_tts_service_compile_warmup.py` (12 rows, all green, unmodified).

---

## 2. Tests

### 2.1 New

**`tests/unit/services/test_compile_priming_resident_model.py` — 26 rows.**

* Request shape per resident model, including all four ways the BASE prompt can
  be unavailable (no manager / no active profile / non-CLONED active profile /
  ref-audio deleted) — all land on the same `no_priming_prompt` skip.
* `test_base_resident_without_a_prompt_dispatches_nothing` — the skip aborts
  **before** any dispatch, so no `ensure_model_loaded` can happen.
* `test_priming_carries_the_resident_checkpoint_path_verbatim`.
* **The no-switch invariant (AC #2's explicit test obligation)** — driven
  through the **real** `ModelRegistry` in the post-preload state, with counting
  spies over `_load_model` / `_unload_model`:
  * `test_priming_never_loads_or_unloads_the_resident_model` (CUSTOM_VOICE
    resident) and `test_priming_never_moves_a_resident_base_model` (BASE
    resident) — zero ops, `current_model_type` unchanged,
    `get_loaded_model()` still the same instance.
  * `test_control_the_pre_20_3_hardcoded_model_type_would_have_switched` —
    **non-vacuity control**: the same rig, fed the pre-20.3 hard-coded
    CUSTOM_VOICE with BASE resident, records `["unload:BASE", "load:CUSTOM_VOICE"]`.
    Without this row, `ops == []` could be a property of a spy that never fires.
* **AC #3** — four rows: model swapped + type contradicted; **model identity
  alone** (priming records no target type, which is what a stubbed or renamed
  priming surface leaves behind); model vanished during priming; and the
  `primed_cold` control that proves `mark_warm` *is* reachable.
* Cold- and warm-path skip telemetry, including that the transient
  "Preparing TTS engine…" indicator is not left stuck on a skip.
* `test_no_model_loaded_is_not_the_outcome_when_a_model_is_resident` (AC #1,
  service half).

**`tests/unit/test_app_compile_warmup_sequencing.py` — 8 rows.**

Guards the ordering defect **at the call site**, which is where it lived
(per `memory/code_review_regression_test_exact_class.md`):

* AST invariant: every compile-warmup schedule line in
  `_initialize_services_async` must sit below every `preload_model` line. The
  two markers are **unioned, not fallback-ordered**, so re-adding a direct
  `warmup_compile_async()` above the preload while the wrapper stays below is
  still caught.
* AST invariant: the warmup is never `await`ed inline on the startup path.
* Source invariant: the hydration task handle is retained.
* Behavioural: the wrapper waits for hydration; and still runs the warmup when
  hydration times out (with `_HYDRATION_WAIT_TIMEOUT_S` monkeypatched to 50 ms),
  raised, or was never scheduled.
* `_run_async_task` returns its future.

### 2.2 Changed

`tests/unit/services/test_compile_priming_audio_suppression.py` — one rig line.
`_build_true_stream_service` now declares `CUSTOM_VOICE` resident on the
registry. The pre-20.3 rig stubbed `get_loaded_model()` while leaving
`current_model_type` at `None`, a state the real registry cannot be in
(`get_loaded_model` returns `None` exactly when there is no resident type).
No assertion in that file changed; all 12 rows still pass.

### 2.3 Mutation testing — 12 of 12 caught

Harness: `scratchpad/mutate.py` + `scratchpad/mutate2.py` (revert each fix in
turn, re-run the story's test files, require red).

| # | mutation | result |
|---|---|---|
| M1 | resident-type resolution → pre-20.3 hard-coded CUSTOM_VOICE | CAUGHT (12 failed) |
| M2 | BASE no-prompt skip reason changed | CAUGHT (7 failed) |
| M3 | AC #3 coherence guard always passes | CAUGHT (4 failed) |
| M4 | model-id check dropped from the guard | CAUGHT (1 failed) |
| M5 | primed-type check dropped from the guard | CAUGHT (1 failed) |
| M6 | `checkpoint_path` dropped from the priming request | CAUGHT (1 failed) |
| M7 | `suppress_audio_output` dropped from the priming request | CAUGHT (11 failed) |
| M8 | `_run_async_task` stops returning the future | CAUGHT (1 failed) |
| M9 | warmup no longer waits for hydration | CAUGHT (1 failed) |
| M10 | hydration handle no longer retained | CAUGHT (1 failed) |
| M11 | warmup block moved back above the preload | CAUGHT (1 failed) |
| M12 | pre-20.3 direct `warmup_compile_async()` above the preload | CAUGHT (1 failed) |

M4 was **MISSED on the first pass** — the AC #3 row that swaps the model also
contradicted the primed type, so the type check alone was carrying it. Two rows
were added (`..._on_model_identity_alone`,
`..._when_the_model_vanished_during_priming`) that isolate the identity signal.
No fix in this story rests on an assertion that does not exist.

---

## 3. Regression sweep (AC #5, Task 5)

`python310\python.exe -m pytest -q`, portable interpreter per
`memory/test_interpreter_portable_python310.md`.

| surface | result | vs. Story 20.2 baseline |
|---|---|---|
| `tests/unit/services tests/unit/observability tests/unit/models` | **926 passed, 0 failed** | 896 → 926 (+26 new rows, +4 elsewhere); **zero failures, unchanged** |
| `tests/unit` (whole tree) | 1,541 passed, 30 failed, 5 errors | identical failure set, **all pre-existing** — verified by stashing this story's source + test changes and re-running `tests/unit/ui/dialogs/voice_design_studio`: **30 failed, 5 errors** on the baseline too |
| `tests/integration tests/test_qwen_tts_internals.py` | 174 passed, **4 failed** | exactly the 4 pre-existing rows 20.2 documented |
| `tests/services tests/settings tests/utils` | 288 passed, **7 failed** | exactly 20.2's 7 pre-existing |
| `tests/ui` | 711 passed, **7 failed** | exactly 20.2's 7 pre-existing |
| new: `test_compile_priming_resident_model.py` | **26 passed** | — |
| new: `test_app_compile_warmup_sequencing.py` | **8 passed** | — |
| unchanged: `test_qwen_tts_service_compile_warmup.py` | **12 passed** | 12 |
| rig-touched: `test_compile_priming_audio_suppression.py` | **12 passed** | 12 |

**Zero new failures.** Every failure above is in the same UI / voice-profile /
session-manager drift set `20-2-warm-path-compile-priming-evidence.md` §5
documents, none of it touching the streaming dispatch chain.

---

## 4. ⚠ AC #4 — measurement through the shipped GUI: **PENDING**

**Status: not performed.** AC #4 requires ≥5 launches of the real application
with a CLONED voice active and a human pressing Generate on each. That is an
operator task on the Ampere+ host, not something the implementation pass can
produce. **The story is not closeable until this section is filled in.**

Everything AC #4 needs is in place: the GUI path is now live, and the shipped
capture Story 20.1 built for exactly this (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`,
driven by `01_Run_MyVoice_With_CSV_Capture.bat`) is unchanged.

### 4.1 Procedure

1. Precondition: warm compile cache (i.e. at least one prior primed launch on
   this host with this torch/model/precision combination), `tts_compile="auto"`,
   a **CLONED** voice set active so BASE is the preloaded resident model, and
   `MYVOICE_DISABLE_COMPILE_WARMUP` / `MYVOICE_DISABLE_WARM_COMPILE_PRIMING`
   **unset**.
2. Launch via `01_Run_MyVoice_With_CSV_Capture.bat`. Wait for the
   "Preparing TTS engine…" indicator to appear **and clear** — that is priming
   running at startup, which pre-20.3 never happened.
3. Generate the Story 17.3 §4.1 long paragraph once. That first generation is
   the measurement; anything after it is steady state.
4. Quit. Repeat for 5 launches. Repeat the set for the short
   (Clear Comms interjection) utterance if comparing to §3.2 of 20.2.
5. From the log, record the `tts_compile_warmup_priming` reason for each launch
   (expected: `primed_warm`; `no_model_loaded` would mean AC #1 regressed;
   `no_priming_prompt` would mean the active profile's prompt was not hydrated
   in time — see §5).

### 4.2 Table to fill

| launch | 1a model load | **1b first-forward** | 2 talker | 3 decode | TTFA(post) | telemetry reason |
|---|---:|---:|---:|---:|---:|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| **median** | | | | | | |

### 4.3 What to compare against

From `20-2-warm-path-compile-priming-evidence.md` §3, harness numbers for the
long cell:

| cell | harness before | harness after |
|---|---:|---:|
| segment 1b (first-forward) | 3,593 ms | **86 ms** |
| first-generation TTFA (TTFA − 1a) | 5,316 ms | **1,526 ms** |
| Story 20.1 pooled steady state | — | 1,785 ms |

**Expected divergences between the harness and the GUI, to be explained rather
than averaged away:**

* **Segment 1a will differ by construction.** The 20.2 harness loads the model
  lazily on first dispatch; the shipped GUI preloads at startup
  (`app.py`), so 1a is ~1 ms in the GUI on *both* sides. Compare
  `TTFA − 1a`, as 20.2 §3.3 does.
* **The GUI primes the model the user will actually use.** The harness primed
  CUSTOM_VOICE. A cloned-voice GUI launch now primes **BASE with the active
  profile's real prompt** — the same model *and* conditioning shape as the
  measured generation. If anything, the GUI number should be at least as good
  as the harness's; a GUI 1b materially above ~100 ms means the prime did not
  reach the graph the generation takes, and should be investigated, not
  averaged.
* **Priming holds `_request_semaphore`** (20.2 §3.4). A Generate pressed while
  priming is still running serialises behind it and can wait up to the
  remaining priming time (~4.4–4.9 s on the 20.2 host). Wait for the indicator
  to clear before generating, or the measurement captures queueing rather than
  first-forward.
* **Startup cost moves, it does not vanish.** The ~4.4 s is now billed to
  startup on every launch. Confirm subjectively that the window still appears
  and is interactive during it — the priming generation runs on the talker
  thread with the coroutine polling via `await asyncio.sleep`, so it should be.

---

## 5. Standing risks

1. **AC #4 is unverified on hardware.** Everything above is unit-level and
   source-level. The claim "the win is now reachable" is *structurally*
   established (the `no_model_loaded` exit is gone, the primed model is the
   resident one) but not *measured* end-to-end. Treat §0's headline as
   "defects closed", not "win delivered", until §4 is filled.
2. **Hydration/priming race on a slow first launch.** If
   `hydrate_voice_clone_prompt_cache` has not put the active profile's prompt
   in the cache — e.g. the voice is brand new and its `.pt` has never been
   computed — the BASE prime skips with `no_priming_prompt` and that launch's
   first generation pays the full lazy inductor reload. This is by design (the
   alternative is switching models, which is strictly worse) but it means
   **a user's very first launch after creating a new cloned voice is not
   primed**. Whether to pre-compute-then-prime is a follow-up, not this story.
3. **The 120 s hydration wait is a judgement call.** It is long enough that a
   large voice library on a slow disk still gets a primed BASE, and bounded so
   a wedged hydration cannot make priming unreachable. No measurement backs the
   specific value.
4. **`VOICE_DESIGN` priming is untested on hardware.** The request shape is
   unit-tested; no VoiceDesign-resident launch has been run through a real
   model. Its failure mode is benign (`priming_failed`, cache stays cold, next
   launch retries).
5. Story 20.2 §7.5's suppression hazards are now **live** rather than latent —
   priming actually runs. The four-channel wall
   (`test_compile_priming_audio_suppression.py`) is what stands between the
   BASE prime and the user's speakers, and it is green.

---

## 6. Out of scope, confirmed untouched

Per the story's Dev Notes: `DEFAULT_CHUNK_SIZE` and `streaming_chunk_buffer.py`
(Follow-ups B/C), Story 20.2's suppression mechanism (reused, not redesigned),
the hard-coded `decode_window_frames=30` in the cache key (Follow-up B —
changing it invalidates every warm cache directory), and PORT-b (Follow-up E).
`git diff --stat` confirms no edits to any of them.

---

## 7. File list

**Source**

* `src/myvoice/app.py`
  * `MyVoiceApp.__init__` — `_voice_clone_prompt_hydration_task` attribute
  * `_initialize_services_async` — hydration handle retained; warmup scheduling
    block moved below the preload and re-pointed at the new wrapper
  * `_warmup_compile_after_preload` (new) + `_HYDRATION_WAIT_TIMEOUT_S`
  * `_run_async_task` — returns the scheduled future
* `src/myvoice/services/qwen_tts_service.py`
  * `CompilePrimingSkipped` (new, module level)
  * `_COMPILE_PRIMING_VOICE_DESCRIPTION` (new constant)
  * `_last_priming_model_type` instance state
  * `_priming_matches_cache_key` (new) — the AC #3 guard
  * `_compile_cache_model_id` (new) — single derivation of the key's model_id
  * `_active_profile_voice_clone_prompt` (new)
  * `_build_compile_priming_request` (new)
  * `_run_compile_priming` — rewritten around the resident model
  * `warmup_compile_async` — `CompilePrimingSkipped` handling on both paths,
    the `mark_warm` coherence guard on the cold path, `_last_priming_model_type`
    reset, docstring updated with the three new reasons and the ordering fix

**Tests**

* `tests/unit/services/test_compile_priming_resident_model.py` (new, 26)
* `tests/unit/test_app_compile_warmup_sequencing.py` (new, 8)
* `tests/unit/services/test_compile_priming_audio_suppression.py` (rig line
  only — registry declares a resident model type)
