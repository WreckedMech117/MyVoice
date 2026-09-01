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

**AC #1 is now verified in the shipped application**, not only in tests — see
§4.0. Two real launches on the Ampere+ host show the warmup handing off, priming
**BASE (Clone)** (the resident model), and completing: `primed_cold` on the
first launch and `primed_warm` on the second, with zero re-entrancy errors.

**AC #4 (the TTFA measurement) is still NOT complete.** It requires ≥5
interactive launches with a human pressing Generate. §4 is the procedure and the
numbers to compare against; the story cannot be signed off until that table is
filled.

> **Iteration note (2026-09-01).** The first AC #1 fix passed every unit test
> and did nothing in the shipped app. Two negative GUI passes and a promoted
> log line pinned it to a qasync task-re-entrancy failure. §1.1a is the full
> account; §2.4 is the verification that would have caught it. **Removing
> `shield`/`wait_for` was necessary but not sufficient** — a plain
> `ensure_future(warmup_compile_async())` at the same point is destroyed
> identically, which is demonstrated rather than asserted.

---

## 1. What changed

### 1.1 AC #1 — sequencing (Task 1)

Two defects had to be closed before the warmup body ran even once in the
shipped app. Defect 1 is the ordering one the story was written against;
defect 1a was found only by the AC #4 GUI capture.

**Defect 1 — ordering.** `src/myvoice/app.py::_initialize_services_async`:

| before | after |
|---|---|
| `:568` hydration scheduled fire-and-forget, handle dropped | handle retained as `self._voice_clone_prompt_hydration_task` |
| `:593` `warmup_compile_async()` scheduled **above** the preload | block deleted |
| `:613` `await preload_model(...)` | unchanged |
| — | the warmup handed off **below** the preload block |

`MyVoiceApp._run_async_task` now **returns** the `asyncio.Future` it schedules
(previously `None`), so the hydration handle is observable.

### 1.1a AC #1 — the qasync re-entrancy failure (2026-09-01)

**Symptom.** Two 5-launch GUI passes, bf16 + compile genuinely engaged
(`cuda_capability=12.0`), segment 1b still 3,585–3,642 ms — Story 20.2's
*before* number — and **zero** `tts_compile_warmup_priming` rows in any CSV. That
metric is recorded on every exit path, so the body never executed at all.

**The log line that pinned it** (`logs/myvoice.log`, 09:34:54,645, immediately
after "Model Base (Clone) preloaded successfully"):

```
qasync._QEventLoop - ERROR - Exception in callback <TaskStepMethWrapper object>()
RuntimeError: Cannot enter into task <Task pending name='Task-6'
  coro=<MyVoiceApp._run_async_task.<locals>._handle_task() running at app.py:1070>>
  while another task <Task pending name='Task-1'
  coro=<async_main() running at main.py:397>> is being executed.
Task was destroyed but it is pending!
```

**Mechanism.** Under qasync, `call_soon` is **not** a ready-queue append. It is
`call_later(0, ...)` → `_SimpleTimer.add_callback` → **`QObject.startTimer(0)`**
(`qasync/__init__.py`). A queued task step is therefore delivered by
`timerEvent` during **any** Qt event processing — including the synchronous
`splash.showMessage(...)` / `processEvents()` stretch that `main.py` runs
*inside* Task-1 straight after `initialize_async()` returns. `main.py:397`
named in the traceback is exactly that `showMessage` call.
`asyncio._enter_task` refuses to enter a second task while Task-1 is mid-step,
the loop's exception handler swallows the RuntimeError, and the task dies
pending.

**Correction to the first diagnosis.** The failure was attributed to
`wait_for`/`shield` over a task handle. That is **not** the cause, and removing
them is not a fix. Measured under a real qasync loop
(`tests/unit/_qasync_warmup_driver.py`, three variants, each in a fresh
process):

| hand-off at the post-preload point | metric rows | outcome |
|---|---:|---|
| `wait_for(shield(hydration_task))` then warmup (as first shipped) | **0** | destroyed, `Cannot enter into task` |
| plain `_run_async_task(warmup_compile_async())` | **0** | destroyed, identical error |
| `await hydration` inline in Task-1, then plain schedule | **0** | destroyed, identical error |
| **`_run_async_task_when_loop_is_idle(...)`** | **1** (`primed_warm`) | **survives, zero errors** |

The first three are the shapes proposed as fixes; all three fail. What matters
is not *which* awaitable machinery is used but *when the task's steps are
delivered* relative to Task-1's synchronous Qt pumping.

**The fix.** `MyVoiceApp._run_async_task_when_loop_is_idle(coro_factory,
on_error=)` creates the task only on a loop pass where
`asyncio.current_task() is None`. That reads the same `_current_tasks` slot
`_enter_task` checks, so it tests the precise precondition rather than a proxy
for it, and it terminates the instant Task-1 parks. It takes a *factory*, so an
abandoned deferral cannot leave a never-awaited coroutine. `_MAX_IDLE_DEFERRALS
= 10000` is a safety valve: exhausting it logs a WARNING and schedules anyway,
so the path cannot go silent.

Measured in the shipped app: **`deferred 2 loop pass(es)`** on both launches.

`MyVoiceApp._compile_warmup_entrypoint(coro_factory)` is the task body. It
checks the hydration handle and **does not await it** — the ordering is
structural rather than enforced by a barrier:

* hydration is scheduled before `await preload_model(...)` and its body is
  fully synchronous (no `await` in the scan), so it completes in a single step
  at that first suspension — measured finishing ~4.5 s before the preload
  returns;
* this task is created later still, on the first idle pass after the whole
  startup window.

If hydration somehow has not finished, the entrypoint logs a WARNING and
proceeds; a BASE resident then skips itself with `no_priming_prompt`, the
designed safe fallback. It never blocks startup and never switches models.

**Qt main thread** (AC #1 last clause, Task 1.2): still fire-and-forget.
`_initialize_services_async` returns immediately and UI construction proceeds
exactly as before; the deferral loop yields to Qt on every pass rather than
blocking it. `test_warmup_is_not_awaited_inline_on_the_startup_path` guards
against "fixing" this by awaiting the warmup inline.

**Neither a sleep nor a readiness poll.** The re-arm condition is not a timer
and not "is the model ready yet" — it is the loop-reentrancy invariant
`_enter_task` itself enforces, and it clears as soon as that invariant holds.

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

**`tests/unit/test_app_compile_warmup_sequencing.py` — 11 rows.**

Guards the ordering defect **at the call site**, which is where it lived
(per `memory/code_review_regression_test_exact_class.md`):

* AST invariant: every compile-warmup schedule line in
  `_initialize_services_async` must sit below every `preload_model` line. The
  two markers are **unioned, not fallback-ordered**, so re-adding a direct
  `warmup_compile_async()` above the preload while the wrapper stays below is
  still caught.
* AST invariant: the warmup is never `await`ed inline on the startup path.
* **AST invariant: the hand-off goes through `_run_async_task_when_loop_is_idle`
  and the warmup is never handed to the plain `_run_async_task`.** This row was
  added after a mutation escaped: reverting the call site to the plain
  scheduler left the qasync file green, because those rows drive the scheduler
  helper directly and never read app.py's call site. The qasync file proves the
  mechanism; this row proves the shipped code uses it.
* Source invariant: the hydration task handle is retained.
* Behavioural: the entrypoint runs the warmup with hydration done; **proceeds
  without awaiting** an unfinished hydration (bounded by `wait_for` in the test,
  so a regression to a blocking wait fails rather than hangs); **warns** when
  hydration is unfinished, so the degraded launch is not silent; runs with no
  hydration task at all; the hand-off re-arms while a task is mid-step
  (`asyncio.current_task` faked for two passes) and gives up bounded rather
  than never scheduling.
* `_run_async_task` returns its future.

The file carries an explicit note that these rows run on a plain asyncio loop
where the qasync hazard is **structurally absent**, so nobody adds a "warmup
runs" row here and believes it protects the shipped path — that is precisely
the mistake that let the first fix ship broken.

### 2.2 Changed

`tests/unit/services/test_compile_priming_audio_suppression.py` — one rig line.
`_build_true_stream_service` now declares `CUSTOM_VOICE` resident on the
registry. The pre-20.3 rig stubbed `get_loaded_model()` while leaving
`current_model_type` at `None`, a state the real registry cannot be in
(`get_loaded_model` returns `None` exactly when there is no resident type).
No assertion in that file changed; all 12 rows still pass.

### 2.4 Verification under a REAL qasync loop (new)

**`tests/unit/test_app_compile_warmup_qasync.py` — 3 rows**, each spawning
`tests/unit/_qasync_warmup_driver.py` in a fresh interpreter.

The driver reproduces the launch shape exactly: schedule hydration
fire-and-forget → `await preload_model(...)` (a real suspension) → hand off the
warmup → **40 synchronous `processEvents()` calls inside Task-1** (standing in
for `main.py:397`) → park. It drives the **real** `MyVoiceApp._run_async_task`,
the **real** `_run_async_task_when_loop_is_idle`, and the **real**
`QwenTTSService.warmup_compile_async`, with a stubbed `_run_compile_priming`
that suspends five times so the task must survive many steps.

| row | assertion |
|---|---|
| `test_warmup_reaches_the_metric_under_a_real_qasync_loop` | exactly one `tts_compile_warmup_priming`, reason `primed_warm`, zero re-entrancy errors |
| `test_the_plain_hand_off_is_destroyed_under_the_same_loop` | **non-vacuity control**: the pre-fix shape records **nothing** and raises `Cannot enter into task` |
| `test_shield_wait_for_hand_off_is_also_destroyed` | the exact shape 20.3 first shipped, pinned |

Rows 2 and 3 are what make row 1 mean anything: they prove the rig actually
reproduces the hazard. **Row 2 is the check that would have caught the original
bug.**

**Why out of process.** Standing a `qasync.QEventLoop` up inside the shared
pytest Qt session hangs the whole suite — measured: `tests/unit` went from 56 s
to >600 s, because `QApplication.exec()` does not return cleanly once other
modules have created Qt state. A fresh process is safer *and* a more faithful
reproduction of a launch. Cost: ~17 s for the file (three torch imports).

### 2.3 Mutation testing — 22 of 22 caught

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
| M9 | unfinished-hydration warning silenced | CAUGHT (1 failed) |
| M10 | hydration handle no longer retained | CAUGHT (1 failed) |
| M11 | warmup block moved back above the preload | CAUGHT (1 failed) |
| M12 | pre-20.3 direct `warmup_compile_async()` above the preload | CAUGHT (1 failed) |
| N1 | hand-off reverted to the plain `_run_async_task` | CAUGHT (1 failed) |
| N2 | qasync deferral guard removed | CAUGHT (1 failed) |
| N3 | deferral condition replaced by a constant | CAUGHT (2 failed) |
| N4 | hand-off moved back above the preload | CAUGHT (1 failed) |
| N5 | `_run_async_task` stops returning the future | CAUGHT (1 failed) |
| N6 | hydration handle no longer retained | CAUGHT (1 failed) |

Two mutations were **MISSED on a first pass** and both were closed by adding a
row, not by weakening the mutation:

* **M4** (model-id half of the AC #3 guard) — the row that swaps the model also
  contradicted the primed type, so the type check alone was carrying it. Two
  rows were added that isolate the identity signal.
* **N1** (call site reverted to the plain scheduler) — the qasync rows drive
  the scheduler helper directly and never read app.py's call site, so the
  mutation was invisible to them. `test_warmup_hand_off_uses_the_qasync_safe_scheduler`
  closes it. This is the same class of gap that let the first AC #1 fix ship:
  a test that exercises a mechanism is not a test that the product uses it.

No fix in this story rests on an assertion that does not exist.

---

## 3. Regression sweep (AC #5, Task 5)

`python310\python.exe -m pytest -q`, portable interpreter per
`memory/test_interpreter_portable_python310.md`.

| surface | result | vs. Story 20.2 baseline |
|---|---|---|
| `tests/unit/services tests/unit/observability tests/unit/models` | **928 passed, 0 failed** | 896 → 928; **zero failures, unchanged** |
| `tests/unit` (whole tree) | 1,548 passed, 30 failed, 4 errors | identical failure set, **all pre-existing** — verified by stashing this story's source + test changes and re-running `tests/unit/ui/dialogs/voice_design_studio`: **30 failed, 5 errors** on the baseline too (the error count flakes 4↔5 on a Windows temp-file lock, as 20.2 also recorded) |
| `tests/integration tests/test_qwen_tts_internals.py` | 174 passed, **4 failed** | exactly the 4 pre-existing rows 20.2 documented |
| `tests/services tests/settings tests/utils` | 288 passed, **7 failed** | exactly 20.2's 7 pre-existing |
| `tests/ui` | 735 passed, **7 failed** | exactly 20.2's 7 pre-existing |
| the whole story surface in one invocation | **64 passed** | — |
| new: `test_compile_priming_resident_model.py` | **26 passed** | — |
| new: `test_app_compile_warmup_sequencing.py` | **11 passed** | — |
| new: `test_app_compile_warmup_qasync.py` | **3 passed** (~17 s, 3 subprocesses) | — |
| unchanged: `test_qwen_tts_service_compile_warmup.py` | **12 passed** | 12 |
| rig-touched: `test_compile_priming_audio_suppression.py` | **12 passed** | 12 |

**Suite runtime.** `tests/unit` goes from 56 s to 128 s: the qasync file spawns
three fresh interpreters and each pays a torch import. That is the price of
testing the hazard in the only place it exists. It was weighed against the
alternative — an in-process qasync loop, which hangs the suite past 600 s.

**Zero new failures.** Every failure above is in the same UI / voice-profile /
session-manager drift set `20-2-warm-path-compile-priming-evidence.md` §5
documents, none of it touching the streaming dispatch chain.

---

## 4. AC #1 verified in the app; AC #4 measurement still **PENDING**

### 4.0 AC #1 — verified end-to-end in the shipped application ✅

The compile warmup runs at **startup**, with no user interaction, so AC #1 is
verifiable without a Generate click. Two real launches
(`python310\python.exe src/myvoice/main.py`, `tts_compile=auto`, bf16, a CLONED
voice active so BASE is the resident model), reading `logs/myvoice.log`:

**Launch 1 — 2026-09-01 09:47** (cold marker: priming had literally never run
in this app, so `mark_warm` had never been called):

```
09:47:33,917  MyVoiceApp    - Voice clone prompt cache hydration: (13, 14)
09:47:38,225  MyVoiceApp    - Model Base (Clone) preloaded successfully
09:47:38,593  MyVoiceApp    - torch.compile warmup handed off to the event loop
                              (deferred 2 loop pass(es) for qasync re-entrancy safety)
09:47:38,595  QwenTTSService- Compile priming: dispatching against the resident model Base (Clone)
09:47:43,100  QwenTTSService- Compile warmup primed cache successfully (duration=4516ms)
```

**Launch 2 — 2026-09-01 09:49** (marker now warm):

```
09:49:00,197  MyVoiceApp    - Voice clone prompt cache hydration: (13, 14)
09:49:04,682  MyVoiceApp    - Model Base (Clone) preloaded successfully
09:49:05,063  MyVoiceApp    - torch.compile warmup handed off to the event loop
                              (deferred 2 loop pass(es) for qasync re-entrancy safety)
09:49:05,064  QwenTTSService- Compile priming: dispatching against the resident model Base (Clone)
09:49:09,731  QwenTTSService- Compile cache hit; warm-path priming completed (duration=4672ms)
                              — the inductor reload is now paid for at startup
```

What this establishes:

| claim | evidence |
|---|---|
| **AC #1** — priming runs after the model is loaded | hand-off logged 368 ms *after* "preloaded successfully"; `no_model_loaded` never appears |
| **AC #1** — the qasync fix works in production | `deferred 2 loop pass(es)`, then a clean hand-off; **zero** `Cannot enter into task` / `Task was destroyed` in either launch |
| **AC #2** — the resident model is primed | "dispatching against the resident model **Base (Clone)**" — not the pre-20.3 hard-coded CustomVoice |
| **AC #3** — key and priming agree | launch 1 reached `mark_warm` (the coherence guard passed); launch 2 read the marker back as a cache hit |
| expected AC #4 telemetry reason | launch 2 is `primed_warm`, exactly what §4.1 step 5 says to look for |
| startup cost | ~4.5–4.7 s, matching Story 20.2 §3.4's ~4.4–4.9 s estimate |

The only `ERROR` in either launch is an unrelated virtual-audio-device probe
(`Virtual device -1 validation failed`) from a host with no VB-Cable configured.

**Note for the AC #4 re-run:** because priming had never once succeeded before,
the compile-cache marker was cold. Launch 1 above paid `primed_cold` and marked
it. A re-run therefore starts warm and should read `primed_warm` from its first
launch — which is the state AC #4 asks to measure.

### 4.1 AC #4 — the TTFA measurement: still not performed

**Status: not performed.** AC #4 requires ≥5 launches of the real application
with a CLONED voice active and a human pressing Generate on each. That is an
operator task on the Ampere+ host, not something the implementation pass can
produce. **The story is not closeable until this section is filled in.**

Everything AC #4 needs is in place: §4.0 shows the GUI path is now genuinely
live, and the shipped capture Story 20.1 built for exactly this
(`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`, driven by
`10_Story_20.3_AC4_GUI_Capture.bat`) is unchanged.

**Diagnostics retained (commit `6428601`).** The two silent warmup gates
(`no_model_registry`, `no_model_loaded`) log at **INFO**, and
`tts_compile_warmup_priming` is in the CSV capture set. Both stay: they are
what made this diagnosable at all — the first negative pass was uninterpretable
because those gates logged at DEBUG and the app runs at INFO. Story 20.3 adds
to that discipline rather than trimming it: the hand-off itself now logs at
INFO with its deferral count, priming logs which resident model it dispatches
against, and an unfinished hydration logs a WARNING. A startup path this epic
depends on should not be able to exit without saying so.

### 4.2 Procedure

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

### 4.3 Table to fill

| launch | 1a model load | **1b first-forward** | 2 talker | 3 decode | TTFA(post) | telemetry reason |
|---|---:|---:|---:|---:|---:|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| **median** | | | | | | |

### 4.4 What to compare against

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

1. **AC #4's TTFA numbers are unmeasured.** §4.0 establishes that priming
   now runs, primes the right model, and completes — but not that the user's
   first generation got faster. Treat §0 as "defects closed and priming
   confirmed running", not "win delivered", until §4.3 is filled.

1a. **The qasync hazard is general, and this story only fixes one call site.**
   Every `_run_async_task` in the app is exposed to the same failure whenever
   its task's steps are delivered while another task is mid-step pumping Qt
   events. The startup window is uniquely dangerous (a long synchronous Qt
   stretch immediately follows), which is why it broke there first and why the
   fix is scoped there — every other caller schedules from a Qt signal handler
   with no task on the stack, where the plain path is correct and has shipped
   for many releases. But a suspended warmup resuming *during* a user
   generation that pumps Qt events could still be destroyed mid-priming. The
   consequence is bounded and non-fatal (`priming_failed`, cache stays cold,
   next launch retries), and in practice priming completes before the user can
   generate. **Not audited beyond the startup path in this story.**
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
  * `_initialize_services_async` — hydration handle retained; warmup hand-off
    moved below the preload and routed through the qasync-safe scheduler
  * `_run_async_task_when_loop_is_idle` (new) + `_MAX_IDLE_DEFERRALS`
  * `_compile_warmup_entrypoint` (new) — hydration check + warmup
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
* `tests/unit/test_app_compile_warmup_sequencing.py` (new, 11)
* `tests/unit/test_app_compile_warmup_qasync.py` (new, 3) +
  `tests/unit/_qasync_warmup_driver.py` (new, out-of-process driver)
* `tests/unit/services/test_compile_priming_audio_suppression.py` (rig line
  only — registry declares a resident model type)
