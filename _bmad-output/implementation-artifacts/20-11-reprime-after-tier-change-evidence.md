# Story 20.11 — Re-prime after a tier change, prime through TRUE_STREAM: evidence

Branch `story-20-11-reprime` from `main` (`013caa7`). Implemented 2026-09-14.
Baseline before this story: 2,992 passed, 1 xfailed, 0 failed, 0 errors.

Source: Story 20.10 evidence §5, the two follow-ups raised from the RTX 3060
build-58 log — a tier switch to `quality` cost 19.2 s to first chunk (17.07 s
model load + cold compile inside the request), and priming under the
sentence-stream override left the first TRUE_STREAM generation to pay the
`CodecStateCache` self-test (3.08 s vs 1.86 s warm).

---

## 0. Headline

* **AC #1/#2** — `_on_settings_changed`'s `set_quality_tier` hand-off now
  passes `_on_quality_tier_updated` as `on_success`. When `changed` is True
  it schedules, through `_schedule_when_loop_is_idle` and inside
  `_compile_warmup_entrypoint`, `_reprime_after_tier_change`: re-hydrate the
  voice_clone_prompt cache for the new tier → preload the active profile's
  model (else CUSTOM_VOICE) → `warmup_compile_async` (untouched). Preload
  failure or raise: WARNING, no priming, gate terminal state `False`.
  `changed=False`: nothing scheduled.
* **AC #3** — `_run_compile_priming` dispatches with
  `effective_streaming_mode(None)` (the D-9 hardware probe) instead of
  `_resolve_streaming_mode()`. The public generators still read the override.
  One extra INFO line names the priming mode; the existing "dispatching
  against the resident model" line is byte-identical.
* **AC #4** — classification below (§2): the Qt slot has no task current;
  the continuation runs inside the `set_quality_tier` task (`Task-2` in the
  driver). Not `None` → idle scheduler, proved under the real qasync loop
  with a control row that is destroyed (§2.2).
* **AC #5** — startup call site untouched (`git diff` shows no hunk in
  `_initialize_services_async`); Story 20.3/20.9's pins pass unchanged; the
  full suite result is in §5.
* **A finding the story did not anticipate** (§3): without a re-hydration
  step, the cloned-voice re-prime would have skipped with
  `no_priming_prompt` on every tier change — the startup hydration is
  per-tier and the BASE priming lookup is in-memory only.
* **A test-harness finding** (§6): the out-of-process qasync driver was
  importing the *main checkout's* `src` when run from a worktree, because
  the portable interpreter's `python310._pth` ignores `PYTHONPATH`. Fixed in
  the driver (it now inserts its own repo's `src`), not in `python310`.

---

## 1. What changed (`git diff --stat`)

| file | change |
|---|---|
| `src/myvoice/app.py` | `_on_settings_changed`: `on_success=self._on_quality_tier_updated` (+ a why-comment). New `_on_quality_tier_updated` (the continuation; logs the pre-20.11 line unchanged, schedules the re-prime iff `changed`). New `_reprime_after_tier_change` (hydrate → preload → warmup). `_compile_warmup_entrypoint` docstring: one paragraph on its reuse. |
| `src/myvoice/services/qwen_tts_service.py` | `_run_compile_priming`: `priming_mode = effective_streaming_mode(None)`; one new INFO line; comment above the request build corrected (it claimed the `_resolve_streaming_mode()` fork). `warmup_compile_async` untouched. |
| `tests/unit/_qasync_call_site_driver.py` | three `tier_change_*` variants; own-repo `src` on `sys.path` (§6). |
| `tests/unit/test_app_reprime_after_tier_change.py` | new, 19 rows (§4). |
| `tests/unit/services/test_compile_priming_streaming_mode.py` | new, 5 rows (§4). |

Not touched: `warmup_compile_async` (gates, telemetry reasons, docstring),
the startup hand-off in `_initialize_services_async`,
`_run_async_task_when_loop_is_idle`, `_schedule_when_loop_is_idle`, any
`QMessageBox`.

---

## 2. AC #4 — scheduling classification

Per Story 20.9's scheme (evidence §2): the *site* fixes which task is on the
stack when the first step is armed; a task dies iff one of its steps is
delivered while another task is mid-step pumping Qt.

| point in the chain | `asyncio.current_task()` | class | measured (driver `tier_change_fixed`) |
|---|---|---|---|
| `_on_settings_changed` (the `settings_changed` slot, Settings-dialog OK) | `None` | **(a)** — 20.9 §2.3 #33 | `"handler_current_task": null` |
| `_on_quality_tier_updated` — the `on_success` continuation, called from `_handle_task` after `await set_quality_tier(...)` returned | the `set_quality_tier` wrapper task | **(b0)** — 20.9 §2.4 shape | `"continuation_current_task": "Task-2"` |
| the idle callback `_schedule_when_idle` (next Qt pass) | `None` | (a) by construction | `"deferrals": [0]` |
| `_reprime_after_tier_change`'s first step | — | created on that idle pass | events all `@parked` |

**Conclusion.** The current task at the moment the continuation schedules is
**not `None`** — it is the `set_quality_tier` task's own `_handle_task`
wrapper. The AC's rule therefore applies and the site uses
`_schedule_when_loop_is_idle`. The written argument for why a plain
`_run_async_task` would *also* survive today (the parent's remaining
synchronous stretch is `return` → task completes, no Qt pump; Task-1 is
parked at `app_close.wait()`) is the same as 20.9 §2.4's for #27–#31 and is
recorded in the method's docstring — but the idle scheduler is the shape
that keeps surviving if a future edit opens a dialog from this continuation,
and it costs exactly one loop pass (`deferred 0 loop pass(es)` on the
shipped path).

### 2.1 Driver rows (real `QApplication` + `qasync.QEventLoop`, out of process)

Task-1 (the `run_until_complete` coroutine) parks at `await asyncio.sleep`,
as at `app_close.wait()`; the handler is invoked from `QTimer.singleShot(0)`
— a plain Qt slot with no task on the stack, exactly as the signal delivers
it. Fakes: `set_quality_tier` (yields once, returns True), `hydrate` (sync
body), `preload_model` (spans passes), `warmup_compile_async`;
`get_active_profile_model_type()` → BASE.

```
tier_change_fixed
  handler_current_task=None  continuation_current_task=Task-2
  events=[tier_set@parked, hydrate@parked, preload_start:BASE@parked,
          preload_done@parked, warmup@parked]
  deferrals=[0]  reentrancy_errors=[]  destroyed_pending=[]

tier_change_fixed_pumped      (shipped continuation + 0.2 s Qt pump inside it)
  continuation_current_task=Task-2  pump_passes=20
  events=[tier_set@parked, hydrate@parked, preload_start:BASE@parked,
          preload_done@parked, warmup@parked]
  deferrals=[20]  reentrancy_errors=[]  destroyed_pending=[]

tier_change_plain_pumped      (control: plain _run_async_task from the
                               continuation + the same pump)
  continuation_current_task=Task-2  pump_passes=19
  events=[tier_set@parked]
  reentrancy_errors=["RuntimeError: Cannot enter into task <Task-4 ...
     _handle_task() running at .../src/myvoice/app.py:1396> while another
     task <Task-2 ... _handle_task() running at .../app.py:1400> is being
     executed."]
  destroyed_pending=["Task was destroyed but it is pending!"]
```

### 2.2 What the three rows prove

* `fixed`: the shipped path runs once, in production order, with Task-1
  parked and no deferral needed (the parent completed before the idle
  callback's pass).
* `plain_pumped` is the **non-vacuity control**: with the pre-fix scheduling
  shape, a pump inside the continuation delivers the re-prime's first step
  while `Task-2` is still current and `_enter_task` kills it — the re-prime
  never runs (no `hydrate`, no `preload`, no `warmup`). If this row ever goes
  green the rig has stopped reproducing the hazard.
* `fixed_pumped`: the shipped continuation under the same pump defers on
  every one of the 20 pump passes (the parent is current for all of them),
  creates the task after the parent completes, and runs it once.

The `app.py:1396`/`:1400` in the control's error message are inside the
**worktree's** `app.py` — the driver imports the code under test (§6).

---

## 3. Finding: the re-prime needs a re-hydration step

The story's AC #1 lists "(1) preload … (2) `warmup_compile_async`". Reading
the two lookups the BASE priming request depends on:

* `hydrate_voice_clone_prompt_cache` reads `self._model_registry.quality_tier`
  **once** and loads `<voice>.<tier>.pt` for that tier only — the RTX 3060
  log says so: `hydrated 12/12 CLONED voices for tier small`.
* `_active_profile_voice_clone_prompt` is **in-memory only** (its docstring:
  "never touches disk beyond the stat"), keyed `(ref_audio, tier)`.

So after `small → quality` the in-memory cache has no `quality` entry for
the active voice, and a re-prime without re-hydration would raise
`CompilePrimingSkipped("no_priming_prompt")` on every cloned-voice tier
change — the shipped default state. The user's own generation does not hit
this because Story 17.2's lazy path reads the disk (`Voice clone prompt
cache hit on disk: …Sarira-F.quality.pt`, 20.10 §5).

`_reprime_after_tier_change` therefore awaits
`hydrate_voice_clone_prompt_cache()` first. The registry's `quality_tier` is
already the new tier at that point (`ModelRegistry.set_quality_tier` flips
it before taking the lock to unload). The scan is idempotent and its body is
synchronous (the same property Story 20.3 §1.1a relies on), so awaiting it
inline adds no task. Log line: `Voice clone prompt cache hydration after
tier change: (hits, total)`. A raise there is a WARNING and does not cost
the preload or the prime. Pinned by
`test_reprime_rehydrates_for_the_new_tier_before_priming` and the AST row
(`hydrate` < `preload_model` < `warmup_compile_async` by line).

---

## 4. Tests

### 4.1 `tests/unit/test_app_reprime_after_tier_change.py` (19)

Plain loop, behaviour:

| row | AC |
|---|---|
| `test_tier_change_reprimes_in_startup_order` — events `[set_app_settings, set_quality_tier:quality, hydrate, preload:BASE, warmup]`, gate stream `[False]` | #1 |
| `test_tier_change_falls_back_to_custom_voice_without_an_active_profile_model` | #1 |
| `test_no_change_schedules_nothing` — `changed=False`: no hydrate/preload/warmup, gate untouched | #1 |
| `test_same_tier_in_settings_never_calls_set_quality_tier` — the pre-existing guard | #1 |
| `test_switching_back_reprimes_again` — two changes, two preloads, two warmups | #1 |
| `test_preload_failure_skips_priming_warns_and_leaves_the_gate_released[preload-returns-false]` / `[preload-raises]` — no warmup, WARNING names why, gate stream exactly `[False]` | #2 |
| `test_hydration_failure_does_not_cost_the_preload_or_the_prime` | #1 |
| `test_reprime_rehydrates_for_the_new_tier_before_priming` | §3 |
| `test_continuation_hand_off_log_lines_use_the_idle_scheduler` — the three hand-off lines with this site's label | #4 |
| `test_continuation_runs_inside_the_set_quality_tier_task` — `current_task()` in the continuation is a `_handle_task` task, not `None` | #4 |

AST pins:

| row | guards |
|---|---|
| `test_settings_handler_wires_the_continuation_as_on_success` | reverting `on_success` to the log-only lambda |
| `test_continuation_routes_through_the_idle_scheduler_and_the_entrypoint` | plain `_run_async_task`/`create_task`/`ensure_future` in the continuation; dropping the entrypoint |
| `test_reprime_body_preloads_then_primes_through_the_untouched_warmup` | calling `_run_compile_priming` directly; order hydrate < preload < warmup |
| `test_startup_call_site_is_byte_identical` | exactly one idle hand-off of `warmup_compile_async` at startup; no re-prime reference there | 

Real qasync loop (module-scoped fixtures, one subprocess each):

| row | driver variant |
|---|---|
| `test_qasync_classification_slot_has_no_task_and_continuation_does` | `tier_change_fixed` |
| `test_qasync_shipped_reprime_runs_once_in_order_with_task1_parked` | `tier_change_fixed` |
| `test_qasync_control_plain_scheduling_from_the_continuation_is_destroyed` | `tier_change_plain_pumped` |
| `test_qasync_shipped_reprime_survives_a_pump_inside_the_continuation` | `tier_change_fixed_pumped` |

### 4.2 `tests/unit/services/test_compile_priming_streaming_mode.py` (5)

| row | AC |
|---|---|
| `test_priming_ignores_the_sentence_stream_override_but_user_generations_honour_it[cuda]` — same service, override `sentence_stream`: priming dispatch sees TRUE_STREAM, `generate_custom_voice` sees SENTENCE_STREAM | #3 (load-bearing) |
| `…[cpu-only]` — both SENTENCE_STREAM: the priming resolver is the D-9 probe, not a hard-coded TRUE_STREAM | #3 |
| `test_priming_mode_on_cuda_is_true_stream_for_the_common_configurations[auto]` / `[true_stream]` | #3, no change where they agreed |
| `test_priming_mode_is_resolved_at_dispatch_time_not_cached` — two primes in one process under different probe results | 20.11 re-prime reuses the body |

### 4.3 Existing rows re-run unchanged

`test_app_compile_warmup_sequencing.py` (11), `test_app_qasync_call_sites.py`
(12), `test_app_compile_warmup_qasync.py`, `test_compile_priming_resident_model.py`,
`test_compile_priming_generate_gate.py`, `test_qwen_tts_service_compile_warmup.py`,
`test_compile_priming_audio_suppression.py`: 125 passed together with the
24 new rows (one run, 78 s).

---

## 5. Mutation check

Each mutation applied to the worktree source, the two new files run, then
reverted (`git diff --stat` afterwards matched the pre-mutation stat). The
qasync rows were deselected for M1–M7 except M3, where they are the point.

| # | mutation | caught by |
|---|---|---|
| M1 | `_run_compile_priming`: `priming_mode = self._resolve_streaming_mode()` (the AC #3 defect) | `test_priming_ignores_the_sentence_stream_override…[cuda]`, `test_priming_mode_is_resolved_at_dispatch_time_not_cached` — 2 failed, 18 passed |
| M2 | `on_success` reverted to the pre-20.11 log-only lambda (the AC #1 defect) | 9 failed: every behaviour row that expects a re-prime + `test_settings_handler_wires_the_continuation_as_on_success` |
| M3 | continuation schedules through the plain `_run_async_task` | 4 failed: the AST pin, the log-lines row, **and both shipped qasync rows** (`…survives_a_pump_inside_the_continuation` — the re-prime is destroyed; `…runs_once_in_order…` — no hand-off line, `deferrals != [0]`) |
| M4 | `if not changed: return` guard dropped | `test_no_change_schedules_nothing` — 1 failed |
| M5 | preload `success=False` no longer returns before the warmup | `test_preload_failure_skips_priming…[preload-returns-false]` — 1 failed |
| M6 | re-hydration step dropped | 4 failed: `…reprimes_in_startup_order`, `…hydration_failure…`, `…rehydrates_for_the_new_tier…`, the AST row |
| M7 | `_compile_warmup_entrypoint` bypassed (factory is `_reprime_after_tier_change` directly) | 4 failed: gate stream is `[]` not `[False]` in `…reprimes_in_startup_order` and both `…preload_failure…` rows; the AST pin |

Every fix has at least one row that is red without it; M3's catch includes
a real-loop behavioural row, not only the AST pin.

---

## 6. Test-harness finding: the driver imported the wrong `src`

First run of the new driver variants from the worktree failed with
`RuntimeError: super-class __init__() of type MyVoiceApp was never called`
at `app_obj._on_quality_tier_updated` — QObject's `__getattr__` for a
*missing* attribute. The method exists in the worktree's `app.py`; the
driver had imported `I:\MyVoiceV2\src\myvoice\app.py`.

Cause: `I:\MyVoiceV2\python310\python310._pth` lists `..\src`, and a `._pth`
file puts the interpreter in isolated mode — `PYTHONPATH` is ignored. The
`env["PYTHONPATH"] = REPO_ROOT / "src"` that `test_app_qasync_call_sites.py`
(and now the new file) sets never reached the subprocess. From the main
checkout the two paths coincide, so Story 20.9's rows were never wrong
there; from any worktree they were exercising `main`'s code.

Fix (test-side only; `python310` untouched): the driver inserts
`Path(__file__).resolve().parents[2] / "src"` at `sys.path[0]` after the
torch/PyQt6/qasync imports and before the first `myvoice` import — the
out-of-process half of what `tests/conftest.py` does in-process. Verified by
the control row's error text, which names the worktree's `app.py` path.

---

## 7. Full suite

Command: `I:\MyVoiceV2\python310\python.exe -m pytest tests -q -p no:cacheprovider`
from the worktree root, one run.

```
3016 passed, 1 xfailed in 135.18s (0:02:15)
```

2,992 baseline + 24 new rows = 3,016; the xfail is the pre-existing one.

Failure-set identity: empty before (baseline 2,992 passed / 1 xfailed on
`main`), empty after.
