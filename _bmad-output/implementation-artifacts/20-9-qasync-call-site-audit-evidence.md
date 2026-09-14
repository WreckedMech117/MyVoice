# Story 20.9 — qasync Call-Site Audit (F7): evidence

Branch `story-20-9-qasync-audit` on top of tooling-5 + ui-2. Implemented 2026-09-14.
Baseline before this story: 2,978 passed, 1 xfailed, 0 failed, 0 errors.

---

## 0. Headline

* **All 71 scheduling sites classified by call path** (§2). The exposure set
  is small and entirely startup-shaped: the only task in the app that pumps
  Qt events on a live path is `main.py`'s Task-1, and it does so in exactly
  two windows — the post-init splash/UI stretch (W1) and shutdown (W2).
  Everything scheduled from a Qt signal handler after startup runs with
  Task-1 parked at `app_close.wait()`, where the plain path is correct.
* **Two sites moved** to the idle-pass scheduler, each with a control row
  that reproduces the pre-20.9 shape being destroyed under a real qasync
  loop and a fixed row that proves the shipped code survives (§3, §4):
  * `_on_voice_service_started` — voice restoration (was a 500 ms `QTimer`
    + `loop.create_task` guess at "init is done");
  * `_setup_api_server_from_settings` — the local TTS API start, whose
    uvicorn `serve()` task is long-lived and wakes every 100 ms.
* **Three sites left with a written argument** (§3.4): hydration (`app.py:589`,
  Story 20.3's verified chain), device-change monitoring (`app.py:4217`),
  and the profile-cache save inside the restore — all complete inside
  Task-1's ≥3 s preload suspension on every logged launch. Eleven more match
  the spec's literal "inside a running task" wording but can never overlap a
  pump (§2.4, (b0)); left, with each stack stated.
* **Nothing touched** in `warmup_compile_async`, its call site, or the audio
  dispatch chain. `git diff --stat` on `src/`: `app.py` only.
* **Findings the story did not anticipate** (§6): four sites run on the
  device-monitor *thread* and never create a task (a different defect
  class, reported not fixed); seven sites are unreachable in the shipped app;
  one is dead code; Story 20.3's safety valve has a measurable limit under a
  tight `processEvents` spin; `cleanup_async` re-enters `close()` inside
  Task-1 at shutdown.
* Full suite, one run: **2,990 passed, 1 xfailed, 0 failed, 0 errors** (failure-set identity unchanged:
  empty before, empty after).

---

## 1. The mechanism, restated precisely

Under qasync `loop.call_soon(cb)` is `call_later(0, cb)` →
`_SimpleTimer.add_callback` → `QObject.startTimer(0)`. Qt registers a
zero-interval timer as a posted `QZeroTimerEvent`, so the callback is
delivered from `timerEvent` on the next Qt event pass — *any* pass, including
one run synchronously by `QCoreApplication.processEvents()`, a modal
`exec()`, or `QSplashScreen.finish()` from inside a task step.

A task step or wake-up (`TaskStepMethWrapper` / `TaskWakeupMethWrapper`) is
such a callback. `_asyncio.task_step` calls `enter_task(loop, task)` first,
which raises `RuntimeError: Cannot enter into task X while another task Y is
being executed` if `_current_tasks[loop]` is set — i.e. if the pump is being
run from inside Y's step. `Handle._run` reports it to the loop's exception
handler and the step is not retried. The task is now a zombie: pending, its
`_fut_waiter` already done, never rescheduled. If nothing holds a reference it
is collected with `Task was destroyed but it is pending!`; **if something does
hold a reference there is no second line at all** (this is the API-server
case, §4.4).

Three consequences that drive the classification:

1. **The scheduling site is not the variable; the pumping task is.** A task
   dies iff one of its steps is delivered while some *other* task is mid-step
   pumping Qt. The site matters only in that it fixes *which task is on the
   stack when the first step is armed* and *how long the task lives*.
2. **A pump with no task on the stack is harmless.** Modal dialogs opened
   from a Qt signal handler (`settings_dialog.exec()`, `QMessageBox.question`
   in a click slot) run a nested Qt loop with `current_task() is None`; tasks
   step normally inside it. This is why the settings dialog, Voice Design
   Studio and every confirmation box have worked for years.
3. **Deferring creation guards the first step only.** A long-lived task
   created on an idle pass is still exposed at every later wake-up if any
   task pumps later. The idle helper is therefore correct only in
   combination with "Task-1 does not pump again until shutdown", which holds
   (§2.1).

### 1.1 Where Qt is pumped from inside a task (the complete inventory)

An AST pass over `src/myvoice` for `processEvents`, `.exec()`, and the static
`QMessageBox`/`QFileDialog`/`QInputDialog` calls, with the enclosing function
and its callers traced (`scratchpad/pumps.py`):

| window | where | inside which task | live? |
|---|---|---|---|
| **W1** startup | `main.py:372-400` (`processEvents` ×4, `splash.showMessage` ×3), `main.py:404` `splash.finish(main_window)` (spins `processEvents` + `msleep(10)` until the window is exposed, ≤1 s) | Task-1 (`async_main`), *after* `initialize_async()` returns | **yes** — measured 133 ms from "initialization completed successfully" to "initialized successfully" on six launches |
| **W2** shutdown | `cleanup_async` → `self._main_window.close()` (`app.py:927`) → `closeEvent` → `QMessageBox.question` (unless `_force_quit`) and `_wait_for_pending_audio_drain` (`processEvents`, measurement mode only) | Task-1, after `lastWindowClosed` woke it | yes, but the API server, TTS, Whisper and the coordinator are already stopped above it, and `_os._exit` follows — a destroyed task here is inconsequential |
| **W3** init failure | `initialize_async` except-path → `_show_error_dialog` → `msg_box.exec()` | Task-1 | app returns 1 immediately after |

Every other pump site in the codebase is a user-click slot, a `closeEvent`
delivered by Qt, or a `QThread`-signal slot — no task on the stack. No
`async def` outside `main.py` contains a pump, and no `on_success`/`on_error`
callback passed to `_run_async_task` opens a dialog (each was read).

Empirical cross-check: `logs/myvoice.log` (4.2 MB, launches back to May)
contains exactly 13 `Cannot enter into task` lines, every one of them Task-6
(`_handle_task`, the warmup) against Task-1 at `main.py:397`, dated
2026-08-31/09-01 — the pre-20.3 defect and nothing else.

### 1.2 What "exposed" means for this audit, per site

A site is **EXPOSED** if the task it creates can have a step or wake-up
delivered during W1. Given §1.1 that reduces to: *scheduled from inside
Task-1, or from a Qt timer that can fire inside W1, with a task that is
still pending (or is woken) when Task-1 reaches W1.* Task-1 suspends for real
inside `initialize_async` at `await preload_model(...)` (3.4–4.5 s on every
logged launch; §2.2), so a task created before that and completing inside it
is safe by timing but not by construction — which is the AC #2 "written
argument" case.

---

## 2. AC #1 — the 71 sites

Line numbers are those of `HEAD` (`d433fd5`), i.e. before this story's edits,
so they match the spec's count (47 in `app.py` after excluding the three
comment-only grep hits, 24 elsewhere). "now" gives the post-edit location
where it moved.

Classes: **(a)** no task on the stack at scheduling time and no live pump can
overlap the task → safe by construction. **(b)** exposed: scheduled inside a
running task or from a path that pumps before the task's first step, *with a
live pump window that can overlap it*. **(b0)** literal-(b) shape — scheduled
inside a running task's step (an `on_success`/`on_error` callback or a
service coroutine) — but the parent's remaining synchronous stretch contains
no Qt pump and Task-1 is parked for the whole interactive phase, so no step
can be delivered re-entrantly; reported separately so the reviewer can see
every site that matches the spec's literal wording. **(t)** runs on a
non-loop thread and never creates a task (different defect, §6.1). **(x)**
unreachable in the shipped app.

### 2.1 `app.py` — the startup chain (called from inside Task-1)

| # | site (HEAD) | enclosing | stack at scheduling | class | evidence / argument |
|---|---|---|---|---|---|
| 1 | `app.py:589` `_run_async_task(hydrate_voice_clone_prompt_cache())` | `_initialize_services_async` | Task-1, directly | **(b)** left, argued | Body is fully synchronous (20.3 §1.1a): one step, delivered at Task-1's next yield (`await preload_model`), measured 13–14 ms after "Preloading model" on all six launches (§2.2), ~3.4 s before W1. The returned handle is load-bearing for `_compile_warmup_entrypoint`; the idle helper returns `None`. Not moved. Residual: a launch where the preload raises synchronously *and* mic/API setup never yield reaches W1 with this step still queued → hydration destroyed → BASE prime skips with `no_priming_prompt` (20.3's designed fallback). |
| 2 | `app.py:671` `_run_async_task_when_loop_is_idle(warmup_compile_async)` | `_initialize_services_async` | Task-1 | fixed (20.3) | Untouched per Dev Notes. Still routes through `_compile_warmup_entrypoint`; wrapper now delegates to the generic helper with byte-identical log lines (§3.1). Re-verified: 20.3's 14 rows green, `deferred 40 loop pass(es)` in the driver, `[0, 40]` in `startup_order_fixed`. |
| 3 | `app.py:1088` `_run_async_task(...)` inside `_schedule_when_idle` (now `:1155`) | idle helper | a plain `call_soon` callback on a pass where `current_task() is None` (or the valve fired, with a WARNING) | (a) by construction | This *is* the mechanism. See §6.4 for the valve limit. |
| 4 | `app.py:1180` `asyncio.ensure_future(_handle_task())` (now `:1245`) | `_run_async_task` | inherits its caller's | — | The primitive; classified through its 36 callers. |
| 5 | `app.py:2562` `_run_async_task(_load_app_settings_on_startup())` | `_on_config_service_started` | — | **(x)** dead | No caller anywhere in `src/` (`grep -rn _on_config_service_started`): the startup path awaits `_config_manager.start()` inline and calls `_on_settings_loaded` directly (`:400`). |
| 6 | `app.py:2597` `loop.create_task(delayed_restore())` in `schedule_restore` (now `_schedule_when_loop_is_idle` at `:2681`) | `_on_voice_service_started` ← `_initialize_services_async:573` | `QTimer.singleShot(500)` slot — a Qt timer that fires on whichever pass is 500 ms later, *including a W1 pass*; the slot itself survives (plain callback) but the task it creates arms a zero-timer delivered in the same pump | **(b) → MOVED** | Normal launch: timer fires at t₀+0.50 s, restore runs t₀+1.00 s, done ~2.4 s before the preload returns (§2.2) — survives by the preload's duration, not by design. Fast-fail launch (preload returns without suspending): reproduced destroyed, §4.2 control. Moved to `_schedule_when_loop_is_idle`; the 0.5 s sleep dropped (it was the guess the helper replaces, and its wake-up was itself an exposed step). Ordering proof §4.3. |
| 7 | `app.py:4217` `asyncio.create_task(_setup_device_change_monitoring_async())` | `_on_audio_coordinator_started` ← `_initialize_services_async:479` | Task-1, directly | **(b)** left, argued | Task body: register callback (sync) + `await start_device_monitoring()`; measured "Setting up device change monitoring" → "started successfully" in 1 ms, 24 ms after coordinator start, i.e. at Task-1's next yield inside `_auto_detect_and_configure_vb_cable`/`tts_service.start()`, ≥3 s before W1 on every logged launch. Same residual as #1 on a synchronously-failing launch (device hot-plug monitoring not started that session; its callbacks are thread-broken anyway, §6.1). Not moved: it would run ~3 s later than today for no live benefit. |
| 8 | `app.py:4911` `_run_async_task(update_device_settings(app_settings))` | `_on_settings_loaded` ← `_initialize_configuration_async:400` and (dead) `#5` | Task-1 at `:400` | **(x)** at startup, dead otherwise | At `:400` `_audio_coordinator` does not exist yet (first assigned `:470`, not in `__init__`) so `hasattr` is False and the branch is skipped — the code's own NOTE says so. The only other caller is #5 (dead). |
| 9 | `server.py:106` `asyncio.ensure_future(self._server.serve())` | `ApiServerController.start` ← `_setup_api_server_from_settings` ← `initialize_async:331` (also ← `_reconcile_api_server`, a `_handle_task`) | Task-1, at the very end of `initialize_async`, ~1–2 ms before W1's first `processEvents()` | **(b) → MOVED** (via its app.py caller) | Long-lived; uvicorn 0.40 `main_loop` is `await asyncio.sleep(0.1)` forever. `start()` yields in `_await_started` (20 ms poll) so the tick is due ≥80 ms after Task-1 resumes and W1's unconditional pump is ~2 ms after — it survives today by an ~80 ms timing accident, and dies if `splash.finish` has to spin. Failure is *silent* (strong ref in `_task`): `is_running` stays True, no ticks, `stop()` hangs into main.py's 8 s `wait_for` → `_os._exit`. Reproduced against real uvicorn, §4.4. Never enabled in any logged launch (0 × "Local TTS API listening"). Start deferred to the first idle pass in `_setup_api_server_from_settings`; the settings-change path is a `_handle_task` (b0) and unchanged. |

### 2.2 Production timing the arguments above rest on

Six consecutive launches in `logs/myvoice.log` (2026-09-14, RTX 5090, BASE
resident), t₀ = "Voice profile service started successfully":

| event | t₀ + | site |
|---|---:|---|
| "Setting up device change monitoring" … "started successfully" | −0.030 s … −0.029 s (24 ms after coordinator start) | #7 |
| "Voice clone prompt cache hydration: (13, 14)" | +0.014 s | #1 |
| "Voice restoration scheduled after initialization delay" | +0.501 s | #6 (legacy) |
| "Restoring voice selection" … "Successfully restored" | +0.997 s … +0.999 s | #6 (legacy) |
| "Model Base (Clone) preloaded successfully" | +3.41 s … +4.53 s | Task-1 resumes |
| "MyVoice application initialization completed successfully" | +0.42 s later | W1 begins |
| "MyVoice application initialized successfully" (main.py) | +0.133 s later (range 0.132–0.136) | W1 ends, Task-1 parks |
| "torch.compile warmup handed off … (deferred 2 loop pass(es))" | +0.001 s later | #2 |

### 2.3 `app.py` — Qt signal handlers, no task on the stack: **(a)**

All emitted by user actions (every `.emit(` in `main_window.py` was read:
button/menu/combo/tray slots, settings-dialog OK, VDS close). None fires
programmatically during `_initialize_ui`. The interactive phase runs with
Task-1 parked at `app_close.wait()`; the only way a click could land inside
W1 is a click on the main window in the ≤133 ms between `show()` and Task-1
parking, with the splash still overlaid — noted once, not per site.

| # | site | enclosing | wired from | note |
|---|---|---|---|---|
| 10–15 | `:1336 :1353 :1366 :1396 :1434 :1456` (six mutually exclusive `generate_*` branches) | `_on_text_generate_requested` | `main_window.text_generate_requested` (`:730`) — Generate/Quick-Speak clicks | also re-entered from `_fire_pending_generation` (§2.4 #31) |
| 16–18 | `:1527 :1537 :1560` `ensure_future(cancel_generation / stop_all_playback / stop_streaming_session)` | `_on_cancel_generation_requested` | `cancel_generation_requested` (`:746`) — Stop click | audio dispatch chain; order-sensitive ("queued before fires before"); untouched |
| 19 | `:1701` `_play_generated_audio` | `_on_replay_last_requested` | `replay_last_requested` (`:740`) | dispatch chain |
| 20 | `:1796` `_play_generated_audio` | `_on_clear_comms_requested` | `clear_comms_requested` (`:742`) | dispatch chain |
| 21 | `:1908` `ensure_future(stop_all_playback)` | `_interrupt_active_playback_for_clear_comms` ← #20 | same | dispatch chain |
| 22 | `:2026` `_play_generated_audio` | `_on_clear_comms_test_playback_requested` | `clear_comms_test_playback_requested` (`:744`) — settings panel button | dispatch chain |
| 23–24 | `:2065 :2073` `set_active_profile` / `update_voice_selection` | `_on_voice_changed` | `voice_changed` (`:731`) — voice selector / settings | #23's `on_success` is §2.4 #27 |
| 25 | `:2629` `_initialize_whisper_service_on_demand` | `_on_whisper_init_requested` | `whisper_init_requested` (`:747`) — VDS open | also invoked by the TTS service's `_whisper_init_callback` from inside `_ensure_transcription_for_clone_voice` (a generation/prepare task) → that path is (b0): the parent raises `RuntimeError` and continues generating; no pump |
| 26 | `:2882` `_initialize_whisper_service_on_demand` | `_on_transcription_requested` | `main_window` transcription request (`:5314`), VDS emotions panel (`:910`), and `_ensure_voice_clone_pipeline` (§2.4) | |
| 32 | `:4138` `_play_generated_audio` (re-entry) | `_dispatch_next_pending` | `QMetaObject.invokeMethod(QueuedConnection)` from `_on_playback_complete` (worker-thread origin) and `_on_cancel_generation_requested:1633` | delivered by the Qt loop with no task current; dispatch chain |
| 33–37 | `:4409 :4418 :4441 :4448 :4455` (quality tier, save, device settings, mic mixing, API reconcile) | `_on_settings_changed` | `settings_changed` (`:733`, settings OK / tray notice) | five FIFO neighbours; untouched |
| 38 | `:4471` `enumerate_all_devices` | `_on_device_refresh_requested` | `audio_device_refresh_requested` (`:734`, also via `QTimer.singleShot(1000, _populate_device_list)` — a Qt timer in the interactive phase) | |
| 39 | `:4516` `enumerate_mic_devices` | `_on_mic_device_refresh_requested` | `mic_device_refresh_requested` (`:748`) | |
| 40–41 | `:4582 :4585` `start_monitor()` / `stop_mic_monitor_to_speakers` | `_on_mic_monitor_toggled` | `mic_monitor_toggled` (`:749`, checkbox) | |
| 42 | `:4617` `create_task(_test_device_async)` | `_on_device_test_requested` | `audio_device_test_requested` (`:735`) | the `else: run_until_complete` branch is unreachable under qasync (loop always running) |
| 43 | `:4725` `create_task(_test_virtual_device_async)` | `_on_virtual_device_test_requested` | `virtual_device_test_requested` (`:736`) | same |
| 46 | `:5126` `_update_voice_manager_directory` | `_on_voice_directory_changed` | `voice_directory_changed` (`:737`) | |
| 47 | `:5195` `force_rescan` | `_on_voice_refresh_requested` | `voice_refresh_requested` (`:738`) | |

### 2.4 `app.py` — scheduled from inside a running task's final step: **(b0)**

Each of these runs inside the `_handle_task` wrapper of a *previous* task,
in the `on_success`/`on_error` callback after `await coro` returned. Stack:
`Qt loop → timerEvent → _handle_task.__step → on_success → <site>`. The
parent's remaining work after the site is: return from the callback →
`_handle_task` returns → task completes → `_leave_task`. No Qt pump exists
anywhere in that stretch (`app.py` has no pump call outside
`_show_error_dialog`; the `main_window` methods these callbacks reach —
`set_generation_status`, `show_service_notification`, `set_emotion_enabled`,
`update_voice_emotions`, `set_playback_active` — are `QStatusBar`/widget
updates, read). The new task's first step is delivered on the next Qt pass
with no task current. Exposure would require Task-1 to pump concurrently,
and Task-1 is parked for the whole interactive phase.

| # | site | enclosing | parent task (whose `on_success` this is) | neighbours / order |
|---|---|---|---|---|
| 27 | `:2133` `preload_model(required_model)` | `_on_voice_profile_set` | #23 `set_active_profile` | then `_ensure_voice_clone_pipeline` → #28 or #26; FIFO with #28 |
| 28 | `:2233` `prepare_voice_clone_prompt` | `_trigger_voice_clone_prompt_prepare` | #27's callback, or #30's rescan callback (`_after_transcription_rescan`) | |
| 29 | `:2798` `_transcribe_voice_file` | `_proceed_with_transcription` | #26's `on_success` (`_continue_transcription_after_init`), or called synchronously from #26's handler when Whisper is already up (then (a)) | |
| 30 | `:2962` `force_rescan` | `_on_transcription_complete` | #29 | `on_success` → `_after_transcription_rescan` → #28 |
| 31 | `:1336–:1456` via `_fire_pending_generation` | `_on_text_generate_requested` | #28's callback (`_on_voice_clone_prompt_prepared`) / prepare-error / rescan-failed | the deferred-Generate replay; same six branches as #10–15 |
| 44 | `:3407` `_play_generated_audio(audio_bytes)` | `_on_tts_generation_complete` | #10–15 (the generation task) | dispatch chain; untouched |
| 45 | `:3540` `ensure_future(_clear_progressive_flag_after_drain())` | `_play_generated_audio` (an `async def`, i.e. inside #19/#20/#22/#32/#44's task) | | dispatch chain; untouched |

### 2.5 `app.py` — called on the device-monitor thread: **(t)**

`WindowsAudioClient._device_monitor_loop` (a `threading.Thread`,
`windows_audio_client.py:1010`) → `_notify_device_change` →
`DeviceResilienceManager._handle_device_change_event` → its
`_recovery_callbacks` / `_device_change_callbacks` / `_emit_notification` →
`AudioCoordinator._handle_device_recovery` / `_handle_device_change` /
`_handle_device_notification` → `MyVoiceApp._on_device_notification`. Every
hop is a synchronous call on the monitor thread; nothing marshals to the loop
thread (`call_soon_threadsafe` does not appear in any of these files;
`_check_for_device_changes` has no other caller).

| # | site | what actually happens |
|---|---|---|
| 48 | `app.py:4972` `_run_async_task(_auto_refresh_device_lists())` in `_on_device_notification` | `asyncio.ensure_future` from a thread with no event loop set → `RuntimeError: There is no current event loop in thread 'Thread-N'`, caught by the handler's `except`, logged; the `_handle_task` coroutine is never awaited |
| 49 | `app.py:5060` `_run_async_task(save_settings())` in `_migrate_disconnected_device_settings` ← `_handle_device_disconnection` ← #48 | same |
| 60 | `audio_coordinator.py:2025` `asyncio.create_task(_restart_mic_capture(device))` in `_handle_device_recovery` | `create_task` → `get_running_loop()` → `RuntimeError: no running event loop`, caught and logged |
| 61 | `audio_coordinator.py:2084` `asyncio.create_task(_stop_mic_on_disconnect())` in `_handle_device_change` | same |

No task is ever created, so the qasync hazard is moot; the paths are simply
broken (§6.1). Not fixed here: making them work would introduce *new* live
behaviour (device hot-plug auto-refresh / mic restart) that has never run in
production, which is the opposite of this story's constraint.

### 2.6 Services

| # | site | enclosing | stack | class |
|---|---|---|---|---|
| 50 | `routes.py:136` `ensure_future(generate_custom_voice(streaming=True))` | `_streaming_response` | a uvicorn request-handler task (created by the h11 protocol from a transport callback, no task on the stack) | (b0) — inside a task; no pump; Task-1 parked |
| 51 | `routes.py:163` `ensure_future(queue.get())` | `gen()` inside the streaming response | same | (b0) |
| 9 | `server.py:106` | see §2.1 | | **(b) → MOVED** via caller |
| 52–53 | `audio_coordinator.py:684 :690` `create_task(monitor/virtual play)` | `play_dual_stream` | inside `_play_generated_audio`'s task (#19/#20/#22/#32/#44) | (b0); dispatch chain; untouched |
| 54 | `audio_coordinator.py:2673` `_mic_monitor_task = create_task(_mic_monitor_loop())` | `start_mic_monitor_to_speakers` ← only `app.py:4699` inside #40's task | (b0); long-lived but only during the interactive phase; stopped by `cleanup_async` before W2's pump |
| 55 | `audio_service.py:3730` `create_task(self.shutdown())` | `AudioManager.__exit__` | — | **(x)**: no `with AudioManager` anywhere in `src/` |
| 56–59 | `background_transcription_manager.py:400 :425 :441 :476` `create_task(_emit_notification / cache_transcription)` | `_on_transcription_progress` / `_on_transcription_completion` / `_update_batch_progress` | callbacks invoked synchronously from `TranscriptionQueueService._process_transcription_item` (a task) | (b0) shape, and **(x)**: `BackgroundTranscriptionManager` is never instantiated in `src/` |
| 62 | `transcription_queue_service.py:131` `create_task(_process_queue())` | `start_service` | only via `BackgroundTranscriptionManager.start` | **(x)** |
| 63 | `transcription_queue_service.py:291` `create_task(_process_transcription_item)` | `_process_queue` task | | **(x)** |
| 64 | `transcription_queue_service.py:417` `create_task(_retry_transcription)` | `_process_transcription_item` task | | **(x)** |
| 65 | `voice_profile_service.py:1366` `_cache_save_task = create_task(_save_profile_cache())` | `set_active_profile` | inside #23's task (interactive, (b0)) **or** inside the restore task at startup (#6) | at startup: created ~t₀+1.0 s (legacy) / ~t₀+0.02 s (now), body is one `run_in_executor` write (ms), woken via `call_soon_threadsafe` → completes inside the preload suspension; same residual as #1 on a synchronously-failing launch |

### 2.7 Voice Design Studio (`voice_design_studio_dialog.py`)

The dialog runs inside `voice_design_studio_dialog.exec()` from
`_on_voice_design_studio_clicked` (a click slot) or from the settings
dialog's `_open_voice_design_studio` (a click slot inside the settings
dialog's own `exec()`): nested Qt loops with **no task on the stack**, so
tasks scheduled here step normally. Its `QMessageBox` pumps (`:471 :480
:1352 :1871 :1900 :1951–:1972 :2064 :2074 :2170`) are all click slots or
`closeEvent`; none is reachable from an `on_success`/`on_error` callback
(`_save_from_description` ← `_on_save_clicked` only).

| # | site | enclosing | stack | class |
|---|---|---|---|---|
| 66 | `:669` `ensure_future(_handle_task())` | VDS `_run_async_task` | inherits | — |
| 67 | `:523` variant generation | `_generate_next_variant` | first call from `_on_generate_requested` (click) → (a); subsequent from `_on_variant_complete`/`_on_variant_failed` (the previous variant's `on_success`/`on_error`) → (b0) self-chain | (a)/(b0) |
| 68 | `:792` emotion variant | `_generate_next_emotion_variant` | same shape (`:776` click path; `:878 :893` self-chain) | (a)/(b0) |
| 69 | `:1079` emotion embedding extraction | `_extract_next_emotion_embedding` | `:1042` click path; `:1224 :1236` self-chain | (a)/(b0) |
| 70 | `:1263` refinement preview | `_generate_refinement_preview` | `_on_preview_requested` (click) → (a); `_finish_batch_extraction` (end of #69's chain) → (b0) | (a)/(b0) |
| 71 | `:1626` clone transcribe | `_on_clone_transcribe_requested` | `description_path_panel.clone_transcribe_requested` (`:1414`, click) | (a) |
| 72 | `:1751` emotion transcribe | `_on_emotion_transcribe_requested` | `emotions_panel.transcription_requested` (`:550 :921`, click) | (a) |

(Numbering runs to 72 because #31 is the same six lines as #10–15 reached
by a second path; the distinct-line count is 71.)

### 2.8 Tally (71 distinct lines)

| class | lines | sites |
|---|---:|---|
| (a) safe by construction | 33 | #10–24, #25, #26, #32–43, #46–47, #71–72 |
| (a)/(b0) both legs (VDS self-chains) | 4 | #67–70 |
| (b0) inside a task's final step, no pump reachable | 11 | #27–30, #44–45, #50–54 |
| (b0) interactive leg + (b) startup leg, argued | 1 | #65 |
| (b) exposed, **moved** | 2 | #6, #9 |
| (b) exposed in theory, **left with argument** | 2 | #1, #7 |
| the mechanism / fixed by 20.3 | 4 | #2, #3, #4, #66 |
| (t) thread, never creates a task | 4 | #48, #49, #60, #61 |
| (x) unreachable / dead | 10 | #5, #8, #55, #56–59, #62–64 |
| (c) uncertain | **0** | — |
| **total** | **71** | (#31 is #10–15 reached by a second path, not a distinct line) |

---

## 3. AC #2 — changes and their behaviour-preservation proofs

`src/` diff: `app.py` only, 162 insertions / 31 deletions, three regions.

### 3.1 The helper: `_run_async_task_when_loop_is_idle` → thin wrapper over `_schedule_when_loop_is_idle`

The deferral loop (`call_soon` re-arm while `current_task() is not None`,
bounded by `_MAX_IDLE_DEFERRALS`, WARNING + schedule-anyway on exhaustion,
INFO hand-off line) moved verbatim into
`_schedule_when_loop_is_idle(coro_factory, *, label, on_success, on_error,
failure_note)`. The warmup wrapper keeps its signature and calls it with
`lambda: self._compile_warmup_entrypoint(coro_factory)`,
`label="torch.compile warmup"` and the original `on_success` debug line.

Proof it is identical:

* the three log lines are byte-identical for the warmup label —
  `test_warmup_wrapper_log_lines_are_byte_identical_to_story_20_3` asserts
  the exact DEBUG / WARNING / INFO strings the AC #4 hardware evidence
  quotes, including `deferred N loop pass(es)`;
* the wrapper still routes through `_compile_warmup_entrypoint` —
  `test_warmup_call_site_and_wrapper_are_untouched` (AST). **Story 20.3's
  own rows did not catch dropping the entrypoint from the wrapper** (mutation
  M5, §4.5: 11 passed) — they test the entrypoint and the call site, not the
  wrapper's routing;
* Story 20.3's 14 rows (`test_app_compile_warmup_sequencing.py`,
  `test_app_compile_warmup_qasync.py`) pass unchanged; the qasync file's
  fixed row still records one `primed_warm` with zero errors;
* the call site at `app.py:671` is byte-identical (`git diff` shows no hunk
  there).

### 3.2 Site #6 — voice restoration (`_on_voice_service_started`)

| | before | after |
|---|---|---|
| shape | `QTimer.singleShot(500, schedule_restore)`; `schedule_restore` does `loop.create_task(delayed_restore())`; `delayed_restore` begins `await asyncio.sleep(0.5)` | `self._schedule_when_loop_is_idle(delayed_restore, label="voice restoration", …)`; `delayed_restore` has no sleep |
| task created | at t₀+0.50 s, from a Qt timer slot | on the first loop pass with no task mid-step — t₀+~0.015 s on a normal launch (Task-1's first yield inside the preload), or after Task-1 parks on a fast-fail launch |
| runs | once | once (`startup_order_fixed`: one `restore_start`, one `restore_done`) |
| order vs neighbours | hydration (t₀+0.014) → restore (t₀+1.0) → preload returns (t₀+3.4) → UI → warmup | hydration → restore → preload returns → UI → warmup — **same**, proved under a real qasync loop: `events == ["hydration@preload", "restore_start@preload", "restore_done@preload", "warmup@parked"]` |
| why the order holds | — | the idle callback is armed (`call_soon`) *before* hydration's `ensure_future` (line 573 precedes 589), so on the first idle pass the callback runs first and creates the restore task, whose first step is queued *behind* hydration's already-queued step; Qt delivers `QZeroTimerEvent`s FIFO. `test_generic_helper_creates_its_task_one_pass_after_a_plain_neighbour` pins the one-pass offset so nobody assumes FIFO parity the other way round. |
| dependencies checked | `get_active_profile_model_type()` for the preload is read at t₀+0 from the voice manager's own cache, before the restore in both shapes; `_on_voice_restoration_complete` is a log + `pass`; the restore does not touch the TTS service; `_main_window` is `None` at both times | |
| absolute timing | restore at t₀+1.0 s | restore at t₀+~0.02 s. Nothing reads the active profile between those two instants except the model preload's *cache* read (above). The warmup, which does read it (`_active_profile_voice_clone_prompt`), runs ≥3 s after either. |
| log lines | "Voice restoration scheduled after initialization delay" (INFO, at scheduling) | "voice restoration handed off to the event loop (deferred N loop pass(es) …)" (INFO, at creation) + "Voice restoration task completed" (DEBUG). No test or script greps the old line. |
| failure mode | fast-fail launch: destroyed silently except for the loop ERROR + "destroyed pending" (§4.2 control) | fast-fail launch: deferred through the pump, runs after park (§4.2 fixed). Valve limit: §6.4. |

### 3.3 Site #9 — local TTS API start (`_setup_api_server_from_settings`)

| | before | after |
|---|---|---|
| shape | `await self._api_server.start(host, port)` inline in Task-1, wrapped in try/except → `logger.exception("Failed to start local TTS API server")` | same `await`/try/except moved into a nested `_start_api_server_if_still_enabled()` handed to `_schedule_when_loop_is_idle(label="local TTS API start")`, with a run-time re-read of `enable_http_api` |
| controller / hub construction | synchronous, always | unchanged — still synchronous, so `_reconcile_api_server` and the Settings panel see a controller immediately |
| task created | serve task created inside Task-1 ~2 ms before W1 | created inside a `_handle_task` on the first idle pass after Task-1 parks |
| runs | once | once (`api_fixed`: `running_before_pump=False`, `running_after_pump=True`, `deferrals=[35]`) |
| order vs neighbours | mic setup → **API up** → `initialize_async` returns → W1 → park → warmup created | mic setup → controller built → `initialize_async` returns → W1 → park → **warmup created, then API start task created** (both idle callbacks, FIFO: the warmup's was armed at `:671`, this one at `:331`'s tail) |
| **ordering change, reported** | | "API listening" now happens ~150 ms later and *after* the warmup task is created rather than before. Nothing depends on it: the warmup holds the TTS request semaphore during priming so an early API request queues exactly as it did when the API was up before priming; `main.py` does not check the server; the Settings panel reads `is_running` only when opened. If the reviewer disagrees, reverting this one hunk restores the inline start and leaves the rest of the story intact. |
| toggle race | none (inline) | a Settings change inside the ~150 ms gap: disable → the deferred start re-reads `self._app_settings.enable_http_api` (swapped by `_on_settings_changed:4525`) and skips with an INFO line; enable is handled by `_reconcile_api_server` itself. `test_deferred_api_start_reads_the_toggle_at_run_time` pins both. |
| close-within-150-ms race | none | `cleanup_async` runs `_api_server.stop()` (no-op, not running) and the deferred start may then fire during cleanup's awaits, starting a server that `_os._exit(0)` kills a few hundred ms later. Harmless; noted. |
| failure mode | W1 tick lands in pump → zombie serve task, `is_running` True, no ticks, `stop()` hangs to main.py's 8 s hard exit (§4.4 control, reproduced against real uvicorn) | survives; `stop()` completes (§4.4 fixed) |

### 3.4 Sites left in place — the written arguments

* **#1 hydration (`:589`)** — body is a single synchronous step (20.3 §1.1a),
  delivered at Task-1's first yield 14 ms later, ≥3.4 s before W1 on every
  logged launch. Its future is retained for `_compile_warmup_entrypoint`;
  the idle helper cannot return one. Residual on a launch whose preload
  raises synchronously: hydration destroyed → BASE prime skips
  (`no_priming_prompt`), 20.3's designed fallback. Left.
* **#7 device monitoring (`:4217`)** — 1 ms body at Task-1's next yield,
  ≥3 s before W1. Moving it would delay device-change monitoring by ~3 s on
  every launch for a benefit only on a synchronously-failing launch, where
  the monitoring's own callbacks are thread-broken anyway (§6.1). Left.
* **#65 profile-cache save (startup leg)** — one executor write, woken by
  `call_soon_threadsafe`, complete within ms of the restore, ≥2.4 s before W1
  (now ≥3.3 s). Left.
* **All (b0) sites** — the spec's literal "called from inside a running task"
  applies, but exposure additionally requires a pump the parent can run
  before the child's first step, and there is none; moving them would add a
  loop pass of latency to every generation/transcription hand-off and change
  FIFO order among the five `_on_settings_changed` neighbours, for no live
  hazard. Left. If the reviewer reads AC #1(b) strictly, these are the rows
  to argue about; §2.4 states each stack.

---

## 4. AC #3 — the hazard under a real qasync loop

`tests/unit/_qasync_call_site_driver.py` (out-of-process, same shape as
20.3's `_qasync_warmup_driver.py`) + `tests/unit/test_app_qasync_call_sites.py`
(**12 rows**: 5 subprocess, 3 AST, 4 plain-loop). File time ≈35 s (five
interpreters, each paying the torch import).

Reading "non-startup exposed site" in AC #3 as "a site other than the one
Story 20.3 already covers": both exercised sites are startup-adjacent
because that is where every real exposure lives (§1.1); there is no exposed
site outside startup to exercise.

### 4.1 Rig

Task-1 drives the *real* `MyVoiceApp` method under test on a `__new__`-built
app object with `SimpleNamespace` collaborators, then runs the W1 pump
**inside Task-1**, then parks. Two pump shapes:

* default — `processEvents()` + `time.sleep(0.01)` per pass: Qt's
  `waitForWindowExposed` loop that `QSplashScreen.finish` runs (≤1 s, ~100
  passes/s), the longest pump main.py can run;
* `tight` — back-to-back `processEvents()` (~10⁶ passes/s). Nothing in
  main.py does this; kept to document the valve limit (§6.4).

### 4.2 Voice restoration, fast-fail launch (preload returns without suspending)

| row | shape | pump | result |
|---|---|---|---|
| `test_legacy_restore_hand_off_is_destroyed_under_a_real_qasync_loop` (control) | pre-20.9 `QTimer.singleShot(500)` + `create_task` + `sleep(0.5)` | 1.3 s, 127 passes | `events == []`; `Cannot enter into task <delayed_restore()> while <task1_restore()>`; `Task was destroyed but it is pending!` |
| `test_shipped_restore_hand_off_survives_the_same_pump` | real `_on_voice_service_started(None)` | same | `deferrals == [127]`; `events == ["restore_start@parked", "restore_done@parked"]`; zero errors |
| (driver only) `restore_fixed_tightspin` | real method | 1.3 s, 1,002,994 passes | `deferrals == [10000]` → valve fires → scheduled into the pump → destroyed. §6.4. |

### 4.3 Startup order, normal launch (preload suspends 50 ms)

`test_startup_sites_run_once_and_in_production_order`: real
`_on_voice_service_started` → real `_run_async_task(hydrate())` → `await
preload` → real `_run_async_task_when_loop_is_idle(warmup)` → 40 × pump →
park.

```
events    = ["hydration@preload", "restore_start@preload", "restore_done@preload", "warmup@parked"]
deferrals = [0, 40]        # restore: created on the first idle pass; warmup: deferred through the pump
errors    = []
```

### 4.4 Local TTS API, against real uvicorn 0.40.0 on a free `127.0.0.1` port

`uvicorn.server.Server.on_tick` is wrapped in the driver process to count
main-loop ticks. Pump 0.35 s (35 passes), then wait for `server.started`,
count ticks over 0.5 s, then `stop()` with a 4 s guard.

| row | shape | result |
|---|---|---|
| `test_legacy_inline_api_start_leaves_a_zombie_serve_task` (control) | pre-20.9: real `ApiServerController.start()` awaited inline in Task-1 | `running_before_pump=True`; `Cannot enter into task <Server.serve() running at uvicorn/server.py:71> wait_for=<Future finished result=None>`; `running_after_pump=True` (**`is_running` lies**); `ticks_while_parked=0` (`ticks_total=1`); `stop_completed=False`; "destroyed pending" once the ref was dropped |
| `test_shipped_deferred_api_start_survives_and_stops_cleanly` | real `_setup_api_server_from_settings()` with `enable_http_api=True` | `running_before_pump=False`; `deferrals=[35]`; `running_after_pump=True`; `ticks_while_parked=4`; `stop_completed=True`; zero errors |

### 4.5 Mutation pass — 6 of 6 caught (`scratchpad/mutate.py`, app.py restored byte-identical after)

| # | mutation | caught by |
|---|---|---|
| M1 | restore reverted to `QTimer.singleShot` + `create_task` | AST pin + `restore_fixed` + `startup_order` (3 failed) |
| M2 | API start awaited inline again | AST pin + `api_fixed` + toggle row (3 failed) |
| M3 | idle check removed from the generic helper (`current = None`) | `restore_fixed` + `api_fixed` + **20.3's** `test_warmup_reaches_the_metric_under_a_real_qasync_loop` (3 failed) |
| M4 | warmup label text changed | log-line row (1 failed) |
| M5 | wrapper drops `_compile_warmup_entrypoint` | `test_warmup_call_site_and_wrapper_are_untouched` (1 failed) — **20.3's sequencing file: 11 passed, missed** |
| M6 | toggle re-check removed | toggle row (1 failed) |

---

## 5. AC #4 — regression sweep

Single `python310\python.exe -m pytest tests/ -q -p no:cacheprovider`:

**2,990 passed, 1 xfailed, 0 failed, 0 errors in 112 s** (exit 0).

Failure-set identity: empty before (tooling-5 baseline 2,978 passed / 1
xfailed / 0 failed / 0 errors), empty after. Delta = +12 rows from
`test_app_qasync_call_sites.py`. `src/` changes: `app.py` only, three hunks,
all scheduling sites (§3). Suite runtime grows by ≈35 s for the five
subprocesses.

---

## 6. Findings, disagreements, follow-ups

### 6.1 Four scheduling sites run on the device-monitor thread and never create a task

#48, #49, #60, #61 (§2.5). `_on_device_notification` and the coordinator's
recovery/change handlers are invoked synchronously from
`WindowsAudioClient._device_monitor_loop`'s thread; `asyncio.create_task` /
`ensure_future` there raise `RuntimeError` (no running / current loop),
which the surrounding `except` blocks log as "Error in device notification
callback" / "Error handling device recovery". Consequences that have been
silently absent since Story 2.5: no automatic device-list refresh in an open
Settings dialog on hot-plug, no mic-capture restart on reconnect, no
mic-capture stop on disconnect, no migrated-settings save. Fix class:
`loop.call_soon_threadsafe(...)` at the thread boundary (the coordinator
already knows the pattern — `_on_playback_complete` uses
`QMetaObject.invokeMethod(QueuedConnection)`). **Not done here**: it is a
different defect and it would add live behaviour. Separately,
`AudioCoordinator.add_device_notification_callback` is defined twice
(`:1127`, `:2098`); the second shadows the first, so the monitor/virtual
services never receive the app's callback — which is why the app callback
only ever arrives via the resilience manager.

### 6.2 Dead and unreachable sites

`_on_config_service_started` (#5) has no caller; `AudioManager.__exit__`
(#55) has no `with` user; `BackgroundTranscriptionManager` and therefore
`TranscriptionQueueService` (#56–59, #62–64) are never instantiated in
`src/`. Ten of the 71 sites cannot run in the shipped app. Left as-is (not
scheduling-site edits in any behavioural sense; a cleanup story).

### 6.3 `cleanup_async` re-enters `close()` inside Task-1 (W2)

`app.py:927` calls `self._main_window.close()` after `lastWindowClosed` has
already fired, from inside Task-1. Qt sends a second `QCloseEvent`; with
`_force_quit` False (the X-button path) `closeEvent` shows the "Close
MyVoice" `QMessageBox.question` a second time, now *inside a task*. Any task
still pending and woken during that modal is destroyed — but by that line
the API server, TTS, Whisper and the coordinator are stopped, and
`_os._exit` follows, so the only effect is the extra prompt (if it is in fact
shown — not verified on hardware here; the tray-Exit path sets `_force_quit`
and would not show it). Reported, not changed: `cleanup_async` is not a
scheduling site.

### 6.4 The safety valve has a measurable limit (Story 20.3 design, untouched)

`_MAX_IDLE_DEFERRALS = 10000` counts *passes*, and under qasync every
`processEvents()` pass delivers the re-armed zero-timer once. A tight
`processEvents` spin (no sleep) exhausts it in ~15 ms, after which the valve
schedules into the pump and the task is destroyed — measured:
`restore_fixed_tightspin`, 1,002,994 passes / 1.3 s, `deferrals=[10000]`,
destroyed. Nothing in main.py spins like that (`splash.finish` sleeps 10 ms
per pass: ≤100 passes/s, so a full 1 s spin is 100 deferrals; production
shows 2), so the valve is not reached on any real launch shape. A time-based
bound (e.g. "defer for up to N seconds") would be the principled form; not
changed here because it is Story 20.3's verified mechanism and the Dev Notes
fence it. The WARNING it emits is the observable, and it is pinned indirectly
by `test_hand_off_gives_up_deferring_rather_than_never_scheduling` (20.3).

### 6.5 Where I read the spec differently

* **AC #1(b) literal vs. mechanism.** The spec's (b) wording ("called from
  inside a running task") matches 11 sites outright, plus the chain legs of
  #65 and #67–70, that are not exposed because no pump can follow them
  (§2.4, §2.6, §2.7). I split them out as (b0) and left them,
  because moving them changes hand-off latency and FIFO order among
  neighbours for no live hazard — the exact regression class the story
  forbids. Each stack is stated so the reviewer can overrule per site.
* **AC #3 "non-startup exposed site".** There is none; every real exposure
  is a task that overlaps W1. I exercised the two startup-adjacent sites
  other than the warmup, against the real methods and (for the API) real
  uvicorn, with controls.
* **The API move is the one judgement call.** It survives today by an ~80 ms
  accident and fails silently when it does not; I moved it. The cost is the
  ordering note in §3.3. Reverting that single hunk is safe if the reviewer
  prefers the accident to the reordering.

### 6.6 Not done, deliberately

No change to `main.py` (the root cause is that Task-1 pumps Qt at all;
`await asyncio.sleep(0)` in place of `processEvents()` and a non-spinning
splash close would remove W1 entirely — but `main.py` is not a scheduling
site, and 20.3's fix was verified with W1 present). No change to
`warmup_compile_async`, `app.py:671`, or any dispatch-chain site.

---

## 7. File list

**Source**
* `src/myvoice/app.py`
  * `_run_async_task_when_loop_is_idle` — now a wrapper (same signature,
    same log lines, same entrypoint); `_schedule_when_loop_is_idle` (new,
    generic) holds the deferral loop
  * `_on_voice_service_started` — restore hand-off via the idle scheduler;
    `QTimer.singleShot` + `create_task` + `sleep(0.5)` removed
  * `_setup_api_server_from_settings` — `start()` deferred via the idle
    scheduler with a run-time toggle re-check

**Tests**
* `tests/unit/_qasync_call_site_driver.py` (new; six variants, one
  driver-only)
* `tests/unit/test_app_qasync_call_sites.py` (new, 12 rows)

**Evidence**
* this file
