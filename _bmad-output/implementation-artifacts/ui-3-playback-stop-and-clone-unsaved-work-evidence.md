# Story ui-3 — Evidence: Stop-All Playback Helpers Missing, Clone Load Not Marking Unsaved Work

Date: 2026-09-14. Branch `ui-3-playback-stop-clone-unsaved` off `main`
(`32ddc50`, PR #15). Interpreter: bundled `python310\python.exe` (3.10.11),
pytest 9.0.2, PyQt6 6.9.1, pytest-timeout 60 s / thread. tooling-4's
raise-on-modal net active throughout. Main advanced by one PR (#16, Story
20.12) while this branch was open; it touches only
`streaming_chunk_buffer.py` and its test — no overlap.

## 0. Result in one paragraph

`pytest tests -q -p no:cacheprovider` → **3006 passed in 116.72 s**, no
xfailed (baseline on main: 2,992 passed, 1 xfailed; +13 new tests, +1
un-xfailed). Defect A had *two* bugs under one log, not one: the coordinator
called two service helpers that never existed (the 19:07:03 lines), and
underneath that the per-task stop it should have reached was itself broken on
every call because `PlaybackStatus.STOPPED` was never defined (the 19:11:30
lines). Defect B was the one-line signal connection tooling-5 predicted.
`src/` changes: 5 files, +130/−20; no production `QMessageBox` touched.

## 1. Root cause of the `STOPPED` raise (Defect A, second half)

Log shape (`20-10-rtx3060-logs/myvoice.log:949–956`, eight lines, one per
completed task):

```
19:11:30,116 monitor_audio_service - ERROR - Error stopping monitor playback monitor_1_1789434423: STOPPED
```

The story's hypothesis was a status setter / state machine rejecting the
transition on a finished task. It is simpler and worse than that:

* `src/myvoice/models/audio_playback_task.py::PlaybackStatus` had four
  members — `PENDING, PLAYING, COMPLETED, FAILED`. **No `STOPPED`.**
* `MonitorAudioService.stop_monitor_playback` (and the mirror
  `VirtualMicrophoneService.stop_virtual_playback`) did
  `task.status = PlaybackStatus.STOPPED` as their first action after the
  lookup. `EnumMeta.__getattr__` raises `AttributeError(name)` — the
  exception's `str()` is the bare member name, which is exactly the
  `: STOPPED` suffix in the log.
* The `except Exception` at the bottom of each stop method turned that into
  the ERROR line and returned `False` **before** reaching the `pop()` — so
  the entry was never removed either.
* Both playback workers already polled `task.status.value in ['failed',
  'stopped']` between chunk writes, i.e. the code was written expecting the
  member and it was simply never added.

So the per-task stop raised on *every* call, not only on finished tasks. It
surfaced at shutdown as eight lines because the workers never remove a task
from `_active_tasks` when playback runs to the end, so by 19:11:30 the dict
held every task of the session, all `COMPLETED`, and `shutdown()` walked them.
It did not surface during the session because the only other caller,
`AudioCoordinator.stop_all_playback`, was failing one layer up (first half of
Defect A) and never got as far as the per-task stop.

Fix: add `STOPPED = "stopped"` (the value the workers poll). Then make the
per-task stop a clean no-op for a task already in a terminal status
(`COMPLETED/FAILED/STOPPED`): drop the entry, leave the terminal status
intact, log at DEBUG, return `False` (nothing live was interrupted). Add the
two missing helpers `stop_all_playback()` /
`stop_all_virtual_microphone_playback()` that walk `_active_tasks` through
the per-task stop and return the count of live tasks interrupted; `shutdown()`
now uses the same helper. The coordinator fan-out is unchanged (AC A1).

## 2. Files changed

| File | Change |
|---|---|
| `src/myvoice/models/audio_playback_task.py` | `PlaybackStatus.STOPPED = "stopped"` |
| `src/myvoice/services/monitor_audio_service.py` | `_FINISHED_STATUSES`; finished-task no-op branch in `stop_monitor_playback`; new `stop_all_playback()`; `shutdown()` routes through it |
| `src/myvoice/services/virtual_microphone_service.py` | mirror of the above with `stop_virtual_playback` / `stop_all_virtual_microphone_playback()` |
| `src/myvoice/services/audio_coordinator.py` | docstring only — no longer claims the helpers pre-existed |
| `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py` | `clone_file_loaded.connect(self._on_clone_file_loaded)` next to the QA8 siblings; new slot calls `set_has_unsaved_work(True)` |
| `tests/unit/services/test_playback_stop_all.py` | new — 13 tests (see §3) |
| `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py` | `xfail(strict=True)` removed from `test_clone_file_loaded_sets_unsaved_work`; new `test_new_voice_after_clone_load_asks_before_discarding` |

## 3. Test list

`tests/unit/services/test_playback_stop_all.py` — uses the **real**
`MonitorAudioService` / `VirtualMicrophoneService` (constructed, never
initialized, so no PortAudio), because `test_audio_coordinator.py`'s
`MagicMock(spec=Service)` fixtures *assign* the missing helpers and so could
never have caught the AttributeError (spec guards reads, not writes).

| Test | AC |
|---|---|
| `TestPlaybackStatusStopped::test_stopped_member_matches_the_value_the_workers_poll` | A2 (root cause) |
| `TestStopFinishedTaskIsCleanNoOp::test_monitor_stop_on_finished_task_removes_entry_without_error[COMPLETED,FAILED]` | A2 |
| `TestStopFinishedTaskIsCleanNoOp::test_virtual_stop_on_finished_task_removes_entry_without_error[COMPLETED,FAILED]` | A2 |
| `TestStopFinishedTaskIsCleanNoOp::test_monitor_shutdown_with_eight_finished_tasks_logs_no_error` — the exact 19:11:30 shape | A2/A3 |
| `TestStopFinishedTaskIsCleanNoOp::test_virtual_shutdown_with_finished_tasks_logs_no_error` | A2 |
| `TestServiceStopAllHelpers::test_monitor_stop_all_playback_stops_live_tasks_and_counts` | A1 |
| `TestServiceStopAllHelpers::test_virtual_stop_all_stops_live_tasks_and_counts` | A1 |
| `TestServiceStopAllHelpers::test_stop_all_on_idle_services_is_a_zero_no_op` | A1 |
| `TestCoordinatorStopAllPlaybackRealServices::test_stop_button_path_stops_both_sinks_and_sums_count` — the exact 19:07:03 shape; asserts count == 2, both per-task stops awaited with their task id, no ERROR record | A3 |
| `TestCoordinatorStopAllPlaybackRealServices::test_stop_button_with_nothing_playing_returns_zero_without_error` | A1 |

`tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py::TestSampleTabSignals`

| Test | AC |
|---|---|
| `test_clone_file_loaded_sets_unsaved_work` — plain test now; drives `panel._load_clone_audio_file`, not the signal | B1/B2 |
| `test_new_voice_after_clone_load_asks_before_discarding` — user-facing symptom: New Voice after a load must pop "Start New Voice"; No keeps `_clone_audio_path`. `QMessageBox.question` patched to answer No (tooling-4 style), never weakened | B1 |

## 4. Mutation check

Targeted set = the 14 tests above. Each mutation applied by exact-string
replacement, run, then the pristine file restored (residue grep = 0 before
the full-suite run).

| # | Mutation | Result |
|---|---|---|
| M0 | All four `src` files reverted to `main`'s version (the shipped code) | **14 failed** — every new/un-xfailed test. Log reproduces the RTX 3060 lines verbatim: `Error stopping monitor playback monitor_1_1789434423: STOPPED` ×8 in the shutdown test |
| M1 | Keep enum fix, disable monitor finished-task no-op branch | 3 failed: `test_monitor_stop_on_finished_task…[COMPLETED]`, `…[FAILED]`, `test_monitor_stop_all_playback_stops_live_tasks_and_counts` (terminal status overwritten; finished task counted) |
| M2 | Same for virtual | 3 failed: the virtual mirrors |
| M3 | Rename `MonitorAudioService.stop_all_playback` (helper missing again) | 5 failed incl. both coordinator tests; log shows `'MonitorAudioService' object has no attribute 'stop_all_playback'` from `audio_coordinator.py:916` — the 19:07:03 line |
| M4 | Rename `stop_all_virtual_microphone_playback` | 5 failed; log shows the second 19:07:03 line from `audio_coordinator.py:923` |
| M5 | `STOPPED = "halted"` (member exists, wrong value for the worker poll) | 1 failed: `test_stopped_member_matches_the_value_the_workers_poll` |
| M6 | `stop_all_playback` counts every entry, not just live ones | 1 failed: `test_monitor_stop_all_playback_stops_live_tasks_and_counts` |
| M7 | Remove the `clone_file_loaded.connect(...)` line (Defect B as shipped) | 2 failed: both `TestSampleTabSignals` tests |
| M8 | Slot connected but calls `set_has_unsaved_work(False)` | 2 failed: both `TestSampleTabSignals` tests |

## 5. Full suite

```
I:\MyVoiceV2\python310\python.exe -m pytest tests -q -p no:cacheprovider
3006 passed in 116.72s (0:01:56)
```

0 failed, 0 errors, 0 xfailed (AC B2). Adjacent files rerun in isolation
beforehand: `test_playback_stop_all.py` + `test_audio_coordinator.py` +
`TestSampleTabSignals` → 48 passed.

## 6. Observations not acted on

* The playback workers never remove a completed task from `_active_tasks`
  (`get_status()['active_tasks']` therefore reports the session total, not
  the live count). The finished-task no-op branch makes this harmless for
  Stop and shutdown; a worker-side `pop` on completion is a separate
  cleanup, not this story's scope.
* `_on_description_content_changed` sets unsaved-work from
  `has_content()` (description / preview text only). A user who loads a
  clone sample, types in the description and then erases it will have the
  flag cleared again. Pre-existing for the emotions path too; product call.
