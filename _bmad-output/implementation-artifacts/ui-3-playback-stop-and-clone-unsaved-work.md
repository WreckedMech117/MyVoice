# Story ui-3: Two Shipped Defects — Stop-All Playback Helpers Missing, Clone Load Not Marking Unsaved Work

Status: ready-for-dev - 2026-09-14

<!-- Source: (A) RTX 3060 log 2026-09-14 19:07:03 + 19:11:30 (build 56, code unchanged since); (B) tooling-5's strict xfail. -->
<!-- Risk: LOW–MEDIUM. (A) touches the Stop button path in batch/sentence mode; (B) one signal connection. -->

## Defect A — `AudioCoordinator.stop_all_playback` fans out to methods that do not exist

```
19:07:03,535 audio_coordinator - ERROR - Error stopping monitor playback: 'MonitorAudioService' object has no attribute 'stop_all_playback'
19:07:03,535 audio_coordinator - ERROR - Error stopping virtual playback: 'VirtualMicrophoneService' object has no attribute 'stop_all_virtual_microphone_playback'
```

`audio_coordinator.py:912/919` (Story 11.4 follow-up, the dual-mode
Clear/Stop button) calls `monitor_service.stop_all_playback()` and
`virtual_service.stop_all_virtual_microphone_playback()`. Neither service has
ever had those methods — only the per-task `stop_monitor_playback(task_id)` /
`stop_virtual_playback(task_id)`. Both calls are inside `try/except` so the
user sees a Stop button that logs two ERRORs and stops nothing on the
task-based (batch / sentence-stream) playback path. TRUE_STREAM playback
stops through the session path and is unaffected, which is why it went
unnoticed on the dev box.

Also in the same log, at shutdown:
```
19:11:30,116 monitor_audio_service - ERROR - Error stopping monitor playback monitor_1_1789434423: STOPPED
```
(eight of them — one per task that had already completed). `shutdown()`
iterates `_active_tasks` and calls `stop_monitor_playback`, which raises on
an already-finished task (the message is the status name). Find what raises
and make stopping a finished task a no-op that still cleans up the entry.

### Acceptance Criteria (A)

**AC A1** — `MonitorAudioService.stop_all_playback()` and
`VirtualMicrophoneService.stop_all_virtual_microphone_playback()` exist,
stop every entry in `_active_tasks` through the existing per-task stop, and
return the count stopped. The coordinator's fan-out is unchanged.

**AC A2** — Stopping an already-completed/stopped task neither raises nor
logs ERROR; the entry is removed. `shutdown()` on a service with finished
tasks logs no ERROR.

**AC A3** — Tests mirror the exact bug class: a coordinator test that
previously would have logged the AttributeError now asserts the count and
that both services' per-task stops were invoked; a service test with a
completed task in `_active_tasks` asserts shutdown is clean.

## Defect B — Clone-tab file load does not mark the studio as having unsaved work

`tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py::TestSampleTabSignals::test_clone_file_loaded_sets_unsaved_work`
is a strict `xfail` from tooling-5: QA8 moved the clone flow into
`DescriptionPathPanel` (which emits `clone_file_loaded(str)` at
`description_path_panel.py:1312`) but the dialog never re-connected that
signal to `set_has_unsaved_work(True)`. Loading a sample and then pressing
New Voice discards it without the confirm prompt.

### Acceptance Criteria (B)

**AC B1** — The dialog connects `clone_file_loaded` → unsaved-work = True,
in the same place its sibling signals are wired.

**AC B2** — The `xfail` marker is removed and the test passes as a plain
test; `--runxfail` is no longer needed. Full-suite summary reads `0 xfailed`.

## Dev Notes

Run pytest with `I:\MyVoiceV2\python310\python.exe -m pytest` (the worktree
has no interpreter of its own; never modify or copy `python310`). Never
weaken a production `QMessageBox` (memory: main_window_close_confirm_dialog_in_tests).
