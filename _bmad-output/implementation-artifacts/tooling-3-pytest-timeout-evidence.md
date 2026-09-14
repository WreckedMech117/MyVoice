# Story tooling-3 — Evidence: Make Suite Hangs Countable (pytest-timeout)

Date: 2026-09-14. Branch `tooling/3-pytest-timeout` off clean merged `main`
(`2936c7f`). Interpreter: bundled `python310\python.exe` (3.10.11), pytest 9.0.2,
pytest-timeout **2.4.0** (newly installed), PyQt6 6.9.1, torch 2.10+cu128, Win11.

Working files for this story live in `tooling-3/` next to this file (scripts,
every raw log, per-test duration JSONL for every run, the generated timeout table).
All are `_bmad-output/`-gitignored; force-add the scripts if they should travel.

## 0. Result in one paragraph

`pytest tests/` on this repo now **ends** instead of stalling. Under the derived
default of `timeout = 60` (thread method) the whole suite was driven to completion
and produced **27 named timeouts** at **2 distinct stall sites** (both a modal
`QMessageBox.warning(...)` reached from test code), plus **47 failed / 4 errors /
2896 passed** in 82 s on the final pass. The pre-existing failure set is
**unchanged in identity**: the 20.8 baseline's 49 FAILED ids = these 47 FAILED ids
+ 2 ids that are now TIMEOUT rows instead of FAILED rows (section 6). Nothing was
fixed; the 27 are listed by id with stall location in section 5.

One deviation from the story's literal AC #1 wording, discovered by reading the
plugin, not assumed: with `timeout_method = thread` — the only method that works on
Windows — pytest-timeout **terminates the pytest process** (`os._exit(1)`) after
naming the test and dumping every thread's stack. So a single `pytest tests/`
"completes" in the sense that it returns, but it returns at the *first* hang; it
does not go on to run the remaining tests. Enumerating all 27 took 28 runs
(`--deselect` the last hang, run again). Details and consequences in sections 2 and 8.

## 1. What changed (tracked files)

| File | Change |
|---|---|
| `pytest.ini` | **New** (no pytest config existed in the repo). `timeout = 60`, `timeout_method = thread`, with a comment block explaining both values and the thread-method exit semantics. rootdir was already the repo root (pytest picks cwd), so node ids are unchanged — verified with `--collect-only` before and after. |
| `requirements.txt` | `pytest-timeout>=2.3  # dev-only — ...` appended directly beneath the `pytest-cov` dev-only entry (lines 84-96 convention), with a Windows note matching the shape of the pytest-cov one. |
| `python310/` (untracked, bundled interpreter) | `pip install pytest-timeout` -> 2.4.0. |

No test file was marked with `@pytest.mark.timeout(N)` — none needed it (section 3).
No change to `tests/conftest.py` (section 4).

## 2. AC #1 — hangs become failures (and what "completes" means on Windows)

- Plugin loads and is active: pytest header now prints
  `timeout: 60.0s / timeout method: thread / timeout func_only: False`
  (`tooling-3/suite-S1-iter28.log` lines 1-12, and every other log).
- Synthetic check before touching the suite: a scratch test spinning in
  `while True: time.sleep(0.01)` under `--timeout=2 -v` produced, in order: the
  test's node id on the `+++ Timeout +++` line, `~~~ Stack of MainThread ~~~` with
  the frame in the test body, a closing `+++ Timeout +++`, exit code 1, and the
  next test in the file was **not** run.
- Read in `python310/Lib/site-packages/pytest_timeout.py`:
  - `timeout_timer()` (thread method, lines 505-542): suspends capture, prints
    captured stdout/stderr, `dump_stacks(terminal)`, then `finally: os._exit(1)`.
  - `timeout_sigalrm()` (signal method) raises `Failed` inside the test and the
    session continues — but `HAVE_SIGALRM = hasattr(signal, "SIGALRM")` is False on
    Windows, so it is unavailable here. The story's requirement of `thread` is the
    only workable choice; its cost is the process exit.
- The node id **is** on the Timeout line only under `-v`. Under `-q` the hung test
  is identifiable only from the stack dump's test frame. `pytest.ini` documents
  "run with -v".
- Consequence: to count every hang in a run, iterate. `tooling-3/suite-until-complete.sh`
  does exactly that (run -> parse the Timeout line -> `--deselect` -> run again -> until
  a pass ends with a normal pytest summary; a pass that ends with neither a Timeout
  block nor a summary is flagged, not counted). It also streams every completed
  test's (setup, call, teardown) durations to JSONL via a tiny in-process plugin
  (`tooling-3/durations_jsonl.py`, registered through `tooling-3/run_pytest.py`
  because the embeddable interpreter's `python310._pth` makes it *isolated*:
  `PYTHONPATH` is ignored, so `-p durations_jsonl` cannot resolve). This was
  necessary because everything pytest prints at session end (`--durations`,
  `--junitxml`) is lost when `os._exit` fires.

## 3. AC #2 — the timeout is derived from evidence

### Measurement run (`tooling-3/measure-durations.sh M1`)

Every directory that directly contains `test_*.py` was run as its own pytest
process with an explicit non-recursive file list (the same shape as Story 20.8's
guard, so the numbers are comparable), under a provisional `--timeout=900`; a
directory that hit the guard was re-run file by file. 11:01 -> 12:37 wall, most of
it the 900 s guard firing five times (section 5).

Per-test totals are **setup + call + teardown**, because that is what
pytest-timeout's default `func_only = False` budget covers.

| statistic (passing tests, n = 2,875) | value |
|---|---|
| max | **7.63 s** — `tests/unit/test_app_compile_warmup_qasync.py::test_warmup_reaches_the_metric_under_a_real_qasync_loop` (setup phase: Story 20.5's out-of-process qasync driver) |
| 2nd / 3rd | 4.90 s / 4.81 s — the other two tests in the same file |
| p99 | 0.93 s |
| p95 | 0.11 s |
| median | 0.002 s |
| tests > 1 s | 29 (of which 20 are the ~2.6-3.0 s **first test in a process**: torch + PyQt6 import cost lands on the first test's setup) |
| tests > 5 s | 1 |
| tests > 10 s | 0 |
| sum of all passing durations | 124.6 s |

Slowest 10 passing (full top-30 list: `tooling-3/analyze_durations.py measure-M1.jsonl 30`):

```
7.63s  tests/unit/test_app_compile_warmup_qasync.py::test_warmup_reaches_the_metric_under_a_real_qasync_loop
4.90s  tests/unit/test_app_compile_warmup_qasync.py::test_the_plain_hand_off_is_destroyed_under_the_same_loop
4.81s  tests/unit/test_app_compile_warmup_qasync.py::test_shield_wait_for_hand_off_is_also_destroyed
3.01s  tests/unit/services/test_audio_coordinator.py::TestStopStreamingSessionDrain::test_wait_for_drain_under_producer_faster_than_realtime_waits_for_queued_audio
2.96s  tests/ui/test_playback_last.py::TestReplayButtonPresence::test_replay_button_exists            (first-in-process import)
2.91s  tests/ui/test_generate_gate_during_priming.py::test_priming_disables_generate_and_releasing_re_enables_it  (first-in-process)
2.85s  tests/ui/test_accessibility.py::TestAccessibleNames::test_text_input_has_accessible_name        (first-in-process)
2.85s  tests/ui/test_model_switching_indicator.py::TestModelCallbackConnectionCode::test_callback_connection_code_exists (first-in-process)
2.78s  tests/ui/test_settings_panel_access.py::TestSettingsDialogInit::test_dialog_creation            (first-in-process)
2.75s  tests/ui/test_compact_always_on_top_window.py::TestWindowDimensions::test_window_default_size   (first-in-process)
```

The whole-suite final pass (`suite-S1-iter28.jsonl`, n = 2,892 passing) agrees:
max 4.89 s, p99 0.31 s, none above 5 s. No integration test constructs a real
model in the default suite (nothing in `tests/integration` exceeded 0.2 s; the
directory runs in ~11 s wall including startup), so the "real model construction"
case the story anticipated does not exist in the current suite and needs no mark.

### Chosen value: `timeout = 60`

- ~8x the slowest honest test (7.63 s), ~65x the p99.
- Leaves room for a cold torch/PyQt6 import on a slower ship-target host
  (RTX 30xx-class laptop), which lands on one test's setup per process.
- Small enough that each hang costs one minute in the iterating run (27 hangs
  = ~27 min of waiting) rather than five.
- No test is individually marked with `@pytest.mark.timeout(N)`: nothing comes
  within an order of magnitude of the default.
- The value was NOT raised at any point to make a hang disappear. The provisional
  900 s guard used only for measurement fired five times; every one of those five
  is a genuine stall (section 5), and all reproduced at 60 s.

Anomaly worth recording: the 20.8 AFTER per-directory run reported `tests/settings`
at 813 s and `tests/unit/ui/dialogs/voice_design_studio` at 538 s. Here the same
directories took 8 s and (until the guard fired) seconds per test. Those 20.8 wall
times were therefore stalls that eventually returned, not slow tests; under a 60 s
default they would now be named. That is the intended behaviour.

## 4. AC #5 — the conftest DLL-ordering invariant survives

Static: `pytest_timeout.py` imports only stdlib (`inspect, os, signal, sys,
threading, time, traceback, collections`) plus `pytest`; it registers **no**
`pytest_load_initial_conftests` hook (the hook pytest-cov used to jump ahead of
`tests/conftest.py`); it installs nothing until `pytest_runtest_protocol` — long
after the conftest has imported torch. It starts one `threading.Timer` per test;
no C tracer, no DLL loads.

Empirical: `grep -c "WinError 1114"` = **0** in every log of this story
(`measure-M1.log`, all 28 `suite-S1-iter*.log`). The streaming suites ran under the
active plugin and passed: `tests/unit/services/tts_streaming` 224/224 (6 files, 6 s
wall), `tests/test_qwen_tts_internals.py` 18/18, `tests/unit/services` 397/397.

No workaround is needed; `python -m pytest ...` is the invocation, exactly as
before. (`tools/run_cov.py` remains the way to run `--cov`; that constraint is
pytest-cov's and unchanged.)

## 5. AC #4 — every timeout, by id, with stall location

### Whole-suite run (`tooling-3/suite-until-complete.sh S1`)

`pytest tests/ -v -rfE` from the repo root, 12:38 -> 13:19 wall, **28 passes**;
passes 1-27 each ended at one timeout (~73-75 s wall each: ~12-15 s of tests plus
the 60 s budget), pass 28 ran to a normal summary:

```
===== 47 failed, 2896 passed, 27 deselected, 4 errors in 82.11s (0:01:22) =====
collected 2970 items / 27 deselected / 2943 selected
```

Between iterations 8 and 9 the loop was stopped and resumed (the initial
`MAXITER` cap was too low once it became clear that a whole test class hangs one
test per pass); the resume seeds `--deselect` from the recorded `.hung` file, so
iteration 9 onward is exactly what an uninterrupted loop would have run. A bogus
"completed" line written by the interrupted pass was removed and its partial log
discarded; the driver now refuses to count a pass that ends without a pytest
summary.

### The 27 timeouts

Table generated by `tooling-3/timeout_table.py S1` from the `.hung` file, each
iteration's stack dump, and the measurement-run JSONL (also saved as
`tooling-3/timeout-table-S1.md`). "Per-directory run" answers the AC's "which
directory-level run it passed in, if it passed anywhere".

| # | test id | stalled at (innermost project frame) | per-directory run (measure-M1) |
|---|---|---|---|
| 1 | `tests/settings/test_reset_to_defaults.py::TestResetQuickSpeak::test_reset_quick_speak_entries` | `quick_speak_settings_widget.py:152 _load_entries -> QMessageBox.warning(` | **passed** (2.64 s) |
| 2 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_control_exists_and_is_a_two_option_combo` | same | TIMED OUT (900 s guard, dir-level AND file-level) |
| 3 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_control_lives_on_the_interface_tab` | same | not reached (file aborted at #2) |
| 4 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_option_labels_are_user_facing` | same | not reached |
| 5 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_reflects_persisted_value_on_open[True]` | same | not reached |
| 6 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_reflects_persisted_value_on_open[False]` | same | not reached |
| 7 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_selection_saves_through_existing_path[True]` | same | not reached |
| 8 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_selection_saves_through_existing_path[False]` | same | not reached |
| 9 | `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_saved_value_round_trips_through_persistence` | same | not reached |
| 10 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_is_last` | same | TIMED OUT (900 s guard, file-level) |
| 11 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_widget_is_panel_instance` | same | not reached |
| 12 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_dialog_exposes_clear_comms_test_playback_signal` | same | not reached |
| 13 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestSignalForward::test_panel_signal_re_emits_through_dialog` | same | not reached |
| 14 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestSignalForward::test_panel_signal_with_file_payload` | same | not reached |
| 15 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestOkPersistence::test_save_current_settings_pushes_panel_state_into_current_settings` | same | not reached |
| 16 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestOkPersistence::test_load_current_settings_hydrates_panel_from_current_settings` | same | not reached |
| 17 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestCancelLosslessness::test_cancel_button_does_not_emit_settings_changed` | same | not reached |
| 18 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestCancelLosslessness::test_persisted_settings_unchanged_after_panel_mutation_then_cancel` | same | not reached |
| 19 | `tests/ui/test_settings_dialog_clear_comms_tab.py::TestResetToDefaults::test_reset_restores_panel_to_defaults` | same | not reached |
| 20 | `tests/ui/test_settings_dialog_streaming_tab.py::TestStreamingTabPresence::test_streaming_tab_exists` | same | TIMED OUT (900 s guard, file-level) |
| 21 | `tests/ui/test_settings_dialog_streaming_tab.py::TestStreamingTabPresence::test_streaming_tab_widget_is_panel_instance` | same | not reached |
| 22 | `tests/ui/test_settings_dialog_streaming_tab.py::TestStreamingPanelBindsToStagingCopy::test_panel_is_bound_to_current_settings_not_original` | same | not reached |
| 23 | `tests/ui/test_settings_dialog_streaming_tab.py::TestStreamingPanelBindsToStagingCopy::test_changing_dropdown_writes_to_current_settings` | same | not reached |
| 24 | `tests/ui/test_settings_dialog_streaming_tab.py::TestStreamingPanelBindsToStagingCopy::test_load_current_settings_hydrates_dropdown` | same | not reached |
| 25 | `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py::TestRegenerateAllVariations::test_regenerate_triggers_generate_flow` | `voice_design_studio_dialog.py:471 _on_generate_requested -> QMessageBox.warning(` | TIMED OUT (900 s guard, dir-level AND file-level) |
| 26 | `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py::TestSessionManagement::test_regenerate_clears_variant_files` | same | not reached |
| 27 | `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py::TestSessionManagement::test_regenerate_preserves_non_variant_files` | same | not reached |

"Not reached" = in the measurement run the file's process had already been
`os._exit`ed by an earlier test in the same file, so this test has no
per-directory result at all; it therefore has never been observed passing in this
story. Only #1 has: it passed in the `tests/settings` directory run (and in the
20.8 AFTER run) and hangs only under whole-suite ordering — the ordering-dependent
class the story called out. #2 is the Story ui-1 test the story named; it hangs
even in isolation here.

### Two stall sites (full project-frame chains, from the MainThread stack dumps)

**Site A — 24 tests (#1-#24): a modal warning inside `SettingsDialog.__init__`**

```
src/myvoice/ui/components/settings_dialog.py:131        __init__            self._create_ui()
src/myvoice/ui/components/settings_dialog.py:171        _create_ui          self._create_quick_speak_tab()
src/myvoice/ui/components/settings_dialog.py:752        _create_quick_speak_tab   QuickSpeakSettingsWidget(self.quick_speak_service)
src/myvoice/ui/components/quick_speak_settings_widget.py:49   __init__      self._load_entries()
src/myvoice/ui/components/quick_speak_settings_widget.py:152  _load_entries QMessageBox.warning(self, "Load Error", ...)
```

`_load_entries` wraps its body in `except Exception` and shows a **modal**
`QMessageBox.warning`, which under pytest waits for a click that never comes. The
exception it is reporting is visible in the 20.8 AFTER log's captured setup log
for the same tests: `Error loading Quick Speak entries: setCurrentText(self, text:
Optional[str]): argument 1 has unexpected type 'MagicMock'` — i.e. the tests'
quick-speak stub returns a `MagicMock` where a profile-name string is expected.
Every test in the four files that builds a `SettingsDialog` with such a stub
reaches this line. (Files that build `SettingsDialog` with a richer stub —
`test_settings_panel_access.py` (16 constructions), `test_window_transparency.py`
(10), `test_compact_always_on_top_window.py` (6) — did not hang.)

Open question for the fix story, recorded not answered: in the 20.8 AFTER run the
*same* exception was logged for `test_clear_comms_tab_is_last` and the test went on
to **fail** on its assertion (`'API Access' == 'Clear Comms'`), i.e. the modal did
not block there; and 20.8's `tests/ui` directory run passed 705 tests in 33 s with
`test_close_to_tray_toggle.py` present (`git log` dates it 2026-09-01; the run is
2026-09-02). Here the modal blocks in every ordering including a single file. None
of the four hanging test files patch `QMessageBox.warning`; whatever neutralised
the modal on 2026-09-02 is not in the tree now or was ordering/environment.
`QT_QPA_PLATFORM` is unset in both. This story does not resolve it.

**Site B — 3 tests (#25-#27): a modal warning on the regenerate path**

```
src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:693  _on_regenerate_requested  self._on_generate_requested(description, preview_text, language)
src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:471  _on_generate_requested    QMessageBox.warning(self, "Service Unavailable", ...)
```

Reached when `self._tts_service` is falsy — the dialog is constructed without a
TTS service in these tests and the regenerate path guards with a modal. Story
20.8's BEFORE run named this file as its one HUNG file; its AFTER run got through
it in 538 s with 30 failures — consistent with the same modal, sometimes
dismissed, sometimes not.

## 6. AC #4 — the pre-existing failure set is unchanged

Baseline: `20-8-regression-AFTER.summary` — 2921 passed, **49 FAILED**, 5 errors,
0 hung files (per-directory guard, `-q -rf`). Its "2975" total double-counts the 5
teardown errors (pytest counts a test that passes and then errors at teardown in
both columns); the population is 2970, identical to this story's `collected 2970`.

Final whole-suite pass here: **47 FAILED**, 4 errors, 2896 passed, 27 deselected
(= the 27 timeouts).

`diff` of sorted FAILED ids, baseline vs final (`tooling-3/baseline-20-8-AFTER.failed`
vs `tooling-3/suite-S1-final.failed`):

```
< FAILED tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_is_last
< FAILED tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_widget_is_panel_instance
```

Those two are rows #10 and #11 of the timeout table: they moved from the FAILED
column to the TIMEOUT column. The other **47 are identical in identity**. So:
49 baseline = 47 FAILED + 2 TIMEOUT, and the 25 other TIMEOUT rows are the
*additional* named rows the AC asked for — tests the baseline reported as
"passed" (#1) or could not report at all.

Errors: the 4 teardown errors here (`test_audio_player_widget.py`:
`test_get_audio_path`, `test_set_audio_file_updates_duration`,
`test_set_none_disables_play`, `TestPlaybackState::test_get_duration`) are 4 of the
baseline's 5. The 5th, `TestAudioFileManagement::test_set_audio_file_enables_play`
teardown, is intermittent: present in the baseline, present in 1 of the 2
measurement passes that reached it, absent in the final pass. All are
`Media player error: Error.FormatError` teardown noise from the same widget.

## 7. AC #3 — the dependency does not ship (verified by reading)

- `requirements.txt`: entry sits under the `# Dev-only test tooling` banner beside
  `pytest-cov`, tagged `# dev-only`, same comment shape including a Windows note.
- `build_tools/requirements-production.txt`: no `pytest*` entry; its "Excluded from
  Production" block explicitly names `pytest, pytest-*`. Unchanged.
- `build_tools/myvoice.spec` line 540: `'pytest'` is in PyInstaller `excludes`;
  nothing under `src/` imports `pytest_timeout`, and PyInstaller bundles by import
  analysis from `src/myvoice/main.py`, so the plugin cannot be collected.
- `build_tools/build_release.bat` [Bundle Prerequisites] probes 1-4 (lines
  142-213) test only `Include\Python.h`, `libs\python310.lib`,
  `site-packages\triton\__init__.py`, and `cuda_toolkit_subset\bin\cudart64_*.dll`.
  Nothing enumerates site-packages or pins its contents. Unaffected.
- Not built, per the AC.

## 8. Findings, deviations, disagreements

1. **`thread` method = process exit, one hang per run.** The story's AC #1 reads as
   if a single run would tally every timeout; on Windows it cannot. This is not a
   configuration choice — it is how the plugin is written (`os._exit(1)` in
   `timeout_timer`). What the story actually needs — countable, named hangs and a
   comparable failure set — is delivered, but by iterating. Anyone running the
   suite by hand should expect `pytest tests/` to stop at the first hang with a
   stack dump, and should run with `-v`. If a future story wants a single run that
   keeps going past hangs, that is a different mechanism (e.g. a per-test
   subprocess runner or a Qt-aware timeout that raises into the event loop), and
   it should be its own decision, not a quiet change here.
2. **One class of defect, 27 rows.** Both stall sites are production code calling a
   blocking `QMessageBox.warning` on an error path that the test's stub triggers.
   The fix story is likely two changes, not 27. The stub cause for Site A is
   already visible (`MagicMock` where a profile name string is expected).
3. **Two tests changed column, not state.** #10 and #11 were "failures" in 20.8
   only because the modal happened not to block that day; they were never passing.
4. **Stale-path `.pyc` files.** 132 `tests/**/__pycache__/*.pyc` embed
   `G:\MyVoicePublicInst\...` as their source path (the stack dumps show test
   frames at `G:\...`). pytest validates its rewritten-pyc cache by source mtime and
   size, so these were compiled from identical sources (a mtime-preserving copy),
   not stale — but tracebacks that point at a drive that may not exist are
   misleading. `__pycache__/` is gitignored; deleting `tests/**/__pycache__` once
   would clean it. Not done here (out of scope, and it would change nothing about
   the run).
5. **`config/settings.json.pre20-3` and `test_localsystem.txt`** were already
   untracked in the working tree before this story and were not touched.
6. **Nothing was committed.** Changes are on the working tree of
   `tooling/3-pytest-timeout`: `pytest.ini` (new), `requirements.txt` (modified),
   plus the gitignored `_bmad-output/` files.

## 9. Reproduce

```
# single run (stops at the first hang, names it, dumps stacks, exit 1)
python310\python.exe -m pytest tests/ -v -rfE

# drive to completion, enumerating every hang (resumable; logs + JSONL per pass)
bash _bmad-output/implementation-artifacts/tooling-3/suite-until-complete.sh <label>

# per-directory duration distribution under a provisional guard
GUARD=900 bash _bmad-output/implementation-artifacts/tooling-3/measure-durations.sh <label>
python310\python.exe _bmad-output/implementation-artifacts/tooling-3/analyze_durations.py <jsonl> 30

# stall frames for every Timeout block in any -v log
python310\python.exe _bmad-output/implementation-artifacts/tooling-3/stall_frames.py <log>
```
