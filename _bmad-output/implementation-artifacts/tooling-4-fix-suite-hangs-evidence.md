# Story tooling-4 — Evidence: Fix the Two Suite-Hang Sites

Date: 2026-09-14. Branch `tooling/4-fix-suite-hangs` off clean merged `main`
(`b2eda6c`, which carries tooling-3's `pytest.ini`: `timeout = 60`,
`timeout_method = thread`). Interpreter: bundled `python310\python.exe`
(3.10.11), pytest 9.0.2, pytest-timeout 2.4.0, pytest-qt 4.5.0, PyQt6 6.9.1.

Raw log and id lists for the single verification run live in `tooling-4/`
next to this file (`_bmad-output/`-gitignored).

## 0. Result in one paragraph

A single `pytest tests/ -v -rfE` now runs to a normal pytest summary:
**49 failed, 2924 passed, 5 errors in 84.02 s**, collected 2973, **0 `+++ Timeout +++`
blocks, no `os._exit`**. All 27 tooling-3 timeouts are gone: 25 pass, 2
(`test_clear_comms_tab_is_last`, `test_clear_comms_tab_widget_is_panel_instance`)
now fail on their real assertion (`'API Access' == 'Clear Comms'`) — the same two
that were FAILED rows in the 20.8 baseline and that tooling-3 saw move to the
TIMEOUT column. The 47 pre-existing failures are unchanged in count and identity;
49 = 47 + those 2, i.e. the failure set is byte-identical to 20.8's 49. Both stall
sites were fixed on the test side only (AC #1, #2); no production `QMessageBox`
was touched (AC #4). The defensive raise-on-modal net (AC #3) was audited and
shipped — it did not fire on anything in the run (0 hits in the log).

## 1. What changed (tracked files)

| File | Change |
|---|---|
| `tests/conftest.py` | **(a)** `quick_speak_service_stub` fixture — `MagicMock(spec=QuickSpeakService)` with typed return values for every call the widget makes at construction and on the reset path. **(b)** `UnexpectedModalDialogError` + session-scoped autouse `_fail_on_unexpected_qmessagebox` that replaces `QMessageBox.warning/critical/information/question` and `QMessageBox.exec` with raisers, and function-scoped autouse `_qmessagebox_passthrough` that honours the `@pytest.mark.qmessagebox_passthrough` opt-out. The existing `_suppress_main_window_close_confirm` fixture is untouched. |
| `pytest.ini` | Registers the `qmessagebox_passthrough` marker (with a comment). `timeout` / `timeout_method` unchanged. |
| `tests/settings/test_reset_to_defaults.py` | `test_reset_quick_speak_entries` takes `quick_speak_service_stub` instead of building `MagicMock()`. |
| `tests/ui/test_close_to_tray_toggle.py` | Module helper `_make_dialog(settings, qtbot)` (bare `MagicMock()` stub) becomes a `make_dialog` fixture factory built on `quick_speak_service_stub`; 8 call sites in `TestInterfaceTabCloseBehaviorControl` updated. The close-event tests (which patch `QMessageBox.question` themselves) are untouched. |
| `tests/ui/test_settings_dialog_clear_comms_tab.py` | `dialog` fixture takes `quick_speak_service_stub`; unused `MagicMock, patch` import removed. |
| `tests/ui/test_settings_dialog_streaming_tab.py` | Same; unused `MagicMock` import removed. |
| `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py` | New `tts_service_stub` fixture (`MagicMock(spec=QwenTTSService)`, `is_running() -> True`) and `_capture_async_dispatch(dialog)` helper. The three regenerate tests construct the dialog with the stub and capture the async hand-off. New class `TestGenerateGuardsOnTtsService` (3 tests) covers both guards with `QMessageBox.warning` patched. |

`src/` is untouched: `git diff --stat` lists only the seven files above.

## 2. AC #1 — Site A: the shared quick-speak stub

Enumerated from the code, not guessed. `QuickSpeakSettingsWidget.__init__`
(`quick_speak_settings_widget.py:36-51`) calls `_create_ui()` (no service
calls) then `_load_entries()` (`:129-156`), which makes exactly three service
calls and feeds each straight into Qt:

| call | annotated return | consumer |
|---|---|---|
| `get_profiles()` | `List[str]` | `QComboBox.addItems(profiles)` |
| `get_current_profile()` | `str` | `QComboBox.setCurrentText(current_profile)` — **this is the one that raised** `TypeError: argument 1 has unexpected type 'MagicMock'` |
| `get_entries()` | `List[QuickSpeakEntry]` | `_populate_table(entries)` |

`SettingsDialog.__init__` itself (`settings_dialog.py:121-126`) only stores the
service when one is passed. The reset path the `test_reset_to_defaults` test
exercises (`_reset_quick_speak_entries`, `:1619-1635`) calls
`_create_default_profile()` and `load_entries()` on the service, then
`widget.refresh_entries()` → `_load_entries()` again.

The fixture sets `get_profiles -> ["default"]`, `get_current_profile ->
"default"`, `get_entries -> []`, `load_entries -> []`, `_create_default_profile
-> None`, and `switch_profile/create_profile/delete_profile -> True` (the
widget's slots read those as `bool`). `spec=QuickSpeakService` makes a mistyped
attribute an `AttributeError` rather than a silent `MagicMock`. It lives in
`tests/conftest.py` because the four consumers span `tests/settings/` and
`tests/ui/`. The old stubs' `load_entries = MagicMock()` line was a no-op
(`MagicMock()` already has it) — it never addressed the three reads above.

Reproduced before the fix: `pytest tests/settings/test_reset_to_defaults.py -x`
alone hit the timeout at `quick_speak_settings_widget.py:152 QMessageBox.warning(`.

## 3. AC #2 — Site B: the regenerate tests

`_on_regenerate_requested` (`voice_design_studio_dialog.py:680-693`) clears
variant files and then calls `_on_generate_requested`, whose first two
statements are the guards: `if not self._tts_service` → `QMessageBox.warning
("Service Unavailable", ...)`; `if not self._tts_service.is_running()` →
`QMessageBox.warning("Service Not Ready", ...)`. `__init__` never touches
`_tts_service`, so the stub only has to satisfy the guards.

Past the guards the flow is real: `set_generating(True)`, store
`_last_description/_last_preview_text/_last_language`, then
`_generate_next_variant` → `_run_async_task(coro)` → `asyncio.ensure_future`.
Under pytest that would leave a pending task wrapping `await
MagicMock().generate_voice_design(...)` on the main-thread loop, to run
inside whichever later test happens to drive that loop, after the dialog is
gone (the `memory/qasync_task_destruction_hazard.md` class). So
`_capture_async_dispatch(dialog)` replaces `dialog._run_async_task` on the
instance with a recorder that closes the coroutine. The single run shows 0
`was never awaited` / `Task was destroyed` lines.

- `test_regenerate_triggers_generate_flow` keeps its spy on
  `_on_generate_requested` and now additionally asserts the flow really
  started: `_is_generating is True`, `_last_description`/`_last_preview_text`
  match, and exactly one coroutine was dispatched. Before, the "routing" it
  claimed to test ended at the guard's modal.
- `test_regenerate_clears_variant_files` and
  `test_regenerate_preserves_non_variant_files` construct
  `VoiceDesignStudioDialog(tts_service=tts_service_stub)` + capture. Their
  file assertions are unchanged.
- `test_regenerate_uses_current_description` / `..._preview_text` replace
  `_on_generate_requested` wholesale and never reach the guard — untouched.

The guard had no test of its own. `TestGenerateGuardsOnTtsService` adds three,
each patching `QMessageBox.warning` via `monkeypatch` and asserting the call
and its title, plus `_is_generating is False`:
`test_generate_without_service_warns_instead_of_starting` (direct),
`test_regenerate_without_service_warns_instead_of_starting` (the exact shape
that hung — per `memory/code_review_regression_test_exact_class.md`), and
`test_generate_with_stopped_service_warns_instead_of_starting`
(`is_running() -> False`). The guards themselves are unchanged.

## 4. AC #3 — the defensive net: audit first, then the decision

### 4.1 Audit: every `QMessageBox` reference under `tests/`

`grep -rn QMessageBox tests/ --include=*.py` (excluding `__pycache__`) — 33 lines
in 6 files:

| file:line | what it does | interaction with a raise-on-unpatched-modal net |
|---|---|---|
| `tests/conftest.py:54,73` | comments in the `_force_quit` close-confirm bypass | none (text). The bypass flips the production flag so `closeEvent` never reaches its `QMessageBox.question`; the net never sees it. Untouched. |
| `tests/settings/test_reset_to_defaults.py:14,139` | `patch.object(QMessageBox, 'exec', return_value=No)` around `_on_reset_defaults` (production builds a `QMessageBox(self)` and calls `.exec()`, `settings_dialog.py:1561-1581`) | patches it itself → unaffected. Note it patches **`exec`**, not one of the four statics — the reason the net covers `exec` too (below). |
| `tests/ui/test_close_to_tray_toggle.py:19,33,325-410` | 4 `patch.object(QMessageBox, "question", return_value=Yes/No)` around `MainWindow.closeEvent` | patch themselves → unaffected. `patch.object` restores what was in `QMessageBox.__dict__` on entry — the raiser — so the net is back after each. |
| `tests/ui/test_settings_dialog_clear_comms_tab.py:28,245-247` | `monkeypatch.setattr(QMessageBox, "exec", fake_exec)` around `_on_reset_defaults` | patches itself → unaffected (monkeypatch's class-attribute undo restores the raiser). |
| `tests/unit/ui/dialogs/voice_design_studio/test_description_path_panel.py:1393-1409` | `patch('PyQt6.QtWidgets.QMessageBox.question')` returning No | string target resolves to the same class attribute → unaffected. |
| `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py:281-414` | 5 tests `monkeypatch.setattr(QMessageBox, 'warning', ...)` (+1 `'question'`) in `TestSaveFromDescription` | patch themselves → unaffected; all 5 pass before and after. (The file's 19 pre-existing failures are in other classes — `TestSampleTab*`, `TestSaveButtonState`, `TestVoiceDesignStudio*`, one in `TestSessionManagement` — and are identical before and after.) |

Beyond the grep: no test **relies on a real dialog**, and none can — every
blocking `QMessageBox` entry point runs a nested event loop waiting for a click,
so any test that reached one unpatched was, by definition, one of the 27
timeouts, and all 27 are now traced to the two stubs fixed above. The single
run confirms it: **0 `UnexpectedModalDialogError` in the log**. Also checked:

- No production module binds a static at import time (`= QMessageBox.warning`
  etc.: 0 matches in `src/`); every one of the 57 call sites is
  `QMessageBox.<name>(` at call time, so a class-attribute replacement covers
  every import alias.
- No test patches a module-level `QMessageBox` name (e.g.
  `patch('myvoice.ui.main_window.QMessageBox')`) — such a patch would still work
  (it replaces the whole class in that module), it just would not go through
  the net.
- No module/class/session-scoped fixture constructs a widget (all 63
  non-function-scoped fixtures are `qapp`-style); so the pytest ordering rule
  "higher-scoped fixtures set up before function-scoped autouse ones" has no
  live consequence today. The net is nevertheless installed **session-wide** so
  it cannot develop that hole later.
- Two tests carry `@pytest.mark.qt_no_exception_capture`
  (`tests/integration/test_streaming_tts_smoke.py:1247,1315`). For every other
  test pytest-qt installs `sys.excepthook` capture for setup/call/teardown, so a
  raiser fired inside a C++-invoked slot (button click → Python slot) is
  reported as `Exceptions caught in Qt event loop` — verified with a scratch
  test (a `QPushButton.clicked` slot calling `QMessageBox.information`: CALL
  ERROR naming the title/text, not `qFatal`). In those two opted-out tests the
  same raise would abort the process via `qFatal` — immediate and named on
  stderr, still not a hang; and neither test reaches a modal today.

### 4.2 Decision: ship it, with two deliberate widenings

Nothing in the audit needs adapting or opting out, so AC #3 is implemented.
Two choices go beyond the AC's literal wording, both because the audit showed
the narrower version had a known hole:

1. **`QMessageBox.exec` is covered as well as the four statics.** Two
   production sites build a box and call `.exec()` (`settings_dialog.py:1561`,
   `app.py:794`); two existing tests patch exactly that. A net over the four
   statics only would leave the instance route — the same defect class — able
   to hang. Setting `exec` on `QMessageBox` shadows the inherited
   `QDialog.exec` for message boxes only; `MonkeyPatch.undo()` `delattr`s it.
2. **`UnexpectedModalDialogError` derives from `BaseException`.** Most of these
   modals sit inside `try: ... except Exception:` (`app.py:793` wraps the
   `.exec()` itself in one). An `Exception`-derived raiser would be swallowed
   there and the test would pass silently — a hole nobody would see. pytest
   reports a `BaseException` subclass as an ordinary failure (verified). Only a
   bare `except:` can hide it now.

Per-test opt-out: `@pytest.mark.qmessagebox_passthrough` (registered in
`pytest.ini`) restores the real implementations for that one test via
`monkeypatch`, and the raisers return afterwards. No test uses it; it exists
for a future test that drives a real box with a `QTimer`.

Scratch verification (temporary file under `tests/`, deleted afterwards;
13 passed + 1 strict xfail): static call raises naming title/text; instance
`exec` raises naming `windowTitle()`/`text()`; escapes `except Exception`;
slot-raise → pytest-qt failure; `monkeypatch.setattr`, `patch.object` and
`patch('PyQt6.QtWidgets.QMessageBox.question')` overrides each win and each
restore to the raiser; marker opt-out restores the real Qt attributes (`exec`
leaves `QMessageBox.__dict__`) and they come back after; the exact Site A
shape (`SettingsDialog(..., quick_speak_service=MagicMock())`) now raises in
milliseconds with `'Load Error'` in the message; the exact Site B shape
(`VoiceDesignStudioDialog()._on_regenerate_requested()`) raises with
`'Service Unavailable'`.

## 5. AC #4 — no production dialog weakened

`git diff --stat` touches no file under `src/`. `quick_speak_settings_widget.py`
and `voice_design_studio_dialog.py` are byte-identical to `main`. The
`_force_quit` bypass fixture in `tests/conftest.py` is unchanged (the new code is
appended below it).

**Finding for a UI story (recorded, not acted on):**
`QuickSpeakSettingsWidget._load_entries` (`quick_speak_settings_widget.py:129-156`)
wraps profile+entry loading in `except Exception` and shows a modal
`QMessageBox.warning` — and it runs from `__init__`, i.e. from inside
`SettingsDialog.__init__` via `_create_quick_speak_tab`. Any failure in the
quick-speak service (corrupt profile file, bad config path) therefore blocks
the *entire* Settings dialog from finishing construction until the user
dismisses a warning that appears before the dialog does, and the dialog then
opens with an empty Quick Speak tab. A non-modal indication in the tab (or
deferring the warning until after the dialog is shown) would be the UX-correct
shape. Same pattern is worth a look at `SettingsDialog.__init__` generally.

## 6. AC #5 — the single run, and the failure-set diff

Command, from the repo root (`tooling-4/suite-T4-run1.log`):

```
python310\python.exe -m pytest tests/ -v -rfE
collected 2973 items
============ 49 failed, 2924 passed, 5 errors in 84.02s (0:01:24) =============
```

- `+++ Timeout +++` blocks: **0**. `os._exit`: none (the summary line was
  printed; exit code 1 is the ordinary "some tests failed").
- `WinError 1114`: 0 (the DLL-ordering invariant survives the net's
  `from PyQt6.QtWidgets import QMessageBox` inside the session fixture — that
  import happens at first-test setup, long after the conftest's torch import).
- `UnexpectedModalDialogError`: 0 hits.
- Population: 2973 = tooling-3's 2970 + the 3 new guard tests.
- Passed: 2924 = 2896 (tooling-3 final) + 25 former timeouts + 3 new.

**All 27 former timeouts, by outcome** (ids from
`tooling-3/timeout-table-S1.md`, matched by fixed string in the `-v` log):

| outcome | rows | ids |
|---|---|---|
| PASSED | 25 | #1-#9, #12-#27 |
| FAILED on a genuine assertion | 2 | #10 `tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_is_last` — `AssertionError: assert 'API Access' == 'Clear Comms'`; #11 `...::test_clear_comms_tab_widget_is_panel_instance` — last tab is an `APIAccessSettingsPanel`, not `ClearCommsSettingsPanel`. The Clear Comms tab is no longer the last tab; the API Access tab was added after it. Real, pre-existing, and not this story's. |

**Failure-set diff** (`tooling-3/suite-S1-final.failed` vs
`tooling-4/suite-T4-run1.failed`, ids only, sorted):

```
8a9,10
> FAILED tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_is_last
> FAILED tests/ui/test_settings_dialog_clear_comms_tab.py::TestClearCommsTabPresence::test_clear_comms_tab_widget_is_panel_instance
```

The 47 are identical in identity; the 2 additions are #10/#11 above, and they
are the 2 ids tooling-3 §6 showed leaving the 20.8 FAILED set for the TIMEOUT
column. So against the 20.8 baseline (`tooling-3/baseline-20-8-AFTER.failed`,
49 ids) the FAILED set is now **identical, 49 = 49**.

**Errors:** 5 vs tooling-3's 4 — the 5th is
`test_audio_player_widget.py::TestAudioFileManagement::test_set_audio_file_enables_play`
teardown, which tooling-3 §6 recorded as intermittent (present in the 20.8
baseline, present in 1 of 2 measurement passes, absent in its final pass). All
5 are the same `Media player error: Error.FormatError` teardown noise. Not
touched.

**Tests that patch a modal themselves, in the same run:**
`test_reset_button_triggers_confirmation` passed; `test_reset_restores_panel_to_defaults`
passed; `test_description_path_panel.py` 158/158 passed; `TestCloseEventHonoursToggle`
4/4; `TestMeasurementModeUnaffected` 3/3; `TestSaveFromDescription` 5/5;
`TestGenerateGuardsOnTtsService` 3/3 (new).

## 7. Findings, deviations, disagreements

1. **The net is wider than the AC's four names** (`exec`, `BaseException`) —
   §4.2 says why. If the reviewer wants the literal four-static version, both
   widenings are one-line removals; but I would ship them.
2. **Session-scoped install rather than a per-test autouse fixture.** The AC
   says "autouse conftest fixture"; the install fixture *is* autouse, at session
   scope, with a function-scoped autouse companion for the opt-out. Reason in
   §4.1 (module-scoped widget fixtures would otherwise sit outside the net).
3. **Site B's tests were not testing what their names said.** Before this
   story `test_regenerate_triggers_generate_flow` stopped at the guard's
   modal; the "flow" was never entered. It now asserts the flow started. That is
   a strengthening, not a behaviour change.
4. **tooling-3's open question** ("why did the same modal *not* block on
   2026-09-02?") is not resolved here and no longer matters for the suite: the
   modal is no longer reached, and if it ever is again the net names it.
5. **`tests/**/__pycache__` on `G:`** still not cleaned (story's "NOT" list; a
   separate commit).
6. **Nothing was committed.** Changes are on the working tree of
   `tooling/4-fix-suite-hangs`; the `_bmad-output/` files are gitignored.
   Line endings: `core.autocrlf=true`, working copy CRLF, repo LF — checked
   for all seven files.

## 8. Reproduce

```
# the single run (should end with a normal summary and no Timeout block)
python310\python.exe -m pytest tests/ -v -rfE

# the two fixed sites alone
python310\python.exe -m pytest tests/settings/test_reset_to_defaults.py tests/ui/test_close_to_tray_toggle.py tests/ui/test_settings_dialog_clear_comms_tab.py tests/ui/test_settings_dialog_streaming_tab.py tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py -q -rfE

# failure-set diff
grep "^FAILED" <log> | sed 's/ - .*//' | sort | diff <(sed 's/ - .*//' _bmad-output/implementation-artifacts/tooling-3/suite-S1-final.failed | sort) -
```
