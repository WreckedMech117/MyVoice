# Story ui-2: No Modal From Inside Settings Dialog Construction

Status: done — 2026-09-14. Load error surfaced in-place; 9 new tests; verified to fail against the old code via the tooling-4 net; 2,978 passed / 1 xfailed.

<!-- Out-of-epic UI story. Source: tooling-4 AC #4 finding. -->
<!-- Risk: LOW. One error-path change in one widget. -->

## Story

As **a MyVoice user opening Settings**,
I want **a Quick Speak load failure to be shown inside the dialog, not as a blocking box before the dialog appears**,
so that **Settings is always reachable even when one of its tabs has a data problem**.

## Context

`QuickSpeakSettingsWidget._load_entries` (`quick_speak_settings_widget.py:145-156`)
catches any exception and calls `QMessageBox.warning(...)`. It runs during
`SettingsDialog.__init__`, so a corrupted `quickspeak.csv` — or any load error —
blocks the **entire** Settings dialog behind a modal that appears before the
dialog itself does. tooling-3/4 found this because test stubs tripped it; a
real user would trip it with bad data.

## Acceptance Criteria

### AC #1 — The load error is surfaced non-modally, inside the widget
**Given** `_load_entries` fails during construction
**Then** the Quick Speak tab shows the error in-place (an inline label, banner,
or disabled table with a message — match the dialog's existing idiom), the rest
of Settings is fully usable, and nothing blocks construction
**And** the error is still logged at the same level as today

### AC #2 — Explicit user actions may still be modal
**Given** the same widget has user-initiated actions (save, import, reset)
**Then** their existing `QMessageBox` behaviour is unchanged — this story
changes only the construction-time path. A modal in response to a click the
user just made is fine; a modal from inside `__init__` is not

### AC #3 — Tested, including the former hang shape
**Then** a test constructs `SettingsDialog` with a quick-speak service whose
load raises, asserts the dialog constructs, the tab shows the error, and no
`QMessageBox` is invoked — the tooling-4 raise-on-modal net will fail the test
if one is

### AC #4 — No regressions
**Then** the full suite's failure set is unchanged apart from anything this
story fixes, reported by id

## Dev Notes
Touch only `quick_speak_settings_widget.py` and its tests. The two stale
Clear-Comms-tab assertions in `test_settings_dialog_clear_comms_tab.py` belong
to tooling-5, not here — leave them.
