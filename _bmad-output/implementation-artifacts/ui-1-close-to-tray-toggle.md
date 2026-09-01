# Story ui-1: Close-to-Tray Toggle (and the minimize button that ignores it)

Status: ready-for-dev

<!-- Out-of-epic UI story, following the `tooling-N` naming precedent for work that sits outside an epic. -->
<!-- Source: Commander, 2026-09-01 — "when I close from the X the app just minimizes to taskbar. The Settings/Interface menu does not have a toggle for that." -->
<!-- Risk: LOW. One setting, one checkbox, one conditional. The only trap is the second defect below, which is easy to fix and easy to miss. -->

## Story

As **a MyVoice user**,
I want **to choose whether closing the window quits the app or hides it to the tray**,
so that **the close button does what I expect instead of what a setting I cannot see decides**.

## Context

`AppSettings.minimize_to_tray` already exists (`app_settings.py:69`, Story 7.2 / FR38),
is persisted (`:599`, `:670`), and governs `MainWindow.closeEvent`
(`main_window.py:2354-2366`). What is missing is any way for a user to change it.

The Settings dialog already has the right home: an **Interface** tab
(`settings_dialog.py:677-724`, added as `_create_ui_tab`). The setting simply was
never surfaced there.

Commander hit this directly: the X minimizes instead of quitting, with no visible
way to change that, and the only workaround is closing from the taskbar — which
routes through the tray-quit path and leaves the process alive long enough to
stall a scripted measurement loop.

### A second, related defect worth fixing in the same pass

The **title-bar minimize button ignores the setting entirely.**
`custom_title_bar.py:172-186` reads:

```python
if hasattr(self._parent_window, '_minimize_to_tray'):
    self._parent_window._minimize_to_tray()
else:
    self._parent_window.showMinimized()
```

The guard is `hasattr`, which is **always true** on `MainWindow` — the method is
defined at `main_window.py:1115`. So the `showMinimized()` branch is dead code, and
the minimize button always hides to tray regardless of `minimize_to_tray`.

That is almost certainly why the behaviour reads as "there is no way to get a normal
window". A user who turns the new toggle off would still find minimize vanishing
their window to the tray, and would reasonably conclude the toggle does nothing.

**Fix both, or the toggle is not believable.**

## Acceptance Criteria

### AC #1 — The setting is visible and editable

**Given** the Settings dialog's existing **Interface** tab (`settings_dialog.py:677`)
**When** the user opens it
**Then** there is a labelled control for the close-button behavior, worded from the
user's point of view rather than after the internal field name — e.g. *"When closing
the window: minimize to system tray / quit MyVoice"*, with a short line of helper text
naming the tray as the way back
**And** it reflects the current persisted `minimize_to_tray` value on open
**And** changing it persists through the dialog's existing save path, with no new
persistence mechanism introduced
**And** the change takes effect without an app restart

### AC #2 — The close button honours the setting

**Given** the toggle is set to **quit**
**When** the user clicks the window's X
**Then** the app quits — it does not hide to the tray
**And** the existing confirm-close dialog behavior is unchanged; this story changes
*whether we minimize*, never *whether we confirm*
**Given** the toggle is set to **minimize to tray**
**When** the user clicks X
**Then** the current behavior is preserved exactly, including the first-use tray
notification at `main_window.py:1115`

### AC #3 — The minimize button honours the setting too (the second defect)

**Given** `custom_title_bar.py:172-186` guards on `hasattr(parent, '_minimize_to_tray')`,
which is always true, so `showMinimized()` is unreachable
**When** the user clicks the title-bar minimize button with the toggle set to **quit**
**Then** the window minimizes to the **taskbar**, not the tray
**And** with the toggle set to **minimize to tray**, it continues to hide to the tray
**And** the dead `hasattr` guard is replaced by an actual read of the setting, so the
`showMinimized()` path is reachable
**And** a regression test covers both branches — the current test suite cannot
distinguish them, because one of them never executes

### AC #4 — Measurement mode is untouched

**Given** `MYVOICE_AUTO_QUIT_ON_CLOSE=1` sets `_force_quit` and bypasses both
tray-minimize and the confirm dialog (`main_window.py:2312-2326`)
**When** that env var is set
**Then** its behavior is identical before and after this story, whatever the toggle says
**And** the measurement launchers that depend on it keep working — they are how Epic 20
gets its numbers

### AC #5 — No regressions

**Given** the existing settings-dialog and main-window suites
**When** the change lands
**Then** they pass with zero new failures
**And** the tree's known pre-existing failures — documented in
`20-2-warm-path-compile-priming-evidence.md` — are unchanged in count and identity

## Tasks / Subtasks

- [ ] **Task 1 — Surface the setting** (AC: #1)
  - [ ] 1.1 Add the control to `_create_ui_tab` (`settings_dialog.py:677-724`), matching the tab's existing widget idiom rather than inventing a new one.
  - [ ] 1.2 Load from and save to `minimize_to_tray` through the dialog's existing path.
  - [ ] 1.3 Confirm the live `app_settings` object the window reads is the one updated, so no restart is needed.

- [ ] **Task 2 — Fix the minimize button** (AC: #3)
  - [ ] 2.1 Replace the `hasattr` guard with a read of `minimize_to_tray`.
  - [ ] 2.2 Handle the case where the parent has no such setting wired — fall back to `showMinimized()`, the safer of the two.
  - [ ] 2.3 Tests for both branches.

- [ ] **Task 3 — Verify close behavior** (AC: #2, #4)
  - [ ] 3.1 Tests for closeEvent under both toggle states.
  - [ ] 3.2 Test that `MYVOICE_AUTO_QUIT_ON_CLOSE=1` still forces a real quit regardless of the toggle.

- [ ] **Task 4 — Regression sweep** (AC: #5)

## Dev Notes

### Do not weaken the confirm dialog

`memory/main_window_close_confirm_dialog_in_tests.md` records that `closeEvent`'s
`QMessageBox` blocks pytest, and that the sanctioned bypass is the production
`_force_quit` flag — never softening the dialog itself. This story keeps that intact:
it changes the minimize-vs-quit decision at `main_window.py:2354`, which sits *before*
the confirm dialog at `:2369`, and touches neither the dialog nor `_force_quit`.

### Default value

`AppSettings.minimize_to_tray` defaults to `False` (`app_settings.py:69`), yet
Commander's `config/settings.json` has it `True` — so something set it, or it predates
the current default. **Do not change the default in this story.** Surfacing the control
lets any user fix their own state, which is the actual complaint; changing a persisted
default would silently alter behavior for existing users who are happy with it.

### What this story is NOT

- Not a tray-icon rework. The tray menu, its Exit item, and the first-use notification
  all stay as they are.
- Not a change to `_force_quit` or the confirm dialog.
- Not related to Epic 20. It surfaced during Story 20.3's AC #4 measurement runs, but
  the fix is independent and should not wait on that epic.

## References

- `src/myvoice/models/app_settings.py:69` — the setting, defaulting to `False`
- `src/myvoice/ui/components/settings_dialog.py:677-724` — the Interface tab that should host it
- `src/myvoice/ui/main_window.py:2354-2366` — closeEvent's minimize decision; `:2312-2326` — the measurement-mode bypass; `:1115` — `_minimize_to_tray`
- `src/myvoice/ui/components/custom_title_bar.py:172-186` — the always-true `hasattr` guard
- `memory/main_window_close_confirm_dialog_in_tests.md`

## Dev Agent Record

### Agent Model Used

_(to be filled by dev agent)_

### Completion Notes List

_(to be filled by dev agent)_

### File List

_(to be filled by dev agent)_

## Change Log

- 2026-09-01 — Drafted by Winston from Commander's report that the X minimizes with no Interface toggle to change it. Scope widened by one defect found while grounding the story: the title-bar minimize button guards on `hasattr`, which is always true, so it ignores the setting and its `showMinimized()` branch is unreachable. Shipping the toggle without that fix would produce a control that appears not to work.
