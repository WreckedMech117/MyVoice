# Story tooling-4: Fix the Two Suite-Hang Sites

Status: in-progress

<!-- Out-of-epic tooling story. Follow-up to tooling-3, which made the hangs countable and named them. -->
<!-- Risk: LOW. Test-side fixes plus one defensive fixture. The only way to get this wrong is to weaken a production dialog, which is forbidden below. -->

## Story

As **anyone running the regression suite**,
I want **a plain `pytest tests/` to complete in one invocation**,
so that **regression evidence is a single countable run again, not a deselect-and-rerun loop**.

## Context

tooling-3 installed `pytest-timeout` and found **27 hangs at exactly 2 stall sites**,
both the same defect class — a blocking `QMessageBox.warning()` on an error path a
test stub trips:

**Site A — 24 tests.** Tests construct `SettingsDialog` with
`quick_speak_service=MagicMock()`. A bare `MagicMock` returns another `MagicMock` for
every attribute, one of which reaches a Qt setter (`setCurrentText(MagicMock)` →
`TypeError`). `QuickSpeakSettingsWidget._load_entries` catches it and pops a modal
(`quick_speak_settings_widget.py:152`) **from inside dialog construction**, waiting for
a click nobody gives. Files: `test_reset_to_defaults.py` (1),
`test_close_to_tray_toggle.py` (8 — Story ui-1's), `test_settings_dialog_clear_comms_tab.py`
(10), `test_settings_dialog_streaming_tab.py` (5).

**Site B — 3 tests.** `test_voice_design_studio_dialog.py`'s regenerate tests call
`_on_regenerate_requested`, which routes to `_on_generate_requested`
(`voice_design_studio_dialog.py:455`), which guards on `_tts_service` and pops a modal
(`:471`) when it is `None` — which it is, because the tests never wire one.

Both are **test defects**: stubs that put the code under test into an error path it
was never meant to exercise. The production modals are doing their jobs.

## Acceptance Criteria

### AC #1 — Site A: give the settings-dialog tests a correct quick-speak stub

**Given** four test files each hand-roll a bare `MagicMock()` as the quick-speak service
**When** the fix lands
**Then** there is **one** shared fixture (in `tests/conftest.py` or a `tests/ui/conftest.py`)
that builds a quick-speak stub whose attributes return values of the **right type** for
everything `QuickSpeakSettingsWidget` reads during construction — no `MagicMock` reaches
a Qt setter
**And** all four files use it instead of their own stub, so the next test that constructs
`SettingsDialog` cannot re-spring the trap
**And** the stub is derived from what the widget actually calls — read
`QuickSpeakSettingsWidget.__init__` / `_load_entries` and enumerate the calls, rather
than guessing attribute names

### AC #2 — Site B: the regenerate tests get the dependency they exercise

**Given** the three regenerate tests want to assert that regenerate **routes into** the
generate flow, and the generate flow guards on `_tts_service`
**When** the fix lands
**Then** those tests provide a `tts_service` stub sufficient to pass the guard (and
`is_running()`), so the routing they exist to test is actually reached
**And** the guard itself — `_tts_service` falsy → warning — is left exactly as it is;
if it has no test of its own, add one that **patches `QMessageBox.warning`** and asserts
it was called, rather than letting a real modal show

### AC #3 — An unexpected modal under test must fail, never hang (defensive)

**Given** both stall sites are the same shape and there is nothing stopping a fifth file
from hand-rolling a bad stub next month
**When** the fix lands
**Then** an **autouse** conftest fixture makes any `QMessageBox.warning` /
`.critical` / `.information` / `.question` that is *not* explicitly patched by the test
**raise** immediately with a message naming the dialog title and text — so a future
regression is a clear failure in milliseconds, not a 60 s timeout followed by
`os._exit(1)`
**And** it must not break tests that legitimately exercise a modal: audit every existing
`QMessageBox` reference under `tests/` first, and report how each one interacts with the
fixture (patches it themselves → unaffected; relies on a real dialog → must be adapted
or the fixture opted out for that test)
**And** the existing close-confirm bypass (`_force_quit`, `memory/main_window_close_confirm_dialog_in_tests.md`)
is untouched — this fixture is a *net*, not a replacement for it
**And** if the audit finds the fixture cannot be made safe, say so and ship AC #1 + #2
without it rather than forcing it

### AC #4 — Do not weaken a production dialog

**Given** `memory/main_window_close_confirm_dialog_in_tests.md` records the rule
**When** any of the above is implemented
**Then** no production `QMessageBox` call is removed, made conditional on a test flag,
or made non-modal. The fixes are on the test side and in test infrastructure only
**And** the one production question this surfaces — a modal fired from inside
`SettingsDialog.__init__` on *any* exception is questionable UX regardless of tests — is
**recorded as a finding for a UI story**, not acted on here

### AC #5 — A single run completes, and the failure set is unchanged

**Given** tooling-3's baseline: 47 failed, 4 errors, 27 timeouts, 2,896 passed
**When** this story lands
**Then** a **single** `pytest tests/` invocation runs to completion with **zero
timeouts** and no `os._exit`
**And** the 47 pre-existing failures are unchanged in count **and identity** — they are
not this story's, and "fixing" any of them here would be scope drift
**And** the 27 former timeouts now **pass** (or fail on a genuine assertion, which is
then reported by id — a former hang that becomes a real failure has been *improved*,
not regressed)

## Tasks / Subtasks

- [ ] **Task 1 — Site A** (AC: #1) — enumerate the widget's construction-time calls, build the shared stub, migrate all four files.
- [ ] **Task 2 — Site B** (AC: #2) — wire a `tts_service` stub; add a guard test if none exists.
- [ ] **Task 3 — Defensive fixture** (AC: #3, #4) — audit existing `QMessageBox` uses under `tests/`, then implement or explicitly decline.
- [ ] **Task 4 — Single-run verification** (AC: #5) — one `pytest tests/`, to completion, failure set diffed against tooling-3's.
- [ ] **Task 5 — Evidence** — `_bmad-output/implementation-artifacts/tooling-4-fix-suite-hangs-evidence.md`.

## Dev Notes

### Why the fixture is a separate AC and may be declined

AC #1 and #2 fix what is broken. AC #3 stops it recurring. But an autouse fixture that
intercepts every modal is exactly the kind of global test-infrastructure change that
quietly breaks something unrelated. Hence the audit-first requirement and the explicit
permission to decline. A repo with the hangs fixed and no net is better than one with a
net that has holes nobody knows about.

### What this story is NOT

- Not a fix for the 47 pre-existing failures. Different defects, different stories.
- Not a change to any production `QMessageBox`. See AC #4.
- Not the `__pycache__`-on-`G:` cleanup tooling-3 noted. Trivial, but a separate commit
  so it does not muddy this one's diff.

## References

- `_bmad-output/implementation-artifacts/tooling-3-pytest-timeout-evidence.md` §5 — the 27 timeouts by id and stall location
- `src/myvoice/ui/components/quick_speak_settings_widget.py:145-156` (Site A), `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:455-476` (Site B)
- `tests/conftest.py:54-80` — the existing close-confirm bypass, as the precedent for *how* this repo handles modals under test
- `memory/main_window_close_confirm_dialog_in_tests.md`

## Dev Agent Record

_(to be filled by dev agent)_

## Change Log

- 2026-09-14 — Drafted by Winston immediately after tooling-3 merged. Both stall sites are test defects; the story fixes them at the root, adds a net against recurrence with an audit-first guard, and forbids touching the production dialogs.
