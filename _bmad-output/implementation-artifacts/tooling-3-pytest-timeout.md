# Story tooling-3: Make Suite Hangs Countable (pytest-timeout)

Status: in-progress

<!-- Out-of-epic tooling story, following the tooling-N precedent. -->
<!-- Source: Story 20.8 AC #4 — `pytest tests/` does not complete on main; an external per-directory guard was needed to produce a regression comparison at all. -->
<!-- Risk: LOW. Adds a dev dependency and a config block. The only way to get this wrong is to ship the dependency or to pick a timeout that turns slow-but-honest tests into false failures. -->

## Story

As **anyone running the regression suite on this repo**,
I want **a hung test to fail and be named instead of stalling the whole run forever**,
so that **regression evidence stays countable and the hangs themselves become fixable**.

## Context

Story 20.8's regression sweep found that a plain `pytest tests/` **does not complete
on current `main`** — CPU pins and the run never returns. The agent had to build an
external per-directory wall-clock guard (`20-8-suite-with-hang-guard.sh`) just to
produce a comparison, and it stated plainly that the result was therefore not the
clean "identical count and identity" every earlier story had.

Two hanging tests were identified reproducing **in isolation on a clean tree**, with
at least one more around the 90 % mark of the suite:

- `tests/settings/test_reset_to_defaults.py::TestResetQuickSpeak::test_reset_quick_speak_entries`
- `tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_control_exists_and_is_a_two_option_combo`

The second is a **Story ui-1 test** — merged in PR #8 on the strength of a per-directory
run that passed. It hangs under whole-suite ordering. That is exactly the class of
defect this story exists to make visible.

`pytest-timeout` is not in the portable interpreter. `requirements.txt` already carries
a `# dev-only` convention for `pytest-cov`, and `build_tools/requirements-production.txt`
excludes `pytest-*`, so there is an established home for it.

## Acceptance Criteria

### AC #1 — Hangs become failures

**Given** `pytest-timeout` installed in the portable `python310` interpreter
**When** `pytest tests/` runs
**Then** it **completes** — every test either passes, fails, errors, or times out; none
stalls the run
**And** a timed-out test is reported by name with a traceback of where it was stuck
**And** the timeout method is **`thread`**, not `signal` — the signal method does not
work on Windows and the repo is Windows-first (`memory/hardware_setup.md`)

### AC #2 — The timeout is chosen from evidence, not guessed

**Given** some legitimate tests are slow (Story 20.5's out-of-process qasync driver,
integration smoke tests with real model construction)
**When** the default timeout is set
**Then** it is derived from the measured distribution of passing-test durations in the
suite — report the slowest legitimately-passing tests and set the default above them
with margin, so no honest test becomes a false failure
**And** any test that genuinely needs longer is marked individually with
`@pytest.mark.timeout(N)` and a comment saying why, rather than the global value being
raised to accommodate it
**And** the chosen value and the evidence for it are recorded in the evidence file

### AC #3 — The dependency does not ship

**Given** the portable interpreter is also the one bundled into the public installer
**When** `pytest-timeout` is added
**Then** it goes into `requirements.txt` under the existing `# dev-only` convention
beside `pytest-cov`, and `build_tools/requirements-production.txt` continues to exclude
it
**And** `build_release.bat`'s probes are unaffected — verify by reading them, not by
building

### AC #4 — Name the hangs; do not fix them here

**Given** this story makes hangs countable
**When** the suite completes under the timeout
**Then** every test that times out is **listed by id** in the evidence file, with what
it was doing when it stalled (from the traceback), and which directory-level run it
passed in if it passed anywhere
**And** they are **not fixed in this story**. Each is a real defect with its own root
cause; this story's job is to make them visible and countable so the next story can
address them from evidence rather than from a hang
**And** the pre-existing failure set — the 49 documented across Stories 20.6–20.8 — is
reported as unchanged in count and identity, now with the timeouts as *additional* named
rows rather than an absent run

### AC #5 — The conftest DLL-ordering invariant survives

**Given** `tests/conftest.py` enforces torch-before-PyQt6 import ordering and
`pytest-cov` already needed a documented workaround for it
(`memory/torch_before_coverage_dll_ordering.md`)
**When** `pytest-timeout` is active
**Then** the conftest preamble still runs first and the DLL invariant holds — confirm
by running the streaming suites under the timeout and observing no `WinError 1114`
**And** if the timeout plugin disturbs the ordering, say so and record the workaround the
way the pytest-cov one was recorded

## Tasks / Subtasks

- [ ] **Task 1 — Install and configure** (AC: #1, #3, #5) — `pytest-timeout` into `python310`, config block with `timeout_method = thread`, `requirements.txt` dev-only entry, verify production requirements and build probes untouched.
- [ ] **Task 2 — Derive the timeout** (AC: #2) — measure passing-test durations, set the default with margin, mark individual slow tests.
- [ ] **Task 3 — Run the whole suite to completion** (AC: #1, #4) — list every timeout by id and stall location; confirm the pre-existing failure set is unchanged.
- [ ] **Task 4 — Evidence file** — `_bmad-output/implementation-artifacts/tooling-3-pytest-timeout-evidence.md`.

## Dev Notes

### The point is countability, not a green suite

This story will make the suite *look worse* — hangs become named failures. That is the
goal. A run that never finishes cannot be compared to anything; a run with three named
timeouts can be compared to the next run. Do not raise the timeout to make them
disappear.

### What this story is NOT

- Not a fix for the hanging tests. They get their own story, from evidence this one
  produces.
- Not a change to `tests/conftest.py`'s DLL ordering, unless AC #5 finds the plugin
  breaks it — in which case the fix is a recorded workaround, not a weakening.
- Not a CI change. There is no CI on this repo; this is for the local run.

## References

- `_bmad-output/implementation-artifacts/20-8-suite-with-hang-guard.sh` — the external guard and the hangs it named
- `_bmad-output/implementation-artifacts/20-8-chunk-size-reopen-evidence.md` §9.4.5 — the regression caveat this story removes
- `requirements.txt:84-96` — the existing `# dev-only` convention and the pytest-cov Windows note
- `memory/test_interpreter_portable_python310.md`, `memory/torch_before_coverage_dll_ordering.md`

## Dev Agent Record

_(to be filled by dev agent)_

## Change Log

- 2026-09-14 — Drafted by Winston at Commander's direction. Chosen over the remaining Epic 20 follow-ups because it costs no listening time and has been degrading every story's regression evidence since Story 20.6.
