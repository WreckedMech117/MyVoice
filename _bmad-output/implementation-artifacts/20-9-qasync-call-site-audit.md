# Story 20.9: qasync Call-Site Audit (Follow-up F7)

Status: in-progress

<!-- Epic 20 follow-up F7. Source: Story 20.3's residual risk. -->
<!-- Risk: MEDIUM. Touches scheduling in app.py. The one thing that must not happen is a behaviour change to a path that currently works. -->

## Story

As **the maintainer of a Qt + asyncio app**,
I want **every fire-and-forget task scheduled in a way that cannot be destroyed mid-step**,
so that **the defect that silently killed compile priming for months cannot recur on another call site**.

## Context

Story 20.3 found that qasync's `call_soon` routes through `QObject.startTimer(0)`,
so a scheduled task's steps are Qt zero-timers delivered during ANY Qt event
processing — including a `processEvents()` inside another task. When that
happens the scheduled task dies with `RuntimeError: Cannot enter into task ...
while another task ... is being executed` and `Task was destroyed but it is
pending!`. Story 20.3 fixed ONLY the startup call site with
`_run_async_task_when_loop_is_idle`, which creates the task only on a loop pass
where `asyncio.current_task() is None`. It stated the hazard is general and that
the wider audit was undone. `memory/qasync_task_destruction_hazard.md` records
this.

Sizing: 37 `_run_async_task(` calls in `app.py`; 71 scheduling sites
(`_run_async_task` / `ensure_future` / `create_task`) across `src/myvoice`.

## Acceptance Criteria

### AC #1 — Every scheduling site is classified, with evidence
**Then** the evidence file lists all 71 sites, each classified as: (a) called
from a Qt signal handler with no task on the stack — safe by construction;
(b) called from inside a running task, or from a path that pumps Qt events
before the scheduled task's first step — EXPOSED; or (c) uncertain, with what
would resolve it
**And** the classification is by reading the call path, not by grep alone —
state the stack for each (b)

### AC #2 — Exposed sites are made safe without changing behaviour
**Given** a (b) site
**Then** it is moved to the idle-scheduling helper or an equivalent, and the
change is proven behaviour-preserving: the task still runs, runs once, and runs
in the same order relative to its neighbours
**And** a site that is exposed only in theory but whose task is short enough
to complete before any event pumping may be left, if that argument is written
down per site

### AC #3 — The trap is caught in tests, not only in the log
**Then** the out-of-process qasync driver from Story 20.3
(`tests/unit/_qasync_warmup_driver.py`) is generalised or duplicated so at
least one non-startup exposed site is exercised under a real qasync loop and
proven to survive — the same class of test that would have caught the original

### AC #4 — No regressions
**Then** the full suite's failure set is unchanged in identity; `src/` changes
are confined to scheduling sites

## Dev Notes
Do not touch `warmup_compile_async` or its call site — Story 20.3 fixed and
verified it on hardware. Do not touch the audio dispatch chain. If a (b) site's
fix would change ordering that something depends on, report it rather than
change it.
