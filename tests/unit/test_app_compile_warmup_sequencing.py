"""Story 20.3 AC #1 — the compile warmup must run AFTER the model preload.

**The exact defect class this file guards.** ``_initialize_services_async`` used to
schedule ``warmup_compile_async`` at ``app.py:594`` — *above* the
``await preload_model(...)`` at ``:613``. ``_run_async_task`` only schedules
(``asyncio.ensure_future``), so the warmup coroutine first executed at the
enclosing coroutine's next suspension point, which was that very
``await preload_model``. At that instant nothing was loaded, every statement
in ``warmup_compile_async`` between entry and its ``get_loaded_model()`` check
is synchronous, and so the worker ran straight through to
``reason="no_model_loaded"`` — deterministically, on every launch. Story
18.4's cold priming and Story 20.2's warm priming had therefore *never run in
the shipped application*.

That defect is a **call-site ordering** bug, so it is guarded here at the call
site (per ``memory/code_review_regression_test_exact_class.md``: the
regression test must mirror the exact bug class). Two complementary rows:

  1. A source/AST invariant over ``_initialize_services_async`` — the warmup
     scheduling must not appear before the preload await. This is the row that
     goes red if someone moves the call back up.
  2. Behavioural coverage of ``_warmup_compile_after_preload`` — it waits for
     Story 17.2's hydration task before handing off, and it hands off anyway
     when hydration is slow, cancelled, or broken (never leaving priming
     unreached, and never blocking startup on it).
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from typing import List

import pytest


APP_PY_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "myvoice" / "app.py"
)


@pytest.fixture(scope="module")
def initialize_services_fn() -> ast.AsyncFunctionDef:
    tree = ast.parse(APP_PY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == "_initialize_services_async"
        ):
            return node
    pytest.fail("app.py no longer defines _initialize_services_async")


def _call_linenos(fn: ast.AST, attr_name: str) -> List[int]:
    """Line numbers of every ``<something>.<attr_name>(...)`` call inside fn."""
    out: List[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == attr_name:
            out.append(node.lineno)
    return out


def _attr_linenos(fn: ast.AST, attr_name: str) -> List[int]:
    """Line numbers of every ``<something>.<attr_name>`` reference inside fn.

    The warmup is now handed over as a *callable*
    (``_run_async_task_when_loop_is_idle(self._tts_service.warmup_compile_async)``)
    rather than invoked, so an ast.Call scan alone would silently find nothing
    and the ordering invariant would stop protecting anything.
    """
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and node.attr == attr_name
    ]


# --------------------------------------------------------------------------- #
# 1. The ordering invariant at the call site
# --------------------------------------------------------------------------- #


def test_compile_warmup_is_scheduled_after_the_model_preload(
    initialize_services_fn,
):
    """AC #1 — the warmup must be scheduled below every ``preload_model``.

    Scheduling it above means it first runs *at* the preload's ``await``, with
    no model loaded, and exits at ``no_model_loaded``. That is not a subtle
    race: it is deterministic on every launch.
    """
    preloads = _call_linenos(initialize_services_fn, "preload_model")
    # Union of every way the warmup can be referenced here, not a fallback
    # chain: if someone re-adds a schedule above the preload while the current
    # hand-off stays below it, the defect is back and this row must catch it.
    warmups = (
        _attr_linenos(initialize_services_fn, "warmup_compile_async")
        + _call_linenos(initialize_services_fn, "_run_async_task_when_loop_is_idle")
    )

    assert preloads, "app.py no longer preloads a model in _initialize_services_async"
    assert warmups, (
        "_initialize_services_async no longer schedules the compile warmup at all; "
        "Story 18.4's cold priming and Story 20.2's warm priming are both "
        "unreachable without it"
    )
    assert min(warmups) > max(preloads), (
        f"the compile warmup is scheduled at line(s) {warmups}, at or above "
        f"the model preload at line(s) {preloads}. _run_async_task only "
        "SCHEDULES the coroutine, so it would first execute at the preload's "
        "own await — with no model loaded — and exit at "
        'reason="no_model_loaded" on every launch (Story 20.2 §6).'
    )


def test_warmup_is_not_awaited_inline_on_the_startup_path(
    initialize_services_fn,
):
    """AC #1 — the reordering must not block the Qt main thread.

    Fixing the ordering by simply ``await``-ing the warmup would park startup
    (and, on the qasync loop, the UI construction that follows) behind a
    multi-second priming generation. The hand-off has to stay
    fire-and-forget.
    """
    offenders = []
    for node in ast.walk(initialize_services_fn):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(
            call.func, ast.Attribute
        ):
            continue
        if call.func.attr in {
            "warmup_compile_async",
            "_warmup_compile_after_preload",
        }:
            offenders.append(node.lineno)

    assert offenders == [], (
        "the compile warmup is awaited inline in _initialize_services_async at "
        f"line(s) {offenders}; startup would block for the whole priming "
        "generation. Schedule it via _run_async_task instead."
    )


def test_warmup_hand_off_uses_the_qasync_safe_scheduler(initialize_services_fn):
    """AC #1 — the call site must use ``_run_async_task_when_loop_is_idle``.

    **This row exists because a mutation escaped without it.** Reverting the
    call site to the plain ``_run_async_task`` left
    ``test_app_compile_warmup_qasync.py`` green, because those rows drive the
    scheduler helper directly and never read app.py's actual call site. The
    qasync file proves the *mechanism* works; this row proves the shipped code
    actually uses it.

    Scheduling the warmup through the plain ``_run_async_task`` here is the
    exact shape that shipped broken: under qasync the task is created and then
    destroyed by the re-entrancy guard during ``main.py``'s synchronous
    splash/``processEvents()`` stretch, having never run a line of its body.
    """
    idle_calls = _call_linenos(
        initialize_services_fn, "_run_async_task_when_loop_is_idle"
    )
    assert idle_calls, (
        "the compile warmup is no longer handed off through "
        "_run_async_task_when_loop_is_idle. Under qasync every call_soon is a "
        "Qt zero-timer, so a plainly-scheduled task is stepped from inside "
        "main.py's processEvents() while Task-1 is mid-step, and "
        "asyncio._enter_task destroys it. See "
        "tests/unit/test_app_compile_warmup_qasync.py."
    )

    # ...and the warmup must not ALSO be handed to the plain scheduler.
    offenders = []
    for node in ast.walk(initialize_services_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_run_async_task"):
            continue
        inner = [
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
        ]
        if "warmup_compile_async" in inner:
            offenders.append(node.lineno)

    assert offenders == [], (
        "the compile warmup is scheduled through the plain _run_async_task at "
        f"line(s) {offenders}; that is the shape that shipped broken under "
        "qasync"
    )


def test_hydration_task_handle_is_retained(initialize_services_fn):
    """AC #1 — priming BASE needs the hydrated prompt, so the warmup wrapper
    has to be able to wait for hydration. That requires keeping the task
    handle rather than dropping it on the floor."""
    src = ast.get_source_segment(
        APP_PY_PATH.read_text(encoding="utf-8"), initialize_services_fn
    )
    assert "_voice_clone_prompt_hydration_task = self._run_async_task(" in src, (
        "the hydrate_voice_clone_prompt_cache task handle is no longer "
        "retained; _warmup_compile_after_preload cannot wait for hydration, "
        "so a BASE prime would skip with no_priming_prompt on exactly the "
        "cloned-voice launches Story 20.3 exists to speed up"
    )


# --------------------------------------------------------------------------- #
# 2. The startup hand-off: _run_async_task_when_loop_is_idle + entrypoint
# --------------------------------------------------------------------------- #
#
# NOTE ON WHAT THESE ROWS CAN AND CANNOT PROVE. They run on a plain asyncio
# loop, where ``call_soon`` appends to a ready queue drained only between task
# steps — so the qasync re-entrancy hazard that actually broke this feature is
# structurally ABSENT here. These rows cover the logic (deferral condition,
# hydration check, hand-off). The hazard itself is covered in
# ``tests/unit/test_app_compile_warmup_qasync.py``, which stands up a real
# qasync loop. Do not add a "warmup runs" row here and believe it protects the
# shipped path; that is exactly the mistake that let the first fix ship broken.


def _bare_app(tts_service=None, hydration_task=None):
    """A ``MyVoiceApp`` shell carrying only what the hand-off reads.

    ``__new__`` avoids the full constructor (QApplication, services, Qt
    signals) — the methods under test touch three attributes.
    """
    import logging

    from myvoice.app import MyVoiceApp

    app = MyVoiceApp.__new__(MyVoiceApp)
    app.logger = logging.getLogger("test-myvoice-app")
    app._tts_service = tts_service
    app._voice_clone_prompt_hydration_task = hydration_task
    return app


def test_entrypoint_runs_the_warmup_when_hydration_is_done():
    pytest.importorskip("PyQt6")

    async def scenario():
        ran: List[str] = []

        async def hydrate():
            ran.append("hydrated")

        async def warmup():
            ran.append("warmup")

        task = asyncio.ensure_future(hydrate())
        await task
        app = _bare_app(hydration_task=task)
        await app._compile_warmup_entrypoint(warmup)
        return ran

    assert asyncio.run(scenario()) == ["hydrated", "warmup"]


def test_entrypoint_proceeds_without_awaiting_an_unfinished_hydration():
    """AC #1 — the entrypoint must NOT await the hydration handle.

    Awaiting another task from inside this one is what the first fix did (via
    ``wait_for``/``shield``); the extra task machinery is needless exposure to
    the qasync hazard. If hydration has somehow not finished, the BASE priming
    path skips itself with ``no_priming_prompt`` — the designed safe fallback —
    so proceeding is correct and hanging would not be.
    """
    pytest.importorskip("PyQt6")

    async def scenario():
        ran: List[str] = []

        async def never_finishes():
            await asyncio.sleep(30)

        async def warmup():
            ran.append("warmup")

        task = asyncio.ensure_future(never_finishes())
        app = _bare_app(hydration_task=task)
        try:
            await asyncio.wait_for(app._compile_warmup_entrypoint(warmup), timeout=2.0)
        finally:
            task.cancel()
        return ran

    assert asyncio.run(scenario()) == ["warmup"], (
        "the entrypoint blocked on an unfinished hydration task"
    )


def test_entrypoint_warns_when_hydration_has_not_finished(caplog):
    """Observability — an unfinished hydration must be SAID, not swallowed.

    Commit 6428601 promoted the warmup's two silent gates to INFO precisely
    because a startup path this epic depends on must not be able to exit
    without saying so; the negative AC #4 passes were undiagnosable until it
    did. The same rule applies to the one remaining condition that can silently
    degrade priming: hydration not finished means a BASE resident primes
    nothing (it skips with ``no_priming_prompt``), and the launch looks
    identical to a healthy one unless this warning exists.
    """
    pytest.importorskip("PyQt6")

    async def scenario():
        ran: List[str] = []

        async def never_finishes():
            await asyncio.sleep(30)

        async def warmup():
            ran.append("warmup")

        task = asyncio.ensure_future(never_finishes())
        app = _bare_app(hydration_task=task)
        try:
            with caplog.at_level(logging.WARNING, logger="test-myvoice-app"):
                await app._compile_warmup_entrypoint(warmup)
        finally:
            task.cancel()
        return ran

    ran = asyncio.run(scenario())

    assert ran == ["warmup"]
    assert any(
        "hydration has not finished" in r.message
        and r.levelno >= logging.WARNING
        for r in caplog.records
    ), (
        "the compile warmup ran with an unfinished voice_clone_prompt "
        "hydration and said nothing; a BASE prime will skip with "
        "no_priming_prompt and the launch will look healthy. Records: "
        f"{[r.message for r in caplog.records]}"
    )


def test_entrypoint_runs_when_there_is_no_hydration_task():
    """Hydration wiring can fail (no VoiceProfileManager); the warmup still has
    to run, because CUSTOM_VOICE and VOICE_DESIGN residents need no prompt."""
    pytest.importorskip("PyQt6")

    async def scenario():
        ran: List[str] = []

        async def warmup():
            ran.append("warmup")

        app = _bare_app(hydration_task=None)
        await app._compile_warmup_entrypoint(warmup)
        return ran

    assert asyncio.run(scenario()) == ["warmup"]


def test_hand_off_defers_while_another_task_is_mid_step(monkeypatch):
    """AC #1 — the deferral condition is ``asyncio.current_task() is not None``.

    That is the exact slot ``asyncio._enter_task`` checks before stepping a
    task, so this is testing the precondition rather than a proxy for it. On a
    plain loop nothing is ever mid-step during a ``call_soon`` callback, so
    ``current_task`` is faked to return a task for the first two passes.
    """
    pytest.importorskip("PyQt6")

    async def scenario():
        ran: List[str] = []
        pretend_busy = {"n": 2}
        real_current_task = asyncio.current_task

        def fake_current_task(*a, **k):
            if pretend_busy["n"] > 0:
                pretend_busy["n"] -= 1
                return "pretend-task-1"
            return real_current_task(*a, **k)

        monkeypatch.setattr(asyncio, "current_task", fake_current_task)

        async def warmup():
            ran.append("warmup")

        app = _bare_app()
        app._run_async_task_when_loop_is_idle(warmup)

        for _ in range(10):
            await asyncio.sleep(0)
        return ran, pretend_busy["n"]

    ran, remaining = asyncio.run(scenario())
    assert remaining == 0, "the hand-off did not re-arm while a task was mid-step"
    assert ran == ["warmup"], (
        "the hand-off never scheduled the warmup after the loop went idle"
    )


def test_hand_off_gives_up_deferring_rather_than_never_scheduling(monkeypatch):
    """The re-arm is bounded. A loop that never goes idle must still get a
    scheduled warmup plus a WARNING, not silence — a startup path this epic
    depends on should not be able to exit without saying so."""
    pytest.importorskip("PyQt6")

    from myvoice.app import MyVoiceApp

    monkeypatch.setattr(MyVoiceApp, "_MAX_IDLE_DEFERRALS", 3)

    async def scenario():
        ran: List[str] = []
        monkeypatch.setattr(asyncio, "current_task", lambda *a, **k: "always-busy")

        async def warmup():
            ran.append("warmup")

        app = _bare_app()
        app._run_async_task_when_loop_is_idle(warmup)
        for _ in range(20):
            await asyncio.sleep(0)
        return ran

    assert asyncio.run(scenario()) == ["warmup"]


def test_run_async_task_returns_the_scheduled_future():
    """AC #1 — the sequencing depends on ``_run_async_task`` handing back a
    handle. Returning None again would silently reintroduce the race."""
    pytest.importorskip("PyQt6")

    async def scenario():
        import logging

        from myvoice.app import MyVoiceApp

        app = MyVoiceApp.__new__(MyVoiceApp)
        app.logger = logging.getLogger("test-myvoice-app")

        seen: List[str] = []

        async def work():
            seen.append("ran")
            return 7

        fut = app._run_async_task(work())
        assert fut is not None, "_run_async_task no longer returns its future"
        await fut
        return seen

    assert asyncio.run(scenario()) == ["ran"]
