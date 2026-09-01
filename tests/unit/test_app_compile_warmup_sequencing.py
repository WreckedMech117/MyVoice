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
    # Union, not a fallback: if someone re-adds a direct
    # ``warmup_compile_async()`` schedule above the preload while the wrapper
    # stays below it, the defect is back and this row must still catch it.
    warmups = _call_linenos(
        initialize_services_fn, "_warmup_compile_after_preload"
    ) + _call_linenos(initialize_services_fn, "warmup_compile_async")

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
# 2. _warmup_compile_after_preload behaviour
# --------------------------------------------------------------------------- #


def _bare_app(tts_service, hydration_task):
    """A ``MyVoiceApp`` shell carrying only what the coroutine reads.

    ``__new__`` avoids the full constructor (QApplication, services, Qt
    signals) — the coroutine under test touches three attributes.
    """
    import logging

    from myvoice.app import MyVoiceApp

    app = MyVoiceApp.__new__(MyVoiceApp)
    app.logger = logging.getLogger("test-myvoice-app")
    app._tts_service = tts_service
    app._voice_clone_prompt_hydration_task = hydration_task
    return app


class _FakeTTSService:
    def __init__(self) -> None:
        self.warmup_calls = 0
        self.hydration_done_at_warmup: List[bool] = []

    def make_warmup(self, hydration_flag: List[bool]):
        async def warmup_compile_async():
            self.warmup_calls += 1
            self.hydration_done_at_warmup.append(bool(hydration_flag))

        return warmup_compile_async


def test_warmup_waits_for_prompt_hydration_before_priming():
    """AC #1 — priming starts only after hydration has finished.

    Racing hydration would make the BASE path skip with ``no_priming_prompt``
    on the common cloned-voice launch, which is the launch the whole story is
    about.
    """
    pytest.importorskip("PyQt6")

    async def scenario():
        hydrated: List[bool] = []
        svc = _FakeTTSService()

        async def slow_hydration():
            await asyncio.sleep(0.05)
            hydrated.append(True)

        task = asyncio.ensure_future(slow_hydration())
        svc.warmup_compile_async = svc.make_warmup(hydrated)
        app = _bare_app(svc, task)

        await app._warmup_compile_after_preload()
        return svc

    svc = asyncio.run(scenario())

    assert svc.warmup_calls == 1
    assert svc.hydration_done_at_warmup == [True], (
        "the compile warmup ran before voice_clone_prompt hydration finished"
    )


def test_warmup_runs_even_when_hydration_times_out(monkeypatch):
    """AC #1 — a stuck hydration must not make priming unreachable.

    The wait is bounded and ``shield``-ed: the timeout abandons the wait, not
    the hydration task, and the warmup runs anyway (a BASE prime with no
    cached prompt skips itself — it never switches models).
    """
    pytest.importorskip("PyQt6")

    from myvoice.app import MyVoiceApp

    monkeypatch.setattr(MyVoiceApp, "_HYDRATION_WAIT_TIMEOUT_S", 0.05)

    async def scenario():
        svc = _FakeTTSService()
        svc.warmup_compile_async = svc.make_warmup([])

        async def never_finishes():
            await asyncio.sleep(30)

        task = asyncio.ensure_future(never_finishes())
        app = _bare_app(svc, task)
        try:
            await app._warmup_compile_after_preload()
        finally:
            task.cancel()
        return svc, task

    svc, task = asyncio.run(scenario())

    assert svc.warmup_calls == 1


def test_warmup_runs_when_hydration_raised():
    """AC #1 — a failed hydration is not a reason to skip priming."""
    pytest.importorskip("PyQt6")

    async def scenario():
        svc = _FakeTTSService()
        svc.warmup_compile_async = svc.make_warmup([])

        async def boom():
            raise RuntimeError("disk on fire")

        task = asyncio.ensure_future(boom())
        await asyncio.gather(task, return_exceptions=True)
        app = _bare_app(svc, task)
        await app._warmup_compile_after_preload()
        return svc

    svc = asyncio.run(scenario())
    assert svc.warmup_calls == 1


def test_warmup_runs_when_there_is_no_hydration_task():
    """AC #1 — hydration wiring can fail (no VoiceProfileManager); the warmup
    still has to run, because CUSTOM_VOICE and VOICE_DESIGN residents need no
    prompt at all."""
    pytest.importorskip("PyQt6")

    async def scenario():
        svc = _FakeTTSService()
        svc.warmup_compile_async = svc.make_warmup([])
        app = _bare_app(svc, None)
        await app._warmup_compile_after_preload()
        return svc

    svc = asyncio.run(scenario())
    assert svc.warmup_calls == 1


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
