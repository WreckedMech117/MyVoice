"""Story 20.9 (F7) — the qasync call-site audit's regression tests.

Story 20.3 fixed one scheduling site (the compile warmup) against qasync's
task-destruction hazard and left the other 70 unaudited. The audit
(``_bmad-output/implementation-artifacts/20-9-qasync-call-site-audit-evidence.md``)
found that the only task that pumps Qt events on a live path is ``main.py``'s
Task-1 during the post-init splash/UI stretch, so the exposed sites are the
ones that schedule from inside Task-1 with a task that can still be pending
when that stretch runs. Two were moved to the idle-pass scheduler:

  * ``_on_voice_service_started`` — voice restoration, previously a
    ``QTimer.singleShot(500)`` + ``loop.create_task`` guess at "init is done";
  * ``_setup_api_server_from_settings`` — the local TTS API start, whose
    uvicorn ``serve()`` task is long-lived and wakes every 100 ms.

The mechanism (``_schedule_when_loop_is_idle``) is the generic form of Story
20.3's ``_run_async_task_when_loop_is_idle``; the warmup wrapper keeps its
signature, body and log lines, and its call site is untouched.

Rows, in three groups:

**Real qasync loop (out of process, ``_qasync_call_site_driver.py``)** — the
same class of test that would have caught Story 20.3's original defect. Each
fixed row has a control row that reproduces the pre-20.9 shape and must be
destroyed; if a control ever goes green the rig has stopped reproducing the
hazard and the fixed row proves nothing.

**Call-site pins (AST)** — a mutation that reverts a call site to the plain
scheduler would escape the driver rows, which drive the app methods directly
but cannot see whether ``app.py`` still routes through them. Story 20.3
learned this the hard way (``test_warmup_hand_off_uses_the_qasync_safe_scheduler``).

**Behaviour-preservation of the refactor (plain loop)** — the warmup wrapper's
three log lines are byte-identical to Story 20.3's (the AC #4 hardware
evidence quotes them), and the API start's run-time re-check of the toggle.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("qasync")


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().parent / "_qasync_call_site_driver.py"


def _run_driver(variant: str) -> Dict[str, Any]:
    assert DRIVER.exists(), f"driver missing at {DRIVER}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), variant],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    marker = "__RESULT__"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    pytest.fail(
        f"qasync driver ({variant}) produced no result.\n"
        f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-3000:]}\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )


@pytest.fixture(scope="module")
def restore_legacy() -> Dict[str, Any]:
    return _run_driver("restore_legacy_fastfail")


@pytest.fixture(scope="module")
def restore_fixed() -> Dict[str, Any]:
    return _run_driver("restore_fixed_fastfail")


@pytest.fixture(scope="module")
def startup_order() -> Dict[str, Any]:
    return _run_driver("startup_order_fixed")


@pytest.fixture(scope="module")
def api_legacy() -> Dict[str, Any]:
    return _run_driver("api_legacy")


@pytest.fixture(scope="module")
def api_fixed() -> Dict[str, Any]:
    return _run_driver("api_fixed")


# =========================================================================== #
# Group 1 — real qasync loop
# =========================================================================== #


def test_legacy_restore_hand_off_is_destroyed_under_a_real_qasync_loop(restore_legacy):
    """Non-vacuity control for the voice-restoration site.

    The pre-20.9 shape — ``QTimer.singleShot(500, ...)`` creating the task —
    on a launch whose preload returns without suspending: the timer fires
    inside Task-1's post-init pump, the QTimer slot survives (plain Qt
    callback), but the task it creates is stepped in the same pump and
    destroyed. The restore never runs.
    """
    assert restore_legacy["events"] == [], (
        "the legacy restore ran, so this rig no longer reproduces the hazard "
        f"and the fixed row is vacuous: {restore_legacy['events']}"
    )
    assert any(
        "delayed_restore" in e for e in restore_legacy["reentrancy_errors"]
    ), f"expected the re-entrancy RuntimeError on the restore task; got {restore_legacy['all_errors']}"
    assert restore_legacy["destroyed_pending"], "expected 'Task was destroyed but it is pending!'"


def test_shipped_restore_hand_off_survives_the_same_pump(restore_fixed):
    """AC #2/#3 — the shipped ``_on_voice_service_started`` under the same
    rig: deferred through every pump pass, created once Task-1 parks, runs to
    completion exactly once, zero re-entrancy errors.
    """
    assert restore_fixed["reentrancy_errors"] == [], restore_fixed["all_errors"]
    assert restore_fixed["destroyed_pending"] == []
    starts = [e for e in restore_fixed["events"] if e.startswith("restore_start@")]
    dones = [e for e in restore_fixed["events"] if e.startswith("restore_done@")]
    assert len(starts) == 1 and len(dones) == 1, (
        f"the restore must run exactly once; events={restore_fixed['events']}"
    )
    assert starts[0] == "restore_start@parked", (
        "the restore's first step must land after Task-1 parked, not inside "
        f"the pump; events={restore_fixed['events']}"
    )
    assert restore_fixed["deferrals"] and restore_fixed["deferrals"][0] > 0, (
        "the hand-off never deferred, so the pump was not reproduced: "
        f"{restore_fixed['deferrals']}"
    )


def test_startup_sites_run_once_and_in_production_order(startup_order):
    """AC #2 — behaviour preservation of the restore move.

    Under a normal launch shape (the preload suspends Task-1) the three
    startup hand-offs must land in the order the shipped app logs today:
    hydration (single synchronous step) -> restore -> [preload returns] ->
    compile warmup. Each exactly once, and the warmup still after the pump.
    """
    assert startup_order["reentrancy_errors"] == [], startup_order["all_errors"]
    assert startup_order["events"] == [
        "hydration@preload",
        "restore_start@preload",
        "restore_done@preload",
        "warmup@parked",
    ], startup_order["events"]
    # Two idle hand-offs: the restore (created on the first idle pass, no
    # deferral because Task-1 was already parked in the preload) and the
    # warmup (deferred through the PUMP_ITERATIONS passes, as in 20.3).
    assert len(startup_order["deferrals"]) == 2, startup_order["deferrals"]
    assert startup_order["deferrals"][1] > 0


def test_legacy_inline_api_start_leaves_a_zombie_serve_task(api_legacy):
    """Non-vacuity control for the local TTS API site, against REAL uvicorn.

    ``ApiServerController.start()`` awaited inline inside Task-1, then the
    post-init pump: the serve task's 100 ms wake-up lands in the pump and it
    is destroyed mid-life. The failure is silent in the shipped app because
    the controller holds a strong reference — ``is_running`` keeps reporting
    True, ``on_tick`` is never called again, and ``stop()`` hangs.
    """
    assert api_legacy["running_before_pump"] is True
    assert any("Server.serve" in e for e in api_legacy["reentrancy_errors"]), (
        f"expected the serve task to hit the re-entrancy guard; got {api_legacy['all_errors']}"
    )
    assert api_legacy["running_after_pump"] is True, "is_running lies about a zombie"
    assert api_legacy["ticks_while_parked"] == 0, (
        "the serve loop kept ticking, so this rig no longer reproduces the hazard"
    )
    assert api_legacy["stop_completed"] is False, "stop() completed on a zombie task"


def test_shipped_deferred_api_start_survives_and_stops_cleanly(api_fixed):
    """AC #2/#3 — the shipped ``_setup_api_server_from_settings`` under the
    same rig: the start is deferred past the pump, the server comes up once,
    its main loop keeps ticking, and ``stop()`` completes.
    """
    assert api_fixed["reentrancy_errors"] == [], api_fixed["all_errors"]
    assert api_fixed["destroyed_pending"] == []
    assert api_fixed["running_before_pump"] is False, (
        "the start was not deferred: the server was already running when "
        "Task-1 reached the pump"
    )
    assert api_fixed["running_after_pump"] is True
    assert api_fixed["ticks_while_parked"] >= 2, api_fixed
    assert api_fixed["stop_completed"] is True
    assert api_fixed["deferrals"] and api_fixed["deferrals"][0] > 0


# =========================================================================== #
# Group 2 — call-site pins (AST)
# =========================================================================== #


def _method_ast(name: str) -> ast.AST:
    from myvoice.app import MyVoiceApp

    src = textwrap.dedent(inspect.getsource(getattr(MyVoiceApp, name)))
    return ast.parse(src).body[0]


def _attr_calls(node: ast.AST) -> List[str]:
    return [
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]


def test_voice_restoration_is_handed_off_through_the_idle_scheduler():
    """The restore must go through ``_schedule_when_loop_is_idle`` and must
    not be recreated via a QTimer or a direct ``create_task`` — the shape the
    control row above destroys."""
    fn = _method_ast("_on_voice_service_started")
    calls = _attr_calls(fn)
    assert "_schedule_when_loop_is_idle" in calls, calls
    assert "create_task" not in calls, "voice restoration reverted to loop.create_task"
    assert "singleShot" not in calls, "voice restoration reverted to QTimer.singleShot"
    assert "sleep" not in calls, (
        "voice restoration re-grew a timing guess (asyncio.sleep); the idle "
        "scheduler is the precise form of that intent"
    )


def test_api_server_start_is_deferred_not_awaited_inline():
    """``_setup_api_server_from_settings`` must not await ``start()`` in its
    own body (inside Task-1); the await has to live in the coroutine handed
    to ``_schedule_when_loop_is_idle``."""
    fn = _method_ast("_setup_api_server_from_settings")
    assert "_schedule_when_loop_is_idle" in _attr_calls(fn)

    # Any ``await ...start(...)`` must be nested inside an inner async def,
    # never directly in the method body.
    inner_async_defs = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.AsyncFunctionDef) and n is not fn
    ]
    inner_nodes = {id(m) for d in inner_async_defs for m in ast.walk(d)}
    top_level_start_awaits = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Await)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "start"
        and id(n) not in inner_nodes
    ]
    assert top_level_start_awaits == [], (
        "the API server is started inline inside Task-1 at line(s) "
        f"{top_level_start_awaits}; that is the shape the control row destroys"
    )


def test_warmup_call_site_and_wrapper_are_untouched():
    """Story 20.9 Dev Notes: do not touch the warmup or its call site. The
    wrapper must still route the warmup through ``_compile_warmup_entrypoint``
    (the hydration check + priming-gate release live there)."""
    fn = _method_ast("_run_async_task_when_loop_is_idle")
    calls = _attr_calls(fn)
    assert "_schedule_when_loop_is_idle" in calls
    assert "_compile_warmup_entrypoint" in calls


# =========================================================================== #
# Group 3 — behaviour preservation of the refactor (plain asyncio loop)
# =========================================================================== #
#
# These rows run on a plain asyncio loop where the qasync hazard is
# structurally absent. They prove contract details of the helpers, not that
# the shipped path survives qasync — Group 1 does that.


def _bare_app():
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp.__new__(MyVoiceApp)
    app.logger = logging.getLogger("test-20-9")
    app._voice_clone_prompt_hydration_task = None
    return app


def test_warmup_wrapper_log_lines_are_byte_identical_to_story_20_3(monkeypatch, caplog):
    """The AC #4 hardware evidence for Story 20.3 quotes these lines; the
    generic helper must reproduce them exactly for the warmup label."""
    from myvoice.app import MyVoiceApp

    monkeypatch.setattr(MyVoiceApp, "_MAX_IDLE_DEFERRALS", 2)
    monkeypatch.setattr(MyVoiceApp, "_on_tts_compile_priming_changed", lambda self, v: None)

    async def scenario():
        busy = {"n": 3}
        real = asyncio.current_task

        def fake_current_task(*a, **k):
            if busy["n"] > 0:
                busy["n"] -= 1
                return real(*a, **k) or "pretend-task-1"
            return real(*a, **k)

        monkeypatch.setattr(asyncio, "current_task", fake_current_task)

        async def warmup():
            pass

        app = _bare_app()
        app._run_async_task_when_loop_is_idle(warmup)
        for _ in range(10):
            await asyncio.sleep(0)

    with caplog.at_level(logging.DEBUG, logger="test-20-9"):
        asyncio.run(scenario())

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(
            "Deferring torch.compile warmup: task 'pretend-task-1' is mid-step "
            "(qasync delivers task steps from Qt timerEvent, and asyncio "
            "refuses to enter a second task re-entrantly)"
        ) or m.startswith("Deferring torch.compile warmup: task ")
        for m in messages
    ), messages
    assert (
        "torch.compile warmup: the event loop never went idle after 2 passes; "
        "scheduling anyway. If the task is destroyed pending, compile priming "
        "will not run this launch and the first generation pays the inductor "
        "reload."
    ) in messages, messages
    assert (
        "torch.compile warmup handed off to the event loop "
        "(deferred 2 loop pass(es) for qasync re-entrancy safety)"
    ) in messages, messages
    assert "torch.compile warmup task completed" in messages, messages


def test_generic_helper_runs_the_factory_once_with_callbacks():
    """Contract: the factory is called exactly once, on the idle pass; the
    task runs once; ``on_success`` receives the result; the helper returns
    ``None`` (creation is deferred, so there is no future to hand back)."""

    async def scenario():
        calls = {"factory": 0, "body": 0}
        got: List[Any] = []

        async def body():
            calls["body"] += 1
            return "result"

        def factory():
            calls["factory"] += 1
            return body()

        app = _bare_app()
        ret = app._schedule_when_loop_is_idle(factory, label="x", on_success=got.append)
        assert ret is None
        assert calls["factory"] == 0, "the factory must not be called eagerly"
        for _ in range(5):
            await asyncio.sleep(0)
        return calls, got

    calls, got = asyncio.run(scenario())
    assert calls == {"factory": 1, "body": 1}
    assert got == ["result"]


def test_generic_helper_creates_its_task_one_pass_after_a_plain_neighbour():
    """Ordering contract, stated precisely so nobody assumes FIFO parity.

    Idle hand-offs keep their order among themselves, and a plain task
    scheduled *before* an idle hand-off still runs first — but a plain task
    scheduled *after* one also runs first, because the idle helper spends a
    pass on its callback before creating the task. This is exactly why the
    restore move is order-preserving: the restore's idle callback is armed
    before hydration's plain task in ``_initialize_services_async``, so
    hydration's single step still lands before the restore's first step
    (proved under qasync by ``test_startup_sites_run_once_and_in_production_order``).
    """

    async def scenario():
        order: List[str] = []

        async def first():
            order.append("idle-first")

        async def second():
            order.append("plain-second")

        async def third():
            order.append("idle-third")

        app = _bare_app()
        app._schedule_when_loop_is_idle(first, label="a")
        app._run_async_task(second())
        app._schedule_when_loop_is_idle(third, label="b")
        for _ in range(6):
            await asyncio.sleep(0)
        return order

    # The plain task's body runs on the pass after its creation; the idle
    # helper creates its task on that same pass and the body runs one pass
    # later — so a plain neighbour scheduled *after* an idle one still runs
    # first. What is preserved is the order among idle hand-offs and the
    # order relative to anything scheduled before them; see the evidence
    # file for why the restore does not depend on beating hydration.
    assert asyncio.run(scenario()) == ["plain-second", "idle-first", "idle-third"]


def test_deferred_api_start_reads_the_toggle_at_run_time():
    """A Settings change inside the deferral gap must not start a server the
    user just disabled; when still enabled it starts exactly once on the
    configured port."""
    from myvoice.models.app_settings import AppSettings

    async def scenario(enabled_at_run_time: bool):
        app = _bare_app()
        app._tts_service = SimpleNamespace()
        app._voice_manager = SimpleNamespace()
        app._stream_hub = None
        app._api_server = None
        app._app_settings = AppSettings(enable_http_api=True, http_api_port=7801)

        started: List[Any] = []

        class FakeController:
            def __init__(self, **kw):
                pass

            async def start(self, host, port):
                started.append((host, port))

        import myvoice.services.api_server.server as server_mod

        original = server_mod.ApiServerController
        server_mod.ApiServerController = FakeController
        try:
            await app._setup_api_server_from_settings()
            assert started == [], "start must be deferred, not awaited inline"
            if not enabled_at_run_time:
                app._app_settings = AppSettings(enable_http_api=False, http_api_port=7801)
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            server_mod.ApiServerController = original
        return started

    assert asyncio.run(scenario(True)) == [("127.0.0.1", 7801)]
    assert asyncio.run(scenario(False)) == []
