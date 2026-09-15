"""Story 20.9 (F7) — out-of-process driver for the qasync call-site audit.

Run as ``python tests/unit/_qasync_call_site_driver.py <variant>``; prints one
JSON object on stdout. Not collected by pytest (no ``test_`` prefix) — it is
spawned by ``test_app_qasync_call_sites.py``. Same shape as Story 20.3's
``_qasync_warmup_driver.py`` and for the same reason: the hazard needs a real
``QApplication`` + ``qasync.QEventLoop`` and a task stepped from inside a
nested ``processEvents()``, and standing that up inside the shared pytest Qt
session hangs the suite.

The hazard, restated: under qasync every ``call_soon`` is a Qt zero-timer, so
a task step is delivered from ``timerEvent`` during ANY Qt event processing —
including a ``processEvents()`` run synchronously inside another task. If that
happens while the other task is current, ``asyncio._enter_task`` raises
``Cannot enter into task``, the loop's exception handler swallows it, and the
stepped task is a zombie: pending forever, never rescheduled.

Two audited sites are exercised here, each with a control row that reproduces
the pre-20.9 shape and must be destroyed, and a fixed row that drives the
shipped code and must survive:

  ``restore_legacy_fastfail`` — pre-20.9 ``_on_voice_service_started``:
        ``QTimer.singleShot(500, lambda: loop.create_task(delayed_restore()))``
        under a launch where the model preload returns without suspending, so
        Task-1 is still pumping Qt when the 500 ms timer fires. The QTimer
        slot survives (it is a plain Qt callback); the task it creates does
        not.
  ``restore_fixed_fastfail`` — the shipped ``_on_voice_service_started``
        (``_schedule_when_loop_is_idle``) under the same rig.
  ``startup_order_fixed`` — the shipped restore + hydration + compile-warmup
        hand-offs in ``_initialize_services_async``'s order under a normal
        launch shape (preload suspends), asserting each runs once and in the
        production order: hydration -> restore -> warmup.
  ``api_legacy`` — pre-20.9 ``_setup_api_server_from_settings``: a REAL
        ``ApiServerController.start()`` awaited inline in Task-1, then the
        post-init pump. uvicorn's ``serve()`` main loop wakes every 100 ms;
        the wake-up lands in the pump and the serve task becomes a zombie.
        Observed as: ``on_tick`` stops being called.
  ``api_fixed`` — the shipped ``_setup_api_server_from_settings`` (deferred
        start) under the same rig: ``on_tick`` keeps being called after the
        pump and ``stop()`` completes.

Story 20.11 adds the post-tier-change re-prime site (AC #4). The handler
(``_on_settings_changed``) is a Qt signal slot with no task on the stack, but
the ``on_success`` continuation that schedules the re-prime runs inside the
``set_quality_tier`` task's final step — so ``current_task()`` there is NOT
``None``. Three variants, all driven with Task-1 parked (the interactive
phase; ``app_close.wait()`` in main.py):

  ``tier_change_fixed`` — the shipped handler, called from a zero-timer Qt
        slot exactly as the signal would. Records which task is current in
        the slot and in the continuation, and that the re-prime runs once in
        production order (hydrate -> preload -> warmup) with no re-entrancy
        error.
  ``tier_change_plain_pumped`` — control: the continuation replaced by the
        plain ``_run_async_task`` shape, plus a Qt pump inside the
        continuation (what a dialog opened from it would do). The re-prime's
        first step is delivered inside the parent's step and destroyed.
  ``tier_change_fixed_pumped`` — the shipped continuation under the same
        pump: deferred through it, created once the parent completes, runs.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch  # noqa: F401  — torch-before-PyQt6 DLL ordering (see memory)

from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtWidgets import QApplication

import qasync

# Story 20.11: import THIS checkout's ``src``, not whichever one the
# interpreter was configured with. The bundled portable interpreter's
# ``python310._pth`` pins ``..\src`` (the main checkout) and — being a
# ``._pth`` — ignores ``PYTHONPATH`` entirely, so the ``env["PYTHONPATH"]`` the
# spawning test sets never reached this process. Run from a git worktree, the
# driver was importing the main checkout's ``app.py`` and proving nothing
# about the code under test. ``conftest.py`` does the same insert for the
# in-process suite; this is the out-of-process half of it.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from myvoice.models.app_settings import AppSettings  # noqa: E402


PUMP_ITERATIONS = 40  # Story 20.3's stand-in for main.py:397-404 (normal launch)


def _pump_for(seconds: float, tight: bool = False) -> int:
    """Synchronously pump Qt for ``seconds`` inside Task-1.

    Default shape mirrors ``QSplashScreen::finish`` waiting for the main
    window to be exposed (Qt's ``waitForWindowExposed``: ``processEvents``
    then ``msleep(10)`` per iteration, up to 1 s) — the longest synchronous
    pump main.py can run after ``initialize_async`` returns, at ~100 passes/s.

    ``tight=True`` spins ``processEvents`` back-to-back (~10^6 passes/s).
    Nothing in main.py does that; it is here to document a limit of the
    idle scheduler's safety valve (see the evidence file): every pass
    delivers the re-armed zero-timer once, so ``_MAX_IDLE_DEFERRALS``
    (10,000) is exhausted in ~15 ms and the valve schedules into the pump.
    """
    n = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        n += 1
        if not tight:
            time.sleep(0.01)
    return n


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: List[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _bare_app(events: List[str], phase: Dict[str, str]):
    from myvoice.app import MyVoiceApp

    app_obj = MyVoiceApp.__new__(MyVoiceApp)
    app_obj.logger = logging.getLogger("qasync-call-site-driver")
    app_obj._main_window = None
    app_obj._voice_clone_prompt_hydration_task = None
    app_obj._stream_hub = None
    app_obj._api_server = None
    app_obj._tts_service = SimpleNamespace()
    app_obj._app_settings = AppSettings()

    async def restore_voice_selection():
        events.append(f"restore_start@{phase['now']}")
        # Span several loop passes, like the real config read + profile set.
        for _ in range(3):
            await asyncio.sleep(0)
        return "Sarira-F"

    async def set_active_profile(name):
        await asyncio.sleep(0)
        events.append(f"restore_done@{phase['now']}")
        return True

    app_obj._config_manager = SimpleNamespace(
        restore_voice_selection=restore_voice_selection
    )
    app_obj._voice_manager = SimpleNamespace(set_active_profile=set_active_profile)
    return app_obj


def _legacy_on_voice_service_started(app_obj, loop):
    """The pre-20.9 body of ``_on_voice_service_started``, verbatim in shape."""

    async def delayed_restore():
        await asyncio.sleep(0.5)  # Wait half second for init to complete
        try:
            await app_obj._restore_voice_selection_on_startup()
            app_obj._on_voice_restoration_complete(None)
        except Exception as e:  # noqa: BLE001
            app_obj._on_voice_restoration_failed(e)

    def schedule_restore():
        loop.create_task(delayed_restore())

    QTimer.singleShot(500, schedule_restore)


def _install_tick_counter(ticks: Dict[str, int]):
    from uvicorn.server import Server

    original = Server.on_tick

    async def counting_on_tick(self, counter):
        ticks["n"] += 1
        return await original(self, counter)

    Server.on_tick = counting_on_tick  # type: ignore[assignment]


def main(variant: str) -> Dict[str, Any]:
    qt_app = QApplication.instance() or QApplication([])

    loop_errors: List[str] = []
    events: List[str] = []
    phase = {"now": "init"}
    ticks = {"n": 0}
    capture = _LogCapture()
    logging.getLogger("qasync-call-site-driver").addHandler(capture)
    logging.getLogger("qasync-call-site-driver").setLevel(logging.DEBUG)

    loop = qasync.QEventLoop(qt_app)
    asyncio.set_event_loop(loop)

    def _handler(_loop, context):
        exc = context.get("exception")
        loop_errors.append(
            f"{type(exc).__name__}: {exc}" if exc else str(context.get("message"))
        )

    loop.set_exception_handler(_handler)

    app_obj = _bare_app(events, phase)
    result: Dict[str, Any] = {"variant": variant}

    # ------------------------------------------------------------------ #
    async def preload_returns_without_suspending():
        # A preload that fails synchronously (no CUDA / missing checkpoint):
        # ``await`` of a coroutine that never suspends does not yield.
        return False, "no model"

    async def preload_suspends():
        await asyncio.sleep(0.05)
        return True, None

    async def task1_restore(fixed: bool, tight: bool = False):
        if fixed:
            app_obj._on_voice_service_started(None)
        else:
            _legacy_on_voice_service_started(app_obj, loop)
        await preload_returns_without_suspending()
        phase["now"] = "pumping"
        # main.py post-init stretch on a fast-fail launch: long enough for the
        # legacy 500 ms QTimer (and its 0.5 s sleep) to fire inside it.
        result["pump_passes"] = _pump_for(1.3, tight=tight)
        phase["now"] = "parked"
        await asyncio.sleep(0.7)

    async def task1_startup_order():
        # _initialize_services_async, in order: voice service started ->
        # hydration scheduled -> preload awaited -> warmup handed off ->
        # main.py pump -> park.
        app_obj._on_voice_service_started(None)

        async def hydrate():
            events.append(f"hydration@{phase['now']}")
            return (13, 14)

        app_obj._voice_clone_prompt_hydration_task = app_obj._run_async_task(hydrate())
        phase["now"] = "preload"
        await preload_suspends()
        phase["now"] = "post-preload"

        async def warmup():
            events.append(f"warmup@{phase['now']}")

        app_obj._run_async_task_when_loop_is_idle(warmup)
        phase["now"] = "pumping"
        for _ in range(PUMP_ITERATIONS):
            QCoreApplication.processEvents()
        phase["now"] = "parked"
        await asyncio.sleep(0.5)

    async def task1_api(fixed: bool):
        _install_tick_counter(ticks)
        from myvoice.services.api_server.server import ApiServerController

        port = _free_port()
        app_obj._app_settings = AppSettings(enable_http_api=True, http_api_port=port)
        app_obj._api_server = ApiServerController(
            tts_service=SimpleNamespace(),
            voice_manager=SimpleNamespace(),
            app_ref=app_obj,
            settings_provider=lambda: app_obj._app_settings,
        )
        ctrl = app_obj._api_server

        if fixed:
            await app_obj._setup_api_server_from_settings()
        else:
            # pre-20.9 body: start awaited inline inside Task-1
            await ctrl.start(host="127.0.0.1", port=port)
        result["running_before_pump"] = ctrl.is_running
        phase["now"] = "pumping"
        # main.py post-init stretch, sized to cover several 100 ms ticks.
        result["pump_passes"] = _pump_for(0.35)
        phase["now"] = "parked"

        # Give a deferred start time to come up, then measure liveness.
        deadline = loop.time() + 3.0
        while loop.time() < deadline and not getattr(ctrl._server, "started", False):
            await asyncio.sleep(0.02)
        result["running_after_pump"] = ctrl.is_running
        before = ticks["n"]
        await asyncio.sleep(0.5)
        result["ticks_while_parked"] = ticks["n"] - before

        stop_task = asyncio.ensure_future(ctrl.stop())
        done, _pending = await asyncio.wait({stop_task}, timeout=4.0)
        result["stop_completed"] = stop_task in done
        if stop_task not in done:
            stop_task.cancel()
        gc.collect()

    async def task1_tier_change(shape: str):
        """Story 20.11 AC #4. Task-1 parks (the interactive phase) and the
        Settings-changed handler fires from a Qt zero-timer slot — no task on
        the stack, exactly as the ``settings_changed`` signal delivers it.

        ``shape``: ``"fixed"`` (shipped continuation), ``"fixed_pumped"``
        (shipped continuation + a Qt pump inside it), ``"plain_pumped"``
        (pre-fix scheduling shape — plain ``_run_async_task`` from inside the
        continuation — + the same pump; the control that must be destroyed).
        """
        from myvoice.models.service_enums import QwenModelType

        async def set_quality_tier(tier):
            await asyncio.sleep(0)
            events.append(f"tier_set@{phase['now']}")
            return True

        async def hydrate_voice_clone_prompt_cache():
            events.append(f"hydrate@{phase['now']}")
            return (12, 12)

        async def preload_model(model_type):
            events.append(f"preload_start:{model_type.name}@{phase['now']}")
            await asyncio.sleep(0.02)
            events.append(f"preload_done@{phase['now']}")
            return True, None

        async def warmup_compile_async():
            events.append(f"warmup@{phase['now']}")

        app_obj._tts_service = SimpleNamespace(
            set_app_settings=lambda s: None,
            set_quality_tier=set_quality_tier,
            hydrate_voice_clone_prompt_cache=hydrate_voice_clone_prompt_cache,
            preload_model=preload_model,
            warmup_compile_async=warmup_compile_async,
        )
        app_obj._voice_manager = SimpleNamespace(
            get_active_profile_model_type=lambda: QwenModelType.BASE
        )

        async def save_settings():
            return True

        app_obj._config_manager = SimpleNamespace(save_settings=save_settings)
        # Non-None sentinels so ``_reconcile_api_server`` takes its "disabled,
        # not running" no-op branch instead of constructing a real controller.
        app_obj._api_server = SimpleNamespace(is_running=False)
        app_obj._stream_hub = object()
        app_obj._app_settings = AppSettings(model_quality_tier="small")

        shipped_continuation = app_obj._on_quality_tier_updated

        def _task_name(task):
            return task.get_name() if task is not None else None

        def continuation(changed):
            result["continuation_current_task"] = _task_name(asyncio.current_task())
            if shape == "plain_pumped":
                # The pre-fix shape: create the task directly from inside the
                # parent's step. Its first step is a zero-timer, delivered
                # by the pump below while the parent is still current.
                app_obj.logger.info(f"Quality tier updated: {'changed' if changed else 'no change'}")
                app_obj._run_async_task(
                    app_obj._compile_warmup_entrypoint(
                        app_obj._reprime_after_tier_change
                    )
                )
            else:
                shipped_continuation(changed)
            if shape.endswith("_pumped"):
                phase["now"] = "pumping-in-continuation"
                result["pump_passes"] = _pump_for(0.2)
                phase["now"] = "parked"

        app_obj._on_quality_tier_updated = continuation

        def qt_slot():
            result["handler_current_task"] = _task_name(asyncio.current_task())
            app_obj._on_settings_changed(AppSettings(model_quality_tier="quality"))

        phase["now"] = "parked"
        QTimer.singleShot(0, qt_slot)
        # Task-1 parked, as at ``app_close.wait()``; long enough for the whole
        # chain (tier set -> continuation -> idle pass -> re-prime) to land.
        await asyncio.sleep(1.0)
        gc.collect()

    if variant == "restore_legacy_fastfail":
        coro = task1_restore(fixed=False)
    elif variant == "restore_fixed_fastfail":
        coro = task1_restore(fixed=True)
    elif variant == "restore_fixed_tightspin":
        # Documents the safety-valve limit; not a shipped launch shape.
        coro = task1_restore(fixed=True, tight=True)
    elif variant == "startup_order_fixed":
        coro = task1_startup_order()
    elif variant == "api_legacy":
        coro = task1_api(fixed=False)
    elif variant == "api_fixed":
        coro = task1_api(fixed=True)
    elif variant == "tier_change_fixed":
        coro = task1_tier_change("fixed")
    elif variant == "tier_change_fixed_pumped":
        coro = task1_tier_change("fixed_pumped")
    elif variant == "tier_change_plain_pumped":
        coro = task1_tier_change("plain_pumped")
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    with loop:
        loop.run_until_complete(coro)
        gc.collect()

    deferred = [
        int(m.group(1))
        for line in capture.lines
        for m in [re.search(r"deferred (\d+) loop pass\(es\)", line)]
        if m
    ]
    result.update(
        {
            "events": events,
            "ticks_total": ticks["n"],
            "deferrals": deferred,
            "reentrancy_errors": [e for e in loop_errors if "Cannot enter into task" in e],
            "destroyed_pending": [e for e in loop_errors if "destroyed but it is pending" in e],
            "all_errors": loop_errors,
        }
    )
    return result


if __name__ == "__main__":
    out = main(sys.argv[1] if len(sys.argv) > 1 else "startup_order_fixed")
    print("__RESULT__" + json.dumps(out))
