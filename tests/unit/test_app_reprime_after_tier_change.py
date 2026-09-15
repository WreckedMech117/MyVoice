"""Story 20.11 AC #1/#2/#4 — a model quality tier change re-runs the startup
preload + compile priming against the new tier.

**The exact defect class this file guards.** ``_on_settings_changed`` handed
``set_quality_tier`` to ``_run_async_task`` with an ``on_success`` that only
logged. ``set_quality_tier`` UNLOADS the resident model ("will take effect on
next generation") and nothing reloaded or primed, so the user's next Generate
paid the 17 s model load plus a cold compile for the new tier's key inside
the request — 19.2 s to first chunk on the RTX 3060 (Story 20.10 evidence §5,
2026-09-14 20:24:35), against 1.90 s once warm.

Three groups:

  1. **Behaviour on a plain asyncio loop** — the continuation schedules the
     re-prime iff ``changed`` is True; the re-prime runs hydrate -> preload
     (active profile's model, else CUSTOM_VOICE) -> ``warmup_compile_async``;
     the preload failure paths skip priming with a WARNING and leave the
     Generate gate released (AC #2); a switch back re-primes again.
  2. **Call-site pins (AST)** — the continuation must stay wired, must route
     through ``_schedule_when_loop_is_idle`` and ``_compile_warmup_entrypoint``
     (the Story 20.7 release-on-exit guarantee lives there), and must not be
     recreated through the plain scheduler.
  3. **Real qasync loop (out of process)** — AC #4. The scheduling
     classification is recorded from the live loop: no task current in the
     Qt slot, the ``set_quality_tier`` task current in the continuation. A
     control row reproduces the plain-scheduling shape under a pump and is
     destroyed; the shipped shape survives the same pump.

The plain-loop rows cannot see the qasync hazard (``call_soon`` there is a
ready-queue append, drained only between task steps). Do not add a "the
re-prime runs" row here and believe it covers the shipped path — group 3 does.
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
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("PyQt6")

from myvoice.models.app_settings import AppSettings
from myvoice.models.service_enums import QwenModelType


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().parent / "_qasync_call_site_driver.py"


# --------------------------------------------------------------------------- #
# Rig
# --------------------------------------------------------------------------- #


class _FakeTTS:
    """The five TTS-service methods the handler and the re-prime touch.

    ``preload_result`` is what ``preload_model`` returns; ``preload_raises``
    makes it raise instead; ``changed`` is ``set_quality_tier``'s answer.
    """

    def __init__(
        self,
        *,
        changed: bool = True,
        preload_result=(True, None),
        preload_raises: Optional[Exception] = None,
        hydrate_raises: Optional[Exception] = None,
    ) -> None:
        self.events: List[str] = []
        self._changed = changed
        self._preload_result = preload_result
        self._preload_raises = preload_raises
        self._hydrate_raises = hydrate_raises

    def set_app_settings(self, settings) -> None:
        self.events.append("set_app_settings")

    async def set_quality_tier(self, tier: str) -> bool:
        await asyncio.sleep(0)
        self.events.append(f"set_quality_tier:{tier}")
        return self._changed

    async def hydrate_voice_clone_prompt_cache(self):
        if self._hydrate_raises is not None:
            raise self._hydrate_raises
        self.events.append("hydrate")
        return (12, 12)

    async def preload_model(self, model_type):
        self.events.append(f"preload:{model_type.name}")
        if self._preload_raises is not None:
            raise self._preload_raises
        await asyncio.sleep(0)
        return self._preload_result

    async def warmup_compile_async(self) -> None:
        self.events.append("warmup")


def _bare_app(tts: _FakeTTS, *, active_model=QwenModelType.BASE, tier="small"):
    """A ``MyVoiceApp`` shell carrying only what ``_on_settings_changed``
    and the re-prime read. ``__new__`` avoids the full constructor.

    The handler's later ``hasattr(self, '_audio_coordinator')`` raises
    ``RuntimeError`` on an un-``__init__``-ed QObject (its ``__getattr__``
    does not raise ``AttributeError``), which the handler's outer ``except``
    logs as an ERROR. That is after the tier-change hand-off — the FIFO
    neighbours behind it (Story 20.9 §2.3 #34–37) are not this file's
    subject — and the same shape the 20.9 driver rig has.
    """
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp.__new__(MyVoiceApp)
    app.logger = logging.getLogger("test-20-11")
    app._tts_service = tts
    app._voice_manager = SimpleNamespace(
        get_active_profile_model_type=lambda: active_model
    )
    app._voice_clone_prompt_hydration_task = None
    app._main_window = None
    app._app_settings = AppSettings(model_quality_tier=tier)

    async def save_settings():
        return True

    app._config_manager = SimpleNamespace(save_settings=save_settings)
    # Non-None sentinels: ``_reconcile_api_server`` then takes its
    # "disabled, not running" no-op branch.
    app._api_server = SimpleNamespace(is_running=False)
    app._stream_hub = object()

    gate: List[bool] = []
    app._on_tts_compile_priming_changed = gate.append
    app._gate_calls = gate
    return app


async def _settle(passes: int = 12) -> None:
    for _ in range(passes):
        await asyncio.sleep(0)


# =========================================================================== #
# Group 1 — behaviour on a plain asyncio loop
# =========================================================================== #


def test_tier_change_reprimes_in_startup_order():
    """AC #1 — ``changed=True`` schedules hydrate -> preload(active model) ->
    warmup, exactly once, after the tier is set."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts, active_model=QwenModelType.BASE, tier="small")
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        return tts.events, app._gate_calls

    events, gate = asyncio.run(scenario())
    assert events == [
        "set_app_settings",
        "set_quality_tier:quality",
        "hydrate",
        "preload:BASE",
        "warmup",
    ], events
    # The entrypoint's ``finally`` lands one release; the fake warmup never
    # engages, so a lone False is the whole stream.
    assert gate == [False], gate


def test_tier_change_falls_back_to_custom_voice_without_an_active_profile_model():
    """AC #1 — same choice logic as startup: no active-profile model type
    means the default CUSTOM_VOICE model is preloaded and primed."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts, active_model=None)
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        return tts.events

    events = asyncio.run(scenario())
    assert "preload:CUSTOM_VOICE" in events, events
    assert events[-1] == "warmup", events


def test_no_change_schedules_nothing():
    """AC #1, last clause — ``changed=False`` (the registry already held that
    tier) must not reload or prime anything."""

    async def scenario():
        tts = _FakeTTS(changed=False)
        app = _bare_app(tts)
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        return tts.events, app._gate_calls

    events, gate = asyncio.run(scenario())
    assert events == ["set_app_settings", "set_quality_tier:quality"], events
    assert gate == [], "nothing was scheduled, so the gate must not be touched"


def test_same_tier_in_settings_never_calls_set_quality_tier():
    """The pre-existing guard: a Settings save that did not touch the tier
    does not reach ``set_quality_tier`` at all, so it cannot re-prime."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts, tier="quality")
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        return tts.events

    assert asyncio.run(scenario()) == ["set_app_settings"]


def test_switching_back_reprimes_again():
    """AC #1 — the compile-cache key is per model_id, so a change back to the
    original tier has to re-prime that tier's key too."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts, tier="small")
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        app._on_settings_changed(AppSettings(model_quality_tier="small"))
        await _settle()
        return tts.events

    events = asyncio.run(scenario())
    assert events.count("warmup") == 2, events
    assert events.count("preload:BASE") == 2, events
    assert [e for e in events if e.startswith("set_quality_tier")] == [
        "set_quality_tier:quality",
        "set_quality_tier:small",
    ]


@pytest.mark.parametrize(
    "fake_kwargs, why",
    [
        ({"preload_result": (False, "no checkpoint")}, "failed to preload"),
        ({"preload_raises": RuntimeError("CUDA OOM")}, "raised"),
    ],
    ids=["preload-returns-false", "preload-raises"],
)
def test_preload_failure_skips_priming_warns_and_leaves_the_gate_released(
    caplog, fake_kwargs, why
):
    """AC #2 — no model means no key to prime; priming is not attempted, a
    WARNING names why, and ``_on_tts_compile_priming_changed(False)`` is the
    terminal (and only) gate state: the gate was never engaged."""

    async def scenario():
        tts = _FakeTTS(changed=True, **fake_kwargs)
        app = _bare_app(tts)
        with caplog.at_level(logging.WARNING, logger="test-20-11"):
            app._on_settings_changed(AppSettings(model_quality_tier="quality"))
            await _settle()
        return tts.events, app._gate_calls

    events, gate = asyncio.run(scenario())
    assert "warmup" not in events, events
    assert "preload:BASE" in events, events
    assert gate == [False], (
        f"gate stream {gate}: the terminal state must be released, and the "
        "gate must never have been engaged for a preload that produced no model"
    )
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any(
        "Post-tier-change re-prime skipped" in m and why in m for m in warnings
    ), warnings


def test_hydration_failure_does_not_cost_the_preload_or_the_prime(caplog):
    """The re-hydration is best-effort: a raise there is a WARNING and the
    preload + prime still run (BASE would then skip with no_priming_prompt —
    the service's designed fallback — but CUSTOM_VOICE/VOICE_DESIGN residents
    need no prompt at all)."""

    async def scenario():
        tts = _FakeTTS(changed=True, hydrate_raises=OSError("disk"))
        app = _bare_app(tts)
        with caplog.at_level(logging.WARNING, logger="test-20-11"):
            app._on_settings_changed(AppSettings(model_quality_tier="quality"))
            await _settle()
        return tts.events

    events = asyncio.run(scenario())
    assert events[-2:] == ["preload:BASE", "warmup"], events
    assert any(
        "hydration after tier change failed" in r.getMessage()
        for r in caplog.records
    )


def test_reprime_rehydrates_for_the_new_tier_before_priming():
    """Story 17.2's startup hydration is per tier and the BASE priming
    request looks the prompt up in memory only, so without a re-hydration
    every cloned-voice re-prime would skip with ``no_priming_prompt``. The
    hydrate must precede the warmup."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts)
        await app._reprime_after_tier_change()
        return tts.events

    events = asyncio.run(scenario())
    assert events.index("hydrate") < events.index("warmup"), events


def test_continuation_hand_off_log_lines_use_the_idle_scheduler(caplog):
    """The hand-off line is the generic helper's, with this site's label —
    the same shape Story 20.3/20.9 quote for the startup warmup."""

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts)
        with caplog.at_level(logging.DEBUG, logger="test-20-11"):
            app._on_settings_changed(AppSettings(model_quality_tier="quality"))
            await _settle()

    asyncio.run(scenario())
    messages = [r.getMessage() for r in caplog.records]
    assert "Quality tier updated: changed" in messages, messages
    assert (
        "post-tier-change compile warmup handed off to the event loop "
        "(deferred 0 loop pass(es) for qasync re-entrancy safety)"
    ) in messages, messages
    assert "post-tier-change compile warmup task completed" in messages, messages


def test_continuation_runs_inside_the_set_quality_tier_task():
    """AC #4, the plain-loop half of the classification: ``on_success`` is
    called from the ``set_quality_tier`` task's final step, so
    ``asyncio.current_task()`` is that task — not ``None`` — when the
    re-prime is scheduled. This is why the site goes through
    ``_schedule_when_loop_is_idle``; the qasync rows below prove the
    consequence under a real loop."""
    from myvoice.app import MyVoiceApp

    seen: Dict[str, Any] = {}
    original = MyVoiceApp._on_quality_tier_updated

    def recording(self, changed):
        seen["current"] = asyncio.current_task()
        return original(self, changed)

    async def scenario():
        tts = _FakeTTS(changed=True)
        app = _bare_app(tts)
        app._on_quality_tier_updated = recording.__get__(app, MyVoiceApp)
        app._on_settings_changed(AppSettings(model_quality_tier="quality"))
        await _settle()
        return tts.events

    events = asyncio.run(scenario())
    assert "warmup" in events
    current = seen["current"]
    assert current is not None, "the continuation ran with no task current"
    assert "_handle_task" in repr(current.get_coro()), repr(current)


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


def test_settings_handler_wires_the_continuation_as_on_success():
    """The tier-change ``_run_async_task`` must pass
    ``_on_quality_tier_updated`` as ``on_success`` — reverting to the old
    log-only lambda is the defect."""
    fn = _method_ast("_on_settings_changed")
    tier_calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_async_task"
        and "set_quality_tier" in _attr_calls(n)
    ]
    assert len(tier_calls) == 1, "expected exactly one set_quality_tier hand-off"
    on_success = [
        kw.value for kw in tier_calls[0].keywords if kw.arg == "on_success"
    ]
    assert on_success and isinstance(on_success[0], ast.Attribute), (
        "set_quality_tier's on_success is not a bound method"
    )
    assert on_success[0].attr == "_on_quality_tier_updated", (
        f"on_success is {on_success[0].attr!r}; the re-prime continuation is "
        "no longer wired and a tier change leaves the next Generate to pay "
        "the model load and a cold compile"
    )


def test_continuation_routes_through_the_idle_scheduler_and_the_entrypoint():
    """AC #4 — the continuation runs inside the ``set_quality_tier`` task's
    step, so it must schedule via ``_schedule_when_loop_is_idle`` (never the
    plain ``_run_async_task``), and it must wrap the re-prime in
    ``_compile_warmup_entrypoint`` for the Story 20.7 release-on-exit
    guarantee (AC #2)."""
    fn = _method_ast("_on_quality_tier_updated")
    calls = _attr_calls(fn)
    assert "_schedule_when_loop_is_idle" in calls, calls
    assert "_compile_warmup_entrypoint" in calls, calls
    assert "_run_async_task" not in calls, (
        "the re-prime is scheduled through the plain _run_async_task from "
        "inside a running task's step; that is the shape the qasync control "
        "row destroys"
    )
    assert "create_task" not in calls and "ensure_future" not in calls, calls


def test_reprime_body_preloads_then_primes_through_the_untouched_warmup():
    """The re-prime must call the service's own ``warmup_compile_async``
    (gates and telemetry untouched, per Dev Notes) after ``preload_model``,
    and must not await the warmup when the preload did not succeed."""
    fn = _method_ast("_reprime_after_tier_change")
    calls = _attr_calls(fn)
    assert "hydrate_voice_clone_prompt_cache" in calls, calls
    assert "preload_model" in calls, calls
    assert "warmup_compile_async" in calls, calls
    assert "_run_compile_priming" not in calls, (
        "the re-prime bypasses warmup_compile_async's gates"
    )
    linenos = {
        n.func.attr: n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"hydrate_voice_clone_prompt_cache", "preload_model", "warmup_compile_async"}
    }
    assert (
        linenos["hydrate_voice_clone_prompt_cache"]
        < linenos["preload_model"]
        < linenos["warmup_compile_async"]
    ), linenos


def test_startup_call_site_is_byte_identical():
    """AC #5 — the startup hand-off in ``_initialize_services_async`` is not
    this story's to touch: still exactly one idle hand-off of
    ``warmup_compile_async`` and no new scheduling of it."""
    fn = _method_ast("_initialize_services_async")
    idle = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_async_task_when_loop_is_idle"
    ]
    assert len(idle) == 1
    assert "warmup_compile_async" in [
        a.attr for a in ast.walk(idle[0]) if isinstance(a, ast.Attribute)
    ]
    assert "_reprime_after_tier_change" not in _attr_calls(fn)
    assert "_on_quality_tier_updated" not in [
        a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)
    ]


# =========================================================================== #
# Group 3 — real qasync loop (out of process)
# =========================================================================== #


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
def tier_fixed() -> Dict[str, Any]:
    pytest.importorskip("qasync")
    return _run_driver("tier_change_fixed")


@pytest.fixture(scope="module")
def tier_fixed_pumped() -> Dict[str, Any]:
    pytest.importorskip("qasync")
    return _run_driver("tier_change_fixed_pumped")


@pytest.fixture(scope="module")
def tier_plain_pumped() -> Dict[str, Any]:
    pytest.importorskip("qasync")
    return _run_driver("tier_change_plain_pumped")


def test_qasync_classification_slot_has_no_task_and_continuation_does(tier_fixed):
    """AC #4 — recorded from the live loop: the Qt slot runs with no task
    current (class (a)); the ``on_success`` continuation runs inside the
    ``set_quality_tier`` task (the (b0) shape), which is why the site uses
    the idle scheduler."""
    assert tier_fixed["handler_current_task"] is None, tier_fixed
    assert tier_fixed["continuation_current_task"] is not None, tier_fixed


def test_qasync_shipped_reprime_runs_once_in_order_with_task1_parked(tier_fixed):
    """AC #1/#4 — the shipped handler under a real qasync loop: the re-prime
    lands on the first idle pass (no deferral needed: the parent completed
    before the callback ran), runs hydrate -> preload -> warmup exactly once,
    zero re-entrancy errors."""
    assert tier_fixed["reentrancy_errors"] == [], tier_fixed["all_errors"]
    assert tier_fixed["destroyed_pending"] == []
    assert tier_fixed["events"] == [
        "tier_set@parked",
        "hydrate@parked",
        "preload_start:BASE@parked",
        "preload_done@parked",
        "warmup@parked",
    ], tier_fixed["events"]
    assert tier_fixed["deferrals"] == [0], tier_fixed["deferrals"]


def test_qasync_control_plain_scheduling_from_the_continuation_is_destroyed(
    tier_plain_pumped,
):
    """Non-vacuity control. The plain ``_run_async_task`` shape from inside
    the continuation, with a Qt pump in that continuation (what a dialog
    opened from it would do): the re-prime's first step is delivered while
    the ``set_quality_tier`` task is still current, ``_enter_task`` refuses,
    and the re-prime never runs. If this row ever goes green the rig has
    stopped reproducing the hazard and the fixed row proves nothing."""
    assert tier_plain_pumped["events"] == ["tier_set@parked"], (
        "the plainly-scheduled re-prime ran, so this rig no longer reproduces "
        f"the hazard: {tier_plain_pumped['events']}"
    )
    assert tier_plain_pumped["reentrancy_errors"], tier_plain_pumped["all_errors"]
    assert tier_plain_pumped["destroyed_pending"]


def test_qasync_shipped_reprime_survives_a_pump_inside_the_continuation(
    tier_fixed_pumped,
):
    """AC #4 — the shipped continuation under the same pump: deferred through
    every pump pass (the parent is current for all of them), created once
    the parent completes, runs once, zero re-entrancy errors."""
    assert tier_fixed_pumped["reentrancy_errors"] == [], tier_fixed_pumped["all_errors"]
    assert tier_fixed_pumped["destroyed_pending"] == []
    assert tier_fixed_pumped["events"] == [
        "tier_set@parked",
        "hydrate@parked",
        "preload_start:BASE@parked",
        "preload_done@parked",
        "warmup@parked",
    ], tier_fixed_pumped["events"]
    assert tier_fixed_pumped["deferrals"] and tier_fixed_pumped["deferrals"][0] > 0, (
        "the hand-off never deferred, so the pump was not reproduced: "
        f"{tier_fixed_pumped['deferrals']}"
    )


# ---------------------------------------------------------------------------
# Review fixes on the 2.3.0 pass — overlapping entrypoint bodies
# ---------------------------------------------------------------------------


def test_overlapping_entrypoints_release_the_safety_net_only_once_at_the_end():
    """Review MEDIUM: the app-side gate safety net must not force Generate
    back on when the FIRST of two overlapping warmup bodies exits while the
    second is still priming; it fires once, when the last one exits."""

    async def scenario():
        from myvoice.app import MyVoiceApp

        app = MyVoiceApp.__new__(MyVoiceApp)
        app.logger = logging.getLogger("test-20-11")
        app._voice_clone_prompt_hydration_task = None
        app._main_window = None
        releases: List[int] = []
        app._on_tts_compile_priming_changed = lambda v: releases.append(v)

        release_first = asyncio.Event()

        async def first():
            await release_first.wait()

        async def second():
            return None

        t1 = asyncio.ensure_future(app._compile_warmup_entrypoint(first))
        await asyncio.sleep(0)
        await app._compile_warmup_entrypoint(second)
        after_second = list(releases)
        release_first.set()
        await t1
        return after_second, releases

    after_second, releases = asyncio.run(scenario())
    assert after_second == [], "the second body's exit must not release"
    assert releases == [False]


def test_an_overlapping_entrypoint_still_propagates_its_exception():
    """Review second pass: a ``return`` inside the entrypoint's ``finally``
    would have swallowed the exception of the earlier of two overlapping
    runs, so its failure never reached ``on_error``. The raise must
    propagate even when another run is still in flight."""

    async def scenario():
        from myvoice.app import MyVoiceApp

        app = MyVoiceApp.__new__(MyVoiceApp)
        app.logger = logging.getLogger("test-20-11")
        app._voice_clone_prompt_hydration_task = None
        app._main_window = None
        app._on_tts_compile_priming_changed = lambda v: None

        release_other = asyncio.Event()

        async def other():
            await release_other.wait()

        async def failing():
            raise RuntimeError("preload exploded")

        t_other = asyncio.ensure_future(app._compile_warmup_entrypoint(other))
        await asyncio.sleep(0)
        try:
            await app._compile_warmup_entrypoint(failing)
        except RuntimeError as exc:
            outcome = str(exc)
        else:
            outcome = "swallowed"
        release_other.set()
        await t_other
        return outcome

    assert asyncio.run(scenario()) == "preload exploded"
