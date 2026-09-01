"""Story 20.3 AC #1 — the compile warmup must survive the qasync startup window.

**Why this file exists.** Story 20.3's first AC #1 fix passed every unit test
and still did nothing in the shipped app. The AC #4 GUI capture came back
negative twice: segment 1b stayed at ~3,607 ms (Story 20.2's *before* number)
and **zero** ``tts_compile_warmup_priming`` rows appeared in any CSV — a metric
recorded on every exit path of ``warmup_compile_async``, so the body had never
executed at all. ``logs\\myvoice.log`` carried the reason::

    qasync._QEventLoop - ERROR - Exception in callback <TaskStepMethWrapper>()
    RuntimeError: Cannot enter into task <Task pending name='Task-6' ...>
      while another task <Task pending name='Task-1'
      coro=<async_main() running at main.py:397>> is being executed.
    Task was destroyed but it is pending!

**The mechanism.** Under qasync, ``call_soon`` is not a ready-queue append — it
is ``QObject.startTimer(0)`` (``qasync/__init__.py``: ``call_soon`` ->
``call_later(0, ...)`` -> ``_SimpleTimer.add_callback`` -> ``startTimer``). The
queued task step is therefore delivered by ``timerEvent`` during **any** Qt
event processing, including the synchronous ``splash.showMessage(...)`` /
``processEvents()`` stretch that ``main.py`` runs *inside* Task-1 immediately
after ``initialize_async()`` returns — ``main.py:397`` is exactly that line.
``asyncio._enter_task`` refuses to enter a second task while Task-1 is mid-step,
the loop's exception handler swallows the RuntimeError, and the task dies having
run nothing.

**Why the ordinary unit tests could not catch it.** They run on a plain
``asyncio`` loop, where ``call_soon`` appends to a ready queue drained only
between task steps. The hazard is structurally absent there. The tests were
green and the feature was dead — so this file stands the *real* loop up.

**Why out of process.** The hazard needs a real ``QApplication``, a real
``qasync.QEventLoop`` driving it through ``QApplication.exec()``, and a task
stepped from inside a nested ``processEvents()``. Standing that up inside the
shared pytest Qt session hangs the whole suite (``exec()`` does not return
cleanly once other modules have created Qt state — measured: ``tests/unit`` went
from 56 s to >600 s). Each row therefore spawns
``tests/unit/_qasync_warmup_driver.py`` in a fresh interpreter, which is also a
more faithful reproduction of a launch.

Rows:

  1. ``test_warmup_reaches_the_metric_under_a_real_qasync_loop`` — the shipped
     hand-off must record ``tts_compile_warmup_priming`` with a real reason.
  2. ``test_the_plain_hand_off_is_destroyed_under_the_same_loop`` — the
     non-vacuity control, and the row that would have caught the bug: the same
     rig scheduling the same coroutine through the plain ``_run_async_task``
     must be destroyed and record NOTHING.
  3. ``test_shield_wait_for_hand_off_is_also_destroyed`` — the exact shape
     Story 20.3 first shipped, pinned so nobody reintroduces it.

Note that row 1 exercises the *mechanism*. That the shipped call site actually
uses it is asserted separately by
``test_app_compile_warmup_sequencing.py::test_warmup_hand_off_uses_the_qasync_safe_scheduler``
— a mutation that reverted the call site escaped this file entirely, because
these rows drive the scheduler helper directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("qasync")


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().parent / "_qasync_warmup_driver.py"


def _run_driver(variant: str) -> Dict[str, Any]:
    assert DRIVER.exists(), f"driver missing at {DRIVER}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("MYVOICE_DISABLE_COMPILE_WARMUP", None)
    env.pop("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", None)

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
def fixed_result() -> Dict[str, Any]:
    return _run_driver("fixed")


@pytest.fixture(scope="module")
def plain_result() -> Dict[str, Any]:
    return _run_driver("plain")


@pytest.fixture(scope="module")
def shield_result() -> Dict[str, Any]:
    return _run_driver("shield")


# --------------------------------------------------------------------------- #


def test_warmup_reaches_the_metric_under_a_real_qasync_loop(fixed_result):
    """AC #1 — the shipped hand-off survives the Qt-pumping startup window.

    This is the assertion the AC #4 capture made against the shipped app: a
    launch must produce a ``tts_compile_warmup_priming`` row. Zero rows means
    the warmup body never ran, which is exactly what both negative GUI passes
    showed.
    """
    assert fixed_result["metric_count"] == 1, (
        "warmup_compile_async did not record tts_compile_warmup_priming exactly "
        "once under a real qasync loop; its body did not run to completion. "
        f"errors={fixed_result['all_errors']}"
    )
    assert fixed_result["reasons"] == ["primed_warm"], (
        f"expected a real telemetry reason, got {fixed_result['reasons']}"
    )
    assert fixed_result["reentrancy_errors"] == [], (
        f"the loop reported a task re-entrancy error: "
        f"{fixed_result['reentrancy_errors']}"
    )


def test_the_plain_hand_off_is_destroyed_under_the_same_loop(plain_result):
    """Non-vacuity control — and the row that would have caught the bug.

    The same rig, scheduling the same coroutine through the plain
    ``_run_async_task`` at the same point, must be destroyed by the qasync
    re-entrancy guard and record nothing. If this row ever goes green, the rig
    has stopped reproducing the hazard and row 1 proves nothing.
    """
    assert plain_result["metric_count"] == 0, (
        "the plain hand-off recorded a metric, so this rig no longer reproduces "
        "the qasync re-entrancy hazard and the row above is vacuous"
    )
    assert plain_result["reentrancy_errors"], (
        "expected the qasync re-entrancy RuntimeError; got "
        f"{plain_result['all_errors']}"
    )


def test_shield_wait_for_hand_off_is_also_destroyed(shield_result):
    """The exact shape Story 20.3 first shipped, pinned as a regression.

    The first fix wrapped the warmup in a coroutine whose first statement was
    ``await asyncio.wait_for(asyncio.shield(hydration_task), timeout=...)``.
    Recorded here so nobody reintroduces it believing the extra task machinery
    is harmless — and to document that removing ``shield``/``wait_for`` alone
    does NOT fix the bug (see the row above, which has neither).
    """
    assert shield_result["metric_count"] == 0
    assert shield_result["reentrancy_errors"]
