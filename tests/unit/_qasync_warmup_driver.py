"""Story 20.3 AC #1 — out-of-process driver for the qasync startup-window test.

Run as ``python tests/unit/_qasync_warmup_driver.py <variant>``; prints one JSON
object on stdout. Not collected by pytest (no ``test_`` prefix) — it is spawned
by ``test_app_compile_warmup_qasync.py``.

**Why a subprocess.** The hazard is process-global: it needs a real
``QApplication``, a real ``qasync.QEventLoop`` driving it via
``QApplication.exec()``, and a task stepped from inside a nested
``processEvents()``. Standing that up inside the shared pytest Qt session hangs
the suite (``exec()`` does not return cleanly once other modules have created Qt
state). A fresh process is both safer and a more faithful reproduction of a
launch.

Variants:

  ``fixed``  — the shipped hand-off, ``MyVoiceApp._run_async_task_when_loop_is_idle``
  ``plain``  — a plain ``_run_async_task`` at the same point (the shape that
               shipped broken after the ordering fix)
  ``shield`` — ``wait_for(shield(hydration_task))`` first (the shape Story 20.3
               originally shipped)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import torch  # noqa: F401  — torch-before-PyQt6 DLL ordering (see memory)

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

import qasync

from myvoice.models.app_settings import AppSettings
from myvoice.models.service_enums import QwenModelType
from myvoice.observability import metrics
from myvoice.services.qwen_tts_service import QwenTTSService


PUMP_ITERATIONS = 40


def _fake_loaded_model(model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    inner = type("FakeInner", (), {})()
    inner.name_or_path = model_id
    inner.dtype = torch.bfloat16
    outer = type("FakeModel", (), {})()
    outer.model = inner
    return outer


def _make_tts_service() -> QwenTTSService:
    """A real QwenTTSService whose warmup reaches ``primed_warm``."""
    service = QwenTTSService(
        device="cpu", dtype="float32", app_settings=AppSettings(tts_compile="auto")
    )
    registry = MagicMock(name="ModelRegistry")
    registry.get_loaded_model.return_value = _fake_loaded_model()
    registry.current_model_type = QwenModelType.BASE
    service._model_registry = registry

    from myvoice.services import tts_streaming
    from myvoice.services.tts_streaming import compile_cache

    torch.cuda.is_available = lambda: True  # type: ignore[assignment]
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)  # type: ignore[assignment]
    tts_streaming.is_ampere_or_newer = lambda: True  # type: ignore[assignment]
    compile_cache.is_warm = lambda key: True  # type: ignore[assignment]

    async def _priming():
        # Suspend repeatedly, like the real dispatch's executor hand-offs and
        # the talker's ``await asyncio.sleep(poll_s)`` poll. Each resumption is
        # another chance for the loop to re-enter this task, so the task must
        # survive many steps for the metric to land.
        for _ in range(5):
            await asyncio.sleep(0)
        service._last_priming_model_type = QwenModelType.BASE

    service._run_compile_priming = _priming  # type: ignore[assignment]
    return service


def _bare_app():
    from myvoice.app import MyVoiceApp

    app_obj = MyVoiceApp.__new__(MyVoiceApp)
    app_obj.logger = logging.getLogger("qasync-warmup-driver")
    app_obj._voice_clone_prompt_hydration_task = None
    return app_obj


def main(variant: str) -> Dict[str, Any]:
    qt_app = QApplication.instance() or QApplication([])

    records: List[Any] = []
    loop_errors: List[str] = []
    unsub = metrics.add_listener(records.append)

    loop = qasync.QEventLoop(qt_app)
    asyncio.set_event_loop(loop)

    def _handler(_loop, context):
        exc = context.get("exception")
        loop_errors.append(
            f"{type(exc).__name__}: {exc}" if exc else str(context.get("message"))
        )

    loop.set_exception_handler(_handler)

    tts = _make_tts_service()
    app_obj = _bare_app()
    app_obj._tts_service = tts

    async def hydrate():
        # Mirrors hydrate_voice_clone_prompt_cache: an ``async def`` with a
        # fully synchronous body, so it completes in a single step.
        return (13, 14)

    async def preload_model():
        await asyncio.sleep(0.05)

    async def task1():
        # --- the real startup shape, in order -------------------------------
        app_obj._voice_clone_prompt_hydration_task = app_obj._run_async_task(
            hydrate()
        )
        await preload_model()

        if variant == "fixed":
            app_obj._run_async_task_when_loop_is_idle(tts.warmup_compile_async)
        elif variant == "plain":
            app_obj._run_async_task(tts.warmup_compile_async())
        elif variant == "shield":
            async def wrapper():
                task = app_obj._voice_clone_prompt_hydration_task
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=120.0)
                await tts.warmup_compile_async()

            app_obj._run_async_task(wrapper())
        else:
            raise SystemExit(f"unknown variant {variant!r}")

        # main.py:397 — splash.showMessage(...) / processEvents(), run
        # SYNCHRONOUSLY inside Task-1. This is the window that kills the task.
        for _ in range(PUMP_ITERATIONS):
            QCoreApplication.processEvents()

        # ``await app_close.wait()`` — Task-1 parks, the loop goes idle.
        await asyncio.sleep(0.5)

    try:
        with loop:
            loop.run_until_complete(task1())
    finally:
        unsub()

    warm = [r for r in records if r.name == "tts_compile_warmup_priming"]
    return {
        "variant": variant,
        "reasons": [r.tags.get("reason") for r in warm],
        "metric_count": len(warm),
        "reentrancy_errors": [e for e in loop_errors if "Cannot enter into task" in e],
        "all_errors": loop_errors,
    }


if __name__ == "__main__":
    result = main(sys.argv[1] if len(sys.argv) > 1 else "fixed")
    print("__RESULT__" + json.dumps(result))
