"""Server lifecycle controller for the local TTS API.

Owns the FastAPI app + the uvicorn ``Server.serve()`` task scheduled on the
existing qasync loop (NOT a new thread). uvicorn runs with
``install_signal_handlers=False`` and ``reload=False`` because Qt owns signals
on the main thread.

Shutdown is bounded (≤2 s) so it fits comfortably inside the 8 s
``cleanup_async`` budget (main.py:435) and never trips the hard
``_os._exit(0)``. On stop we first cancel any in-flight streaming generation
tasks so no generation outlives the TTS service teardown (tech-spec G7/H2).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional, Set

from .app_factory import build_app
from .security import generate_api_key  # re-exported for callers (Task 10)

logger = logging.getLogger(__name__)

__all__ = ["ApiServerController", "generate_api_key"]

# Bounded stop budget; stays well under the 8 s cleanup_async wait_for window.
_STOP_TIMEOUT_SECONDS = 2.0


class ApiServerController:
    """Start/stop a uvicorn server on the shared qasync loop."""

    def __init__(self, tts_service, voice_manager, app_ref, settings_provider: Callable):
        self._tts_service = tts_service
        self._voice_manager = voice_manager
        self._app_ref = app_ref
        self._settings_provider = settings_provider

        self._app = None
        self._server = None
        self._task: Optional[asyncio.Task] = None
        self._host: str = "127.0.0.1"
        self._port: Optional[int] = None
        # Registry of in-flight streaming generation tasks (Task 8 registers
        # into this; stop() cancels them before tearing down the server).
        self._active_stream_tasks: Set[asyncio.Task] = set()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def url(self) -> Optional[str]:
        if self._port is None:
            return None
        return f"http://{self._host}:{self._port}"

    async def start(self, host: str = "127.0.0.1", port: int = 7778) -> None:
        """Build the app and run ``Server.serve()`` as a loop task.

        Binds ``127.0.0.1`` only (v1). Idempotent: a no-op if already running.
        """
        if self.is_running:
            logger.debug("ApiServerController.start() ignored; already running")
            return

        import uvicorn

        # Security-by-default enforced at the server layer, not only the UI
        # (F2): the API can be enabled via a hand-edited config that never went
        # through the panel's auto-key path. Warn loudly when coming up keyless
        # so an open localhost generation surface is never silent.
        try:
            current_key = (self._settings_provider().http_api_key or "")
        except Exception:  # pragma: no cover - defensive
            current_key = ""
        if not current_key:
            logger.warning(
                "Local TTS API is starting WITHOUT an API key — any local "
                "process or web page can drive generation. Set an API key in "
                "Settings > API Access (recommended)."
            )

        self._host = host
        self._port = port
        self._app = build_app(
            tts_service=self._tts_service,
            voice_manager=self._voice_manager,
            app_ref=self._app_ref,
            settings_provider=self._settings_provider,
            controller=self,
        )

        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="warning",
            reload=False,
            access_log=False,
        )
        config.install_signal_handlers = False  # Qt owns signals on main thread
        self._server = uvicorn.Server(config)

        self._task = asyncio.ensure_future(self._server.serve())

        # Await readiness so callers (and the Settings UI) see a live server.
        await self._await_started()
        logger.info("Local TTS API listening on %s", self.url)

    async def _await_started(self, timeout: float = 5.0) -> None:
        """Poll ``server.started`` until ready, or fail loudly on timeout/error.

        On a hung bind (server never reports ``started``) we tear the half-
        started task down and raise, rather than returning success for a wedged
        server that ``is_running`` would then report as live (F9).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._task is not None and self._task.done():
                # Surface a startup failure (e.g. port in use) to the caller.
                exc = self._task.exception()
                if exc is not None:
                    raise exc
                return
            if getattr(self._server, "started", False):
                return
            await asyncio.sleep(0.02)

        # Hung start: tear down so we don't leave a wedged "running" server.
        logger.error("Local TTS API did not report 'started' within %.1fs; tearing down", timeout)
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
        self._server = None
        self._port = None
        raise TimeoutError(f"Local TTS API did not start within {timeout:.1f}s")

    async def stop(self) -> None:
        """Cancel in-flight streams, then bounded-stop the uvicorn task."""
        # 1) Cancel every in-flight streaming gen_task first so no generation
        #    runs after the TTS service is torn down (G7).
        if self._active_stream_tasks:
            for task in list(self._active_stream_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._active_stream_tasks, return_exceptions=True)
            self._active_stream_tasks.clear()

        # 2) Ask uvicorn to exit, bounded by the stop budget.
        if self._server is not None:
            self._server.should_exit = True

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=_STOP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("uvicorn stop exceeded %.1fs; cancelling", _STOP_TIMEOUT_SECONDS)
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except asyncio.CancelledError:
                pass

        self._task = None
        self._server = None
        self._app = None
        self._port = None
        logger.info("Local TTS API stopped")
