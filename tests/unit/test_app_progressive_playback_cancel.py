"""Story 17.3 — orchestrator-level progressive-playback cancel chain tests.

Covers AC #4 (Task 4.3):
  - ``test_cancel_mid_stream_stops_streaming_session`` — when
    ``_progressive_playback_active is True``, ``_on_cancel_generation_requested``
    schedules ``AudioCoordinator.stop_streaming_session()`` and clears the
    flag so a subsequent generation opens a fresh session.
  - ``test_cancel_when_progressive_inactive_no_extra_call`` — when the
    flag is False, no ``stop_streaming_session`` schedule fires (defends
    against double-stop on an already-closed session).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def app_with_mocked_services(qapp):
    """Construct a ``MyVoiceApp`` with mocked TTS service + audio
    coordinator so ``_on_cancel_generation_requested`` can run end-to-end
    without touching real services."""
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(qapp)

    tts_service = MagicMock()
    tts_service.cancel_generation = AsyncMock()
    app._tts_service = tts_service

    coordinator = MagicMock()
    coordinator.stop_all_playback = AsyncMock(return_value={})
    coordinator.stop_streaming_session = AsyncMock(
        return_value={"monitor": True, "virtual": True}
    )
    app._audio_coordinator = coordinator

    # _on_cancel_generation_requested calls asyncio.ensure_future(...) so
    # we need a loop. The handler also touches _session_registry — leave
    # it as None so the registry-close branch is short-circuited.
    yield app, tts_service, coordinator


def _drive_cancel(app):
    """Run the synchronous cancel handler on a fresh asyncio loop. The
    handler schedules fire-and-forget coros via ``asyncio.ensure_future``;
    we close the loop after to suppress 'coro was never awaited' warnings
    while keeping the call/await assertions on the mocks intact.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        app._on_cancel_generation_requested()
        # Drain any scheduled coroutines so AsyncMock.assert_awaited works.
        pending = asyncio.all_tasks(loop=loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class TestProgressivePlaybackCancel:
    """Story 17.3 AC #4 — cancel chain integration."""

    def test_cancel_mid_stream_stops_streaming_session(
        self, app_with_mocked_services
    ):
        app, tts_service, coordinator = app_with_mocked_services
        app._progressive_playback_active = True
        app._progressive_playback_sample_rate = 24000

        _drive_cancel(app)

        # Existing chain fires unchanged: tts cancel + stop_all_playback.
        tts_service.cancel_generation.assert_awaited_once()
        coordinator.stop_all_playback.assert_awaited_once()
        # Story 17.3 additive call: stop_streaming_session fires when
        # progressive playback was active.
        coordinator.stop_streaming_session.assert_awaited_once()
        # Flag cleared so a subsequent generation re-opens fresh.
        assert app._progressive_playback_active is False

    def test_cancel_when_progressive_inactive_no_extra_call(
        self, app_with_mocked_services
    ):
        app, tts_service, coordinator = app_with_mocked_services
        # Default: _progressive_playback_active is False from __init__.
        assert app._progressive_playback_active is False

        _drive_cancel(app)

        # Existing chain still fires.
        tts_service.cancel_generation.assert_awaited_once()
        coordinator.stop_all_playback.assert_awaited_once()
        # No double-stop — no streaming session was open.
        coordinator.stop_streaming_session.assert_not_awaited()
        assert app._progressive_playback_active is False
