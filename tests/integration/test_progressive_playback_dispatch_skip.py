"""Story 17.3 — integration tests for the dispatch-skip branch in
``_play_generated_audio`` (AC #2 step 5 / AC #6 / Task 6.3).

Verifies the surgical skip semantics:
  - When ``_progressive_playback_active`` is True, ``_play_generated_audio``
    skips ``play_dual_stream`` AND releases the queue slot so subsequent
    dispatches can advance — without this release the queue would stay
    stuck because no dual-fire ``_on_playback_complete`` fires.
  - When ``_progressive_playback_active`` is False, the existing batch
    dispatch path runs unchanged (Story 13.2 / 13.3 / 14.3 contracts).
  - The flag is consume-once: ``_play_generated_audio``'s skip-branch
    clears it so a subsequent generation re-arms via the normal chunk-0
    callback path.

The cached WAV file (Replay's source-of-truth per
``QwenTTSService.get_cached_audio_path()``) is written inside
``_generate_true_stream`` itself, so this test does not need to assert
WAV-file writing — it only needs to assert ``_play_generated_audio``'s
skip-branch is semantically correct.
"""

from __future__ import annotations

import asyncio
from typing import Optional
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
def app_with_queue(qapp, monkeypatch):
    """Construct a ``MyVoiceApp`` with a real ``PlaybackQueue`` + a stubbed
    registry.post_mutation. Mirrors the helper in
    ``tests/integration/test_session_lifecycle.py`` so the queue gating
    runs unmodified — exactly what AC #6 / Task 6.2 require to verify."""
    from myvoice.app import MyVoiceApp
    from myvoice.services.sessions import PlaybackQueue, SessionRegistry

    app = MyVoiceApp(qapp)
    app._session_registry = SessionRegistry(parent=app)
    app._playback_queue = PlaybackQueue(parent=app)
    monkeypatch.setattr(
        app._session_registry, "post_mutation", lambda *a, **kw: None
    )

    # Mocked AudioCoordinator surface — only the calls the dispatch path
    # makes are stubbed.
    coordinator = MagicMock()
    coordinator.play_dual_stream = AsyncMock(
        return_value=MagicMock(any_successful=True, monitor_task=None, virtual_task=None)
    )
    coordinator.monitor_service = MagicMock()
    coordinator.monitor_service.enumerate_monitor_devices = AsyncMock(
        return_value=[]
    )
    coordinator.monitor_service.play_monitor_audio = AsyncMock(return_value=None)
    coordinator.windows_audio_client = None
    app._audio_coordinator = coordinator

    # AppSettings is consulted for device-id preferences; provide a stub
    # with no virtual device id so the monitor-only fallback path is used.
    class _StubSettings:
        monitor_device_id: Optional[str] = None
        virtual_microphone_device_id: Optional[str] = None
        monitor_device_name = None
        monitor_device_host_api = None
        virtual_microphone_device_name = None
        virtual_microphone_device_host_api = None

    app._app_settings = _StubSettings()
    return app, coordinator


def _drive(app, audio_bytes):
    """Run ``_play_generated_audio`` end-to-end on a fresh asyncio loop.
    Caller is expected to set ``_progressive_playback_active`` before
    invocation if the skip branch is the target.
    """
    return asyncio.run(app._play_generated_audio(audio_bytes))


class TestProgressiveDispatchSkip:
    """AC #2 step 5 / AC #6 — skip-branch preserves queue continuity and
    consumes the flag once."""

    def test_progressive_active_skips_play_dual_stream(self, app_with_queue):
        app, coordinator = app_with_queue
        app._progressive_playback_active = True

        _drive(app, b"some-wav-bytes")

        # play_dual_stream must NOT fire — audio already played progressively.
        coordinator.play_dual_stream.assert_not_awaited()
        # Monitor-only fallback also does not fire.
        coordinator.monitor_service.play_monitor_audio.assert_not_awaited()
        # Flag consumed by the skip-branch — next generation re-arms cleanly.
        assert app._progressive_playback_active is False
        # Queue slot was released so subsequent dispatches can advance
        # (verified by _dispatching_session_id being None).
        assert app._dispatching_session_id is None
        assert app._playback_queue.depth == 0

    def test_progressive_inactive_runs_existing_dispatch(self, app_with_queue):
        app, coordinator = app_with_queue
        # Default: _progressive_playback_active is False from __init__.
        assert app._progressive_playback_active is False

        _drive(app, b"some-wav-bytes")

        # The monitor-only fallback path runs because the stub settings
        # have no virtual device id and enumerate returns []. The
        # important contract is that the skip-branch did NOT fire.
        # Assert via "play_dual_stream is never reached because the
        # virtual_device_id branch is gated on virtual_device_id being
        # truthy" — i.e., the existing branching is exercised.
        coordinator.play_dual_stream.assert_not_awaited()
        coordinator.monitor_service.enumerate_monitor_devices.assert_awaited_once()
        # Flag stayed False (never touched).
        assert app._progressive_playback_active is False

    def test_progressive_skip_does_not_block_subsequent_dispatch(
        self, app_with_queue
    ):
        """After a progressive-skip dispatch, a SECOND _play_generated_audio
        call (e.g. user clicks Replay) must still be able to run through
        the batch path — proves the queue slot was released cleanly."""
        app, coordinator = app_with_queue

        # First call: progressive-skip path.
        app._progressive_playback_active = True
        _drive(app, b"first-bytes")
        coordinator.play_dual_stream.assert_not_awaited()
        assert app._dispatching_session_id is None

        # Second call: batch path (flag now False after consume).
        _drive(app, b"second-bytes")
        # Monitor enumeration ran (proving the dispatch path was entered)
        # — even though no devices are returned, the path was reachable.
        assert (
            coordinator.monitor_service.enumerate_monitor_devices.await_count
            >= 1
        )
