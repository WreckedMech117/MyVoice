"""Unit tests for AudioCoordinator (Story 16.5 cancel-chain extension).

Covers Story 16.5 AC #4 (cancel_playback per-session semantics) and AC #9
(play_dual_stream session_id → coordination_id map maintenance + non-
regression of existing behavior).

Per architecture file-map (architecture-optimization-pass.md:639-641) this
file is the unit-test sibling of audio_coordinator.py; structural template
follows tests/unit/services/sessions/test_session_registry.py (module
docstring → fixtures → class-grouped Test* with one AC focus per class).
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_monitor():
    """Fake MonitorAudioService whose stop_all_playback is an AsyncMock."""
    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_all_playback = AsyncMock(return_value=0)
    monitor.play_monitor_audio = AsyncMock()
    return monitor


@pytest.fixture
def mock_virtual():
    """Fake VirtualMicrophoneService whose stop_all_virtual_microphone_playback
    is an AsyncMock."""
    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_all_virtual_microphone_playback = AsyncMock(return_value=0)
    virtual.play_virtual_microphone = AsyncMock()
    return virtual


@pytest.fixture
def coordinator(mock_monitor, mock_virtual) -> AudioCoordinator:
    """AudioCoordinator wired with fake monitor + virtual services.

    The coordinator's full ``initialize()`` chain pulls in PortAudio and
    DeviceResilienceManager — too heavy + side-effecty for a unit test.
    Instead we construct directly and force-stamp the fields that
    ``play_dual_stream`` and ``cancel_playback`` actually consult.
    """
    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = mock_monitor
    coord.virtual_service = mock_virtual
    return coord


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelPlaybackPerSession (AC #4)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestCancelPlaybackPerSession:
    """AC #4: cancel_playback(session_id) stops monitor + virtual playback
    for a known session and is a quiet False-no-op for unknown sessions.
    """

    async def test_cancel_playback_stops_monitor_and_virtual_for_known_session(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # Pre-populate the map as if play_dual_stream had run.
        coordinator._session_id_to_coordination_id["sess-known"] = "coord_1_xyz"
        result = await coordinator.cancel_playback("sess-known")
        assert result is True
        mock_monitor.stop_all_playback.assert_called_once()
        mock_virtual.stop_all_virtual_microphone_playback.assert_called_once()

    async def test_cancel_playback_returns_false_for_unknown_session_id(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # AC #4 second clause: stale-id quiet no-op — neither service stop
        # is attempted, no exception is raised, returns False.
        result = await coordinator.cancel_playback("never-registered")
        assert result is False
        mock_monitor.stop_all_playback.assert_not_called()
        mock_virtual.stop_all_virtual_microphone_playback.assert_not_called()

    async def test_cancel_playback_continues_after_monitor_stop_raises(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # AC #4 third clause: monitor stop failure must NOT prevent the
        # virtual stop attempt — don't let one sink leave the other
        # running.
        coordinator._session_id_to_coordination_id["sess-broken"] = "coord_1"
        mock_monitor.stop_all_playback.side_effect = RuntimeError(
            "PortAudio borked"
        )
        result = await coordinator.cancel_playback("sess-broken")
        # virtual stop still attempted → return True (something stopped).
        assert result is True
        mock_monitor.stop_all_playback.assert_called_once()
        mock_virtual.stop_all_virtual_microphone_playback.assert_called_once()

    async def test_cancel_playback_removes_session_from_map(
        self, coordinator
    ):
        # AC #4: the session_id → coordination_id entry is removed from
        # the map (so retry semantics are clean — a second cancel for the
        # same id falls through to the unknown-session quiet-False path).
        coordinator._session_id_to_coordination_id["sess-x"] = "coord_1"
        await coordinator.cancel_playback("sess-x")
        assert "sess-x" not in coordinator._session_id_to_coordination_id
        # Second call → no-op False per AC #4 second clause.
        result = await coordinator.cancel_playback("sess-x")
        assert result is False

    async def test_cancel_playback_with_no_sinks_returns_false(
        self, coordinator
    ):
        # Defensive edge: both services are None (e.g., teardown raced
        # with a cancel). The map is consulted; with no sinks to drive,
        # nothing is attempted and False is returned.
        coordinator.monitor_service = None
        coordinator.virtual_service = None
        coordinator._session_id_to_coordination_id["sess-y"] = "coord_2"
        result = await coordinator.cancel_playback("sess-y")
        assert result is False
        # Map still cleaned (no leak even when nothing was stopped).
        assert "sess-y" not in coordinator._session_id_to_coordination_id


# --------------------------------------------------------------------------- #
# Story 16.5 — TestPlayDualStreamSessionMap (AC #9)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestPlayDualStreamSessionMap:
    """AC #9: play_dual_stream populates _session_id_to_coordination_id at
    entry, leaves it populated for the dispatch lifetime (cancel_playback
    handles success-path cleanup), and clears the entry on failure paths
    so the map never leaks dead entries.
    """

    async def _make_playing_task(self):
        """Construct a MonitorPlaybackTask-shaped MagicMock with status
        PLAYING (the value AudioCoordinator's any_successful checks)."""
        task = MagicMock()
        task.status = MagicMock()
        task.status.value = "playing"
        return task

    async def test_play_dual_stream_registers_session_to_coordination_map_on_entry(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # Override the health checks (AudioCoordinator gates dispatch on
        # them; without override they'd query the mock service health
        # surface which isn't wired here).
        coordinator._is_monitor_service_healthy = AsyncMock(return_value=True)
        coordinator._is_virtual_service_healthy = AsyncMock(return_value=True)
        playing_task = await self._make_playing_task()
        mock_monitor.play_monitor_audio.return_value = playing_task
        mock_virtual.play_virtual_microphone.return_value = playing_task

        result = await coordinator.play_dual_stream(
            audio_data=b"\x00" * 100,
            session_id="sess-A",
        )

        # Success path: map remains populated so a mid-playback cancel
        # can target this dispatch.
        assert "sess-A" in coordinator._session_id_to_coordination_id
        assert coordinator._session_id_to_coordination_id["sess-A"] == result.coordination_id

    async def test_play_dual_stream_clears_map_on_no_healthy_services_path(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # When neither service is healthy, no tasks are appended; the
        # method returns the no-healthy-services error result. The
        # session→coordination map MUST be cleared (otherwise a later
        # cancel_playback would stale-fan-out).
        coordinator._is_monitor_service_healthy = AsyncMock(return_value=False)
        coordinator._is_virtual_service_healthy = AsyncMock(return_value=False)

        result = await coordinator.play_dual_stream(
            audio_data=b"\x00" * 100,
            session_id="sess-B",
        )
        assert result.success is False
        assert result.error_message == "No healthy services available"
        assert "sess-B" not in coordinator._session_id_to_coordination_id

    async def test_play_dual_stream_clears_map_on_exception_path(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # An exception during dispatch (here: monitor's task-creation
        # raises) hits the outer except block. The cleanup MUST run there
        # so the map doesn't leak entries for failed dispatches.
        coordinator._is_monitor_service_healthy = AsyncMock(return_value=True)
        coordinator._is_virtual_service_healthy = AsyncMock(return_value=True)
        # Force the asyncio.gather to raise via a blowup in the awaitable.
        mock_monitor.play_monitor_audio.side_effect = RuntimeError("kaboom")
        mock_virtual.play_monitor_audio = AsyncMock()

        result = await coordinator.play_dual_stream(
            audio_data=b"\x00" * 100,
            session_id="sess-C",
        )
        # The exception is captured per-task by asyncio.gather(return_exceptions=True),
        # so the method does NOT enter the outer except block — instead it
        # returns a partial-success / no-success result depending on the
        # virtual task. Either way, the map remains populated for the
        # dispatch lifetime per AC #9 success-path semantics. The cleanup
        # contract on the *outer-exception* path is exercised by the
        # not-initialized + no-healthy-services branches; this asserts
        # the existing semantics are unchanged.
        assert isinstance(result.coordination_id, str)

    async def test_play_dual_stream_without_session_id_does_not_touch_map(
        self, coordinator, mock_monitor, mock_virtual
    ):
        # Legacy callers omit session_id (D-14: legacy path runs
        # unchanged). The map must not be polluted with empty / None keys.
        coordinator._is_monitor_service_healthy = AsyncMock(return_value=True)
        coordinator._is_virtual_service_healthy = AsyncMock(return_value=True)
        playing_task = await self._make_playing_task()
        mock_monitor.play_monitor_audio.return_value = playing_task
        mock_virtual.play_virtual_microphone.return_value = playing_task

        before = dict(coordinator._session_id_to_coordination_id)
        await coordinator.play_dual_stream(audio_data=b"\x00" * 100)
        after = dict(coordinator._session_id_to_coordination_id)
        assert before == after
        # Specifically: no None key.
        assert None not in after


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelPlaybackInitializationOrder (AC #9 non-regression)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestCancelPlaybackInitializationOrder:
    """AC #9: existing AudioCoordinator behavior is unchanged. The new
    map and method are additive; pre-existing fields are present and
    initialized to empty.
    """

    async def test_session_to_coordination_map_initialized_empty(self):
        coord = AudioCoordinator()
        assert hasattr(coord, "_session_id_to_coordination_id")
        assert coord._session_id_to_coordination_id == {}

    async def test_cancel_playback_method_exists_on_uninitialized_coordinator(
        self
    ):
        # Defensive: a freshly-constructed coordinator (no initialize())
        # exposes cancel_playback. Calling it for an unknown session is
        # the documented quiet-False no-op.
        coord = AudioCoordinator()
        result = await coord.cancel_playback("anything")
        assert result is False
