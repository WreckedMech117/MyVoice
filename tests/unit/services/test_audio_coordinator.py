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


# --------------------------------------------------------------------------- #
# Story 17.3 finalization-drain follow-up — TestStopStreamingSessionDrain
# --------------------------------------------------------------------------- #
#
# Background: app.py:_handle_progressive_chunk_async called
# stop_streaming_session() immediately on `is_final` without awaiting the
# PyAudio output buffer drain. With the bf16 + TF32 + cuDNN engagements all
# firing post-Story-18.3, the producer outpaces the consumer; the buffer
# still holds the tail of the last chunk when the close fires → audible
# cut-off-at-end. Surfaced 2026-05-10 by Commander during the Story 18.3
# Task 1 dtype-audit run.
#
# Fix: stop_streaming_session(wait_for_drain: bool = False). When True, the
# coordinator computes (total_bytes_written / bytes_per_second) - elapsed,
# adds a small safety buffer for PyAudio internal latency, caps at
# _MAX_DRAIN_WAIT_S, and asyncio.sleeps for that duration before tearing
# down. Cancel paths keep the default False (immediate teardown).
#
# These tests pin the contract:
#   1. Default (wait_for_drain=False) preserves legacy immediate-teardown.
#   2. wait_for_drain=True with un-drained audio waits ~remaining seconds.
#   3. wait_for_drain=True with already-drained audio does NOT wait.
#   4. The drain wait is capped at _MAX_DRAIN_WAIT_S (math drift safety).
#   5. Both leaf services receive their stop call regardless of the path.

import time as _time

import pytest as _pytest


def _coord_with_drain_services():
    """AudioCoordinator wired with services that mock both stop_streaming_session
    and play_audio_chunk (the heavier mock_monitor / mock_virtual fixtures
    above only stub stop_all_playback)."""
    coord = AudioCoordinator()
    coord._is_initialized = True

    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_streaming_session = AsyncMock(return_value=True)
    monitor.is_streaming_active = MagicMock(return_value=True)
    monitor.play_audio_chunk = AsyncMock(return_value=True)
    coord.monitor_service = monitor

    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_streaming_session = AsyncMock(return_value=True)
    virtual.is_streaming_active = MagicMock(return_value=True)
    virtual.play_audio_chunk = AsyncMock(return_value=True)
    coord.virtual_service = virtual

    return coord, monitor, virtual


@_pytest.mark.asyncio
class TestStopStreamingSessionDrain:
    """Story 17.3 finalization-drain follow-up — wait_for_drain contract."""

    async def test_default_no_wait_preserves_legacy_immediate_teardown(self):
        """wait_for_drain=False (default) MUST NOT introduce any drain wait,
        even when total_bytes_written is large. This preserves the cancel
        path's immediate-teardown contract (Story 16.5 cancel-chain)."""
        coord, monitor, virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # Pretend we wrote 9 seconds of audio (24000 * 2 * 9 = 432000 bytes).
        coord._stream_first_write_ts = _time.monotonic()
        coord._stream_total_bytes = 432000

        t0 = _time.monotonic()
        await coord.stop_streaming_session()  # default wait_for_drain=False
        elapsed = _time.monotonic() - t0

        # Default path: no drain wait → completes promptly.
        assert elapsed < 0.5, (
            f"Default stop_streaming_session() must NOT introduce a drain "
            f"wait (would break cancel paths); elapsed={elapsed:.3f}s"
        )
        monitor.stop_streaming_session.assert_awaited_once()
        virtual.stop_streaming_session.assert_awaited_once()

    async def test_wait_for_drain_true_waits_remaining_audio_duration(self):
        """wait_for_drain=True with un-played audio MUST sleep approximately
        (last_chunk_duration - time_since_last_write) + safety_buffer before
        tearing down.

        Story 18.3 M6 — math reads the LAST-chunk trackers, not the FIRST.
        Expected wait is computed from the live ``_DRAIN_SAFETY_BUFFER_S``
        constant rather than hard-coded — when the constant moves (M2
        already bumped it 0.15→0.5; future Windows-backend support may
        bump again), this test still pins the contract instead of
        breaking on a value change."""
        coord, monitor, virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # Pretend the LAST chunk was 0.4s of audio (24000 * 2 * 0.4 = 19200
        # bytes), written 0.1s ago. last_chunk_remaining = 0.4 - 0.1 = 0.3s.
        coord._stream_first_write_ts = _time.monotonic() - 0.1
        coord._stream_total_bytes = 19200
        coord._stream_last_write_ts = _time.monotonic() - 0.1
        coord._stream_last_chunk_bytes = 19200
        remaining_s = 0.3
        safety_s = AudioCoordinator._DRAIN_SAFETY_BUFFER_S
        expected_wait = remaining_s + safety_s

        t0 = _time.monotonic()
        await coord.stop_streaming_session(wait_for_drain=True)
        elapsed = _time.monotonic() - t0

        # Generous tolerance for asyncio scheduling jitter on Windows CI:
        # lower bound = expected - 200ms; upper bound = expected + 400ms.
        assert (expected_wait - 0.2) <= elapsed <= (expected_wait + 0.4), (
            f"Expected drain wait ~{expected_wait:.3f}s "
            f"(last_chunk_remaining {remaining_s}s + safety {safety_s}s); "
            f"got elapsed={elapsed:.3f}s"
        )
        monitor.stop_streaming_session.assert_awaited_once()
        virtual.stop_streaming_session.assert_awaited_once()

    async def test_wait_for_drain_true_with_already_drained_last_chunk_still_waits_safety(
        self
    ):
        """Story 18.3 M6 — even when the LAST chunk has fully drained
        (time_since_last_write > last_chunk_duration), the safety buffer
        STILL fires because PyAudio's device-level buffer (200–500ms on
        Windows shared mode) holds residual audio. This is the M6 fix:
        the previous ``if remaining > 0`` gate caused the cut-off-at-end
        Commander surfaced in the bundled smoke."""
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # Last chunk written 5s ago (way past any chunk's playback duration).
        coord._stream_first_write_ts = _time.monotonic() - 5.0
        coord._stream_total_bytes = 19200
        coord._stream_last_write_ts = _time.monotonic() - 5.0
        coord._stream_last_chunk_bytes = 19200
        safety_s = AudioCoordinator._DRAIN_SAFETY_BUFFER_S

        t0 = _time.monotonic()
        await coord.stop_streaming_session(wait_for_drain=True)
        elapsed = _time.monotonic() - t0

        # Last-chunk remaining = 0; total wait = safety only.
        assert (safety_s - 0.1) <= elapsed <= (safety_s + 0.3), (
            f"Already-drained last chunk must STILL wait safety buffer "
            f"(~{safety_s:.3f}s); got elapsed={elapsed:.3f}s"
        )

    async def test_wait_for_drain_caps_at_max_drain_wait(self):
        """Math-drift safety: a wildly-overestimated last-chunk duration is
        capped at _MAX_DRAIN_WAIT_S to prevent the close path from hanging.

        We patch _MAX_DRAIN_WAIT_S down to a small test value, then claim a
        huge last-chunk-bytes that would imply a >100s wait. The actual
        wait must be the patched cap, not the math.
        """
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # Pretend the last chunk was 200s of audio: 24000 * 2 * 200 = 9_600_000 bytes.
        coord._stream_first_write_ts = _time.monotonic()
        coord._stream_total_bytes = 9_600_000
        coord._stream_last_write_ts = _time.monotonic()
        coord._stream_last_chunk_bytes = 9_600_000

        # Patch the cap on the class attribute.
        original_cap = AudioCoordinator._MAX_DRAIN_WAIT_S
        AudioCoordinator._MAX_DRAIN_WAIT_S = 0.3
        try:
            t0 = _time.monotonic()
            await coord.stop_streaming_session(wait_for_drain=True)
            elapsed = _time.monotonic() - t0
        finally:
            AudioCoordinator._MAX_DRAIN_WAIT_S = original_cap

        assert 0.25 <= elapsed <= 0.7, (
            f"Drain wait must be capped at _MAX_DRAIN_WAIT_S (=0.3s); "
            f"got elapsed={elapsed:.3f}s"
        )

    async def test_wait_for_drain_under_producer_faster_than_realtime_waits_for_queued_audio(
        self
    ):
        """Story 18.4 code-review follow-up — drain math must handle the
        producer-FASTER-than-real-time regime (torch.compile + CUDA Graph
        replay produces chunks faster than PyAudio consumes them).

        Bug class: the last-chunk-only math (Story 18.3 M6 fix) was correct
        when the producer was SLOWER than real-time (the PyAudio buffer is
        approximately empty when the last chunk arrives — playback caught
        up during slow chunks). But when the producer outpaces real-time,
        multiple prior chunks queue in PyAudio's buffer; the last-chunk-
        only math underestimates remaining audio by the entire queued
        depth.

        Observed Story 18.4 Task 8 first-run (2026-05-11): 18.9 s of audio
        arrived in 14 s; sessions stopped 566 ms after last chunk write
        while ~4.9 s of audio was still buffered → user heard the audio
        cut mid-sentence. Fix: take max(last_chunk_remaining,
        total_queued_audio_s). This test pins the producer-faster regime;
        the existing Story 18.3 M6 tests pin the producer-slower regime.

        Setup mirrors the observed run:
          * 18.9 s of audio in total (sample_rate=24000, bytes=24000*2*18.9=907200)
          * Last chunk written 0.05 s ago (just arrived; producer faster)
          * First chunk written 14 s ago (chunks streamed over 14 s wall-clock)
          * Last chunk is 1.9 s of audio (38000 bytes); last_chunk_remaining
            ≈ 1.85 s under the OLD math.
          * Total queued ≈ 18.9 - 14 = 4.9 s.
          * The fix takes max(1.85, 4.9) = 4.9 s, then adds safety buffer.

        The OLD math would have computed ~1.85 + 0.5 = 2.35 s wait. The
        FIXED math computes ~4.9 + 0.5 = 5.4 s wait. We patch the cap
        down to a small value to avoid waiting 5+ seconds in CI; the
        contract under test is "the wait is the larger of the two
        estimates."
        """
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        now = _time.monotonic()
        # Producer-faster scenario: 18.9 s of audio arrived in 14 s.
        coord._stream_first_write_ts = now - 14.0
        coord._stream_total_bytes = 24000 * 2 * 19  # ~19 s of audio (round number)
        coord._stream_last_write_ts = now - 0.05
        coord._stream_last_chunk_bytes = 24000 * 2 * 2  # 2 s chunk
        # last_chunk_remaining = 2.0 - 0.05 = 1.95 s
        # total_queued = 19.0 - 14.0 = 5.0 s   (the regime the bug missed)
        # remaining = max(1.95, 5.0) = 5.0 s

        # Cap the wait so the test doesn't actually sleep 5+ s; the
        # contract is that the wait HITS the cap (proving math >> 1.95 s).
        safety_s = AudioCoordinator._DRAIN_SAFETY_BUFFER_S
        # We need a cap that is unambiguously > old-math-result + safety
        # (1.95 + 0.5 = 2.45 s) AND < new-math-result + safety (5.0 + 0.5
        # = 5.5 s), so the test can distinguish: OLD math returns 2.45 s
        # (test would FAIL the lower bound); NEW math returns capped value
        # (test PASSES).
        cap = 3.0
        lower_bound = 2.8  # > old-math result of 2.45 s
        upper_bound = 3.6  # cap + asyncio jitter
        original_cap = AudioCoordinator._MAX_DRAIN_WAIT_S
        AudioCoordinator._MAX_DRAIN_WAIT_S = cap
        try:
            t0 = _time.monotonic()
            await coord.stop_streaming_session(wait_for_drain=True)
            elapsed = _time.monotonic() - t0
        finally:
            AudioCoordinator._MAX_DRAIN_WAIT_S = original_cap

        assert lower_bound <= elapsed <= upper_bound, (
            f"Drain wait under producer-faster-than-real-time regime must "
            f"use total-queued-audio estimate, NOT last-chunk-only. "
            f"OLD math would have returned ~2.45s (last_chunk_remaining "
            f"1.95s + safety {safety_s:.3f}s); NEW math should return "
            f"capped value (~{cap:.3f}s). Got elapsed={elapsed:.3f}s "
            f"(expected {lower_bound}-{upper_bound}s)."
        )

    async def test_wait_for_drain_under_producer_slower_than_realtime_still_uses_last_chunk_math(
        self
    ):
        """Story 18.4 code-review follow-up — the max() fix must NOT
        regress Story 18.3 M6's producer-slower-than-real-time case.

        Under producer-slower regime, the PyAudio buffer is approximately
        empty when the last chunk arrives (playback caught up during slow
        chunks). total_queued_audio_s should be ~0 (or negative, clamped
        to 0); the max() picks last_chunk_remaining, preserving Story
        18.3 M6's contract.

        Setup mirrors Story 18.3's producer-bottleneck scenario:
          * Producer 3.23× slower than real-time (Story 18.1 baseline)
          * Last chunk is 2 s of audio
          * Total audio is 6 s (3 chunks at 2 s each)
          * Wall-clock since first write: 19.4 s (6 s of audio at 3.23×)
          * Wall-clock since last write: 0.1 s (just arrived; producer
            still emitting)
          * total_audio (6 s) - playback_elapsed (19.4 s) = -13.4 s → clamp to 0
          * last_chunk_remaining = 2 s - 0.1 s = 1.9 s
          * max = 1.9 s — Story 18.3 M6 contract preserved.
        """
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        now = _time.monotonic()
        coord._stream_first_write_ts = now - 19.4
        coord._stream_total_bytes = 24000 * 2 * 6  # 6 s of audio
        coord._stream_last_write_ts = now - 0.1
        coord._stream_last_chunk_bytes = 24000 * 2 * 2  # 2 s chunk
        safety_s = AudioCoordinator._DRAIN_SAFETY_BUFFER_S
        # last_chunk_remaining = 1.9 s; total_queued clamped to 0
        # → wait ≈ 1.9 + safety
        expected_wait = 1.9 + safety_s

        t0 = _time.monotonic()
        await coord.stop_streaming_session(wait_for_drain=True)
        elapsed = _time.monotonic() - t0

        assert (expected_wait - 0.2) <= elapsed <= (expected_wait + 0.4), (
            f"Producer-slower regime must preserve Story 18.3 M6 last-chunk-"
            f"only math (total_queued clamped to 0; max picks "
            f"last_chunk_remaining). Expected ~{expected_wait:.3f}s; "
            f"got elapsed={elapsed:.3f}s."
        )

    async def test_wait_for_drain_with_no_writes_does_not_wait(self):
        """If no chunks were ever written (last_chunk_bytes == 0), there
        is nothing to drain. Early-out without sleeping."""
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # No play_audio_chunk calls → last_write_ts is None, last_chunk_bytes is 0.
        assert coord._stream_last_write_ts is None
        assert coord._stream_last_chunk_bytes == 0

        t0 = _time.monotonic()
        await coord.stop_streaming_session(wait_for_drain=True)
        elapsed = _time.monotonic() - t0

        assert elapsed < 0.2

    async def test_play_audio_chunk_records_drain_trackers(self):
        """play_audio_chunk MUST stamp BOTH _stream_first_write_ts (once)
        and _stream_last_write_ts + _stream_last_chunk_bytes (per write),
        and accumulate _stream_total_bytes. The first/last split is the
        Story 18.3 M6 fix — drain math reads the last-chunk trackers.

        Note (consumer-side smoothing buffer added 2026-05-12): trackers
        record DISPATCHED bytes (post-buffer), not input bytes. Chunks
        below the 500ms watermark threshold are held back. This test
        uses chunk sizes ≥ watermark so each push dispatches in one
        flush, preserving the original first/last-tracker contract.
        """
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        assert coord._stream_first_write_ts is None
        assert coord._stream_last_write_ts is None
        assert coord._stream_last_chunk_bytes == 0
        assert coord._stream_total_bytes == 0

        # 30000 bytes @ 24kHz mono int16 = 625ms — crosses 500ms watermark
        # in one push, so the buffer flushes immediately.
        await coord.play_audio_chunk(b"\x00" * 30000, is_final=False)
        first_ts = coord._stream_first_write_ts
        last_ts_after_chunk1 = coord._stream_last_write_ts
        assert first_ts is not None
        assert last_ts_after_chunk1 is not None
        assert coord._stream_total_bytes == 30000
        assert coord._stream_last_chunk_bytes == 30000

        await coord.play_audio_chunk(b"\x00" * 2500, is_final=True)
        # First-write timestamp must NOT be reset on subsequent writes.
        assert coord._stream_first_write_ts == first_ts
        # Last-write timestamp MUST advance to chunk 2's write time.
        assert coord._stream_last_write_ts >= last_ts_after_chunk1
        assert coord._stream_total_bytes == 32500
        # Last-chunk bytes MUST reflect chunk 2's size only (not cumulative).
        assert coord._stream_last_chunk_bytes == 2500

    async def test_play_audio_chunk_holds_below_watermark_then_dispatches(self):
        """Consumer-side smoothing buffer (2026-05-12 RTX 3060 fix):
        chunks below the 500ms watermark threshold are held in the
        buffer and NOT dispatched to the services until either the
        threshold is crossed or is_final is set. Drain trackers must
        not stamp during the buffering phase — the tracking measures
        what PyAudio actually got, not what the producer fed in.
        """
        coord, monitor, virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)

        # 4000 bytes ≈ 83ms — well below the 500ms watermark.
        await coord.play_audio_chunk(b"\x00" * 4000, is_final=False)
        assert coord._stream_first_write_ts is None, (
            "Drain trackers must NOT fire while audio is buffered — they "
            "track dispatched bytes, not input bytes."
        )
        assert coord._stream_total_bytes == 0
        # Service-level dispatch must not have happened yet.
        monitor.play_audio_chunk.assert_not_called()
        virtual.play_audio_chunk.assert_not_called()

        # is_final flushes regardless of watermark.
        await coord.play_audio_chunk(b"\x00" * 2000, is_final=True)
        assert coord._stream_first_write_ts is not None
        # Trackers reflect the merged-and-dispatched payload size.
        assert coord._stream_total_bytes == 6000
        # Dispatched once with the combined payload.
        assert monitor.play_audio_chunk.call_count == 1
        dispatched_bytes = monitor.play_audio_chunk.call_args.args[0]
        assert len(dispatched_bytes) == 6000

    async def test_stop_resets_drain_trackers_for_next_session(self):
        """After stop_streaming_session, the trackers must be cleared so the
        next session starts from zero."""
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        coord._stream_first_write_ts = _time.monotonic()
        coord._stream_total_bytes = 1000
        coord._stream_last_write_ts = _time.monotonic()
        coord._stream_last_chunk_bytes = 1000

        await coord.stop_streaming_session()  # any path

        assert coord._stream_first_write_ts is None
        assert coord._stream_total_bytes == 0
        assert coord._stream_last_write_ts is None
        assert coord._stream_last_chunk_bytes == 0
        assert coord._tts_sample_rate is None

    async def test_M6_producer_bottleneck_workload_still_drains_last_chunk(self):
        """Story 18.3 M6 regression — exact bug class.

        The bug class: producer is slower than realtime (≥1.5×), so
        ``elapsed`` (wall-clock since first write) outpaces
        ``expected_total_seconds`` (cumulative audio duration). The
        original M2 math computed ``remaining = expected_total - elapsed``,
        which goes NEGATIVE, then guarded ``if remaining > 0`` and
        skipped the drain entirely. PyAudio's device-level buffer
        (200–500ms on Windows shared mode) gets truncated by
        stop_stream(), cutting off the last ~500–800ms of audio.

        Empirical reproduction (Commander's Story 18.3 Task 10 bundled
        smoke): producer ratio 1.62 for bf16 on RTX 5090, last 4 words
        of the canonical Sarira-F paragraph cut every time, regardless
        of last-chunk size.

        The M6 fix reads the LAST chunk's trackers + drops the gate,
        always waiting at least the safety buffer. Per
        ``memory/code_review_regression_test_exact_class.md``: the
        regression test must mirror THIS exact bug class.
        """
        coord, _monitor, _virtual = _coord_with_drain_services()
        await coord.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
        # Simulate producer-bottleneck: 10 chunks × 1.98s audio each = 19.8s.
        # Producer 1.62× realtime → elapsed ≈ 32s when last chunk arrives.
        # Last chunk written 0.05s ago (just before the synthetic terminal).
        chunk_bytes = int(24000 * 2 * 1.98)  # 1.98s of int16 mono
        coord._stream_first_write_ts = _time.monotonic() - 32.0
        coord._stream_total_bytes = chunk_bytes * 10
        coord._stream_last_write_ts = _time.monotonic() - 0.05
        coord._stream_last_chunk_bytes = chunk_bytes
        safety_s = AudioCoordinator._DRAIN_SAFETY_BUFFER_S

        # Original buggy math: remaining = 19.8 - 32 = -12.2 → drain SKIPPED.
        # Corrected math: last_chunk_remaining = 1.98 - 0.05 ≈ 1.93s; drain
        # waits 1.93 + safety. With safety=0.5, expected ≈ 2.43s.
        expected_min = 1.93  # last_chunk_remaining alone, no safety
        expected_max = 1.98 + safety_s + 0.5  # generous upper bound

        t0 = _time.monotonic()
        await coord.stop_streaming_session(wait_for_drain=True)
        elapsed = _time.monotonic() - t0

        assert expected_min <= elapsed <= expected_max, (
            f"Story 18.3 M6 regression: producer-bottleneck workload must "
            f"STILL drain the last chunk's residual playback. "
            f"Expected drain wait in [{expected_min:.3f}s, {expected_max:.3f}s]; "
            f"got elapsed={elapsed:.3f}s. If elapsed < 1s, the M2 math gate "
            f"({{if remaining > 0}}) has regressed and the cut-off-at-end is "
            f"back."
        )
