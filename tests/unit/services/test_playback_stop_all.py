"""Story ui-3 Defect A — stop-all playback helpers that never existed.

RTX 3060 log 2026-09-14 (build 56, code unchanged since):

    19:07:03 audio_coordinator - ERROR - Error stopping monitor playback:
        'MonitorAudioService' object has no attribute 'stop_all_playback'
    19:07:03 audio_coordinator - ERROR - Error stopping virtual playback:
        'VirtualMicrophoneService' object has no attribute
        'stop_all_virtual_microphone_playback'
    19:11:30 monitor_audio_service - ERROR - Error stopping monitor
        playback monitor_1_1789434423: STOPPED          (x8, at shutdown)

Two bugs, one test file:

* ``AudioCoordinator.stop_all_playback`` fanned out to two service methods
  that did not exist; its try/except turned the AttributeError into an
  ERROR line and the Stop button stopped nothing on the task-based path.
  ``tests/unit/services/test_audio_coordinator.py`` never caught this
  because its fixtures are ``MagicMock(spec=Service)`` with the missing
  method *assigned* (spec only guards reads, not writes). The coordinator
  tests here use the REAL services (constructed, never initialized — no
  PortAudio) so a missing method is an AttributeError again.

* ``PlaybackStatus.STOPPED`` did not exist. ``stop_monitor_playback`` /
  ``stop_virtual_playback`` assigned it, so EnumMeta raised
  ``AttributeError('STOPPED')`` — the bare status name in the log — on
  EVERY stop, before the entry was popped. The workers never remove a
  finished task from ``_active_tasks``, which is why shutdown hit eight.

Structural template follows test_audio_coordinator.py (module docstring →
fixtures → class-grouped Test* with one AC focus per class).
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

from myvoice.models.audio_playback_task import PlaybackStatus
from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import (
    MonitorAudioService,
    MonitorPlaybackTask,
)
from myvoice.services.virtual_microphone_service import (
    VirtualMicrophoneService,
    VirtualPlaybackTask,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _monitor_task(task_id: str, status: PlaybackStatus) -> MonitorPlaybackTask:
    return MonitorPlaybackTask(
        playback_id=task_id, audio_data=b"", device=None, status=status
    )


def _virtual_task(task_id: str, status: PlaybackStatus) -> VirtualPlaybackTask:
    return VirtualPlaybackTask(
        playback_id=task_id, audio_data=b"", device=None, status=status
    )


def _error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.fixture
def monitor() -> MonitorAudioService:
    """Real service, constructed only. ``initialize()`` would open PyAudio;
    ``_active_tasks`` is populated directly instead, exactly as the shipped
    log shape (entries present, no worker thread behind them)."""
    return MonitorAudioService()


@pytest.fixture
def virtual() -> VirtualMicrophoneService:
    return VirtualMicrophoneService()


@pytest.fixture
def coordinator(monitor, virtual) -> AudioCoordinator:
    """Coordinator over the REAL services (contrast test_audio_coordinator.py,
    whose spec'd MagicMocks let the missing helpers slip through)."""
    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = monitor
    coord.virtual_service = virtual
    return coord


# --------------------------------------------------------------------------- #
# AC A2 — the STOPPED raise
# --------------------------------------------------------------------------- #


class TestPlaybackStatusStopped:
    def test_stopped_member_matches_the_value_the_workers_poll(self):
        # Both playback workers break their write loop on
        # ``task.status.value in ['failed', 'stopped']``; the member the
        # stop paths assign must produce exactly that string.
        assert PlaybackStatus.STOPPED.value == "stopped"


@pytest.mark.asyncio
class TestStopFinishedTaskIsCleanNoOp:
    """AC A2: stopping an already-completed task neither raises nor logs
    ERROR; the entry is removed; shutdown with finished tasks is clean."""

    @pytest.mark.parametrize(
        "terminal", [PlaybackStatus.COMPLETED, PlaybackStatus.FAILED]
    )
    async def test_monitor_stop_on_finished_task_removes_entry_without_error(
        self, monitor, caplog, terminal
    ):
        caplog.set_level(logging.DEBUG)
        task = _monitor_task("monitor_1_1789434423", terminal)
        monitor._active_tasks[task.playback_id] = task

        result = await monitor.stop_monitor_playback(task.playback_id)

        assert result is False  # nothing live was interrupted
        assert task.playback_id not in monitor._active_tasks
        assert task.status is terminal  # terminal state not overwritten
        assert _error_records(caplog) == []

    @pytest.mark.parametrize(
        "terminal", [PlaybackStatus.COMPLETED, PlaybackStatus.FAILED]
    )
    async def test_virtual_stop_on_finished_task_removes_entry_without_error(
        self, virtual, caplog, terminal
    ):
        caplog.set_level(logging.DEBUG)
        task = _virtual_task("virtual_1_1789434423", terminal)
        virtual._active_tasks[task.playback_id] = task

        result = await virtual.stop_virtual_playback(task.playback_id)

        assert result is False
        assert task.playback_id not in virtual._active_tasks
        assert task.status is terminal
        assert _error_records(caplog) == []

    async def test_monitor_shutdown_with_eight_finished_tasks_logs_no_error(
        self, monitor, caplog
    ):
        # Exact log shape: eight completed tasks still in _active_tasks at
        # shutdown, each of which produced an ERROR line on the RTX 3060.
        caplog.set_level(logging.DEBUG)
        for i in (1, 2, 3, 5, 6, 7, 8, 9):
            tid = f"monitor_{i}_178943{i:04d}"
            monitor._active_tasks[tid] = _monitor_task(
                tid, PlaybackStatus.COMPLETED
            )

        assert await monitor.shutdown() is True

        assert monitor._active_tasks == {}
        assert _error_records(caplog) == []

    async def test_virtual_shutdown_with_finished_tasks_logs_no_error(
        self, virtual, caplog
    ):
        caplog.set_level(logging.DEBUG)
        for i in range(3):
            tid = f"virtual_{i}_1789434423"
            virtual._active_tasks[tid] = _virtual_task(
                tid, PlaybackStatus.COMPLETED
            )

        assert await virtual.shutdown() is True

        assert virtual._active_tasks == {}
        assert _error_records(caplog) == []


# --------------------------------------------------------------------------- #
# AC A1 — the stop-all helpers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestServiceStopAllHelpers:
    """AC A1: the helpers exist, go through the per-task stop, and return
    the count of live tasks stopped."""

    async def test_monitor_stop_all_playback_stops_live_tasks_and_counts(
        self, monitor, caplog
    ):
        caplog.set_level(logging.DEBUG)
        live_a = _monitor_task("monitor_1_1", PlaybackStatus.PLAYING)
        live_b = _monitor_task("monitor_2_1", PlaybackStatus.PENDING)
        done = _monitor_task("monitor_3_1", PlaybackStatus.COMPLETED)
        for t in (live_a, live_b, done):
            monitor._active_tasks[t.playback_id] = t

        with patch.object(
            monitor, "stop_monitor_playback", wraps=monitor.stop_monitor_playback
        ) as per_task:
            count = await monitor.stop_all_playback()

        assert count == 2  # finished task cleaned up, not counted
        assert per_task.call_count == 3
        assert monitor._active_tasks == {}
        # The status the workers poll for
        assert live_a.status is PlaybackStatus.STOPPED
        assert live_b.status is PlaybackStatus.STOPPED
        assert done.status is PlaybackStatus.COMPLETED
        assert _error_records(caplog) == []

    async def test_virtual_stop_all_stops_live_tasks_and_counts(
        self, virtual, caplog
    ):
        caplog.set_level(logging.DEBUG)
        live = _virtual_task("virtual_1_1", PlaybackStatus.PLAYING)
        done = _virtual_task("virtual_2_1", PlaybackStatus.FAILED)
        for t in (live, done):
            virtual._active_tasks[t.playback_id] = t

        with patch.object(
            virtual, "stop_virtual_playback", wraps=virtual.stop_virtual_playback
        ) as per_task:
            count = await virtual.stop_all_virtual_microphone_playback()

        assert count == 1
        assert per_task.call_count == 2
        assert virtual._active_tasks == {}
        assert live.status is PlaybackStatus.STOPPED
        assert done.status is PlaybackStatus.FAILED
        assert _error_records(caplog) == []

    async def test_stop_all_on_idle_services_is_a_zero_no_op(
        self, monitor, virtual
    ):
        assert await monitor.stop_all_playback() == 0
        assert await virtual.stop_all_virtual_microphone_playback() == 0


# --------------------------------------------------------------------------- #
# AC A3 — the coordinator fan-out against the real services
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestCoordinatorStopAllPlaybackRealServices:
    """AC A3: the test that would have logged the two AttributeErrors at
    19:07:03 now asserts the count and that both per-task stops ran."""

    async def test_stop_button_path_stops_both_sinks_and_sums_count(
        self, coordinator, monitor, virtual, caplog
    ):
        caplog.set_level(logging.DEBUG)
        m_task = _monitor_task("monitor_4_1789434440", PlaybackStatus.PLAYING)
        v_task = _virtual_task("virtual_4_1789434440", PlaybackStatus.PLAYING)
        monitor._active_tasks[m_task.playback_id] = m_task
        virtual._active_tasks[v_task.playback_id] = v_task

        with patch.object(
            monitor, "stop_monitor_playback", wraps=monitor.stop_monitor_playback
        ) as m_stop, patch.object(
            virtual, "stop_virtual_playback", wraps=virtual.stop_virtual_playback
        ) as v_stop:
            total = await coordinator.stop_all_playback()

        assert total == 2
        m_stop.assert_awaited_once_with(m_task.playback_id)
        v_stop.assert_awaited_once_with(v_task.playback_id)
        assert m_task.status is PlaybackStatus.STOPPED
        assert v_task.status is PlaybackStatus.STOPPED
        assert monitor._active_tasks == {} and virtual._active_tasks == {}
        # The exact shipped symptom: two "Error stopping ... playback" ERRORs
        assert _error_records(caplog) == []

    async def test_stop_button_with_nothing_playing_returns_zero_without_error(
        self, coordinator, caplog
    ):
        caplog.set_level(logging.DEBUG)
        assert await coordinator.stop_all_playback() == 0
        assert _error_records(caplog) == []


class TestStopBeforeWorkerStarts:
    """Story ui-3 review: a STOPPED written before the worker thread reaches
    ``mark_started()`` used to be overwritten with PLAYING, so the clip
    played to the end and the stop reported success. ``mark_started`` now
    only promotes a PENDING task."""

    def test_monitor_mark_started_does_not_revive_a_stopped_task(self):
        task = _monitor_task("monitor_1", PlaybackStatus.PENDING)
        task.status = PlaybackStatus.STOPPED  # the stop path, racing the worker
        task.mark_started()
        assert task.status is PlaybackStatus.STOPPED
        assert task.start_time is None

    def test_virtual_mark_started_does_not_revive_a_stopped_task(self):
        task = _virtual_task("virtual_1", PlaybackStatus.PENDING)
        task.status = PlaybackStatus.STOPPED
        task.mark_started()
        assert task.status is PlaybackStatus.STOPPED
        assert task.start_time is None

    def test_pending_task_still_starts(self):
        task = _monitor_task("monitor_2", PlaybackStatus.PENDING)
        task.mark_started()
        assert task.status is PlaybackStatus.PLAYING
        assert task.start_time is not None

    @pytest.mark.asyncio
    async def test_stop_join_does_not_block_the_event_loop(self, monitor):
        """The join on the worker thread now happens in the default
        executor: a worker that ignores STOPPED for a while must not stall
        other coroutines on the loop for that time."""
        import threading
        import time

        task = _monitor_task("monitor_3", PlaybackStatus.PLAYING)
        monitor._active_tasks[task.playback_id] = task
        worker = threading.Thread(target=lambda: time.sleep(0.4))
        worker.start()
        monitor._playback_threads[task.playback_id] = worker

        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(8):
                await asyncio.sleep(0.05)
                ticks += 1

        t = asyncio.ensure_future(ticker())
        result = await monitor.stop_monitor_playback(task.playback_id)
        ticks_while_stopping = ticks
        await t
        assert result is True
        # A blocking join gives the loop no turns for the whole 0.4 s, so
        # the ticker would still read 0 when the stop returned.
        assert ticks_while_stopping >= 4
        assert task.playback_id not in monitor._active_tasks
