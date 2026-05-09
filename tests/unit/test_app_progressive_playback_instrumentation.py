"""Story 18.1 Task 1.2 + 1.3 — orchestrator-side per-chunk instrumentation tests.

Verifies that ``MyVoiceApp._handle_progressive_chunk_async`` emits two new
metrics after each successful ``play_audio_chunk`` call:

  * ``progressive_chunk_playback_arrival_ms`` — tagged with
    ``chunk_index``, ``is_final``, ``audio_data_size``.
  * ``progressive_chunk_audio_duration_ms`` — tagged with ``chunk_index``;
    value is ``size / sample_rate * 1000``.

These metrics, paired with the producer-side ``progressive_chunk_emit_ms``
metric (qwen_tts_service.py ``_wrapped_post``), are the data layer that
turns Task 1.5's choice between Option 1 / 2 / 3 into an empirical decision.

Mirrors the test pattern at ``tests/unit/test_app_progressive_playback.py``
(_StubChunk + mocked AudioCoordinator + direct ``await`` of the handler).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from myvoice.observability import metrics


@dataclass
class _StubChunk:
    audio_data: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool = False
    text_segment: str = ""
    session_id: "str | None" = None


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def app_with_mocked_coordinator(qapp):
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(qapp)
    coordinator = AsyncMock()
    coordinator.start_streaming_session = AsyncMock(
        return_value={"monitor": "m-1", "virtual": "v-1"}
    )
    coordinator.play_audio_chunk = AsyncMock(
        return_value={"monitor": True, "virtual": True}
    )
    coordinator.stop_streaming_session = AsyncMock(
        return_value={"monitor": True, "virtual": True}
    )
    app._audio_coordinator = coordinator
    app.loop = asyncio.get_event_loop_policy().new_event_loop()
    yield app, coordinator
    try:
        app.loop.close()
    except Exception:
        pass


@pytest.fixture
def metric_capture():
    """Collect every ``MetricRecord`` emitted during the test body.

    Uses ``metrics.add_listener`` (the documented public surface for
    in-process subscription) and the unsubscribe callback it returns to
    clean up after each test. Listener leaks would mask later test
    pollution, so the teardown is unconditional.
    """
    captured: list = []
    unsub = metrics.add_listener(captured.append)
    yield captured
    unsub()


def _drive(app, chunk):
    return asyncio.run(app._handle_progressive_chunk_async(chunk))


class TestProgressivePlaybackInstrumentation:
    """Story 18.1 Task 1.2 + 1.3 — per-chunk consumer-side metrics."""

    def test_arrival_metric_emitted_after_play_audio_chunk(
        self, app_with_mocked_coordinator, metric_capture
    ):
        app, coordinator = app_with_mocked_coordinator
        chunk0 = _StubChunk(
            audio_data=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            session_id="sid-arrival",
        )
        _drive(app, chunk0)

        arrivals = [
            r for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ]
        assert len(arrivals) == 1, (
            "Expected exactly one progressive_chunk_playback_arrival_ms "
            f"emission for a single non-empty chunk; got {len(arrivals)}"
        )
        rec = arrivals[0]
        assert rec.tags["chunk_index"] == 0
        assert rec.tags["is_final"] is False
        assert rec.tags["audio_data_size"] == 3
        # Code-review pass M1: session_id plumbed through from the chunk
        # so the consumer-side row joins to the producer-side emit row by
        # (session_id, chunk_index), not chunk_index alone.
        assert rec.session_id == "sid-arrival"
        # Value is wall-clock ms — must be a positive float in the
        # year-2026 epoch range (loose sanity bound).
        assert isinstance(rec.value, float)
        assert rec.value > 0.0

    def test_audio_duration_metric_emitted_per_data_chunk(
        self, app_with_mocked_coordinator, metric_capture
    ):
        app, _ = app_with_mocked_coordinator
        # 2400 samples @ 24000 Hz = 100 ms exactly.
        chunk0 = _StubChunk(
            audio_data=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            session_id="sid-duration",
        )
        _drive(app, chunk0)

        durations = [
            r for r in metric_capture
            if r.name == "progressive_chunk_audio_duration_ms"
        ]
        assert len(durations) == 1, (
            "Expected exactly one progressive_chunk_audio_duration_ms "
            f"emission per data chunk; got {len(durations)}"
        )
        rec = durations[0]
        assert rec.tags["chunk_index"] == 0
        # Code-review pass M1: session_id plumbed through from the chunk.
        assert rec.session_id == "sid-duration"
        # 2400 / 24000 * 1000 = 100.0
        assert rec.value == pytest.approx(100.0, rel=1e-6)

    def test_no_metrics_emitted_for_zero_size_terminal_chunk(
        self, app_with_mocked_coordinator, metric_capture
    ):
        """TRUE_STREAM's synthetic terminal chunk (audio_data.size == 0)
        skips the play_audio_chunk path AND therefore must not emit
        either consumer-side metric — emitting either would muddy the
        Task 1.4 CSV with a non-data row whose audio_duration is 0
        (would skew the median/p95 stats).

        Variant-(a): start with a non-zero chunk to flip
        ``_progressive_playback_active`` True (otherwise the terminal
        falls through the early-return at the no-active branch).
        """
        app, _ = app_with_mocked_coordinator

        # Open the session.
        chunk0 = _StubChunk(
            audio_data=np.array([0.5, 0.5], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)
        metric_capture.clear()

        # Now send the synthetic terminal (TRUE_STREAM shape).
        terminal = _StubChunk(
            audio_data=np.zeros(0, dtype=np.float32),
            sample_rate=24000,
            chunk_index=1,
            is_final=True,
        )
        _drive(app, terminal)

        arrivals = [
            r for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ]
        durations = [
            r for r in metric_capture
            if r.name == "progressive_chunk_audio_duration_ms"
        ]
        assert arrivals == [], (
            "Zero-size terminal chunk must not emit "
            "progressive_chunk_playback_arrival_ms — the play_audio_chunk "
            "branch is gated on ``audio_data.size > 0``."
        )
        assert durations == [], (
            "Zero-size terminal chunk must not emit "
            "progressive_chunk_audio_duration_ms — the metric pair is "
            "co-located inside the same ``size > 0`` branch."
        )

    def test_arrival_and_duration_emitted_for_each_progressive_chunk(
        self, app_with_mocked_coordinator, metric_capture
    ):
        """Three-chunk smoke: each non-empty chunk emits exactly one
        of each metric. This is the regression guard against future
        edits accidentally moving either ``metrics.record`` outside
        the ``size > 0`` branch.
        """
        app, _ = app_with_mocked_coordinator

        for idx in range(3):
            chunk = _StubChunk(
                audio_data=np.full(1200, 0.1, dtype=np.float32),
                sample_rate=24000,
                chunk_index=idx,
                is_final=(idx == 2),
            )
            _drive(app, chunk)

        arrivals = [
            r for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ]
        durations = [
            r for r in metric_capture
            if r.name == "progressive_chunk_audio_duration_ms"
        ]
        assert len(arrivals) == 3, (
            f"Expected 3 arrival emissions; got {len(arrivals)}"
        )
        assert len(durations) == 3, (
            f"Expected 3 duration emissions; got {len(durations)}"
        )
        # Per-chunk pairing is intact: arrivals[i].tags['chunk_index']
        # matches durations[i].tags['chunk_index'] for every i.
        for i, (a, d) in enumerate(zip(arrivals, durations)):
            assert a.tags["chunk_index"] == i
            assert d.tags["chunk_index"] == i
        # Final chunk's is_final tag carries through.
        assert arrivals[2].tags["is_final"] is True
        assert arrivals[0].tags["is_final"] is False
        assert arrivals[1].tags["is_final"] is False

    def test_session_id_separates_chunks_from_two_generations(
        self, app_with_mocked_coordinator, metric_capture
    ):
        """Code-review pass M1 — when a single CSV captures two back-to-
        back generations, rows must be joinable to the right producer
        emit row by (session_id, chunk_index), not chunk_index alone.

        Both generations re-use chunk_index 0..N-1 from the producer's
        per-session counter, so without session_id on the consumer-side
        rows the rows from generation 2 collide with generation 1's
        chunk_index values and the post-processing join is corrupt.
        """
        app, _ = app_with_mocked_coordinator

        # Generation 1: 2 chunks with session_id="gen-1".
        for idx in range(2):
            chunk = _StubChunk(
                audio_data=np.full(2400, 0.05, dtype=np.float32),
                sample_rate=24000,
                chunk_index=idx,
                is_final=(idx == 1),
                session_id="gen-1",
            )
            _drive(app, chunk)

        # Generation 2: 2 chunks with session_id="gen-2", indices restart at 0.
        for idx in range(2):
            chunk = _StubChunk(
                audio_data=np.full(2400, 0.05, dtype=np.float32),
                sample_rate=24000,
                chunk_index=idx,
                is_final=(idx == 1),
                session_id="gen-2",
            )
            _drive(app, chunk)

        arrivals = [
            r for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ]
        assert len(arrivals) == 4
        # Two distinct session_ids each appear on two records.
        gen_1 = [r for r in arrivals if r.session_id == "gen-1"]
        gen_2 = [r for r in arrivals if r.session_id == "gen-2"]
        assert len(gen_1) == 2 and len(gen_2) == 2
        # Within each generation, chunk_index restarts at 0 — the
        # session_id is what disambiguates the duplicate (sid, ci) pairs.
        assert sorted(r.tags["chunk_index"] for r in gen_1) == [0, 1]
        assert sorted(r.tags["chunk_index"] for r in gen_2) == [0, 1]

    def test_fallback_restart_does_not_double_emit_chunk_zero_metrics(
        self, app_with_mocked_coordinator, metric_capture
    ):
        """Code-review pass M4 — guards the Story 17.3 commit-abd842c
        invariant ("no spurious second session open after fallback
        restart") in the metrics dimension.

        When chunk_index=0 arrives while ``_progressive_playback_active``
        is already True (NFR7 fallback variant (b): TRUE_STREAM raised
        mid-stream and SENTENCE_STREAM took over), the handler closes
        the stale session and opens a fresh one — but each NEW chunk
        emits exactly ONE arrival metric. A buggy edit that moved the
        ``metrics.record`` call ABOVE the size-gate (or duplicated it
        across the close+open path) would surface here.
        """
        app, _ = app_with_mocked_coordinator

        # First stream — chunk 0.
        first = _StubChunk(
            audio_data=np.full(2400, 0.1, dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            session_id="stream-A",
        )
        _drive(app, first)
        # Sanity: one arrival metric emitted so far.
        assert sum(
            1 for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ) == 1
        # Second stream — fresh chunk 0 while _progressive_playback_active
        # is already True (the fallback-restart shape).
        assert app._progressive_playback_active is True
        restart = _StubChunk(
            audio_data=np.full(2400, 0.1, dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            session_id="stream-B",
        )
        _drive(app, restart)

        arrivals = [
            r for r in metric_capture
            if r.name == "progressive_chunk_playback_arrival_ms"
        ]
        # Exactly one MORE arrival metric — not two. The stale-session
        # close path must not emit a metric of its own.
        assert len(arrivals) == 2, (
            f"Expected 2 arrival emissions across the 2-stream "
            f"fallback-restart sequence; got {len(arrivals)}"
        )
        assert arrivals[0].session_id == "stream-A"
        assert arrivals[1].session_id == "stream-B"
