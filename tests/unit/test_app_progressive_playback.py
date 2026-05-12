"""Story 17.3 — orchestrator-level progressive-playback consumer tests.

Covers AC #2 (Task 2.6):
  - ``test_first_chunk_opens_streaming_session`` — chunk_index=0 invokes
    ``AudioCoordinator.start_streaming_session`` with the chunk's
    sample rate exactly once.
  - ``test_subsequent_chunks_call_play_audio_chunk`` — chunks 1..N invoke
    ``play_audio_chunk`` with PCM16 bytes and ``is_final=False``.
  - ``test_final_chunk_closes_streaming_session`` — terminal
    ``AudioChunk(is_final=True)`` invokes ``stop_streaming_session`` and
    leaves ``_progressive_playback_active`` True until the dispatch path
    consumes it (the deferred-clear contract).

The handler is exercised directly by ``await``-ing
``_handle_progressive_chunk_async`` so the run-coroutine-threadsafe
trampoline (``_on_audio_chunk_ready``) need not be involved — the
trampoline is a thin synchronous shim with no branching logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")


@dataclass
class _StubChunk:
    """Mirrors ``myvoice.services.qwen_tts_service.AudioChunk``'s public
    surface — keeps these tests free of the heavy qwen_tts_service import
    chain (which transitively pulls torch + qwen-tts).
    """
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
    """Construct a ``MyVoiceApp`` with a mocked ``AudioCoordinator``.

    The orchestrator's progressive-playback methods only need
    ``_audio_coordinator``, ``self.loop``, and the slot fields — full
    service initialization is not required. Returns
    ``(app, coordinator_mock)``.

    NOTE on ``start_streaming_session`` mock: production
    ``AudioCoordinator.start_streaming_session``
    (audio_coordinator.py:1018-1072) catches all exceptions internally
    and returns the result dict with None values on failure; the
    consumer inspects the dict (NOT a raise) to detect failure.
    The default mock here returns BOTH ids non-None — i.e. the
    happy-path open. Failure-path tests override
    ``return_value={"monitor": None, "virtual": None}`` to mirror
    production reality (per memory/code_review_regression_test_exact_class.md).
    """
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
    # Provide a loop so _on_audio_chunk_ready could schedule onto it (the
    # tests below await the async handler directly, so this is mostly
    # for completeness / parity with production).
    app.loop = asyncio.get_event_loop_policy().new_event_loop()
    yield app, coordinator
    try:
        app.loop.close()
    except Exception:
        pass


def _drive(app, chunk):
    """Run the handler on a fresh loop; keeps tests free of pytest-asyncio
    fixture wiring and matches the integration suite's runner pattern."""
    return asyncio.run(app._handle_progressive_chunk_async(chunk))


class TestProgressivePlaybackConsumer:
    """Story 17.3 AC #2 — orchestrator wires the chunk callback to
    ``AudioCoordinator``'s start/play/stop streaming triplet."""

    def test_first_chunk_opens_streaming_session(
        self, app_with_mocked_coordinator
    ):
        app, coordinator = app_with_mocked_coordinator
        assert app._progressive_playback_active is False

        chunk0 = _StubChunk(
            audio_data=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)

        coordinator.start_streaming_session.assert_awaited_once_with(
            sample_rate=24000, channels=1, sample_width=2
        )
        assert app._progressive_playback_active is True
        assert app._progressive_playback_sample_rate == 24000
        # Chunk 0's audio data is also written to the open session.
        coordinator.play_audio_chunk.assert_awaited_once()
        first_call = coordinator.play_audio_chunk.await_args
        assert isinstance(first_call.args[0], (bytes, bytearray))
        assert first_call.kwargs.get("is_final") is False
        # stop_streaming_session must NOT fire on a non-final chunk.
        coordinator.stop_streaming_session.assert_not_awaited()

    def test_subsequent_chunks_call_play_audio_chunk(
        self, app_with_mocked_coordinator
    ):
        app, coordinator = app_with_mocked_coordinator
        # Skip chunk 0 — pre-set the flag so this test exercises the
        # "session already open" branch without coupling to chunk 0.
        app._progressive_playback_active = True
        app._progressive_playback_sample_rate = 24000

        for idx in range(1, 4):
            chunk = _StubChunk(
                audio_data=np.array(
                    [0.05 * idx, -0.05 * idx], dtype=np.float32
                ),
                sample_rate=24000,
                chunk_index=idx,
            )
            _drive(app, chunk)

        coordinator.start_streaming_session.assert_not_awaited()
        assert coordinator.play_audio_chunk.await_count == 3
        for call in coordinator.play_audio_chunk.await_args_list:
            assert isinstance(call.args[0], (bytes, bytearray))
            # PCM16 → 4 bytes for two float32 samples.
            assert len(call.args[0]) == 4
            assert call.kwargs.get("is_final") is False
        coordinator.stop_streaming_session.assert_not_awaited()
        assert app._progressive_playback_active is True

    def test_final_chunk_closes_streaming_session(
        self, app_with_mocked_coordinator
    ):
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_active = True
        app._progressive_playback_sample_rate = 24000

        terminal = _StubChunk(
            audio_data=np.zeros(0, dtype=np.float32),
            sample_rate=24000,
            chunk_index=5,
            is_final=True,
        )
        _drive(app, terminal)

        coordinator.stop_streaming_session.assert_awaited_once()
        # Story 17.3 deviation from AC #2 step 4: the flag is intentionally
        # NOT cleared on is_final. _play_generated_audio's skip-branch is
        # the canonical consumer; clearing here would race the dispatch
        # path on the asyncio loop ordering. See app.py:_handle_progressive
        # _chunk_async docstring.
        assert app._progressive_playback_active is True
        # No further play_audio_chunk on the terminal chunk (zero-length).
        coordinator.play_audio_chunk.assert_not_awaited()

    def test_sentence_stream_final_chunk_with_audio_is_played(
        self, app_with_mocked_coordinator
    ):
        """Code-review HIGH-1 regression: SENTENCE_STREAM emits its last
        data chunk with ``is_final=True`` AND real ``audio_data``
        (qwen_tts_service.py:3071-3082) — NOT a separate zero-length
        synthetic terminal chunk like TRUE_STREAM. The consumer must play
        that chunk's audio BEFORE closing the session, otherwise the last
        sentence of every SENTENCE_STREAM utterance is silently dropped.

        Mirrors the EXACT bug class per
        memory/code_review_regression_test_exact_class.md: a non-empty
        ``audio_data`` paired with ``is_final=True``, asserted to reach
        ``play_audio_chunk`` before ``stop_streaming_session``.
        """
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_active = True
        app._progressive_playback_sample_rate = 24000

        # SENTENCE_STREAM-shape final chunk: real audio AND is_final=True.
        sentence_final = _StubChunk(
            audio_data=np.array(
                [0.1, -0.1, 0.2, -0.2], dtype=np.float32
            ),
            sample_rate=24000,
            chunk_index=4,
            is_final=True,
            text_segment="last sentence.",
        )
        _drive(app, sentence_final)

        coordinator.play_audio_chunk.assert_awaited_once()
        play_call = coordinator.play_audio_chunk.await_args
        assert isinstance(play_call.args[0], (bytes, bytearray))
        # 4 float32 samples → 8 bytes of int16 PCM.
        assert len(play_call.args[0]) == 8
        # is_final passed through so the underlying service can drain.
        assert play_call.kwargs.get("is_final") is True
        # Session close fires AFTER the audio is written.
        coordinator.stop_streaming_session.assert_awaited_once()

    def test_sentence_stream_play_then_close_ordering(
        self, app_with_mocked_coordinator
    ):
        """Code-review HIGH-1 follow-up: explicit await-order assertion —
        ``play_audio_chunk`` MUST be awaited before
        ``stop_streaming_session`` on a SENTENCE_STREAM final chunk.
        """
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_active = True
        app._progressive_playback_sample_rate = 24000

        order: list[str] = []

        async def record_play(*args, **kwargs):
            order.append("play")
            return {"monitor": True, "virtual": True}

        async def record_stop(*args, **kwargs):
            order.append("stop")
            return {"monitor": True, "virtual": True}

        coordinator.play_audio_chunk = AsyncMock(side_effect=record_play)
        coordinator.stop_streaming_session = AsyncMock(side_effect=record_stop)

        sentence_final = _StubChunk(
            audio_data=np.array([0.3, -0.3], dtype=np.float32),
            sample_rate=24000,
            chunk_index=2,
            is_final=True,
        )
        _drive(app, sentence_final)

        assert order == ["play", "stop"], (
            f"play_audio_chunk must run before stop_streaming_session; "
            f"got {order}"
        )


class TestProgressivePlaybackSampleRateAndFailure:
    """Story 17.3 AC #3 — sample-rate handshake + open-failure graceful
    degradation (Task 3.3)."""

    def test_session_open_failure_falls_through_to_batch(
        self, app_with_mocked_coordinator
    ):
        """Code-review HIGH-2/HIGH-3 regression: production
        ``AudioCoordinator.start_streaming_session`` (audio_coordinator.py:
        1018-1072) wraps its body in ``try/except Exception``, swallows
        the exception, and returns ``{"monitor": None, "virtual": None}``
        — it never re-raises on PyAudio open failure. The consumer must
        therefore inspect the returned dict (NOT a try/except) to
        detect failure; otherwise the flag is set True on a non-existent
        session and the user hears nothing (progressive skipped + batch
        skipped).

        Mirrors the EXACT bug class per
        memory/code_review_regression_test_exact_class.md.
        """
        app, coordinator = app_with_mocked_coordinator
        coordinator.start_streaming_session.return_value = {
            "monitor": None,
            "virtual": None,
        }

        chunk0 = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)

        # Open was attempted exactly once and returned both-None.
        assert coordinator.start_streaming_session.await_count == 1
        # Flag stays False so _play_generated_audio's skip-check is a
        # no-op and the assembled buffer plays via the batch path.
        assert app._progressive_playback_active is False
        # Chunk is discarded — no play_audio_chunk on a session that
        # failed to open.
        coordinator.play_audio_chunk.assert_not_awaited()
        coordinator.stop_streaming_session.assert_not_awaited()

    def test_session_open_partial_success_keeps_progressive_active(
        self, app_with_mocked_coordinator
    ):
        """Defensive boundary: if EITHER service opened (e.g., monitor
        succeeded but virtual failed), progressive playback proceeds —
        only the both-None case is treated as total failure."""
        app, coordinator = app_with_mocked_coordinator
        coordinator.start_streaming_session.return_value = {
            "monitor": "m-1",
            "virtual": None,
        }

        chunk0 = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)

        assert app._progressive_playback_active is True
        coordinator.play_audio_chunk.assert_awaited_once()

    def test_session_open_exception_falls_through_to_batch(
        self, app_with_mocked_coordinator
    ):
        """Defense-in-depth: even though production never raises (see
        ``test_session_open_failure_falls_through_to_batch``), the
        consumer's try/except is still load-bearing if a future refactor
        of ``AudioCoordinator.start_streaming_session`` removes the inner
        swallow. Verify that an unexpected raise also leaves the flag
        False so batch playback runs.
        """
        app, coordinator = app_with_mocked_coordinator
        coordinator.start_streaming_session.side_effect = RuntimeError(
            "hypothetical future refactor: PyAudio open re-raises"
        )

        chunk0 = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)

        assert coordinator.start_streaming_session.await_count == 1
        assert app._progressive_playback_active is False
        coordinator.play_audio_chunk.assert_not_awaited()
        coordinator.stop_streaming_session.assert_not_awaited()

    def test_chunk_sample_rate_passed_to_session_open(
        self, app_with_mocked_coordinator
    ):
        """Defensive against hardcoded 24000: the orchestrator must pass
        the chunk's actual ``sample_rate`` so future model changes (or
        mocked test scenarios) work."""
        app, coordinator = app_with_mocked_coordinator

        chunk0 = _StubChunk(
            audio_data=np.array([0.0], dtype=np.float32),
            sample_rate=22050,
            chunk_index=0,
        )
        _drive(app, chunk0)

        coordinator.start_streaming_session.assert_awaited_once_with(
            sample_rate=22050, channels=1, sample_width=2
        )
        assert app._progressive_playback_sample_rate == 22050

    def test_session_open_log_includes_session_ids(
        self, app_with_mocked_coordinator, caplog
    ):
        """Code-review MEDIUM-3 regression: AC #3 specifies the log line
        as ``"Progressive playback session opened: sample_rate=24000Hz,
        monitor_session=<id>, virtual_session=<id>"`` — the session-id
        dict from ``start_streaming_session`` must be reflected in the
        log so postmortem on dual-service failures can identify which
        side opened.
        """
        import logging as _logging

        app, coordinator = app_with_mocked_coordinator
        coordinator.start_streaming_session.return_value = {
            "monitor": "monitor-abc",
            "virtual": "virtual-xyz",
        }

        chunk0 = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        with caplog.at_level(_logging.INFO, logger=app.logger.name):
            _drive(app, chunk0)

        opened = [
            r for r in caplog.records
            if "Progressive playback session opened" in r.getMessage()
        ]
        assert len(opened) == 1, (
            f"Expected one session-open INFO log; got {len(opened)}"
        )
        msg = opened[0].getMessage()
        assert "sample_rate=24000Hz" in msg
        assert "monitor_session=monitor-abc" in msg
        assert "virtual_session=virtual-xyz" in msg


def _drive_sequence(app, chunks):
    """Run a sequence of chunks through the handler on a single asyncio
    loop so Task 5's NFR7 fallback scenarios can be expressed in the
    natural order chunks would arrive in production."""
    async def runner():
        for chunk in chunks:
            await app._handle_progressive_chunk_async(chunk)

    asyncio.run(runner())


class TestProgressivePlaybackNFR7Fallback:
    """Story 17.3 AC #5 — when TRUE_STREAM raises mid-stream and
    SENTENCE_STREAM takes over (the three-mode fallback chain), the
    orchestrator's progressive-playback handler must close the stale
    session cleanly and open a fresh one on the new chunk_index=0
    (variant (b): clean cut + restart)."""

    def test_true_stream_raises_mid_progressive_then_sentence_stream_restarts(
        self, app_with_mocked_coordinator
    ):
        app, coordinator = app_with_mocked_coordinator
        # Simulate: TRUE_STREAM emits 3 chunks (partial progressive
        # playback), then raises (no terminal chunk). NFR7 routes to
        # SENTENCE_STREAM which restarts from chunk 0 and emits 5 + final.
        true_stream_chunks = [
            _StubChunk(
                audio_data=np.array([0.1 * i], dtype=np.float32),
                sample_rate=24000,
                chunk_index=i,
            )
            for i in range(3)
        ]
        sentence_stream_chunks = [
            _StubChunk(
                audio_data=np.array([0.2 * i], dtype=np.float32),
                sample_rate=24000,
                chunk_index=i,
            )
            for i in range(5)
        ]
        sentence_stream_chunks.append(
            _StubChunk(
                audio_data=np.zeros(0, dtype=np.float32),
                sample_rate=24000,
                chunk_index=5,
                is_final=True,
            )
        )

        _drive_sequence(
            app, true_stream_chunks + sentence_stream_chunks
        )

        # start_streaming_session called twice: once at TRUE_STREAM chunk 0,
        # once at SENTENCE_STREAM chunk 0 (after the stale-close).
        assert coordinator.start_streaming_session.await_count == 2, (
            f"Expected 2 session opens (TRUE_STREAM + SENTENCE_STREAM); "
            f"got {coordinator.start_streaming_session.await_count}"
        )
        # stop_streaming_session called twice: once on the stale-close
        # at SENTENCE_STREAM chunk 0, once on the SENTENCE_STREAM final.
        assert coordinator.stop_streaming_session.await_count == 2, (
            f"Expected 2 session closes (stale + final); "
            f"got {coordinator.stop_streaming_session.await_count}"
        )
        # Audio chunks: 3 from TRUE_STREAM partial + 5 from SENTENCE_STREAM.
        assert coordinator.play_audio_chunk.await_count == 8

    def test_true_stream_raises_before_chunk_0(
        self, app_with_mocked_coordinator
    ):
        """When TRUE_STREAM raises BEFORE emitting any chunks (the typical
        empty-chunks-guard path at qwen_tts_service.py:4035), the flag
        stays False and SENTENCE_STREAM proceeds with a single open/close
        cycle — no stale-session restart."""
        app, coordinator = app_with_mocked_coordinator
        # No TRUE_STREAM chunks (raised pre-chunk-0). SENTENCE_STREAM emits
        # its full sequence.
        sentence_chunks = [
            _StubChunk(
                audio_data=np.array([0.05 * i], dtype=np.float32),
                sample_rate=24000,
                chunk_index=i,
            )
            for i in range(4)
        ]
        sentence_chunks.append(
            _StubChunk(
                audio_data=np.zeros(0, dtype=np.float32),
                sample_rate=24000,
                chunk_index=4,
                is_final=True,
            )
        )

        _drive_sequence(app, sentence_chunks)

        # Single open/close pair — no stale-session abort.
        assert coordinator.start_streaming_session.await_count == 1
        assert coordinator.stop_streaming_session.await_count == 1
        assert coordinator.play_audio_chunk.await_count == 4

    def test_dispatch_chain_unchanged_under_normal_path(
        self, app_with_mocked_coordinator
    ):
        """Happy-path TRUE_STREAM (no raise) emits chunks + final → exactly
        one open/close pair; no fallback artifacts."""
        app, coordinator = app_with_mocked_coordinator
        chunks = [
            _StubChunk(
                audio_data=np.array([0.05 * i], dtype=np.float32),
                sample_rate=24000,
                chunk_index=i,
            )
            for i in range(4)
        ]
        chunks.append(
            _StubChunk(
                audio_data=np.zeros(0, dtype=np.float32),
                sample_rate=24000,
                chunk_index=4,
                is_final=True,
            )
        )

        _drive_sequence(app, chunks)

        assert coordinator.start_streaming_session.await_count == 1
        assert coordinator.stop_streaming_session.await_count == 1
        assert coordinator.play_audio_chunk.await_count == 4


class TestProgressivePlaybackSkipRaceRegression:
    """Build #11 regression: terminal-chunk handler racing with
    ``_play_generated_audio``'s skip-branch.

    Repro pattern from `I:/MyVoice/logs/myvoice.log` 11:08 / 11:10 / 11:11
    entries: gen completes, skip-branch clears the flag, then the
    TRUE_STREAM synthetic terminal AudioChunk's handler runs, sees
    ``not self._progressive_playback_active``, and (pre-fix) opened a
    fresh PyAudio session — which implicitly closed the still-playing
    session via ``MonitorAudioService.start_streaming_session``'s
    "close existing" prelude. On Win11 MME this surfaced as audible
    chunk repeats. Fix: only ``chunk_index == 0`` legitimately opens a
    session; non-zero chunks arriving with the flag cleared are stale
    and dropped (with best-effort close on terminal chunks).
    """

    def test_stale_terminal_chunk_does_not_reopen_session(
        self, app_with_mocked_coordinator
    ):
        """The exact production race: chunk_index > 0, is_final=True,
        audio_data.size == 0 (TRUE_STREAM synthetic terminal),
        progressive_playback_active=False (just cleared by
        _play_generated_audio's skip-branch). MUST NOT call
        ``start_streaming_session`` (that's the spurious open) and MUST
        call ``stop_streaming_session`` for cleanup.
        """
        app, coordinator = app_with_mocked_coordinator
        # Simulate the post-skip state: flag cleared by dispatch path.
        app._progressive_playback_active = False

        terminal = _StubChunk(
            audio_data=np.zeros(0, dtype=np.float32),
            sample_rate=24000,
            chunk_index=2,  # non-zero — would trigger spurious open pre-fix
            is_final=True,
        )
        _drive(app, terminal)

        # The spurious open is the bug: must NOT fire.
        coordinator.start_streaming_session.assert_not_awaited()
        # play_audio_chunk must not fire either (audio_data.size == 0).
        coordinator.play_audio_chunk.assert_not_awaited()
        # Best-effort close: keep PyAudio resources clean even if
        # there's nothing to close.
        coordinator.stop_streaming_session.assert_awaited_once()
        # Flag stays False — no spurious set-True from a session reopen.
        assert app._progressive_playback_active is False

    def test_stale_non_terminal_chunk_silent_drop(
        self, app_with_mocked_coordinator
    ):
        """Defensive: a stale non-terminal chunk (chunk_index > 0, is_final=
        False) arriving with the flag cleared must drop silently — no open,
        no play, no close (no session to close)."""
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_active = False

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=3,
            is_final=False,
        )
        _drive(app, chunk)

        coordinator.start_streaming_session.assert_not_awaited()
        coordinator.play_audio_chunk.assert_not_awaited()
        coordinator.stop_streaming_session.assert_not_awaited()
        assert app._progressive_playback_active is False

    def test_chunk_zero_still_opens_when_flag_cleared(
        self, app_with_mocked_coordinator
    ):
        """The fix must NOT regress legitimate chunk-0 behavior. When a
        new generation's chunk 0 fires with flag cleared (the normal
        post-prior-gen state), the session opens and audio plays."""
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_active = False

        chunk0 = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            is_final=False,
        )
        _drive(app, chunk0)

        coordinator.start_streaming_session.assert_awaited_once()
        coordinator.play_audio_chunk.assert_awaited_once()
        assert app._progressive_playback_active is True


class TestProgressivePlaybackCancelEpoch:
    """Code-review MEDIUM-1 regression: cancel-vs-chunk race.

    Without the epoch guard, a chunk that the producer queued via
    ``run_coroutine_threadsafe`` BEFORE the cancel handler ran can land
    on the loop AFTER the cancel cleared ``_progressive_playback_active``
    — at which point the handler's ``if not self._progressive_playback_
    active:`` branch opens a fresh PyAudio session that nothing will
    ever close (no further chunks, no is_final). The cancel handler
    bumps ``_progressive_playback_epoch``; the trampoline captures the
    epoch at schedule time; the handler verifies the captured value
    matches under the lock and drops stale chunks.
    """

    def test_chunk_with_stale_epoch_is_dropped(
        self, app_with_mocked_coordinator
    ):
        app, coordinator = app_with_mocked_coordinator

        # Simulate: chunk was scheduled when epoch was 0; cancel ran and
        # bumped the epoch to 1; the chunk now arrives at the handler.
        chunk = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        app._progressive_playback_epoch = 1
        # Pass the captured (stale) epoch=0.
        asyncio.run(
            app._handle_progressive_chunk_async(chunk, epoch=0)
        )

        # Stale chunk must be dropped — no session opened, no audio
        # written, no leak.
        coordinator.start_streaming_session.assert_not_awaited()
        coordinator.play_audio_chunk.assert_not_awaited()
        coordinator.stop_streaming_session.assert_not_awaited()
        assert app._progressive_playback_active is False

    def test_chunk_with_current_epoch_is_processed(
        self, app_with_mocked_coordinator
    ):
        """Defensive boundary: a chunk whose captured epoch matches the
        current value MUST process normally — the epoch guard only
        rejects stale chunks."""
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_epoch = 5

        chunk = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        asyncio.run(
            app._handle_progressive_chunk_async(chunk, epoch=5)
        )

        coordinator.start_streaming_session.assert_awaited_once()
        coordinator.play_audio_chunk.assert_awaited_once()
        assert app._progressive_playback_active is True

    def test_legacy_none_epoch_skips_check(
        self, app_with_mocked_coordinator
    ):
        """``epoch=None`` is the direct-test calling convention used by
        the rest of this file; it must skip the epoch check entirely so
        existing tests keep working without re-plumbing."""
        app, coordinator = app_with_mocked_coordinator
        app._progressive_playback_epoch = 99

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        asyncio.run(
            app._handle_progressive_chunk_async(chunk, epoch=None)
        )

        coordinator.start_streaming_session.assert_awaited_once()
        coordinator.play_audio_chunk.assert_awaited_once()


class TestProgressivePlaybackTrampoline:
    """Code-review MEDIUM-2 regression: the synchronous trampoline
    ``_on_audio_chunk_ready`` is the production-only path between the
    producer thread and the orchestrator's event loop. The other tests
    in this file ``await`` ``_handle_progressive_chunk_async`` directly,
    so the trampoline's loop-availability guard, epoch capture, and
    ``run_coroutine_threadsafe`` scheduling have zero coverage without
    these tests.
    """

    def test_trampoline_short_circuits_when_loop_missing(
        self, app_with_mocked_coordinator
    ):
        """If ``self.loop`` is ``None`` (pre-init or post-shutdown), the
        trampoline must return without raising and without scheduling
        anything onto a non-existent loop."""
        app, coordinator = app_with_mocked_coordinator
        app.loop = None

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        # Must not raise.
        app._on_audio_chunk_ready(chunk)
        coordinator.start_streaming_session.assert_not_awaited()

    def test_trampoline_short_circuits_when_loop_closed(
        self, app_with_mocked_coordinator, monkeypatch
    ):
        """Closed loop = same handling as missing loop. Avoids the
        ``run_coroutine_threadsafe(... ,closed_loop)`` raise that would
        otherwise surface on shutdown."""
        app, coordinator = app_with_mocked_coordinator

        class _ClosedLoop:
            def is_closed(self):
                return True

        app.loop = _ClosedLoop()

        scheduled: list = []

        def fake_rcts(coro, loop):
            scheduled.append((coro, loop))
            coro.close()
            from unittest.mock import MagicMock
            return MagicMock()

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_rcts)

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        app._on_audio_chunk_ready(chunk)
        assert scheduled == [], (
            f"Trampoline must not schedule on a closed loop; got {scheduled}"
        )

    def test_trampoline_schedules_with_captured_epoch(
        self, app_with_mocked_coordinator, monkeypatch
    ):
        """The trampoline must call ``run_coroutine_threadsafe`` with the
        handler coroutine targeted at ``self.loop``, AND the epoch it
        threads in must be the value at schedule time (so a later cancel
        bump leaves the in-flight chunk's captured value stale and
        droppable).
        """
        from unittest.mock import MagicMock

        app, coordinator = app_with_mocked_coordinator

        class _RunningLoop:
            def is_closed(self):
                return False

        app.loop = _RunningLoop()
        app._progressive_playback_epoch = 7

        scheduled: list = []

        def fake_rcts(coro, loop):
            scheduled.append((coro, loop))
            coro.close()
            return MagicMock()

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_rcts)

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        app._on_audio_chunk_ready(chunk)

        assert len(scheduled) == 1
        scheduled_coro, scheduled_loop = scheduled[0]
        assert scheduled_loop is app.loop
        # Coroutine name confirms we scheduled the handler (not e.g. the
        # trampoline itself).
        assert (
            "_handle_progressive_chunk_async" in scheduled_coro.__qualname__
        )

    def test_trampoline_swallows_scheduling_exception(
        self, app_with_mocked_coordinator, monkeypatch, caplog
    ):
        """A raise inside ``run_coroutine_threadsafe`` (e.g. loop being
        torn down between the ``is_closed()`` check and the schedule
        call) must NOT propagate back to the producer thread — the
        trampoline's outer try/except must log and swallow.
        """
        import logging as _logging

        app, coordinator = app_with_mocked_coordinator

        class _RunningLoop:
            def is_closed(self):
                return False

        app.loop = _RunningLoop()

        def fake_rcts(coro, loop):
            coro.close()
            raise RuntimeError("loop tearing down")

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_rcts)

        chunk = _StubChunk(
            audio_data=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        with caplog.at_level(_logging.ERROR, logger=app.logger.name):
            # Must not raise.
            app._on_audio_chunk_ready(chunk)

        assert any(
            "Failed to schedule progressive-playback chunk handler"
            in r.getMessage()
            for r in caplog.records
        )
