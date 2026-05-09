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


class TestProgressivePlaybackSampleRateAndFailure:
    """Story 17.3 AC #3 — sample-rate handshake + open-failure graceful
    degradation (Task 3.3)."""

    def test_session_open_failure_falls_through_to_batch(
        self, app_with_mocked_coordinator
    ):
        """When ``start_streaming_session`` raises (PyAudio error, device
        unavailable), the orchestrator must clear the flag and discard the
        chunk so the eventual ``_play_generated_audio`` call falls through
        to the existing batch dispatch path (NFR7-style graceful
        degradation)."""
        app, coordinator = app_with_mocked_coordinator
        coordinator.start_streaming_session.side_effect = RuntimeError(
            "simulated PyAudio open failure"
        )

        chunk0 = _StubChunk(
            audio_data=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
        )
        _drive(app, chunk0)

        # Open was attempted exactly once and raised.
        assert coordinator.start_streaming_session.await_count == 1
        # Flag stays False so _play_generated_audio's skip-check is a
        # no-op and the assembled buffer plays via the batch path.
        assert app._progressive_playback_active is False
        # Chunk is discarded — no play_audio_chunk on a session that
        # failed to open.
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
