"""Clear Comms clicked (interrupt mode) while a generation is STREAMING.

Lucy's report (2026-11): "Cannot hear own voice (playback) after using
repeat or the clear text button." Clear Comms dispatches through the same
``_play_generated_audio`` path as Replay; mid-stream it also hit this
Clear-Comms-only defect:

``_interrupt_active_playback_for_clear_comms`` called
``stop_all_playback`` (batch tasks only) but left the progressive streaming
session and ``_progressive_playback_active`` untouched. The Clear Comms
dispatch therefore took ``_play_generated_audio``'s "progressive playback
already active" skip-branch:

  1. the Clear Comms clip never played;
  2. the skip-branch's deferred flag-clear made every remaining chunk of
     the stream drop as stale -- own voice cut out mid-sentence;
  3. the generation-complete dispatch then saw the flag cleared and played
     the whole generation again through the batch path.

The fix mirrors the Stop button's streaming block (epoch bump +
``stop_streaming_session`` + flag clear) WITHOUT cancelling generation,
per the documented interrupt contract.

Real ``MyVoiceApp`` handlers drive the exact click sequence; only the
``AudioCoordinator`` is mocked (pattern: test_app_progressive_playback.py).
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

import soundfile as sf  # noqa: E402


@dataclass
class _StubChunk:
    audio_data: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool = False
    text_segment: str = ""
    session_id: "str | None" = None


def _wav(level: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(2400, level, np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunk(i: int, final: bool = False) -> _StubChunk:
    return _StubChunk(np.full(100, 0.1 * (i + 1), np.float32), 24000, i, final)


CLIP = _wav(0.5)
GENERATION = _wav(0.9)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp):
    from myvoice.app import MyVoiceApp
    from myvoice.models.app_settings import AppSettings

    app = MyVoiceApp(qapp)
    coord = AsyncMock()
    coord.start_streaming_session = AsyncMock(return_value={"monitor": "m", "virtual": "v"})
    coord.play_audio_chunk = AsyncMock(return_value={"monitor": True, "virtual": True})
    coord.stop_streaming_session = AsyncMock(return_value={"monitor": True, "virtual": True})
    coord.stop_all_playback = AsyncMock(return_value=0)

    batch = []

    async def play_monitor_audio(audio_data, device=None, volume=1.0):
        batch.append(audio_data)
        return MagicMock(playback_id=f"task-{len(batch)}")

    monitor = MagicMock()
    monitor.enumerate_monitor_devices = AsyncMock(
        return_value=[MagicMock(device_id="pyaudio_20", name="Speakers")]
    )
    monitor.play_monitor_audio = play_monitor_audio
    coord.monitor_service = monitor

    app._audio_coordinator = coord
    app._app_settings = AppSettings()
    app._app_settings.virtual_microphone_device_id = None  # monitor-only batch path
    app._resolve_clear_comms_wav_bytes = lambda *a: CLIP
    yield app, coord, batch


async def _click_clear_comms_mid_stream(app):
    """chunks 0,1 stream -> Clear Comms click -> chunks 2,3(final) arrive."""
    app.loop = asyncio.get_running_loop()
    handle = app._handle_progressive_chunk_async
    for i in (0, 1):
        await handle(_chunk(i), app._progressive_playback_epoch)

    # The real click handler body: interrupt, then dispatch the clip.
    app._interrupt_active_playback_for_clear_comms()
    await app._play_generated_audio(CLIP)
    await asyncio.sleep(0.05)  # drain ensure_future'd stop + deferred clear

    for i, final in ((2, False), (3, True)):
        await handle(_chunk(i, final), app._progressive_playback_epoch)


class TestClearCommsInterruptsStreaming:
    def test_clip_plays_when_clicked_mid_stream(self, app):
        app_, _, batch = app
        asyncio.run(_click_clear_comms_mid_stream(app_))
        assert batch and batch[0] == CLIP

    def test_streaming_session_is_stopped_on_click(self, app):
        app_, coord, _ = app

        async def run():
            app_.loop = asyncio.get_running_loop()
            for i in (0, 1):
                await app_._handle_progressive_chunk_async(_chunk(i), app_._progressive_playback_epoch)
            app_._interrupt_active_playback_for_clear_comms()
            await asyncio.sleep(0.05)

        asyncio.run(run())
        coord.stop_streaming_session.assert_awaited()
        assert app_._progressive_playback_active is False

    def test_rest_of_interrupted_stream_does_not_reach_device(self, app):
        """Remaining chunks are dropped by the epoch bump, not written over
        the clip (they belong to the interrupted playback)."""
        app_, coord, _ = app
        asyncio.run(_click_clear_comms_mid_stream(app_))
        # chunks 0 and 1 only
        assert coord.play_audio_chunk.await_count == 2

    def test_completed_generation_dispatches_once_after_clip(self, app):
        """Documented contract: generation is not cancelled; its complete
        audio plays after the clip -- exactly once."""
        app_, _, batch = app

        async def run():
            await _click_clear_comms_mid_stream(app_)
            await app_._play_generated_audio(GENERATION)  # generation-complete dispatch

        asyncio.run(run())
        assert batch == [CLIP, GENERATION]
