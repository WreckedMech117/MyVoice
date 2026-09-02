"""AC12b / F4 — session-keyed origin gating in _handle_progressive_chunk_async.

An API-origin session (id in ``_api_origin_sessions``) must be published to the
StreamHub and short-circuit the handler WITHOUT opening the desktop device
session; a GUI/None session must fall through to the device path and NOT be
published to the hub.

The handler is exercised on a bare instance built via ``__new__`` with only the
attributes the gate path and the first device call touch — avoiding a full
MyVoiceApp/QApplication bring-up.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from myvoice.app import MyVoiceApp
from myvoice.services.api_server.stream_hub import StreamHub


def _make_app():
    app = MyVoiceApp.__new__(MyVoiceApp)
    app.logger = logging.getLogger("test_origin_gating")
    app._stream_hub = StreamHub()
    app._api_origin_sessions = set()
    app._progressive_playback_lock = None
    app._progressive_playback_epoch = 0
    app._progressive_playback_active = False
    app._progressive_playback_sample_rate = 0
    # PyQt6's QObject.__getattr__ raises (rather than returning a default) for
    # a missing attr on a __new__'d instance, so pre-set the optional ones the
    # device path reads via getattr.
    app._pending_progressive_text_length = None
    app._audio_coordinator = AsyncMock()
    return app


def _chunk(session_id, chunk_index=0, is_final=False, size=10):
    audio = np.zeros(size, dtype=np.float32)
    return SimpleNamespace(
        session_id=session_id,
        chunk_index=chunk_index,
        is_final=is_final,
        audio_data=audio,
        sample_rate=24000,
        text_segment="",
    )


@pytest.mark.asyncio
async def test_api_session_publishes_and_suppresses_device():
    app = _make_app()
    sid = "api-1"
    app._api_origin_sessions.add(sid)
    queue = app._stream_hub.subscribe(sid)

    await app._handle_progressive_chunk_async(
        _chunk(sid, chunk_index=0, is_final=True, size=10), epoch=None
    )

    # Published to the HTTP fan-out...
    audio_bytes, is_final = queue.get_nowait()
    assert is_final is True
    assert len(audio_bytes) == 20  # 10 samples * int16

    # ...and the desktop device was never touched.
    app._audio_coordinator.start_streaming_session.assert_not_called()
    app._audio_coordinator.play_audio_chunk.assert_not_called()
    app._audio_coordinator.stop_streaming_session.assert_not_called()


@pytest.mark.asyncio
async def test_gui_session_reaches_device_and_is_not_published():
    app = _make_app()
    app._audio_coordinator.start_streaming_session.return_value = {
        "monitor": "m",
        "virtual": "v",
    }
    sid = "gui-1"  # NOT registered as API-origin
    queue = app._stream_hub.subscribe(sid)

    await app._handle_progressive_chunk_async(
        _chunk(sid, chunk_index=0, is_final=False, size=10), epoch=None
    )

    # Gate not taken -> nothing published to the hub.
    assert queue.empty()
    # Device path reached.
    app._audio_coordinator.start_streaming_session.assert_called_once()
    app._audio_coordinator.play_audio_chunk.assert_called_once()
