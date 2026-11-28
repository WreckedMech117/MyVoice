"""Field crash on build 59 (tester log, 2026-11-26 and 2026-11-27).

A TRUE_STREAM generation stalled past its 90 s timeout with the talker thread
still inside the compiled model. The fallback chain then ran the same model
from a second thread (logged "Offset increment outside graph capture"), then
from a third, and the process died. These pin that a model still busy with a
stalled generation is never started again.
"""
import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.services import qwen_tts_service as module
from myvoice.services.qwen_tts_service import (
    QwenModelType, QwenTTSRequest, QwenTTSService,
)
from myvoice.services.tts_streaming.streaming_mode import StreamingMode


@pytest.fixture
def service():
    svc = QwenTTSService(device="cpu", dtype="float32", app_settings=AppSettings())
    svc._generate_true_stream = AsyncMock()
    svc._generate_streaming = AsyncMock()
    svc._generate = AsyncMock()
    return svc


@pytest.fixture
def stalled_thread():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    yield thread, release
    release.set()
    thread.join(timeout=2)


def _dispatch(service, mode=StreamingMode.TRUE_STREAM):
    request = QwenTTSRequest(text="sounds good to me", model_type=QwenModelType.BASE)
    return asyncio.new_event_loop().run_until_complete(
        service._dispatch_by_streaming_mode(request, mode)
    )


def test_fallback_does_not_run_the_model_beside_a_stalled_talker(service, stalled_thread):
    """The reported sequence: TRUE_STREAM times out leaving its talker alive;
    SENTENCE_STREAM and BATCH must not be attempted."""
    thread, _ = stalled_thread

    async def true_stream_times_out(request):
        service._stalled_engine_thread = thread  # what the timeout handler records
        raise RuntimeError("TRUE_STREAM talker/worker did not complete within 60.0s")

    service._generate_true_stream.side_effect = true_stream_times_out
    failed = []
    service.set_generation_failed_callback(failed.append)

    response = _dispatch(service)

    assert not response.success
    assert response.error_message == module._ENGINE_STALLED_MESSAGE
    service._generate_streaming.assert_not_awaited()
    service._generate.assert_not_awaited()
    assert failed == [module._ENGINE_STALLED_MESSAGE]


def test_next_line_is_refused_while_the_talker_is_still_running(service, stalled_thread):
    thread, _ = stalled_thread
    service._stalled_engine_thread = thread

    response = _dispatch(service)

    assert not response.success
    service._generate_true_stream.assert_not_awaited()


def test_generations_resume_once_the_stalled_talker_returns(service, stalled_thread):
    thread, release = stalled_thread
    service._stalled_engine_thread = thread
    release.set()
    thread.join(timeout=2)
    service._generate_true_stream.return_value = "spoken"

    assert _dispatch(service) == "spoken"
    assert service._stalled_engine_thread is None


def test_ordinary_failure_still_falls_back(service):
    """No stalled talker: the fallback chain is unchanged."""
    service._generate_true_stream.side_effect = RuntimeError("0 audio chunks")
    service._generate_streaming.return_value = "sentence-stream"

    assert _dispatch(service) == "sentence-stream"
