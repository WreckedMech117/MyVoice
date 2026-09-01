"""Story 20.2 AC #2 / AC #3 — compile-priming suppression, across EVERY
channel that reaches a user.

Story 18.4 shipped a compile-priming generation that dispatches a **real**
TRUE_STREAM utterance. Its audio safety rested on two claims in its own
docstring — that priming runs before ``set_audio_chunk_ready_callback`` wires a
consumer, and that "the generation is short enough that any audible artifact is
bounded". The first is a startup ordering race (already lost once in
production); the second is not a safety property at all.

Story 20.2 makes warm-cache launches prime too — every launch instead of only
the first-ever — so the race would become recurring.

**What the 20.2 review pass corrected in this suite.** The first version of
these tests asserted only that the progressive-chunk callback stayed silent.
That is one channel of four, and the suite reported green while the loudest
channel leaked: TRUE_STREAM calls ``audio_coordinator.play_dual_stream`` — the
monitor device and the virtual microphone — directly, with no request
consultation. The rig even stubbed that call and never counted it. The
channels, all now covered here:

  1. ``_audio_chunk_ready_callback``  → progressive playback
  2. ``audio_coordinator.play_dual_stream`` → the user's speakers + virtual mic
  3. ``_save_audio_to_cache`` → ``myvoice_current.wav``, i.e. Replay Last
     (and, transitively, ``_generation_complete_callback``)
  4. ``SessionRegistry.create_session`` → ``_saveable`` / ``_focal``, i.e. the
     Save button and the Stop / Clear Comms focal-cancel paths

Rows:

  1. ``test_priming_reaches_no_user_facing_channel`` (AC #2) — all four
     channels wired **before** priming; all four untouched afterwards, with a
     producer-side metric proving chunks were genuinely produced.
  2. ``test_cold_path_warmup_reaches_no_user_facing_channel`` (AC #2, "the
     existing cold-path priming is brought under the same mechanism").
  3. ``test_unsuppressed_request_uses_every_channel`` — the control that makes
     row 1 meaningful: without suppression, all four channels fire.
  4. ``test_user_generation_during_priming_still_reaches_consumer`` (AC #3 +
     review F2) — a user generation dispatched mid-priming gets its audio
     **and** keeps its cancel bookkeeping.
  5. ``test_priming_does_not_clear_a_pending_user_cancel`` (review F2).
  6. ``TestSuppressionMechanismIsHardToBypass`` — a source invariant over
     *every* user-facing channel call site, not one attribute name, plus the
     F6 trip-wire.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Mirror the Story 17.3 TRUE_STREAM suite's gate so torch-less / Qt-less envs
# skip cleanly.
pytest.importorskip("PyQt6")

from myvoice.observability import metrics
from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService
from myvoice.services.sessions import SessionRegistry
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService
from myvoice.services.qwen_tts_service import (
    AudioChunk,
    QwenModelType,
    QwenTTSRequest,
    QwenTTSService,
)


SERVICE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "myvoice"
    / "services"
    / "qwen_tts_service.py"
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def coordinator() -> AudioCoordinator:
    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_all_playback = AsyncMock(return_value=0)
    monitor.play_monitor_audio = AsyncMock()
    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_all_virtual_microphone_playback = AsyncMock(return_value=0)
    virtual.play_virtual_microphone = AsyncMock()

    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = monitor
    coord.virtual_service = virtual
    # Spied, NOT silently stubbed: every test that runs a suppressed generation
    # asserts this was never called. The pre-review rig stubbed it and never
    # looked, which is how the leak stayed green.
    coord.play_dual_stream = AsyncMock(return_value=MagicMock())
    return coord


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def registry(qapp) -> SessionRegistry:
    return SessionRegistry()


@pytest.fixture
def emit_records():
    """Producer-side ``progressive_chunk_emit_ms`` records.

    These fire inside ``_wrapped_post`` *before* any sink is consulted, so a
    non-zero count proves the priming generation genuinely produced audio
    chunks — which is what makes "no channel was touched" a real assertion
    rather than a vacuous one.
    """
    captured: List[metrics.MetricRecord] = []

    def listener(record: metrics.MetricRecord) -> None:
        if record.name == "progressive_chunk_emit_ms":
            captured.append(record)

    unsub = metrics.add_listener(listener)
    try:
        yield captured
    finally:
        unsub()


# --------------------------------------------------------------------------- #
# Rig
# --------------------------------------------------------------------------- #


def _build_true_stream_service(registry, coordinator):
    """A ``QwenTTSService`` wired for TRUE_STREAM dispatch with no real model."""
    from myvoice.models.app_settings import AppSettings
    from myvoice.services.core.base_service import ServiceStatus

    settings = AppSettings(
        streaming_mode_override="true_stream", tts_compile="auto"
    )
    service = QwenTTSService(
        audio_coordinator=coordinator,
        device="cpu",
        dtype="float32",
        session_registry=registry,
        app_settings=settings,
    )
    service.status = ServiceStatus.RUNNING

    from concurrent.futures import ThreadPoolExecutor

    service._executor = ThreadPoolExecutor(max_workers=2)
    service._request_semaphore = asyncio.Semaphore(1)

    mock_model = MagicMock()
    service._model_registry.ensure_model_loaded = AsyncMock(
        return_value=(True, None)
    )
    service._model_registry.get_loaded_model = MagicMock(return_value=mock_model)
    service._model_registry.device = "cpu"
    return service, mock_model


class ChannelSpies:
    """Records every user-facing channel a generation can reach.

    Wired **before** the generation under test runs, so the ordering race the
    pre-20.2 docstring relied on is deliberately violated in every row.
    """

    def __init__(self, service, coordinator, registry, monkeypatch):
        self.service = service
        self.coordinator = coordinator
        self.registry = registry
        self.chunks: List[AudioChunk] = []
        self.cache_writes: List[Any] = []
        self.created_sessions: List[str] = []
        self.completed_files: List[Path] = []

        service.set_audio_chunk_ready_callback(self.chunks.append)
        service.set_generation_complete_callback(self.completed_files.append)

        real_cache = service._save_audio_to_cache

        def _spy_cache(audio_data, sample_rate):
            self.cache_writes.append((np.asarray(audio_data).size, sample_rate))
            return Path("cache-write-happened.wav")

        monkeypatch.setattr(service, "_save_audio_to_cache", _spy_cache)
        assert real_cache is not None  # sanity: the attribute exists to patch

        real_create = registry.create_session

        def _spy_create(*a, **kw):
            sid = real_create(*a, **kw)
            self.created_sessions.append(sid)
            return sid

        monkeypatch.setattr(registry, "create_session", _spy_create)

    # -- assertions -------------------------------------------------------- #

    def touched(self) -> Dict[str, bool]:
        return {
            "audio_chunk_ready_callback": bool(self.chunks),
            "play_dual_stream": self.coordinator.play_dual_stream.called,
            "save_audio_to_cache": bool(self.cache_writes),
            "generation_complete_callback": bool(self.completed_files),
            "registry_create_session": bool(self.created_sessions),
            "registry_focal_session": self.registry.focal_session_id is not None,
            "registry_saveable_session": (
                self.registry.saveable_session_id is not None
            ),
        }

    def assert_nothing_reached_the_user(self, what: str) -> None:
        touched = {k: v for k, v in self.touched().items() if v}
        assert touched == {}, (
            f"{what} reached user-facing channel(s): {sorted(touched)}. "
            "A suppressed generation must reach NONE of them "
            "(Story 20.2 AC #2)."
        )
        assert self.service._current_session_id is None, (
            f"{what} published a session id on the shared "
            "_current_session_id singleton"
        )


def _fake_talker_factory(token_count: int):
    def builder(model, request, streamer):
        def run():
            for tok in range(token_count):
                streamer.put([tok])
                time.sleep(0.001)
            streamer.end()

        return run

    return builder


def _fake_decode_fn(model):
    return lambda chunk: np.array([t * 0.01 for t in chunk], dtype=np.float32)


def _install_fakes(service, monkeypatch, *, tokens: int = 60, talker=None):
    monkeypatch.setattr(
        service,
        "_build_true_stream_talker",
        talker if talker is not None else _fake_talker_factory(tokens),
    )
    monkeypatch.setattr(service, "_build_true_stream_decode_fn", _fake_decode_fn)


def _run_with_qt_drain(qapp, coro_factory, *, settle_iterations: int = 50):
    """Run ``coro_factory()`` on a fresh loop while pumping the Qt event queue
    so the SessionRegistry's queued-connection posts land on the main thread."""

    async def runner():
        stop_evt = asyncio.Event()

        async def drainer():
            while not stop_evt.is_set():
                qapp.processEvents()
                await asyncio.sleep(0.005)

        drain_task = asyncio.create_task(drainer())
        try:
            return await coro_factory()
        finally:
            stop_evt.set()
            await drain_task

    result = asyncio.run(runner())
    for _ in range(settle_iterations):
        qapp.processEvents()
        time.sleep(0.005)
    return result


def _user_request(session_id: str = "user-session") -> QwenTTSRequest:
    return QwenTTSRequest(
        text="the user's actual utterance",
        language="Auto",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        streaming=True,
        session_id=session_id,
    )


# --------------------------------------------------------------------------- #
# AC #2 — priming reaches no user-facing channel
# --------------------------------------------------------------------------- #


def test_priming_reaches_no_user_facing_channel(
    qapp, registry, coordinator, monkeypatch, emit_records
):
    """AC #2 — every channel wired BEFORE priming; all stay untouched.

    The pre-20.2 docstring's "priming runs before the callback is wired"
    ordering claim is exactly the precondition this test violates.
    """
    service, _ = _build_true_stream_service(registry, coordinator)
    _install_fakes(service, monkeypatch)
    spies = ChannelSpies(service, coordinator, registry, monkeypatch)

    _run_with_qt_drain(qapp, lambda: service._run_compile_priming())

    spies.assert_nothing_reached_the_user("compile priming")
    # Non-vacuity: the producer really did emit chunks, they just went nowhere.
    assert len(emit_records) >= 1, (
        "Expected the priming generation to actually produce audio chunks "
        "(progressive_chunk_emit_ms); zero would make every assertion above "
        "vacuous"
    )


def test_cold_path_warmup_reaches_no_user_facing_channel(
    qapp, registry, coordinator, monkeypatch, emit_records
):
    """AC #2 — "the existing cold-path priming is brought under the same
    mechanism". Drives the full ``warmup_compile_async`` cold path
    (``is_warm`` False → prime → ``mark_warm``) with every channel wired first.
    """
    import torch

    service, mock_model = _build_true_stream_service(registry, coordinator)
    mock_model.model.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mock_model.model.dtype = torch.bfloat16
    _install_fakes(service, monkeypatch)

    monkeypatch.delenv("MYVOICE_DISABLE_COMPILE_WARMUP", raising=False)
    monkeypatch.delenv("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm", lambda key: False
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm", mark_warm_mock
    )

    spies = ChannelSpies(service, coordinator, registry, monkeypatch)

    _run_with_qt_drain(qapp, lambda: service.warmup_compile_async())

    spies.assert_nothing_reached_the_user("cold-path compile warmup")
    assert len(emit_records) >= 1, "cold-path priming produced no chunks at all"
    mark_warm_mock.assert_called_once()


def test_unsuppressed_request_uses_every_channel(
    qapp, registry, coordinator, monkeypatch
):
    """Control — without suppression the SAME rig drives every channel.

    This is what makes the two rows above meaningful: it proves the rig can
    reach each channel, so "untouched" is a property of suppression and not of
    a rig that never wired anything up.
    """
    service, _ = _build_true_stream_service(registry, coordinator)
    _install_fakes(service, monkeypatch)
    spies = ChannelSpies(service, coordinator, registry, monkeypatch)

    response = _run_with_qt_drain(
        qapp, lambda: service._generate_true_stream(_user_request())
    )

    assert response.success is True, f"dispatch failed: {response.error_message}"
    untouched = [k for k, v in spies.touched().items() if not v]
    assert untouched == [], (
        "The control generation should exercise every user-facing channel; "
        f"these stayed silent, so the suppression rows they guard are "
        f"vacuous: {untouched}"
    )
    coordinator.play_dual_stream.assert_called()
    assert len([c for c in spies.chunks if c.is_final]) == 1


# --------------------------------------------------------------------------- #
# AC #3 + review F2 — a user generation during priming
# --------------------------------------------------------------------------- #


def test_user_generation_during_priming_still_reaches_consumer(
    qapp, registry, coordinator, monkeypatch
):
    """AC #3 — priming is in flight when the user's generation is dispatched.

    The user's chunks must arrive, AND (review F2) the user's cancel
    bookkeeping must survive priming's ``finally``. ``_current_session_id`` and
    ``_current_generation_task`` are process-wide singletons that a user
    request claims *before* it parks on the request semaphore, so an
    unconditional clear in the prime's finally would leave Stop with no session
    and no task while the user's audio keeps playing.
    """
    service, _ = _build_true_stream_service(registry, coordinator)

    priming_talker_started = threading.Event()
    release_priming = threading.Event()

    def talker_builder(model, request, streamer):
        is_priming = getattr(request, "suppress_audio_output", False)

        def run():
            if is_priming:
                priming_talker_started.set()
                release_priming.wait(timeout=10.0)
            for tok in range(40):
                streamer.put([tok])
                time.sleep(0.001)
            streamer.end()

        return run

    _install_fakes(service, monkeypatch, talker=talker_builder)
    spies = ChannelSpies(service, coordinator, registry, monkeypatch)

    observed: Dict[str, Any] = {}

    async def scenario():
        priming_task = asyncio.create_task(service._run_compile_priming())

        deadline = time.monotonic() + 10.0
        while not priming_talker_started.is_set():
            assert time.monotonic() < deadline, "priming talker never started"
            await asyncio.sleep(0.01)

        # Priming holds the semaphore. The user's dispatch claims its
        # bookkeeping and then parks behind it.
        user_task = asyncio.create_task(
            service._generate_true_stream(_user_request())
        )
        await asyncio.sleep(0.05)
        assert not user_task.done()
        observed["sid_while_parked"] = service._current_session_id
        observed["task_while_parked"] = service._current_generation_task

        release_priming.set()

        # Let priming's finally run while the user's generation is still live.
        await priming_task
        observed["sid_after_prime"] = service._current_session_id
        observed["task_after_prime"] = service._current_generation_task

        return await user_task

    user_response = _run_with_qt_drain(qapp, scenario)

    assert user_response.success is True, (
        f"user dispatch failed: {user_response.error_message}"
    )

    # --- AC #3: the user's audio reached the user ------------------------- #
    data_chunks = [c for c in spies.chunks if not c.is_final]
    assert len(data_chunks) >= 1, (
        "The user's generation was dispatched while priming was in flight and "
        "its audio never reached the consumer — the suppression mechanism is "
        "not scoped to the priming generation"
    )
    assert len([c for c in spies.chunks if c.is_final]) == 1
    assert {c.session_id for c in spies.chunks} == {"user-session"}, (
        "A chunk from a session other than the user's reached the consumer: "
        f"{sorted({str(c.session_id) for c in spies.chunks})}"
    )
    coordinator.play_dual_stream.assert_called()
    assert all(
        call.kwargs.get("session_id") == "user-session"
        for call in coordinator.play_dual_stream.call_args_list
    ), "play_dual_stream was called for a session other than the user's"
    assert spies.created_sessions == ["user-session"], (
        "exactly one session — the user's — should have been registered; got "
        f"{spies.created_sessions}"
    )

    # --- review F2: the user's cancel bookkeeping survived ---------------- #
    assert observed["sid_while_parked"] == "user-session"
    assert observed["task_while_parked"] is not None
    assert observed["sid_after_prime"] == "user-session", (
        "priming's finally cleared the parked user generation's session id; "
        "Stop / Clear Comms would find no in-flight session and the user's "
        "audio would not stop"
    )
    assert observed["task_after_prime"] is observed["task_while_parked"], (
        "priming's finally cleared the parked user generation's task handle; "
        "cancel_generation() would have nothing to cancel"
    )


def test_priming_does_not_clear_a_pending_user_cancel(
    qapp, registry, coordinator, monkeypatch
):
    """Review F2 — ``_cancel_requested`` is shared too.

    A suppressed generation entering while the user has a cancel pending must
    not reset the flag to False, or the user's Stop is silently swallowed.
    """
    service, _ = _build_true_stream_service(registry, coordinator)
    _install_fakes(service, monkeypatch, tokens=10)

    service._cancel_requested = True

    _run_with_qt_drain(qapp, lambda: service._run_compile_priming())

    assert service._cancel_requested is True, (
        "the compile prime reset the shared _cancel_requested flag; a Stop "
        "pressed just before priming started would be swallowed"
    )


# --------------------------------------------------------------------------- #
# Source invariants — the mechanism is hard to bypass
# --------------------------------------------------------------------------- #


def _code_lines() -> List[str]:
    """Source lines with comments stripped, so a call named in prose does not
    register as a call site."""
    out = []
    for line in SERVICE_PATH.read_text(encoding="utf-8").splitlines():
        out.append(line.split("#", 1)[0])
    return out


class TestSuppressionMechanismIsHardToBypass:
    """AC #2's latitude asks for the mechanism "hardest to get wrong from a
    future caller's perspective".

    The first version of this class grepped a single attribute name,
    ``self._audio_chunk_ready_callback``. That is structurally blind to the
    three channels that do not go through the sink — and the loudest of them
    (``play_dual_stream``) was leaking while this class reported green. The
    invariant below covers every user-facing channel instead.
    """

    # channel call-site marker -> how far back to look for the guard
    USER_FACING_CHANNELS = {
        "self.audio_coordinator.play_dual_stream(": 16,
        "self._save_audio_to_cache(": 4,
        "self._session_registry.create_session(": 4,
    }

    def test_every_user_facing_channel_call_site_is_suppression_gated(self):
        lines = _code_lines()
        offenders = []
        seen = {marker: 0 for marker in self.USER_FACING_CHANNELS}

        for idx, line in enumerate(lines):
            for marker, lookback in self.USER_FACING_CHANNELS.items():
                if marker not in line:
                    continue
                seen[marker] += 1
                window = "\n".join(lines[max(0, idx - lookback): idx + 1])
                if "suppressed" not in window:
                    offenders.append((idx + 1, line.strip()))

        # Guard against the invariant silently covering nothing (e.g. after a
        # rename): each channel must still have call sites to check.
        missing = [m for m, n in seen.items() if n == 0]
        assert missing == [], (
            "This invariant found no call sites for: "
            + ", ".join(missing)
            + ". It has stopped protecting those channels — update the marker "
            "or the guard, do not delete the row."
        )

        assert offenders == [], (
            "Every channel that can reach a user must be gated on the "
            "per-request suppression flag, so a compile-priming generation "
            "cannot reach the speakers, the replay cache, or the session "
            "registry (Story 20.2 AC #2). Ungated call sites: "
            + "; ".join(f"line {n}: {t}" for n, t in offenders)
        )

    def test_no_emit_site_reads_the_raw_callback(self):
        """Narrow companion invariant: the chunk callback is reachable only
        through ``_audio_chunk_sink``.

        This one guards a single channel by construction. It is kept because it
        is the only channel with many call sites, but it is explicitly NOT the
        whole guarantee — see the test above.
        """
        offenders = []
        for lineno, code in enumerate(_code_lines(), start=1):
            if "self._audio_chunk_ready_callback" not in code:
                continue
            stripped = code.strip()
            sanctioned = (
                stripped.startswith("self._audio_chunk_ready_callback:")
                or stripped == "self._audio_chunk_ready_callback = callback"
                or stripped == "return self._audio_chunk_ready_callback"
            )
            if not sanctioned:
                offenders.append((lineno, stripped))

        assert offenders == [], (
            "Every audio-chunk emit site must resolve its consumer through "
            "QwenTTSService._audio_chunk_sink(request). Direct reads found at: "
            + "; ".join(f"line {n}: {t}" for n, t in offenders)
        )

    def test_audio_chunk_sink_returns_none_for_suppressed_request(
        self, registry, coordinator
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        sentinel = MagicMock()
        service.set_audio_chunk_ready_callback(sentinel)

        normal = _user_request()
        suppressed = QwenTTSRequest(
            text="Hello world.", suppress_audio_output=True
        )

        assert service._audio_chunk_sink(normal) is sentinel
        assert service._audio_chunk_sink(suppressed) is None
        assert service._is_suppressed(normal) is False
        assert service._is_suppressed(suppressed) is True

    def test_priming_request_carries_the_flag(self, registry, coordinator, monkeypatch):
        """The priming dispatch must actually set ``suppress_audio_output``."""
        service, _ = _build_true_stream_service(registry, coordinator)
        seen: List[QwenTTSRequest] = []

        async def _capture(request, mode):
            seen.append(request)
            return MagicMock(success=True)

        monkeypatch.setattr(service, "_dispatch_by_streaming_mode", _capture)
        asyncio.run(service._run_compile_priming())

        assert len(seen) == 1
        assert seen[0].suppress_audio_output is True
        assert seen[0].text == QwenTTSService._COMPILE_PRIMING_TEXT

    def test_priming_refuses_to_dispatch_when_the_flag_is_lost(
        self, registry, coordinator, monkeypatch
    ):
        """Review F6 — ``_is_suppressed`` fails OPEN (a request it does not
        recognise is treated as user-facing, because silencing a real user is
        the worse failure). That tolerance would let a field rename turn
        priming back into audible output silently, so ``_run_compile_priming``
        re-checks its own request and refuses to dispatch."""
        service, _ = _build_true_stream_service(registry, coordinator)
        dispatched: List[Any] = []

        async def _capture(request, mode):
            dispatched.append(request)
            return MagicMock(success=True)

        monkeypatch.setattr(service, "_dispatch_by_streaming_mode", _capture)
        # Simulate the field having been renamed / dropped out from under us.
        monkeypatch.setattr(
            QwenTTSService, "_is_suppressed", staticmethod(lambda request: False)
        )

        with pytest.raises(RuntimeError, match="not suppressed"):
            asyncio.run(service._run_compile_priming())

        assert dispatched == [], (
            "priming dispatched a generation it could not prove was suppressed"
        )

    def test_suppress_flag_defaults_to_false(self):
        """No ordinary request is ever born suppressed."""
        assert QwenTTSRequest(text="hi").suppress_audio_output is False

    def test_sentence_stream_chunk_copy_preserves_the_flag(self):
        """The SENTENCE_STREAM per-chunk request copy must carry the flag.

        Harmless today (the copy only reaches ``_generate_sync``, which never
        emits), but a request-copy site that silently drops the flag becomes a
        leak the moment anyone routes it through a dispatcher."""
        lines = SERVICE_PATH.read_text(encoding="utf-8").splitlines()
        starts = [
            i for i, ln in enumerate(lines)
            if "chunk_request = QwenTTSRequest(" in ln
        ]
        assert len(starts) == 1, (
            f"expected exactly one chunk_request copy site, found {len(starts)}"
        )
        block = "\n".join(lines[starts[0]: starts[0] + 30])
        assert "suppress_audio_output=request.suppress_audio_output" in block, (
            "the SENTENCE_STREAM chunk_request copy drops "
            "suppress_audio_output"
        )
