"""Story 20.5 Phase 4 — the consumer crossfade is neutralised ONLY on the
stream that is provably continuous.

WHY THIS EXISTS
---------------
``StreamingChunkBuffer`` cross-fades the last K samples of one released chunk
with the first K of the next. Those are *different moments in time* — sample
``n+i`` is mixed with sample ``n-64+i`` — so the operation bridges a genuine
step but **combs** a signal that was already continuous.

Through Story 20.4 every TRUE_STREAM chunk began from a cold codec state, so
there was a genuine step and the cross-fade was a repair whose harm was
invisible under a cold-start error 20x larger. Story 20.5 removed the cold
start. The Phase 2 evidence (§4.2) then measured the cross-fade making the
state-cached output **2.6x / 3.3x worse against ground truth** on the two long
fixtures, and predicted it would become audible on its own. The Phase 3
audition confirmed that in the sharpest possible way: two single-seam trials
(``m-020-t2``, ``s-020-t2``) where the reference's cold start happened NOT to
be audible and the candidate's newly-unmasked comb was — the only two blocking
rows in the round.

THE SCOPING HAZARD THESE TESTS GUARD
------------------------------------
The same buffer serves **SENTENCE_STREAM**, whose chunks are independently
generated sentences butt-spliced together. There the discontinuity is real and
the cross-fade is doing its job. Story 20.5 measured nothing on that path, so
turning the cross-fade off globally would silently change audio this story
never looked at.

Hence: the producer *declares* whether its chunk stream is sample-continuous,
and only a declared-continuous stream gets the neutralised buffer. Every
dispatch path that emits AudioChunks must set the flag at the top of that path
so a TRUE_STREAM generation cannot leave a stale ``True`` behind for a
following SENTENCE_STREAM one — which is exactly the silent cross-path change
this file is here to prevent.

WHY 0 AND NOT A SMALLER K
-------------------------
The harm is qualitative, not proportional. A cross-dissolve of a signal with
its own past is a comb filter with notches at odd multiples of
``sample_rate / (2K)``; halving K moves the notches up and shortens the
artefact but does not make the operation correct. On a stream that is already
sample-continuous, **0 is not "less of a bad thing" — it is the correct
value**, because the concatenation is then exactly the codec's output. §4.2
measured that directly: crossfade-off is 2.6-3.3x closer to ground truth.
``test_streaming_chunk_buffer.py::test_crossfade_disabled_passes_through_unchanged``
pins that 0 really does mean untouched.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# AudioCoordinator — the pass-through, and its default
# ============================================================================


def _coordinator():
    """A coordinator with the init flag flipped, mirroring
    ``test_audio_coordinator.py``: the real ``initialize()`` chain pulls in
    PortAudio, and none of that is under test here."""
    from myvoice.services.audio_coordinator import AudioCoordinator

    coord = AudioCoordinator()
    coord._is_initialized = True
    return coord


@pytest.mark.asyncio
async def test_default_keeps_the_64_sample_crossfade():
    """Every pre-existing caller passes nothing and must be unaffected."""
    from myvoice.services import audio_coordinator as ac

    coord = _coordinator()
    await coord.start_streaming_session(sample_rate=24000)
    assert coord._streaming_buffer is not None
    assert (coord._streaming_buffer.crossfade_samples
            == ac._DEFAULT_STREAMING_CROSSFADE_SAMPLES == 64)


@pytest.mark.asyncio
async def test_explicit_zero_neutralises_the_crossfade():
    coord = _coordinator()
    await coord.start_streaming_session(sample_rate=24000, crossfade_samples=0)
    assert coord._streaming_buffer.crossfade_samples == 0


@pytest.mark.asyncio
async def test_explicit_none_is_the_default_not_zero():
    """The consumer passes ``None`` for a non-continuous stream. ``None`` must
    mean "use the default", not "no crossfade" — getting this backwards would
    silently strip the crossfade from SENTENCE_STREAM."""
    coord = _coordinator()
    await coord.start_streaming_session(
        sample_rate=24000, crossfade_samples=None
    )
    assert coord._streaming_buffer.crossfade_samples == 64


@pytest.mark.asyncio
async def test_non_default_crossfade_is_logged(caplog):
    """A silently different audio path is the failure mode this whole story
    is about. Turning the crossfade off must be visible in the log."""
    import logging

    coord = _coordinator()
    with caplog.at_level(logging.INFO):
        await coord.start_streaming_session(
            sample_rate=24000, crossfade_samples=0
        )
    assert any("cross-fade" in r.message or "cross-fade" in r.getMessage()
               for r in caplog.records), caplog.text


# ============================================================================
# QwenTTSService — who declares continuity, and when
# ============================================================================


def _service():
    from myvoice.services.qwen_tts_service import QwenTTSService
    return QwenTTSService(
        audio_coordinator=None, device="cpu", quality_tier="fast",
        session_registry=None, app_settings=MagicMock(),
    )


def test_service_declares_not_continuous_before_any_generation():
    """The safe default. A consumer that opens a session before any dispatch
    has run must get today's behaviour."""
    assert _service().progressive_stream_is_continuous is False


def test_flag_is_true_only_for_a_state_carrying_decode_fn():
    """The declaration is read off the decode_fn rather than assumed, so the
    stateless fallback and the ``MYVOICE_CODEC_STATE_CACHE`` kill switch both
    do the right thing without a second place to keep in sync."""
    svc = _service()

    stateful = MagicMock()
    stateful.carries_codec_state = True
    svc._progressive_stream_continuous = bool(
        getattr(stateful, "carries_codec_state", False))
    assert svc.progressive_stream_is_continuous is True

    def stateless_fn(chunk):  # no attribute at all — the stock adapter
        return chunk
    svc._progressive_stream_continuous = bool(
        getattr(stateless_fn, "carries_codec_state", False))
    assert svc.progressive_stream_is_continuous is False


def test_every_audio_chunk_producer_declares_stream_continuity():
    """Source invariant, in the spirit of the Story 20.2 ``_audio_chunk_sink``
    rule.

    Every dispatch method that emits AudioChunks must assign
    ``_progressive_stream_continuous`` in its own body. Without this, a
    TRUE_STREAM generation would leave a stale ``True`` and the NEXT
    SENTENCE_STREAM generation would silently lose its crossfade — a
    cross-path audio change on a path this story never measured, and one no
    single-generation test would catch.
    """
    from myvoice.services import qwen_tts_service

    source = Path(inspect.getfile(qwen_tts_service)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    producers = set()
    assigners = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(node)
        if "_audio_chunk_sink" in body and node.name.startswith("_generate"):
            producers.add(node.name)
        if "_progressive_stream_continuous" in body and node.name.startswith(
            "_generate"
        ):
            assigners.add(node.name)

    assert producers, (
        "no _generate* method appears to emit AudioChunks any more; this "
        "invariant test has lost its subject and must be re-derived."
    )
    missing = producers - assigners
    assert not missing, (
        f"{sorted(missing)} emit AudioChunks but never declare whether the "
        f"stream is sample-continuous. A stale declaration from the previous "
        f"generation would decide the consumer crossfade for this one. Set "
        f"self._progressive_stream_continuous at the top of each."
    )


def test_sentence_stream_declares_discontinuous():
    """SENTENCE_STREAM butt-splices independently generated sentences, so the
    consumer crossfade is bridging a real step there. Story 20.5 measured
    nothing on that path and must not change it."""
    from myvoice.services import qwen_tts_service

    source = inspect.getsource(qwen_tts_service.QwenTTSService._generate_streaming)
    assert "self._progressive_stream_continuous = False" in source, (
        "_generate_streaming no longer declares its chunk stream "
        "discontinuous. SENTENCE_STREAM would inherit whatever the previous "
        "generation left behind."
    )


def test_true_stream_reads_the_declaration_off_the_decode_fn():
    from myvoice.services import qwen_tts_service

    source = inspect.getsource(
        qwen_tts_service.QwenTTSService._generate_true_stream)
    assert "carries_codec_state" in source and (
        "self._progressive_stream_continuous" in source), (
        "_generate_true_stream no longer derives its continuity declaration "
        "from the decode_fn. If it hard-codes True, the stateless fallback "
        "and the MYVOICE_CODEC_STATE_CACHE kill switch would both lose the "
        "crossfade they still need."
    )


# ============================================================================
# The consumer — asks the producer rather than assuming
# ============================================================================


def test_consumer_asks_the_producer_and_defaults_to_today_s_behaviour():
    """The orchestrator must read the declaration defensively: an app without
    a TTS service wired yet (and every test double) has to fall through to the
    64-sample default rather than raise or silently neutralise."""
    from myvoice import app as app_module

    source = inspect.getsource(app_module.MyVoiceApp._handle_progressive_chunk_async)
    assert "progressive_stream_is_continuous" in source
    assert "crossfade_samples=" in source
    # ``getattr(getattr(self, "_tts_service", None), ...)`` — both hops
    # guarded, because the failure mode of the inner one is an AttributeError
    # inside the session-open try block, which degrades to batch playback.
    assert 'getattr(self, "_tts_service", None)' in source, (
        "the service lookup is not guarded; an app without _tts_service "
        "raises inside the session-open try and silently falls back to batch "
        "playback."
    )


# ============================================================================
# End-to-end through the real orchestrator: producer -> consumer -> coordinator
# ============================================================================


@pytest.fixture
def _qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _app_with_service(_qapp, continuous):
    """A real ``MyVoiceApp`` with a mocked coordinator and a stub TTS service
    that declares the given continuity."""
    import asyncio

    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(_qapp)
    coordinator = AsyncMock()
    coordinator.start_streaming_session = AsyncMock(
        return_value={"monitor": "m-1", "virtual": "v-1"})
    coordinator.play_audio_chunk = AsyncMock(
        return_value={"monitor": True, "virtual": True})
    app._audio_coordinator = coordinator
    if continuous is not None:
        service = MagicMock()
        service.progressive_stream_is_continuous = continuous
        app._tts_service = service
    app.loop = asyncio.get_event_loop_policy().new_event_loop()
    return app, coordinator


class _Chunk:
    def __init__(self):
        import numpy as np
        self.audio_data = np.zeros(480, dtype=np.float32)
        self.sample_rate = 24000
        self.chunk_index = 0
        self.is_final = False
        self.text_segment = ""
        self.session_id = None


@pytest.mark.parametrize(
    "continuous, expected",
    [
        (True, 0),      # TRUE_STREAM with codec state caching
        (False, None),  # SENTENCE_STREAM, or the stateless TRUE_STREAM fallback
        (None, None),   # no TTS service wired at all
    ],
)
def test_orchestrator_requests_the_right_crossfade(_qapp, continuous, expected):
    """The whole chain, behaviourally: what the producer declares is what the
    coordinator is asked for.

    The three rows are the three regimes that must not be conflated. Only the
    first neutralises the cross-fade; the other two have to reach the
    coordinator with ``None`` so it applies its 64-sample default. Getting the
    third row wrong would strip the cross-fade from every path in any build
    where the service is not wired yet.
    """
    import asyncio

    app, coordinator = _app_with_service(_qapp, continuous)
    try:
        asyncio.run(app._handle_progressive_chunk_async(_Chunk()))
        coordinator.start_streaming_session.assert_awaited_once()
        kwargs = coordinator.start_streaming_session.await_args.kwargs
        assert kwargs["crossfade_samples"] == expected
    finally:
        try:
            app.loop.close()
        except Exception:
            pass
