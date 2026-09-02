"""Story 20.6 — the 5-frame lookahead is retired ONLY where carried codec
state has already made the chunks continuous.

WHY THIS FILE EXISTS (AC #2 — the load-bearing constraint)
----------------------------------------------------------
Story 16.4's *future-lookahead overlap-add* has the streamer emit
``chunk_size + lookahead`` frames and slide by ``chunk_size``: each chunk
re-decodes the previous chunk's last five frames so the next rendition is
better primed. Story 20.5 made the codec carry its real state across the
boundary, so that priming is now exact rather than approximate and the five
frames buy nothing — they cost five talker steps of TTFA, a two-pass decode,
and a trim plus a blend to manage an overlap nobody needs.

But the streamer is not always fed by the state-carrying decoder.
``_build_true_stream_decode_fn`` falls back to a cold-state adapter — three
ways: the ``MYVOICE_CODEC_STATE_CACHE`` operator kill switch, ``probe_decoder``
refusing a decoder graph it has not verified, and the numerical self-test
failing on the loaded weights. On that path every chunk is still an
independent rendering of an overlapping token span, so **the lookahead, the
post-decode trim and the Story 20.4 seam blend are the only seam handling it
has.**

Retiring the lookahead by editing ``DEFAULT_LOOKAHEAD`` to 0 would therefore
strip all three defences from the path a user lands on *precisely when
something else has already gone wrong*, and reintroduce the artefact Epic 20
spent six audition rounds eliminating. Every row below exists to make that
mistake fail loudly rather than ship silently.

THE SHAPE, AND WHY IT IS STORY 20.5'S
--------------------------------------
Producer declares, consumer acts — the same shape Story 20.5 used for the
consumer crossfade (``test_consumer_crossfade_scoping.py``). The decode_fn
declares ``carries_codec_state``; the dispatch layer reads it once and hands
the answer to two consumers: ``_progressive_stream_continuous`` (the crossfade)
and ``streamer.apply_codec_state_geometry`` (the lookahead). One declaration,
two acts, no second place to keep in sync.

THE STALENESS TRAP, AND WHY THE INVARIANTS ARE SOURCE-DERIVED
--------------------------------------------------------------
Story 20.5's hazard was a declaration set on one dispatch path and inherited
by the next. The same shape applies here, so the same defence does: a runtime
test can only prove the generation *it* ran was right, while an invariant over
the source proves every path is. Three of them below —

  * every ``CodecTokenStreamer`` construction site declares its geometry in
    the same function (a path that constructs one and never declares would
    silently emit at whatever geometry it inherited);
  * nothing outside the streamer module assigns ``.lookahead`` directly (the
    single reversible entry point is what makes a half-retired geometry
    unreachable); and
  * ``DEFAULT_LOOKAHEAD`` is still 5 (the global-change failure mode itself).
"""

from __future__ import annotations

import ast
import inspect
import queue
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myvoice.services.tts_streaming import codec_token_streamer
from myvoice.services.tts_streaming.codec_token_streamer import (
    CodecTokenStreamer,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOOKAHEAD,
    RETIRED_LOOKAHEAD,
    effective_lookahead,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src" / "myvoice"
STREAMER_MODULE = SRC / "services" / "tts_streaming" / "codec_token_streamer.py"


# ============================================================================
# The rule itself — a pure function of the declaration
# ============================================================================


def test_carried_state_retires_the_lookahead():
    assert effective_lookahead(True) == RETIRED_LOOKAHEAD == 0
    assert effective_lookahead(True, 5) == 0
    assert effective_lookahead(True, 17) == 0


def test_no_carried_state_keeps_the_configured_lookahead_exactly():
    """The fallback path's whole seam defence, restated as a test."""
    assert effective_lookahead(False) == DEFAULT_LOOKAHEAD == 5
    assert effective_lookahead(False, 5) == 5
    assert effective_lookahead(False, 17) == 17


def test_the_module_constant_is_not_globally_retired():
    """AC #2's failure mode, named directly.

    Retiring the lookahead by editing this constant is the change this whole
    story is arranged to prevent: it would take the trim and the seam blend
    away from the stateless fallback (both are gated on ``lookahead > 0``)
    without any test of the state-cached path noticing, because the
    state-cached path wants exactly that behaviour.
    """
    assert codec_token_streamer.DEFAULT_LOOKAHEAD == 5, (
        "DEFAULT_LOOKAHEAD is the STATELESS path's lookahead and must stay 5. "
        "Story 20.6 retires the lookahead conditionally, via "
        "effective_lookahead() / apply_codec_state_geometry(), never by "
        "changing this constant."
    )
    assert codec_token_streamer.DEFAULT_CHUNK_SIZE == 25, (
        "AC #5: this story does not touch geometry beyond the lookahead; the "
        "chunk-size reopen is a separate story with its own audition."
    )


# ============================================================================
# The streamer — the consumer half
# ============================================================================


def test_apply_retires_the_lookahead_and_the_overlap_with_it():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    assert s.apply_codec_state_geometry(True) == 0
    assert s.lookahead == 0
    assert s._chunk_with_lookahead == 4

    s.put([1, 2, 3, 4, 5, 6, 7, 8])
    assert s.queue.get_nowait() == [1, 2, 3, 4]
    assert s.queue.get_nowait() == [5, 6, 7, 8], (
        "chunks still overlap; the streamer kept a lookahead tail it no "
        "longer has"
    )
    assert s._buffer == []


def test_apply_without_carried_state_leaves_the_pre_20_6_geometry_exactly():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    assert s.apply_codec_state_geometry(False) == 2
    assert s.lookahead == 2
    assert s._chunk_with_lookahead == 6

    s.put([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert s.queue.get_nowait() == [1, 2, 3, 4, 5, 6]
    # ...and the next chunk re-sends [5, 6]: the shared lookahead tail is
    # exactly the overlap the trim and the seam blend exist to reconcile.
    assert s.queue.get_nowait() == [5, 6, 7, 8, 9, 10]
    assert s._buffer == [9, 10]


def test_flipping_the_declaration_never_leaves_a_half_retired_geometry():
    """AC #2 — "flipping the kill switch at runtime does not leave a
    half-retired geometry".

    The geometry is a pure function of the declaration and of the lookahead
    the streamer was constructed with, so no sequence of flips can accumulate
    a state that disagrees with the flag currently in force. Asserted over an
    adversarial sequence rather than one round trip.
    """
    s = CodecTokenStreamer(chunk_size=25, lookahead=5)
    for carries in (True, True, False, True, False, False, True, False):
        applied = s.apply_codec_state_geometry(carries)
        assert applied == (0 if carries else 5)
        assert s.lookahead == applied
        assert s._chunk_with_lookahead == s.chunk_size + applied
    assert s.lookahead == 5 and s._chunk_with_lookahead == 30


def test_geometry_may_not_change_mid_generation():
    """Same caller contract ``reset()`` carries. Changing the window while
    ``put()`` is sliding the buffer — or while the talker forward-hook is
    chunking against a snapshot of it — emits one malformed chunk, which is
    inaudible as a bug and audible only as "the codec got worse"."""
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put([1, 2, 3])
    with pytest.raises(RuntimeError, match="between generations"):
        s.apply_codec_state_geometry(True)

    s.reset()
    s.apply_codec_state_geometry(True)  # clean now

    s.queue.put(["a chunk nobody drained"])
    with pytest.raises(RuntimeError, match="between generations"):
        s.apply_codec_state_geometry(False)


def test_a_reset_streamer_can_be_re_declared():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.apply_codec_state_geometry(True)
    s.put([1, 2, 3, 4])
    s.reset()
    assert s.apply_codec_state_geometry(False) == 2
    assert isinstance(s.queue, queue.Queue) and s.queue.empty()


# ============================================================================
# The dispatch layer — who declares, who acts, and the staleness invariants
# ============================================================================


def _service():
    from myvoice.services.qwen_tts_service import QwenTTSService

    return QwenTTSService(
        audio_coordinator=None, device="cpu", quality_tier="fast",
        session_registry=None, app_settings=MagicMock(),
    )


def _fake_model():
    """A model whose ``model.speech_tokenizer.model.decoder`` is reachable —
    all ``_build_true_stream_decode_fn`` needs before it consults
    ``codec_state_cache``."""
    return MagicMock()


def test_the_stateful_decoder_is_built_in_the_retired_geometry(monkeypatch):
    """AC #1 / Task 2 — the two-pass decode collapses, without touching
    ``codec_state_cache``.

    ``StatefulCodecDecoder`` is constructed with
    ``window_frames = chunk_size + lookahead``. Handing it the retired
    lookahead makes ``window_frames == commit_frames``, and its own commit
    rule then takes the ``commit = n_frames`` branch on every chunk — so the
    snapshot / decode-the-lookahead-on-the-snapshot / restore second pass is
    never entered. Story 20.5's wrapper is the foundation this stands on and
    is not modified; the collapse is a consequence of the argument.
    """
    from myvoice.services.tts_streaming import codec_state_cache

    seen = {}

    def _fake_build(decoder, *, chunk_size, lookahead, device=None, **kw):
        seen["chunk_size"] = chunk_size
        seen["lookahead"] = lookahead
        fn = MagicMock()
        fn.carries_codec_state = True
        return fn, "enabled"

    monkeypatch.setattr(codec_state_cache, "build_stateful_decode_fn", _fake_build)
    decode_fn = _service()._build_true_stream_decode_fn(
        _fake_model(), chunk_size=25, lookahead=5
    )
    assert getattr(decode_fn, "carries_codec_state", False) is True
    assert seen == {"chunk_size": 25, "lookahead": 0}


def test_the_kill_switch_returns_the_stateless_adapter_undeclared(monkeypatch):
    """AC #2 — with the switch set the real builder declines, and the adapter
    it falls back to carries no declaration at all, so every consumer of the
    declaration (crossfade and lookahead alike) keeps today's behaviour."""
    monkeypatch.setenv("MYVOICE_CODEC_STATE_CACHE", "0")
    decode_fn = _service()._build_true_stream_decode_fn(
        _fake_model(), chunk_size=25, lookahead=5
    )
    assert getattr(decode_fn, "carries_codec_state", False) is False
    assert getattr(decode_fn, "reset", None) is None


def test_a_build_time_refusal_also_returns_the_stateless_adapter(monkeypatch):
    """The kill switch is not the only way onto the fallback path.
    ``probe_decoder`` refusing an unrecognised graph, or the numerical
    self-test failing on the loaded weights, land there too — and they are
    the cases where a user reaches the fallback without having asked for it.
    """
    from myvoice.services.tts_streaming import codec_state_cache

    monkeypatch.setattr(
        codec_state_cache, "build_stateful_decode_fn",
        lambda *a, **k: (None, "decoder graph not supported: synthetic"),
    )
    decode_fn = _service()._build_true_stream_decode_fn(
        _fake_model(), chunk_size=25, lookahead=5
    )
    assert getattr(decode_fn, "carries_codec_state", False) is False


def test_true_stream_dispatch_acts_on_the_declaration_it_reads():
    """Source arm: the dispatch reads ``carries_codec_state`` once and drives
    the streamer geometry from it. Hard-coding either side would decouple the
    lookahead from the decode path it exists to serve."""
    from myvoice.services import qwen_tts_service

    source = inspect.getsource(
        qwen_tts_service.QwenTTSService._generate_true_stream)
    assert "carries_codec_state" in source
    assert "apply_codec_state_geometry" in source
    assert "apply_codec_state_geometry(True)" not in source, (
        "the streamer geometry is hard-coded to the retired one; the "
        "stateless fallback would lose its only seam handling"
    )


def test_every_streamer_construction_declares_its_geometry():
    """Source invariant — the staleness trap, closed the Story 20.5 way.

    A dispatch path that constructs a ``CodecTokenStreamer`` and never
    declares its geometry emits chunks at whatever the module constants say,
    regardless of which decode path it actually built. Today that means a
    30-frame window fed to a decoder built for 25 — the worker would then
    trim and blend audio that carries real codec state, at a splice point
    that no longer matches. No single-generation test catches that on a path
    it does not happen to run.
    """
    producers = set()
    declarers = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.dump(node)
            where = f"{path.relative_to(REPO_ROOT)}::{node.name}"
            if "'CodecTokenStreamer'" in body and "Call(" in body:
                # Only real constructions, not the class definition itself.
                if any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id == "CodecTokenStreamer"
                    for c in ast.walk(node)
                ):
                    producers.add(where)
            if "apply_codec_state_geometry" in body:
                declarers.add(where)

    assert producers, (
        "no function constructs a CodecTokenStreamer any more; this "
        "invariant has lost its subject and must be re-derived."
    )
    missing = producers - declarers
    assert not missing, (
        f"{sorted(missing)} construct a CodecTokenStreamer but never declare "
        f"its geometry. Call streamer.apply_codec_state_geometry(...) with "
        f"the decode_fn's carries_codec_state, before the worker snapshots "
        f"the geometry and before the talker reads it."
    )


def test_nothing_sets_the_lookahead_behind_the_single_entry_point():
    """Source invariant — one reversible door, so a half-retired geometry is
    unreachable.

    ``apply_codec_state_geometry`` sets ``lookahead`` and
    ``_chunk_with_lookahead`` together, from the constructed value. A direct
    ``streamer.lookahead = 0`` elsewhere sets one and not the other: the
    streamer would then emit 30-frame chunks while the worker, which
    snapshots ``streamer.lookahead``, believed there were none to trim.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == STREAMER_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in (
                    "lookahead", "_chunk_with_lookahead",
                ):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                    )
    assert not offenders, (
        "the streamer geometry must only be set through "
        f"apply_codec_state_geometry(). Offending sites: {offenders}"
    )


# ============================================================================
# The fallback keeps ALL THREE defences — end to end, in one place
# ============================================================================


SPF = 1920
EDGE = 555


def _stateless_codec_like_decode(chunk):
    """A decode_fn with the pre-20.5 cold-state geometry: every decode returns
    ``1920*N - 555``, and the sample VALUE is the absolute audio position so a
    mis-splice shows as a jump or a repeat. Declares nothing, exactly like the
    stock adapter ``_build_true_stream_decode_fn`` falls back to."""
    import numpy as np

    n = len(chunk)
    start = int(chunk[0]) * SPF
    return np.arange(start, start + SPF * n - EDGE, dtype=np.float64).astype(
        np.float32
    )


def test_the_fallback_path_keeps_lookahead_trim_and_blend_together():
    """AC #2, stated as one behavioural row rather than three inferences.

    The stateless path is reached whenever the kill switch is set OR
    ``codec_state_cache`` declines the loaded decoder. On it, chunks are
    independent renderings of overlapping token spans, so all three defences
    must still be live at once:

      1. the streamer emits ``chunk_size + lookahead`` frames per chunk;
      2. the worker trims back to the splice at ``chunk_size * 1920``;
      3. the worker retains the trimmed tail and cross-fades it into the next
         chunk's head.

    Retiring the lookahead globally would remove all three in one edit — (2)
    and (3) are both gated on ``lookahead > 0`` — and no test of the
    state-cached path would notice, because that path wants exactly that.
    """
    from myvoice.services.tts_streaming.streaming_decoder import (
        END_OF_STREAM,
        StreamingDecoderWorker,
    )

    cs, la = 10, 5
    streamer = CodecTokenStreamer(chunk_size=cs, lookahead=la)
    # The declaration the stock adapter makes: none at all.
    assert streamer.apply_codec_state_geometry(
        getattr(_stateless_codec_like_decode, "carries_codec_state", False)
    ) == la

    # 1. the streamer still overlaps.
    assert streamer._chunk_with_lookahead == cs + la

    posted = []
    for k in range(3):
        streamer.queue.put(list(range(k * cs, k * cs + cs + la)))
    streamer.queue.put(END_OF_STREAM)
    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_stateless_codec_like_decode,
        post_mutation=lambda m, s, *a: (
            posted.append(a[0]) if m == "append_chunk" else None),
        session_id="fallback",
    )
    worker.start()
    worker.join(timeout=5.0)

    # 2. the trim is live: each posted chunk is exactly the splice, not the
    #    whole 30-frame decode.
    assert [p.size for p in posted] == [cs * SPF] * 3

    # 3. the blend is live: a tail was retained past every splice, which is
    #    the input ``_apply_overlap_add`` cross-fades into the next chunk's
    #    head. Under the retirement this is None on every chunk — that is the
    #    difference. (What the blend then DOES with the tail is pinned by
    #    ``test_streaming_decoder.py::test_overlap_add_blends_the_previously_discarded_tail``;
    #    the position-valued decode_fn here cannot observe it, because both
    #    sides of the blend carry the same values by construction.)
    assert worker._pending_overlap is not None
    assert worker._pending_overlap.size > 0


# ============================================================================
# The compile geometry — one derivation point, and it is conditional too
# ============================================================================


def test_compile_geometry_follows_the_retirement():
    from myvoice.services.tts_streaming import resolve_streamer_geometry

    assert resolve_streamer_geometry() == (DEFAULT_CHUNK_SIZE, 0)
    assert sum(resolve_streamer_geometry()) == 25


def test_compile_geometry_reverts_under_the_kill_switch(monkeypatch):
    from myvoice.services.tts_streaming import resolve_streamer_geometry

    monkeypatch.setenv("MYVOICE_CODEC_STATE_CACHE", "off")
    assert resolve_streamer_geometry() == (DEFAULT_CHUNK_SIZE, DEFAULT_LOOKAHEAD)
    assert sum(resolve_streamer_geometry()) == 30
