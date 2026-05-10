"""Tests for StreamingDecoderWorker (Story 16.4).

Verifies P-6 (architecture-optimization-pass.md:431-441) — the four-step
decoder contract; P-7 (lines 443-451) — cancellation propagation; D-11
(line 261) — cooperative cancel via shared threading.Event; the import
rule at line 674 — `streaming_decoder.py` may NOT import sessions.
"""

import ast
import inspect
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from myvoice.observability import metrics as _metrics_module
from myvoice.services.tts_streaming import (
    CodecTokenStreamer,
    END_OF_STREAM,
    StreamingDecoderWorker,
)
from myvoice.services.tts_streaming import streaming_decoder as _streaming_decoder


# -- shared fixtures --------------------------------------------------- #


def _make_decoded_pcm(chunk):
    """Deterministic decode_fn: each token id maps to one PCM sample
    of value = token_id * 0.1. Token-to-PCM ratio is 1.0 — trim count
    in samples equals trim count in tokens.
    """
    return np.array([t * 0.1 for t in chunk], dtype=np.float32)


class _RecordingPostMutation:
    """Records every post_mutation call as a tuple in self.calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        self.calls.append(args)


class _RecordingMetrics:
    """Records every metrics.record(...) call. Installed via monkeypatch
    onto myvoice.observability.metrics.record (the source attribute).

    Signature mirrors the real `metrics.record(name, value, *,
    session_id=None, **tags)` exactly so tests cannot drift away from
    production wire format (H2 review-fix).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, name, value, *, session_id=None, **tags) -> None:
        self.calls.append(
            {
                "metric_name": name,
                "value": value,
                "session_id": session_id,
                "tags": dict(tags),
            }
        )


def _build_streamer(chunk_size: int = 4, lookahead: int = 2) -> CodecTokenStreamer:
    """Construct a streamer; the worker snapshots its chunk geometry."""
    return CodecTokenStreamer(chunk_size=chunk_size, lookahead=lookahead)


def _wait_for_thread_alive(worker: StreamingDecoderWorker, timeout: float = 1.0) -> None:
    """Spin briefly until the worker's thread reports alive (post-start)."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if worker.is_alive():
            return
        time.sleep(0.001)


# ============================================================================
# AC #1 — public surface: importable from package; documented method set;
#         not a Thread subclass; no auto-start on construction.
# ============================================================================


def test_worker_importable_from_package_top_level():
    # Direct re-export: no need to reach into streaming_decoder.py.
    from myvoice.services.tts_streaming import StreamingDecoderWorker as W
    assert W is StreamingDecoderWorker


def test_package_all_lists_expected_symbols_in_declaration_order():
    import myvoice.services.tts_streaming as pkg
    # Story 18.2 widened the package surface with two new exports
    # (`enable_tf32_and_cudnn_benchmark`, `is_ampere_or_newer`) per
    # the append-only declaration-order discipline.
    assert pkg.__all__ == [
        "StreamingMode",
        "default_streaming_mode_for_hardware",
        "effective_streaming_mode",
        "CodecTokenStreamer",
        "END_OF_STREAM",
        "StreamingDecoderWorker",
        "enable_tf32_and_cudnn_benchmark",
        "is_ampere_or_newer",
    ]


def test_worker_class_has_documented_public_surface():
    # Public methods only: start, join, is_alive. No other public method.
    public_attrs = {
        name
        for name in dir(StreamingDecoderWorker)
        if not name.startswith("_")
    }
    assert public_attrs == {"start", "join", "is_alive"}


def test_worker_is_not_a_thread_subclass_owns_thread_internally():
    # Worker holds a Thread; it is not itself a Thread subclass.
    assert not issubclass(StreamingDecoderWorker, threading.Thread)
    streamer = _build_streamer()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-12345",
    )
    assert isinstance(worker._thread, threading.Thread)


def test_worker_does_not_auto_start_on_construction():
    streamer = _build_streamer()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-12345",
    )
    # Construction does not auto-start; caller controls .start().
    assert worker.is_alive() is False


# ============================================================================
# AC #2 — constructor: identity preservation, snapshot, shared cancel event,
#         tag passthrough, validation.
# ============================================================================


def test_constructor_holds_streamer_decode_post_mutation_session_id_by_identity():
    streamer = _build_streamer()
    fake_decode = _make_decoded_pcm
    fake_post = _RecordingPostMutation()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=fake_decode,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    assert worker._streamer is streamer
    assert worker._decode_fn is fake_decode
    assert worker._post_mutation is fake_post
    assert worker._session_id == "abc-123"


def test_constructor_snapshots_streamer_chunk_size_and_lookahead():
    streamer = _build_streamer(chunk_size=8, lookahead=3)
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
    )
    assert worker._chunk_size == 8
    assert worker._lookahead == 3

    # Mutating the streamer's geometry post-construction does NOT change
    # the worker's snapshot — the documented contract.
    streamer.chunk_size = 999
    streamer.lookahead = 999
    assert worker._chunk_size == 8
    assert worker._lookahead == 3


def test_constructor_shares_cancel_event_with_streamer():
    streamer = _build_streamer()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
    )
    # AC #2: same threading.Event the streamer was constructed with.
    # Story 16.5 wires registry.cancel() to flip this event; both
    # producer and consumer observe the single flip.
    assert worker._cancel_event is streamer._cancel_event


def test_constructor_default_tags():
    streamer = _build_streamer()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
    )
    assert worker._model_type == "qwen3_tts"
    assert worker._hardware == "gpu"
    assert worker.is_alive() is False


def test_constructor_custom_tags_pass_through_verbatim():
    streamer = _build_streamer()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
        model_type="qwen3_tts_voice_clone",
        hardware="cpu",
    )
    assert worker._model_type == "qwen3_tts_voice_clone"
    assert worker._hardware == "cpu"


@pytest.mark.parametrize(
    "kwargs, offending",
    [
        ({"streamer": None}, "streamer"),
        ({"decode_fn": None}, "decode_fn"),
        ({"post_mutation": None}, "post_mutation"),
        ({"session_id": ""}, "session_id"),
        ({"session_id": None}, "session_id"),
    ],
)
def test_constructor_rejects_invalid_inputs(kwargs, offending):
    base = {
        "streamer": _build_streamer(),
        "decode_fn": _make_decoded_pcm,
        "post_mutation": _RecordingPostMutation(),
        "session_id": "abc-123",
    }
    base.update(kwargs)
    with pytest.raises(ValueError) as exc_info:
        StreamingDecoderWorker(**base)
    assert offending in str(exc_info.value)


def test_worker_uses_snapshot_geometry_after_streamer_mutated():
    """Trim semantics use the constructor-time snapshot, not live values."""
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    # Mutate the streamer's geometry AFTER construction. The trim must
    # still use the snapshot (chunk_size=4, lookahead=2).
    streamer.chunk_size = 99
    streamer.lookahead = 99

    # Push a full-size chunk (6 tokens = chunk_size + lookahead from snapshot)
    # plus END_OF_STREAM. The trim should yield the leading 4 samples.
    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put(END_OF_STREAM)

    worker.start()
    worker.join(timeout=2.0)

    # Expected: ('append_chunk', sid, [0.1, 0.2, 0.3, 0.4]) then ('finalize', sid).
    assert len(fake_post.calls) == 2
    assert fake_post.calls[0][0] == "append_chunk"
    np.testing.assert_array_almost_equal(
        fake_post.calls[0][2], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    )
    assert fake_post.calls[1] == ("finalize", "abc-123")


# ============================================================================
# AC #3 — start()/join()/is_alive() lifecycle and double-start semantics.
# ============================================================================


def test_start_spawns_daemon_thread_with_named_for_session():
    streamer = _build_streamer()
    streamer.queue.put(END_OF_STREAM)  # immediate clean exit
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-12345-XYZ",
    )
    # Thread name uses the first 8 chars of session_id.
    assert worker._thread.name == "StreamingDecoder-abc-1234"
    # Daemon: pytest will not hang if the worker leaks.
    assert worker._thread.daemon is True
    worker.start()
    worker.join(timeout=2.0)


def test_join_blocks_until_thread_exits():
    streamer = _build_streamer()
    streamer.queue.put(END_OF_STREAM)
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)
    assert worker.is_alive() is False


def test_double_start_raises_runtime_error():
    streamer = _build_streamer()
    streamer.queue.put(END_OF_STREAM)
    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)
    with pytest.raises(RuntimeError):
        worker.start()


# ============================================================================
# AC #4 — happy path: pull → decode + trim → post append_chunk → END_OF_STREAM
#         → post finalize → exit.
# ============================================================================


def test_happy_path_three_chunks_then_end_of_stream():
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    # Pre-populate three full-size chunks + END_OF_STREAM (mirrors what
    # CodecTokenStreamer.put would produce for tokens 1..14 with chunk
    # geometry 4+2; see Story 16.3 AC #3).
    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False

    # Expected: 3 append_chunk posts (leading 4 samples each) + 1 finalize.
    assert len(fake_post.calls) == 4

    for posted, expected_first_sample in zip(fake_post.calls[:3], [0.1, 0.5, 0.9]):
        method, sid, pcm = posted
        assert method == "append_chunk"
        assert sid == "abc-123"
        assert len(pcm) == 4  # trimmed to leading chunk_size = 4 samples
        # First sample of leading chunk_size matches expectation.
        np.testing.assert_almost_equal(float(pcm[0]), expected_first_sample)

    assert fake_post.calls[3] == ("finalize", "abc-123")


def test_happy_path_trim_arithmetic_per_chunk():
    """Verifies the exact trim values for the canonical multi-chunk path."""
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    np.testing.assert_array_almost_equal(
        fake_post.calls[0][2],
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )
    np.testing.assert_array_almost_equal(
        fake_post.calls[1][2],
        np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    )
    np.testing.assert_array_almost_equal(
        fake_post.calls[2][2],
        np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
    )


def test_residual_final_chunk_posted_without_trim():
    """The residual chunk (len < chunk_size + lookahead) is posted whole —
    no trim because no following chunk to overlap with.
    """
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    streamer.queue.put([1, 2, 3])  # length 3 < 6 → residual
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    assert len(fake_post.calls) == 2
    method, sid, pcm = fake_post.calls[0]
    assert method == "append_chunk"
    np.testing.assert_array_almost_equal(
        pcm, np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    assert fake_post.calls[1] == ("finalize", "abc-123")


# ============================================================================
# AC #5 — drain-on-cancel: 3 paths (pre-consumption, mid-stream, mid-decode).
# ============================================================================


def test_cancel_set_before_consumption_drains_queue_no_decode():
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    decode_calls = []

    def counting_decode(chunk):
        decode_calls.append(chunk)
        return _make_decoded_pcm(chunk)

    # Pre-load chunks + sentinel; flip cancel BEFORE start.
    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)
    streamer._cancel_event.set()

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=counting_decode,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    # No decode happened.
    assert decode_calls == []
    # Exactly one cancel post; queue fully drained.
    assert fake_post.calls == [("cancel", "abc-123")]
    assert streamer.queue.empty() is True


def test_cancel_set_mid_stream_drains_remaining_chunks():
    """Cancel flips between iterations: chunk 1 lands, chunks 2/3 don't."""
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    cancel_event = streamer._cancel_event

    # post_mutation flips the cancel event after the first append_chunk.
    class _FlipOnFirstPost:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def __call__(self, *args) -> None:
            self.calls.append(args)
            if len(self.calls) == 1:
                cancel_event.set()

    fake_post = _FlipOnFirstPost()
    decode_calls = []

    def counting_decode(chunk):
        decode_calls.append(chunk)
        return _make_decoded_pcm(chunk)

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=counting_decode,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    # Decoded only chunk 1.
    assert len(decode_calls) == 1
    assert decode_calls[0] == [1, 2, 3, 4, 5, 6]

    # Posts: append_chunk then cancel (no finalize).
    assert len(fake_post.calls) == 2
    assert fake_post.calls[0][0] == "append_chunk"
    assert fake_post.calls[1] == ("cancel", "abc-123")

    # Queue fully drained.
    assert streamer.queue.empty() is True


def test_cancel_event_flipped_during_decode_still_posts_in_flight_chunk_then_cancels():
    """Decode_fn flips event; the in-flight post still happens, then cancel."""
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    cancel_event = streamer._cancel_event

    def decode_then_set_cancel(chunk):
        # Flip the event mid-decode; the worker still posts the chunk it
        # already finished decoding, then drains and posts cancel on the
        # next loop iteration.
        cancel_event.set()
        return _make_decoded_pcm(chunk)

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=decode_then_set_cancel,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    # In-flight chunk posted; followed by cancel; no finalize.
    assert len(fake_post.calls) == 2
    assert fake_post.calls[0][0] == "append_chunk"
    assert fake_post.calls[1] == ("cancel", "abc-123")
    assert streamer.queue.empty() is True


# ============================================================================
# AC #6 — decoder exception: catch → record decode_error metric → cancel +
#         drain → exit cleanly.
# ============================================================================


def test_decode_exception_caught_posts_cancel_and_records_decode_error_metric(
    monkeypatch,
):
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    fake_metrics = _RecordingMetrics()
    monkeypatch.setattr(
        "myvoice.observability.metrics.record", fake_metrics
    )

    def always_raise(chunk):
        raise RuntimeError("CUDA OOM")

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=always_raise,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    # Exactly one cancel post; no append_chunk; no finalize.
    assert fake_post.calls == [("cancel", "abc-123")]
    # Queue drained.
    assert streamer.queue.empty() is True
    # Worker exited cleanly.
    assert worker.is_alive() is False
    # decode_error metric recorded with numeric value (so the real
    # metrics.record's int|float guard at metrics.py:95-98 does not
    # raise TypeError mid-cancel — H1 review-fix).
    decode_error_calls = [
        c for c in fake_metrics.calls if c["metric_name"] == "decode_error"
    ]
    assert len(decode_error_calls) == 1
    rec = decode_error_calls[0]
    assert isinstance(rec["value"], (int, float))
    assert rec["session_id"] == "abc-123"
    # The exception's repr now lives in tags, not in value.
    assert "CUDA OOM" in rec["tags"]["error_repr"]


def test_decode_exception_drains_remaining_queue(monkeypatch):
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    monkeypatch.setattr(
        "myvoice.observability.metrics.record", _RecordingMetrics()
    )

    def always_raise(chunk):
        raise RuntimeError("boom")

    # Many chunks; the drain must clear them all.
    for _ in range(5):
        streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=always_raise,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    assert streamer.queue.empty() is True
    assert fake_post.calls == [("cancel", "abc-123")]


def test_decode_exception_on_second_chunk_posts_first_then_cancels(monkeypatch):
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    monkeypatch.setattr(
        "myvoice.observability.metrics.record", _RecordingMetrics()
    )

    call_count = {"n": 0}

    def raise_on_second(chunk):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("second-chunk failure")
        return _make_decoded_pcm(chunk)

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=raise_on_second,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    # Posts: first chunk's append_chunk, then cancel. No finalize.
    assert len(fake_post.calls) == 2
    assert fake_post.calls[0][0] == "append_chunk"
    assert fake_post.calls[1] == ("cancel", "abc-123")


# ============================================================================
# Review-fix regressions — exact bug class per
# user-memory `code_review_regression_test_exact_class.md`.
# ============================================================================


def test_decode_exception_uses_numeric_metric_value_so_real_metrics_record_does_not_crash():
    """H1 regression: the worker previously called
    `metrics.record('decode_error', repr(exc), ...)` — but the real
    `metrics.record` validates `value: int|float` (metrics.py:95-98)
    and raises `TypeError` for str. The TypeError fired from inside
    `_run`'s except block, killed the thread, and `_drain_and_post_cancel`
    was never reached. The session was stuck in GENERATING with no
    cancel post — the "cancelled-but-still-emitting" window P-7 forbids.

    This test exercises the **exact bug class** by using the REAL
    `metrics.record` (no monkeypatch) plus an `add_listener()` capture.
    A regression that re-introduces a non-numeric value fails this
    test before any tag-shape assertion can mask it.
    """
    captured: list = []
    unsub = _metrics_module.add_listener(captured.append)
    try:
        streamer = _build_streamer(chunk_size=4, lookahead=2)
        fake_post = _RecordingPostMutation()

        def always_raise(chunk):
            raise RuntimeError("CUDA OOM")

        streamer.queue.put([1, 2, 3, 4, 5, 6])
        streamer.queue.put(END_OF_STREAM)

        worker = StreamingDecoderWorker(
            streamer=streamer,
            decode_fn=always_raise,
            post_mutation=fake_post,
            session_id="abc-123",
        )
        worker.start()
        worker.join(timeout=2.0)

        # Worker did NOT crash mid-cancel — the cancel post landed.
        assert worker.is_alive() is False
        assert fake_post.calls == [("cancel", "abc-123")]
        assert streamer.queue.empty() is True

        # decode_error metric flowed through the real metrics.record.
        decode_error_records = [
            r for r in captured if r.name == "decode_error"
        ]
        assert len(decode_error_records) == 1
        rec = decode_error_records[0]
        # Value is numeric — the bug had it as str(repr(exc)).
        assert isinstance(rec.value, (int, float))
        # The exception's repr lives in tags now.
        assert "CUDA OOM" in rec.tags.get("error_repr", "")
        # session_id at top level on the LogRecord (real signature
        # splits it out — H2 fixture-fidelity fix exercises this).
        assert rec.session_id == "abc-123"
    finally:
        unsub()


def test_drain_records_metric_on_unexpected_exception_and_still_posts_cancel():
    """H3 regression: the worker previously caught bare `Exception` in
    `_drain_and_post_cancel`, silently swallowing any non-`queue.Empty`
    raise from `get_nowait()`. The narrow fix catches `queue.Empty`
    explicitly; a defensive outer guard records exotic exceptions as
    `drain_error` and still posts cancel (P-7 invariant).

    This test injects a `get_nowait` that raises `RuntimeError` (NOT
    `queue.Empty`) on its first call, verifying:
      1. The exception is NOT silently swallowed — a `drain_error`
         metric fires.
      2. The cancel post still happens — P-7 cannot be violated.
      3. The worker thread exits cleanly.
    """
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    fake_metrics = _RecordingMetrics()

    # Replace get_nowait with a one-shot that raises RuntimeError. The
    # cancel-event path enters drain BEFORE any chunk decode happens.
    def explode_get_nowait():
        raise RuntimeError("queue corruption")

    streamer.queue.get_nowait = explode_get_nowait  # type: ignore[method-assign]
    streamer._cancel_event.set()  # force immediate drain path

    import myvoice.observability.metrics as _m
    original_record = _m.record
    _m.record = fake_metrics
    try:
        worker = StreamingDecoderWorker(
            streamer=streamer,
            decode_fn=_make_decoded_pcm,
            post_mutation=fake_post,
            session_id="abc-123",
        )
        worker.start()
        worker.join(timeout=2.0)
    finally:
        _m.record = original_record

    # Worker exited cleanly — RuntimeError did NOT kill the thread.
    assert worker.is_alive() is False

    # drain_error metric was recorded with numeric value + repr in tags.
    drain_errors = [
        c for c in fake_metrics.calls if c["metric_name"] == "drain_error"
    ]
    assert len(drain_errors) == 1
    assert isinstance(drain_errors[0]["value"], (int, float))
    assert "queue corruption" in drain_errors[0]["tags"]["error_repr"]

    # Cancel post still fired despite drain raising.
    assert fake_post.calls == [("cancel", "abc-123")]


def test_post_terminal_records_metric_when_post_mutation_raises_and_does_not_kill_thread():
    """M1 regression: `_post_terminal` previously called `_post_mutation`
    unguarded. Story 16.6's wiring will pass `registry.post_mutation`
    (a bound method that wraps `QMetaObject.invokeMethod` and validates
    against `_MUTATION_SLOT_NAMES`). A raising post_mutation would have
    killed the worker thread silently mid-handoff, leaving no record.

    This test verifies a `post_mutation_error` metric is recorded
    instead, and the worker thread exits cleanly.
    """
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    streamer.queue.put(END_OF_STREAM)  # immediate finalize path
    fake_metrics = _RecordingMetrics()

    def raising_post(*args):
        raise RuntimeError("registry rejected mutation")

    import myvoice.observability.metrics as _m
    original_record = _m.record
    _m.record = fake_metrics
    try:
        worker = StreamingDecoderWorker(
            streamer=streamer,
            decode_fn=_make_decoded_pcm,
            post_mutation=raising_post,
            session_id="abc-123",
        )
        worker.start()
        worker.join(timeout=2.0)
    finally:
        _m.record = original_record

    # Thread did NOT die from the raising post_mutation.
    assert worker.is_alive() is False

    # post_mutation_error metric recorded with method_name + error_repr.
    pm_errors = [
        c for c in fake_metrics.calls if c["metric_name"] == "post_mutation_error"
    ]
    assert len(pm_errors) == 1
    assert pm_errors[0]["tags"]["method_name"] == "finalize"
    assert "registry rejected mutation" in pm_errors[0]["tags"]["error_repr"]


# ============================================================================
# AC #7 — metrics.record('decode_chunk_latency_ms', ...) per decoded chunk.
# ============================================================================


def test_metrics_record_decode_chunk_latency_ms_per_chunk(monkeypatch):
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    fake_metrics = _RecordingMetrics()
    monkeypatch.setattr(
        "myvoice.observability.metrics.record", fake_metrics
    )

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put([9, 10, 11, 12, 13, 14])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
        model_type="qwen3_tts",
        hardware="gpu",
    )
    worker.start()
    worker.join(timeout=2.0)

    latency_calls = [
        c
        for c in fake_metrics.calls
        if c["metric_name"] == "decode_chunk_latency_ms"
    ]
    assert len(latency_calls) == 3
    for c in latency_calls:
        assert isinstance(c["value"], float)
        assert c["value"] > 0.0  # nonzero perf_counter delta
        # session_id is a top-level kwarg-only param on the real
        # metrics.record signature — NOT a tag. The recording fake
        # mirrors this so tests cannot drift from production wire
        # format (H2 review-fix).
        assert c["session_id"] == "abc-123"
        assert c["tags"]["model_type"] == "qwen3_tts"
        assert c["tags"]["hardware"] == "gpu"


def test_metrics_record_skipped_for_failed_decode(monkeypatch):
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    fake_post = _RecordingPostMutation()
    fake_metrics = _RecordingMetrics()
    monkeypatch.setattr(
        "myvoice.observability.metrics.record", fake_metrics
    )

    def always_raise(chunk):
        raise RuntimeError("fail")

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=always_raise,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    latency_calls = [
        c
        for c in fake_metrics.calls
        if c["metric_name"] == "decode_chunk_latency_ms"
    ]
    assert len(latency_calls) == 0
    # decode_error was recorded instead.
    error_calls = [
        c for c in fake_metrics.calls if c["metric_name"] == "decode_error"
    ]
    assert len(error_calls) == 1


# ============================================================================
# AC #8 — module imports honor architecture line 674.
# ============================================================================


def test_module_imports_match_architecture_line_674_via_ast():
    """ast.parse the module's source and assert its import set matches the
    architecturally-mandated allow-list. Avoids literal-string scans
    (which self-trip on the module's own "may NOT import" docstring).

    Resolves the source path via `inspect.getsourcefile` so a future
    test reorganization cannot silently break the path arithmetic
    (M3 review-fix; was Path(__file__).parents[4]).
    """
    src_path = Path(inspect.getsourcefile(_streaming_decoder))
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # node.module is None for `from . import x` (relative); not used here.
            if node.module is not None:
                imported_modules.add(node.module)

    # `queue` is added per H3 review-fix (narrow `except queue.Empty:`
    # in `_drain_and_post_cancel`). It's stdlib like `threading` /
    # `time`, both already on the architecture line 674 allow-list.
    allowed = {
        "queue",
        "threading",
        "time",
        "typing",
        "numpy",
        "myvoice.observability",
        "myvoice.services.tts_streaming.codec_token_streamer",
    }
    assert imported_modules == allowed, (
        f"streaming_decoder.py imports diverged from architecture line 674. "
        f"Expected {allowed}, got {imported_modules}. "
        f"Diff: extra={imported_modules - allowed}, missing={allowed - imported_modules}"
    )

    # Forbidden module-name prefixes (defensive — if any future edit adds
    # one of these, this test fails before it can land).
    forbidden_prefixes = (
        "myvoice.services.sessions",
        "myvoice.services.audio_coordinator",
        "myvoice.services.qwen_tts_service",
        "myvoice.models",
        "myvoice.ui",
        "PyQt6",
        "qwen_tts",
    )
    for mod in imported_modules:
        for prefix in forbidden_prefixes:
            assert not mod.startswith(prefix), (
                f"streaming_decoder.py imports forbidden module {mod!r} "
                f"(matches forbidden prefix {prefix!r}); architecture "
                f"line 674-675 forbids this. The decoder worker is "
                f"decoder-agnostic and posts via callable supplied at init."
            )


# ============================================================================
# AC #10 — test discovery + DLL preamble work without per-directory conftest.
# ============================================================================


def test_existing_tests_discover_and_import_module():
    """Smoke test confirming that pytest discovery + the conftest preamble
    (tests/conftest.py:12-50) produce a working import for this module.
    If this file's tests are running, the discovery worked. The assertion
    is a tautology against module-level state to make the AC #10 check
    explicit in the test report.
    """
    from myvoice.services.tts_streaming import streaming_decoder
    assert streaming_decoder.StreamingDecoderWorker is StreamingDecoderWorker
    # Metrics helper imports cleanly via the bundled portable Python.
    from myvoice.observability import metrics as _metrics
    assert callable(_metrics.record)
