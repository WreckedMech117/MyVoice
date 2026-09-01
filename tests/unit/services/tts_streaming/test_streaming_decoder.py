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
    # (`enable_tf32_and_cudnn_benchmark`, `is_ampere_or_newer`); Story 18.3
    # added one more (`resolve_tts_precision`); Story 18.4 appends two
    # (`engage_compile_optimizations` + the `compile_cache` module re-export);
    # the compile-disengage-post-generation-reload spec appends
    # `apply_reload_compile_fix` + `collect_compile_gate_diagnostic` —
    # append-only declaration-order discipline.
    assert pkg.__all__ == [
        "StreamingMode",
        "default_streaming_mode_for_hardware",
        "effective_streaming_mode",
        "CodecTokenStreamer",
        "END_OF_STREAM",
        "StreamingDecoderWorker",
        "apply_reload_compile_fix",
        "collect_compile_gate_diagnostic",
        "enable_tf32_and_cudnn_benchmark",
        "is_ampere_or_newer",
        "resolve_tts_precision",
        "engage_compile_optimizations",
        "compile_cache",
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
    instead, and the worker thread exits cleanly. Exercises the
    finalize-success path by feeding one chunk before END_OF_STREAM so
    the worker's empty-chunks branch (try_set_error + discard, added to
    avoid the no-chunks finalize ValueError) is not the one under test.
    """
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    streamer.queue.put([1, 2, 3, 4, 5, 6])  # one chunk → appended > 0
    streamer.queue.put(END_OF_STREAM)
    fake_metrics = _RecordingMetrics()

    def raising_post(*args):
        # Only raise on the finalize terminal (which is what M1 covers);
        # let append_chunk through so the appended counter increments and
        # the END_OF_STREAM branch chooses the finalize post.
        if len(args) >= 1 and args[0] == "finalize":
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


def test_end_of_stream_with_no_chunks_routes_to_try_set_error_and_discard():
    """When the TRUE_STREAM talker raises mid-stream and pushes END_OF_STREAM
    without any chunks ever being appended (qwen_tts_service.py:4006-4014
    talker-exception push), the worker MUST NOT post finalize — that would
    raise ValueError inside the registry slot (session_registry.py:433 ->
    generation_session.py:163) which the global exception handler
    surfaces as a user-visible dialog even though the dispatch's fallback
    chain produces audio.

    Regression test for the May-2026 'finalize() called with no chunks'
    dialog observed when a newly-added cloned voice's first compile-
    primed generation hit a torch.compile failure.
    """
    streamer = _build_streamer(chunk_size=4, lookahead=2)
    streamer.queue.put(END_OF_STREAM)  # talker raised → no chunks appended
    fake_post = _RecordingPostMutation()

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,
        post_mutation=fake_post,
        session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    # try_set_error then discard — no finalize.
    assert fake_post.calls == [
        ("try_set_error", "abc-123"),
        ("discard", "abc-123"),
    ]


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

# ============================================================================
# Story 20.1 Task 2.1 — ttfa_first_decode_complete_ms (segment-3 boundary).
# ============================================================================


def test_metrics_record_ttfa_first_decode_complete_only_for_first_chunk(
    monkeypatch,
):
    """The segment-3 boundary closes when the FIRST token chunk has become
    PCM. It must be one-shot per worker even though ``decode_chunk_latency_ms``
    fires per chunk, and it must be instance state so two concurrent
    sessions do not suppress each other's boundary.
    """
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
    )
    worker.start()
    worker.join(timeout=2.0)

    boundary = [
        c
        for c in fake_metrics.calls
        if c["metric_name"] == "ttfa_first_decode_complete_ms"
    ]
    latency = [
        c
        for c in fake_metrics.calls
        if c["metric_name"] == "decode_chunk_latency_ms"
    ]
    assert len(latency) == 3, "per-chunk latency metric must be unchanged"
    assert len(boundary) == 1, (
        "ttfa_first_decode_complete_ms must fire exactly once per session, "
        f"not once per chunk; got {len(boundary)}"
    )
    rec = boundary[0]
    assert rec["session_id"] == "abc-123"
    assert rec["tags"]["chunk_index"] == 0
    assert rec["tags"]["pcm_samples"] > 0
    # Absolute wall-clock ms (joins the other boundaries by subtraction),
    # NOT the elapsed value decode_chunk_latency_ms carries.
    assert rec["value"] > 1_600_000_000_000.0

    # Instance state, not module state: a second worker re-arms.
    streamer2 = _build_streamer(chunk_size=4, lookahead=2)
    streamer2.queue.put([1, 2, 3, 4, 5, 6])
    streamer2.queue.put(END_OF_STREAM)
    worker2 = StreamingDecoderWorker(
        streamer=streamer2,
        decode_fn=_make_decoded_pcm,
        post_mutation=_RecordingPostMutation(),
        session_id="def-456",
    )
    worker2.start()
    worker2.join(timeout=2.0)

    boundary_all = [
        c
        for c in fake_metrics.calls
        if c["metric_name"] == "ttfa_first_decode_complete_ms"
    ]
    assert len(boundary_all) == 2, (
        "a second worker must emit its own segment-3 boundary; the one-shot "
        "guard must be per-instance"
    )
    assert boundary_all[1]["session_id"] == "def-456"


# ============================================================================
# Story 20.4 — exact splice + decoder-side overlap-add.
#
# The pre-20.4 tests above all use ``_make_decoded_pcm``, which returns ONE
# sample per token. That does not satisfy the codec's output-length identity
# (``1920*frames - 555``), so those tests exercise the legacy proportional
# trim — deliberately, and they still pin it, because it is the live
# fallback for an unrecognised codec geometry.
#
# The rows below use a decode_fn that reproduces the REAL codec's geometry,
# so they cover the path production actually takes.
# ============================================================================


SPF = _streaming_decoder._CODEC_SAMPLES_PER_FRAME
EDGE = _streaming_decoder._CODEC_EDGE_LOSS_SAMPLES


def _codec_like_decode(chunk):
    """A decode_fn with the real codec's measured output geometry.

    Returns ``1920*N - 555`` samples for N frames, and — importantly for
    the alignment assertions — makes the sample VALUE a function of the
    absolute audio position, so two chunks decoding the same moment produce
    the same value. Token ``t`` stands in for codec frame index ``t``.

    The fixed edge loss is taken off the TAIL here. Which end it actually
    comes off does not matter to the splice arithmetic (the cross-
    correlation lag is ``chunk_size*1920`` either way, because both decodes
    lose the same amount), and this file cannot observe it.
    """
    n = len(chunk)
    first_frame = int(chunk[0])
    total = SPF * n - EDGE
    start = first_frame * SPF
    return np.arange(start, start + total, dtype=np.float32)


def _frames(first, count):
    return list(range(first, first + count))


def test_exact_splice_advances_by_chunk_size_frames_of_audio():
    """Story 20.4 defect 1 — the posted chunk must be cs*1920 samples.

    The pre-20.4 arithmetic posted ``1920*(cs+la) - 555 - round(la*(...)/(cs+la))``
    = ``cs*1920 - 555*cs/(cs+la)``, i.e. 370 samples short at chunk_size=10.
    Those samples are real speech, and they were deleted at every seam.
    """
    cs, la = 10, 5
    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    streamer.queue.put(_frames(0, cs + la))
    streamer.queue.put(_frames(cs, cs + la))
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_codec_like_decode,
        post_mutation=fake_post, session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=5.0)

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    assert len(posted) == 2
    assert posted[0].size == cs * SPF == 19200
    # The pre-20.4 value, pinned so a revert is unmistakable.
    legacy = (SPF * (cs + la) - EDGE) - round(
        la * (SPF * (cs + la) - EDGE) / (cs + la)
    )
    assert legacy == 18830
    assert posted[0].size - legacy == 370


def test_stitched_stream_is_time_contiguous_across_the_seam():
    """No audio is dropped or duplicated at a chunk boundary.

    ``_codec_like_decode`` makes each sample's VALUE its absolute position,
    so a contiguous stitch is an arithmetic sequence with step 1 straight
    through the seam. A dropped span shows as a jump; a duplicated one as a
    repeat. This is the property the cross-correlation established on real
    audio, asserted directly here.
    """
    cs, la = 10, 5
    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    for k in range(3):
        streamer.queue.put(_frames(k * cs, cs + la))
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_codec_like_decode,
        post_mutation=fake_post, session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=5.0)

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    stitched = np.concatenate(posted)
    # Both decodes agree on the shared audio, so the blend is an identity
    # here and the whole stream must be exactly position-valued.
    expected = np.arange(stitched.size, dtype=np.float32)
    # atol is float32-ULP-aware, not slack: at these magnitudes (3e4) one
    # ULP is ~2e-3 and the blend rounds once. A single dropped or
    # duplicated sample would show as an error of 1.0 - twenty times this
    # tolerance - so the assertion still has all its teeth.
    np.testing.assert_allclose(stitched, expected, rtol=0, atol=0.05)


def test_overlap_add_blends_the_previously_discarded_tail():
    """Story 20.4 defect 2 — the retained tail is cross-faded, not dropped.

    Alignment alone butts two independently-decoded renditions of the same
    instant together and measurably makes the click WORSE (8.5x -> 18.0x
    the non-seam baseline at chunk_size=25). The tail this module used to
    discard is that same instant, decoded with future context, so it is
    retained and ramped in.

    Here the second chunk's decode is offset by a constant, so the blended
    head must be a linear ramp between the two renditions rather than
    either one of them.
    """
    cs, la = 10, 5
    offset = 100.0

    def _offset_decode(chunk):
        pcm = _codec_like_decode(chunk)
        # Only the SECOND chunk is offset, so the seam has something to
        # reconcile.
        return pcm + offset if int(chunk[0]) == cs else pcm

    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    streamer.queue.put(_frames(0, cs + la))
    streamer.queue.put(_frames(cs, cs + la))
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_offset_decode,
        post_mutation=fake_post, session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=5.0)

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    w = _streaming_decoder._OVERLAP_ADD_SAMPLES
    head = posted[1][:w]
    base = np.arange(cs * SPF, cs * SPF + w, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    # previous chunk's rendition = base; this chunk's = base + offset.
    np.testing.assert_allclose(head, base + offset * ramp, rtol=0, atol=1e-2)
    # First sample takes the PREVIOUS chunk's value, last takes this one's:
    # that is what makes the transition continuous with what was already
    # posted rather than stepping to the new rendition instantly.
    assert abs(float(head[0]) - float(base[0])) < 1e-2
    assert abs(float(head[-1]) - float(base[-1] + offset)) < 1e-1
    # Past the blend the segment is untouched.
    np.testing.assert_allclose(
        posted[1][w:w + 5],
        np.arange(cs * SPF + w, cs * SPF + w + 5, dtype=np.float32) + offset,
        rtol=0, atol=1e-2,
    )


def test_overlap_add_is_clamped_to_the_audio_the_chunks_actually_share():
    """The blend can never read past the shared region.

    The two chunks only both cover ``lookahead*1920 - 555`` samples. A
    larger ``_OVERLAP_ADD_SAMPLES`` must clamp to that rather than reaching
    into audio the previous chunk never decoded.
    """
    cs, la = 10, 5
    shared = la * SPF - EDGE
    assert shared == 9045
    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    streamer.queue.put(_frames(0, cs + la))
    streamer.queue.put(_frames(cs, cs + la))
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_codec_like_decode,
        post_mutation=fake_post, session_id="abc-123",
    )
    # Far wider than the shared region.
    worker._pending_overlap = None
    original = _streaming_decoder._OVERLAP_ADD_SAMPLES
    try:
        _streaming_decoder._OVERLAP_ADD_SAMPLES = 1_000_000
        worker.start()
        worker.join(timeout=5.0)
    finally:
        _streaming_decoder._OVERLAP_ADD_SAMPLES = original

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    assert len(posted) == 2
    assert posted[0].size == cs * SPF
    # Still contiguous — the clamp did not corrupt the splice.
    stitched = np.concatenate(posted)
    np.testing.assert_allclose(
        stitched[:cs * SPF + 100],
        np.arange(cs * SPF + 100, dtype=np.float32), rtol=0, atol=0.05,
    )


def test_residual_chunk_still_posted_whole_and_still_blended():
    """The residual carries no trim, but it DOES start at a seam.

    Its head covers the same audio the previous full chunk's retained tail
    does, so it must be blended like any other chunk head. Missing this
    would leave one un-treated seam per utterance — the last one, right
    before the audio ends, which is a conspicuous place for a click.
    """
    cs, la = 10, 5
    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    streamer.queue.put(_frames(0, cs + la))
    streamer.queue.put(_frames(cs, 6))          # residual: 6 < cs + la
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=_codec_like_decode,
        post_mutation=fake_post, session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=5.0)

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    assert len(posted) == 2
    assert posted[0].size == cs * SPF
    assert posted[1].size == SPF * 6 - EDGE      # posted whole, no trim
    stitched = np.concatenate(posted)
    np.testing.assert_allclose(
        stitched, np.arange(stitched.size, dtype=np.float32),
        rtol=0, atol=0.05,
    )


def test_unrecognised_codec_geometry_falls_back_and_says_so(monkeypatch):
    """A codec/pin change must degrade to the old behaviour LOUDLY.

    The exact splice is only correct while ``decode(N) == 1920*N - 555``.
    If that identity ever stops holding, computing a splice from it would
    mis-cut every chunk — silently, and worse than the bug this replaces.
    So the identity is verified per chunk, and its failure drops the
    session to the pre-20.4 proportional trim with a metric.
    """
    cs, la = 4, 2
    streamer = _build_streamer(chunk_size=cs, lookahead=la)
    fake_post = _RecordingPostMutation()
    recorder = _RecordingMetrics()
    monkeypatch.setattr("myvoice.observability.metrics.record", recorder)

    streamer.queue.put([1, 2, 3, 4, 5, 6])
    streamer.queue.put([5, 6, 7, 8, 9, 10])
    streamer.queue.put(END_OF_STREAM)

    worker = StreamingDecoderWorker(
        streamer=streamer,
        decode_fn=_make_decoded_pcm,   # one sample per token: identity fails
        post_mutation=fake_post, session_id="abc-123",
    )
    worker.start()
    worker.join(timeout=5.0)

    posted = [c[2] for c in fake_post.calls if c[0] == "append_chunk"]
    # Legacy proportional trim: 6 samples - round(2*6/6) = 4.
    assert [p.size for p in posted] == [4, 4]

    names = [c["metric_name"] for c in recorder.calls]
    assert "decode_geometry_unverified" in names, (
        "an unrecognised codec geometry must be visible in telemetry, not "
        "silently absorbed"
    )
    # And exactly once per session, not once per chunk.
    assert names.count("decode_geometry_unverified") == 1


def test_codec_geometry_constants_match_the_measured_model():
    """Pin the two measured numbers and the invariants that depend on them.

    ``1920`` samples/frame is 12.5 Hz, not the 12 Hz this codebase's prose
    assumed from Story 16.3 to Story 20.3. Both constants were solved from
    posted chunk lengths and cross-checked against 14 residual lengths
    (``20-4-seam-analysis.py``). A change to either without re-running that
    measurement is a silent audio-quality regression.
    """
    assert SPF == 1920
    assert EDGE == 555
    assert 24000 / SPF == 12.5
    # The overlap-add width must fit the audio two chunks actually share at
    # the committed lookahead, or the blend would be clamped and the swept
    # value would not be the value in effect.
    from myvoice.services.tts_streaming import codec_token_streamer
    shared = codec_token_streamer.DEFAULT_LOOKAHEAD * SPF - EDGE
    assert _streaming_decoder._OVERLAP_ADD_SAMPLES <= shared
