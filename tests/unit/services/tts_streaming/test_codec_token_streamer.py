"""Tests for CodecTokenStreamer (Story 16.3).

Verifies P-5 (architecture-optimization-pass.md:415-429) — the streamer's
three-method contract; D-10 (line 259) — bounded queue with backpressure;
D-11 (line 261) — cooperative cancellation via threading.Event.
"""

import ast
import queue
import threading
from pathlib import Path

import pytest
import torch
from transformers.generation.streamers import BaseStreamer

from myvoice.services.tts_streaming import (
    CodecTokenStreamer,
    END_OF_STREAM,
)
from myvoice.services.tts_streaming import codec_token_streamer


# -- AC #1: importable from package; inherits from BaseStreamer --------- #


def test_streamer_inherits_basestreamer():
    assert issubclass(CodecTokenStreamer, BaseStreamer)
    assert BaseStreamer in CodecTokenStreamer.__mro__


def test_end_of_stream_is_module_singleton():
    # Decoder worker (Story 16.4) compares via `is`, not `==`.
    from myvoice.services.tts_streaming.codec_token_streamer import (
        END_OF_STREAM as direct_import,
    )
    assert direct_import is END_OF_STREAM


def test_end_of_stream_is_not_a_list():
    # Defensive: a confused decoder iterating chunk-as-list must not
    # accidentally treat the sentinel as a single-token chunk.
    assert not isinstance(END_OF_STREAM, list)
    assert not isinstance(END_OF_STREAM, tuple)


def test_package_all_lists_expected_symbols_in_order():
    import myvoice.services.tts_streaming as pkg
    # Story 16.2's three names, then Story 16.3's two, then Story 16.4's
    # one, then Story 18.2's two, then Story 18.3's one, then Story 18.4's
    # two (engage_compile_optimizations + compile_cache module), then the
    # compile-disengage-post-generation-reload spec's two
    # (apply_reload_compile_fix + collect_compile_gate_diagnostic) —
    # declaration order matches each story's append-only precedent.
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


# -- AC #2: construction defaults + validation -------------------------- #


def test_default_construction_uses_documented_constants():
    """A bare ``CodecTokenStreamer()`` picks up the module constants.

    Deliberately derived, not literal: the point of this row is that the
    constructor defaults track the module constants, which is what makes
    a retune a one-line edit. The committed VALUES are pinned separately
    by ``test_committed_chunk_geometry_is_story_20_4_optimum``.
    """
    s = CodecTokenStreamer()
    assert s.chunk_size == codec_token_streamer.DEFAULT_CHUNK_SIZE
    assert s.lookahead == codec_token_streamer.DEFAULT_LOOKAHEAD
    # D-10: maxsize = 4 * chunk_size
    assert s.queue.maxsize == (
        codec_token_streamer.DEFAULT_QUEUE_MAX_FACTOR
        * codec_token_streamer.DEFAULT_CHUNK_SIZE
    )
    assert isinstance(s.queue, queue.Queue)
    assert isinstance(s._cancel_event, threading.Event)
    assert not s._cancel_event.is_set()
    assert s._buffer == []


def test_committed_chunk_geometry_is_25_plus_5_after_the_20_4_revert():
    """Story 20.4 AC #1 — pin the committed geometry, and why it is 25.

    Story 20.1 SS5.2/SS5.3 measured chunk_size = 10 as the latency optimum
    and Story 20.4 shipped it as far as the NFR3 gate, where it failed
    twice: seam artefacts scale with seam count, and 10 has 2.5x the seams
    of 25. It was reverted. A round-3 audition then showed the SEAM FIX is
    good on its own at 25, so the two changes are separable and only the
    stitching fix ships.

    This row exists so a future retune cannot be done on the latency sweep
    alone — that sweep already said 10, and the ear disagreed. Moving this
    number requires an NFR3 audition, and chunk_size = 15 in particular is
    perceptually untested (1.5x the seams of 25).
    """
    assert codec_token_streamer.DEFAULT_CHUNK_SIZE == 25
    assert codec_token_streamer.DEFAULT_LOOKAHEAD == 5
    # The consumer's 500 ms static watermark must stay a no-op: a chunk
    # carries chunk_size/12.5 s of audio at the codec's measured frame
    # rate, and if that falls under the watermark the consumer holds two
    # chunks and hands the producer-side gain straight back (Story 20.1
    # SS5.4 measured exactly that at chunk_size = 5).
    assert codec_token_streamer.DEFAULT_CHUNK_SIZE / 12.5 >= 0.5


def test_custom_construction_honors_all_parameters():
    injected = threading.Event()
    s = CodecTokenStreamer(
        chunk_size=10,
        lookahead=2,
        queue_max_factor=8,
        cancel_event=injected,
    )
    assert s.chunk_size == 10
    assert s.lookahead == 2
    assert s.queue.maxsize == 80
    assert s._cancel_event is injected


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_chunk_size_must_be_positive(bad):
    with pytest.raises(ValueError, match="chunk_size"):
        CodecTokenStreamer(chunk_size=bad)


@pytest.mark.parametrize("bad", [-1, -5])
def test_lookahead_must_be_non_negative(bad):
    with pytest.raises(ValueError, match="lookahead"):
        CodecTokenStreamer(lookahead=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_queue_max_factor_must_be_positive(bad):
    # maxsize=0 is "infinite" in queue.Queue — that's the D-10
    # invariant violation this guard prevents.
    with pytest.raises(ValueError, match="queue_max_factor"):
        CodecTokenStreamer(queue_max_factor=bad)


# -- AC #3: buffering, push at threshold, slide-by-chunk_size ---------- #


def test_put_below_threshold_does_not_push():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(torch.tensor([[1, 2, 3]]))
    assert s.queue.empty()


def test_put_at_threshold_pushes_first_chunk_and_keeps_lookahead_tail():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(torch.tensor([[1, 2, 3]]))
    s.put(torch.tensor([[4, 5, 6]]))
    chunk = s.queue.get_nowait()
    assert chunk == [1, 2, 3, 4, 5, 6]
    # Lookahead tail kept for next chunk's left-context (overlap-add).
    assert s._buffer == [5, 6]


def test_subsequent_chunks_share_lookahead_with_previous():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    s.queue.get_nowait()  # discard first chunk
    s.put(torch.tensor([[7, 8, 9, 10]]))
    chunk = s.queue.get_nowait()
    # Shares [5, 6] with previous chunk's tail.
    assert chunk == [5, 6, 7, 8, 9, 10]
    assert s._buffer == [9, 10]


def test_single_put_releasing_multiple_chunks():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    # 14 tokens → three chunks: [1..6], [5..10], [9..14]; tail = [13, 14]
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]))
    c1 = s.queue.get_nowait()
    c2 = s.queue.get_nowait()
    c3 = s.queue.get_nowait()
    assert c1 == [1, 2, 3, 4, 5, 6]
    assert c2 == [5, 6, 7, 8, 9, 10]
    assert c3 == [9, 10, 11, 12, 13, 14]
    assert s._buffer == [13, 14]
    assert s.queue.empty()


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[1, 2, 3, 4, 5, 6]]),  # batch-1 2-D tensor
        torch.tensor([1, 2, 3, 4, 5, 6]),    # 1-D tensor
        [1, 2, 3, 4, 5, 6],                   # list
        (1, 2, 3, 4, 5, 6),                   # tuple
    ],
)
def test_extract_tokens_handles_multiple_input_shapes(value):
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(value)
    chunk = s.queue.get_nowait()
    assert chunk == [1, 2, 3, 4, 5, 6]


def test_extract_tokens_handles_zero_dim_scalar_tensor():
    # _extract_tokens has a defensive branch for 0-D scalar tensors:
    # `value.tolist() if dim>0 else [value.item()]`. HF .generate() in
    # some configurations may emit per-token scalars (dim=0) rather than
    # 1-D / batched tensors. Verify the branch round-trips correctly.
    s = CodecTokenStreamer(chunk_size=2, lookahead=0)
    s.put(torch.tensor(5))   # 0-D scalar tensor
    s.put(torch.tensor(6))   # 0-D scalar — completes the 2-token chunk
    chunk = s.queue.get_nowait()
    assert chunk == [5, 6]


def test_put_with_batch_greater_than_one_tensor_raises_value_error():
    # HF streaming contracts batch=1 per put() call. A [batch>1, seq_len]
    # tensor arriving here is a contract violation that previously
    # produced nested-list "tokens" silently. _extract_tokens must reject
    # the misuse loudly so the breakage surfaces at the boundary.
    s = CodecTokenStreamer(chunk_size=2, lookahead=0)
    with pytest.raises(ValueError, match="batch=1"):
        s.put(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    # Buffer is unchanged — the raise happens before any extend().
    assert s._buffer == []
    assert s.queue.empty()


@pytest.mark.parametrize(
    "empty",
    [
        torch.tensor([[]], dtype=torch.long),  # batch-1 empty 2-D tensor
        torch.tensor([], dtype=torch.long),    # 1-D empty tensor
        [],                                     # empty list
        (),                                     # empty tuple
    ],
)
def test_put_with_empty_input_is_a_safe_noop(empty):
    # HF .generate() can deliver a zero-token batch in pathological
    # short-input cases (e.g., the talker emits EOS as the very first
    # token). Defensive: no chunk pushed, buffer unchanged, no crash.
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(empty)
    assert s.queue.empty()
    assert s._buffer == []


def test_lookahead_zero_produces_non_overlapping_chunks():
    # AC #2 allows lookahead=0 ("no overlap-add"). Verify the slide-by-
    # chunk_size semantics still hold: consecutive chunks share zero
    # tokens, buffer is empty after each push.
    s = CodecTokenStreamer(chunk_size=3, lookahead=0)
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    c1 = s.queue.get_nowait()
    c2 = s.queue.get_nowait()
    c3 = s.queue.get_nowait()
    assert c1 == [1, 2, 3]
    assert c2 == [4, 5, 6]
    assert c3 == [7, 8, 9]
    assert s.queue.empty()
    assert s._buffer == []  # no lookahead tail kept


# -- AC #4: backpressure (block on full queue) ------------------------- #


def test_put_blocks_when_queue_is_full():
    # maxsize = 1 * 2 = 2
    s = CodecTokenStreamer(chunk_size=2, lookahead=0, queue_max_factor=1)
    s.put(torch.tensor([[1, 2, 3, 4]]))  # pushes 2 chunks; queue full
    assert s.queue.qsize() == 2

    completed = []

    def background_put():
        s.put(torch.tensor([[5, 6]]))
        completed.append(True)

    t = threading.Thread(target=background_put, daemon=True)
    t.start()
    # join() with a small timeout returns without the thread completing
    # if put() is correctly blocked on the full queue. If put() raised
    # queue.Full or returned without pushing, the thread would complete
    # immediately and join() would return with is_alive() == False.
    t.join(timeout=0.1)
    assert t.is_alive(), "put() should be blocked on full queue"
    assert completed == []

    # Drain one slot — blocked put() unblocks.
    s.queue.get_nowait()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert completed == [True]


# -- AC #5: end() flushes residual + END_OF_STREAM marker -------------- #


def test_end_flushes_residual_buffer_then_pushes_marker():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(torch.tensor([[1, 2, 3]]))
    s.end()
    residual = s.queue.get_nowait()
    marker = s.queue.get_nowait()
    assert residual == [1, 2, 3]
    assert marker is END_OF_STREAM
    assert s.queue.empty()
    assert s._buffer == []


def test_end_with_empty_buffer_pushes_only_marker():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    # No put() calls — buffer is empty.
    s.end()
    first = s.queue.get_nowait()
    assert first is END_OF_STREAM
    assert s.queue.empty()


def test_end_when_buffer_emptied_by_chunk_boundary_pushes_only_marker():
    # AC #5 explicit scenario: "the last put() happened to land exactly
    # at a chunk boundary" — buffer slid to empty after a chunk push,
    # end() must push only END_OF_STREAM (no zero-length chunk).
    # With lookahead=0 and chunk_size=3, putting exactly 6 tokens pushes
    # two chunks and slides the buffer to []. (Lookahead>0 cannot reach
    # this state because the lookahead tail always remains.)
    s = CodecTokenStreamer(chunk_size=3, lookahead=0)
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    s.queue.get_nowait()  # discard chunk 1
    s.queue.get_nowait()  # discard chunk 2
    assert s._buffer == []
    s.end()
    first = s.queue.get_nowait()
    assert first is END_OF_STREAM
    assert s.queue.empty()


# -- AC #6: cancel event makes put() a no-op --------------------------- #


def test_set_cancel_event_makes_put_a_noop():
    s = CodecTokenStreamer(chunk_size=2, lookahead=0)
    s._cancel_event.set()
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    assert s.queue.empty()
    assert s._buffer == []


def test_cancel_does_not_drain_existing_chunks():
    # Pre-cancel chunks remain — Story 16.4's decoder is responsible
    # for draining them on the cancel side. P-7 invariant.
    s = CodecTokenStreamer(chunk_size=2, lookahead=0)
    s.put(torch.tensor([[1, 2, 3, 4]]))
    depth_before = s.queue.qsize()
    assert depth_before == 2
    s._cancel_event.set()
    s.put(torch.tensor([[5, 6]]))
    assert s.queue.qsize() == depth_before


def test_injected_cancel_event_observed_by_streamer():
    # Story 16.5 wires the registry's event into both the streamer and
    # the decoder worker. This test verifies the streamer end of that.
    injected = threading.Event()
    s = CodecTokenStreamer(chunk_size=2, lookahead=0, cancel_event=injected)
    injected.set()
    s.put(torch.tensor([[1, 2, 3, 4]]))
    assert s.queue.empty()


def test_cancel_set_mid_multi_chunk_batch_stops_subsequent_pushes():
    # If a single put() call would release multiple chunks (multi-token
    # batch from HF) and cancel fires after the first chunk lands, the
    # while loop must recheck the cancel event and stop pushing the rest.
    # Without the recheck, a decoder that has already exited on cancel
    # leaves the producer pushing chunks into a queue with no consumer
    # and eventually deadlocks on backpressure.
    cancel = threading.Event()

    class CancelOnFirstPut(queue.Queue):
        def put(self, item, block=True, timeout=None):
            super().put(item, block, timeout)
            cancel.set()

    s = CodecTokenStreamer(
        chunk_size=2, lookahead=0, cancel_event=cancel
    )
    # Replace the bounded queue with one that flips cancel after the
    # first push lands. Buffer-extend already happened in put(), so the
    # while loop's recheck is the only thing that prevents the rest.
    s.queue = CancelOnFirstPut(maxsize=8)
    # 8 tokens → 4 chunks if uncancelled; expect only 1 with the recheck.
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
    assert s.queue.qsize() == 1
    assert s.queue.get_nowait() == [1, 2]


# -- AC #7: reset() clears state -------------------------------------- #


def test_reset_clears_buffer_queue_and_cancel_event():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    s._cancel_event.set()
    s.reset()
    assert s._buffer == []
    assert s.queue.empty()
    assert not s._cancel_event.is_set()


def test_reset_allows_reuse_for_subsequent_session():
    s = CodecTokenStreamer(chunk_size=4, lookahead=2)
    # Cycle 1
    s.put(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    s.end()
    s.reset()
    # Cycle 2 — same instance, expect identical observable behavior.
    s.put(torch.tensor([[10, 20, 30, 40, 50, 60]]))
    chunk = s.queue.get_nowait()
    assert chunk == [10, 20, 30, 40, 50, 60]
    assert s._buffer == [50, 60]


# -- AC #8: forbidden-imports invariant --------------------------------- #


def test_module_does_not_import_forbidden_peers():
    # Architecture line 671 ("may NOT import: sessions, services") covers
    # both `from X import Y` and bare `import X` forms. Parse the AST so
    # the check is symmetric across forms and immune to forbidden-name
    # prose appearing inside docstrings or comments.
    from myvoice.services.tts_streaming import codec_token_streamer as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_prefixes = (
        "myvoice.services.sessions",
        "myvoice.services.audio_coordinator",
        "myvoice.services.qwen_tts_service",
        "myvoice.observability",
        "myvoice.models",
        "PyQt6",
    )

    def is_forbidden(name: str) -> str:
        for prefix in forbidden_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return prefix
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = is_forbidden(alias.name)
                assert not hit, (
                    f"P-5/architecture line 671 violation: "
                    f"codec_token_streamer.py imports forbidden peer "
                    f"{alias.name!r} (matches {hit!r})"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            hit = is_forbidden(mod_name)
            assert not hit, (
                f"P-5/architecture line 671 violation: "
                f"codec_token_streamer.py imports from forbidden peer "
                f"{mod_name!r} (matches {hit!r})"
            )
