"""CodecTokenStreamer — bounded-queue HF streamer for true streaming TTS.

Story 16.3 — Phase ⊥ of D-20 (architecture-optimization-pass.md).

Architecture references:
  - P-5 (lines 415-429): three responsibilities — put(value) buffers and
    pushes chunks with backpressure, end() flushes + END_OF_STREAM marker,
    reset() clears state between sessions. Forbidden: never calls into
    session, registry, or audio coordinator.
  - D-10 (line 259): bounded queue.Queue(maxsize = 4 * chunk_size) with
    backpressure when full. HF .generate() yields naturally on block.
  - D-11 (line 261): cooperative cancellation via threading.Event. put()
    becomes no-op when set; HF iterates a few more times then completes;
    results discarded. No exceptions raised through HF internals; CUDA
    state stays clean.
  - Import rule (line 671): may import only transformers (BaseStreamer),
    torch, queue, threading. May NOT import myvoice.services.sessions,
    myvoice.services.audio_coordinator, myvoice.services.qwen_tts_service,
    myvoice.observability, myvoice.models, or PyQt6.

Public surface (consumed by Stories 16.4-16.6):
  - CodecTokenStreamer: HF BaseStreamer subclass with bounded queue.
  - END_OF_STREAM: module-level singleton; decoder loop-exit signal.

Defaults: chunk_size=10, lookahead=5 (Story 20.4). The original
chunk_size=25 was inherited verbatim from the example in
01-streaming-tts-research.md:184 and never tuned; Story 20.1 SS5.2/5.3
swept {5, 10, 15, 25} on both utterance classes and found 10 to be the
optimum -- see the class docstring for the numbers. Story 16.7's
empirical-validation harness may revise via direct module-constant edit.
"""

import queue
import threading
from typing import Any, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer


# Module-level singleton sentinel (compared via `is`). Decoder worker
# (Story 16.4) treats `chunk is END_OF_STREAM` as the loop-exit signal.
END_OF_STREAM = object()


# Story 20.4 (Epic 20, Follow-up B) -- the committed geometry.
#
# chunk_size was 25 from Story 16.3 through Story 20.3, inherited verbatim
# from the 01-streaming-tts-research.md:184 example and never tuned. Story
# 20.1 SS5.2/SS5.3 swept {5, 10, 15, 25} with lookahead held at 5, on the
# RTX 5090, in the shipping tts_compile="auto" regime:
#
#   cs  window  audio/chunk  TTFA long  TTFA short  ratio  short first-emit
#    5      10       417 ms     951 ms      899 ms  0.760  threshold 5/5
#   10      15       833 ms     875 ms      921 ms  0.676  threshold 5/5   <- optimum
#   15      20     1,250 ms   1,172 ms    1,174 ms  0.677  threshold 5/5
#   25      30     2,083 ms   1,785 ms    1,651 ms  0.665  residual_flush 11/20
#
# The optimum is 10, not the smallest value: at chunk_size=5 each chunk
# carries 417 ms of audio, BELOW the consumer's 500 ms static watermark
# (audio_coordinator.py), so the consumer holds two chunks and hands back
# ~270 ms -- wiping out most of the producer-side gain. chunk_size >= 6
# keeps the watermark a no-op.
#
# chunk_size=10 also drops the first-emit threshold from 30 frames (2.5 s
# of audio at 12 Hz) to 15 (1.25 s), which is what moves SHORT utterances
# off the ``residual_flush`` dispatch path onto the ``threshold`` path.
#
# ANY change to these two constants must be threaded into
# ``torch_runtime.engage_compile_optimizations`` -- it derives D-25's
# ``decode_window_frames`` from them (it imports this module for exactly
# that reason), and the value is one of compile_cache's seven key
# dimensions, so a retune auto-invalidates the compile cache (D-24).
# Story 20.1 SS5.4 documents the trap: before Story 20.4 the compile path
# carried its own hard-coded 25/5 literals and the sole production call
# site passed neither, so ``decode_window_frames`` was pinned at 30
# regardless of the streamer's real geometry.
DEFAULT_CHUNK_SIZE = 10
DEFAULT_LOOKAHEAD = 5
DEFAULT_QUEUE_MAX_FACTOR = 4  # D-10: maxsize = factor * chunk_size


class CodecTokenStreamer(BaseStreamer):
    """HF BaseStreamer subclass: codec-token producer for the streaming
    decoder worker (Story 16.4).

    Buffers tokens delivered via put(), pushes fixed-size chunks of
    (chunk_size + lookahead) tokens onto a bounded queue, slides the
    buffer forward by chunk_size after each push (keeping the last
    `lookahead` tokens as the next chunk's left-context — overlap-add
    per architecture line 184), signals end-of-stream via the
    END_OF_STREAM sentinel on end(), and goes silent (no-op) while a
    threading.Event cancel hook is set.

    Forbidden by P-5 (architecture line 429): this class does NOT call
    into the session, registry, or audio coordinator. Composition with
    the decoder worker is the registry's job (Stories 16.5 and 16.6).

    **Story 16.8 deviation note.** The TRUE_STREAM dispatch path
    (``QwenTTSService._build_true_stream_talker``) does NOT call
    ``put()`` or ``end()`` on this streamer. The qwen-tts talker is
    multi-codebook (returns ``(batch, num_code_groups)`` per step), but
    HF ``GenerationMixin._sample``'s standard ``streamer.put(next_tokens)``
    callback only fires with the codec_head's main-codebook sample — the
    other codebooks live in ``Qwen3TTSTalkerOutputWithPast.hidden_states[1]``.
    Story 16.8's forward-hook captures multi-codebook ``codec_ids``
    directly from the talker's per-step output and pushes whole
    ``(N_steps, num_code_groups)`` tensors to ``self.queue`` directly,
    bypassing the int-buffer ``put()/end()`` chunking machinery here.

    On the TRUE_STREAM path, this class is effectively a queue-holder
    plus the shared ``_cancel_event``; ``put``, ``end``, ``_buffer``,
    and ``_extract_tokens`` remain live for any future HF-streamer
    consumer (e.g., a SENTENCE_STREAM-style adapter, or a future
    qwen-tts release that emits single-codebook tokens). The chunking
    arithmetic is duplicated in ``_build_true_stream_talker`` —
    intentional duplication: the two paths chunk different shapes
    (flat token list vs. per-step tensors) and conflating them would
    require a more invasive refactor of ``put()`` to accept tensor
    inputs. If a third consumer ever needs the same overlap-add
    chunking on per-step tensors, factor it out then.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        lookahead: int = DEFAULT_LOOKAHEAD,
        queue_max_factor: int = DEFAULT_QUEUE_MAX_FACTOR,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}"
            )
        if lookahead < 0:
            raise ValueError(
                f"lookahead must be >= 0, got {lookahead}"
            )
        if queue_max_factor <= 0:
            raise ValueError(
                f"queue_max_factor must be positive, got "
                f"{queue_max_factor}"
            )

        self.chunk_size = chunk_size
        self.lookahead = lookahead
        self._chunk_with_lookahead = chunk_size + lookahead
        # D-10: bounded queue. maxsize = factor * chunk_size keeps the
        # backpressure characteristic stable across chunk-size choices.
        self.queue: queue.Queue = queue.Queue(
            maxsize=queue_max_factor * chunk_size
        )
        # Architecture line 420: streamer "owns" the cancel event but
        # accepts injection so Story 16.5 can wire the same event to the
        # registry's session.cancel() and the decoder worker's loop.
        self._cancel_event = (
            cancel_event if cancel_event is not None else threading.Event()
        )
        self._buffer: List[Any] = []

    # P-5 / D-10 / D-11
    def put(self, value: Any) -> None:
        """Buffer incoming token(s); push a chunk when ready; backpressure.

        HF .generate() calls this once per token (or per token batch)
        during streaming generation. The streamer accumulates tokens
        until the buffer reaches chunk_size + lookahead, then pushes the
        next chunk onto the bounded queue (D-10) and slides the buffer
        forward by chunk_size, keeping the last `lookahead` tokens as
        left-context for the next chunk's overlap-add decode.

        Cancellation (D-11): when self._cancel_event is set, this method
        returns immediately as a no-op. HF .generate() will iterate a
        few more times producing tokens we drop, then complete cleanly.
        No exception is raised through HF internals; CUDA state stays
        clean. The decoder worker (Story 16.4) is responsible for
        draining any chunks the streamer pushed before cancel landed.

        Backpressure (D-10): queue.put() blocks when the queue is full.
        HF .generate() yields the GPU naturally between iterations, so
        a blocked streamer throttles the talker without explicit
        cooperation.
        """
        if self._cancel_event.is_set():
            return

        tokens = self._extract_tokens(value)
        self._buffer.extend(tokens)

        # Push every chunk that became ready during this put() call.
        # Some HF streamers deliver tokens in batches large enough to
        # release multiple chunks in one call.
        while len(self._buffer) >= self._chunk_with_lookahead:
            # Re-check between chunks so a cancel that fires mid-batch
            # stops pushing the rest. Without this, a multi-chunk batch
            # combined with a decoder that exits on cancel can deadlock
            # the producer on a full queue with no consumer.
            if self._cancel_event.is_set():
                return
            chunk = self._buffer[: self._chunk_with_lookahead]
            # Blocks on full queue (backpressure).
            self.queue.put(chunk)
            # Slide forward by chunk_size; keep the lookahead tail as
            # the next chunk's left-context (overlap-add per arch:184).
            del self._buffer[: self.chunk_size]

    # P-5
    def end(self) -> None:
        """Final flush: push residual buffer (if any), then END_OF_STREAM.

        Called by HF .generate() when token generation completes. The
        decoder worker (Story 16.4) treats END_OF_STREAM as the
        loop-exit signal and posts registry.post_mutation('finalize',
        session_id) before exiting.
        """
        if self._buffer:
            self.queue.put(list(self._buffer))
            self._buffer.clear()
        self.queue.put(END_OF_STREAM)

    # P-5 (MyVoice-specific extension; not in HF BaseStreamer contract)
    def reset(self) -> None:
        """Clear all internal state. MUST NOT be called mid-generation.

        Caller contract (architecture line 427): the registry/dispatch
        layer is responsible for ensuring no in-flight generation is
        using the streamer when reset() runs. Story 16.5's cancellation
        chain plus Story 16.6's dispatch wiring is what guarantees this.

        Drains the queue, clears the token buffer, and clears the cancel
        event. After reset(), the streamer is functionally indistinguish-
        able from a freshly-constructed instance with the same chunk
        sizing.
        """
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self._buffer.clear()
        self._cancel_event.clear()

    @staticmethod
    def _extract_tokens(value: Any) -> List[Any]:
        """Convert HF-streamer put() value into a list of token ids.

        HF .generate() typically delivers a torch.Tensor at each step,
        often shape [batch=1, seq_len] or [seq_len]. We unwrap the batch
        dimension if present and convert to a Python list. Lists and
        tuples pass through directly. Scalars wrap into single-element
        lists (defensive).

        Raises ValueError on a >1-D tensor whose first dim is not 1 —
        HF streaming contracts batch=1, and a batch>1 tensor would
        otherwise produce nested-list "tokens" that corrupt the buffer
        silently.
        """
        if isinstance(value, torch.Tensor):
            if value.dim() > 1:
                if value.size(0) != 1:
                    raise ValueError(
                        f"CodecTokenStreamer expects batch=1 tensors; "
                        f"got shape={tuple(value.shape)}. HF streaming "
                        f"contracts batch=1 per put() call."
                    )
                value = value.squeeze(0)
            return value.tolist() if value.dim() > 0 else [value.item()]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
