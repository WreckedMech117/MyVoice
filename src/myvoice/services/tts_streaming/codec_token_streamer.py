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

Defaults: chunk_size=25, lookahead=5. Story 20.4 attempted a retune to
10 -- measured as the throughput/latency optimum by Story 20.1 SS5.2/5.3 --
and REVERTED it after the NFR3 perceptual gate failed twice. See the
constants below for the full record. Story 16.7's empirical-validation
harness may revise via direct module-constant edit.

Story 20.6 retires the lookahead on the STATE-CARRYING decode path only:
``DEFAULT_LOOKAHEAD`` stays 5 (it is the stateless fallback's only seam
handling) and the live value is resolved per stream by
:func:`effective_lookahead` / :meth:`CodecTokenStreamer.apply_codec_state_geometry`.
"""

import queue
import threading
from typing import Any, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer


# Module-level singleton sentinel (compared via `is`). Decoder worker
# (Story 16.4) treats `chunk is END_OF_STREAM` as the loop-exit signal.
END_OF_STREAM = object()


# THE COMMITTED GEOMETRY -- 10, and the record of how it got here.
#
# Story 20.4 tried chunk_size = 10 and REVERTED it. Story 20.8 re-measured
# and committed it. Both of those are load-bearing and neither supersedes
# the other by being newer, so the whole chain is recorded here rather than
# only in the story files.
#
# WHY 20.4 REVERTED IT. Not on latency -- on the ear. Every chunk boundary
# is a seam and 10 has 2.5x as many as 25. Commander flagged audible
# defects on 1 of 7 fixtures in round 1 and 3 of 7 in round 2, and
# preferred 25 on every utterance where the two differed. A round-3
# audition then isolated the variables and showed the SEAM FIX in
# ``streaming_decoder.py`` was good on its own at 25, so only the fix
# shipped. Story 20.4 SS17's closing line is the one to remember: *the
# sweep optimised perceived latency and never asked the ear.*
#
# WHAT CHANGED UNDERNEATH IT. Two things, in this order, and neither was
# a chunk-size change:
#
#   * Story 20.5 -- codec state caching. The seam harm that killed cs10 was
#     a cold-state residual at every boundary, MASKED by the Story 20.4
#     blend. Carrying the codec's real state removes it at the cause: head
#     NRMSE 0.406 -> 0.0078, lag jitter 0 samples on every seam, edge loss
#     555 -> 0, so ``decode(N) == 1920*N`` exactly. Story 20.4 SS17 named
#     precisely this as its reopening condition.
#   * Story 20.6 -- the lookahead retirement. ``chunk_size = N`` now means
#     first emit at N frames, not N + 5. That makes the geometry lever
#     bigger than the old curve implied and it is why the numbers below do
#     not match Story 20.1's.
#
# THE RE-BASELINE (Story 20.8 Phase 1). Story 20.1's curve was measured
# before both of those, and with ``decode_window_frames`` pinned at 30
# regardless of geometry, so it was re-measured from scratch: one sitting,
# one machine, cs25 captured as the control TWICE (first and last, drift
# +14.0 ms), n = 10 warm runs per point, RTX 5090:
#
#   cs   seams  first chunk  TTFA long  TTFA short  ratio  decoder work
#   25       9     1,977 ms   1,167 ms    1,176 ms  0.539        366 ms
#   15      16     1,177 ms     741 ms      738 ms  0.547        577 ms
#   10      24       777 ms     528 ms      516 ms  0.558        840 ms
#    7      34       537 ms     392 ms      393 ms  0.567      1,135 ms
#
# WHY 10 AND NOT 7. Three non-perceptual reasons (Story 20.8 SS8.1):
#   1. the marginal rate collapses -- each step buys 60.9, then 26.6, then
#      13.0 ms per added seam. cs10 already takes 82.5 % of the whole
#      available win;
#   2. the WATERMARK FLOOR is 7 and cs7 clears it by 0.46 of a frame.
#      cs10 clears it by 3.5 frames. That floor MOVED during Story 20.8
#      itself (Story 20.1 SS5.4 said 6; the exact solve on current code is
#      ``N*1920 - 555 >= 12000`` -> N >= 6.54 -> 7), and it is a function
#      of three constants that have all moved within this epic;
#   3. per-chunk decode time is FLAT in chunk size, so decoder work scales
#      with chunk count -- cs7 is 35 % more than cs10, and the sub-16 GiB
#      tier where the OFR-E ratio has least room is still unmeasured.
#
# THE GATE THAT IS NOT YET CLOSED. This constant ships subject to an NFR3
# audition whose falsifiable prediction is recorded in
# ``20-8-chunk-size-reopen-evidence.md`` SS8.4 BEFORE the round. If cs10
# flags a blocking seam defect again, the mechanism argument above is
# wrong -- state caching did not remove what actually made cs10 worse --
# and the geometry question closes for good rather than retuning to 15.
# Story 20.4 shipped a retune this far and then reverted it; that is the
# precedent, not a reason to assume this one lands.
#
# ANY change to these two constants must be threaded into
# ``torch_runtime.engage_compile_optimizations`` -- it derives D-25's
# ``decode_window_frames`` from them via ``resolve_streamer_geometry()``,
# and the value is one of compile_cache's seven key dimensions, so a retune
# auto-invalidates the compile cache (D-24) and costs exactly one cold
# compile on first launch (measured at +19.2 s, Story 20.8 SS3.5).
# Story 20.1 SS5.4 documents the trap that made this necessary: before
# Story 20.4 the compile path carried its own hard-coded 25/5 literals and
# the sole production call site passed neither, so ``decode_window_frames``
# was pinned at 30 regardless of the streamer's real geometry. Story 20.6
# verified the threading carries a change in BOTH directions.
DEFAULT_CHUNK_SIZE = 10
DEFAULT_LOOKAHEAD = 5
DEFAULT_QUEUE_MAX_FACTOR = 4  # D-10: maxsize = factor * chunk_size

# Story 20.6 — the lookahead a STATE-CARRYING stream runs with.
#
# ``DEFAULT_LOOKAHEAD`` above stays 5 and that is not an oversight: it is the
# lookahead of the STATELESS decode path, which is still reachable two ways
# (the ``MYVOICE_CODEC_STATE_CACHE`` kill switch, and any build-time refusal
# by ``codec_state_cache.build_stateful_decode_fn``). On that path chunks are
# still independent renderings of overlapping token spans, so the lookahead,
# the post-decode trim and the Story 20.4 seam blend are the ONLY seam
# handling it has. Retiring the constant globally would strip all three from
# the path a user reaches precisely when something else has already failed,
# and would reintroduce the artefact Epic 20 spent six audition rounds
# eliminating.
#
# So retirement is CONDITIONAL, resolved by :func:`effective_lookahead` from
# the decode_fn's own ``carries_codec_state`` declaration — the same
# producer-declares / consumer-acts shape Story 20.5 used for the consumer
# crossfade (``qwen_tts_service._progressive_stream_continuous``).
RETIRED_LOOKAHEAD = 0


def effective_lookahead(
    carries_codec_state: bool, lookahead: Optional[int] = None
) -> int:
    """Return the lookahead a stream should actually run with (Story 20.6).

    ``carries_codec_state`` is the decode_fn's own declaration (see
    ``codec_state_cache.StatefulCodecDecoder.carries_codec_state``). When it
    is true, consecutive chunks are continuous *by construction* — the codec
    resumes each chunk from the previous chunk's real state — so the five
    future-lookahead frames Story 16.4 introduced re-establish, at a cost of
    five talker steps of TTFA and a two-pass decode, priming the carried
    state already provides exactly.

    When it is false the pre-20.6 geometry is returned unchanged.

    This is a **pure function of the flag**, deliberately: the geometry can
    never end up half-retired, no matter how often the kill switch is
    flipped, because there is no accumulated state to get out of step.

    Args:
        carries_codec_state: the decode_fn's declaration.
        lookahead: the stream's configured lookahead. ``None`` resolves
            :data:`DEFAULT_LOOKAHEAD`.
    """
    base = DEFAULT_LOOKAHEAD if lookahead is None else lookahead
    return RETIRED_LOOKAHEAD if carries_codec_state else base


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

    Story 20.6: with ``lookahead == 0`` (what
    :meth:`apply_codec_state_geometry` sets on the state-carrying path) the
    chunks are exactly ``chunk_size`` tokens and do not overlap at all, so
    the slide keeps nothing back and the first emit lands five talker steps
    earlier. The arithmetic below already degenerates correctly; no branch
    is needed.

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
        # Story 20.6 — the lookahead this instance was CONSTRUCTED with, kept
        # so ``apply_codec_state_geometry`` is a pure function of its argument
        # rather than a one-way door. Retiring and then un-retiring returns
        # the exact pre-20.6 geometry, which is what makes a runtime kill-
        # switch flip incapable of leaving a half-retired streamer behind.
        self._configured_lookahead = lookahead
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

    # Story 20.6 — the CONSUMER half of producer-declares / consumer-acts.
    def apply_codec_state_geometry(self, carries_codec_state: bool) -> int:
        """Set this streamer's lookahead from the decode_fn's declaration.

        The dispatch layer builds the decode_fn, reads
        ``decode_fn.carries_codec_state`` off it, and hands the answer here.
        With carried state the lookahead retires to 0 — chunks become
        non-overlapping, the streamer's first emit waits for ``chunk_size``
        frames instead of ``chunk_size + lookahead``, and the decoder worker's
        trim and seam blend fall away *by construction* (its ``is_full_window``
        predicate requires ``lookahead > 0``). Without it, the pre-20.6
        geometry is restored exactly.

        Reversible and idempotent: the result is a pure function of the
        argument and of the lookahead this streamer was constructed with, so
        a kill-switch flip between generations cannot leave a half-retired
        geometry behind.

        MUST NOT be called mid-generation — the same caller contract
        :meth:`reset` carries. Changing ``_chunk_with_lookahead`` while
        ``put()`` is sliding the buffer, or while
        ``qwen_tts_service._build_true_stream_talker``'s forward-hook is
        chunking against a snapshot of it, would emit one malformed chunk.
        Guarded rather than documented, because "the geometry changed
        mid-stream" is exactly the class of defect that is inaudible as a bug
        and audible only as "the codec got worse".

        Returns:
            The lookahead now in effect.
        """
        if self._buffer or not self.queue.empty():
            raise RuntimeError(
                "apply_codec_state_geometry() called on a streamer with "
                "buffered tokens or queued chunks; the geometry may only be "
                "set between generations (same contract as reset())."
            )
        self.lookahead = effective_lookahead(
            carries_codec_state, self._configured_lookahead
        )
        self._chunk_with_lookahead = self.chunk_size + self.lookahead
        return self.lookahead

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
