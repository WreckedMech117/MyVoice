"""StreamingDecoderWorker — single-thread-per-session decoder for TRUE_STREAM TTS.

Story 16.4 — Phase ⊥ of D-20 (architecture-optimization-pass.md).

Architecture references:
  - P-6 (lines 431-441): four-step decoder contract — pull token chunks
    from streamer.queue.get(), decode with overlap-add (chunk+lookahead,
    trim trailing lookahead-portion), post each PCM segment via
    post_mutation('append_chunk', session_id, pcm), post finalize on
    END_OF_STREAM, drain-and-post-cancel on _cancel_event.is_set().
    Forbidden: write to disk, emit signals, touch audio devices.
  - P-7 (lines 443-451): cancellation propagation — drain on cancel;
    no exception raised through HF / CUDA back into the streamer.
  - D-11 (line 261): cooperative cancel via threading.Event shared with
    the streamer (single event; both producer and consumer observe).
  - P-9 (lines 463-476): per-chunk decode latency metric; the worker
    is the architecturally-named owner of decode_chunk_latency_ms.
  - Import rule (line 674): may import tts_streaming.codec_token_streamer,
    qwen_tts internals (NOT exercised here — see Story 16.6),
    numpy, threading, observability.metrics. May NOT import
    myvoice.services.sessions (posts via callback supplied at init).
    Review-fix deviation: also imports `queue` (stdlib) so drain can
    catch `queue.Empty` narrowly instead of bare-Except'ing — same
    stdlib tier as `threading` / `time`.

Public surface (consumed by Stories 16.5, 16.6):
  - StreamingDecoderWorker: thread-owning decoder worker class.

Overlap-add interpretation: future-lookahead. Each non-final chunk
decodes (chunk_size + lookahead) tokens; the worker posts only the
leading chunk_size's worth of PCM samples (the trailing lookahead's
worth is discarded — the next chunk's leading lookahead tokens
re-establish the same audio with better priming). The final residual
chunk (length < chunk_size + lookahead) is posted whole — there is
no following chunk to overlap with.
"""

import queue
import threading
import time
from typing import Any, Callable, Optional, Protocol

import numpy as np

from myvoice.observability import metrics
from myvoice.services.tts_streaming.codec_token_streamer import (
    CodecTokenStreamer,
    END_OF_STREAM,
)


class _PostMutationCallable(Protocol):
    """Structural type for the post_mutation callback.

    Matches registry.post_mutation's bound-method shape:
        post_mutation('append_chunk', session_id, pcm)
        post_mutation('finalize', session_id)
        post_mutation('cancel', session_id)
    """

    def __call__(self, method_name: str, session_id: str, *args: Any) -> None: ...


class StreamingDecoderWorker:
    """One decoder thread per active streaming session (P-6).

    Pulls token chunks from the streamer's bounded queue, decodes via
    the injected `decode_fn`, applies future-lookahead overlap-add trim,
    and posts each PCM segment + the eventual finalize/cancel mutation
    via the injected `post_mutation` callable. Architecture line 674
    forbids importing myvoice.services.sessions; the post_mutation
    callable is how the worker reaches the registry without the import.

    Lifecycle: caller constructs, then calls .start() to spawn the
    thread. Caller is responsible for eventually calling .join() (or
    the worker exits cleanly on END_OF_STREAM / cancel / decode error).
    """

    def __init__(
        self,
        streamer: CodecTokenStreamer,
        decode_fn: Callable[[list[Any]], np.ndarray],
        post_mutation: _PostMutationCallable,
        session_id: str,
        *,
        model_type: str = "qwen3_tts",
        hardware: str = "gpu",
    ) -> None:
        # Snapshots the streamer's chunk_size + lookahead at construction;
        # do not mutate the streamer's geometry while a worker is bound to it.
        if streamer is None:
            raise ValueError("streamer must not be None")
        if decode_fn is None:
            raise ValueError("decode_fn must not be None")
        if post_mutation is None:
            raise ValueError("post_mutation must not be None")
        if not session_id:
            raise ValueError("session_id must be a non-empty string")

        self._streamer = streamer
        self._decode_fn = decode_fn
        self._post_mutation = post_mutation
        self._session_id = session_id
        self._model_type = model_type
        self._hardware = hardware
        self._chunk_size = streamer.chunk_size
        self._lookahead = streamer.lookahead
        # Same threading.Event the streamer was constructed with — Story 16.5
        # wires registry.cancel() to flip this event; both producer and
        # consumer observe the single flip.
        self._cancel_event = streamer._cancel_event

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"StreamingDecoder-{session_id[:8]}",
        )

    def start(self) -> None:
        """Spawn the worker thread. Calling twice raises (Thread.start
        semantics). Caller is responsible for join() or the worker
        exits on END_OF_STREAM / cancel / decode error.
        """
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ----- internal: thread loop ----------------------------------- #

    def _run(self) -> None:
        """Decoder loop body. P-6 four-step contract + P-7 cancel."""
        while True:
            # P-6 step 5 + P-7: cancel takes priority over decode.
            if self._cancel_event.is_set():
                self._drain_and_post_cancel()
                return

            chunk = self._streamer.queue.get()

            # P-6 step 4: END_OF_STREAM → finalize → exit.
            if chunk is END_OF_STREAM:
                self._post_terminal("finalize")
                return

            # P-6 steps 2 + 3: decode + post; any decode/post exception
            # → record decode_error metric → cancel + drain + exit.
            try:
                self._decode_and_post(chunk)
            except Exception as exc:  # noqa: BLE001 — see AC #6
                # Numeric value (1.0) so the real metrics.record (which
                # validates value to int|float at metrics.py:95-98) does
                # not raise TypeError from inside this except block and
                # kill the thread before the cancel post fires.
                metrics.record(
                    "decode_error",
                    1.0,
                    session_id=self._session_id,
                    model_type=self._model_type,
                    hardware=self._hardware,
                    error_repr=repr(exc),
                )
                self._drain_and_post_cancel()
                return

    def _decode_and_post(self, chunk: list[Any]) -> None:
        """Decode one chunk, trim trailing lookahead-tokens'-worth of
        PCM samples (future-lookahead overlap-add), post via callable.
        Final residual chunk (len < chunk_size + lookahead) is posted
        whole — no trim because no following chunk to overlap.
        """
        t_start = time.perf_counter()
        pcm_full = self._decode_fn(chunk)
        t_end = time.perf_counter()

        metrics.record(
            "decode_chunk_latency_ms",
            (t_end - t_start) * 1000.0,
            session_id=self._session_id,
            model_type=self._model_type,
            hardware=self._hardware,
        )

        if len(chunk) >= self._chunk_size + self._lookahead and self._lookahead > 0:
            samples_per_token = len(pcm_full) / len(chunk)
            trim_samples = int(round(self._lookahead * samples_per_token))
            pcm_segment = pcm_full[: len(pcm_full) - trim_samples]
        else:
            # Final residual chunk OR lookahead == 0 → no trim.
            pcm_segment = pcm_full

        self._post_mutation("append_chunk", self._session_id, pcm_segment)

    def _post_terminal(self, method_name: str) -> None:
        """Post a single terminal mutation — 'finalize' or 'cancel'.

        A raising post_mutation must not silently kill the worker thread
        mid-handoff (P-7 spirit). Catches and records as a metric so a
        failing registry handoff is visible without violating the
        no-exception-escapes-the-worker invariant.
        """
        try:
            self._post_mutation(method_name, self._session_id)
        except Exception as exc:  # noqa: BLE001
            metrics.record(
                "post_mutation_error",
                1.0,
                session_id=self._session_id,
                model_type=self._model_type,
                hardware=self._hardware,
                method_name=method_name,
                error_repr=repr(exc),
            )

    def _drain_and_post_cancel(self) -> None:
        """P-7: drain remaining queue (drop chunks, drop sentinel) and
        post a single ('cancel', session_id). queue.Empty is the expected
        drain-complete signal; any other exception is recorded as a
        metric but does NOT prevent the cancel post (P-7 invariant:
        cancel must propagate to the registry).
        """
        try:
            while True:
                try:
                    self._streamer.queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as exc:  # noqa: BLE001 — keep cancel path resilient
            metrics.record(
                "drain_error",
                1.0,
                session_id=self._session_id,
                model_type=self._model_type,
                hardware=self._hardware,
                error_repr=repr(exc),
            )
        self._post_terminal("cancel")
