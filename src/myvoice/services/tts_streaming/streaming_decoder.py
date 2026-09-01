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
leading chunk_size's worth of PCM samples. The final residual
chunk (length < chunk_size + lookahead) is posted whole — there is
no following chunk to overlap with.

Story 20.4 — the overlap is now actually ADDED
----------------------------------------------
Through Story 20.3 the "overlap-add" was an overlap-*discard*: the
trailing lookahead's worth of PCM was thrown away and the next chunk's
independent decode of the same tokens was butt-spliced on. Story 20.4's
failed NFR3 audition traced two distinct defects to that splice, both
measured against captured pcm_full (see
``_bmad-output/implementation-artifacts/20-4-seam-analysis.py``):

1. **A splice-alignment bug.** ``decode(N frames)`` returns exactly
   ``1920 * N - 555`` samples — 1920 samples per codec frame plus a
   FIXED 555-sample convolution edge loss. The trim was computed as
   ``round(lookahead * len(pcm_full) / len(chunk))``, which treats that
   fixed loss as if it were proportional, so every posted chunk fell
   short by ``555 * chunk_size / (chunk_size + lookahead)`` samples:
   370 at chunk_size=10, 463 at 25. Cross-correlating consecutive
   pcm_full arrays puts the true splice point at exactly
   ``chunk_size * 1920``, so those samples were real speech — measured
   RMS at or above the utterance's own — silently deleted at every
   chunk boundary. This shipped from Story 16.4 onward and is why the
   round-1 audition flagged long-form seams at chunk_size=25 too.

2. **A codec-state mismatch.** The two decodes of the shared lookahead
   frames differ by ~35 % NRMSE (correlation ~0.93) because each decode
   starts from a cold codec state. Alignment cannot fix that; correcting
   the trim ALONE measurably makes the click worse (seam step 8.5x ->
   18.0x the non-seam baseline at chunk_size=25), because it butts two
   genuinely different waveforms together at the same instant.

The fix pairs them. The tail this module used to discard is exactly the
audio the next chunk re-decodes at its head, so it is now retained and
cross-faded into that head over ``_OVERLAP_ADD_SAMPLES``. Unlike widening
the CONSUMER crossfade in ``streaming_chunk_buffer.py``, this consumes no
audio and no duration: both sides of the blend are the same moment in
time. Measured effect, seam step as a multiple of the non-seam baseline:

    chunk_size=25   shipped 8.46x  ->  aligned+OLA 1.25x
    chunk_size=10   shipped 13.06x ->  aligned+OLA 0.85x

Both land at the baseline, i.e. a seam becomes statistically
indistinguishable from any other point in the audio.

This does NOT make the codec carry state across chunks — the reference
implementations do that and we still do not (see the Story 20.4 evidence
file, §11). It masks the consequence at the boundary rather than removing
the cause.
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


# --------------------------------------------------------------------------- #
# Story 20.4 — the codec's measured output geometry.
#
# ``speech_tokenizer.decode`` on N codec frames returns exactly
# ``_CODEC_SAMPLES_PER_FRAME * N - _CODEC_EDGE_LOSS_SAMPLES`` samples. Both
# numbers were MEASURED, not assumed: solved from posted chunk lengths at two
# chunk sizes and then cross-checked against 14 independent residual-chunk
# lengths, every one of which is integral under the model
# (``20-4-seam-capture.py`` / ``20-4-seam-analysis.py``).
#
# Note 1920 samples/frame = 12.5 Hz, NOT the 12 Hz this codebase's prose has
# assumed since Story 16.3. Nothing depended on the 12 Hz figure except
# human-facing "seconds of audio per chunk" arithmetic in comments and
# evidence files, which is ~4 % optimistic wherever it appears.
#
# These are verified at runtime rather than trusted: ``_decode_and_post``
# checks the identity on every chunk and falls back to the pre-20.4
# proportional trim (with no overlap-add) if it ever fails, so a codec or
# pin change degrades to the old behaviour loudly instead of producing
# mis-spliced audio silently.
_CODEC_SAMPLES_PER_FRAME = 1920
_CODEC_EDGE_LOSS_SAMPLES = 555

# Width of the decoder-side overlap-add, in samples (~42.7 ms at 24 kHz).
#
# Chosen from the offline sweep in ``20-4-seam-fix-sweep.py``, which
# re-stitched captured pcm_full at eight widths and scored each seam
# against non-seam positions in the same audio:
#
#   width   seam step (x baseline)      excess spectral jump
#            cs25      cs10              cs25      cs10
#      0    18.04     12.32             +4.43 dB  +2.04 dB   <- alignment only
#     64     1.31      1.00             +4.36     +1.93
#    256     1.06      0.75             +4.90     +2.36
#   1024     1.25      0.85             +0.71     +1.99      <- chosen
#   2048     1.46      0.83             +1.27     +1.63
#   4096     1.50      0.96             +1.59     +1.23
#   9045     1.46      0.80             +2.21     +1.81      <- the whole budget
#
# The step metric plateaus at the baseline by 64-256; the spectral metric
# keeps improving to ~1024-4096. 1024 is the smallest width that is at the
# plateau on BOTH, which matters because the blend is not free in every
# sense: inside the window the signal is the average of two decodes, which
# mildly softens fine structure. That cost scales with the fraction of the
# stream inside a blend — 1024/(10*1920) = 5.3 % at chunk_size=10, 2.1 % at
# 25 — so the smallest sufficient width is the right pick, not the largest.
#
# The hard ceiling is the audio the two chunks actually share:
# ``lookahead * 1920 - 555`` = 9,045 samples (377 ms) at lookahead=5.
# ``_decode_and_post`` clamps to it; a larger constant cannot over-read.
_OVERLAP_ADD_SAMPLES = 1024


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
        # Story 20.1 (TTFA spike) — one-shot flag for the segment-3
        # boundary metric below. Instance state, not module state, so
        # concurrent workers (one per session) never share it.
        self._ttfa_first_decode_recorded = False
        # Story 20.4 overlap-add state. ``_pending_overlap`` is the tail of
        # the previous chunk's pcm_full BEYOND its splice point — the audio
        # the next chunk is about to re-decode at its head. Per-instance, so
        # concurrent per-session workers never share it.
        self._pending_overlap: Optional[np.ndarray] = None
        # One-shot latch: the codec output-length identity failed, so this
        # session has fallen back to the pre-20.4 proportional trim.
        self._geometry_fallback = False
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
        # Track whether any append_chunk has been posted so the END_OF_STREAM
        # branch can choose finalize vs. try_set_error + discard. The talker
        # exception path in qwen_tts_service.py:4006-4014 pushes
        # END_OF_STREAM after raising — with zero chunks accumulated — and a
        # blind finalize post here raises ValueError inside the Qt slot
        # (session_registry.py:433 -> generation_session.py:163), which the
        # global exception handler surfaces as a user-visible dialog even
        # though the dispatch's fallback chain produces audio.
        appended = 0
        while True:
            # P-6 step 5 + P-7: cancel takes priority over decode.
            if self._cancel_event.is_set():
                self._drain_and_post_cancel()
                return

            chunk = self._streamer.queue.get()

            # P-6 step 4: END_OF_STREAM → finalize → exit. When no chunks
            # were appended (talker raised mid-stream and pushed
            # END_OF_STREAM from the exception path), finalize would raise
            # in the registry slot — route to try_set_error + discard so the
            # session settles into a terminal state cleanly and the
            # dispatch's own cleanup races harmlessly.
            if chunk is END_OF_STREAM:
                if appended > 0:
                    self._post_terminal("finalize")
                else:
                    self._post_terminal("try_set_error")
                    self._post_terminal("discard")
                return

            # P-6 steps 2 + 3: decode + post; any decode/post exception
            # → record decode_error metric → cancel + drain + exit.
            try:
                self._decode_and_post(chunk)
                appended += 1
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

        # Story 20.1 (TTFA spike) segment boundary #3 — the first token
        # chunk has been turned into PCM. Closes segment 3 (first decode)
        # and opens segment 4 (consumer-side cushion). Wall-clock ms to
        # join against the other TTFA boundaries; ``decode_chunk_latency_ms``
        # below stays the per-chunk elapsed measure and is unchanged.
        # One-shot: fires only for the first decoded chunk of a session.
        if not self._ttfa_first_decode_recorded:
            self._ttfa_first_decode_recorded = True
            metrics.record(
                "ttfa_first_decode_complete_ms",
                time.time() * 1000.0,
                session_id=self._session_id,
                chunk_index=0,
                pcm_samples=int(len(pcm_full)),
            )

        metrics.record(
            "decode_chunk_latency_ms",
            (t_end - t_start) * 1000.0,
            session_id=self._session_id,
            model_type=self._model_type,
            hardware=self._hardware,
        )

        n_frames = len(chunk)
        is_full_window = (
            n_frames >= self._chunk_size + self._lookahead
            and self._lookahead > 0
        )

        # Story 20.4 — exact splice, guarded by the codec's own arithmetic.
        # The identity is checked on EVERY chunk (residuals included, where
        # it also holds) rather than trusted, so a codec/pin change cannot
        # silently produce mis-spliced audio.
        expected = (
            _CODEC_SAMPLES_PER_FRAME * n_frames - _CODEC_EDGE_LOSS_SAMPLES
        )
        geometry_ok = len(pcm_full) == expected and not self._geometry_fallback
        if not geometry_ok and not self._geometry_fallback:
            self._geometry_fallback = True
            metrics.record(
                "decode_geometry_unverified",
                1.0,
                session_id=self._session_id,
                model_type=self._model_type,
                hardware=self._hardware,
                frames=int(n_frames),
                pcm_samples=int(len(pcm_full)),
                expected_samples=int(expected),
            )

        next_overlap: Optional[np.ndarray] = None
        if geometry_ok and is_full_window:
            # The stream advances by exactly chunk_size frames of audio.
            splice = self._chunk_size * _CODEC_SAMPLES_PER_FRAME
            if splice >= len(pcm_full):
                # Defensive: cannot happen while the identity holds and
                # lookahead > 0, since the remainder is lookahead*1920-555.
                pcm_segment = pcm_full
            else:
                pcm_segment = pcm_full[:splice]
                # Retain, do not discard: this is the audio the NEXT chunk
                # re-decodes at its head, and it is what the overlap-add
                # blends against.
                keep = min(_OVERLAP_ADD_SAMPLES, len(pcm_full) - splice)
                next_overlap = np.asarray(
                    pcm_full[splice: splice + keep], dtype=np.float32
                )
        elif is_full_window:
            # Pre-20.4 proportional trim. Retained verbatim as the fallback
            # for an unrecognised codec geometry — and it is also the path
            # every synthetic test decode_fn takes, since a fake that
            # returns one sample per token does not satisfy the identity.
            samples_per_token = len(pcm_full) / n_frames
            trim_samples = int(round(self._lookahead * samples_per_token))
            pcm_segment = pcm_full[: len(pcm_full) - trim_samples]
        else:
            # Final residual chunk OR lookahead == 0 → no trim.
            pcm_segment = pcm_full

        pcm_segment = self._apply_overlap_add(pcm_segment)
        self._pending_overlap = next_overlap

        self._post_mutation("append_chunk", self._session_id, pcm_segment)

    def _apply_overlap_add(self, pcm_segment: np.ndarray) -> np.ndarray:
        """Cross-fade the previous chunk's retained tail into this head.

        Both sides cover the SAME moment in time — the previous chunk
        decoded it with future context, this chunk decoded it cold — so the
        blend costs no audio and no duration; it only chooses how to
        reconcile two renditions of one instant.

        A linear ramp is correct here rather than an equal-power one: the
        two decodes correlate at ~0.93 (measured), so they add coherently
        and a linear ramp preserves amplitude. An equal-power ramp would
        bulge the level mid-blend on signals this correlated.
        """
        pending = self._pending_overlap
        if pending is None or pending.size == 0 or pcm_segment.size == 0:
            return pcm_segment
        m = int(min(pending.size, pcm_segment.size))
        if m <= 0:
            return pcm_segment
        ramp = np.linspace(0.0, 1.0, m, dtype=np.float32)
        head = (
            np.asarray(pcm_segment[:m], dtype=np.float32) * ramp
            + pending[:m] * (1.0 - ramp)
        )
        return np.concatenate(
            [head, np.asarray(pcm_segment[m:], dtype=np.float32)]
        )

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
