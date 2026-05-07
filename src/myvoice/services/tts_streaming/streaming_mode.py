"""StreamingMode enum + hardware probe + override-aware resolver.

Story 16.2 — Phase ⊥ of D-20 (architecture-optimization-pass.md).

Architecture references:
  - D-9 (line 257): "At startup, probe torch.cuda.is_available(). If true,
    default streaming_mode = TRUE_STREAM. If false, default
    streaming_mode = SENTENCE_STREAM. User can override in settings."
  - NFR12 (line 75): CPU-only support — streaming default must NOT regress
    latency for CPU-only users; the False branch returning SENTENCE_STREAM
    protects this requirement.
  - Import rule (lines 669-674): this module may import only torch (for
    cuda.is_available probe) + stdlib. May NOT import myvoice.* peers.

Public surface (stable for Stories 16.3-16.6):
  - StreamingMode: three-member enum (BATCH, SENTENCE_STREAM, TRUE_STREAM)
  - default_streaming_mode_for_hardware() -> StreamingMode: pure probe
  - effective_streaming_mode(override) -> StreamingMode: override-aware

Module boundary: signal-free, side-effect-free, no metrics emission.
The streaming_mode metric (P-9, line 469) fires from Story 16.6's
dispatch site, not from this module's pure decision functions.
"""

from enum import Enum
from typing import Optional


class StreamingMode(Enum):
    """Three-mode dispatch enum (D-9, FR2/FR3/NFR7/NFR12)."""

    BATCH = "batch"
    SENTENCE_STREAM = "sentence_stream"
    TRUE_STREAM = "true_stream"


def default_streaming_mode_for_hardware() -> StreamingMode:
    """Hardware probe per D-9.

    Lazy-imports torch and reads torch.cuda.is_available() *by attribute
    access* (not by binding a function reference at import time) so
    monkeypatch.setattr("torch.cuda.is_available", ...) is honored in
    tests without import-order gymnastics.

    Returns TRUE_STREAM when CUDA is reported available; SENTENCE_STREAM
    otherwise (NFR12 protection — CPU-only hosts stay on the V2 baseline
    sentence-stream behavior and avoid the latency regression that an
    unconditional default flip would introduce).

    Returning BATCH is intentionally NOT a default branch — BATCH is the
    last-resort fallback in Story 16.6's three-mode dispatch chain.
    """
    import torch  # lazy: see docstring rationale
    if torch.cuda.is_available():
        return StreamingMode.TRUE_STREAM
    return StreamingMode.SENTENCE_STREAM


def effective_streaming_mode(
    override: Optional[StreamingMode],
) -> StreamingMode:
    """Override-aware resolver per D-9 ("User can override in settings").

    If override is None, delegates to default_streaming_mode_for_hardware().
    If override is a StreamingMode value, returns it verbatim — the
    resolver does NOT second-guess the user. The "user picked TRUE_STREAM
    on a CPU-only machine" case is handled at dispatch time by Story
    16.6's three-mode fallback chain (TRUE_STREAM -> SENTENCE_STREAM
    -> BATCH on failure), not here. This function's contract is
    "what mode did the user-or-hardware pick"; "what mode actually
    runs" is a separate (dispatch-layer) concern.

    Raises TypeError if override is not None and not a StreamingMode
    instance — the parameter is typed Optional[StreamingMode] precisely
    to keep string-vs-enum confusion out of the call sites; conversion
    from the AppSettings.streaming_mode_override string field is the
    caller's responsibility (Story 16.6 will do that conversion via
    StreamingMode(settings.streaming_mode_override) at dispatch time).
    """
    if override is None:
        return default_streaming_mode_for_hardware()
    if not isinstance(override, StreamingMode):
        raise TypeError(
            f"override must be StreamingMode or None, got "
            f"{type(override).__name__}"
        )
    return override
