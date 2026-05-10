"""tts_streaming subpackage — Phase ⊥ of D-20 (architecture-optimization-pass.md).

The decision-layer for streaming-mode selection. Story 16.2 ships the enum +
hardware probe + override-aware resolver; Stories 16.3–16.5 add the
streamer/decoder/cancel infrastructure; Story 16.6 wires dispatch in
QwenTTSService. This file's only job is to re-export the three public
symbols so call sites can `from myvoice.services.tts_streaming import ...`
without reaching into the leaf module.
"""

from myvoice.services.tts_streaming.streaming_mode import (
    StreamingMode,
    default_streaming_mode_for_hardware,
    effective_streaming_mode,
)
from myvoice.services.tts_streaming.codec_token_streamer import (
    CodecTokenStreamer,
    END_OF_STREAM,
)
from myvoice.services.tts_streaming.streaming_decoder import (
    StreamingDecoderWorker,
)
from myvoice.services.tts_streaming.torch_runtime import (
    enable_tf32_and_cudnn_benchmark,
    is_ampere_or_newer,
    resolve_tts_precision,
)

__all__ = [
    "StreamingMode",
    "default_streaming_mode_for_hardware",
    "effective_streaming_mode",
    "CodecTokenStreamer",
    "END_OF_STREAM",
    "StreamingDecoderWorker",
    "enable_tf32_and_cudnn_benchmark",
    "is_ampere_or_newer",
    "resolve_tts_precision",
]
