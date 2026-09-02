"""Codec state caching across TRUE_STREAM chunks — Story 20.5 Phase 2 (AC #3).

WHAT THIS REMOVES
-----------------
``Qwen3TTSTokenizerV2CausalConvNet.forward``
(``modeling_qwen3_tts_tokenizer_v2.py:189-192``) left-pads every decode with
**zeros** where the previous chunk's real audio should be::

    hidden_state = F.pad(hidden_state, (self.padding, extra_padding),
                         mode="constant", value=0)

Through Story 20.4 every streamed chunk therefore began from a cold codec
state.  That single fact produced both defects Story 20.4 chased for four
audition rounds:

  * the deterministic 555-sample edge loss (``decode(N) == 1920*N - 555``),
    and
  * a cold-start error at each chunk head — ~35 % NRMSE, 0.55 median
    correlation falling to 0.11 over the blend region, +/-35 samples of lag
    jitter — which the Story 20.4 seam blend *masks* rather than removes.

This module carries the codec's real state across the boundary instead.
Story 20.5 Phase 1 measured the result on an RTX 5090 (evidence file
``20-5-codec-state-caching-evidence.md``): edge loss 555 -> **0**, head
NRMSE 115-144 % -> **0.56-0.82 %** (bit-exact in fp32 with TF32 off), lag
jitter +/-1200 -> **0 samples on every seam**, error-by-position **flat**
rather than head-weighted, at <= 2.52 MiB per session and no decode-time
regression.

THREE SUB-STACKS, THREE BOUNDARY BEHAVIOURS
-------------------------------------------
All three are necessary and none is sufficient (Phase 1 ran the full 2**3
ablation; leave-one-out from the full arm costs 0.008 -> 1.54 / 0.93 / 0.24):

1. **Causal conv** (``Qwen3TTSTokenizerV2CausalConvNet``).  Every instance in
   the decoder is stride-1 — :func:`probe_decoder` asserts this — so
   ``_get_extra_padding_for_conv1d`` returns 0 and the module reduces to
   "left-pad ``k_eff-1`` zeros, conv, output length == input length".  The
   streaming form keeps the last ``k_eff-1`` **input** samples as the next
   call's left context.  Exact.  Deepest buffer: 54 samples.

2. **Transposed conv** (``Qwen3TTSTokenizerV2CausalTransConvNet``).  Built two
   ways.  The ``upsampling_ratios`` instances have ``kernel == stride``, so
   ``pad == 0`` and they are already stateless and exact.  The
   ``upsample_rates`` instances have ``kernel == 2*stride``, so
   ``left_pad == right_pad == stride``: the module conv-transposes and then
   discards ``stride`` samples at each end.  That discard is the *entire*
   555-sample edge loss::

       stride 8 x downstream upsample 60 = 480
       stride 5 x downstream upsample 12 =  60
       stride 4 x downstream upsample  3 =  12
       stride 3 x downstream upsample  1 =   3
                                   TOTAL = 555

   The streaming form overlap-adds the discarded right tail into the next
   chunk's head.  Exact.

3. **Transformer** (``Qwen3TTSTokenizerV2DecoderTransformerModel``).  Eight
   layers, ``layer_types == ["sliding_attention"] * 8``, ``sliding_window=72``
   — so its carried state is a **bounded** KV cache (measured 2.22 MiB at 231
   frames vs a 2.25 MiB analytic cap; it stops growing, it does not
   accumulate over a long utterance).

THE BIAS DOUBLE-COUNT TRAP — READ BEFORE TOUCHING :func:`_stream_tconv`
----------------------------------------------------------------------
``nn.ConvTranspose1d`` adds its bias to **every output position of each
partial convolution**, so a naive overlap-add of the two partials
double-counts it.  Uncorrected, that leaves a bias-shaped transient at every
seam which decays over ~2,000 samples, **survives an fp32 pass**, and reads as
a 17-21 % residual cold start — i.e. it looks exactly like the Phase 1 NO-GO
verdict ("the state we can reach does not determine the output").  It is not
that.  Subtracting one copy of ``conv.bias`` over the overlap region is what
makes the streaming decode bit-exact.  Phase 1 localised it with
``20-5-stage-probe.py`` to ``decoder[1].block[1]``, the first stride-8
transposed conv.  It cost that spike two runs.
``test_codec_state_cache.py::test_transposed_conv_bias_is_not_double_counted``
pins it directly, because a regression here would present as "state caching
does not work" rather than as a bug.

WHY A WRAPPER AND NOT A SUBCLASS OR A MONKEY-PATCH
--------------------------------------------------
The state has to thread through every conv and transposed conv, and those are
reached through ``nn.ModuleList`` nesting inside ``DecoderBlock`` ->
``ResidualUnit``; there is no single method to override.  Monkey-patching
``forward`` on the module *classes* would make the state process-global,
which fails the per-session requirement below.  So this module re-walks
``Qwen3TTSTokenizerV2Decoder.forward``'s module traversal, calling the
**loaded** submodules' inner ``nn.Conv1d`` / ``nn.ConvTranspose1d`` directly.
It subclasses nothing, patches nothing, copies no weights, and vendors no
file.

The cost of a wrapper is that it restates the traversal, so an upstream pin
bump that reorders ``decoder`` / ``upsample`` — or inserts a new
time-mixing module — could silently desync it.  Two mitigations, both
load-bearing:

  * :func:`probe_decoder` walks the whole module graph at build time and
    **refuses** (falls back to the stock decode) on any leaf type it does not
    know how to stream.  Silence beats wrong audio; a wrong-audio failure
    here would be inaudible as a bug and audible only as "the codec got
    worse".
  * ``tests/test_qwen_tts_internals.py`` pins the module chain this file
    walks, per the Story 16.1 / 16.4 trip-wire pattern.

PER-SESSION, NOT GLOBAL
-----------------------
All state lives in one :class:`CodecStreamState` instance owned by one
:class:`StatefulCodecDecoder` instance, built per dispatch.  There is no
module-level or class-level storage, so concurrent generations — reachable
via the HTTP API — each get their own.  :class:`StreamingDecoderWorker` calls
``reset()`` on session start, on cancel, and on completion.

GEOMETRY — AND WHY THE FIRST CHUNK IS 555 SAMPLES SHORTER
---------------------------------------------------------
The streamer emits ``chunk_size + lookahead`` = 30 frames per chunk and
slides by ``chunk_size`` = 25, so chunk *k* covers frames
``[25k, 25k+30)``.  State must therefore be committed at the **splice**
(frame ``25k+25``), not at the end of the window, or chunk *k+1* would resume
5 frames in the future.  :meth:`StatefulCodecDecoder.__call__` does that in
two passes: it decodes the first ``chunk_size`` frames with the live state,
snapshots, decodes the remaining ``lookahead`` frames on the snapshot, and
restores.  The snapshot is O(1) tensor *references*, not copies — every
state slot is reassigned rather than mutated in place, upstream included
(``DynamicSlidingWindowLayer.update`` rebinds ``self.keys``), so holding the
old references is a valid, free snapshot.

Consequences the worker relies on:

  * the first decode of a session returns ``1920*N - 555`` samples (the
    module's own left-pad trim, which the whole-sequence decode also pays);
  * **every later decode returns exactly ``1920*N``** — the 555 does not
    shrink, it *moves* to the single stream-start call;
  * so the first posted chunk is ``25*1920 - 555`` samples and every later one
    is exactly ``25*1920``.  Total for N frames is ``1920*N - 555``: identical
    to a whole-sequence decode, to the sample.

  * the audio this chunk retains past its splice is decoded from the *same
    snapshot* the next chunk resumes from, so the Story 20.4 overlap-add now
    blends a signal against a numerically identical copy of itself.  It is an
    identity operation, not a mask.  That is measured, not assumed — see
    ``test_overlap_add_is_an_identity_under_carried_state``.  AC #3 says
    re-evaluate the blend on evidence rather than assume; leaving it in place
    and *proving* it inert is what keeps Phase 2 to one variable.

NOT REPLICATED, DELIBERATELY
----------------------------
``Qwen3TTSTokenizerV2Model.decode`` (``:1194``) truncates its output to
``(audio_codes[..., 0] > 0).sum(1) * decode_upsample_rate`` — an
encoder-side heuristic for *padded batch* decode, where a zero code means
padding.  A streaming chunk from the talker carries no padding, and Story
20.4 verified the ``1920*N - 555`` identity on 14 independent residual
lengths plus every full chunk, so the clamp never fires today.  Replicating
it would let a single legitimate code-0 frame silently delete 1,920 samples
of real speech mid-stream.  This wrapper does not replicate it and records
the divergence here rather than in a commit message.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from myvoice.services.tts_streaming.streaming_decoder import (
    _CODEC_EDGE_LOSS_SAMPLES,
    _CODEC_SAMPLES_PER_FRAME,
)


logger = logging.getLogger(__name__)


# Operator kill switch.  Set to "0" / "false" / "off" to run the pre-20.5
# stateless decode on an otherwise unchanged binary.  It exists for two
# reasons: an escape hatch if the field ever disagrees with the bench, and —
# concretely — so the Phase 3 audition can generate BOTH arms from one build,
# which is what keeps "reference == what ships today" honest.
_ENV_ENABLE = "MYVOICE_CODEC_STATE_CACHE"

_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


def state_cache_enabled() -> bool:
    """True unless the operator kill switch is set to a disabling value."""
    raw = os.environ.get(_ENV_ENABLE)
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLED_VALUES


class UnsupportedDecoderGraph(RuntimeError):
    """The loaded decoder is not the module graph this wrapper knows how to
    stream.  Raised only by :func:`probe_decoder`, i.e. at build time, never
    mid-stream — the caller falls back to the stock stateless decode."""


# --------------------------------------------------------------------------- #
# Lazy handles on the upstream classes
# --------------------------------------------------------------------------- #


class _UpstreamTypes:
    """Resolved-once handles on the qwen-tts classes the traversal switches on.

    Imported lazily: ``qwen_tts`` pulls torch/CUDA and prints third-party
    banners, and this module is imported by ``qwen_tts_service`` at a point
    where test environments without CUDA-bound torch DLLs must stay viable.
    """

    _cache: Optional["_UpstreamTypes"] = None

    def __init__(self) -> None:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            SnakeBeta,
            Qwen3TTSTokenizerV2CausalConvNet,
            Qwen3TTSTokenizerV2CausalTransConvNet,
            Qwen3TTSTokenizerV2ConvNeXtBlock,
            Qwen3TTSTokenizerV2DecoderDecoderBlock,
            Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
        )

        self.CausalConv = Qwen3TTSTokenizerV2CausalConvNet
        self.CausalTConv = Qwen3TTSTokenizerV2CausalTransConvNet
        self.ConvNeXt = Qwen3TTSTokenizerV2ConvNeXtBlock
        self.DecBlock = Qwen3TTSTokenizerV2DecoderDecoderBlock
        self.ResUnit = Qwen3TTSTokenizerV2DecoderDecoderResidualUnit
        self.SnakeBeta = SnakeBeta

    @classmethod
    def get(cls) -> "_UpstreamTypes":
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache


# --------------------------------------------------------------------------- #
# Per-session state
# --------------------------------------------------------------------------- #


class CodecStreamState:
    """Per-session codec state: conv left contexts, transposed-conv overlap
    tails, and the transformer KV cache.

    Phase 1 measured the whole object at **37 tensors, <= 2.52 MiB in bf16**
    (21 conv/tconv buffers = 276 KiB; KV cache 2.25 MiB, self-bounded by
    ``sliding_window=72`` so it is constant in utterance length).  Nothing
    here is module- or class-level: one instance per stream.
    """

    __slots__ = ("conv", "tconv", "kv", "frames")

    def __init__(self) -> None:
        self.conv: Dict[str, Any] = {}    # module key -> [B, C, k_eff-1] left context
        self.tconv: Dict[str, Any] = {}   # module key -> [B, C, stride]  overlap tail
        self.kv: Any = None               # transformers Cache
        self.frames: int = 0              # frames already committed to state

    def clear(self) -> None:
        """Drop everything.  Session start, cancel, and completion each call
        this exactly once; there is no partial teardown."""
        self.conv = {}
        self.tconv = {}
        self.kv = None
        self.frames = 0

    # -- snapshot / restore -------------------------------------------- #
    #
    # FREE, because nothing here is ever mutated in place. ``_stream_conv``
    # and ``_stream_tconv`` rebind their dict entries to freshly cloned
    # tensors, and upstream's ``DynamicSlidingWindowLayer.update`` rebinds
    # ``self.keys`` / ``self.values`` to new slices of a new ``torch.cat``
    # result rather than writing into the old ones.  So a snapshot is a
    # shallow dict copy plus one tuple per KV layer -- no tensor copies, no
    # deepcopy of the config that a ``copy.deepcopy(cache)`` would drag in.

    def snapshot(self) -> Tuple[Any, ...]:
        kv_snap: Optional[List[Tuple[Any, ...]]] = None
        if self.kv is not None:
            kv_snap = [
                (
                    getattr(layer, "keys", None),
                    getattr(layer, "values", None),
                    getattr(layer, "cumulative_length", None),
                    getattr(layer, "is_initialized", False),
                )
                for layer in self.kv.layers
            ]
        return (dict(self.conv), dict(self.tconv), kv_snap, self.frames)

    def restore(self, snap: Tuple[Any, ...]) -> None:
        conv, tconv, kv_snap, frames = snap
        self.conv = conv
        self.tconv = tconv
        self.frames = frames
        if kv_snap is not None and self.kv is not None:
            for layer, (keys, values, cumulative, initialized) in zip(
                self.kv.layers, kv_snap
            ):
                layer.keys = keys
                layer.values = values
                if cumulative is not None:
                    layer.cumulative_length = cumulative
                layer.is_initialized = initialized

    # -- observability -------------------------------------------------- #

    def nbytes(self) -> Tuple[int, int]:
        """``(bytes, tensor_count)`` currently held.  Used by the cost metric
        and by the Phase 1 / Phase 2 evidence, not by the decode path."""
        total = 0
        count = 0
        for store in (self.conv, self.tconv):
            for tensor in store.values():
                total += tensor.numel() * tensor.element_size()
                count += 1
        if self.kv is not None:
            for layer in self.kv.layers:
                for attr in ("keys", "values"):
                    tensor = getattr(layer, attr, None)
                    if tensor is not None and hasattr(tensor, "numel"):
                        total += tensor.numel() * tensor.element_size()
                        count += 1
        return total, count


# --------------------------------------------------------------------------- #
# Structural probe — the build-time gate
# --------------------------------------------------------------------------- #


class DecoderGeometry:
    """What :func:`probe_decoder` derived from the loaded module graph.

    ``samples_per_frame`` and ``edge_loss_samples`` are *computed by symbolic
    length simulation over the real modules*, not assumed — which is what
    lets the caller compare them against the two measured constants in
    ``streaming_decoder.py`` and refuse if a pin bump moved the geometry.
    """

    __slots__ = ("samples_per_frame", "edge_loss_samples", "conv_modules",
                 "tconv_modules", "kv_layers")

    def __init__(self, samples_per_frame: int, edge_loss_samples: int,
                 conv_modules: int, tconv_modules: int, kv_layers: int) -> None:
        self.samples_per_frame = samples_per_frame
        self.edge_loss_samples = edge_loss_samples
        self.conv_modules = conv_modules
        self.tconv_modules = tconv_modules
        self.kv_layers = kv_layers

    def output_samples(self, n_frames: int, first_call: bool) -> int:
        """Length identity this wrapper guarantees.

        ``1920*N - 555`` on the first decode of a session (the module's own
        left-pad trim, which the whole-sequence decode pays too) and exactly
        ``1920*N`` on every later one.
        """
        loss = self.edge_loss_samples if first_call else 0
        return self.samples_per_frame * n_frames - loss


def probe_decoder(decoder: Any) -> DecoderGeometry:
    """Verify the loaded decoder is the graph this wrapper knows how to stream.

    Raises :class:`UnsupportedDecoderGraph` on anything unexpected.  The caller
    must treat that as "fall back to the stock stateless decode", never as
    "carry on anyway": every check below is a precondition for the traversal
    producing *correct audio*, and a wrong-audio failure would present as a
    codec regression rather than as a bug.
    """
    T = _UpstreamTypes.get()

    for attr in ("quantizer", "pre_conv", "pre_transformer", "upsample", "decoder"):
        if not hasattr(decoder, attr):
            raise UnsupportedDecoderGraph(
                f"Qwen3TTSTokenizerV2Decoder has no attribute {attr!r}; the "
                f"module chain codec_state_cache walks has changed upstream."
            )
    if not callable(getattr(decoder.quantizer, "decode", None)):
        raise UnsupportedDecoderGraph(
            "decoder.quantizer.decode is not callable."
        )

    transformer = decoder.pre_transformer
    config = getattr(transformer, "config", None)
    layer_types = list(getattr(config, "layer_types", []) or [])
    if not layer_types:
        raise UnsupportedDecoderGraph(
            "pre_transformer.config exposes no layer_types; the KV cache "
            "shape (and its self-bounding property) cannot be verified."
        )
    if any(t != "sliding_attention" for t in layer_types):
        # A full-attention layer would make the KV cache grow without bound
        # over a long utterance, which invalidates the AC #2 cost claim.
        raise UnsupportedDecoderGraph(
            f"pre_transformer has non-sliding layer types {sorted(set(layer_types))}; "
            f"the carried KV cache would no longer be bounded by "
            f"sliding_window and the per-session cost claim would not hold."
        )

    counts = {"conv": 0, "tconv": 0}

    def check_leaf(module: Any, path: str) -> None:
        """Assert every leaf is a type the traversal streams correctly."""
        if isinstance(module, T.CausalConv):
            counts["conv"] += 1
            # stride-1 is what makes ``_get_extra_padding_for_conv1d`` return
            # 0, which is what makes "left-pad k_eff-1, conv" exact and the
            # output length equal to the input length.
            if getattr(module, "stride", None) != 1:
                raise UnsupportedDecoderGraph(
                    f"{path}: CausalConvNet stride={module.stride!r}, expected 1. "
                    f"A strided causal conv needs extra right padding and the "
                    f"left-context form in _stream_conv is no longer exact."
                )
            if module.padding != module.kernel_size - module.stride:
                raise UnsupportedDecoderGraph(
                    f"{path}: CausalConvNet padding={module.padding} != "
                    f"kernel_size - stride."
                )
            return
        if isinstance(module, T.CausalTConv):
            counts["tconv"] += 1
            conv = getattr(module, "conv", None)
            stride = int(conv.stride[0])
            kernel = int(conv.kernel_size[0])
            if module.left_pad != module.right_pad:
                raise UnsupportedDecoderGraph(
                    f"{path}: CausalTransConvNet left_pad != right_pad; the "
                    f"overlap-add tail width in _stream_tconv assumes they match."
                )
            if module.left_pad != kernel - stride:
                raise UnsupportedDecoderGraph(
                    f"{path}: CausalTransConvNet pad={module.left_pad} != "
                    f"kernel({kernel}) - stride({stride})."
                )
            if kernel not in (stride, 2 * stride):
                raise UnsupportedDecoderGraph(
                    f"{path}: CausalTransConvNet kernel={kernel} is neither "
                    f"stride nor 2*stride ({stride}); only those two shapes "
                    f"have the overlap structure _stream_tconv implements."
                )
            return
        if isinstance(module, (T.SnakeBeta,)):
            # Pointwise in time: safe to call as-is, chunk by chunk.
            return
        raise UnsupportedDecoderGraph(
            f"{path}: unknown leaf module {type(module).__name__}. "
            f"codec_state_cache refuses to stream a module graph it has not "
            f"been shown to be exact on — a new time-mixing module here would "
            f"corrupt audio at every chunk boundary rather than fail loudly."
        )

    def walk(module: Any, path: str) -> None:
        if isinstance(module, T.ConvNeXt):
            check_leaf(module.dwconv, path + ".dwconv")
            return
        if isinstance(module, T.ResUnit):
            check_leaf(module.conv1, path + ".conv1")
            check_leaf(module.conv2, path + ".conv2")
            return
        if isinstance(module, T.DecBlock):
            for i, sub in enumerate(module.block):
                walk(sub, f"{path}.block.{i}")
            return
        check_leaf(module, path)

    check_leaf(decoder.pre_conv, "pre_conv")
    for i, blocks in enumerate(decoder.upsample):
        for j, block in enumerate(blocks):
            walk(block, f"upsample.{i}.{j}")
    for i, block in enumerate(decoder.decoder):
        walk(block, f"decoder.{i}")

    spf, edge = _simulate_length(decoder)
    return DecoderGeometry(
        samples_per_frame=spf,
        edge_loss_samples=edge,
        conv_modules=counts["conv"],
        tconv_modules=counts["tconv"],
        kv_layers=len(layer_types),
    )


def _simulate_length(decoder: Any) -> Tuple[int, int]:
    """Symbolically walk the traversal to derive ``(samples_per_frame,
    edge_loss)`` from the loaded modules.

    Length arithmetic, stage by stage:
      * causal conv (stride 1)                 -> length unchanged
      * transposed conv, kernel == stride      -> length * stride
      * transposed conv, kernel == 2*stride    -> length * stride - stride on a
        cold start (the module trims ``left_pad`` and ``right_pad``), and
        exactly ``length * stride`` once the previous chunk's tail is
        overlap-added in.

    So the loss contributed by each 2*stride transposed conv is ``stride``
    samples *at that stage's resolution*, which the remaining upsampling
    multiplies up.  For the shipping model that is
    ``8*60 + 5*12 + 4*3 + 3*1 = 555`` — exactly the constant Story 20.4
    measured on 14 independent residual lengths.
    """
    T = _UpstreamTypes.get()
    # (multiplier, accumulated_loss) in the current stage's units, folded
    # forward so a loss incurred early is scaled by everything downstream.
    mult = 1
    loss = 0

    def visit(module: Any) -> None:
        nonlocal mult, loss
        if isinstance(module, T.CausalTConv):
            stride = int(module.conv.stride[0])
            mult *= stride
            loss = loss * stride + (module.left_pad if module.left_pad else 0)
            return
        if isinstance(module, T.ConvNeXt):
            return
        if isinstance(module, (T.DecBlock,)):
            for sub in module.block:
                visit(sub)
            return
        return

    for blocks in decoder.upsample:
        for block in blocks:
            visit(block)
    for block in decoder.decoder:
        visit(block)
    return mult, loss


# --------------------------------------------------------------------------- #
# The streaming traversal
# --------------------------------------------------------------------------- #


def _stream_conv(module: Any, x: Any, state: CodecStreamState, key: str) -> Any:
    """Streaming ``CausalConvNet``: real left context instead of zeros.

    Upstream left-pads ``padding = k_eff - stride`` zeros on every call.  Here
    the previous chunk's last ``padding`` **input** samples take their place,
    which is exactly what a whole-sequence decode would have seen.
    """
    pad = module.padding
    if pad == 0:
        return module.conv(x).contiguous()
    buf = state.conv.get(key)
    if buf is None:
        # First chunk of the session: zeros, same as the whole-sequence
        # decode's own start. Not an approximation — it is the ground truth.
        buf = x.new_zeros(x.shape[0], x.shape[1], pad)
    y = _cat_time(buf, x)
    state.conv[key] = y[..., y.shape[-1] - pad:].clone()
    return module.conv(y).contiguous()


def _stream_tconv(module: Any, x: Any, state: CodecStreamState, key: str) -> Any:
    """Streaming ``CausalTransConvNet``: overlap-add the discarded tail.

    Upstream computes ``conv_transpose(x)`` and then *throws away*
    ``left_pad`` samples at the head and ``right_pad`` at the tail.  The tail
    it throws away is the first ``stride`` samples of the next chunk's
    contribution; retaining it and adding it to the next chunk's head is what
    removes the 555-sample edge loss entirely.

    **The bias correction is not optional.**  ``nn.ConvTranspose1d`` adds its
    bias to every output position of *each* partial convolution, so the naive
    sum double-counts it.  See this module's docstring: uncorrected, the
    result reads as a 17-21 % residual cold start that survives fp32 — i.e.
    it looks exactly like the mechanism failing.
    """
    left_pad, right_pad = module.left_pad, module.right_pad
    if left_pad == 0 and right_pad == 0:
        # kernel == stride: no overlap, already stateless and exact.
        return module.conv(x).contiguous()

    y = module.conv(x)                       # length L*stride + left_pad
    prev = state.tconv.get(key)
    if prev is not None:
        head = y[..., :left_pad] + prev
        bias = module.conv.bias
        if bias is not None:
            head = head - bias.view(1, -1, 1)
        y = _cat_time(head, y[..., left_pad:])
    state.tconv[key] = y[..., y.shape[-1] - right_pad:].clone()
    out = y[..., : y.shape[-1] - right_pad]
    if prev is None:
        # Cold start: keep the module's own left trim, which is what the
        # whole-sequence decode also loses. This is the entire edge loss,
        # and carrying state moves it here — to one call per session —
        # rather than removing it from the codec.
        out = out[..., left_pad:]
    return out.contiguous()


def _cat_time(a: Any, b: Any) -> Any:
    return torch.cat([a, b], dim=-1)


def _stream_transformer(transformer: Any, hidden: Any,
                        state: CodecStreamState) -> Any:
    """Streaming ``DecoderTransformerModel``: a bounded sliding KV cache."""
    from transformers.cache_utils import DynamicCache

    if state.kv is None:
        state.kv = DynamicCache(config=transformer.config)
    n = hidden.shape[1]
    cache_position = torch.arange(
        state.frames, state.frames + n, device=hidden.device
    )
    out = transformer(
        inputs_embeds=hidden,
        past_key_values=state.kv,
        use_cache=True,
        cache_position=cache_position,
    )
    state.kv = out.past_key_values
    return out.last_hidden_state


def _run_module(module: Any, x: Any, state: CodecStreamState, key: str) -> Any:
    """Streaming twin of one node in the decoder's module tree.

    Mirrors upstream's ``forward`` bodies exactly; only the conv calls are
    replaced.  :func:`probe_decoder` has already refused any leaf type this
    function would otherwise pass through blind.
    """
    T = _UpstreamTypes.get()
    if isinstance(module, T.CausalConv):
        return _stream_conv(module, x, state, key)
    if isinstance(module, T.CausalTConv):
        return _stream_tconv(module, x, state, key)
    if isinstance(module, T.ConvNeXt):
        residual = x
        h = _stream_conv(module.dwconv, x, state, key + ".dwconv")
        h = h.permute(0, 2, 1)
        h = module.norm(h)
        h = module.pwconv1(h)
        h = module.act(h)
        h = module.pwconv2(h)
        h = module.gamma * h
        h = h.permute(0, 2, 1)
        return residual + h
    if isinstance(module, T.ResUnit):
        residual = x
        h = module.act1(x)
        h = _stream_conv(module.conv1, h, state, key + ".conv1")
        h = module.act2(h)
        h = _stream_conv(module.conv2, h, state, key + ".conv2")
        return h + residual
    if isinstance(module, T.DecBlock):
        h = x
        for i, sub in enumerate(module.block):
            h = _run_module(sub, h, state, f"{key}.block.{i}")
        return h
    # SnakeBeta and anything else pointwise in time. probe_decoder has
    # already established that nothing time-mixing reaches here.
    return module(x)


def stream_forward(decoder: Any, codes: Any, state: CodecStreamState) -> Any:
    """Streaming twin of ``Qwen3TTSTokenizerV2Decoder.forward`` (``:876``).

    ``codes`` is ``(batch, num_quantizers, n_frames)`` — the shape
    ``Qwen3TTSTokenizerV2Model.decode`` hands to ``chunked_decode``.
    """
    n_frames = codes.shape[-1]
    hidden = decoder.quantizer.decode(codes)
    hidden = _stream_conv(decoder.pre_conv, hidden, state, "pre_conv")
    hidden = hidden.transpose(1, 2)
    hidden = _stream_transformer(decoder.pre_transformer, hidden, state)
    hidden = hidden.permute(0, 2, 1)
    for i, blocks in enumerate(decoder.upsample):
        for j, block in enumerate(blocks):
            hidden = _run_module(block, hidden, state, f"up.{i}.{j}")
    wav = hidden
    for i, block in enumerate(decoder.decoder):
        wav = _run_module(block, wav, state, f"dec.{i}")
    state.frames += n_frames
    return wav.clamp(min=-1, max=1)


# --------------------------------------------------------------------------- #
# The decode_fn the worker consumes
# --------------------------------------------------------------------------- #


class StatefulCodecDecoder:
    """P-6 ``decode_fn`` that carries codec state across chunk boundaries.

    Same contract as the stock adapter in
    ``QwenTTSService._build_true_stream_decode_fn``: called with one chunk
    from the talker forward-hook (a ``torch.Tensor`` of shape
    ``(n_frames, num_code_groups)``), returns a flat float32 PCM array.

    What differs is the geometry, and the worker is told about it via
    :attr:`carries_codec_state`:

      * first decode of a session -> ``1920*N - 555`` samples
      * every later decode        -> exactly ``1920*N`` samples

    State is committed at the **splice** (``commit_frames``), not at the end
    of the window, so the next chunk resumes at the right instant.  See the
    module docstring's geometry section.
    """

    carries_codec_state = True

    def __init__(
        self,
        decoder: Any,
        geometry: DecoderGeometry,
        commit_frames: int,
        window_frames: int,
        device: Any = None,
    ) -> None:
        if commit_frames <= 0:
            raise ValueError("commit_frames must be positive")
        if window_frames < commit_frames:
            raise ValueError("window_frames must be >= commit_frames")
        self._decoder = decoder
        self._geometry = geometry
        self._commit_frames = commit_frames
        self._window_frames = window_frames
        self._device = device
        self._state = CodecStreamState()
        self._calls = 0

    # -- lifecycle ------------------------------------------------------ #

    @property
    def commit_frames(self) -> int:
        return self._commit_frames

    @property
    def state_frames(self) -> int:
        """Frames committed to codec state so far.  0 before the first
        decode and immediately after every reset."""
        return self._state.frames

    def reset(self) -> None:
        """Drop all carried state.  Called by the worker on session start, on
        cancel, and on completion — the three points AC #3 names.  Idempotent
        and safe to call from the worker's cancel path, which must never
        raise."""
        self._state.clear()
        self._calls = 0

    def state_nbytes(self) -> Tuple[int, int]:
        return self._state.nbytes()

    # -- P-6 decode_fn -------------------------------------------------- #

    def __call__(self, chunk: Any) -> np.ndarray:
        codes = self._as_codes(chunk)
        n_frames = int(codes.shape[-1])
        if n_frames == 0:
            return np.asarray([], dtype=np.float32)

        # Commit exactly the splice's worth of frames; decode the trailing
        # lookahead on a snapshot and roll the state back, so the next chunk
        # resumes at frame ``state.frames + commit`` rather than 5 frames in
        # its own future.  ``is_full_window`` here MUST agree with
        # ``StreamingDecoderWorker._decode_and_post``; both derive it from the
        # same snapshotted streamer geometry, and
        # ``test_commit_predicate_matches_the_workers_full_window_predicate``
        # pins the agreement.
        commit = (
            self._commit_frames
            if n_frames >= self._window_frames and self._window_frames > self._commit_frames
            else n_frames
        )

        first_call = self._state.frames == 0
        with torch.inference_mode():
            wav = stream_forward(self._decoder, codes[..., :commit], self._state)
            if commit < n_frames:
                snap = self._state.snapshot()
                try:
                    tail = stream_forward(
                        self._decoder, codes[..., commit:], self._state
                    )
                finally:
                    self._state.restore(snap)
                wav = torch.cat([wav, tail], dim=-1)

        audio = wav.squeeze(0).squeeze(0).to(torch.float32).detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).flatten()

        expected = self._geometry.output_samples(n_frames, first_call)
        if audio.size != expected:
            # A theorem, not a hope: probe_decoder verified every stage's
            # length arithmetic at build time. Reaching here means the graph
            # changed under us mid-session, and the honest response is to say
            # so rather than post mis-aligned audio.
            raise RuntimeError(
                f"codec state cache produced {audio.size} samples for "
                f"{n_frames} frames (first_call={first_call}); expected "
                f"{expected}. The decoder module graph changed after "
                f"probe_decoder verified it."
            )
        self._calls += 1
        return audio

    # -- helpers -------------------------------------------------------- #

    def _as_codes(self, chunk: Any) -> Any:
        """Normalise the talker forward-hook's chunk into the
        ``(batch, num_quantizers, n_frames)`` shape the decoder wants.

        Mirrors the stock adapter's coercions (which exist for the
        residual-flush path and for robustness against upstream shape tweaks)
        plus the two reshapes ``Qwen3TTSTokenizer.decode`` and
        ``Qwen3TTSTokenizerV2Model.decode`` would have applied:
        ``(N, Q)`` -> ``(1, N, Q)`` -> ``(1, Q, N)``.
        """
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.as_tensor(chunk, dtype=torch.long)
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(-1)
        if chunk.dim() != 2:
            raise ValueError(
                f"expected a (n_frames, num_quantizers) chunk, got shape "
                f"{tuple(chunk.shape)}"
            )
        if chunk.dtype != torch.long:
            chunk = chunk.long()
        if self._device is not None:
            chunk = chunk.to(self._device)
        return chunk.unsqueeze(0).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------- #
# Build-time construction + self-test
# --------------------------------------------------------------------------- #


def build_stateful_decode_fn(
    decoder: Any,
    *,
    chunk_size: int,
    lookahead: int,
    device: Any = None,
    self_test: bool = True,
    log: Optional[logging.Logger] = None,
) -> Tuple[Optional[StatefulCodecDecoder], str]:
    """Return ``(decoder_or_None, reason)``.

    ``None`` means "use the stock stateless decode" and ``reason`` says why —
    the caller logs it.  Every rejection path here is deliberate: this wrapper
    re-walks upstream internals, so the only safe failure mode is to decline
    loudly and ship today's audio, never to stream a graph it has not verified.

    The numerical self-test runs **once per loaded decoder** (memoised on the
    module object).  In the shipping configuration that first build is the
    compile-priming generation at startup, so no user-visible generation pays
    for it; if priming is off it costs one small decode on the first
    generation of the process.
    """
    lg = log or logger
    if not state_cache_enabled():
        return None, f"disabled by {_ENV_ENABLE}"

    try:
        geometry = probe_decoder(decoder)
    except UnsupportedDecoderGraph as exc:
        return None, f"decoder graph not supported: {exc}"
    except Exception as exc:  # noqa: BLE001 — probing must never break dispatch
        return None, f"decoder probe failed: {exc!r}"

    if geometry.samples_per_frame != _CODEC_SAMPLES_PER_FRAME:
        return None, (
            f"decoder upsamples {geometry.samples_per_frame} samples/frame, "
            f"but streaming_decoder pins {_CODEC_SAMPLES_PER_FRAME}"
        )
    if geometry.edge_loss_samples != _CODEC_EDGE_LOSS_SAMPLES:
        return None, (
            f"decoder edge loss is {geometry.edge_loss_samples} samples, but "
            f"streaming_decoder pins {_CODEC_EDGE_LOSS_SAMPLES}"
        )

    fn = StatefulCodecDecoder(
        decoder=decoder,
        geometry=geometry,
        commit_frames=chunk_size,
        window_frames=chunk_size + lookahead,
        device=device,
    )

    if self_test and not getattr(decoder, "_myvoice_state_cache_verified", False):
        ok, detail = _numerical_self_test(decoder, geometry, device=device)
        if not ok:
            return None, f"numerical self-test failed: {detail}"
        try:
            decoder._myvoice_state_cache_verified = True
        except Exception:  # pragma: no cover (defensive)
            pass
        lg.info("[CodecStateCache] self-test passed: %s", detail)

    return fn, "enabled"


def _numerical_self_test(decoder: Any, geometry: DecoderGeometry, *,
                         device: Any = None) -> Tuple[bool, str]:
    """Verify the traversal against the LOADED weights, three ways.

    What this gate is for: catching a mismatch between *this code* and *this
    model* — a pin bump that reordered something ``probe_decoder``'s
    structural walk could not see. It is not a substitute for the CI suite;
    see the note at the bottom.

    1. **Single-call streaming == ``decoder.forward``.**  Hand the whole
       sequence to :func:`stream_forward` in one call and there is no chunk
       boundary, so any difference at all is a transcription error in the
       module walk. Measured on the shipping RTX 5090 / bf16 configuration
       this is **exactly 0.0** — bit-for-bit, on random and on real token
       sequences alike, because it is the same ops on the same shapes. That
       makes it a precision-independent gate, which is what a runtime check
       needs to be.

    2. **The chunked length identity.**  ``1920*N - 555`` once, then exactly
       ``1920*N``. Phase 1's ablation attributed 100 % of the edge loss to
       the transposed convs — every arm carrying tconv state reached +0
       samples, every arm without it stayed at −555 per seam — so this single
       integer comparison proves the transposed-conv overlap-add is live.

    3. **Chunked content is in the state-carried regime, not the cold-start
       one.**  The two are three orders apart: Phase 1 measured ~1.0-1.4
       NRMSE with no state against ~0.01 with it, and this host reproduces
       0.94-1.13 vs 0.007-0.009 on real tokens. The tolerance sits between
       them with wide margin on both sides.

    **What this deliberately does NOT gate: the ConvTranspose1d bias
    double-count.**  On *random* codes the bf16 floor (3.3e-02 - 5.4e-02
    measured here) is too close to the bug's own signature (5.9e-02) to
    separate them, and the self-test has no real tokens at build time. That
    is fine, and stating it is better than a tolerance that pretends
    otherwise: the bias trap is a code defect, not a model mismatch, so it
    cannot appear at runtime without the source changing — and
    ``test_codec_state_cache.py::test_transposed_conv_bias_is_not_double_counted``
    pins it in CI with six orders of margin, in float64 on CPU, where it
    separates cleanly.

    Cost: three short decodes of 24 frames, ~10 ms on the reference RTX 5090,
    paid once per model load. In the shipping configuration that is the
    compile-priming generation at startup, so no user-visible generation pays
    for it.
    """
    try:
        param = next(decoder.parameters())
    except StopIteration:  # pragma: no cover (defensive)
        return False, "decoder has no parameters"
    dev = device if device is not None else param.device
    dtype = param.dtype

    num_q = int(getattr(decoder.config, "num_quantizers", 0)) or 1
    codebook = int(getattr(decoder.config, "codebook_size", 2)) or 2
    n = 24
    half = n // 2
    gen = torch.Generator(device="cpu").manual_seed(2050)
    codes = torch.randint(
        1, max(2, codebook), (1, num_q, n), generator=gen, dtype=torch.long
    ).to(dev)

    with torch.inference_mode():
        whole = decoder(codes)
        one_call = stream_forward(decoder, codes, CodecStreamState())
        state = CodecStreamState()
        first = stream_forward(decoder, codes[..., :half], state)
        second = stream_forward(decoder, codes[..., half:], state)
    chunked = torch.cat([first, second], dim=-1)

    # -- 1. the traversal itself ---------------------------------------- #
    if one_call.shape != whole.shape:
        return False, (
            "single-call streaming returned {} samples, forward returned {}"
            .format(one_call.shape[-1], whole.shape[-1])
        )
    traversal = _nrmse(whole, one_call)
    if not (traversal == traversal) or traversal > 1e-4:
        return False, (
            "single-call streaming diverged from decoder.forward "
            "(NRMSE {:.3e}); the module traversal no longer matches this "
            "model's graph".format(traversal)
        )

    # -- 2. the length identity ----------------------------------------- #
    want_first = geometry.output_samples(half, first_call=True)
    want_second = geometry.output_samples(n - half, first_call=False)
    if first.shape[-1] != want_first or second.shape[-1] != want_second:
        return False, (
            "chunk lengths {}/{} != expected {}/{}; the transposed-conv "
            "overlap-add is not moving the edge loss to the stream-start "
            "call".format(first.shape[-1], second.shape[-1],
                          want_first, want_second)
        )

    # -- 3. state-carried regime, not cold-start ------------------------ #
    carried = _nrmse(whole, chunked)
    tol = 0.15 if dtype in (torch.bfloat16, torch.float16) else 1e-3
    if not (carried == carried) or carried > tol:
        return False, (
            "chunked NRMSE {:.3e} > tol {:.1e} (dtype={}); this is the "
            "cold-start regime, not the state-carried one".format(
                carried, tol, dtype)
        )
    return True, (
        "traversal {:.3e} (bit-exact target), chunked {:.3e} <= {:.1e}, "
        "lengths {}+{} exact (dtype={})".format(
            traversal, carried, tol, want_first, want_second, dtype)
    )


def _nrmse(reference: Any, test: Any) -> float:
    a = reference.to(torch.float64).reshape(-1)
    b = test.to(torch.float64).reshape(-1)
    denom = torch.linalg.vector_norm(a).item()
    if denom == 0:
        return 0.0
    return torch.linalg.vector_norm(a - b).item() / denom


__all__ = [
    "CodecStreamState",
    "DecoderGeometry",
    "StatefulCodecDecoder",
    "UnsupportedDecoderGraph",
    "build_stateful_decode_fn",
    "probe_decoder",
    "state_cache_enabled",
    "stream_forward",
]
