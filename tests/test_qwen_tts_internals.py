"""Story 16.1 / D-12 — pin the qwen-tts attribute surface against silent upstream renames.

These three tests assert that every symbol MyVoice imports from `qwen_tts` and every
method MyVoice calls on a `Qwen3TTSModel` instance is present on the pinned install.
If any test fails, MyVoice's runtime calls into qwen-tts will break in production.
Before bumping the pinned commit hash in requirements.txt + build_tools/requirements-production.txt,
run this file. Failures name the missing symbols; fix the call sites in
src/myvoice/services/qwen_tts_service.py and src/myvoice/services/model_registry.py
before merging the bump.

Story 16.3 extension: This file also pins `transformers.generation.streamers.BaseStreamer`
— the parent class of `CodecTokenStreamer` (Phase ⊥ true-streaming). The same trip-wire
pattern applies: a silent upstream rename or refactor in HF transformers fails this file
before reaching production import.

Story 16.4 extension: also pins the deep-decoder method `Qwen3TTSTokenizerV1Model.decode`
for the upcoming Story 16.6 TRUE_STREAM dispatch adapter. The Story 16.4 worker itself
does not import qwen_tts — but Story 16.6's `decode_fn` adapter wraps this method, and
this trip-wire fails loudly in CI before a silent qwen-tts rename can break the adapter.

Story 16.8 extension: pins the talker-patch surface that
`_build_true_stream_talker` depends on at runtime — the
`Qwen3TTSForConditionalGeneration` class, the `Qwen3TTSTalkerForConditionalGeneration`
class our patch wraps, and the call-site invariant that
`Qwen3TTSForConditionalGeneration.generate` invokes `self.talker.generate(...)`
exactly once (the interposition point our patch lands on).
"""


def test_qwen3_tts_model_class_is_top_level_importable():
    """services/model_registry.py:34 imports `Qwen3TTSModel` from the package root."""
    from qwen_tts import Qwen3TTSModel
    assert Qwen3TTSModel is not None


def test_voice_clone_prompt_item_class_is_deep_path_importable():
    """services/qwen_tts_service.py:32 imports `VoiceClonePromptItem as LibraryVoiceClonePromptItem`
    from the deep path (aliased to avoid shadowing the local wrapper class at line 176)
    AND services/qwen_tts_service.py:1324 performs an isinstance-via-__module__ runtime
    check that depends on this exact module path string being stable.
    """
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    assert VoiceClonePromptItem is not None
    assert VoiceClonePromptItem.__module__ == "qwen_tts.inference.qwen3_tts_model"


def test_qwen3_tts_model_method_surface_intact():
    """Pin every callable MyVoice invokes on a Qwen3TTSModel instance.

    Story 18.4 / D-22 Branch B extension: `enable_streaming_optimizations` is the
    upstream-blessed compile+CUDA-graph engagement API introduced by the
    dffdeeq/Qwen3-TTS-streaming fork at commit `3fdb4682` ("compile and fast codebook",
    2026-02-03). Story 18.4 wires it into the model-load path via
    services/tts_streaming/torch_runtime.py:engage_compile_optimizations. The architecture
    binds the call (P-11 invariant assertion at startup); this trip-wire fails CI before
    a silent fork-pin reversion can break that wire-up.
    """
    from qwen_tts import Qwen3TTSModel
    expected_methods = (
        "from_pretrained",                    # classmethod — services/model_registry.py:459
        "create_voice_clone_prompt",          # services/qwen_tts_service.py:1285, 1296
        "generate_voice_clone",               # services/qwen_tts_service.py:2606, 2632
        "generate_voice_design",              # services/qwen_tts_service.py:2596
        "generate_custom_voice",              # services/qwen_tts_service.py:2588
        "enable_streaming_optimizations",     # Story 18.4 — services/tts_streaming/torch_runtime.py
    )
    missing = [m for m in expected_methods if not callable(getattr(Qwen3TTSModel, m, None))]
    assert not missing, (
        f"qwen-tts upstream renamed or removed: {missing}. "
        f"MyVoice depends on these. Update qwen_tts_service.py / model_registry.py "
        f"before bumping the pinned commit hash in requirements.txt + "
        f"build_tools/requirements-production.txt."
    )


def test_transformers_basestreamer_importable_from_deep_path():
    """Story 16.3 — pin transformers.generation.streamers.BaseStreamer.

    services/tts_streaming/codec_token_streamer.py imports BaseStreamer
    from this exact deep path (architecture line 671 names the path
    verbatim). A future transformers refactor that moves BaseStreamer
    out of generation.streamers — or splits it into separate producer/
    consumer abstract classes — would silently break MyVoice's TRUE_STREAM
    path; this trip-wire fails loudly in CI before the rename can ship.
    """
    from transformers.generation.streamers import BaseStreamer

    assert isinstance(BaseStreamer, type), (
        "transformers.generation.streamers.BaseStreamer is not a class — "
        "upstream may have renamed or refactored. Update the import in "
        "src/myvoice/services/tts_streaming/codec_token_streamer.py and "
        "this test in lockstep."
    )
    # The HF contract is: streamers must implement put() and end().
    # Story 16.3's CodecTokenStreamer extends with reset() (MyVoice-
    # specific). If BaseStreamer ever drops put or end, our subclass
    # contract breaks; pin both attributes here.
    for method_name in ("put", "end"):
        assert hasattr(BaseStreamer, method_name), (
            f"transformers.generation.streamers.BaseStreamer is missing "
            f"abstract method {method_name!r}. The HF contract for HF-"
            f"streamers has changed; update CodecTokenStreamer's "
            f"three-method contract before bumping the transformers pin."
        )


def test_qwen3_tts_tokenizer_v1_decode_method_pinned():
    """Story 16.4 / Story 16.6 — pin Qwen3TTSTokenizerV1Model.decode.

    services/tts_streaming/streaming_decoder.py's `decode_fn` callable
    wraps a `model.speech_tokenizer.decode(...)` call where
    `model.speech_tokenizer` is an instance of Qwen3TTSTokenizerV1Model
    per qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py
    (lines 1487-1525, signature `decode(audio_codes, xvectors, ref_mels,
    return_dict=None) -> tuple[Tensor, ...]`). The Story 16.6 adapter
    composes that callable from the model loaded by services/model_registry.py.
    A silent rename of `decode` → `synthesize` (or any other refactor)
    breaks Story 16.6's adapter at construction time; this trip-wire
    fails loudly in CI before the rename can ship.
    """
    from qwen_tts.core.tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import (
        Qwen3TTSTokenizerV1Model,
    )

    assert isinstance(Qwen3TTSTokenizerV1Model, type), (
        "Qwen3TTSTokenizerV1Model is not a class — upstream may have "
        "renamed or refactored. Update the trip-wire AND the Story 16.6 "
        "adapter in lockstep."
    )
    assert callable(getattr(Qwen3TTSTokenizerV1Model, "decode", None)), (
        "Qwen3TTSTokenizerV1Model has no callable attribute 'decode'. "
        "The deep-decoder method that Story 16.6's TRUE_STREAM adapter "
        "wraps has been renamed or removed. Update the adapter before "
        "bumping the qwen-tts pin."
    )


def test_qwen3_tts_for_conditional_generation_class_is_deep_path_importable():
    """Story 16.8 — pin Qwen3TTSForConditionalGeneration at the deep path.

    services/qwen_tts_service.py:_build_true_stream_talker uses
    `model.model.talker.generate` where `model.model` is an instance of
    Qwen3TTSForConditionalGeneration (constructed by the qwen_tts loader,
    not directly by MyVoice). Pinning the class here ensures a silent
    rename or relocation in `qwen_tts.core.models.modeling_qwen3_tts`
    fails CI before the talker patch can dereference a stale class.
    """
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    assert isinstance(Qwen3TTSForConditionalGeneration, type), (
        "Qwen3TTSForConditionalGeneration is not a class. "
        "Story 16.8's _build_true_stream_talker reaches `model.model.talker` "
        "on instances of this class — a structural change here breaks the "
        "TRUE_STREAM wire-up. Update qwen_tts_service.py before bumping the "
        "pin."
    )


def test_qwen3_tts_talker_for_conditional_generation_class_is_callable():
    """Story 16.8 — pin Qwen3TTSTalkerForConditionalGeneration.generate.

    services/qwen_tts_service.py:_build_true_stream_talker monkey-patches
    `model.model.talker.generate` for the duration of one dispatch to
    inject `streamer=streamer` into HF GenerationMixin's standard
    streaming hook. The patch's correctness depends on:
      (a) `Qwen3TTSTalkerForConditionalGeneration` being a real class
          (the type of `model.model.talker`), and
      (b) its `.generate` being a callable (inherited from
          `transformers.GenerationMixin.generate` per the qwen-tts
          0.0.4 class hierarchy).
    A silent upstream rename of either fails this test before the
    talker patch can interpose on a non-existent method.
    """
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    assert isinstance(Qwen3TTSTalkerForConditionalGeneration, type), (
        "Qwen3TTSTalkerForConditionalGeneration is not a class. "
        "Story 16.8's talker-patch wire-up depends on `model.model.talker` "
        "being an instance of this class. Update _build_true_stream_talker "
        "before bumping the pin."
    )
    assert callable(getattr(
        Qwen3TTSTalkerForConditionalGeneration, "generate", None
    )), (
        "Qwen3TTSTalkerForConditionalGeneration has no callable "
        "attribute 'generate'. Story 16.8's talker-patch interposes on "
        "this method (HF GenerationMixin protocol). The streamer kwarg "
        "is forwarded through this method — without it, TRUE_STREAM "
        "cannot stream at all."
    )
    # Story 16.8 forward-hook: services/qwen_tts_service.py:_build_true_stream_talker
    # patches `model.model.talker.forward` to capture per-step `codec_ids`
    # from `Qwen3TTSTalkerOutputWithPast.hidden_states[1]`. HF's
    # `GenerationMixin._sample` loop drives forward via the model's
    # `__call__`, which dispatches to `forward`. A future qwen-tts that
    # renames or refactors the forward entrypoint breaks the hook.
    assert callable(getattr(
        Qwen3TTSTalkerForConditionalGeneration, "forward", None
    )), (
        "Qwen3TTSTalkerForConditionalGeneration has no callable "
        "attribute 'forward'. Story 16.8's forward-hook interposes on "
        "this method to capture multi-codebook codec_ids per step "
        "(modeling_qwen3_tts.py:1738 returns "
        "`hidden_states=(..., codec_ids)` from forward). Without this "
        "method, the talker-patch's chunk capture mechanism breaks and "
        "TRUE_STREAM produces silence."
    )


def test_qwen3_tts_wrapper_constructs_talker_attribute_in_init():
    """Story 16.8 — pin `self.talker = Qwen3TTSTalkerForConditionalGeneration(...)`
    in Qwen3TTSForConditionalGeneration.__init__.

    services/qwen_tts_service.py:_build_true_stream_talker dereferences
    `model.model.talker` — that attribute is created during
    Qwen3TTSForConditionalGeneration.__init__ at modeling_qwen3_tts.py:1820
    (`self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)`).
    A future qwen-tts version that renames the attribute (e.g., `self.gen_model`,
    `self._talker`) would silently break the talker-patch wire-up — the patch
    would `setattr` a new attribute rather than overriding the real one,
    and HF's existing `self.talker.generate(...)` call site would still
    hit the original unpatched method, leaving streaming broken.

    Source-level inspection here pins the attribute name. Brittle (it
    asserts on string content) but the alternative — instantiating the
    full model — is too expensive for a trip-wire.
    """
    import inspect

    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    init_source = inspect.getsource(
        Qwen3TTSForConditionalGeneration.__init__
    )
    assert "self.talker" in init_source, (
        "Qwen3TTSForConditionalGeneration.__init__ no longer assigns "
        "`self.talker = ...`. Story 16.8's talker-patch wire-up at "
        "qwen_tts_service.py:_build_true_stream_talker dereferences "
        "`model.model.talker` — if upstream renamed the attribute, the "
        "patch must be updated in lockstep before bumping the pin."
    )
    assert "Qwen3TTSTalkerForConditionalGeneration" in init_source, (
        "Qwen3TTSForConditionalGeneration.__init__ no longer constructs "
        "the talker via Qwen3TTSTalkerForConditionalGeneration. The class "
        "Story 16.8 depends on may have been replaced; verify the patch "
        "still targets the right type."
    )


def test_qwen3_tts_wrapper_calls_self_talker_generate_in_generate():
    """Story 16.8 — pin the call-site invariant our patch interposes on.

    services/qwen_tts_service.py:_build_true_stream_talker installs a
    streamer-injecting wrapper around `model.model.talker.generate` and
    expects the wrapper's `Qwen3TTSForConditionalGeneration.generate(...)`
    to invoke `self.talker.generate(...)` exactly once during preprocessing.
    If a future qwen-tts version refactors to call the talker through a
    different method (e.g., `self.talker.sample()`, `self.talker.forward()`,
    or hand-rolled token sampling), our patch never fires and the
    empty-chunks guard at qwen_tts_service.py:2845-2861 routes every
    TRUE_STREAM dispatch to fallback — silently regressing latency
    without any visible failure.

    This source-level inspection is the trip-wire: it fails CI when the
    call-site invariant breaks, before the silent regression can ship.
    See modeling_qwen3_tts.py:2272-2278 for the current call site.
    """
    import inspect

    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    generate_source = inspect.getsource(
        Qwen3TTSForConditionalGeneration.generate
    )
    assert "self.talker.generate(" in generate_source, (
        "Qwen3TTSForConditionalGeneration.generate no longer invokes "
        "`self.talker.generate(...)`. Story 16.8's talker-patch "
        "interposes on that exact call site — without it, the streamer "
        "kwarg is never injected and TRUE_STREAM silently falls back to "
        "SENTENCE_STREAM on every dispatch. Update "
        "_build_true_stream_talker (likely by switching to literal Path A "
        "preprocessing replication) before bumping the qwen-tts pin."
    )


# ============================================================================
# Story 20.5 extension — pin the 12Hz V2 decoder module chain
#
# ``services/tts_streaming/codec_state_cache.py`` does something no other part
# of MyVoice does: it re-walks ``Qwen3TTSTokenizerV2Decoder.forward``'s module
# traversal itself, calling the loaded submodules' inner ``nn.Conv1d`` /
# ``nn.ConvTranspose1d`` directly so it can thread per-session codec state
# through them. That buys the whole of Story 20.5 (edge loss 555 -> 0, chunk-
# head NRMSE ~130 % -> 0.8 %, lag jitter +/-1200 -> 0 samples) without
# vendoring a single upstream file.
#
# The price is that the traversal is a RESTATEMENT. If a pin bump reorders
# ``decoder`` / ``upsample``, renames an attribute, or inserts a new
# time-mixing module, the wrapper would walk a graph that no longer matches
# and produce subtly wrong audio at every chunk boundary — a failure that is
# inaudible as a bug and audible only as "the codec got worse". Two things
# stop that. ``codec_state_cache.probe_decoder`` refuses at runtime, falling
# back to the stateless decode; and these tests fail in CI first, which is the
# outcome we actually want.
#
# Note the division of labour: ``probe_decoder`` checks the loaded INSTANCE
# (strides, pads, layer types) and is exercised by
# ``tests/unit/services/tts_streaming/test_codec_state_cache.py`` against a
# real tiny decoder. The rows below pin the SYMBOLS and the source-level
# invariants, which is what a trip-wire can do without loading a model.
# ============================================================================


def _v2_module():
    from qwen_tts.core.tokenizer_12hz import (
        modeling_qwen3_tts_tokenizer_v2 as v2,
    )
    return v2


def test_qwen3_tts_tokenizer_v2_decoder_module_classes_are_importable():
    """Story 20.5 — pin every class ``codec_state_cache`` switches on.

    ``_run_module`` dispatches on these five types by isinstance. A rename
    would send a module down the "pointwise in time, safe to call as-is"
    fallthrough, which for a time-mixing module means silently decoding it
    from a cold state at every chunk boundary. ``probe_decoder`` catches that
    at runtime by refusing unknown leaves; this test catches it in CI.
    """
    v2 = _v2_module()
    expected = (
        "Qwen3TTSTokenizerV2Decoder",
        "Qwen3TTSTokenizerV2CausalConvNet",
        "Qwen3TTSTokenizerV2CausalTransConvNet",
        "Qwen3TTSTokenizerV2ConvNeXtBlock",
        "Qwen3TTSTokenizerV2DecoderDecoderBlock",
        "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
        "Qwen3TTSTokenizerV2DecoderTransformerModel",
        "SnakeBeta",
    )
    missing = [name for name in expected if not isinstance(getattr(v2, name, None), type)]
    assert not missing, (
        f"qwen-tts renamed or removed {missing}. "
        f"src/myvoice/services/tts_streaming/codec_state_cache.py dispatches "
        f"its streaming traversal on these classes by isinstance — update "
        f"_UpstreamTypes and re-run the Story 20.5 exactness suite "
        f"(tests/unit/services/tts_streaming/test_codec_state_cache.py) "
        f"before bumping the qwen-tts pin."
    )


def test_qwen3_tts_v2_decoder_forward_traversal_is_what_the_wrapper_mirrors():
    """Pin the traversal ``codec_state_cache.stream_forward`` restates.

    Source-level and therefore brittle (it asserts on string content), which
    is the point: the alternative is discovering the divergence as a chunk-
    boundary artefact in an audition. The five stages below are exactly the
    five ``stream_forward`` reproduces, in this order.
    """
    import inspect

    v2 = _v2_module()
    source = inspect.getsource(v2.Qwen3TTSTokenizerV2Decoder.forward)
    for fragment in (
        "self.quantizer.decode(codes)",
        "self.pre_conv(hidden).transpose(1, 2)",
        "self.pre_transformer(inputs_embeds=hidden)",
        "hidden.permute(0, 2, 1)",
        "for blocks in self.upsample",
        "for block in self.decoder",
        "clamp(min=-1, max=1)",
    ):
        assert fragment in source, (
            f"Qwen3TTSTokenizerV2Decoder.forward no longer contains "
            f"{fragment!r}. codec_state_cache.stream_forward is a restatement "
            f"of this method; the two have diverged and the streamed audio "
            f"will be wrong at every chunk boundary. Re-derive stream_forward "
            f"against the new source before bumping the pin."
        )


def test_causal_conv_still_left_pads_with_zeros():
    """Pin the premise of the entire story.

    Story 20.5 exists because this line puts ZEROS where the previous chunk's
    real audio should be. If upstream ever fixes it — or changes the padding
    formula — the wrapper's left-context substitution stops being the right
    correction and the story needs re-deriving, not just re-testing.
    """
    import inspect

    v2 = _v2_module()
    source = inspect.getsource(v2.Qwen3TTSTokenizerV2CausalConvNet.forward)
    assert 'mode="constant", value=0' in source and "F.pad(hidden_state" in source, (
        "Qwen3TTSTokenizerV2CausalConvNet.forward no longer zero-left-pads. "
        "That padding is the defect Story 20.5 removes; if upstream changed "
        "it, re-derive codec_state_cache._stream_conv from the new source."
    )
    init_source = inspect.getsource(v2.Qwen3TTSTokenizerV2CausalConvNet.__init__)
    assert "self.padding = self.kernel_size - self.stride" in init_source, (
        "the causal conv's padding width changed. codec_state_cache carries "
        "exactly `module.padding` samples of left context; that width is no "
        "longer derived the way _stream_conv assumes."
    )


def test_causal_trans_conv_still_discards_left_and_right_pad():
    """Pin the mechanism that owns 100 % of the 555-sample edge loss.

    Phase 1's ablation attributed the whole edge loss to this discard: every
    arm carrying transposed-conv state reached +0 samples, every arm without
    it stayed at -555 per seam. ``_stream_tconv`` overlap-adds the discarded
    right tail into the next chunk's head — and subtracts one copy of
    ``conv.bias`` from the overlap, because ConvTranspose1d adds its bias to
    every output position of each partial convolution.
    """
    import inspect

    v2 = _v2_module()
    source = inspect.getsource(v2.Qwen3TTSTokenizerV2CausalTransConvNet.forward)
    assert "self.left_pad : hidden_state.shape[-1] - self.right_pad" in source, (
        "Qwen3TTSTokenizerV2CausalTransConvNet.forward no longer trims "
        "[left_pad : -right_pad]. That trim IS the 555-sample edge loss "
        "codec_state_cache._stream_tconv recovers by overlap-add; re-derive "
        "it against the new source."
    )
    init_source = inspect.getsource(
        v2.Qwen3TTSTokenizerV2CausalTransConvNet.__init__
    )
    assert "nn.ConvTranspose1d" in init_source, (
        "the transposed conv is no longer an nn.ConvTranspose1d. The bias "
        "double-count correction in _stream_tconv is specific to "
        "ConvTranspose1d's bias semantics (it adds bias to every output "
        "position of each partial convolution) and must be re-derived."
    )


def test_v2_decoder_submodule_attribute_names_are_stable():
    """``stream_forward`` and ``probe_decoder`` dereference these by name."""
    import inspect

    v2 = _v2_module()
    init_source = inspect.getsource(v2.Qwen3TTSTokenizerV2Decoder.__init__)
    for attr in ("self.quantizer", "self.pre_conv", "self.pre_transformer",
                 "self.upsample", "self.decoder"):
        assert f"{attr} =" in init_source, (
            f"Qwen3TTSTokenizerV2Decoder.__init__ no longer assigns {attr}. "
            f"codec_state_cache walks the decoder through these five "
            f"attributes; probe_decoder will refuse and MyVoice will silently "
            f"drop back to the pre-20.5 stateless decode (audible as the "
            f"return of the chunk-boundary artefact Story 20.4 chased)."
        )

    for cls_name, attrs in (
        ("Qwen3TTSTokenizerV2ConvNeXtBlock",
         ("self.dwconv", "self.norm", "self.pwconv1", "self.act",
          "self.pwconv2", "self.gamma")),
        ("Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
         ("self.act1", "self.conv1", "self.act2", "self.conv2")),
        ("Qwen3TTSTokenizerV2DecoderDecoderBlock", ("self.block",)),
    ):
        source = inspect.getsource(getattr(v2, cls_name).__init__)
        for attr in attrs:
            assert f"{attr} =" in source, (
                f"{cls_name}.__init__ no longer assigns {attr}; "
                f"codec_state_cache._run_module reproduces this module's "
                f"forward body and dereferences it by name."
            )


def test_v2_decoder_transformer_layers_are_all_sliding_attention():
    """The KV cache Story 20.5 carries is bounded ONLY because every layer is
    sliding. A full-attention layer would make it grow without bound over a
    long utterance, voiding AC #2's <= 2.52 MiB per-session cost claim.

    ``probe_decoder`` refuses such a config at runtime; this fails first.
    """
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )

    config = Qwen3TTSTokenizerV2DecoderConfig()
    assert config.sliding_window == 72
    assert set(config.layer_types) == {"sliding_attention"}, (
        "Qwen3TTSTokenizerV2DecoderConfig.layer_types is no longer all "
        "sliding_attention. codec_state_cache's carried KV cache is bounded "
        "by sliding_window; a full-attention layer makes it unbounded."
    )


def test_v2_decoder_upsample_geometry_still_yields_1920_and_555():
    """The two constants ``streaming_decoder`` pins, re-derived from the
    config rather than trusted.

    ``codec_state_cache.build_stateful_decode_fn`` refuses to engage if the
    loaded decoder's geometry disagrees with these, so a change here silently
    disables Story 20.5 in production. Fail in CI instead.
    """
    import math

    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from myvoice.services.tts_streaming.streaming_decoder import (
        _CODEC_EDGE_LOSS_SAMPLES,
        _CODEC_SAMPLES_PER_FRAME,
    )

    config = Qwen3TTSTokenizerV2DecoderConfig()
    rates = tuple(config.upsample_rates)
    ratios = tuple(config.upsampling_ratios)

    assert math.prod(rates + ratios) == _CODEC_SAMPLES_PER_FRAME

    # Each kernel==2*stride transposed conv discards `stride` samples at its
    # own resolution; everything downstream multiplies that up.
    edge = 0
    for i, stride in enumerate(rates):
        edge += stride * math.prod(rates[i + 1:])
    assert edge == _CODEC_EDGE_LOSS_SAMPLES, (
        f"the decoder's upsample_rates now imply a {edge}-sample edge loss, "
        f"not {_CODEC_EDGE_LOSS_SAMPLES}. Both streaming_decoder's splice "
        f"arithmetic and codec_state_cache's length identity depend on this."
    )


def test_v2_model_decode_still_routes_to_chunked_decode_then_forward():
    """Pin the path the STATELESS fallback takes.

    ``_build_true_stream_decode_fn``'s fallback adapter calls
    ``speech_tokenizer.decode([...])`` -> ``Qwen3TTSTokenizerV2Model.decode``
    -> ``decoder.chunked_decode`` -> plain ``decoder.forward``. Phase 1
    established that this path never reaches ``forward_optimized``, which is
    why carrying state trades away no CUDA-graph speedup — Story 18.4's win
    comes from the talker and code-predictor compiles, not this one.
    """
    import inspect

    v2 = _v2_module()
    decode_source = inspect.getsource(v2.Qwen3TTSTokenizerV2Model.decode)
    assert "self.decoder.chunked_decode(" in decode_source, (
        "Qwen3TTSTokenizerV2Model.decode no longer routes through "
        "chunked_decode. The stateless fallback adapter's measured output "
        "geometry (1920*N - 555) may no longer hold."
    )
    chunked_source = inspect.getsource(v2.Qwen3TTSTokenizerV2Decoder.chunked_decode)
    assert "forward_optimized" not in chunked_source, (
        "chunked_decode now reaches forward_optimized. Story 20.5 Phase 1's "
        "finding that the compiled decoder graph is NOT on MyVoice's decode "
        "path no longer holds; re-audit the CUDA-graph interaction before "
        "trusting the AC #2 'no trade' conclusion."
    )


def test_transformers_dynamic_cache_snapshot_contract_is_intact():
    """Story 20.5 — pin the KV-cache surface ``CodecStreamState`` relies on.

    The snapshot/restore that holds the lookahead frames out of committed
    state is free ONLY because ``DynamicSlidingWindowLayer.update`` rebinds
    ``self.keys`` / ``self.values`` to new tensors rather than writing into
    the old ones. If a future transformers mutates in place, holding the old
    references stops being a valid snapshot and the state would silently
    advance past the splice — every chunk after the first would then skip
    ``lookahead`` frames of audio.
    """
    import inspect

    from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

    assert "config" in inspect.signature(DynamicCache.__init__).parameters, (
        "DynamicCache no longer accepts a `config` kwarg; "
        "codec_state_cache._stream_transformer builds the cache that way so "
        "the sliding-window layer types come from the model's own config."
    )

    update_source = inspect.getsource(DynamicSlidingWindowLayer.update)
    assert "self.keys = full_key_states[" in update_source, (
        "DynamicSlidingWindowLayer.update no longer REBINDS self.keys. "
        "CodecStreamState.snapshot() holds the previous tensors by reference "
        "and assumes they are never mutated in place; if that changed, the "
        "snapshot must start cloning or the lookahead frames will be "
        "committed to state and every seam will skip audio."
    )
    for attr in ("keys", "values", "cumulative_length"):
        assert attr in update_source or hasattr(DynamicSlidingWindowLayer, attr), (
            f"DynamicSlidingWindowLayer no longer exposes {attr!r}; "
            f"CodecStreamState.snapshot/restore reads it by name."
        )
