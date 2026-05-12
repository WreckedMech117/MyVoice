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
