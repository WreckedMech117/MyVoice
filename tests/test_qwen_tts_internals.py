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
    """Pin every callable MyVoice invokes on a Qwen3TTSModel instance."""
    from qwen_tts import Qwen3TTSModel
    expected_methods = (
        "from_pretrained",            # classmethod — services/model_registry.py:459
        "create_voice_clone_prompt",  # services/qwen_tts_service.py:1285, 1296
        "generate_voice_clone",       # services/qwen_tts_service.py:2606, 2632
        "generate_voice_design",      # services/qwen_tts_service.py:2596
        "generate_custom_voice",      # services/qwen_tts_service.py:2588
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
