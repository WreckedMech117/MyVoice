"""Story 16.1 / D-12 — pin the qwen-tts attribute surface against silent upstream renames.

These three tests assert that every symbol MyVoice imports from `qwen_tts` and every
method MyVoice calls on a `Qwen3TTSModel` instance is present on the pinned install.
If any test fails, MyVoice's runtime calls into qwen-tts will break in production.
Before bumping the pinned commit hash in requirements.txt + build_tools/requirements-production.txt,
run this file. Failures name the missing symbols; fix the call sites in
src/myvoice/services/qwen_tts_service.py and src/myvoice/services/model_registry.py
before merging the bump.
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
