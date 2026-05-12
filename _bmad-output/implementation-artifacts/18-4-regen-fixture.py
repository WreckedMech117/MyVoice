"""Story 18.4 Task 9.2 — regenerate the 20-WAV joint audition fixture.

Drives ``Qwen3TTSModel.generate_voice_clone`` directly (bypasses MyVoice's
UI + AudioCoordinator) to produce the 10 paired WAVs at
``_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/``:

  - A rendition: ``{utt}-fp32_eager.wav`` (tts_precision=fp32, tts_compile=off)
  - B rendition: ``{utt}-bf16_compile.wav`` (tts_precision=bf16, tts_compile=on)

Both renditions use the canonical Sarira-F voice (the same voice every
prior audition used). The precomputed voice_clone_prompt at
``voice_files/Sarira-F.quality.pt`` (Story 17.2 cache; pin 3fdb4682) is
loaded directly — no Whisper transcription, no cold-precompute.

Usage:
    python310/python.exe _bmad-output/implementation-artifacts/18-4-regen-fixture.py [A|B|both]

Defaults to ``both``. Loads the model twice (once per dtype) since dtype
must be set at from_pretrained time.

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FIXTURE_DIR = SCRIPT_DIR / "18-4-perceptual-fixtures"
VOICE_FILES_DIR = REPO_ROOT / "voice_files"
SARIRA_F_PROMPT_PATH = VOICE_FILES_DIR / "Sarira-F.quality.pt"


UTTERANCES: dict[str, str] = {
    "s-014": "Bit, bat, bot, but, bet.",
    "s-015": "Tap, tip, top, tup, tep.",
    "s-016": "Sip ship sap shop sup.",
    "s-017": "Six sticks split.",
    "m-011": "She sells seashells by the seashore.",
    "m-012": "Sister Susie sews shirts for soldiers on Sunday.",
    "m-013": "The bell rang clear at noon and echoed across the field.",
    "m-014": "A sharp whistle pierced the silence over the still water.",
    "l-013": (
        "She sells seashells by the seashore, and the shells she sells are "
        "surely seashells, so if she sells shells on the seashore, I'm sure "
        "she sells seashore shells."
    ),
    "l-014": (
        "Six slick slim sycamore saplings stood swaying silently as the soft "
        "summer storm slowly swirled across the steep slopes south of the "
        "silver stream below us."
    ),
}


# Add src/ to sys.path so we can import myvoice modules for compile engagement.
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_voice_clone_prompt(model_dtype) -> list:
    """Load the precomputed Sarira-F voice_clone_prompt from disk, normalize
    to the qwen-tts library's ``VoiceClonePromptItem``, and wrap in the list
    form the ``generate_voice_clone(voice_clone_prompt=...)`` API expects.

    The cached .pt holds MyVoice's wrapper ``VoiceClonePromptItem`` (per
    ``qwen_tts_service.py:178`); the library's ``Qwen3TTSModel.generate_voice_clone``
    expects ``List[qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem]``
    that gets dict-folded via ``_prompt_items_to_voice_clone_prompt`` internally.
    This function mirrors ``QwenTTSService._normalize_voice_clone_prompt``
    (qwen_tts_service.py:2483) inline so the regen script stays
    self-contained.
    """
    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem as LibraryItem
    if not SARIRA_F_PROMPT_PATH.exists():
        raise SystemExit(
            f"FATAL: Sarira-F voice_clone_prompt not found at {SARIRA_F_PROMPT_PATH}.\n"
            "Regenerate via the production GUI (open MyVoice with Sarira-F selected,\n"
            "generate any short utterance — the lazy precompute will populate the .pt)."
        )
    # weights_only=False required for VoiceClonePromptItem deserialization per
    # qwen_tts_service.py:1589's comment.
    raw = torch.load(SARIRA_F_PROMPT_PATH, weights_only=False, map_location="cpu")
    # Read the ref text from Sarira-F.txt — the cache may not carry it.
    sarira_f_txt_path = VOICE_FILES_DIR / "Sarira-F.txt"
    ref_text_from_disk = (
        sarira_f_txt_path.read_text(encoding="utf-8").strip()
        if sarira_f_txt_path.exists() else None
    )
    ref_text = getattr(raw, "ref_text", None) or ref_text_from_disk
    icl_mode = getattr(raw, "icl_mode", True)
    # If ref_text is missing, the library will crash inside the talker's
    # tokenize step — fall back to x-vector-only mode.
    if not ref_text:
        icl_mode = False
    normalized = LibraryItem(
        ref_code=getattr(raw, "ref_code", None),
        ref_spk_embedding=getattr(raw, "ref_spk_embedding", None),
        x_vector_only_mode=getattr(raw, "x_vector_only_mode", False),
        icl_mode=icl_mode,
        ref_text=ref_text,
    )
    print(
        f"  Loaded voice_clone_prompt from {SARIRA_F_PROMPT_PATH.name} "
        f"(normalized; ref_text {'present' if ref_text else 'missing'}, "
        f"icl_mode={icl_mode})"
    )
    # The library API expects a LIST (List[VoiceClonePromptItem]); a
    # single item must be wrapped.
    return [normalized]


def _load_model(dtype_str: str):
    """Load Qwen3TTSModel with the given dtype on cuda:0."""
    import torch
    from qwen_tts import Qwen3TTSModel
    dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
    print(f"  Loading Qwen3TTSModel (dtype={dtype}, device=cuda:0)...")
    t0 = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        torch_dtype=dtype,
    )
    print(f"  Model loaded in {(time.monotonic() - t0)*1000:.0f}ms")
    return model


def _engage_compile_if_needed(model, branch_name: str) -> None:
    """For Branch B (bf16+compile), engage compile via the production function.
    For Branch A (fp32+eager), do nothing — eager-mode is the baseline."""
    if branch_name != "B":
        return
    print("  Engaging torch.compile via production engage_compile_optimizations...")

    class _Settings:
        tts_compile = "auto"

    from myvoice.services.tts_streaming import engage_compile_optimizations
    from myvoice.services.qwen_tts_service import QwenTTSService

    t0 = time.monotonic()
    result = engage_compile_optimizations(
        model,
        app_settings=_Settings(),
        qwen_tts_pin_hash=QwenTTSService._QWEN_TTS_PIN_HASH,
    )
    print(
        f"  Compile engagement returned in {(time.monotonic() - t0)*1000:.0f}ms: "
        f"engaged={result['engaged']}, reason={result['reason']}"
    )
    if not result["engaged"]:
        raise SystemExit(
            f"FATAL: compile engagement failed for Branch B: reason={result['reason']!r}. "
            "Cannot proceed — the B rendition requires actual compile engagement."
        )


def _save_wav(audio_array, sample_rate: int, path: Path) -> None:
    """Save audio array as 16-bit PCM WAV."""
    import numpy as np
    import soundfile as sf
    # The qwen-tts API may return audio as a 1D float array or a list of
    # per-chunk arrays — normalize defensively.
    if isinstance(audio_array, (list, tuple)):
        audio_array = np.concatenate([np.asarray(a).flatten() for a in audio_array])
    audio_array = np.asarray(audio_array).flatten().astype("float32")
    # Clip to [-1, 1] and convert to int16 PCM.
    audio_int16 = (audio_array.clip(-1.0, 1.0) * 32767).astype("int16")
    sf.write(str(path), audio_int16, sample_rate, subtype="PCM_16")


def _generate_branch(branch_name: str, dtype_str: str, mode_suffix: str) -> None:
    """Generate the 10 WAVs for one branch."""
    print()
    print("=" * 70)
    print(f"Branch {branch_name}: dtype={dtype_str}, mode_suffix={mode_suffix}")
    print("=" * 70)

    model = _load_model(dtype_str)
    _engage_compile_if_needed(model, branch_name)

    # Load voice_clone_prompt AFTER compile engagement so the prompt's
    # tensors get migrated correctly if needed.
    import torch
    voice_clone_prompt = _load_voice_clone_prompt(model_dtype=None)

    # Generate the 10 WAVs.
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(UTTERANCES)
    branch_t0 = time.monotonic()
    for idx, (utt_id, text) in enumerate(UTTERANCES.items(), start=1):
        out_path = FIXTURE_DIR / f"{utt_id}-{mode_suffix}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{idx:2d}/{total}] {utt_id}: SKIP (already exists at {out_path.name})")
            continue
        print(f"  [{idx:2d}/{total}] {utt_id}: generating ...", end=" ", flush=True)
        t0 = time.monotonic()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
        )
        gen_ms = (time.monotonic() - t0) * 1000
        _save_wav(wavs, sr, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"OK ({gen_ms:.0f}ms gen, {size_kb:.0f}KB) -> {out_path.name}")

    print(
        f"\nBranch {branch_name} complete in "
        f"{(time.monotonic() - branch_t0)/60:.1f}min"
    )

    # Free GPU memory before next branch's model load.
    del model
    torch.cuda.empty_cache()


def main(branches: str) -> int:
    print(f"CUDA_PATH = {os.environ.get('CUDA_PATH', '(unset)')}")
    print(f"Fixture dir = {FIXTURE_DIR}")
    print(f"Voice prompt = {SARIRA_F_PROMPT_PATH}")
    print()
    overall_t0 = time.monotonic()

    if branches in ("A", "both"):
        _generate_branch("A", dtype_str="fp32", mode_suffix="fp32_eager")
    if branches in ("B", "both"):
        _generate_branch("B", dtype_str="bf16", mode_suffix="bf16_compile")

    print()
    print("=" * 70)
    print(
        f"All requested branches complete in "
        f"{(time.monotonic() - overall_t0)/60:.1f}min"
    )
    print("=" * 70)
    print("Next: run the truth-table generator:")
    print(
        "  python310/python.exe "
        "_bmad-output/implementation-artifacts/18-4-generate-truthtable.py"
    )
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) >= 2 else "both"
    if arg not in ("A", "B", "both"):
        print(f"Usage: python {sys.argv[0]} [A|B|both]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(arg))
