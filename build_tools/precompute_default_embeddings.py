"""Pre-generate voice_clone_prompt .pt files for the bundled default voices.

Mirrors the on-disk cache schema written by
``QwenTTSService._ensure_voice_clone_prompt_for_voice`` (qwen_tts_service.py
:1604-1720). The runtime's startup hydration
(``hydrate_voice_clone_prompt_cache``, qwen_tts_service.py:1722) picks up any
valid ``<stem>.<tier>.pt`` + ``<stem>.<tier>.pt.meta.json`` pair found next to
a CLONED voice's .wav and skips the first-use Base-model precompute step,
which is the latency the end user sees on their very first generation with a
default voice.

Run this once before ``build_release.bat`` so the generated artifacts ship
inside both install paths:

  - Inno installer: ``src/install_files/default_voices/*`` -> ``{app}/voice_files/``
    (installer.iss:127)
  - PyInstaller bundle: ``voice_files/`` (project root) -> ``_internal/voice_files/``
    -> copied to ``{app}/voice_files/`` on first run by ``_copy_bundled_voice_files``
    (portable_paths.py:210)

The .pt payload is identical across the two dirs; only the meta JSON differs
(its ``ref_audio_mtime`` / ``ref_audio_size`` / ``txt_mtime`` fields are
local-fingerprint values pulled from each directory's copy of the .wav and
.txt). Inno and shutil.copy2 both preserve source mtimes, so the meta written
here survives the install and validates at first launch.

Usage::

    python build_tools/precompute_default_embeddings.py
    python build_tools/precompute_default_embeddings.py --tier quality
    python build_tools/precompute_default_embeddings.py --target voice_files
    python build_tools/precompute_default_embeddings.py --force --verbose
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Force HuggingFace Hub into offline mode for this build-time script. The
# bundled portable python310 lacks a working CA cert bundle, so `from_pretrained`
# fails its HEAD freshness-check against huggingface.co with
# SSLCertVerificationError even when the model is already cached on disk.
# Setting these env vars BEFORE qwen_tts / transformers / huggingface_hub get
# imported makes them skip the network round-trip and read straight from
# `~/.cache/huggingface/hub/`. Models must be pre-cached on the build host;
# `build_release.bat`'s [Bundle Prerequisites] probes already enforce that.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Windows torch-before-everything-else DLL preamble. Same invariant enforced
# by ``src/myvoice/main.py`` and ``tests/conftest.py`` (see
# memory/torch_pyqt6_dll_ordering.md). Required before any module that
# transitively imports torch CUDA on Win11.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _repo_root = Path(__file__).resolve().parent.parent
    _torch_lib = _repo_root / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

import torch  # noqa: E402  — must follow the DLL preamble

# Pull the pin hash from the runtime so a future bump in qwen_tts_service.py
# automatically invalidates artifacts written by an older precompute run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from myvoice.services.qwen_tts_service import QwenTTSService  # noqa: E402

PIN_HASH: str = QwenTTSService._QWEN_TTS_PIN_HASH
SCHEMA_VERSION: str = "1.1"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS: Tuple[Path, ...] = (
    REPO_ROOT / "src" / "install_files" / "default_voices",
    REPO_ROOT / "voice_files",
)

# Tier strings match ``ModelQualityTier.value`` (service_enums.py:42-43) and
# the on-disk cache filename suffix produced by QwenTTSService.
TIERS: Tuple[str, ...] = ("quality", "small")
MODEL_ID_BY_TIER: Dict[str, str] = {
    "quality": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "small": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

logger = logging.getLogger("precompute_default_embeddings")


def _persist_paths(wav: Path, tier: str) -> Tuple[Path, Path]:
    """Mirror ``QwenTTSService._voice_clone_prompt_persist_paths`` (line 1492)."""
    pt = wav.with_name(f"{wav.stem}.{tier}.pt")
    meta = wav.with_name(f"{wav.stem}.{tier}.pt.meta.json")
    return pt, meta


def _txt_sidecar_mtime(wav: Path) -> Optional[float]:
    """Mirror ``QwenTTSService._txt_sidecar_mtime`` (line 1175)."""
    txt = wav.with_suffix(".txt")
    try:
        return txt.stat().st_mtime if txt.exists() else None
    except OSError:
        return None


def _read_transcription(wav: Path) -> str:
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        raise FileNotFoundError(
            f"Missing transcription sidecar for {wav.name}: expected {txt}"
        )
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty transcription sidecar: {txt}")
    return text


def _meta_valid(meta_path: Path, wav: Path, tier: str) -> bool:
    """Read-only mirror of ``_voice_clone_prompt_meta_is_valid`` (line 1506).

    Used for idempotent skipping: if a previous run (or normal app usage)
    already wrote a valid meta for this (wav, tier), don't recompute.
    """
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        stat = wav.stat()
    except FileNotFoundError:
        return False
    if meta.get("tier") != tier:
        return False
    if meta.get("qwen_tts_pin") != PIN_HASH:
        return False
    meta_mtime = meta.get("ref_audio_mtime")
    if not isinstance(meta_mtime, (int, float)):
        return False
    if abs(float(meta_mtime) - stat.st_mtime) > 1e-3:
        return False
    if meta.get("ref_audio_size") != stat.st_size:
        return False
    meta_txt = meta.get("txt_mtime")
    curr_txt = _txt_sidecar_mtime(wav)
    if meta_txt is None and curr_txt is None:
        return True
    if meta_txt is None or curr_txt is None:
        return False
    return abs(float(meta_txt) - float(curr_txt)) <= 1e-3


def _move_prompt_tensors_to_cpu(prompt) -> None:
    """Mirror qwen_tts_service.py:1650-1661 — required before persisting so a
    CUDA-trained tensor doesn't get pickled with a device tag that fails to
    reload on a CPU-only end-user machine.
    """
    try:
        if getattr(prompt, "ref_code", None) is not None:
            prompt.ref_code = prompt.ref_code.cpu()
        if getattr(prompt, "ref_spk_embedding", None) is not None:
            prompt.ref_spk_embedding = prompt.ref_spk_embedding.cpu()
    except Exception as exc:
        logger.warning(f"CPU-move on prompt tensors failed: {exc}")


def _save_prompt(prompt, wav: Path, tier: str) -> None:
    """Persist (pt, meta) atomically, then verify by reloading.

    Mirrors the persist + verify discipline at qwen_tts_service.py:1688-1714.
    On verification failure both files are deleted so a subsequent run starts
    clean.
    """
    pt_path, meta_path = _persist_paths(wav, tier)
    stat = wav.stat()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "ref_audio_mtime": stat.st_mtime,
        "ref_audio_size": stat.st_size,
        "txt_mtime": _txt_sidecar_mtime(wav),
        "tier": tier,
        "qwen_tts_pin": PIN_HASH,
    }
    torch.save(prompt, str(pt_path))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    try:
        reloaded = torch.load(
            str(pt_path), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        pt_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Verification reload of {pt_path} failed: {exc}"
        ) from exc
    if getattr(reloaded, "ref_spk_embedding", None) is None:
        pt_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Verification reload produced empty embedding for {pt_path}"
        )
    pt_size_mb = pt_path.stat().st_size / (1024 * 1024)
    logger.info(f"  wrote {pt_path.name} ({pt_size_mb:.2f} MB) + meta JSON")


def _enumerate_wavs(target_dirs: Iterable[Path]) -> Dict[Path, List[Path]]:
    out: Dict[Path, List[Path]] = {}
    for d in target_dirs:
        d = d.resolve()
        if not d.exists():
            logger.warning(f"target dir missing, skipping: {d}")
            continue
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            out[d] = wavs
            logger.info(f"  {d}: {len(wavs)} .wav file(s)")
        else:
            logger.warning(f"target dir has no .wav files: {d}")
    return out


def _process_tier(
    tier: str,
    wavs_by_dir: Dict[Path, List[Path]],
    force: bool,
    normalizer: "QwenTTSService",
) -> Tuple[int, int, int]:
    """Load the Base model at ``tier`` once, iterate all (dir, wav) pairs.

    ``normalizer`` is a no-model-loaded QwenTTSService used only to call its
    ``_normalize_voice_clone_prompt`` method — required because the pinned
    qwen_tts fork returns ``create_voice_clone_prompt()`` as a list/tuple
    rather than a VoiceClonePromptItem object, and the runtime expects the
    library's native class on reload (qwen_tts_service.py:2508-2665).

    Returns (generated, skipped_cached, failed).
    """
    from qwen_tts import Qwen3TTSModel  # local import — heavy

    work: List[Tuple[Path, Path, str]] = []  # (dir, wav, ref_text)
    skipped = 0
    for target_dir, wavs in wavs_by_dir.items():
        for wav in wavs:
            _pt, meta = _persist_paths(wav, tier)
            if not force and _meta_valid(meta, wav, tier):
                logger.info(
                    f"[skip] {target_dir.name}/{wav.name} @ {tier}: "
                    f"meta already valid"
                )
                skipped += 1
                continue
            try:
                ref_text = _read_transcription(wav)
            except (FileNotFoundError, ValueError) as exc:
                logger.error(f"[fail] {wav}: {exc}")
                continue
            work.append((target_dir, wav, ref_text))

    if not work:
        logger.info(
            f"[tier={tier}] nothing to compute ({skipped} already cached)"
        )
        return (0, skipped, 0)

    model_id = MODEL_ID_BY_TIER[tier]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Resolve the local snapshot path to bypass transformers' deeper
    # network calls (e.g. tokenization_utils_base._patch_mistral_regex ->
    # is_base_mistral -> model_info()) which ignore local_files_only and
    # fail under HF_HUB_OFFLINE. Passing a filesystem path triggers
    # transformers' `_is_local` short-circuit and skips that lookup.
    from huggingface_hub import snapshot_download
    local_snapshot = snapshot_download(
        model_id,
        local_files_only=True,
    )
    logger.info(
        f"[tier={tier}] loading {model_id} (cached at {local_snapshot}) "
        f"on {device} for {len(work)} voice(s)..."
    )
    t_load = time.time()
    model = Qwen3TTSModel.from_pretrained(
        local_snapshot,
        device_map=device,
        torch_dtype=dtype,
        local_files_only=True,
    )
    logger.info(
        f"[tier={tier}] model loaded in {time.time() - t_load:.1f}s"
    )

    generated = 0
    failed = 0
    try:
        for target_dir, wav, ref_text in work:
            t0 = time.time()
            logger.info(
                f"[tier={tier}] computing {target_dir.name}/{wav.name}"
            )
            try:
                raw_prompt = model.create_voice_clone_prompt(
                    ref_audio=str(wav),
                    ref_text=ref_text,
                )
                # Normalize via the runtime's helper. The pinned qwen_tts fork
                # returns a list/tuple — the runtime expects the library's
                # VoiceClonePromptItem on reload (qwen_tts_service.py:2580).
                prompt = normalizer._normalize_voice_clone_prompt(raw_prompt)
                _move_prompt_tensors_to_cpu(prompt)
                _save_prompt(prompt, wav, tier)
                generated += 1
                logger.info(f"  done in {time.time() - t0:.1f}s")
            except Exception as exc:
                logger.exception(
                    f"[fail] {target_dir.name}/{wav.name} @ {tier}: {exc}"
                )
                failed += 1
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    return (generated, skipped, failed)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target",
        type=Path,
        action="append",
        metavar="DIR",
        help=(
            "Target directory to scan (default: both install-source dirs: "
            "src/install_files/default_voices AND voice_files). Pass multiple "
            "times for multiple dirs."
        ),
    )
    p.add_argument(
        "--tier",
        choices=("quality", "small", "all"),
        default="all",
        help="Tier(s) to compute (default: all).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a valid meta already exists.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    targets = list(args.target) if args.target else list(DEFAULT_TARGETS)
    logger.info(f"qwen_tts pin: {PIN_HASH}")
    logger.info(f"Targets: {[str(t) for t in targets]}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning(
            "CUDA not available — running on CPU (much slower). "
            "Embedding computation on CPU may take many minutes per voice."
        )

    wavs_by_dir = _enumerate_wavs(targets)
    if not wavs_by_dir:
        logger.error("No .wav files found in any target directory.")
        return 1

    # No-model-loaded service used only as a vehicle for
    # _normalize_voice_clone_prompt. The constructor builds a ModelRegistry
    # but does not load weights (per the unit-test docstring at
    # qwen_tts_service.py:1287; mirrors tests/unit/services/test_voice_clone_prompt_cache.py).
    normalizer = QwenTTSService(device="cpu", dtype="float32")

    tiers = TIERS if args.tier == "all" else (args.tier,)
    total_gen = 0
    total_skip = 0
    total_fail = 0
    t_start = time.time()
    for tier in tiers:
        gen, skip, fail = _process_tier(tier, wavs_by_dir, args.force, normalizer)
        total_gen += gen
        total_skip += skip
        total_fail += fail

    elapsed = time.time() - t_start
    total_wavs = sum(len(v) for v in wavs_by_dir.values())
    logger.info(
        f"Done in {elapsed:.1f}s. Generated {total_gen}, skipped "
        f"{total_skip} (cached), failed {total_fail}, across {total_wavs} "
        f"voice file(s) x {len(tiers)} tier(s)."
    )
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
