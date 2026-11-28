"""Reference codec codes for the Lightning engine.

Accent-matching (ICL) generation needs the reference clip as codec codes. The
standard engine already computes and persists them: every cloned voice's
``<voice>.<tier>.pt`` prompt cache holds ``ref_code``, and the codec is shared
by both quality tiers. Handing those to qwen3-tts.cpp means its own codec
encoder never loads (~200 MB of VRAM, ~0.4 s per line) and the codes are the
exact ones the standard engine uses.

A voice with no ``.pt`` cache yet simply gets no code file; the DLL then encodes
the reference itself.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CODEBOOKS = 16
_TIERS = ("small", "quality")


def _cache_dir() -> Path:
    from myvoice.utils.portable_paths import get_data_root

    path = get_data_root() / "cache" / "lightning_ref_codes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prompt_cache_for(ref_audio: Path) -> Optional[Path]:
    for tier in _TIERS:
        candidate = ref_audio.with_name(f"{ref_audio.stem}.{tier}.pt")
        if candidate.is_file():
            return candidate
    return None


def _code_file_path(ref_audio: Path, prompt_cache: Path, cache_dir: Path) -> Path:
    # Keyed on both files' identity, so replacing the clip or regenerating the
    # prompt cache produces a new code file instead of reusing a stale one.
    audio_stat, cache_stat = ref_audio.stat(), prompt_cache.stat()
    key = "|".join(
        str(part)
        for part in (
            ref_audio.resolve(), audio_stat.st_mtime_ns, audio_stat.st_size,
            prompt_cache.name, cache_stat.st_mtime_ns, cache_stat.st_size,
        )
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.txt"


def write_code_file(ref_code, destination: Path) -> None:
    """Write codes in the text form qwen3-tts.cpp reads: one frame per line,
    16 whitespace-separated integers."""
    frames = ref_code.reshape(-1, _CODEBOOKS).tolist()
    if not frames:
        raise ValueError("reference codes are empty")
    text = "\n".join(" ".join(str(int(v)) for v in frame) for frame in frames)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(text, encoding="ascii")
    temporary.replace(destination)


def ensure_ref_code_file(
    ref_audio: Path, cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """Return a code file for ``ref_audio``, creating it from the voice's
    prompt cache if needed. ``None`` when the voice has no usable cache —
    never raises, because the engine has a working fallback.
    """
    try:
        prompt_cache = _prompt_cache_for(ref_audio)
        if prompt_cache is None:
            return None
        destination = _code_file_path(ref_audio, prompt_cache, cache_dir or _cache_dir())
        if destination.is_file():
            return destination

        import torch

        # weights_only=False: the cache is a pickled VoiceClonePromptItem
        # written by this app (same call as qwen_tts_service's reload).
        loaded = torch.load(str(prompt_cache), map_location="cpu", weights_only=False)
        item = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
        ref_code = (
            item.get("ref_code") if isinstance(item, dict) else getattr(item, "ref_code", None)
        )
        if ref_code is None:
            return None
        write_code_file(ref_code.detach().cpu(), destination)
        logger.info("Lightning reference codes written for %s", ref_audio.name)
        return destination
    except Exception as exc:
        logger.warning(
            "Lightning reference codes unavailable for %s (%s); the engine will "
            "encode the reference itself", ref_audio.name, exc,
        )
        return None
