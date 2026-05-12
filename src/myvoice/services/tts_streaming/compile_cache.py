"""Persistent torch.compile cache helpers — Story 18.4 / D-23 + D-24.

Phase ⊥-Polish-2 of D-20 (architecture-optimization-pass.md), Cluster E of
architecture-streaming-acceleration-and-lightning-tier.md (sealed 2026-05-10).

This module exposes the seven-dimension cache-key helper that gates the
torch.compile-engaged TRUE_STREAM pipeline. It mirrors Story 18.2's
``torch_runtime.py`` discipline:

  * Lazy stdlib imports at function-call time — no top-level ``import torch``
    so ``monkeypatch.setattr("os.environ", ...)`` and ``sys.platform``
    monkeypatches are honored by tests without import-order gymnastics.
  * Pure-decision compute (``compute_key``): no logging, no metric emission,
    no filesystem touch — caller drives I/O.
  * Side-effecting helpers (``cache_root``, ``cache_dir_for_key``, ``mark_warm``,
    ``set_torchinductor_cache_dir``) are individually idempotent.
  * No peer imports from ``myvoice.services.*`` / ``myvoice.tts_engines.*`` /
    ``myvoice.ui.*`` per architecture's module-boundary rule (line 1157-1166):
    "compile_cache.py may import: hashlib, pathlib, torch (cuda capability
    probe), stdlib; may NOT import services.*, tts_engines.*, ui.*".

Architecture references:
  * D-23 — background warmup at app start + persistent compile cache; the
    cache-cold path runs one priming generation and writes ``meta.json`` on
    success; cache-warm path short-circuits.
  * D-24 — seven cache-key dimensions:
        (qwen_tts_pin_hash, model_id, precision_str, torch_version,
         decode_window_frames, cuda_capability, compile_mode)
    Joined by ``"|"`` and SHA-256-hexed. Constructed in EXACTLY one place
    (this module's ``compute_key``); P-10 single-helper enforcement.
  * D-25 — decode-window invariant assertion at startup (lives in
    ``torch_runtime.engage_compile_optimizations``, NOT here — this module's
    job is purely the cache surface).
  * P-10 — single-helper cache key (any inline key construction at the call
    site is a P-10 violation).
  * P-11 — invariant assertions at startup (lives in ``torch_runtime``).

Storage discipline mirrors Story 17.2's ``_voice_clone_prompt_persist_paths``:
  * Per-key directory ``cache_root() / key[:16]`` (16-char prefix is collision-
    resistant for a single user's cache; full key lands in ``meta.json``).
  * ``is_warm()`` validates the full-key match in ``meta.json`` — defensive
    against directory-prefix collisions (the H1+H2 cache-invalidation
    discipline from ``memory/code_review_regression_test_exact_class.md``).
  * Windows path: ``%LOCALAPPDATA%/MyVoice/torch_compile_cache/`` (V2 baseline
    storage discipline per ``architecture.md``).
  * Non-Windows fallback: ``~/.cache/MyVoice/torch_compile_cache/``.

Public surface:
  * ``compute_key(*, qwen_tts_pin_hash, model_id, precision_str, torch_version,
    decode_window_frames, cuda_capability, compile_mode) -> str``
  * ``cache_root() -> Path``
  * ``cache_dir_for_key(key: str) -> Path``
  * ``is_warm(key: str) -> bool``
  * ``mark_warm(key: str) -> None``
  * ``set_torchinductor_cache_dir(key: str) -> None``

Caller contract (architecture D-23):
  * Cache-warm path → caller short-circuits the priming generation.
  * Cache-cold path → caller runs one priming generation; on success calls
    ``mark_warm(key)``; on failure leaves the cache cold (next run retries).
  * ``set_torchinductor_cache_dir(key)`` is invoked BEFORE the
    ``model.enable_streaming_optimizations(...)`` call so PyTorch's inductor
    cache reads/writes from the per-key directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


_logger = logging.getLogger(__name__)

# Architecture D-24 — the seven cache-key dimensions, ordered for the
# canonical join. Changing the order is a backwards-incompatible cache
# invalidation (acceptable on a pin bump; not silently changeable).
_KEY_DIMENSIONS = (
    "qwen_tts_pin_hash",
    "model_id",
    "precision_str",
    "torch_version",
    "decode_window_frames",
    "cuda_capability",
    "compile_mode",
)

# The per-key directory uses the 16-char prefix (architecture D-24
# implementation note). 16 hex chars = 64 bits of entropy → collision-
# resistant for a single user's cache (single-user lifetime < 2^32 entries
# in any realistic scenario). The full key in meta.json is the canonical
# warm signal (defends against the residual collision case).
_KEY_DIR_PREFIX_CHARS = 16

# Set-once guard for the single INFO log emitted by
# ``set_torchinductor_cache_dir``. Mirrors the Story 18.2 idempotency
# pattern — log noise is bounded per process even when the function is
# called multiple times (e.g., during test isolation).
_TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE = False


def compute_key(
    *,
    qwen_tts_pin_hash: str,
    model_id: str,
    precision_str: str,
    torch_version: str,
    decode_window_frames: int,
    cuda_capability: Tuple[int, int],
    compile_mode: str,
) -> str:
    """Pure-decision: compute the seven-dimension compile-cache key.

    Returns the SHA-256 hex digest of the seven dimensions joined by ``"|"``.
    Architecture D-24 mandates this exact construction; P-10 enforces the
    single-helper discipline (any call-site inline construction is a
    violation).

    The function is **side-effect-free**: no logging, no metric emission, no
    filesystem touch. Mirrors Story 18.2's ``is_ampere_or_newer`` and Story
    18.3's ``resolve_tts_precision`` pure-decision pattern. Caller drives
    everything observable.

    Args:
        qwen_tts_pin_hash: First 8 hex chars of the active qwen-tts pin
            (mirrors ``QwenTTSService._QWEN_TTS_PIN_HASH``).
        model_id: HuggingFace model identifier (e.g., ``"Qwen/Qwen3-TTS-12Hz-1.7B-Base"``).
        precision_str: One of ``"bf16"`` or ``"fp32"`` (the resolved dtype
            string; not the raw ``torch.dtype`` to keep the key stringly-
            typed and round-trippable through ``meta.json``).
        torch_version: ``torch.__version__`` (typically ``"2.10.0+cu128"``).
            Compile artifacts are wedded to the torch version; a torch
            upgrade must invalidate the cache.
        decode_window_frames: Architecture D-21's fixed value (today 30).
            A future streamer re-tune would change this dimension and
            auto-invalidate the cache.
        cuda_capability: ``torch.cuda.get_device_capability()`` tuple
            (e.g., ``(12, 0)`` for RTX 5090). Compile artifacts target a
            specific compute capability; cross-host caching is forbidden.
        compile_mode: torch.compile mode (typically ``"reduce-overhead"``).

    Returns:
        SHA-256 hex digest (64 lowercase hex chars).
    """
    parts = [
        str(qwen_tts_pin_hash),
        str(model_id),
        str(precision_str),
        str(torch_version),
        str(decode_window_frames),
        f"{cuda_capability[0]}.{cuda_capability[1]}",
        str(compile_mode),
    ]
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def cache_root() -> Path:
    """Return (and create) the torch-compile-cache root directory.

    Windows: ``%LOCALAPPDATA%/MyVoice/torch_compile_cache/``. Falls back to
    ``Path.home() / "AppData" / "Local" / "MyVoice" / "torch_compile_cache"``
    if ``LOCALAPPDATA`` is unset (defensive; the env var is normally always
    present on Windows since XP).

    Non-Windows: ``~/.cache/MyVoice/torch_compile_cache/`` (XDG-aligned;
    matches Story 17.2's storage-path discipline for non-Windows hosts).

    Idempotent — creates the directory if absent, leaves it intact if
    present. Returns the ``Path`` either way.
    """
    if sys.platform == "win32":
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            root = Path(localappdata) / "MyVoice" / "torch_compile_cache"
        else:
            root = (
                Path.home()
                / "AppData"
                / "Local"
                / "MyVoice"
                / "torch_compile_cache"
            )
    else:
        root = Path.home() / ".cache" / "MyVoice" / "torch_compile_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir_for_key(key: str) -> Path:
    """Return (and create) the per-key cache directory.

    The directory's name is the first ``_KEY_DIR_PREFIX_CHARS`` (= 16) chars
    of the key — the full key lands in ``meta.json`` inside the directory.
    Two keys whose first 16 hex chars collide share the directory but each
    has its own ``meta.json`` slot only if the second writer overwrites the
    first; ``is_warm()`` validates the full-key match to defend against
    this residual collision case (defensive — collisions are astronomically
    unlikely for a single-user cache, but the architecture P-10 + D-24
    discipline mandates the full-key check anyway).
    """
    root = cache_root()
    directory = root / key[:_KEY_DIR_PREFIX_CHARS]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_warm(key: str) -> bool:
    """Return True iff this key's compile cache is warm.

    "Warm" means: the per-key directory's ``meta.json`` exists AND its
    ``key`` field matches the input. The full-key match is the canonical
    signal — a directory-prefix collision where ``key2[:16] == key1[:16]``
    is detected as "not warm" because ``meta.json`` records the original
    full key (mirrors Story 17.2's ``_voice_clone_prompt_meta_is_valid``
    discipline at ``qwen_tts_service.py``).

    The function is defensive: an unparseable ``meta.json``, a missing
    ``key`` field, or any I/O error during read is treated as "not warm"
    (the safe choice — a re-compile costs 10-30s once, whereas trusting a
    broken cache could ship corrupt audio).
    """
    meta_path = cache_dir_for_key(key) / "meta.json"
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(meta, dict) and meta.get("key") == key


def mark_warm(key: str) -> None:
    """Mark this key's compile cache as warm by writing the ``meta.json`` sidecar.

    Caller (the warmup worker at ``QwenTTSService.warmup_compile_async``)
    invokes this AFTER a priming generation completes without error
    (architecture D-23 — "warm = compile-priming generation completed
    without error"). On priming failure, the caller does NOT call this;
    the cache stays cold and the next run retries.
    """
    directory = cache_dir_for_key(key)
    meta_path = directory / "meta.json"
    payload = {
        "key": key,
        "warmed_at": datetime.now().isoformat(),
        "torch_compile_cache_dir": str(directory),
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def set_torchinductor_cache_dir(key: str) -> None:
    """Set ``os.environ["TORCHINDUCTOR_CACHE_DIR"]`` to the per-key directory.

    PyTorch's inductor compile cache honors this env var; setting it to a
    per-key directory means each cache key's compiled artifacts live in
    isolation (no cross-key pollution when the user toggles precision,
    upgrades torch, or moves between GPUs).

    Caller must invoke this BEFORE the ``model.enable_streaming_optimizations(...)``
    call so the inductor cache reads/writes from the per-key directory
    during the compile pass. Idempotent — overwriting on re-call is safe
    (and expected; tests will re-call across test rows).

    Logs at INFO **once per process** (set-once guard); subsequent calls
    log at DEBUG. The set-once guard bounds log noise during test isolation
    where this function is called repeatedly with different keys.
    """
    global _TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE
    directory = cache_dir_for_key(key)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(directory)
    if not _TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE:
        _logger.info(
            "TORCHINDUCTOR_CACHE_DIR set to %s",
            directory,
        )
        _TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE = True
    else:
        _logger.debug(
            "TORCHINDUCTOR_CACHE_DIR re-set to %s",
            directory,
        )
