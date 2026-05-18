"""Install-time model predownload entrypoint.

Invoked from `MyVoice.exe --predownload-models --tier=<small|quality>`,
typically from the Inno Setup installer's post-extraction phase. Pre-
populates the HuggingFace cache with the chosen tier's three model IDs
(CustomVoice, VoiceDesign, Base) so the first user-facing generation
doesn't have to wait for the multi-GB download.

Design constraints:
  * Must NOT import torch, transformers, or PyQt6 — those are heavy and
    irrelevant for a download. Only `huggingface_hub.snapshot_download`
    is required.
  * Must NOT call MyVoiceApp / configuration_service / model_registry —
    those drag in the full graph.
  * Cache location is the default HuggingFace cache. UAC-elevated
    installers still run under the same user session, so the cache
    populated here is the same one the runtime app reads later.
  * Must restrict the download to ONLY the files `from_pretrained` would
    pull. Vanilla `snapshot_download` grabs the entire repo — for Qwen3-
    TTS repos that includes demo audios, embedded model-card media,
    notebooks, and sometimes redundant `.bin` weight copies. 2026-05-17
    smoke surfaced this: snapshot_download ran 30+ minutes while
    `from_pretrained` from the same repo finishes in 1-2 minutes. The
    `_ALLOW_PATTERNS` list below mirrors the file set observed in the
    HF cache after a successful `from_pretrained` (config + tokenizer
    + safetensors + speech_tokenizer subfolder).
  * Exit codes: 0 = success, 1 = bad args, 2 = download failure. The
    installer is expected to treat exit code 2 as "warn user, app will
    download on first launch instead" rather than failing the install.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, List, Optional


# Hard-coded HuggingFace model ID templates. Duplicated from
# `models/service_enums.py` so this module can run without importing
# the full myvoice tree (which pulls torch via transitive imports).
_BASE_QUALITY_IDS: List[str] = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]

# File-set filter for `snapshot_download`. Derived from the actual
# HuggingFace cache contents after a successful `from_pretrained` on a
# Qwen3-TTS-12Hz repo (verified 2026-05-17, both 0.6B and 1.7B tiers).
# Each model snapshot contained exactly these files:
#   config.json, generation_config.json, preprocessor_config.json,
#   tokenizer_config.json, vocab.json, merges.txt, model.safetensors,
#   speech_tokenizer/{config,configuration,preprocessor_config}.json,
#   speech_tokenizer/model.safetensors
# The patterns below are deliberately a superset (e.g. allowing
# `model-*.safetensors` for future sharded weights, `tokenizer.model`
# for sentencepiece variants, `*.py` for trust_remote_code-style
# repos) without admitting the bloat that vanilla snapshot_download
# pulls: README media, demo audios, .gguf quantizations, etc.
_ALLOW_PATTERNS: List[str] = [
    # Root-level config / metadata / tokenizer
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.model",
    "*.py",
    # Speech tokenizer sub-model directory (one level deep)
    "speech_tokenizer/*.json",
    "speech_tokenizer/*.safetensors",
    "speech_tokenizer/*.safetensors.index.json",
    "speech_tokenizer/tokenizer.model",
    "speech_tokenizer/*.py",
]


def _resolve_model_ids_for_tier(tier: str) -> List[str]:
    """Return the HuggingFace model IDs to download for the given tier.

    Mirrors `QwenModelType.get_model_id(tier)` and the
    `available_in_small_tier` rule from service_enums.py — VoiceDesign
    is QUALITY-only.
    """
    tier_lc = tier.lower().strip()
    if tier_lc == "quality":
        return list(_BASE_QUALITY_IDS)
    if tier_lc == "small":
        # 0.6B replaces 1.7B in the templates; VoiceDesign is dropped.
        return [
            mid.replace("1.7B", "0.6B")
            for mid in _BASE_QUALITY_IDS
            if "VoiceDesign" not in mid
        ]
    raise ValueError(
        f"Unknown tier '{tier}'. Expected 'small' or 'quality'."
    )


def _parse_argv(argv: Iterable[str]) -> Optional[str]:
    """Extract the value of `--tier=...` from argv. Returns None if
    `--predownload-models` is not present (caller should fall through to
    normal app startup). Raises ValueError if the flag is present but
    `--tier=...` is missing or malformed.
    """
    arg_list = list(argv)
    if "--predownload-models" not in arg_list:
        return None
    tier: Optional[str] = None
    for arg in arg_list:
        if arg.startswith("--tier="):
            tier = arg.split("=", 1)[1].strip()
            break
    if not tier:
        raise ValueError(
            "--predownload-models requires --tier=<small|quality>"
        )
    return tier


def run_predownload(argv: Iterable[str]) -> int:
    """Entry point. Returns process exit code.

    Logging is minimal stdout so the installer can capture progress in
    its own progress page. INFO and WARNING go to stdout; ERROR also
    goes to stdout so the installer sees it without needing to wire up
    a stderr pipe.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger("myvoice.predownload")

    try:
        tier = _parse_argv(argv)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    if tier is None:
        # Caller should not have invoked this function. Return 1.
        logger.error(
            "run_predownload called without --predownload-models in argv"
        )
        return 1

    try:
        model_ids = _resolve_model_ids_for_tier(tier)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    logger.info(
        "Pre-downloading %d model(s) for tier=%s",
        len(model_ids), tier,
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        logger.error(
            "huggingface_hub not available in bundle: %s. "
            "Cannot pre-download; app will download on first launch.",
            exc,
        )
        return 2

    overall_ok = True
    for idx, model_id in enumerate(model_ids, start=1):
        logger.info(
            "[%d/%d] Downloading %s ...",
            idx, len(model_ids), model_id,
        )
        try:
            snapshot_download(
                repo_id=model_id,
                # `allow_patterns` restricts the download to the files
                # `from_pretrained` would pull. Without it,
                # snapshot_download grabs the entire repo — 30+ minutes
                # observed on RTX 3060 smoke 2026-05-17 vs. 1-2 min for
                # the equivalent from_pretrained call. See module
                # docstring for the derivation.
                allow_patterns=_ALLOW_PATTERNS,
                # Default cache_dir — matches what from_pretrained will
                # look at later. Don't override unless the runtime also
                # overrides.
                # `resume_download=True` lets partial downloads from an
                # interrupted prior run continue rather than restart.
                resume_download=True,
            )
            logger.info("[%d/%d] %s done", idx, len(model_ids), model_id)
        except Exception as exc:  # noqa: BLE001 — log any failure
            overall_ok = False
            logger.error(
                "[%d/%d] %s FAILED: %s",
                idx, len(model_ids), model_id, exc,
            )
            # Continue to the next model — partial cache population is
            # still useful (the user may have downloaded one of the
            # three before the network dropped).

    if overall_ok:
        logger.info("Pre-download complete.")
        return 0
    logger.warning(
        "One or more models failed to pre-download. Remaining models "
        "will be fetched on first generation."
    )
    return 2
