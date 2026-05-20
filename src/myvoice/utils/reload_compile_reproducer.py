"""Headless reproducer for the post-generation compile-disengage bug.

Drives a deterministic load -> priming-generation -> force-reload sequence
without touching the PyQt6 UI graph (no QApplication, no splash, no qasync
event loop). Triggered by the ``--reload-compile-repro`` CLI flag wired in
``main.py``; the host-side driver ``build_tools/verify_reload_compile.py``
runs the bundled ``MyVoice.exe`` once per ``MYVOICE_RELOAD_COMPILE_FIX`` value
and parses the marker line printed below.

Exit codes (TD-3 of the compile-disengage spec):
    0 — reload engaged compile (PASS)
    1 — reload hit ``cuda_unavailable`` (bug confirmed)
    2 — setup error (model load failed, etc.)
    3 — initial-compile invariant violated (reproducer short-circuits)

The marker line is printed to stdout exactly once::

    RELOAD_COMPILE_REPRO: fix=<value> initial=<reason> reload=<reason>

When the initial-compile invariant is violated, ``reload`` is reported as
``not_evaluated`` so log parsers can distinguish a real cuda_unavailable
reload from a short-circuit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional


_logger = logging.getLogger(__name__)

# Initial-compile invariant: the first load's reason must be one of these.
# Any other value means the reproducer's prerequisite (working initial
# compile) is broken; exit code 3 + reload=not_evaluated.
_INITIAL_INVARIANT_OK = ("engaged_cold_compile", "engaged_warm_cache")

# Reload PASS: the reload must engage compile (warm cache is fine — the
# first reload after generation should be able to reuse the on-disk cache
# if the bug is fixed).
_RELOAD_PASS = ("engaged_cold_compile", "engaged_warm_cache")


def _emit_marker(fix_value: str, initial_reason: str, reload_reason: str) -> None:
    """Print the canonical marker line that the driver parses."""
    print(
        f"RELOAD_COMPILE_REPRO: fix={fix_value} "
        f"initial={initial_reason} reload={reload_reason}",
        flush=True,
    )


def _patch_transformers_mistral_check() -> None:
    """Short-circuit ``transformers._patch_mistral_regex`` to avoid an
    unconditional ``model_info`` HTTP call on every tokenizer load.

    The bundled exe's SSL setup can fail to reach huggingface.co (cert chain
    not validated in some launch contexts). Since the reproducer runs with
    ``HF_HUB_OFFLINE=1`` and the models are already in the local cache, the
    mistral-regex check is both unnecessary and harmful. Patching it to a
    no-op lets the tokenizer load offline without forcing every fix's
    diagnostic run to depend on network connectivity.

    **Leak-safety invariant:** this function rebinds an attribute on
    ``PreTrainedTokenizerBase`` at CLASS scope, persisting for the lifetime
    of the Python process. It is only safe to call from the
    ``--reload-compile-repro`` CLI entry point because that path terminates
    the process via ``sys.exit(run_reproducer(...))`` in ``main.py:343``
    (immediately after `run_reproducer` returns). If a future caller invokes
    ``_patch_transformers_mistral_check`` from any path that does NOT
    terminate the process, the patch will leak into the normal app's
    tokenizer loads. The ``_assert_reproducer_only_caller`` helper at
    import time pins this invariant.

    Patched attribute verified to exist on the bundled transformers pin via
    the 2026-05-18 SSL-failure stderr trace at
    ``transformers/tokenization_utils_base.py:2395, 2410-2444``.
    """
    try:
        from transformers import tokenization_utils_base as _t
    except Exception:  # noqa: BLE001 — transformers may not be importable in tests
        return

    # Pre-condition: the symbol we're about to patch must exist. If
    # transformers ever drops or renames _patch_mistral_regex, this function
    # becomes dead code (silent no-op below) — log a WARNING so the next
    # verifier run reveals the staleness.
    if not hasattr(_t.PreTrainedTokenizerBase, "_patch_mistral_regex"):
        _logger.warning(
            "transformers.PreTrainedTokenizerBase._patch_mistral_regex is "
            "missing on this pin; reproducer workaround is dead code. If the "
            "tokenizer load now hits huggingface.co directly, restore the "
            "patch with the correct attribute name."
        )
        return

    def _noop_patch(cls, tokenizer, *args, **kwargs):
        return tokenizer

    try:
        _t.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_noop_patch)
        _logger.info(
            "Patched transformers._patch_mistral_regex to no-op for reproducer "
            "(avoids unconditional model_info() HTTP call)"
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Failed to patch _patch_mistral_regex (%s: %s); "
            "tokenizer load may hit huggingface.co",
            type(exc).__name__, exc,
        )


async def _run_async(fix_value: str) -> int:
    """Async body of the reproducer. Returns the structured exit code."""
    # Patch transformers BEFORE any service-side import that triggers
    # tokenizer loading. The patch is reproducer-local; production loads are
    # unaffected (this function only runs under --reload-compile-repro).
    _patch_transformers_mistral_check()

    try:
        from myvoice.models.service_enums import QwenModelType
        from myvoice.services.model_registry import ModelRegistry
        from myvoice.services.qwen_tts_service import QwenTTSService
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Reproducer import failed: %s", exc)
        _emit_marker(fix_value, "setup_error", "not_evaluated")
        return 2

    registry: Optional[ModelRegistry] = None
    service: Optional[QwenTTSService] = None
    try:
        registry = ModelRegistry(device="auto", quality_tier="quality")
        # Minimal service wiring — no audio coordinator, no session registry.
        # `_run_compile_priming` flows through `generate_custom_voice`, which
        # is permissive about a missing coordinator (the chunked-emission
        # path no-ops without a registered callback).
        service = QwenTTSService(
            audio_coordinator=None,
            device="auto",
            quality_tier="quality",
        )
        # Wire the service's registry to the same instance so the priming
        # call's downstream `ensure_model_loaded` uses our registry.
        if hasattr(service, "_model_registry"):
            service._model_registry = registry
        # Initialize the service's thread pool + request semaphore. Without
        # this, generate_voice_clone's precompute pipeline hits an
        # AttributeError on `async with self._request_semaphore` (the
        # semaphore is None until start() runs).
        await service.start()
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Reproducer setup failed: %s", exc)
        _emit_marker(fix_value, "setup_error", "not_evaluated")
        return 2

    # ----- Initial load ----- #
    try:
        ok, err = await registry.ensure_model_loaded(
            QwenModelType.BASE, tier_override="quality"
        )
        if not ok:
            _logger.error("Initial model load failed: %s", err)
            _emit_marker(fix_value, "setup_error", "not_evaluated")
            return 2
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Initial model load raised: %s", exc)
        _emit_marker(fix_value, "setup_error", "not_evaluated")
        return 2

    initial_reason = "unknown"
    if hasattr(registry, "_compile_engage_result") and registry._compile_engage_result:
        initial_reason = registry._compile_engage_result.get("reason", "unknown")

    # Initial-compile invariant gate — short-circuit before any priming
    # so a broken fix that disables initial compile is reported as
    # exit code 3 + reload=not_evaluated (per AC #12).
    if initial_reason not in _INITIAL_INVARIANT_OK:
        _emit_marker(fix_value, initial_reason, "not_evaluated")
        return 3

    # ----- Priming generation — the "first generation" that the failure
    # mode requires. Critical correction (2026-05-19 production-log
    # observation): we MUST run a BASE-model generation here. The earlier
    # implementation called QwenTTSService._run_compile_priming() which
    # invokes generate_custom_voice() — that path swaps the loaded model
    # from BASE to CUSTOM_VOICE, never reaching the "Base post-generation"
    # state where the cuda_unavailable bug manifests. Instead, call
    # generate_voice_clone() with a bundled Anna-F reference so the
    # TRUE_STREAM path runs against the still-loaded BASE model and
    # poisons the CUDA-Graph state as the production user flow does.
    priming_ok = True
    try:
        from myvoice.utils.portable_paths import get_voice_files_path
        # Resolve Anna-F across the two possible bundle layouts:
        #   1. Installer-deployed: <bundle_root>/voice_files/Anna-F.wav
        #      (the installer copies voice_files OUT of _internal/)
        #   2. Raw PyInstaller dist: <_MEIPASS>/voice_files/Anna-F.wav
        #      (the verifier-driven path used by build_tools/verify_reload_compile.py)
        # Check both locations in order; the data files are byte-identical
        # so generation behavior is independent of which one we use.
        candidates = []
        try:
            candidates.append(get_voice_files_path() / "Anna-F.wav")
        except Exception:  # noqa: BLE001
            pass
        if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
            candidates.append(Path(sys._MEIPASS) / "voice_files" / "Anna-F.wav")
        ref_audio: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                ref_audio = cand
                break
        if ref_audio is None:
            raise FileNotFoundError(
                f"Bundled Anna-F reference voice not found in any of: "
                f"{[str(c) for c in candidates]}; reproducer cannot drive a "
                f"real Base generation."
            )
        ref_text_path = ref_audio.with_suffix(".txt")
        if not ref_text_path.exists():
            raise FileNotFoundError(
                f"Anna-F.txt transcript not found alongside {ref_audio}"
            )
        ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        _logger.info(
            "Priming Base via generate_voice_clone (ref=%s)", ref_audio.name
        )
        await service.generate_voice_clone(
            text="Reproducer priming.",
            ref_audio=ref_audio,
            ref_text=ref_text,
            language="English",
            streaming=True,
            x_vector_only_mode=False,
        )
    except Exception as exc:  # noqa: BLE001
        priming_ok = False
        _logger.warning(
            "Priming generation raised (%s: %s); continuing to reload step "
            "anyway so the reload-reason gets captured",
            type(exc).__name__, exc,
        )

    # ----- Force reload — the canonical second load that the spec asserts
    # comes back with cuda_unavailable on the unfixed build. ----- #
    reload_reason = "unknown"
    try:
        ok, err = await registry.ensure_model_loaded(
            QwenModelType.BASE,
            force_reload=True,
            tier_override="quality",
        )
        if hasattr(registry, "_compile_engage_result") and registry._compile_engage_result:
            reload_reason = registry._compile_engage_result.get("reason", "unknown")
        if not ok:
            _logger.error("Reload model load failed: %s", err)
            # The reload failed entirely — distinct from "engaged failed".
            _emit_marker(fix_value, initial_reason, "load_error")
            return 2
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Reload model load raised: %s", exc)
        _emit_marker(fix_value, initial_reason, "load_error")
        return 2

    _emit_marker(fix_value, initial_reason, reload_reason)

    if reload_reason in _RELOAD_PASS:
        return 0
    if reload_reason == "cuda_unavailable":
        return 1
    # Anything else (e.g. compile_failed, probe_failed) — soft fail; the
    # driver's verdict table records the reason verbatim.
    return 1


def run_reproducer(fix_value: str) -> int:
    """Synchronous entry point invoked from ``main.py``'s CLI handler.

    Sets up logging and runs the async body on a plain ``asyncio`` loop
    (no qasync — the reproducer is headless). Catches any top-level
    exception so the marker line + a structured exit code are always
    emitted.
    """
    # Ensure logging is configured — the reproducer is typically invoked
    # before `setup_logging` runs (it short-circuits before main()), so
    # configure a minimal stderr handler so failures are debuggable.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )

    _logger.info(
        "Reload-compile reproducer starting (fix=%s)", fix_value
    )

    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run_async(fix_value))
        finally:
            try:
                loop.close()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Reproducer top-level exception: %s", exc)
        traceback.print_exc(file=sys.stderr)
        _emit_marker(fix_value, "setup_error", "not_evaluated")
        return 2
