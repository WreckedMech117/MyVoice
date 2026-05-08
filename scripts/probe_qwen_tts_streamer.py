#!/usr/bin/env python
"""Story 16.8 — qwen-tts streamer-kwarg forwarding probe (AC #1).

One-shot empirical probe that determines whether the qwen-tts wrapper's
public ``generate_custom_voice`` entrypoint forwards a ``streamer`` kwarg
through to the inner ``self.talker.generate(...)`` call (HF
``GenerationMixin``'s standard streaming hook).

Source-read of ``qwen_tts/core/models/modeling_qwen3_tts.py`` at the
pinned commit (``1ab0dd75`` / qwen-tts 0.0.4) predicts Path B will fail:

  - ``Qwen3TTSModel.generate_custom_voice(..., **kwargs)`` at
    ``qwen3_tts_model.py:732`` forwards ``kwargs`` through
    ``_merge_generate_kwargs`` and into ``self.model.generate(**gen_kwargs)``
    at line 829.
  - ``Qwen3TTSForConditionalGeneration.generate(..., **kwargs)`` at
    ``modeling_qwen3_tts.py:2022`` reads only ``output_hidden_states``
    and ``return_dict_in_generate`` from its ``kwargs`` (lines 2064-2065)
    and builds a local ``talker_kwargs`` dict that does NOT include
    ``streamer`` before calling ``self.talker.generate(**talker_kwargs)``
    at line 2272-2278.

This probe is the empirical confirmation per AC #1's rule that the
outcome must be recorded before any production code is written.

Usage:

    python310\\python.exe scripts\\probe_qwen_tts_streamer.py

Exit codes:
    0 — probe ran to completion; outcome printed to stdout (one of i/ii/iii).
    2 — CUDA unavailable (probe is GPU-only).
    3 — model failed to load.
    4 — probe raised an unexpected non-TypeError exception.

Outcomes (per AC #1):
    (i)  STREAMER_FORWARDED — ``streamer.put`` was called >=1 times
         before ``generate_custom_voice`` returned. Path B viable; commit
         to Path B.
    (ii) STREAMER_DROPPED — ``streamer.put`` was NEVER called and
         ``generate_custom_voice`` returned a non-streaming wav list.
         Path A required (replicate preprocessing locally).
    (iii) STREAMER_REJECTED — ``TypeError`` from the wrapper or HF
         indicating ``streamer`` is not a recognized kwarg. Path A
         required.

Mirrors Story 16.7's torch-before-PyQt6 DLL preamble per
``memory/torch_pyqt6_dll_ordering.md``.
"""

# DLL ordering: torch MUST import before PyQt6 on Windows.
# See memory/torch_pyqt6_dll_ordering.md and src/myvoice/main.py:25-49.
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _torch_lib = (
        _REPO_ROOT / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    )
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"
        ),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

import torch  # noqa: E402  — must precede PyQt6/qwen_tts imports below

import logging  # noqa: E402
import threading  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe_qwen_tts_streamer")


def _print_outcome_block(outcome_label: str, *lines: str) -> None:
    border = "=" * 60
    logger.info(border)
    logger.info("PROBE OUTCOME: %s", outcome_label)
    for ln in lines:
        logger.info(ln)
    logger.info(border)


def main() -> int:
    if not torch.cuda.is_available():
        logger.error(
            "CUDA not available; probe is GPU-only per Story 16.8 AC #1"
        )
        return 2
    logger.info("CUDA detected: %s", torch.cuda.get_device_name(0))

    # Local imports below are deferred so the DLL-ordering preamble has run.
    from qwen_tts import Qwen3TTSModel  # noqa: E402

    from myvoice.models.service_enums import (  # noqa: E402
        ModelQualityTier,
        QwenModelType,
    )
    from myvoice.services.tts_streaming.codec_token_streamer import (  # noqa: E402
        CodecTokenStreamer,
    )

    # Mirror ModelRegistry._load_qwen_model's CUSTOM_VOICE/Quality flow.
    model_id = QwenModelType.CUSTOM_VOICE.get_model_id(
        ModelQualityTier.QUALITY
    )
    load_kwargs = {
        "device_map": "cuda:0",
        "torch_dtype": torch.bfloat16,
    }
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using attn_implementation=flash_attention_2")
    except ImportError:
        logger.info("flash_attn not available; using default attention")

    logger.info("Loading qwen-tts model: %s", model_id)
    try:
        model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        logger.exception("Model load failed: %s", exc)
        return 3

    supported = sorted(list(model.get_supported_speakers() or []))
    logger.info(
        "Model loaded. Supported speakers (first 8): %s", supported[:8]
    )
    speaker = "Ryan" if "ryan" in [s.lower() for s in supported] else (
        supported[0] if supported else ""
    )
    logger.info("Probing with speaker='%s', language='English'", speaker)

    cancel_event = threading.Event()
    streamer = CodecTokenStreamer(
        chunk_size=25, lookahead=5, cancel_event=cancel_event,
    )
    # Spy on streamer.put so we count HF -> streamer hand-offs. Counts
    # invocations of put(value), not queue-pushes (the streamer only
    # pushes a chunk once the buffer reaches chunk_size+lookahead = 30
    # tokens, and a probe with max_new_tokens=64 would see 2-3 pushes
    # at most — so we instrument at the HF-callback boundary instead).
    put_call_count = [0]
    real_put = streamer.put

    def spy_put(value):
        put_call_count[0] += 1
        return real_put(value)

    streamer.put = spy_put  # type: ignore[method-assign]

    try:
        wavs, sr = model.generate_custom_voice(
            text="hi",
            speaker=speaker,
            language="English",
            streamer=streamer,
            max_new_tokens=64,
        )
    except TypeError as exc:
        msg = str(exc).lower()
        if "streamer" in msg or "unexpected keyword" in msg:
            _print_outcome_block(
                "(iii) STREAMER_REJECTED",
                f"Exception: {exc!r}",
                "Decision: Path A required (replicate preprocessing locally).",
                "Update Story 16.8 Change Log with this outcome before",
                "writing production code per AC #1.",
            )
            return 0
        logger.exception(
            "Unexpected TypeError (does not mention 'streaming'): %s", exc
        )
        return 4
    except Exception as exc:
        logger.exception("Unexpected non-TypeError during probe: %s", exc)
        return 4

    if put_call_count[0] >= 1:
        _print_outcome_block(
            "(i) STREAMER_FORWARDED",
            f"streamer.put() invoked {put_call_count[0]} times before return.",
            f"generate_custom_voice returned {len(wavs)} wav(s) at sr={sr}.",
            "Decision: Path B viable — commit to Path B.",
            "Update Story 16.8 Change Log with this outcome before",
            "writing production code per AC #1.",
        )
    else:
        _print_outcome_block(
            "(ii) STREAMER_DROPPED",
            "streamer.put() was NEVER invoked.",
            f"generate_custom_voice returned {len(wavs)} wav(s) at sr={sr} "
            "via the non-streaming inner path (talker_kwargs dropped the kwarg).",
            "Decision: Path A required (replicate preprocessing locally and",
            "call model.model.talker.generate(streamer=streamer, ...) directly).",
            "Update Story 16.8 Change Log with this outcome before",
            "writing production code per AC #1.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
