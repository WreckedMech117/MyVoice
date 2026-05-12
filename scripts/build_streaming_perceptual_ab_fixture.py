#!/usr/bin/env python
"""Story 16.7 — Perceptual A/B fixture builder for the streaming-default gate.

For every utterance in the input set tagged ``is_perceptual_difficult=true``,
runs the production Qwen3-TTS dispatch path TWICE (once via
``_generate_true_stream`` for the TRUE_STREAM rendition, once via
``_generate_streaming`` for the SENTENCE_STREAM rendition) and writes the
resulting audio as paired WAV files. Truth-table file naming is preserved on
disk for the report's analysis; the per-listener manifest randomizes the
A/B labels per-listener-per-utterance for blind audition (Story 16.7 AC #2).

Outputs to ``--output-dir``:
  - ``{utterance_id}-A-true_stream.wav`` (canonical: always the TRUE_STREAM
    rendition; this is the truth-table-preserved name)
  - ``{utterance_id}-B-sentence_stream.wav`` (canonical: always the
    SENTENCE_STREAM rendition)
  - ``_perlistener_truthtable.json`` (per-listener-per-utterance A/B
    randomization; the report's section 4 joins audition results against
    this file to compute per-system defect counts)
  - ``LISTENING-INSTRUCTIONS.md`` (blind-audition protocol)

Usage::

    python scripts/build_streaming_perceptual_ab_fixture.py \\
        --input-set _bmad-output/implementation-artifacts/16-7-input-set.csv \\
        --output-dir _bmad-output/implementation-artifacts/16-7-perceptual-fixtures/ \\
        --listener-count 3

The script REQUIRES a CUDA-available host so the TRUE_STREAM rendition can be
generated; on a CPU-only host the script aborts with a non-zero exit code
(rationale: A/B against a SENTENCE_STREAM rendition that itself falls back to
SENTENCE_STREAM produces two identical files — useless for audition).

Architecture references:
  - NFR3 (architecture-optimization-pass.md:803): no audio stuttering — the
    perceptual A/B audition is the gate; the architecture's framing is "D-8
    chunk + overlap-add with seam-quality A/B testing before flipping
    streaming default".
  - Story 16.7 AC #2: blind audition with per-listener randomization,
    canonical disk naming preserves truth-table for analysis.
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
    _torch_lib = _REPO_ROOT / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

import torch  # noqa: E402  — must precede PyQt6 import below

from PyQt6.QtWidgets import QApplication  # noqa: E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from myvoice.models.app_settings import AppSettings  # noqa: E402
from myvoice.models.service_enums import QwenModelType  # noqa: E402
from myvoice.services.audio_coordinator import AudioCoordinator  # noqa: E402
from myvoice.services.monitor_audio_service import MonitorAudioService  # noqa: E402
from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService  # noqa: E402
from myvoice.services.sessions import SessionRegistry  # noqa: E402
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.build_streaming_perceptual_ab_fixture")


@dataclass
class PerceptualUtterance:
    utterance_id: str
    text: str


def _load_perceptual_utterances(path: Path) -> List[PerceptualUtterance]:
    rows: List[PerceptualUtterance] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw.get("is_perceptual_difficult", "false").strip().lower() != "true":
                continue
            rows.append(PerceptualUtterance(
                utterance_id=raw["utterance_id"].strip(),
                text=raw["text"],
            ))
    return rows


def _build_mock_audio_coordinator() -> AudioCoordinator:
    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_all_playback = AsyncMock(return_value=0)
    monitor.play_monitor_audio = AsyncMock()
    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_all_virtual_microphone_playback = AsyncMock(return_value=0)
    virtual.play_virtual_microphone = AsyncMock()
    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = monitor
    coord.virtual_service = virtual
    return coord


def _build_request(utterance: PerceptualUtterance) -> QwenTTSRequest:
    return QwenTTSRequest(
        text=utterance.text,
        language="English",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        instruct=None,
        streaming=True,
    )


async def _render_pair(
    service: QwenTTSService,
    utterance: PerceptualUtterance,
    output_dir: Path,
) -> Dict[str, str]:
    """Render TRUE_STREAM and SENTENCE_STREAM versions of one utterance.

    Returns a dict mapping ``"true_stream_filename"`` and
    ``"sentence_stream_filename"`` to the resulting WAV paths (relative to
    ``output_dir``). Raises if either rendition fails — the audition is
    pair-based; an unpaired utterance is dropped from the fixture.
    """
    request = _build_request(utterance)
    true_response = await service._generate_true_stream(request)
    if not true_response.success or true_response.audio_data is None:
        raise RuntimeError(
            f"TRUE_STREAM rendition failed for {utterance.utterance_id}: "
            f"{true_response.error_message!r}"
        )
    sentence_response = await service._generate_streaming(request)
    if not sentence_response.success or sentence_response.audio_data is None:
        raise RuntimeError(
            f"SENTENCE_STREAM rendition failed for {utterance.utterance_id}: "
            f"{sentence_response.error_message!r}"
        )

    true_name = f"{utterance.utterance_id}-A-true_stream.wav"
    sentence_name = f"{utterance.utterance_id}-B-sentence_stream.wav"

    sf.write(
        str(output_dir / true_name),
        np.asarray(true_response.audio_data, dtype=np.float32),
        samplerate=int(true_response.sample_rate or 24000),
        subtype="PCM_16",
    )
    sf.write(
        str(output_dir / sentence_name),
        np.asarray(sentence_response.audio_data, dtype=np.float32),
        samplerate=int(sentence_response.sample_rate or 24000),
        subtype="PCM_16",
    )

    return {
        "true_stream_filename": true_name,
        "sentence_stream_filename": sentence_name,
    }


def _build_truthtable(
    rendered: Dict[str, Dict[str, str]],
    listener_count: int,
    seed: int,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Build the per-listener A/B randomization manifest.

    Schema::

        {
          "L1": {
            "u01": {
              "trial_A_filename": "u01-A-true_stream.wav",
              "trial_B_filename": "u01-B-sentence_stream.wav",
              "trial_A_actual_mode": "true_stream",
              "trial_B_actual_mode": "sentence_stream"
            },
            ...
          },
          "L2": { ... }
        }

    For each (listener, utterance) pair, randomly decide whether trial A
    is the TRUE_STREAM or SENTENCE_STREAM rendition. The randomization is
    deterministic given the seed so the manifest can be regenerated for
    re-runs.
    """
    rng = random.Random(seed)
    manifest: Dict[str, Dict[str, Dict[str, str]]] = {}
    for i in range(1, listener_count + 1):
        listener_id = f"L{i}"
        per_utterance: Dict[str, Dict[str, str]] = {}
        for uid, names in sorted(rendered.items()):
            true_filename = names["true_stream_filename"]
            sentence_filename = names["sentence_stream_filename"]
            if rng.random() < 0.5:
                trial_a, trial_b = true_filename, sentence_filename
                a_mode, b_mode = "true_stream", "sentence_stream"
            else:
                trial_a, trial_b = sentence_filename, true_filename
                a_mode, b_mode = "sentence_stream", "true_stream"
            per_utterance[uid] = {
                "trial_A_filename": trial_a,
                "trial_B_filename": trial_b,
                "trial_A_actual_mode": a_mode,
                "trial_B_actual_mode": b_mode,
            }
        manifest[listener_id] = per_utterance
    return manifest


_LISTENING_INSTRUCTIONS_TEMPLATE = """# Story 16.7 — Perceptual A/B Audition Instructions

Thank you for participating in the streaming-TTS perceptual audition. The goal
is to detect any audible defects in the new chunked-streaming TTS path versus
the existing sentence-streaming baseline. Your judgment is the architecture's
gate (NFR3 — no audio stuttering / no audible overlap-add seams); the data
informs whether MyVoice flips the streaming default for GPU users.

## Your packet

This directory contains paired WAV files. For each utterance you will hear
TWO renditions of the same text — labelled `A` and `B` in the per-listener
manifest at `_perlistener_truthtable.json`. The two renditions come from
different code paths; you do NOT need to know which is which (and the labels
are randomized per-listener so guessing won't help).

## Protocol — for each utterance

1. Open `_perlistener_truthtable.json` and look up your listener id (L1, L2,
   L3, ...). Note the `trial_A_filename` and `trial_B_filename` for the
   utterance you're auditioning.
2. Listen to **trial A** end to end. Then listen to **trial B** end to end.
   Use headphones if you have them. Listen at a comfortable volume; do not
   crank the volume past your normal Discord-call level.
3. Record the following in the audition CSV (one row per utterance):
   - `utterance_id`: from the file name
   - `listener_id`: your assigned listener id (L1, L2, L3)
   - `a_or_b_preferred`: which rendition sounded better overall to you (`A`,
     `B`, or `equivalent`)
   - `a_defects_observed`: any defects you noticed in trial A — pick from
     the controlled vocabulary below; `none` if you heard nothing
   - `b_defects_observed`: same vocabulary, for trial B
   - `free_text_notes`: anything else worth flagging (one or two sentences)

## Controlled defect vocabulary

Pick exactly one. If you heard a defect not on this list, choose
`other_describe_in_notes` and add a sentence in `free_text_notes`.

| Value | What it means |
|---|---|
| `none` | No audible defects |
| `audible_seam` | Audible click, gap, or discontinuity between phrases |
| `clipping` | Distortion as if the audio is too loud / clipped |
| `phase_artifact` | Unnatural phasing or comb-filter sound |
| `tonal_distortion` | Pitch wandering or unnatural intonation |
| `other_describe_in_notes` | Some other defect — describe in notes column |

## What the gate is

Per Story 16.7 AC #2, the perceptual gate is **PASS if and only if zero
listeners flagged `audible_seam` for any TRUE_STREAM pair**. Preference is
informational at N=3; defect detection is the architectural concern.

## Submitting your results

Write your audition rows to a CSV at
`_bmad-output/implementation-artifacts/16-7-perceptual-ab-results.csv` with
the header:

```
utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes
```

If you prefer, send the maintainer a plain-text or spreadsheet copy and they
will fold the results into the canonical CSV. Do not edit the manifest file
or the WAV files.

Thank you for your time and attention.
"""


def write_listening_instructions(path: Path) -> None:
    path.write_text(_LISTENING_INSTRUCTIONS_TEMPLATE, encoding="utf-8")
    logger.info("Wrote %s", path)


async def _amain(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available — the TRUE_STREAM rendition cannot be "
            "produced. Run this script on the GPU host (RTX 5090 Blackwell "
            "per memory/hardware_setup.md)."
        )
        return 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    utterances = _load_perceptual_utterances(args.input_set)
    if not utterances:
        logger.error(
            "Input set %s contains no rows tagged is_perceptual_difficult="
            "true; nothing to render.",
            args.input_set,
        )
        return 2
    logger.info("Loaded %d perceptual-difficult utterances", len(utterances))

    coord = _build_mock_audio_coordinator()
    settings = AppSettings()
    registry = SessionRegistry()
    service = QwenTTSService(
        audio_coordinator=coord,
        session_registry=registry,
        app_settings=settings,
    )

    rendered: Dict[str, Dict[str, str]] = {}
    try:
        ok = await service.start()
        if not ok:
            logger.error("QwenTTSService failed to start; aborting fixture build.")
            return 3

        for idx, utterance in enumerate(utterances, 1):
            try:
                names = await _render_pair(service, utterance, args.output_dir)
            except Exception:
                logger.exception(
                    "[%d/%d] Failed to render utterance %s; dropping from "
                    "the fixture.",
                    idx, len(utterances), utterance.utterance_id,
                )
                continue
            rendered[utterance.utterance_id] = names
            logger.info(
                "[%d/%d] Rendered %s: %s + %s",
                idx, len(utterances), utterance.utterance_id,
                names["true_stream_filename"], names["sentence_stream_filename"],
            )

        if not rendered:
            logger.error("No utterances rendered successfully — fixture is empty.")
            return 5

        manifest = _build_truthtable(
            rendered=rendered,
            listener_count=args.listener_count,
            seed=args.seed,
        )
        manifest_path = args.output_dir / "_perlistener_truthtable.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Wrote %s", manifest_path)

        write_listening_instructions(args.output_dir / "LISTENING-INSTRUCTIONS.md")

        # Summary so the operator knows what to commit.
        wav_files = sorted(args.output_dir.glob("*.wav"))
        total_bytes = sum(p.stat().st_size for p in wav_files)
        total_mb = total_bytes / (1024 * 1024)
        logger.info(
            "Fixture build complete: %d WAV files, %.1f MB total, %d listeners "
            "in manifest. (Threshold for committing the fixture directory is "
            "50 MB per Story 16.7 AC #7 / Task 6.2.)",
            len(wav_files), total_mb, args.listener_count,
        )
        return 0
    finally:
        try:
            await service.stop()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.exception("service.stop() raised during teardown")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Story 16.7 — Build the perceptual A/B fixture for the streaming-"
            "default audition (TRUE_STREAM vs SENTENCE_STREAM paired WAVs + "
            "per-listener randomized truth-table)."
        ),
    )
    parser.add_argument(
        "--input-set", type=Path, required=True,
        help="Path to the fixed input-set CSV; rows with "
             "is_perceptual_difficult=true are rendered.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the paired WAVs, manifest, and instructions.",
    )
    parser.add_argument(
        "--listener-count", type=int, default=3,
        help="Number of listener slots to create in the manifest (default: 3, "
             "matching AC #2 Listener selection L1/L2/L3).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260507,
        help="Deterministic seed for per-listener A/B randomization "
             "(default: 20260507 — story-draft date).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Story 11.2 D-2: SessionRegistry lives on the Qt main thread, so a
    # QApplication instance must exist before the registry is constructed.
    _qapp = QApplication.instance() or QApplication([])
    _ = _qapp  # keep alive for the duration of the run

    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
