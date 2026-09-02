"""Story 20.1 Gate C (AC #3 + AC #4) - faster-qwen3-tts benchmark + mode-parity probe.

MUST be run with the THROWAWAY venv interpreter, never ``python310\\python.exe``:
faster-qwen3-tts pulls transformers 5.x + ``qwen-tts-hf``, whose top-level
package name (``qwen_tts``) collides head-on with MyVoice's pinned
``dffdeeq/Qwen3-TTS-streaming`` fork. See evidence file section 1.

    <venv>\\Scripts\\python.exe tools\\ttfa_spike_faster_qwen3_probe.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --runs 5 --out bench-1p7b.json

What it measures
----------------
AC #3 - RTF and TTFA on the SAME RTX 5090 host as the MyVoice baseline, for
        both 0.6B (published-number sanity check) and 1.7B (our shipped size).
        TTFA is the wall-clock interval from the ``generate_*_streaming``
        call to the first yielded audio chunk - the closest analogue to
        MyVoice's ``ttfa_generation_start_ms -> progressive_chunk_emit_ms[0]``.
AC #4 - mode parity across CustomVoice / VoiceDesign / voice-clone, plus
        whether MyVoice's Story 17.2 ``<voice>.pt`` prompt cache is consumable
        as-is by the qwen-tts-hf ``VoiceClonePromptItem``.

Spike hygiene (AC #7): scratch script under ``tools/``; imports nothing from
``src/myvoice/`` (the stub-module shim below deliberately avoids importing
MyVoice so the venv never pulls the pinned fork onto its path).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = REPO_ROOT / "voice_files"
REF_WAV = VOICE_DIR / "Sarira-F.wav"
REF_TXT = VOICE_DIR / "Sarira-F.txt"
PROMPT_PT = VOICE_DIR / "Sarira-F.quality.pt"

TEXT_LONG = (
    "This is a longer-form test designed to expose the difference between "
    "metric-side first-chunk emission and user-perceived first-audio latency. "
    "On the pre-Story-17.3 build, the user would wait approximately forty "
    "seconds for this utterance to start playing, even though the streaming "
    "pipeline emitted the first chunk internally at around five seconds."
)
TEXT_SHORT = "Hold on a second, say that again."


def _install_myvoice_stub() -> None:
    """Defensive fallback for a ``.pt`` that pickles MyVoice's wrapper class.

    **This is a belt-and-braces path that is NOT exercised by any ``.pt`` in
    the shipped tree** (review finding A7). A scan of all 24 files under
    ``voice_files/`` shows every one pickles
    ``qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem`` - the library
    class - because ``QwenTTSService`` normalises to it before persisting.
    Under this venv that module path resolves to *qwen-tts-hf's* identically
    named dataclass, so ``torch.load`` succeeds with no shim at all. That is
    the whole basis of the "consumable as-is" verdict in evidence section 4.2,
    and it is worth stating plainly: the stub below was dead code on every
    file measured.

    It is retained only because ``_normalize_voice_clone_prompt`` can also
    accept MyVoice's own wrapper (``qwen_tts_service.VoiceClonePromptItem``),
    so a legacy or hand-made ``.pt`` could still carry that class path.
    Importing the real module instead would drag MyVoice's pinned ``qwen_tts``
    fork onto the venv's path - exactly the collision this venv exists to
    avoid - so a stub is the only safe way to keep that path loadable.
    """
    import dataclasses
    from typing import Optional as _Opt

    pkg_myvoice = types.ModuleType("myvoice")
    pkg_services = types.ModuleType("myvoice.services")
    mod = types.ModuleType("myvoice.services.qwen_tts_service")

    @dataclasses.dataclass
    class VoiceClonePromptItem:  # noqa: D401 - pickle target only
        ref_code: Any = None
        ref_spk_embedding: Any = None
        x_vector_only_mode: bool = False
        icl_mode: bool = True
        ref_text: _Opt[str] = None

    mod.VoiceClonePromptItem = VoiceClonePromptItem
    pkg_myvoice.services = pkg_services
    pkg_services.qwen_tts_service = mod
    sys.modules.setdefault("myvoice", pkg_myvoice)
    sys.modules.setdefault("myvoice.services", pkg_services)
    sys.modules.setdefault("myvoice.services.qwen_tts_service", mod)


def _load_myvoice_prompt_item():
    """Return the Story 17.2 prompt converted to qwen-tts-hf's item class.

    Answers AC #4 / Task 4.2: consumable as-is / regenerable / incompatible.
    """
    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem as HFItem

    _install_myvoice_stub()
    raw = torch.load(str(PROMPT_PT), map_location="cpu", weights_only=False)
    # A7: record which class path pickle actually resolved, so the evidence
    # file can state whether the stub above did anything.
    _resolved = type(raw).__module__ + "." + type(raw).__name__
    _via_stub = _resolved.startswith("myvoice.")
    ref_text = getattr(raw, "ref_text", None)
    if not ref_text and REF_TXT.exists():
        ref_text = REF_TXT.read_text(encoding="utf-8").strip()
    item = HFItem(
        ref_code=getattr(raw, "ref_code", None),
        ref_spk_embedding=getattr(raw, "ref_spk_embedding", None),
        x_vector_only_mode=bool(getattr(raw, "x_vector_only_mode", False)),
        icl_mode=bool(ref_text),
        ref_text=ref_text,
    )
    shapes = {
        "ref_code": None if item.ref_code is None else tuple(item.ref_code.shape),
        "ref_spk_embedding": (
            None if item.ref_spk_embedding is None
            else tuple(item.ref_spk_embedding.shape)
        ),
        "source_class": _resolved,
        "loaded_via_myvoice_stub": _via_stub,
        "icl_mode": item.icl_mode,
        "x_vector_only_mode": item.x_vector_only_mode,
    }
    return item, shapes


def _load(model_name: str, dtype: str = "bf16"):
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    t0 = time.perf_counter()
    m = FasterQwen3TTS.from_pretrained(
        model_name,
        device="cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
        local_files_only=True,
    )
    return m, (time.perf_counter() - t0) * 1000.0


def _bench_clone(model, text: str, chunk_size: int, runs: int,
                 prompt_item=None) -> List[Dict[str, Any]]:
    """Stream one utterance N times; record TTFA + RTF per run."""
    import numpy as np

    ref_text = REF_TXT.read_text(encoding="utf-8").strip() if REF_TXT.exists() else ""
    out: List[Dict[str, Any]] = []
    for i in range(runs):
        kwargs: Dict[str, Any] = dict(
            text=text,
            language="English",
            chunk_size=chunk_size,
        )
        if prompt_item is not None:
            kwargs["voice_clone_prompt"] = [prompt_item]
        else:
            kwargs["ref_audio"] = str(REF_WAV)
            kwargs["ref_text"] = ref_text
        t0 = time.perf_counter()
        ttfa_ms: Optional[float] = None
        total_samples = 0
        sr = 24000
        for chunk, chunk_sr, _timing in model.generate_voice_clone_streaming(**kwargs):
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t0) * 1000.0
            sr = chunk_sr
            total_samples += int(np.asarray(chunk).size)
        wall_s = time.perf_counter() - t0
        audio_s = total_samples / float(sr) if sr else 0.0
        # C5: a zero-sample run is a FAILED run, not an RTF of 0.0. Reporting
        # 0.0 would drag the median down while looking like data.
        rtf = (audio_s / wall_s) if (wall_s > 0 and total_samples > 0) else None
        out.append({
            "run": i,
            "ttfa_ms": ttfa_ms,
            "wall_s": wall_s,
            "audio_s": audio_s,
            "rtf": rtf,
            "samples": total_samples,
            "chunk_size": chunk_size,
        })
        print("    run {i}: TTFA={t} ms  audio={a:.2f}s  wall={w:.2f}s  RTF={r}".format(
            i=i, t=None if ttfa_ms is None else round(ttfa_ms),
            a=audio_s, w=wall_s,
            r=None if not wall_s else round(audio_s / wall_s, 3)))
    return out


def _probe_modes(model_ids: Dict[str, str]) -> Dict[str, Any]:
    """AC #4 - does each of MyVoice's three shipped modes have an entrypoint?"""
    from faster_qwen3_tts import FasterQwen3TTS

    surface = {}
    for name in (
        "generate_custom_voice", "generate_custom_voice_streaming",
        "generate_voice_design", "generate_voice_design_streaming",
        "generate_voice_clone", "generate_voice_clone_streaming",
        "generate",
    ):
        surface[name] = callable(getattr(FasterQwen3TTS, name, None))
    return {"api_surface": surface, "model_ids_probed": model_ids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--text", choices=("long", "short"), default="long")
    ap.add_argument("--use-myvoice-prompt", action="store_true",
                    help="AC #4 Task 4.2 - feed the Story 17.2 <voice>.pt prompt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import transformers
    print("transformers", transformers.__version__)
    import faster_qwen3_tts
    print("faster_qwen3_tts", faster_qwen3_tts.__version__)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device", p.name, round(p.total_memory / 1024 ** 3, 1), "GiB")

    result: Dict[str, Any] = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "text_class": args.text,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "faster_qwen3_tts": faster_qwen3_tts.__version__,
    }
    result.update(_probe_modes({"base": args.model}))

    prompt_item = None
    if args.use_myvoice_prompt:
        try:
            prompt_item, shapes = _load_myvoice_prompt_item()
            result["myvoice_prompt_load"] = {"status": "loaded", **shapes}
            print("MyVoice 17.2 prompt loaded:", shapes)
        except Exception as exc:
            result["myvoice_prompt_load"] = {
                "status": "failed", "error": repr(exc),
            }
            print("MyVoice 17.2 prompt load FAILED:", repr(exc))

    print("Loading", args.model, "...")
    try:
        model, load_ms = _load(args.model)
    except Exception as exc:
        result["load_error"] = repr(exc)
        print("LOAD FAILED:", repr(exc))
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 3
    result["load_ms"] = load_ms
    print("  loaded in {:.0f} ms".format(load_ms))

    text = TEXT_LONG if args.text == "long" else TEXT_SHORT
    print("Warmup ({}) ...".format(args.warmup))
    try:
        warm = _bench_clone(model, text, args.chunk_size, args.warmup, prompt_item)
        result["warmup_runs"] = warm
        print("Measured ({}) ...".format(args.runs))
        rows = _bench_clone(model, text, args.chunk_size, args.runs, prompt_item)
    except Exception as exc:
        result["bench_error"] = repr(exc)
        print("BENCH FAILED:", repr(exc))
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 4

    result["runs"] = rows
    ttfas = [r["ttfa_ms"] for r in rows if r["ttfa_ms"] is not None]
    rtfs = [r["rtf"] for r in rows if r["rtf"] is not None]
    # A1: interpolated quantile, and ``max`` reported separately. The old
    # expression returned the maximum under a p95 label for every n <= 10.
    def _q(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    result["summary"] = {
        "ttfa_ms_median": statistics.median(ttfas) if ttfas else None,
        "ttfa_ms_p95_interpolated": _q(ttfas, 0.95),
        "ttfa_ms_max": max(ttfas) if ttfas else None,
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "rtf_max": max(rtfs) if rtfs else None,
        "n_ttfa": len(ttfas),
        "n_rtf": len(rtfs),
        "n_runs": len(rows),
    }
    print(json.dumps(result["summary"], indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
