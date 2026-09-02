"""Story 20.5 Phase 1 - stage-by-stage attribution of the residual.

``20-5-state-cache-bench.py`` showed the ALL-state arm removes the 555-sample
edge loss outright and drops head NRMSE from ~130 % to ~17-21 %, but did NOT
reach the numerical floor: the residual is a transient concentrated in the
first ~2,000 samples of each chunk and it survives an fp32 pass, so it is not
bf16 rounding.

AC #1 requires each remaining discrepancy to be attributed to a sub-stack
rather than reported as one aggregate. This script does that directly: it
runs the SAME streaming traversal (a) as one whole chunk -- which the bench
proved bit-identical to ``Qwen3TTSTokenizerV2Decoder.forward`` -- and (b) in
25-frame chunks with state carried, recording every intermediate tensor, then
reports where in the module chain the two first diverge.

It also runs with TF32 disabled so the reported floor is a real fp32 floor.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-stage-probe.py

Working file - gitignored under ``_bmad-output/``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bench205", SCRIPT_DIR / "20-5-state-cache-bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def cat_stage(rec, key, time_dim):
    return torch.cat(rec[key], dim=time_dim)


def nrmse(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    d = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / d) if d else float("nan")


def probe(dec, codes, label, chunk=B.CHUNK):
    T = codes.shape[-1]
    stw = B.StreamState(True, True, True)
    stw.rec, stw.order = {}, []
    with torch.inference_mode():
        B.stream_forward(dec, codes, stw)

    stc = B.StreamState(True, True, True)
    stc.rec, stc.order = {}, []
    with torch.inference_mode():
        for a in range(0, T, chunk):
            B.stream_forward(dec, codes[..., a:min(a + chunk, T)], stc)

    print("\n  {} -- stage divergence (streaming-chunked vs whole)".format(label))
    print("  {:<26} {:>10} {:>10} {:>12} {:>12}".format(
        "stage", "len whole", "len chunk", "NRMSE all", "NRMSE head"))
    prev = 0.0
    for key, td in stw.order:
        if key not in stc.rec:
            continue
        w = cat_stage(stw.rec, key, td).numpy()
        c = cat_stage(stc.rec, key, td).numpy()
        lw = w.shape[td]
        lc = c.shape[td]
        n = min(lw, lc)
        ws = np.take(w, range(n), axis=td)
        cs = np.take(c, range(n), axis=td)
        full = nrmse(ws, cs)
        # head-of-chunk error at this stage's own sample rate
        per = lw / float((T + chunk - 1) // chunk)
        h = max(1, int(per * 0.05))
        heads = []
        off = 0
        for i, t in enumerate(stc.rec[key]):
            L = t.shape[td]
            if i > 0 and off + h <= n:
                heads.append(nrmse(np.take(ws, range(off, off + h), axis=td),
                                   np.take(cs, range(off, off + h), axis=td)))
            off += L
        hm = float(np.median(heads)) if heads else float("nan")
        flag = "  <== JUMP" if full > max(3 * prev, 1e-5) and full > 1e-4 else ""
        print("  {:<26} {:>10} {:>10} {:>12.3e} {:>12.3e}{}".format(
            key, lw, lc, full, hm, flag))
        prev = max(prev, full)


async def run():
    import ttfa_spike_harness as H
    from myvoice.services.qwen_tts_service import QwenTTSService

    tokens = {}
    for p in sorted((SCRIPT_DIR / "20-5-tokens").glob("*.npz")):
        tokens[p.stem] = torch.from_numpy(np.load(p)["tokens"]).long()
    if not tokens:
        raise SystemExit("run 20-5-state-cache-bench.py first (no token npz)")

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        raise SystemExit("service.start() failed")
    try:
        from myvoice.models.service_enums import QwenModelType
        ok, err = await service._model_registry.ensure_model_loaded(
            QwenModelType.BASE)
        if not ok:
            raise SystemExit("model load failed: {}".format(err))
        model = service._model_registry.get_loaded_model()
        dec = model.model.speech_tokenizer.model.decoder
        device = next(dec.parameters()).device

        utt = sorted(tokens)[0]
        codes = tokens[utt].t().unsqueeze(0).to(device)
        print("=" * 78)
        print("Story 20.5 stage probe -- {} ({} frames)".format(
            utt, codes.shape[-1]))
        print("=" * 78)

        probe(dec, codes, "bf16 (shipping)")

        dec.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("\n  TF32 disabled: matmul={} cudnn={}".format(
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32))
        probe(dec, codes, "fp32, TF32 off")
    finally:
        await service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
