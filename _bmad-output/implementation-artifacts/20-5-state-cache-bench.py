"""Story 20.5 Phase 1 (AC #1 + AC #2) - codec state-caching bench spike.

THE QUESTION. ``Qwen3TTSTokenizerV2CausalConvNet.forward`` left-pads every
decode with ZEROS where the previous chunk's real audio should be
(modeling_qwen3_tts_tokenizer_v2.py:189-192). Story 20.4 measured the
consequence: ~35 % NRMSE between two decodes of the same frames, 0.55
median correlation falling to 0.11 over the blend region, +/-35 samples of
lag jitter, and a deterministic ``decode(N) == 1920*N - 555`` edge loss.

This bench decodes a REAL token sequence three ways and scores the last two
against the first:

  (1) WHOLE      - ``decoder(codes)`` over the entire sequence. Ground truth.
  (2) INDEP      - 25-frame chunks decoded independently. What ships today
                   (``chunked_decode`` -> ``forward``, one call per chunk).
  (3) STREAM     - 25-frame chunks with codec state carried across the
                   boundary: causal-conv left-context buffers, transposed-
                   conv overlap-add tails, and a transformer KV cache.

Arm (3) is run over the full 2^3 ablation matrix of the three sub-stacks so
each remaining discrepancy is ATTRIBUTED rather than aggregated (AC #1).

NO PRODUCTION CODE IS MODIFIED. The streaming decode is a re-implementation
of ``Qwen3TTSTokenizerV2Decoder.forward``'s module traversal that calls the
LOADED module objects' inner ``nn.Conv1d`` / ``nn.ConvTranspose1d`` directly.
It subclasses nothing, patches nothing, and copies no weights. ``src/myvoice``
is untouched.

WHY THE STREAMING MATH IS WHAT IT IS (derived from the source, not guessed):

  * Every ``CausalConvNet`` in the decoder is built with stride=1, so
    ``_get_extra_padding_for_conv1d`` returns 0 and the module reduces to
    "left-pad k_eff-1 zeros, conv, output length == input length". The
    streaming form keeps the last k_eff-1 input samples as the next call's
    left context. Exact.
  * ``CausalTransConvNet`` is built two ways. The ``upsampling_ratios``
    (2, 2) instances have kernel == stride, so pad == 0 and they are
    already stateless and exact. The four ``upsample_rates`` (8, 5, 4, 3)
    instances have kernel == 2*stride, so pad == left_pad == right_pad ==
    stride: the module conv-transposes, then throws away ``stride`` samples
    at each end. The streaming form overlap-adds the discarded right tail
    into the next chunk's head. Exact.
  * That right-tail discard is the WHOLE of the 555-sample edge loss:
        8*(5*4*3) + 5*(4*3) + 4*(3) + 3*(1) = 480 + 60 + 12 + 3 = 555
    so the prediction stated up front is that carrying transposed-conv
    state moves the 555 from EVERY decode call to the FIRST one only.
  * ``pre_transformer`` is 8 layers of sliding attention, window 72
    (configuration_qwen3_tts_tokenizer_v2.py:82, ``layer_types``), so its
    carried state is a bounded KV cache, not an unbounded one.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-state-cache-bench.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TOK_DIR = SCRIPT_DIR / "20-5-tokens"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2CausalConvNet as CausalConv,
    Qwen3TTSTokenizerV2CausalTransConvNet as CausalTConv,
    Qwen3TTSTokenizerV2ConvNeXtBlock as ConvNeXt,
    Qwen3TTSTokenizerV2DecoderDecoderBlock as DecBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit as ResUnit,
)

SR = 24000
SAMPLES_PER_FRAME = 1920
EDGE_LOSS = 555
CHUNK = 25              # the committed geometry (codec_token_streamer.py:98)
HEAD = 1024             # the Story 20.4 seam-fix blend width

UTTERANCES = {
    "m-020": "She sells seashells by the seashore on a still summer morning.",
    "l-020": (
        "This is a longer-form test designed to expose the difference "
        "between metric-side first-chunk emission and user-perceived "
        "first-audio latency. On the pre-Story-17.3 build, the user would "
        "wait approximately forty seconds for this utterance to start "
        "playing, even though the streaming pipeline emitted the first "
        "chunk internally at around five seconds."
    ),
    "l-021": (
        "Six slick slim sycamore saplings stood swaying silently as the "
        "soft summer storm slowly swirled across the steep slopes south of "
        "the silver stream below us, and the sound of it carried further "
        "than anyone standing there expected it to."
    ),
}


# --------------------------------------------------------------------------- #
# Streaming state + streaming re-implementation of the decoder traversal
# --------------------------------------------------------------------------- #


class StreamState:
    """Per-session codec state. Nothing here is global: one instance per
    stream, which is the AC #2 requirement (the HTTP API added this session
    makes concurrent generations reachable)."""

    def __init__(self, use_conv=True, use_tconv=True, use_xf=True):
        self.use_conv = use_conv
        self.use_tconv = use_tconv
        self.use_xf = use_xf
        self.conv = {}      # module path -> [B, C, k_eff-1] left context
        self.tconv = {}     # module path -> [B, C, stride]   overlap tail
        self.kv = None      # transformers Cache
        self.pos = 0        # frames already fed to the transformer
        self.rec = None     # optional {stage_key: [tensor, ...]} recorder
        self.order = []     # stage keys in traversal order

    def nbytes(self):
        n = 0
        cnt = 0
        for d in (self.conv, self.tconv):
            for t in d.values():
                n += t.numel() * t.element_size()
                cnt += 1
        kv_bytes, kv_cnt = 0, 0
        if self.kv is not None:
            for attr in ("key_cache", "value_cache", "layers"):
                obj = getattr(self.kv, attr, None)
                if obj is None:
                    continue
                for item in obj:
                    for t in _tensors_of(item):
                        kv_bytes += t.numel() * t.element_size()
                        kv_cnt += 1
                break
        return n, cnt, kv_bytes, kv_cnt


def _tensors_of(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    out = []
    for attr in ("keys", "values", "key_cache", "value_cache"):
        v = getattr(obj, attr, None)
        if isinstance(v, torch.Tensor):
            out.append(v)
    return out


def _rec(st, key, t, time_dim):
    if st.rec is None:
        return t
    if key not in st.rec:
        st.rec[key] = []
        st.order.append((key, time_dim))
    st.rec[key].append(t.detach().to(torch.float32).cpu())
    return t


def s_conv(m, x, st, key):
    """Streaming ``CausalConvNet``: real left context instead of zeros."""
    p = m.padding
    if not st.use_conv or p == 0:
        return m(x)
    buf = st.conv.get(key)
    if buf is None:
        buf = x.new_zeros(x.shape[0], x.shape[1], p)
    y = torch.cat([buf, x], dim=-1)
    st.conv[key] = y[..., y.shape[-1] - p:].clone()
    return m.conv(y).contiguous()


def s_tconv(m, x, st, key):
    """Streaming ``CausalTransConvNet``: overlap-add the discarded tail."""
    lp, rp = m.left_pad, m.right_pad
    if lp == 0 and rp == 0:
        return m(x)                      # kernel == stride: already exact
    if not st.use_tconv:
        return m(x)
    y = m.conv(x)                        # length L*s + lp
    prev = st.tconv.get(key)
    if prev is not None:
        # The overlap region is a SUM of two partial convolutions, and
        # ConvTranspose1d adds its bias to every output position of each --
        # so a naive overlap-add double-counts it. Subtracting one copy is
        # what makes the streaming form bit-exact against the whole-sequence
        # decode; without it the seam carries a bias-shaped transient that
        # decays over ~2,000 samples and reads as a residual "cold start".
        head = y[..., :lp] + prev
        if m.conv.bias is not None:
            head = head - m.conv.bias.view(1, -1, 1)
        y = torch.cat([head, y[..., lp:]], dim=-1)
    st.tconv[key] = y[..., y.shape[-1] - rp:].clone()
    out = y[..., : y.shape[-1] - rp]
    if prev is None:
        out = out[..., lp:]              # first chunk keeps the module's trim
    return out.contiguous()


def s_transformer(tf, h, st):
    if not st.use_xf:
        return tf(inputs_embeds=h).last_hidden_state
    from transformers.cache_utils import DynamicCache
    if st.kv is None:
        st.kv = DynamicCache(config=tf.config)
    n = h.shape[1]
    cp = torch.arange(st.pos, st.pos + n, device=h.device)
    out = tf(inputs_embeds=h, past_key_values=st.kv, use_cache=True,
             cache_position=cp)
    st.pos += n
    st.kv = out.past_key_values
    return out.last_hidden_state


def run_module(m, x, st, key):
    if isinstance(m, CausalConv):
        return s_conv(m, x, st, key)
    if isinstance(m, CausalTConv):
        return s_tconv(m, x, st, key)
    if isinstance(m, ConvNeXt):
        inp = x
        h = s_conv(m.dwconv, x, st, key + ".dwconv")
        h = h.permute(0, 2, 1)
        h = m.norm(h)
        h = m.pwconv1(h)
        h = m.act(h)
        h = m.pwconv2(h)
        h = m.gamma * h
        h = h.permute(0, 2, 1)
        return inp + h
    if isinstance(m, ResUnit):
        res = x
        h = m.act1(x)
        h = s_conv(m.conv1, h, st, key + ".conv1")
        h = m.act2(h)
        h = s_conv(m.conv2, h, st, key + ".conv2")
        return h + res
    if isinstance(m, DecBlock):
        h = x
        for i, b in enumerate(m.block):
            h = run_module(b, h, st, "{}.block.{}".format(key, i))
        return h
    return m(x)             # SnakeBeta and anything else pointwise in time


def stream_forward(dec, codes, st):
    """Streaming twin of ``Qwen3TTSTokenizerV2Decoder.forward`` (:876)."""
    hidden = dec.quantizer.decode(codes)
    hidden = s_conv(dec.pre_conv, hidden, st, "pre_conv").transpose(1, 2)
    _rec(st, "00 pre_conv_out", hidden, 1)
    hidden = s_transformer(dec.pre_transformer, hidden, st)
    hidden = hidden.permute(0, 2, 1)
    _rec(st, "01 pre_transformer_out", hidden, 2)
    for i, blocks in enumerate(dec.upsample):
        for j, b in enumerate(blocks):
            hidden = run_module(b, hidden, st, "up.{}.{}".format(i, j))
            _rec(st, "02 up.{}.{}".format(i, j), hidden, 2)
    wav = hidden
    for i, b in enumerate(dec.decoder):
        if isinstance(b, DecBlock):
            for j, sub in enumerate(b.block):
                wav = run_module(sub, wav, st, "dec.{}.block.{}".format(i, j))
                _rec(st, "03 dec.{}.block.{}".format(i, j), wav, 2)
        else:
            wav = run_module(b, wav, st, "dec.{}".format(i))
            _rec(st, "03 dec.{}".format(i), wav, 2)
    return wav.clamp(min=-1, max=1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def nrmse(ref, test):
    d = np.linalg.norm(ref)
    if d == 0:
        return float("nan")
    return float(np.linalg.norm(ref - test) / d)


def best_lag(gt, probe, centre, search=1200):
    n = probe.size
    nb = np.linalg.norm(probe)
    best, blag = -2.0, None
    lo = max(0, centre - search)
    hi = min(gt.size - n, centre + search)
    if hi < lo or nb == 0:
        return None, float("nan")
    for lag in range(lo, hi + 1):
        seg = gt[lag:lag + n]
        na = np.linalg.norm(seg)
        if na == 0:
            continue
        c = float(np.dot(seg, probe) / (na * nb))
        if c > best:
            best, blag = c, lag
    return blag, best


def score_arm(gt, chunks, label):
    """Head metrics for every seam: NRMSE at the nominal splice, plus the
    best-lag correlation / NRMSE / lag delta over the first 1024 samples."""
    off = 0
    rows = []
    for i, c in enumerate(chunks):
        if i > 0 and c.size >= HEAD and off + HEAD <= gt.size:
            probe = c[:HEAD].astype(np.float64)
            ref = gt[off:off + HEAD].astype(np.float64)
            nom = nrmse(ref, probe)
            lag, corr = best_lag(gt.astype(np.float64), probe, off)
            if lag is None:
                al = float("nan")
                delta = float("nan")
            else:
                al = nrmse(gt[lag:lag + HEAD].astype(np.float64), probe)
                delta = lag - off
            rows.append((i, nom, corr, al, delta))
        off += c.size
    if not rows:
        return None
    nom = np.array([r[1] for r in rows])
    cor = np.array([r[2] for r in rows])
    ali = np.array([r[3] for r in rows])
    dl = np.array([r[4] for r in rows], dtype=np.float64)
    total = int(sum(c.size for c in chunks))
    return {
        "arm": label,
        "seams": len(rows),
        "head_nrmse_med": float(np.median(nom)),
        "head_nrmse_max": float(np.max(nom)),
        "corr_med": float(np.median(cor)),
        "corr_min": float(np.min(cor)),
        "aligned_nrmse_med": float(np.median(ali)),
        "lag_delta_min": float(np.min(dl)),
        "lag_delta_max": float(np.max(dl)),
        "lag_delta_absmax": float(np.max(np.abs(dl))),
        "total_len": total,
        "gt_len": int(gt.size),
        "len_delta": total - int(gt.size),
    }


# --------------------------------------------------------------------------- #
# Token capture (reuses the Story 20.4 harness + service wiring)
# --------------------------------------------------------------------------- #


async def capture_tokens():
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        raise SystemExit("FATAL: service.start() returned False")

    grabbed = []
    real_builder = service._build_true_stream_decode_fn

    def wrapped_builder(model):
        inner = real_builder(model)

        def _decode(chunk):
            grabbed.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)
        return _decode

    service._build_true_stream_decode_fn = wrapped_builder

    # Reuse a previous capture when one exists, so repeat runs score the
    # SAME token sequence (talker sampling is stochastic; re-generating
    # would move every number between runs).
    cached = {u: TOK_DIR / "{}.npz".format(u) for u in UTTERANCES}
    if all(p.exists() for p in cached.values()):
        from myvoice.models.service_enums import QwenModelType as _QMT
        ok, err = await service._model_registry.ensure_model_loaded(_QMT.BASE)
        if not ok:
            raise SystemExit("model load failed: {}".format(err))
        out = {u: torch.from_numpy(np.load(p)["tokens"]).long()
               for u, p in cached.items()}
        for u, t in sorted(out.items()):
            print("  reused   {:<6} frames={} quantizers={}".format(
                u, t.shape[0], t.shape[1]))
        model = service._model_registry.get_loaded_model()
        return service, model.model.speech_tokenizer.model.decoder, out

    prompt = H._load_voice_clone_prompt(service)

    out = {}
    try:
        warm = QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True,
        )
        await service._generate_true_stream(warm)
        await asyncio.sleep(0.2)

        for utt_id, text in UTTERANCES.items():
            grabbed.clear()
            req = QwenTTSRequest(
                text=text, language="English",
                model_type=QwenModelType.BASE, streaming=True,
                voice_clone_prompt=prompt, suppress_audio_output=True,
            )
            resp = await service._generate_true_stream(req)
            await asyncio.sleep(0.2)
            if not resp.success or not grabbed:
                print("  SKIP {} (success={} chunks={})".format(
                    utt_id, getattr(resp, "success", None), len(grabbed)))
                continue
            # Chunk i covers frames [i*CHUNK, i*CHUNK + len_i); the lookahead
            # tail is re-sent as the next chunk's head, so writing by
            # position de-duplicates without assuming the residual shape.
            q = grabbed[0].shape[-1]
            total = (len(grabbed) - 1) * CHUNK + grabbed[-1].shape[0]
            buf = torch.zeros((total, q), dtype=torch.long)
            for i, g in enumerate(grabbed):
                a = i * CHUNK
                buf[a:a + g.shape[0]] = g.to(torch.long)
            out[utt_id] = buf
            np.savez_compressed(TOK_DIR / "{}.npz".format(utt_id),
                                tokens=buf.numpy())
            print("  captured {:<6} frames={} quantizers={} "
                  "({} streamer chunks)".format(
                      utt_id, total, q, len(grabbed)))
        model = service._model_registry.get_loaded_model()
        dec = model.model.speech_tokenizer.model.decoder
        return service, dec, out
    except Exception:
        await service.stop()
        raise


# --------------------------------------------------------------------------- #
# Bench
# --------------------------------------------------------------------------- #


def decode_whole(dec, codes):
    with torch.inference_mode():
        return dec(codes).squeeze(1).squeeze(0).to(torch.float32).cpu().numpy()


def decode_indep(dec, codes, chunk=CHUNK):
    outs = []
    T = codes.shape[-1]
    with torch.inference_mode():
        for a in range(0, T, chunk):
            c = codes[..., a:min(a + chunk, T)]
            outs.append(dec(c).squeeze(1).squeeze(0).to(torch.float32)
                        .cpu().numpy())
    return outs


def decode_stream(dec, codes, st, chunk=CHUNK):
    outs = []
    T = codes.shape[-1]
    with torch.inference_mode():
        for a in range(0, T, chunk):
            c = codes[..., a:min(a + chunk, T)]
            outs.append(stream_forward(dec, c, st).squeeze(1).squeeze(0)
                        .to(torch.float32).cpu().numpy())
    return outs


def main():
    print("=" * 78)
    print("Story 20.5 Phase 1 - codec state-caching bench")
    print("=" * 78)

    service, dec, tokens = asyncio.run(capture_tokens())
    try:
        _bench(dec, tokens, "bf16 -- SHIPPING PRECISION")
        # Numerical control. Every carried-state formulation below is
        # mathematically exact; the only way to tell "the state does not
        # determine the output" from "bf16 rounds differently when the same
        # arithmetic is split across two kernel launches" is to remove the
        # rounding. This casts the LOADED module in memory only.
        dec.float()
        _bench(dec, tokens, "fp32 -- NUMERICAL CONTROL")
    finally:
        asyncio.run(service.stop())
    return 0


def _profile(gt, chunks, label):
    """RMS error by position into the chunk, normalised by the ground truth's
    RMS in the same window. Story 20.4 measured the cold-start error as worst
    at the head and decaying over ~4,000 samples; a flat profile instead means
    broadband rounding, not a boundary defect."""
    bins = [0, 256, 512, 1024, 2048, 4096, 8192]
    acc = [[] for _ in range(len(bins) - 1)]
    off = 0
    for i, c in enumerate(chunks):
        if i > 0:
            for b in range(len(bins) - 1):
                lo, hi = bins[b], min(bins[b + 1], c.size)
                if hi <= lo or off + hi > gt.size:
                    continue
                e = c[lo:hi].astype(np.float64) - gt[off + lo:off + hi].astype(np.float64)
                r = gt[off + lo:off + hi].astype(np.float64)
                den = float(np.sqrt(np.mean(r * r)))
                if den > 0:
                    acc[b].append(float(np.sqrt(np.mean(e * e))) / den)
        off += c.size
    txt = "  ".join("{}-{}: {:.3f}".format(bins[b], bins[b + 1],
                                           float(np.median(acc[b])))
                    for b in range(len(bins) - 1) if acc[b])
    print("    {:<22} {}".format(label, txt))


def _bench(dec, tokens, tag):
    cfg = dec.config
    device = next(dec.parameters()).device
    dtype = next(dec.parameters()).dtype
    print("\n" + "#" * 78)
    print("# PASS: {}".format(tag))
    print("#" * 78)
    print("\ndecoder: device={} dtype={} upsample_rates={} "
          "upsampling_ratios={} total_upsample={}".format(
              device, dtype, tuple(cfg.upsample_rates),
              tuple(cfg.upsampling_ratios), int(dec.total_upsample)))
    print("transformer: layers={} hidden={} heads={} kv_heads={} "
          "sliding_window={} layer_types={}".format(
              cfg.num_hidden_layers, cfg.hidden_size,
              cfg.num_attention_heads, cfg.num_key_value_heads,
              cfg.sliding_window, set(cfg.layer_types)))

    # --- structural audit: is every CausalConvNet stride-1? --------------
    strides = set()
    tconv_shapes = []
    for m in dec.modules():
        if isinstance(m, CausalConv):
            strides.add((m.stride, m.padding, m.kernel_size))
        if isinstance(m, CausalTConv):
            tconv_shapes.append((m.conv.kernel_size[0], m.conv.stride[0],
                                 m.left_pad, m.right_pad))
    print("\nCausalConvNet (stride, padding, k_eff) set: {}".format(
        sorted(strides)))
    print("CausalTransConvNet (k, stride, left_pad, right_pad): {}".format(
        tconv_shapes))
    assert all(s[0] == 1 for s in strides), \
        "a stride>1 CausalConvNet exists; the streaming form needs revisiting"

    # analytic edge-loss decomposition
    rates = list(cfg.upsample_rates)
    contrib = []
    for i, r in enumerate(rates):
        remain = int(np.prod(rates[i + 1:])) if i + 1 < len(rates) else 1
        contrib.append((r, remain, r * remain))
    print("\nedge-loss decomposition (transposed-conv left/right trim):")
    for r, remain, c in contrib:
        print("   stride {:>2}  x downstream upsample {:>4}  = {:>4} samples"
              .format(r, remain, c))
    print("   TOTAL = {} samples   (measured constant: {})".format(
        sum(c[2] for c in contrib), EDGE_LOSS))

    # --- CUDA-graph / compile interaction audit -------------------------
    print("\ncompile / CUDA-graph state on the LOADED decoder:")
    print("   _compiled_forward set : {}".format(dec._compiled_forward is not None))
    print("   _compile_mode         : {}".format(dec._compile_mode))
    print("   _cuda_graph captured  : {}".format(dec._cuda_graph is not None))
    print("   _graph_window_size    : {}".format(dec._graph_window_size))

    results = {}
    for utt_id, tok in sorted(tokens.items()):
        codes = tok.t().unsqueeze(0).to(device)      # [T, Q] -> [1, Q, T]
        T = codes.shape[-1]
        print("\n" + "=" * 78)
        print("{}  frames={}  ({:.2f} s of audio)".format(
            utt_id, T, T * SAMPLES_PER_FRAME / SR))
        print("=" * 78)

        gt = decode_whole(dec, codes)
        gt2 = decode_whole(dec, codes)
        floor = nrmse(gt.astype(np.float64), gt2.astype(np.float64))
        print("  WHOLE decode length = {}  (1920*{} - {} = {})".format(
            gt.size, T, EDGE_LOSS, SAMPLES_PER_FRAME * T - EDGE_LOSS))
        print("  determinism floor (whole vs whole, same input) NRMSE = "
              "{:.3e}".format(floor))

        # ---- equivalence control: streaming code path, ONE chunk --------
        st1 = StreamState(True, True, True)
        one = decode_stream(dec, codes, st1, chunk=T)[0]
        print("  control: stream_forward as a single chunk vs WHOLE -> "
              "len {} vs {}, NRMSE {:.3e}".format(
                  one.size, gt.size,
                  nrmse(gt.astype(np.float64)[:min(one.size, gt.size)],
                        one.astype(np.float64)[:min(one.size, gt.size)])))

        # ---- arms --------------------------------------------------------
        arms = {}
        indep = decode_indep(dec, codes)
        arms["INDEP (ships today)"] = indep
        print("  INDEP per-chunk lengths (first 5): {} "
              "(1920*25-555 = {})".format(
                  [c.size for c in indep[:5]],
                  SAMPLES_PER_FRAME * CHUNK - EDGE_LOSS))

        matrix = [
            (False, False, False, "state: none"),
            (True, False, False, "state: conv"),
            (False, True, False, "state: tconv"),
            (False, False, True, "state: xformer"),
            (True, True, False, "state: conv+tconv"),
            (True, False, True, "state: conv+xformer"),
            (False, True, True, "state: tconv+xformer"),
            (True, True, True, "state: ALL"),
        ]
        states = {}
        for uc, ut, ux, label in matrix:
            st = StreamState(uc, ut, ux)
            arms[label] = decode_stream(dec, codes, st)
            states[label] = st

        all_chunks = arms["state: ALL"]
        print("  ALL-state per-chunk lengths (first 5): {}  (1920*25 = {})"
              .format([c.size for c in all_chunks[:5]],
                      SAMPLES_PER_FRAME * CHUNK))

        rows = []
        for label, chunks in arms.items():
            r = score_arm(gt, chunks, label)
            if r:
                r["floor"] = floor
                rows.append(r)
        results[utt_id] = rows

        print("\n  {:<22} {:>5} {:>10} {:>10} {:>9} {:>9} {:>10} {:>12}"
              .format("arm", "seams", "headNRMSE", "maxNRMSE", "corr med",
                      "corr min", "aligNRMSE", "lag delta"))
        for r in rows:
            print("  {:<22} {:>5} {:>10.4f} {:>10.4f} {:>9.4f} {:>9.4f} "
                  "{:>10.4f} {:>5.0f}..{:<5.0f}".format(
                      r["arm"], r["seams"], r["head_nrmse_med"],
                      r["head_nrmse_max"], r["corr_med"], r["corr_min"],
                      r["aligned_nrmse_med"], r["lag_delta_min"],
                      r["lag_delta_max"]))
        for r in rows:
            print("  {:<22} concat len {:>8}  vs GT {:>8}  delta {:+}"
                  .format(r["arm"], r["total_len"], r["gt_len"],
                          r["len_delta"]))

        # ---- whole-signal residual for the ALL arm ----------------------
        cat = np.concatenate(all_chunks)
        n = min(cat.size, gt.size)
        print("\n  ALL-state FULL-SIGNAL NRMSE vs WHOLE = {:.4e}  "
              "(determinism floor {:.3e})".format(
                  nrmse(gt.astype(np.float64)[:n], cat.astype(np.float64)[:n]),
                  floor))

        print("\n  error by position into the chunk "
              "(RMS error / GT RMS, median over seams):")
        _profile(gt, indep, "INDEP")
        _profile(gt, arms["state: conv+tconv"], "conv+tconv")
        _profile(gt, all_chunks, "ALL")

        # ---- transformer isolated (latent-level) -------------------------
        _transformer_probe(dec, codes)

        # ---- cost -------------------------------------------------------
        _cost(dec, codes, states["state: ALL"], cfg, dtype)

    _verdict(results)
    (SCRIPT_DIR / "20-5-state-cache-bench.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\nwrote 20-5-state-cache-bench.json")


def _transformer_probe(dec, codes):
    """Per-sub-stack attribution at the source: does a carried KV cache
    reproduce the whole-sequence transformer latents exactly?"""
    tf = dec.pre_transformer
    with torch.inference_mode():
        hidden = dec.quantizer.decode(codes)
        hidden = dec.pre_conv(hidden).transpose(1, 2)
        whole = tf(inputs_embeds=hidden).last_hidden_state
        T = hidden.shape[1]

        st = StreamState(True, True, True)
        parts = []
        for a in range(0, T, CHUNK):
            parts.append(s_transformer(tf, hidden[:, a:a + CHUNK], st))
        cached = torch.cat(parts, dim=1)

        st0 = StreamState(True, True, False)
        parts0 = []
        for a in range(0, T, CHUNK):
            parts0.append(s_transformer(tf, hidden[:, a:a + CHUNK], st0))
        naive = torch.cat(parts0, dim=1)

    w = whole.to(torch.float32).cpu().numpy().reshape(-1)
    c = cached.to(torch.float32).cpu().numpy().reshape(-1)
    n = naive.to(torch.float32).cpu().numpy().reshape(-1)
    print("  transformer latents, chunked vs whole:  KV-cached NRMSE "
          "{:.4e}   |   no-cache NRMSE {:.4f}".format(
              nrmse(w.astype(np.float64), c.astype(np.float64)),
              nrmse(w.astype(np.float64), n.astype(np.float64))))


def _cost(dec, codes, st, cfg, dtype):
    b, cnt, kvb, kvn = st.nbytes()
    esz = torch.finfo(dtype).bits // 8
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    bounded = (cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * hd
               * cfg.sliding_window * esz)
    print("\n  COST (AC #2)")
    print("    conv + tconv state : {} tensors, {:,} bytes ({:.1f} KiB)"
          .format(cnt, b, b / 1024.0))
    print("    KV cache observed  : {} tensors, {:,} bytes ({:.2f} MiB) "
          "[DynamicCache, unbounded growth]".format(kvb and kvn or kvn, kvb,
                                                    kvb / 1048576.0))
    print("    KV cache BOUNDED   : {:,} bytes ({:.2f} MiB) if a sliding "
          "window-{} cache is used (8 layers x K,V x {} heads x {} dim)"
          .format(bounded, bounded / 1048576.0, cfg.sliding_window,
                  cfg.num_key_value_heads, hd))
    print("    TOTAL per session  : {:.2f} MiB observed / {:.2f} MiB bounded"
          .format((b + kvb) / 1048576.0, (b + bounded) / 1048576.0))

    T = codes.shape[-1]
    reps = 3

    def timeit(fn):
        for _ in range(2):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    n_chunks = (T + CHUNK - 1) // CHUNK
    t_ind = timeit(lambda: decode_indep(dec, codes))
    t_str = timeit(lambda: decode_stream(dec, codes, StreamState(True, True, True)))
    print("    decode time, {} chunks: INDEP {:.1f} ms  STREAM {:.1f} ms  "
          "delta {:+.1f} ms ({:+.1f} %)  |  per chunk {:+.2f} ms".format(
              n_chunks, t_ind * 1e3, t_str * 1e3, (t_str - t_ind) * 1e3,
              100.0 * (t_str - t_ind) / t_ind, (t_str - t_ind) * 1e3 / n_chunks))


def _verdict(results):
    print("\n" + "=" * 78)
    print("GO / NO-GO (thresholds fixed before the work, AC #1)")
    print("=" * 78)
    print("  GO     : edge loss reaches zero AND head NRMSE < ~5 %")
    print("  NO-GO  : edge loss persists OR head NRMSE stays above ~15 %")
    for utt_id, rows in sorted(results.items()):
        ind = next((r for r in rows if r["arm"].startswith("INDEP")), None)
        alls = next((r for r in rows if r["arm"] == "state: ALL"), None)
        if not (ind and alls):
            continue
        print("\n  {}:".format(utt_id))
        print("    INDEP     head NRMSE {:.1%}  corr {:.3f}/{:.3f}  "
              "lag {:+.0f}..{:+.0f}  len delta {:+}".format(
                  ind["head_nrmse_med"], ind["corr_med"], ind["corr_min"],
                  ind["lag_delta_min"], ind["lag_delta_max"],
                  ind["len_delta"]))
        print("    ALL-state head NRMSE {:.2%}  corr {:.4f}/{:.4f}  "
              "lag {:+.0f}..{:+.0f}  len delta {:+}".format(
                  alls["head_nrmse_med"], alls["corr_med"], alls["corr_min"],
                  alls["lag_delta_min"], alls["lag_delta_max"],
                  alls["len_delta"]))


if __name__ == "__main__":
    raise SystemExit(main())
