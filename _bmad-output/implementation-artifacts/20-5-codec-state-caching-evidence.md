# Story 20.5 — Phase 1 bench evidence (AC #1 + AC #2)

Date: 2026-09-01 · Host: RTX 5090 / Win11 / torch 2.10+cu128 / transformers 4.57.3
Model: Qwen3-TTS 12 Hz tokenizer (`qwen3_tts_tokenizer_12hz`), quality tier, `tts_precision="auto"` → **bf16**, `tts_compile="auto"` → engaged.

**Verdict: GO on both stated thresholds, by a wide margin.**

---

## 0. Artifacts

| file | what |
|---|---|
| `20-5-state-cache-bench.py` | the bench: WHOLE / INDEP / 8-way state ablation, metrics, cost, CUDA-graph audit |
| `20-5-stage-probe.py` | stage-by-stage divergence probe (which sub-stack, which module) |
| `20-5-tokens/*.npz` | the captured codec token sequences (reused across runs so numbers are comparable) |
| `20-5-bench-run03.log` | the run the numbers below come from |
| `20-5-stage-probe2.log` | the post-fix stage probe (bit-exactness proof) |
| `20-5-state-cache-bench-bf16.json`, `-fp32.json` | machine-readable per-arm metrics |

Reproduce: `python310\python.exe _bmad-output\implementation-artifacts\20-5-state-cache-bench.py`

**No production code was modified.** `git diff --stat -- src/` is empty; `git status --porcelain -- src/ tests/ tools/ build_tools/` is empty. The bench holds a reference to the *loaded* decoder modules and re-implements the traversal of `Qwen3TTSTokenizerV2Decoder.forward`; it subclasses nothing, patches nothing, and copies no weights.

---

## 1. The mechanism, derived from the source before measuring

Three sub-stacks, three different boundary behaviours:

**Causal conv** (`Qwen3TTSTokenizerV2CausalConvNet`, `:159-192`). Every instance in the decoder is stride-1 — the bench asserts this — so `_get_extra_padding_for_conv1d` returns 0 and the module reduces to *left-pad `k_eff-1` zeros → conv → output length == input length*. Observed `(stride, padding, k_eff)` set: `[(1,0,1), (1,2,3), (1,6,7), (1,18,19), (1,54,55)]`. Streaming form: keep the last `k_eff-1` **input** samples as the next call's left context. Exact, and the largest single buffer is 54 samples deep.

**Transposed conv** (`Qwen3TTSTokenizerV2CausalTransConvNet`, `:195-208`). Built two ways. The two `upsampling_ratios=(2,2)` instances have kernel == stride, so `pad == 0` and they are **already stateless and exact** — no state needed. The four `upsample_rates=(8,5,4,3)` instances have kernel == 2·stride, so `left_pad == right_pad == stride`: the module conv-transposes and then discards `stride` samples at each end. That discard is the **entire** 555-sample edge loss:

```
stride  8 × downstream upsample  60 = 480
stride  5 × downstream upsample  12 =  60
stride  4 × downstream upsample   3 =  12
stride  3 × downstream upsample   1 =   3
                            TOTAL = 555   ← exactly the measured constant
```

The bench prints this decomposition and checks it against 555. Streaming form: overlap-add the discarded right tail into the next chunk's head.

**Transformer** (`Qwen3TTSTokenizerV2DecoderTransformerModel`, `:475`). 8 layers, hidden 512, 16 heads, `sliding_window=72`, and `layer_types` is `["sliding_attention"] * 8` — *every* layer is sliding. So its carried state is a **bounded** KV cache, not an unbounded one.

### 1.1 The trap that cost the first two runs — and would have cost Phase 2

The naive overlap-add is wrong. `nn.ConvTranspose1d` carries a **bias**, and it is added to every output position of *each* partial convolution — so summing the two partials double-counts it. Uncorrected, this leaves a bias-shaped transient at every seam that decays over ~2,000 samples and reads exactly like a residual "cold start": head NRMSE stuck at 17–21 % and *surviving an fp32 pass*, which is precisely the signature the go/no-go calls NO-GO ("the state we can reach does not determine the output").

It is not that. The stage probe localised the divergence to a single module — `decoder[1].block[1]`, the first stride-8 transposed conv — with everything upstream at the 1e-6 float floor. Subtracting one copy of `conv.bias` in the overlap region makes the whole thing bit-exact. **Phase 2 must carry this correction; without it the story looks like a failure.**

---

## 2. AC #1 — the gate

Chunk geometry: 25 frames, no lookahead, no overlap. Ground truth: `decoder(codes)` over the whole sequence. bf16 = shipping precision.

### 2.1 Edge loss — GONE

| arm | per-chunk decode length | concat length vs whole-sequence ground truth |
|---|---|---|
| INDEP (ships today) | `47445 = 1920·25 − 555` every chunk | **−555 per seam** (l-020: −4,995 over 9 seams) |
| state carried | first chunk `47445`, **every later chunk `48000 = 1920·25`** | **+0, exactly** |

`decode(N) == 1920·N − 555` becomes `decode(N) == 1920·N`. The 555 does not shrink — it *moves*, from every decode call to the single stream-start call, which is where the whole-sequence decode loses it too. That is why the totals match to the sample. Ablation shows this is **100 % attributable to the transposed conv**: every arm carrying tconv state hits +0; every arm without it stays at −555/seam, regardless of conv or transformer state.

### 2.2 Head fidelity — 130 % → 0.8 %

Median over seams, first 1024 samples (the Story 20.4 blend width), bf16:

| utterance | frames | seams | | INDEP head NRMSE | corr med/min | lag delta | | state-carried head NRMSE | corr med/min | lag delta |
|---|---|---|---|---|---|---|---|---|---|---|
| l-020 | 231 | 9 | | 144.0 % | 0.364 / 0.116 | −1200 … +1110 | | **0.82 %** | 1.0000 / 0.9916 | **0 … 0** |
| l-021 | 176 | 7 | | 115.2 % | 0.165 / 0.009 | −901 … +1165 | | **0.80 %** | 1.0000 / 0.9998 | **0 … 0** |
| m-020 | 44 | 1 | | 128.7 % | 0.959 / 0.959 | +891 … +891 | | **0.56 %** | 1.0000 / 1.0000 | **0 … 0** |

Against the Story 20.4 reference points: ~35 % NRMSE → **0.6–0.8 %**; 0.55 median / 0.11 min correlation → **1.0000 / 0.9916**; ±35 samples of lag jitter → **0 samples, on every seam of every utterance**.

*(The INDEP arm scores worse than 20.4's 35 % because the basis differs: 20.4 compared two decodes of the same frames to each other, this compares each chunk to the whole-sequence decode at the nominal splice — which also carries the 555-sample misalignment. The `aligNRMSE` column, which realigns first, gives 0.94/0.99/0.29 for the same seams. Both bases give the same answer.)*

### 2.3 Error by position into the chunk

RMS error / ground-truth RMS, median over seams (l-020, bf16):

| samples into chunk | 0–256 | 256–512 | 512–1024 | 1024–2048 | 2048–4096 | 4096–8192 |
|---|---|---|---|---|---|---|
| INDEP | 1.101 | 1.170 | 1.679 | 1.545 | 1.292 | 1.324 |
| conv+tconv, no KV | 0.156 | 0.237 | 0.251 | 0.340 | 0.308 | 0.364 |
| **all state** | **0.014** | **0.009** | **0.007** | **0.008** | **0.010** | **0.011** |

The state-carried profile is **flat**. That is the shape of rounding, not of a cold start. Story 20.4's "worst at the head, decaying over ~4,000 samples" signature is gone, not reduced.

### 2.4 Per-sub-stack attribution (AC #1's explicit requirement)

Full 2³ ablation, head NRMSE median, bf16 (l-020 / l-021 / m-020):

| state carried | l-020 | l-021 | m-020 | edge loss |
|---|---|---|---|---|
| none | 1.440 | 1.152 | 1.287 | −555/seam |
| conv only | 1.567 | 1.168 | 1.180 | −555/seam |
| tconv only | 1.036 | 0.908 | 0.838 | **0** |
| transformer only | 1.420 | 1.115 | 1.129 | −555/seam |
| conv + tconv | 0.244 | 0.213 | 0.211 | **0** |
| conv + transformer | 1.542 | 1.143 | 1.075 | −555/seam |
| tconv + transformer | 0.931 | 0.896 | 0.649 | **0** |
| **all three** | **0.008** | **0.008** | **0.006** | **0** |

Read as leave-one-out from the full arm (l-020): dropping **transposed-conv** state costs the most (0.008 → 1.54), then **causal-conv** state (→ 0.93), then the **transformer** KV cache (→ 0.24). All three are necessary; none is sufficient. Notably, conv state *alone* is slightly worse than no state at all — carrying left context without fixing the transposed-conv alignment just moves the error around.

Transformer measured in isolation, at the latent level, chunked vs whole:

| | KV cache carried | no cache |
|---|---|---|
| bf16 | 5.8e-03 | **0.300** |
| fp32 | 1.8e-04 | **0.300** |

### 2.5 Is 0.8 % the mechanism or the arithmetic? — fp32 and TF32-off controls

| precision | state-carried head NRMSE | full-signal NRMSE vs whole |
|---|---|---|
| bf16 (ships today) | 0.56 – 0.82 % | 9.9e-03 – 1.0e-02 |
| fp32 (TF32 on) | 0.03 – 0.05 % | 4.0e-04 – 1.8e-03 |
| fp32, TF32 off (stage probe, final output) | — | **7.7e-07** |

With the arithmetic made exact, the streaming decode is **bit-exact against the whole-sequence decode**. Every stage in the probe sits at 7e-07 – 2.6e-06. The residual 0.8 % in the shipping regime is bf16 rounding, it enters at the transformer stage (`pre_transformer_out` = 5.8e-03) and stays flat thereafter, and it is the *same* class of error a chunked decode has always had. The state we can reach **fully determines the output**.

### 2.6 Go / no-go, against the thresholds fixed before the work

| threshold | result |
|---|---|
| GO: edge loss reaches zero | **met** — `1920·N`, +0 samples vs ground truth, on all three utterances |
| GO: head NRMSE below ~5 % | **met** — 0.56–0.82 % bf16, 0.03–0.05 % fp32 |
| NO-GO: edge loss persists | not triggered |
| NO-GO: NRMSE above ~15 % | not triggered |

**GO.**

---

## 3. AC #2 — the true cost of carrying state

### 3.1 Size and shape

| component | tensors | bf16 | fp32 |
|---|---|---|---|
| causal-conv left context (≤ 54 samples deep) + transposed-conv overlap tails | 21 | 276.3 KiB | 552.6 KiB |
| transformer KV cache (bounded by `sliding_window=72`) | 16 | 2.25 MiB | 4.50 MiB |
| **total per session** | **37** | **≤ 2.52 MiB** | **≤ 5.04 MiB** |

The KV cache **self-bounds**: the config marks all 8 layers `sliding_attention`, so `DynamicCache(config=…)` allocates sliding-window layers that cap at 71 entries. Measured 2,326,528 B at 231 frames vs 2,359,296 B at the analytic cap — it stops growing, it does not accumulate over a long utterance. So per-session cost is **constant, ~2.5 MiB in the shipping bf16 regime**, independent of utterance length.

**Per-session, not global — and it must be.** All 37 tensors plus two scalar counters live in one plain object with no module-level or class-level storage. Concurrent generations (reachable via the HTTP API added this session) each get their own; nothing is shared. This is the one property that a monkey-patch implementation would break, and it is why the bench threads an explicit state object rather than patching module `forward` methods.

Reset points for Phase 2: session start, cancel, completion — one `state = None` each, no partial teardown.

### 3.2 Decode-time delta

3 timed repeats after 2 warmups, CUDA-synchronised, bf16, RTX 5090:

| utterance | chunks | INDEP | state-carried | delta |
|---|---|---|---|---|
| l-020 | 10 | 423.2 ms | 209.8 ms | −213.4 ms (−50.4 %) |
| l-021 | 8 | 137.5 ms | 103.3 ms | −34.2 ms (−24.9 %) |
| m-020 | 2 | 32.5 ms | 25.6 ms | −6.9 ms (−21.1 %) |

Carrying state is **faster**, by roughly **3.4–5.1 ms per chunk** (the −50 % l-020 figure is the first-timed case and includes cuDNN autotuning; the −21 %/−25 % figures are the trustworthy ones, and the fp32 pass reproduces −26 %/−30 %/−37 %).

**Caveat, stated plainly:** this is not all attributable to state. The streaming traversal also skips one `nn.Module.__call__` layer and the `F.pad` allocation on ~30 convs per chunk, and these are tiny tensors where launch overhead dominates. Part of the win is call overhead. The honest claim is the one that matters for the gate: **carrying state costs no decode time** — the trade this AC was written to price does not exist.

### 3.3 Subclass/wrapper, or vendoring?

**Wrapper. This is a story, not an architecture pass.**

The bench implements it as ~130 lines: a `StreamState` object plus a re-implementation of `Qwen3TTSTokenizerV2Decoder.forward`'s module walk that calls the loaded submodules' inner `nn.Conv1d` / `nn.ConvTranspose1d` directly. It reads upstream module *objects*; it copies no weights and forks no file.

Why not a plain subclass: the state has to thread through every conv and transposed conv, and those are reached through `nn.ModuleList` nesting inside `DecoderBlock` → `ResidualUnit`. There is no single method to override. Why not monkey-patching `forward` on the module classes: that state would be process-global, which fails the per-session requirement above.

The cost of the wrapper is that it restates the traversal, so an upstream pin bump that reorders `decoder`/`upsample` could silently desync it. The mitigation already exists in this codebase as a pattern: the Story 16.1/16.4 trip-wire test in `tests/test_qwen_tts_internals.py`. Phase 2 should extend it to pin the module chain the wrapper walks — and can assert the wrapper is correct *cheaply and exactly*, because §2.5 shows single-chunk streaming == `forward` bit-for-bit and fp32 chunked == whole to 7e-07. That is an unusually strong unit test for a change of this class.

### 3.4 Interaction with `enable_streaming_optimizations` / CUDA-graph capture

The AC anticipated a trade: carried state may be incompatible with graph capture, which would cost Story 18.4's decoder speedup. **Measured on the loaded model, that trade does not exist**, for a reason worth recording:

```
_compiled_forward set : True      (mode "reduce-overhead")
_compile_mode         : reduce-overhead
_cuda_graph captured  : False     (manual capture skipped — reduce-overhead
                                   already uses CUDA graphs internally)
_graph_window_size    : None
```

`_compiled_forward` is only ever reached through `Decoder.forward_optimized`, which is only called from `decode_padded` / `Qwen3TTSTokenizerV2Model.decode_streaming`. The production TRUE_STREAM path does not go there: `_build_true_stream_decode_fn` calls `speech_tokenizer.decode([{...}])` → `Qwen3TTSTokenizerV2Model.decode` → `decoder.chunked_decode` → plain `decoder.forward`. **The compiled decoder graph is not on our decode path today.** Story 18.4's measured win comes from the talker and code-predictor compiles, which this story does not touch.

So there is no decoder speedup to trade away, and a stateful wrapper that likewise bypasses `forward_optimized` loses nothing. Two forward-looking notes:

- Carried state makes the decoder **more** graph-friendly, not less: every chunk after the first is exactly 25 frames in → 48,000 samples out, a single static shape, where today the shipped path decodes 30 frames and the residual flush decodes a ragged one. A future manual capture would need the state tensors as static buffers, which is mechanical. Out of Phase 2 scope; worth knowing the door is open.
- The D-25 decode-window invariant stays decorative, exactly as Story 20.1 §5.4 recorded. Phase 2 as scoped does not move onto `decode_streaming`, so `decode_window_frames` still never reaches a runtime shape decision. If a later story does move there, 20.4's geometry threading is what makes it safe.

---

## 4. What this changes about how Phase 2 should be scoped

Not authorisations — inputs to the Commander's decision.

1. **The lookahead may become unnecessary, and that is a TTFA change, not just a quality one.** `CodecTokenStreamer` emits `chunk_size + lookahead` = 30 frames per chunk and `StreamingDecoderWorker._decode_and_post` trims. With state carried, a chunk decodes exactly its own 25 frames and emits exactly 48,000 samples with **zero trim** — the trim arithmetic that Story 20.4 found mis-modelled has no remaining job. Dropping lookahead to 0 also means the first chunk fires after 25 talker steps instead of 30. That is a real latency effect and it is *not* a chunk-size retune, so it does not collide with AC #5. It does need its own evidence.
2. **The Story 20.4 seam fix and the 64-sample consumer crossfade both become suspect at once.** AC #3 already says re-evaluate the 1024-sample blend rather than assume. Add the `StreamingChunkBuffer` crossfade to that list: both exist to mask a discontinuity that §2.2 shows is now 0-sample-aligned with correlation 1.0000. Evidence, not assumption — 20.4 round 3 showed the blend currently helps.
3. **Carry the bias correction or the story fails.** §1.1. It is one line and it is the difference between "bit-exact" and "17 % residual that survives fp32".
4. **The regression bar can be exact, not statistical.** Phase 2 can assert single-chunk-streaming == `forward` bit-for-bit and fp32 chunked == whole to ~1e-06. Most changes to a decode path cannot be tested that sharply.
5. **Precision interacts.** The 0.8 % residual is entirely bf16, and it enters at the transformer. If Phase 3's audition ever flags something, `tts_precision="fp32"` is a diagnostic lever that cleanly separates "state carry is wrong" from "bf16 rounds at the seam" — the bench already provides both arms.
6. **Cost is not a constraint.** ~2.5 MiB per concurrent session, bounded, constant in utterance length, and decode time does not regress. Nothing here needs a memory budget conversation.
