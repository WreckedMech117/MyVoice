# Story 20.5 — Phase 2 evidence (AC #3) + Phase 3 pre-registration (AC #4)

Date: 2026-09-01 · Host: RTX 5090 / Win11 / torch 2.10+cu128 / transformers 4.57.3
Model: Qwen3-TTS 12 Hz tokenizer, quality tier, `tts_precision="auto"` → **bf16**,
`tts_compile="auto"` → engaged.

Phase 1 evidence: `20-5-codec-state-caching-evidence.md`. This file covers the
production implementation and the audition that has not yet run.

**One variable.** Phase 2 carries codec state across chunk boundaries and does
nothing else. `DEFAULT_CHUNK_SIZE` stays at 25, `DEFAULT_LOOKAHEAD` stays at 5,
the post-decode trim is unchanged, the Story 20.4 1,024-sample seam blend stays
in, and the 64-sample `StreamingChunkBuffer` crossfade stays in. Both smoothing
layers were **evaluated on evidence** (§4) and neither was changed.

---

## 0. Artifacts

| file | what |
|---|---|
| `src/myvoice/services/tts_streaming/codec_state_cache.py` | the implementation (new) |
| `src/myvoice/services/tts_streaming/streaming_decoder.py` | worker: state-aware geometry + the three reset points |
| `src/myvoice/services/qwen_tts_service.py` | `_build_true_stream_decode_fn` prefers the state-cached decoder, falls back loudly |
| `tests/unit/services/tts_streaming/test_codec_state_cache.py` | 37 tests, incl. the exact regression bar and the bias trap |
| `tests/unit/services/tts_streaming/test_streaming_decoder.py` | +21 rows for the state-cached geometry and the reset points |
| `tests/test_qwen_tts_internals.py` | +9 trip-wire rows pinning the module chain the wrapper walks |
| `20-5-phase2-verify.py` / `20-5-phase2-verify.json` | this file's measurements, on the real model |
| `20-5-regen-audition-fixture.py` | Phase 3 fixture generator (paired-take A/B) |
| `20-5-l1-audition-helper.py` | Phase 3 audition helper, Commander solo |

---

## 1. What was implemented

`_build_true_stream_decode_fn` now returns a `StatefulCodecDecoder` in place of
the cold-state adapter, when three build-time gates pass. It re-walks
`Qwen3TTSTokenizerV2Decoder.forward`'s module traversal — calling the loaded
submodules' inner `nn.Conv1d` / `nn.ConvTranspose1d` directly — and threads one
per-session state object through every conv, transposed conv and the
transformer's KV cache. It subclasses nothing, patches nothing, copies no
weights and vendors no file, exactly as Phase 1 §3.3 predicted.

**The bias correction is carried.** `_stream_tconv` subtracts one copy of
`conv.bias` over the overlap region, because `nn.ConvTranspose1d` adds its bias
to every output position of each partial convolution. Phase 1 §1.1 recorded that
omitting it produces a 17–21 % residual that survives fp32 and reads exactly like
the NO-GO verdict. It is pinned by
`test_transposed_conv_bias_is_not_double_counted`, which runs the naive form
deliberately and requires it to be six orders of magnitude worse (§3.2).

### 1.1 The geometry, and the one asymmetry

The streamer emits `chunk_size + lookahead` = 30 frames per chunk and slides by
25, so chunk *k* covers frames `[25k, 25k+30)`. State must be committed at the
**splice** (frame `25k+25`), not at the end of the window, or chunk *k+1* would
resume five frames in its own future. `StatefulCodecDecoder.__call__` therefore
decodes in two passes: the first 25 frames with the live state, then a snapshot,
then the trailing 5 frames on the snapshot, then a restore.

The snapshot is **free** — O(1) tensor *references*, no copies. Every state slot
is reassigned rather than mutated in place, upstream included
(`DynamicSlidingWindowLayer.update` rebinds `self.keys` to a slice of a fresh
`torch.cat`), so holding the old references is a valid snapshot. That invariant
is pinned by a trip-wire
(`test_transformers_dynamic_cache_snapshot_contract_is_intact`), because if a
future transformers mutates in place the state would silently advance past the
splice and every chunk would skip 5 frames of audio.

Resulting identities, which the worker's splice arithmetic depends on:

| | first decode of a session | every later decode |
|---|---|---|
| decode length | `1920·N − 555` | **`1920·N`** |
| posted chunk | `25·1920 − 555` = 47,445 | **`25·1920`** = 48,000 |

The 555 does not shrink; it *moves* to the single stream-start call, which is
where the whole-sequence decode loses it too. Total for N frames is
`1920·N − 555` — identical to a single-shot decode, to the sample. Getting the
first splice wrong duplicates or drops 23 ms at the first seam of every
utterance and is invisible to every per-chunk check; that is the exact class of
defect Story 20.4 spent four audition rounds finding, and
`test_state_cached_stream_is_time_contiguous_through_every_seam` pins it.

### 1.2 AC #3's lifecycle requirement

State is strictly per session. One `CodecStreamState` per `StatefulCodecDecoder`
per dispatch; no module-level or class-level storage, so concurrent generations
via the HTTP API each get their own
(`test_state_is_per_instance_not_module_or_class_level` drives two decoders in
different orders and requires the second's output to be byte-identical to a
fresh one's).

`StreamingDecoderWorker` calls `decode_fn.reset()` at exactly the three points
the AC names:

| point | site |
|---|---|
| session start | `_run`, before the loop |
| completion | `_run`, on `END_OF_STREAM`, **before** the terminal post |
| cancel | `_drain_and_post_cancel`, after the drain, **before** the cancel post |

**The Story 16.5 cooperative-cancel chain is unaffected.** `_reset_codec_state`
never raises — it swallows and records `codec_state_reset_error` — precisely so
that a misbehaving reset cannot stop `('cancel', session_id)` reaching the
registry (P-7). `test_a_raising_reset_cannot_break_the_cancel_chain` installs a
reset that raises and requires the cancel post to happen anyway.
`test_codec_state_is_reset_on_cancel_before_the_cancel_post` pins the ordering.
No cancel-path control flow changed; the existing 20.4 cancel tests pass
unmodified.

### 1.3 Failure policy: decline loudly, never degrade silently

The wrapper restates upstream internals, so the only safe failure mode is to
decline and ship today's audio. Three build-time gates, any of which returns the
stateless adapter with a logged reason:

1. **Kill switch.** `MYVOICE_CODEC_STATE_CACHE=0`. Also how the Phase 3 fixture
   renders both arms from one build.
2. **Structural probe.** `probe_decoder` walks the whole module graph and
   refuses on: an unknown leaf type, a strided causal conv, a transposed conv
   whose kernel is neither `stride` nor `2·stride`, a non-sliding transformer
   layer, or a derived geometry that disagrees with `streaming_decoder`'s
   measured 1920 / 555. The unknown-leaf check matters most: without it a new
   time-mixing module would fall through the traversal's pointwise
   passthrough and corrupt audio at every boundary — a failure that is
   inaudible as a bug and audible only as "the codec got worse".
3. **Numerical self-test on the loaded weights** (§3.3), memoised per decoder
   object. In the shipping configuration the compile-priming generation at
   startup pays for it, so no user-visible generation does.

At runtime the length identity is re-checked on every chunk and a mismatch
raises rather than posting mis-spliced audio; and the worker's own
`decode_geometry_unverified` guard stays live on the new path
(`test_state_cached_geometry_violation_still_falls_back_and_says_so`).

### 1.4 One deliberate divergence from the stock path

`Qwen3TTSTokenizerV2Model.decode` truncates its output to
`(audio_codes[..., 0] > 0).sum(1) * decode_upsample_rate` — an encoder-side
heuristic for *padded batch* decode, where a zero code means padding. A
streaming chunk from the talker carries no padding, and Story 20.4 verified the
`1920·N − 555` identity on 14 independent residual lengths plus every full
chunk, so the clamp never fires. Replicating it would let a single legitimate
code-0 frame silently delete 1,920 samples of real speech mid-stream. The
wrapper does not replicate it. Recorded here rather than in a commit message.

---

## 2. Does the shipped path reconstruct the codec's true output?

Measured by `20-5-phase2-verify.py`. Ground truth is `decoder(codes)` over the
whole captured token sequence. Both arms are driven through a **real**
`StreamingDecoderWorker` with the real splice and the real overlap-add, from the
**same** captured token chunks, so the only difference is the decode.

Reference = stateless (ships today). Candidate = state-cached.

### 2.1 Length

| utterance | frames | chunks | ground truth | stateless Δ | state-cached Δ |
|---|---|---|---|---|---|
| l-020 | 231 | 10 | 442,965 | +0 | **+0** |
| l-021 | 176 | 7 | 337,365 | +0 | **+0** |
| m-020 | 44 | 2 | 83,925 | +0 | **+0** |

Both arms total correctly — Story 20.4's splice fix already ensured that. Length
was never the remaining defect; alignment and content were.

### 2.2 Content

| | l-020 | l-021 | m-020 |
|---|---|---|---|
| whole-signal NRMSE vs ground truth — **stateless** | 2.08e-01 | 6.03e-01 | 1.63e-01 |
| whole-signal NRMSE vs ground truth — **state-cached** | **9.88e-03** | **1.02e-02** | **7.91e-02** |
| improvement | 21× | 59× | 2.1× |

m-020's 7.9e-02 is not a state-carrying residual: its per-chunk profile is
`[0.106, 0.008]` — the error is in **chunk 0**, which is decoded from a genuinely
cold state on *both* arms because it is the start of the stream. m-020 opens
near-silent ("She sells seashells…"), so NRMSE against a low-RMS reference reads
high. Its error-by-position profile (§2.4) is flat, which is the rounding
signature, not the cold-start one.

### 2.3 Seam fidelity — the Story 20.4 measurements, repeated

First 1,024 samples after each seam (the blend width), median over seams:

| | head NRMSE | correlation med / min | lag delta |
|---|---|---|---|
| **stateless** l-020 | 0.406 | 0.928 / 0.732 | +0 … +368 |
| **state-cached** l-020 | **0.0078** | **1.0000 / 0.9937** | **+0 … +0** |
| **stateless** l-021 | 0.477 | 0.888 / 0.811 | +0 … +114 |
| **state-cached** l-021 | **0.0075** | **1.0000 / 0.9999** | **+0 … +0** |
| **stateless** m-020 | 0.220 | 0.985 / 0.985 | +1 … +1 |
| **state-cached** m-020 | **0.0055** | **1.0000 / 1.0000** | **+0 … +0** |

Against Story 20.4's reference points (~35 % NRMSE, 0.55 median / 0.11 min
correlation, ±35 samples of lag jitter): head error falls **50–60×**, correlation
reaches 1.0000 on every seam of every utterance, and **lag jitter is exactly
zero on every seam** — the streamed audio is sample-aligned with what a
single-shot decode would have produced.

### 2.4 Error by position into the chunk — the regime discriminator

RMS error / ground-truth RMS, median over seams. Phase 1 §2.3's test: a **cold
start** is head-weighted and decays over ~4,000 samples; **rounding** is flat.

| samples into chunk | 0–256 | 256–512 | 512–1024 | 1024–2048 | 2048–4096 | 4096–8192 |
|---|---|---|---|---|---|---|
| l-020 stateless | 0.151 | 0.289 | 0.538 | 0.422 | 0.297 | 0.278 |
| l-020 **state-cached** | **0.012** | **0.009** | **0.007** | **0.008** | **0.010** | **0.011** |
| l-021 stateless | 0.162 | 0.379 | 0.665 | 0.530 | 0.503 | 0.259 |
| l-021 **state-cached** | **0.005** | **0.005** | **0.006** | **0.018** | **0.018** | **0.017** |
| m-020 stateless | 0.131 | 0.163 | 0.264 | 0.292 | 0.314 | 0.220 |
| m-020 **state-cached** | **0.015** | **0.008** | **0.005** | **0.006** | **0.006** | **0.013** |

The state-cached profile is flat on all three. Story 20.4's cold-start signature
is removed, not reduced — the Phase 1 bench result reproduced by the shipped
code on the production path.

---

## 3. The regression bar — exact, not statistical

Phase 1 §2.5 established that a streaming decode with state carried is bit-exact
against `forward` in fp32 and identical to the whole-sequence decode to the float
floor. That, not a tolerance, is what the test suite holds the implementation to.

### 3.1 On a real decoder, in CI, on CPU

`test_codec_state_cache.py` builds a genuine `Qwen3TTSTokenizerV2Decoder` from
the real upstream classes — tiny (2 transformer layers, latent 8, 8
samples/frame, 6-sample edge loss) with randomised weights — so the exactness
assertions run in float64 on CPU in 3 seconds. Nothing there is a mock of the
decoder; a mock could not detect the one bug that matters.

| assertion | result |
|---|---|
| single-chunk streaming == `decoder.forward` | **`torch.equal` — bit-for-bit** |
| chunked with state == whole sequence, chunk ∈ {5, 8, 10, 13} | 2.8e-16 – 3.4e-16 (float64 floor) |
| stitched worker output == whole-sequence decode, cs=10 la=3, 43 frames | ≤ 1e-06 |
| retained overlap tail == next chunk's head | **`assert_array_equal` — bit-for-bit** |
| independent decodes lose `edge_loss` per seam | exact |
| stateless control still shows the cold-start defect | 0.377 head NRMSE — the fixture reproduces Story 20.4's ~35 % |

### 3.2 The bias trap, pinned as its own row

`test_transposed_conv_bias_is_not_double_counted` monkeypatches the naive
overlap-add in and requires it to be catastrophically worse:

| | NRMSE vs whole-sequence | length |
|---|---|---|
| corrected (shipped) | 3.2e-16 | correct |
| naive — bias double-counted | **2.2e-02** | **correct** |

Seven orders of magnitude apart, and note the second column: **the naive form
gets the length right**, which is exactly why it is dangerous — every geometry
check still passes and the failure presents as "state caching does not work".

### 3.3 On the loaded model, at build time

The runtime self-test measured on this host (bf16, RTX 5090):

| | random 24 frames | random 50 frames | real l-020 tokens, 50 frames |
|---|---|---|---|
| single-call streaming vs `forward` | **0.0 (bit-exact)** | **0.0 (bit-exact)** | **0.0 (bit-exact)** |
| chunked with state vs `forward` | 5.1e-02 | 3.3e-02 | **8.6e-03** |
| naive bias vs `forward` | 6.0e-02 | 5.9e-02 | 3.5e-02 |
| no state (INDEP) vs `forward` | 9.4e-01 | 1.13e+00 | 1.07e+00 |

Two things this table settles.

**The traversal is bit-exact against `forward` on the real model in bf16.** That
is a stronger statement than Phase 1 made and it is precision-independent, so it
is the gate the self-test is built on.

**And a limitation, stated rather than papered over.** On *random* codes the
bf16 floor (3.3e-02 – 5.1e-02) is too close to the bias bug's signature
(5.9e-02) to separate them, and the self-test has no real tokens at build time.
So the runtime self-test does **not** gate the bias trap. That is acceptable
because the bias trap is a code defect, not a model mismatch — it cannot appear
at runtime without the source changing — and §3.2 pins it in CI with six orders
of margin. An earlier draft used a 5e-02 tolerance that would have *failed on
correct code* (measured 5.13e-02 on the first real run); the fix was to
re-derive the gate, not to loosen the number.

### 3.4 The trip-wire extension

Nine rows added to `tests/test_qwen_tts_internals.py`, pinning what the wrapper
depends on and could otherwise lose silently: the eight module classes the
traversal dispatches on; the seven fragments of `Decoder.forward` that
`stream_forward` restates; the zero left-pad that is this story's whole premise;
the `[left_pad : −right_pad]` discard that owns 100 % of the 555; the five
decoder attribute names; `layer_types == ["sliding_attention"] * 8`; the
`upsample_rates` arithmetic that yields 1920 and 555; `Model.decode` still
routing through `chunked_decode` and *not* `forward_optimized` (Phase 1's
no-CUDA-graph-trade finding); and the `DynamicCache` rebind-not-mutate contract
the free snapshot depends on.

---

## 4. AC #3's other requirement: the two smoothing layers, on evidence

AC #3 says the Story 20.4 seam fix must be **re-evaluated, not assumed**. Phase 1
§4.2 added the 64-sample consumer crossfade to that. Both were measured. **Neither
was changed** — that is a follow-up with its own audition, and bundling it here
would repeat the two-variable confound that cost Story 20.4 its round-2 audition.

### 4.1 The Story 20.4 1,024-sample decoder seam blend — now inert

The blend cross-fades the tail a chunk retains past its splice against the head
of the chunk that follows. Under carried state the retained tail is decoded from
the *same state snapshot* the next chunk resumes from, so the two are the same
audio.

| how different are the blend's two inputs? (NRMSE, median over seams / max) | l-020 | l-021 | m-020 |
|---|---|---|---|
| stateless (ships today) | 0.472 / 0.987 | 0.556 / 0.591 | 0.223 / 0.223 |
| **state-cached** | **0.0037 / 0.087** | **0.0044 / 0.012** | **0.0046 / 0.0046** |

The blend's inputs now agree to about **−48 dB median**, a ~125× reduction. On
CPU in float64 they are bit-for-bit identical
(`test_overlap_add_is_an_identity_under_carried_state`); the residual on GPU is
bf16 kernel-selection noise between a 25-frame pass and a 5-frame pass over the
same frames.

**Verdict: the blend is now an identity operation.** It neither helps nor harms.
Leaving it in place costs nothing measurable and keeps Phase 2 to one variable;
removing it would change the output by ~0.4 % over the 2.1 % of the stream inside
a blend window. It should be removed eventually — dead weight in the hot path —
but there is no quality argument either way, and no reason to spend an audition
on it alone.

### 4.2 The 64-sample consumer crossfade — now the dominant error term

`StreamingChunkBuffer._apply_crossfade_and_update_tail` blends the **last** 64
samples of one released chunk with the **first** 64 of the next. Those are
*different moments in time*: sample `n+i` gets mixed with sample `n−64+i`. On a
genuine discontinuity that bridges a step. On continuous audio it is a 2.7 ms
comb over material that needed no repair.

Measured against ground truth — does it move the output toward or away from what
the codec actually produced?

| arm | samples touched | max │Δ│ | NRMSE vs ground truth, crossfade **off** | crossfade **on** | effect |
|---|---|---|---|---|---|
| l-020 stateless | 566 / 442,965 (0.128 %) | 7,949 LSB | 0.208193 | 0.209932 | +0.8 % worse |
| l-020 **state-cached** | 565 / 442,965 (0.128 %) | 8,123 LSB | **0.009876** | **0.025432** | **2.6× worse** |
| l-021 stateless | 378 / 337,365 (0.112 %) | 10,910 LSB | 0.602772 | 0.603511 | +0.1 % worse |
| l-021 **state-cached** | 377 / 337,365 (0.112 %) | 11,518 LSB | **0.010198** | **0.033743** | **3.3× worse** |
| m-020 stateless | 63 / 83,925 (0.075 %) | 4,218 LSB | 0.163148 | 0.163729 | +0.4 % worse |
| m-020 **state-cached** | 63 / 83,925 (0.075 %) | 3,250 LSB | **0.079059** | **0.079474** | +0.5 % worse |

Read carefully. The crossfade was *already* mildly harmful on the shipped arm —
it is applied at the consumer buffer's **release** boundaries, which since Story
20.4's decoder-side blend have been continuous audio anyway — but its harm was
invisible under a cold-start error 20× larger. Remove the cold start and the
crossfade becomes **the largest remaining deviation from the codec's true
output**: it triples the error on both long fixtures while touching 0.1 % of
samples, with local excursions up to 0.35 full scale.

**Verdict: this one should probably go, and it is a real change with a real
expected effect** — unlike §4.1. It is therefore exactly the kind of thing that
must not ride along on this audition. Recorded as the primary Story 20.5
follow-up: *remove or shorten the 64-sample `StreamingChunkBuffer` crossfade,
audition it on its own.* It is one constant
(`streaming_chunk_buffer.py:94 crossfade_samples=64`) and the class already
supports 0.

Note also that it is **not** load-bearing for underrun smoothing — that job
belongs to the 500 ms watermark (Story 18.x), which is untouched.

---

## 5. Cost, measured on the shipped path

### 5.1 Memory — as Phase 1 predicted

| | tensors | bytes |
|---|---|---|
| after 4 chunks (KV sliding window full) | 37 | 2,609,472 (2.49 MiB) |
| after all 10 chunks (231 frames) | 37 | 2,609,472 (2.49 MiB) |

**Bounded: identical.** The KV cache self-bounds at `sliding_window=72`, so
per-session cost is constant in utterance length. 2.49 MiB per concurrent
session; nothing here needs a memory budget conversation.

### 5.2 Decode time — a regression, and it is the lookahead's fault

This is where the shipped implementation **differs from the Phase 1 bench**, and
the difference is worth stating plainly rather than burying.

| utterance | chunks | stateless | state-cached | Δ | per chunk |
|---|---|---|---|---|---|
| l-020 | 10 | 199.4 ms | 285.0 ms | +42.9 % | **+8.6 ms** |
| l-021 | 7 | 150.3 ms | 223.0 ms | +48.4 % | **+10.4 ms** |
| m-020 | 2 | 63.7 ms | 77.6 ms | +21.8 % | **+7.0 ms** |

Best of two timed passes each, after a warm-up pass on the same shapes.

Phase 1 §3.2 measured carried state as **21–30 % faster**. It was — but its bench
decoded 25 frames in **one** call with no lookahead. The shipped path must
preserve the 5-frame lookahead and the post-decode trim (this story's explicit
constraint), so each chunk is decoded in **two** passes: 25 frames committed,
then 5 frames on a snapshot that is thrown away. The decoder is launch-overhead
dominated at these tensor sizes, so two calls cost close to twice the fixed
overhead of one. That is the whole of the +8.6 ms.

Is it acceptable? Yes, and the numbers say why:

- A chunk carries **2,000 ms** of audio. Decode goes from ~20 ms to ~29 ms per
  chunk — **1.0 % → 1.5 % of real time**. Story 18.1 pinned the producer
  bottleneck at the talker (31 % of real time, ratio 3.23×); this moves that
  ratio by about 0.02.
- TTFA takes the hit once: the first chunk's decode is ~9 ms longer against a
  measured 1,491 ms long-form TTFA — **+0.6 %**.

And it is recoverable, two ways, both of which are follow-ups rather than this
story:

1. **Drop the lookahead** (Phase 1 §4.1). With state carried, a chunk decodes
   exactly its own 25 frames and the trim has no remaining job. That deletes the
   second pass — decode becomes *faster* than today, per Phase 1 — and fires the
   first chunk after 25 talker steps instead of 30, which is a TTFA win on top.
   Blocked here only because it also removes the 1,024-sample blend's input,
   which §4.1 shows is inert but which this story is not changing.
2. **Single-pass state capture.** The frame-25 boundary maps to a known sample
   index at every stage's resolution, so the state could be sliced out mid-pass
   instead of re-running 5 frames. More code, more risk, no behavioural change.
   Recorded, not recommended yet.

### 5.3 CUDA graphs — Phase 1's finding holds

`_compiled_forward` is set but is only reachable through `forward_optimized`,
which `chunked_decode` does not call. The compiled decoder graph is not on
MyVoice's decode path, so nothing is traded away. Pinned by
`test_v2_model_decode_still_routes_to_chunked_decode_then_forward`.

---

## 6. AC #6 — regressions

| suite | result |
|---|---|
| `tests/unit/services/tts_streaming/` (incl. 37 new + 21 new rows) | pass |
| `tests/test_qwen_tts_internals.py` (18, incl. 9 new) | pass |
| `tests/unit/services/test_qwen_tts_service_dispatch.py`, `…_true_stream_callback.py`, `…_true_stream_instrumentation.py`, `test_compile_priming_audio_suppression.py`, `test_decode_window_geometry_coherence.py`, `test_streaming_chunk_buffer.py` | pass (344 total with the above) |
| `tests/integration/` | 166 pass, **4 fail** |

The four integration failures are **pre-existing and unchanged in count and
identity** — verified by `git stash`-ing the entire change and re-running:
`test_emotion_preset_instructs_match_qwen_service`,
`test_qwen_service_emotions_have_instructs`, `test_generate_v2_metadata`,
`test_audio_chunk_field_set_unchanged`. None touches streaming.

**One test-side change was required.** `_build_true_stream_decode_fn` grew two
optional geometry parameters (the state-cached decoder commits at the streamer's
splice, so it must be built *from* the streamer's geometry rather than from
module defaults). Fifteen test doubles of that builder took `(model)` only and
were widened to `(model, *_geometry, **_kwargs)`. They return a stateless
decode_fn with no `carries_codec_state` attribute, so those tests keep exercising
the pre-20.5 geometry model — which is correct: they are about callback and
metric wiring, not about the codec.

---

## 7. Phase 3 (AC #4) — pre-registration

**Nothing has been auditioned yet.** This section is written *before* the round,
which is the practice that made Story 20.4 rounds 3 and 4 readable.

### 7.1 The arms

    reference = cs25 + the Story 20.4 seam fix                 <- what ships today
    candidate = cs25 + the Story 20.4 seam fix + state caching

Both carry the same geometry (25, 5), the same 1,024-sample decoder blend and the
same 64-sample consumer crossfade. Both are rendered through a real
`StreamingDecoderWorker` and a real `StreamingChunkBuffer`, from one model load,
in one process. The generator's preflight refuses to run if `DEFAULT_CHUNK_SIZE`
or `DEFAULT_LOOKAHEAD` has drifted.

Same seven utterances as Story 20.4 rounds 1–4, so every round stays comparable.

### 7.2 The variance requirement — answered by removing the confound, not averaging it

AC #4 offers two options: multiple takes per condition, or state plainly that the
round can only detect a large effect. This round takes a third, which Story 20.4
could not:

> **Both files in every pair are decoded from one talker run.**

The talker runs once per take; its codec-token chunks are captured and decoded
twice. Within a pair the wording, prosody, pauses and total duration are
identical *to the sample*; the only difference is whether the decoder carried
state. Story 20.4's arms were necessarily different takes — a chunk-size change
perturbs what the streamer emits and therefore what the talker samples — and
§17 recorded the consequence: the same configuration flagging differently across
takes. Nothing upstream of the decoder is touched by this story, so the tokens
are literally reused.

So a single pair per utterance is already sufficient for **attribution**: any
difference heard is caused by the decode, because there is nothing else it could
be caused by. That is a stronger position than either option the AC offered, and
it means the round is **not** limited to detecting a large effect.

**Two takes per utterance are still generated**, for a different reason: to
sample the *content* lottery — whether a held vowel or a plosive happens to land
on a boundary is luck, and one take might contain no seam-sensitive material at
all. 7 utterances × 2 takes = **14 trials**.

Levels are deliberately **not** normalised. Story 20.4 round 4 normalised because
its arms were different takes that differed by 8 dB; here they share a take, so
any level difference is *caused by the change* and is a finding. The generator
prints the within-pair delta and warns above 0.2 dB.

### 7.3 The falsifiable prediction, recorded before listening

- **P1 (BLOCKING).** No blocking seam defect — `audible_seam`,
  `click_or_discontinuity`, `prosody_break_at_stitch` — on any candidate trial.
  *Falsified by one.*
- **P2 (DIRECTION).** Where the two differ, the candidate is preferred; the
  reference is never preferred on seam grounds. *Falsified if the reference is
  preferred on ≥ 4 of 14 trials.*
- **P3 (MAGNITUDE).** `equivalent` is the modal answer, **6–11 of 14 trials.**
  Story 20.4 round 3 already certified cs25 + blend as clean to this listener, so
  there is little audible headroom left at this seam density — the blend was
  masking the defect adequately at 25. *Falsified in either direction:* ≥ 10
  candidate-preferred is a large effect; ≤ 4 equivalent means the arms are far
  more distinguishable than predicted and the round needs re-reading before any
  conclusion.
- **P4 (LOCATION).** Any heard difference is on l-020 / l-021 (8–9 seams).
  s-020 / s-021 / s-022 carry 0–1 seams and should be indistinguishable.
  *Falsified by a difference heard on a short fixture but not a long one.*

**P3 is the interesting one, and it predicts against the exciting outcome.** The
offline numbers in §2 are dramatic — 50–60× less seam error, zero lag jitter —
and it would be easy to assume that must be audible. It probably is not, at
cs25, because Story 20.4 already established that the 1,024-sample blend masks
this defect well enough at this seam density for Commander to call it clean.
A large effect *is* plausible — Phase 1 removed the cause rather than masking it,
and masking is never perfect — but it is offered here as a prediction to be
falsified, not as an expectation.

A round that returns "equivalent everywhere, no new defects" is a **PASS**. The
prize is not audible quality at cs25; it is that the harm which killed cs10 is
gone at the cause, and AC #5 reopens the chunk-size question as its own story.
If P3 *is* falsified by a large candidate win, that is itself evidence the blend
was masking less well than round 3 suggested — and it strengthens the AC #5 case.

### 7.4 Verdict gate (blocking, per AC #4)

FAIL if any chunk-boundary artefact is flagged on a candidate trial that the
paired reference does not also carry. A defect on **both** files of a pair is
upstream of the decode — here demonstrably, since they are the same take — and is
recorded rather than blocking.

### 7.5 The fixture, generated and checked

Generated 2026-09-01: **28 WAVs, 14 trials**, in
`20-5-perceptual-fixtures/`, with `_perlistener_truthtable.json` and
`_manifest.json`. Chunk counts run from 1 (s-020 take 1) to 10 (l-020),
i.e. 0 to 9 seams — the span the round needs.

Four validity checks on the generated files, before anyone listens:

| check | result |
|---|---|
| length delta within each pair | **+0 samples on all 14** — the arms share a timeline exactly |
| full-signal RMS delta within each pair | −0.21 dB … +0.19 dB (worst 0.21 dB) |
| the zero-seam trial (s-020 take 1, 29 frames → 1 chunk) | **byte-identical between arms** |
| where the arms differ | 36–72 % of the squared difference falls within ±4,096 samples of a seam, which is 9–16 % of the timeline — a 2.5–5.8× concentration |

The last two are the ones worth reading. **s-020 take 1 has no chunk
boundary, and the two arms come out bit-identical** — a built-in control
proving the change touches nothing but the boundaries. And the difference
energy concentrates around the seams and decays over ~4,000 samples, which is
Story 20.4 §13.2's cold-start profile: the fixture is carrying the defect
under test and not something else.

The generator's own level warning fired once (m-021 take 2, +0.317 dB) and is
a **false alarm**: that figure comes from a −50 dBFS-gated active-speech mean,
and frames sitting on the gate flip in and out between arms. The ungated
full-signal delta for the same pair is +0.19 dB, in line with the rest.
Nothing was normalised.

### 7.6 What the operator runs

The fixture is already generated. Commander runs one command:

```
python310\python.exe _bmad-output\implementation-artifacts\20-5-l1-audition-helper.py L1
```

14 trials, headphones, normal Discord-call volume; trial A then trial B, `[r]`
replays, `[q]` quits and keeps what is recorded. Results append to
`20-5-state-cache-audition.csv` and re-running skips rows already recorded.
The helper unblinds at the end and scores the result against P1–P4 itself.

To regenerate the fixture from scratch (new takes):

```
python310\python.exe _bmad-output\implementation-artifacts\20-5-regen-audition-fixture.py
```

To reproduce this file's measurements:

```
python310\python.exe _bmad-output\implementation-artifacts\20-5-phase2-verify.py
```

---

## 8. Follow-ups this story deliberately did not take

| # | item | why it was left | evidence |
|---|---|---|---|
| 1 | Remove the 64-sample consumer crossfade | a real change with a real expected effect — needs its own audition | §4.2 |
| 2 | Drop the 5-frame lookahead and the post-decode trim | recovers the +8.6 ms/chunk *and* fires the first chunk 5 talker steps earlier; also removes the (inert) blend's input | §5.2, Phase 1 §4.1 |
| 3 | Remove the 1,024-sample decoder seam blend | now provably an identity — dead weight, no quality argument either way | §4.1 |
| 4 | Reopen `chunk_size` | AC #5's own story, gated on Phase 3 passing | AC #5 |
| 5 | Single-pass state capture | recovers the decode-time cost without touching the trim contract | §5.2 |
