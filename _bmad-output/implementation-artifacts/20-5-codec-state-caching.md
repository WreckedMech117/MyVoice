# Story 20.5: Codec State Caching Across Chunks (Phase ⊥-Polish-3)

Status: in-progress — Phase 1 GO; Phase 2 COMPLETE and verified; Phase 3 round 1 MIXED (2 blocking rows, cause named pre-audition); Phase 4 neutralises that cause; round 2 fixture built, audition NOT yet run

<!-- Phase tag: Phase ⊥-Polish-3. Fifth story of Epic 20. -->
<!-- Source: Mary's research Finding 1 (re-filed as audio-quality per 20-4 evidence §11.9); Story 20.4 §11/§13/§17. -->
<!-- Story class: PHASE-GATED. Phase 1 is a bounded bench spike with a hard go/no-go. Phases 2-3 do not start until it passes. -->
<!-- Risk: HIGH. Forks upstream decoder internals and reopens the audited decode path. The gate exists so we find out cheaply. -->

## Story

As **a MyVoice user generating long-form speech**,
I want **chunk boundaries to be inaudible because they carry the codec's real state, not zeros**,
so that **streamed audio sounds like one continuous take rather than pieces stitched together**.

## Context — this is now a quality item, not a throughput one

Mary's 2026-08-31 research filed codec state caching as a **speed** optimisation,
from Nari's technique list: *"maintains cached Transformer context and convolutional
state across chunks… avoids replaying the full history."* Story 20.4 established it is
also — and for us primarily — an **audio-quality** item.

**What Story 20.4 measured.** Consecutive chunks decode the same lookahead frames
independently, and the two renditions differ by **~35 % NRMSE**. In the region that
matters — the head of the next chunk — correlation is **0.55 median falling to 0.11**,
with **±35 samples (1.5 ms) of lag jitter**. Decode error by position into the chunk is
worst at the very start and decays over roughly 4,000 samples.

**Why.** `Qwen3TTSTokenizerV2CausalConvNet.forward`
(`modeling_qwen3_tts_tokenizer_v2.py:189-192`) does:

```python
hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
```

with `self.padding = kernel_size - stride`. **Every chunk's decode begins with zeros
where the previous chunk's real audio should be.** That single fact produces both
defects Story 20.4 chased for four audition rounds:

- the deterministic **555-sample fixed edge loss** (`decode(N) == 1920·N − 555`), which
  the shipped trim mis-modelled as proportional and which deleted **19.3 ms of real
  speech at every boundary** from Story 16.4 until Story 20.4 fixed it, and
- the **cold-start error** at each chunk head, which the current seam fix *masks* with a
  1024-sample blend rather than removing.

**Why it matters commercially.** Story 20.4 closed the chunk-size question at `cs25`
because `cs10` regressed perceptually — and the harm that killed `cs10` **is this
residual**, scaled by seam count. Headless, `cs10` measured **829 ms against `cs25`'s
1,491 ms — 44 % faster**. Removing the cause is what makes that latency reachable.
Story 20.4 §17 states this explicitly as the thing that would reopen the question.

## Phase gate

**Phase 1 is a bench spike. Phases 2 and 3 do not begin until Commander approves the
Phase 1 result.** This mirrors Story 20.1, which is the reason Epic 20 has been able to
kill bad ideas cheaply.

## Acceptance Criteria

### AC #1 — Phase 1: prove the mechanism offline, on a bench (the gate)

**Given** the causal convs left-pad with zeros, and the decoder also contains a causal
**transformer** stack (`Qwen3TTSTokenizerV2DecoderTransformerModel`) and a
**transposed**-conv upsampling path (`Qwen3TTSTokenizerV2CausalTransConvNet`), each with
its own boundary behaviour
**When** the developer builds a bench that decodes a known token sequence (a) whole, and
(b) in chunks with state carried across the boundary
**Then** it reports, against the whole-sequence decode as ground truth:
  - whether `decode(N) == 1920·N − 555` becomes `1920·N` — i.e. the fixed edge loss is
    **gone**, not merely smaller
  - the NRMSE at the chunk head, against the current **~35 %**
  - the correlation and lag jitter in the first 1024 samples, against the current
    **0.55 median / 0.11 min / ±35 samples**
**And** it states which of the three sub-stacks (causal conv, transposed conv,
transformer) each remaining discrepancy is attributable to, rather than reporting one
aggregate number
**And** **no production code is modified in Phase 1** — the bench may subclass, monkey-
patch, or vendor a copy, but `src/myvoice` is untouched

> **These metrics are legitimate here, and that is a deliberate distinction.** Story 20.4
> §13.1 established that offline metrics do **not** predict audibility and cannot gate
> AC #5. These do not attempt to. They measure whether the *cause* is removed — the edge
> loss is deterministic and the ground truth is exact. Mechanism metrics gate Phase 1;
> only the ear gates Phase 3.

**Go/no-go, stated before the work:**
- **GO** if the edge loss reaches zero **and** head NRMSE drops below ~5 %
- **NO-GO** if the edge loss persists, or NRMSE stays above ~15 % — meaning the state we
  can reach does not determine the output, and masking remains the only option
- **In between** — report and let Commander decide; do not self-authorise Phase 2

### AC #2 — Phase 1: report the true cost of carrying state

**Given** state caching trades memory and complexity for continuity
**When** the bench works
**Then** it reports: the per-session state size, the number of tensors that must be
carried, whether state is per-session or global (it must be per-session — concurrent
generations are possible via the HTTP API added this session), and the decode-time delta
**And** it states plainly whether the change is expressible as a **subclass/wrapper** or
requires **vendoring** decoder internals, since that decides whether this is a story or
an architecture pass
**And** it names the interaction with `enable_streaming_optimizations` / CUDA-graph
capture: the fork's `decode_streaming` and `capture_cuda_graph` assume a fixed window
with no carried state, and `decode_padded` pads inputs — carried state may be
incompatible with graph capture, which would trade Story 18.4's decoder speedup for
quality

### AC #3 — Phase 2 (gated): implement on the real decode path

**Given** a GO verdict
**When** implemented
**Then** state is carried across chunks in `_build_true_stream_decode_fn`'s decode path,
scoped **per session**, and reset on session start, cancel, and completion
**And** the Story 20.4 seam fix is **re-evaluated, not assumed**: if state caching makes
boundaries continuous, the 1024-sample blend is masking a defect that no longer exists
and should be reduced or removed — but only on evidence, since Story 20.4 round 3 showed
it currently helps
**And** the geometry stays at `cs25`; this story does not retune chunk size
**And** cancellation and the Story 16.5 cooperative-cancel chain are unaffected

### AC #4 — Phase 3 (gated): the ear decides

**Given** Story 20.4's four audition rounds established the ear as the only instrument
that settles this
**When** Phase 2 lands
**Then** an NFR3 audition runs, Commander solo, using the **same 7 utterances** so all
prior rounds remain comparable
**And** the reference arm is **`cs25` + the Story 20.4 seam fix** — what ships today
**And** a falsifiable prediction is recorded **before** the audition, per the practice
that made Story 20.4 rounds 3 and 4 readable
**And** long-form take-to-take variance is accounted for: Story 20.4 §17 found the same
configuration flagging differently across takes, so **a single pair per utterance cannot
separate configurations at this effect size**. Either use multiple takes per condition or
state plainly that the round can only detect a large effect

### AC #5 — The prize is re-measured, not assumed

**Given** the point of this work is to make a smaller `chunk_size` reachable
**When** Phase 3 passes
**Then** the chunk-size question is reopened as its **own** story, not smuggled in here
**And** Story 20.4's closure (§17) is amended to record that its stated precondition has
been met

### AC #6 — No regressions

**Then** the accumulated Epic 20 suites pass with zero new failures, and the tree's known
pre-existing failures are unchanged in count and identity

## Tasks / Subtasks

- [x] **Task 1 — Phase 1 bench** (AC: #1, #2) — chunked-with-state vs whole-sequence, error attributed per sub-stack, cost and CUDA-graph interaction reported. **No `src/` changes.** ✅ `git diff --stat -- src/` empty.
- [x] **Task 2 — GATE.** Reported. **Stopped.** Verdict GO; Phase 2 not begun.
- [x] **Task 3 — Phase 2** (AC: #3) — state carried in `_build_true_stream_decode_fn`'s decode path via `services/tts_streaming/codec_state_cache.py`; per-session, reset on start/cancel/completion; `cs25` and the 5-frame lookahead + post-decode trim untouched; both smoothing layers re-evaluated on evidence and left in place. Verified on the real model.
- [x] **Task 4 — Phase 3 round 1** (AC: #4) — paired-take fixture + helper built, prediction pre-registered, audition run. **MIXED**: candidate preferred 5-1 wherever the seam was exposed, byte-identical zero-seam control passed, but 2 single-seam rows blocked on a defect this story's own evidence had already named.
- [x] **Task 5 — Regression** (AC: #6) — 923 streaming / dispatch / progressive-playback / trip-wire unit tests pass (47 of them new). Integration 166 pass / 4 fail; a combined unit+integration process shows 9, all of them cross-test pollution. Every failure verified pre-existing and unchanged in count and identity by stashing the whole change and re-running.
- [x] **Task 6 — Phase 4** (AC: #3 follow-through) — neutralise the 64-sample `StreamingChunkBuffer` crossfade, **gated on producer-declared continuity** so SENTENCE_STREAM (where its discontinuities are real) is untouched. One variable.
- [ ] **Task 7 — Phase 3 round 2** (AC: #4) — fixture + helper built, prediction pre-registered. **Audition not yet run — awaiting Commander.**

## Dev Notes

### What is already known — do not re-derive it

- `decode(N frames) == 1920·N − 555`, verified on 14 independent residual lengths.
- Two decodes of identical frames: ~35 % NRMSE, 0.93 correlation over a settled window
  but **0.55/0.11 over the blend region**, ±35 samples of lag jitter.
- Error by position into the chunk: worst at the head (0.824–1.163 normalised), decaying
  by ~4,000 samples.
- The codec is **12.5 Hz** (1920 samples at 24 kHz), not the 12 Hz this codebase's prose
  said until Story 20.4.
- Story 20.4's harness, seam-analysis and fixture-generation scripts exist and are
  committed; reuse them rather than rebuilding.

### Why this is not simply "use the fork's streaming path"

`Qwen3TTSTokenizerV2Model.decode_streaming` (`:1256`) looks like the answer and is not.
It is a **speed** path — single window, CUDA graphs, optional padding — and carries **no
state** across calls. `stream_generate_pcm` and `capture_cuda_graph` likewise assume a
fixed window. The fork does not implement codec state caching; the reference
implementations Mary's research surveyed do. This is ours to build.

### The D-25 consequence, if a streaming decode path is ever adopted

Story 20.1 §5.4 noted the D-25 invariant is currently decorative because our decode path
calls `speech_tokenizer.decode(...)` directly and `decode_window_frames` never reaches a
runtime shape decision. If Phase 2 moves onto a windowed streaming decode, **that
invariant becomes live**. Story 20.4's geometry threading is what makes that safe — it
was built for a retune that did not ship, and this would be its real justification.

### What this story is NOT

- **Not a chunk-size retune.** Story 20.4 closed that at `cs25` after four auditions.
  AC #5 reopens it as a separate story only after Phase 3 passes.
- **Not PORT-b.** That is the `faster-qwen3-tts` talker/predictor work, still unscoped
  and now weaker: it was ranked against a 5,051 ms baseline that is already 1,353 ms.
- **Not the qasync call-site audit** (Story 20.3's residual risk).

## References

- `_bmad-output/planning-artifacts/research/technical-qwen3-tts-ttfa-optimization-2026-08-31.md` — Finding 1, Nari's technique list
- `_bmad-output/implementation-artifacts/20-4-chunk-size-and-adaptive-cushion-evidence.md` §11 (the splice/edge-loss discovery), §13.1 (why offline metrics do not gate), §13.2-13.3 (error profiles, correlation correction), §17 (chunk-size closure and what reopens it), §11.9 (re-filing Finding 1 as quality)
- `python310/Lib/site-packages/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:170-192` (the zero left-pad), `:195+` (transposed conv), `:475+` (causal transformer), `:1256` (`decode_streaming` — not stateful)
- `src/myvoice/services/qwen_tts_service.py` `_build_true_stream_decode_fn`; `src/myvoice/services/tts_streaming/streaming_decoder.py` (the seam fix)

## Dev Agent Record

### Phase 1 result — 2026-09-01 (RTX 5090, bf16 shipping precision, 25-frame chunks)

Full evidence: `20-5-codec-state-caching-evidence.md`. Bench: `20-5-state-cache-bench.py`;
stage-attribution probe: `20-5-stage-probe.py`; run log: `20-5-bench-run03.log`.

**AC #1 — GO on both thresholds.**

| | ships today (INDEP) | state carried | threshold |
|---|---|---|---|
| edge loss | `1920·N − 555` per chunk (−4,995 over l-020's 9 seams) | **`1920·N`, +0 samples vs whole-sequence** | zero → met |
| head NRMSE (1024 samples, median over seams) | 115–144 % | **0.56–0.82 %** (fp32: 0.03–0.05 %) | < ~5 % → met |
| correlation, first 1024 samples | 0.165–0.959 med / 0.009 min | **1.0000 med / 0.9916 min** | vs 0.55 / 0.11 |
| lag jitter | −1200 … +1165 samples | **0 samples, every seam** | vs ±35 |

Error-by-position is **flat** under carried state (0.014 / 0.009 / 0.007 / 0.008 / 0.010 / 0.011
across 0–8192 samples), not head-weighted — Story 20.4's cold-start signature is removed, not reduced.
With TF32 disabled in fp32 the streaming decode is **bit-exact** against `decoder(codes)`
(final-output NRMSE 7.7e-07, every intermediate stage ≤ 2.6e-06). The 0.8 % bf16 residual
enters at the transformer stage and is rounding, not mechanism.

**Per-sub-stack attribution.** Full 2³ ablation (head NRMSE, l-020): none 1.440 · conv 1.567 ·
tconv 1.036 · transformer 1.420 · conv+tconv 0.244 · conv+xf 1.542 · tconv+xf 0.931 · **all 0.008**.
The 555-sample edge loss is **100 % transposed conv** — every arm carrying tconv state reaches +0,
every arm without it stays at −555/seam. Leave-one-out cost from the full arm: tconv largest
(→1.54), then causal conv (→0.93), then the KV cache (→0.24). All three necessary, none sufficient.
Transformer isolated at latent level: no-cache 0.300 vs KV-cached 5.8e-03 (bf16) / 1.8e-04 (fp32).

**AC #2 — cost.** 37 tensors, **≤ 2.52 MiB per session** in bf16 (21 conv/tconv buffers = 276 KiB;
KV cache 2.25 MiB, **self-bounded** because all 8 layers are `sliding_attention` window 72 —
measured 2.22 MiB at 231 frames, constant in utterance length). Strictly **per session**: one plain
state object, no module- or class-level storage, so concurrent HTTP-API generations are safe.
Decode time **does not regress** — carried state measured 21–30 % *faster* per chunk
(3.4–5.1 ms), partly from skipping `F.pad` and a `Module.__call__` layer.
Expressible as a **wrapper** (~130 lines re-walking `Decoder.forward`, calling the loaded
submodules' inner `nn.Conv1d`/`nn.ConvTranspose1d`) — **no vendoring**. This is a story, not an
architecture pass. A monkey-patch would be process-global and is ruled out by the per-session rule.

**CUDA-graph interaction — the feared trade does not exist.** On the loaded model
`_compiled_forward` is set (`reduce-overhead`) but `_cuda_graph` is `None`, and the production
TRUE_STREAM path never reaches `forward_optimized`: `_build_true_stream_decode_fn` →
`speech_tokenizer.decode` → `Qwen3TTSTokenizerV2Model.decode` → `chunked_decode` → plain
`forward`. The compiled decoder graph is **not on our decode path today**; Story 18.4's win comes
from the talker/code-predictor compiles. Nothing is traded away. Carried state is in fact *more*
graph-friendly (every chunk after the first is a single static 25-frame → 48,000-sample shape).
D-25 stays decorative, per Story 20.1 §5.4.

**One correctness trap Phase 2 must carry.** `nn.ConvTranspose1d` adds its bias to every output
position of each partial convolution, so a naive overlap-add **double-counts it**. Uncorrected it
leaves a bias-shaped transient decaying over ~2,000 samples that survives an fp32 pass and reads
as a 17–21 % residual cold start — i.e. it looks exactly like NO-GO. Subtracting one copy of
`conv.bias` in the overlap region is what makes the result bit-exact. Localised with
`20-5-stage-probe.py` to `decoder[1].block[1]`, the first stride-8 transposed conv.

**Scoping inputs for Phase 2** (evidence §4, not authorisations): the 5-frame lookahead and the
post-decode trim may both become unnecessary — which is also a TTFA effect (first chunk at 25
talker steps instead of 30) and is *not* a chunk-size retune; the Story 20.4 seam blend **and**
the 64-sample `StreamingChunkBuffer` crossfade both become suspect together and both need
evidence rather than assumption; Phase 2 can assert correctness **exactly** (single-chunk ==
`forward` bit-for-bit; fp32 chunked == whole to ~1e-06) rather than statistically; and a
trip-wire extension in `tests/test_qwen_tts_internals.py` should pin the module chain the
wrapper walks.

**AC #1 "no production code modified" — verified.** `git diff --stat -- src/` empty;
`git status --porcelain -- src/ tests/ tools/ build_tools/` empty.

### Phase 2 result — 2026-09-01 (AC #3)

Full evidence: `20-5-phase2-evidence.md`. Implementation:
`src/myvoice/services/tts_streaming/codec_state_cache.py` (new, wrapper — no vendoring, as
Phase 1 §3.3 predicted). Verification: `20-5-phase2-verify.py` / `.json`.

**On the real model, on the production decode path** (bf16, RTX 5090; both arms replayed from the
SAME captured token chunks through a real `StreamingDecoderWorker`, so the only difference is the
decode). Columns are l-020 / l-021 / m-020:

| | ships today (stateless) | state carried |
|---|---|---|
| whole-signal NRMSE vs whole-sequence decode | 0.208 / 0.603 / 0.163 | **0.0099 / 0.0102 / 0.079** |
| head NRMSE at seams (1024 samples, median) | 0.406 / 0.477 / 0.220 | **0.0078 / 0.0075 / 0.0055** |
| correlation med / min | 0.93/0.73 · 0.89/0.81 · 0.98/0.98 | **1.0000 med, 0.9937 worst** |
| lag jitter | +0…+368 samples | **0 on every seam** |
| error by position into chunk | head-weighted (0.15 → 0.54 → 0.28) | **flat (0.012/0.009/0.007/0.008/0.010/0.011)** |
| per-session state | — | **37 tensors, 2.49 MiB, bounded** (identical after 4 and after 10 chunks) |

**The regression bar is exact, not statistical.**
`tests/unit/services/tts_streaming/test_codec_state_cache.py` builds a genuine tiny
`Qwen3TTSTokenizerV2Decoder` from the real upstream classes and asserts single-chunk streaming ==
`forward` **bit-for-bit** (`torch.equal`), fp64 chunked == whole at 3e-16 across four chunk sizes,
the stitched worker output == the whole-sequence decode to 1e-06, and the retained overlap tail ==
the next chunk's head bit-for-bit. On the loaded bf16 model the single-call traversal is likewise
**bit-exact (0.0)**, on random and on real token sequences alike.

**The `nn.ConvTranspose1d` bias double-count is carried and pinned.**
`test_transposed_conv_bias_is_not_double_counted` runs the naive form deliberately: 2.2e-02 against
the corrected form's 3.2e-16 — seven orders apart, and **with the correct length**, which is why it
reads as NO-GO rather than as a bug. It is also why the runtime self-test was re-derived rather than
loosened: a first-draft 5e-02 bf16 tolerance failed on *correct* code (5.13e-02 on random codes).

**Trip-wire extended.** Nine rows in `tests/test_qwen_tts_internals.py` pin the module chain the
wrapper walks: the eight classes it dispatches on, seven fragments of `Decoder.forward`, the zero
left-pad premise, the `[left_pad : −right_pad]` discard that owns 100 % of the 555, the five decoder
attribute names, all-sliding `layer_types`, the `upsample_rates` arithmetic that yields 1920/555,
`Model.decode` still routing through `chunked_decode` and not `forward_optimized`, and the
`DynamicCache` rebind-not-mutate contract the free state snapshot depends on.

**AC #3's other clauses.** State is one plain object per `StatefulCodecDecoder` per dispatch — no
module- or class-level storage, so concurrent HTTP-API generations are isolated (pinned by a test
that drives two decoders in different orders). `reset()` fires on session start, on `END_OF_STREAM`
before the terminal post, and in `_drain_and_post_cancel` after the drain and before the cancel
post; it can never raise, so the **Story 16.5 cooperative-cancel chain is unaffected** — pinned by a
reset that raises and a required cancel post. `cs25`, the 5-frame lookahead and the post-decode trim
are untouched.

**Both smoothing layers re-evaluated, neither changed** (evidence §4):

- The **Story 20.4 1,024-sample seam blend is now an identity.** Its two inputs — the tail a chunk
  retains and the head of the next — are decoded from the same state snapshot; they differ by NRMSE
  0.0037–0.0046 median against 0.22–0.56 today, a ~125× reduction, and are bit-identical in fp64 on
  CPU. No quality argument either way; left in place, and leaving it is what keeps Phase 2 to one
  variable.
- The **64-sample `StreamingChunkBuffer` crossfade is now the dominant remaining error term.** It
  blends *different moments in time* (sample `n+i` with `n−64+i`), so on continuous audio it is a
  2.7 ms comb. Measured against ground truth it makes the state-cached output **2.6× / 3.3× worse**
  on the two long fixtures (0.0099 → 0.0254, 0.0102 → 0.0337) while touching 0.13 % of samples, with
  local excursions to 0.35 full scale. It was already mildly harmful pre-20.5 — the cold-start error
  was simply 20× larger. **Primary follow-up, with its own audition. Not bundled here.**

**One honest regression: decode time.** +7.0 to +10.4 ms per chunk (+22 % to +48 %), where Phase 1's
bench measured −21 % to −30 %. The bench decoded 25 frames in ONE call with no lookahead; the shipped
path must preserve the lookahead and the trim, so each chunk is two passes (25 frames committed, 5 on
a discarded snapshot) and the decoder is launch-overhead dominated at these tensor sizes. Against a
chunk carrying 2,000 ms of audio that is 1.0 % → 1.5 % of real time, and +0.6 % on a measured
1,491 ms long-form TTFA. Recoverable by dropping the lookahead (follow-up 2, which also *gains* five
talker steps of TTFA) or by single-pass state capture. Reported rather than fixed, because fixing it
means changing the trim — which this story is not doing.

**Failure policy: decline loudly, never degrade silently.** Three build-time gates — the
`MYVOICE_CODEC_STATE_CACHE` kill switch, a structural probe that refuses any module graph it has not
been shown exact on, and a numerical self-test against the loaded weights — each return the pre-20.5
stateless adapter with a logged reason. At runtime the length identity is re-checked on every chunk
and a mismatch raises rather than posting mis-spliced audio.

### Phase 3 preparation — 2026-09-01 (AC #4). NOT YET RUN.

Fixture: `20-5-regen-audition-fixture.py` → `20-5-perceptual-fixtures/`. Helper:
`20-5-l1-audition-helper.py`, Commander solo. Reference = `cs25` + the Story 20.4 seam fix (what
ships today); candidate = the same plus state caching. Same seven utterances as Story 20.4 rounds
1–4, so every round stays comparable.

**The variance requirement is answered by removing the confound, not averaging over it.** Both files
in every pair are decoded from **one talker run** — the talker's token chunks are captured once and
decoded twice — so within a pair the wording, prosody, pauses and total duration are identical *to
the sample*. Story 20.4's arms were necessarily different takes, because a chunk-size change perturbs
what the streamer emits and therefore what the talker samples; that is what §17 measured. Nothing
upstream of the decoder is touched by this story, so the tokens are literally reused, a single pair
per utterance is already sufficient for **attribution**, and the round is **not** limited to
detecting a large effect. Two takes per utterance are generated anyway — to sample the *content*
lottery (whether a held vowel or plosive lands on a boundary), not to average arm variance. 14
trials. Levels are deliberately **not** normalised: sharing a take, any level difference would itself
be a finding, and the generator warns above 0.2 dB.

**Prediction, recorded before the audition:**

- **P1 (BLOCKING)** — no blocking seam defect (`audible_seam`, `click_or_discontinuity`,
  `prosody_break_at_stitch`) on any candidate trial. *Falsified by one.*
- **P2 (DIRECTION)** — where the two differ the candidate is preferred; the reference is never
  preferred on seam grounds. *Falsified if the reference is preferred on ≥ 4 of 14.*
- **P3 (MAGNITUDE)** — `equivalent` is the modal answer, **6–11 of 14**. Story 20.4 round 3 already
  certified `cs25` + blend as clean to this listener, so there is little audible headroom left at
  this seam density. *Falsified either way:* ≥ 10 candidate-preferred is a large effect; ≤ 4
  equivalent means the arms are far more separable than predicted.
- **P4 (LOCATION)** — any difference is on l-020 / l-021 (8–9 seams), not on the short fixtures (0–1
  seams). *Falsified by a difference on a short fixture but not a long one.*

P3 predicts *against* the exciting outcome deliberately. The offline numbers are dramatic — 50–60×
less seam error, zero lag jitter — and it would be easy to assume that must be audible. It probably
is not, at `cs25`, because Story 20.4 established that the 1,024-sample blend masks this defect well
enough at this seam density for Commander to call it clean. A round returning "equivalent everywhere,
no new defects" is a **PASS**: the prize is not audible quality at `cs25`, it is that the harm which
killed `cs10` is gone at the cause, and AC #5 reopens the chunk-size question as its own story. A
large effect *is* plausible — Phase 1 removed the cause rather than masking it, and masking is never
perfect — but it is offered as a prediction to be falsified, not as an expectation. If P3 is
falsified by a large candidate win, that is itself evidence the blend was masking less well than
round 3 suggested, which strengthens the AC #5 case.

### Phase 3 round 1 result — 2026-09-01 (AC #4). MIXED.

Full record: `20-5-phase2-evidence.md` §"Phase 3 audition"; raw data
`20-5-state-cache-audition.csv`. 14 trials, L1 solo, blinded, both arms of every pair decoded
from one talker run.

**Preference: state 5 — reference 3 — equivalent 6. Two blocking rows** (`m-020-t2`,
`s-020-t2`), both single-seam, both candidate-only.

**The mechanism result stands.** Where both arms flagged, the candidate was preferred **5–1**,
and the listener's notes are directional and consistent: *"B click was minor compared to A"*,
*"A clicks were very minor"*, *"very minor pops at the seam compared to the rest of the clicks
noted"*. State caching is audibly *better* wherever the seam is exposed. The byte-identical
zero-seam control (`s-020-t1`) came back `equivalent`, so the fixture is sound.

**The blocker had already been named, in this file, before the round ran** — evidence §4.2: the
64-sample `StreamingChunkBuffer` crossfade blends *different moments* (sample `n+i` with
`n−64+i`), so on continuous audio it is a 2.7 ms comb, measured at 2.6×/3.3× worse against
ground truth. It was masking while the cold start dominated; remove the cold start and it is the
loudest thing left. `m-020-t2` and `s-020-t2` are exactly the rows where the reference's cold
start happened not to be audible and the candidate's newly-unmasked comb is.

**Leaving the crossfade in was the right call.** One variable is why the blocker can be *named*
rather than guessed at. P3 (`equivalent` modal, 6–11) landed at exactly 6 because **both** arms
flagged far more than Story 20.4 rounds 3–4 did on the same utterances — a fixture-construction
difference (one talker run per pair is a different content realisation, and 14 trials sample
more content than 7), not a code regression.

### Phase 4 — 2026-09-01. Neutralise the crossfade, scoped.

**Decision: gate it, not remove it and not reduce it.** Evidence §9.1.

- *Not reduce* — the harm is qualitative. A cross-dissolve of a signal with its own past is a
  comb with notches at odd multiples of `sample_rate/2K`; halving K moves the notches and
  shortens the artefact but never makes the operation correct. It would trade one arbitrary
  constant for another and still need an audition.
- *Not remove globally* — the same buffer is on **SENTENCE_STREAM**, which butt-splices
  independently generated sentences. There the discontinuity is real and the crossfade is doing
  its job. Story 20.5 measured nothing on that path.
- *Gate* — the crossfade's premise ("consecutive chunks are independent renderings butt-spliced
  together") is a property of the **producer**, and it is now false for exactly one producer.
  So the producer declares it and the consumer acts on it. On a declared-continuous stream 0 is
  not "less of a bad thing", it is the **correct** value: the concatenation is then exactly what
  the codec produced.

**Blast radius, mapped rather than assumed.** The buffer is built in one place, consumed in one
place, and that consumer serves two producers — app.py's own comment says so. An AST sweep
confirms only `_generate_streaming` and `_generate_true_stream` emit AudioChunks; **batch never
opens a progressive session** (it plays via `play_dual_stream`, which does not touch the buffer).

| path | continuous? | crossfade after Phase 4 |
|---|---|---|
| TRUE_STREAM + state caching | **yes** | **0** |
| TRUE_STREAM, stateless fallback / kill switch | no | 64 (unchanged) |
| SENTENCE_STREAM | no | 64 (unchanged) |
| BATCH | n/a | n/a |

**Wiring** — three changes, each in the layer that owns the knowledge:
`QwenTTSService._progressive_stream_continuous` set at the top of *every* AudioChunk-emitting
dispatch path (TRUE_STREAM reads it off `decode_fn.carries_codec_state`; SENTENCE_STREAM sets
`False`), exposed as `progressive_stream_is_continuous`;
`AudioCoordinator.start_streaming_session(crossfade_samples=None)` where `None` means today's
64 and a non-default is logged; `MyVoiceApp` passes 0 iff the producer declares continuity, via
a double-guarded `getattr`. **`StreamingChunkBuffer` is unchanged** — it already accepted 0.

Deriving the declaration from the decode_fn rather than hard-coding `True` is load-bearing: the
stateless fallback and the `MYVOICE_CODEC_STATE_CACHE` kill switch each keep the crossfade they
still need, automatically.

**The staleness trap is closed by a source invariant.** If only TRUE_STREAM set the flag, a
following SENTENCE_STREAM generation would inherit a stale `True` and silently lose its
crossfade — a cross-path change on an unmeasured path that no single-generation test would
catch. `test_every_audio_chunk_producer_declares_stream_continuity` derives the producer set
from the AST and requires every member to declare. 10 tests in
`tests/unit/services/test_consumer_crossfade_scoping.py`.

**The Story 20.4 1,024-sample seam blend stays.** The bar was *provably* inert. It is not — §4.1
measured its inputs differing by NRMSE 0.0037–0.0046 median with a **worst case of 8.7e-02** on
one l-020 seam. Bit-identical in fp64 on CPU is not the same claim as inert in bf16 on a 5090.
*Nearly* inert is not the standard that was set, so it stays and this round changes one thing.

### Phase 3 round 2 preparation — 2026-09-01 (AC #4). NOT YET RUN.

Fixture `20-5-regen-audition-fixture-r2.py` → `20-5-perceptual-fixtures-r2/`; helper
`20-5-l1-audition-helper.py L1 r2` (round 1 replayable with `... L1 r1`).

    reference = cs25 + fix + state caching + 64-sample consumer crossfade
    candidate = cs25 + fix + state caching + NO consumer crossfade

Both arms carry state caching; **the only variable is the crossfade.** Each arm's width is
*derived* by the same rule the shipped wiring uses rather than hard-coded, and the generator's
preflight refuses to run unless that wiring is actually present — so the round cannot audition a
configuration unreachable in the product. A third arm (`cs25fix`, what ships today) is rendered
from the same takes but **not auditioned**, so a later close-out comparison is a truth-table edit
rather than a regeneration.

**Fixture validation — the sharpest isolation yet:**

| check | result |
|---|---|
| length delta within each pair | +0 samples on all 14 |
| worst within-pair level delta | **0.004 dB** (round 1: 0.32 dB) |
| zero-seam trial (s-020 t2, 1 chunk) | **byte-identical between arms** |
| where the arms differ | **100 % of the squared difference is inside a ±4,096-sample window around a seam, on every trial** |
| how many samples differ | **exactly 63 per boundary** — the crossfade's own width |

The two arms are identical everywhere except the 63 samples at each chunk boundary the crossfade
touches. Peak excursions inside those windows reach 12,414 LSB (0.38 full scale). Three trials
have peak deltas of only 9–34 LSB (quiet boundaries) and form a built-in low-effect control set.

*Limitation, stated:* round 2's takes are new realisations — neither generator persisted its
tokens — so Q1 tests the utterance *class* that blocked, not the identical waveforms. Both arms
of every pair still share a take, so attribution within a pair is exact. Persisting tokens is
recorded as follow-up 6.

**Prediction, recorded before the round:**

- **Q1 (BLOCKING)** `m-020` and `s-020` come back clean on the candidate. *Falsified if either
  still flags candidate-only.*
- **Q2 (NO NEW HARM)** no blocking seam defect on any candidate trial. *Falsified by one.*
- **Q3 (DIRECTION)** candidate preferred at least as often as reference. *Falsified if reference
  preferred on ≥ 4 of 14.*
- **Q4 (MAGNITUDE)** `equivalent` modal, **7–12 of 14** — the arms differ on 0.1 % of samples.
  *Falsified either way:* ≥ 10 candidate-preferred is more effect than 63 samples per boundary
  should produce; ≤ 4 equivalent means they are far more separable than 2.7 ms can explain.
- **Q5 (LOCATION — the one that can embarrass the diagnosis)** round 1's blocking rows were
  **single-seam**, the opposite of where seam-density reasoning would put them. The crossfade is
  a *per-boundary* artefact, so it should be more exposed where it is not buried under
  neighbouring seams. The improvement should therefore show on the low-seam rows at least as
  much as on the 8–9-seam ones. *Falsified if only the long fixtures improve* — which would mean
  the blocking rows had a different cause and Phase 4 fixed the wrong thing.

Q5 is stated first-class deliberately: a round that cannot embarrass its own hypothesis is not
worth running.

## Change Log

- 2026-09-01 — Phase 3 round 1 run: MIXED. State caching preferred 5-1 wherever the seam was exposed and the zero-seam control passed, but two single-seam rows blocked on the 64-sample consumer crossfade — a defect this story's own evidence had named, with numbers, before the round ran. Phase 4 neutralises it, **gated on producer-declared continuity** so SENTENCE_STREAM (real discontinuities) and the stateless fallback keep it; batch never reaches it. Story 20.4's 1,024-sample seam blend stays: *nearly* inert is not *provably* inert. Round 2 fixture built (both arms state-cached, crossfade the only variable, difference 100 % confined to 63 samples per boundary, zero-seam control byte-identical) and the prediction pre-registered. **Audition not run; awaiting Commander.**
- 2026-09-01 — Phase 2 implemented and verified on the real model; Phase 3 fixture + helper built and the prediction pre-registered. State is carried per session by a wrapper that re-walks the decoder traversal (no vendoring); the `nn.ConvTranspose1d` bias correction is carried and pinned by its own test; the regression bar is bit-exactness rather than a tolerance; the trip-wire now pins the module chain. `cs25`, the lookahead and the trim are untouched. Both smoothing layers measured: the 20.4 seam blend is now an identity (left in), the 64-sample consumer crossfade is now the dominant error term (left in, flagged as the primary follow-up with its own audition). Decode time regresses +8.6 ms/chunk — the price of keeping the lookahead — which is 0.5 % of a chunk's audio duration and is recoverable by follow-up 2. **Audition not run; awaiting Commander.**
- 2026-09-01 — Phase 1 executed. Bench + stage probe built and run on RTX 5090. Verdict **GO**: the 555-sample edge loss reaches zero exactly, head NRMSE falls 115–144 % → 0.56–0.82 % (bit-exact in fp32 with TF32 off), lag jitter ±1200 → 0, cost ≤ 2.52 MiB/session with no decode-time regression, expressible as a wrapper, and no CUDA-graph trade because the compiled decoder graph is not on the production decode path. Stopped at the gate per the story's phase rule; Phase 2 not begun.
- 2026-09-01 — Drafted by Winston after Story 20.4 closed the chunk-size question at `cs25` and named this as the thing that would reopen it. Phase-gated deliberately: the mechanism is understood but the reachable state may not determine the output, and that is answerable on a bench for a fraction of the cost of finding out in the dispatch chain.
