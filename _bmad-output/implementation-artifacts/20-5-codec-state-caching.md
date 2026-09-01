# Story 20.5: Codec State Caching Across Chunks (Phase ⊥-Polish-3)

Status: ready-for-dev

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

- [ ] **Task 1 — Phase 1 bench** (AC: #1, #2) — chunked-with-state vs whole-sequence, error attributed per sub-stack, cost and CUDA-graph interaction reported. **No `src/` changes.**
- [ ] **Task 2 — GATE.** Report to Commander. Stop.
- [ ] **Task 3 — Phase 2** (AC: #3), gated.
- [ ] **Task 4 — Phase 3 audition prep** (AC: #4), gated.
- [ ] **Task 5 — Regression** (AC: #6).

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

_(to be filled by dev agent)_

## Change Log

- 2026-09-01 — Drafted by Winston after Story 20.4 closed the chunk-size question at `cs25` and named this as the thing that would reopen it. Phase-gated deliberately: the mechanism is understood but the reachable state may not determine the output, and that is answerable on a bench for a fraction of the cost of finding out in the dispatch chain.
