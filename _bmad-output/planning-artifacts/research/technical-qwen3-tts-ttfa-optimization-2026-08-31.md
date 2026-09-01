---
date: '2026-08-31'
analyst: Mary (Business Analyst)
user_name: Commander
researchType: technical
trigger: 'Commander flagged QwenLM/Qwen3-TTS discussion #358 (Nari Labs ultrafast implementation)'
primarySources:
  - https://github.com/QwenLM/Qwen3-TTS/discussions/358
  - https://nari-labs.com/blog/qwen3-tts-speed-cost-frontier/
  - https://github.com/nari-labs/nari-qwen3-tts
  - https://github.com/andimarafioti/faster-qwen3-tts
  - https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md
internalEvidence:
  - _bmad-output/implementation-artifacts/18-4-torch-compile-decoder-persistent-cache-evidence.md
  - src/myvoice/services/tts_streaming/codec_token_streamer.py
  - src/myvoice/services/tts_streaming/torch_runtime.py
  - src/myvoice/services/qwen_tts_service.py
relatedScopes:
  - epic_18_producer_bottleneck   # closed 2026-05-11 (ratio 0.670×)
  - ttfa_gap                      # OPEN — this document's subject
status: findings-complete
---

# Technical Research — Qwen3-TTS TTFA Optimization

**Question asked:** Does the Nari Labs result in Qwen3-TTS discussion #358 contain
improvements MyVoice can adopt?

---

## Answer First

**The Nari headline is not our number, but the trail leads to one that is.**

Nari's "10 RPS, sub-50 ms p95 TTFA" is a *concurrency-scheduling* result on an
H100 SXM. Roughly half of their engineering — the unified scheduler, anchor-request
batching, cohort CUDA-graph pre-capture — buys throughput under simultaneous load.
MyVoice runs **batch size 1, one user, one utterance**. That half is worth zero to us.

The other half is not. Following the same thread's ecosystem surfaced
**`andimarafioti/faster-qwen3-tts`** (MIT, 1.3k stars, 336 commits) — a *single-stream*
optimization of the *same* Qwen3-TTS-12Hz model family we ship, benchmarked on an
**RTX 4090**, running **natively on Windows**, supporting **0.6B and 1.7B**, and covering
**voice clone + CustomVoice + VoiceDesign** — our exact three modes.

It reports **4.1× RTF and 3.0× TTFA improvement on RTX 4090** and, critically, states
that **`torch.compile` delivered zero speedup because dynamic KV-cache shapes defeat
the compiler**.

That sentence is a third-party reproduction of our own Story 18.4 result. We measured
compile at **−7.46 % on first-chunk latency** and logged the exact diagnostic warning
their finding predicts (`We have observed 9 distinct sizes`). Our TTFA sits at
**~5.5 s median on an RTX 5090** — against their ~152 ms (0.6B) on weaker silicon.

**The strategic read:** Epic 18 closed the *producer-bottleneck* question (ratio 3.23× →
0.670×, gaps structurally impossible). It did **not** close the *time-to-first-audio*
question, and the evidence file says why in plain language: `compile_talker=False`, so
the talker — which alone determines first-chunk latency — is still eager. This research
identifies the specific, published, permissively-licensed technique that unblocks it.

---

## Finding 1 — Talker CUDA Graph + StaticCache (HIGH — the load-bearing one)

**Evidence.** `faster-qwen3-tts` applies `torch.cuda.CUDAGraph` directly (no Triton, no
Flash Attention, no vLLM) over the talker and predictor decode loops, paired with
Transformers' `StaticCache` — pre-allocated fixed KV tensors with in-place `index_copy_`
updates. Fixed shapes are the precondition for graph capture. Reported:

| GPU | Baseline RTF | Optimized RTF | Speedup |
|---|---|---|---|
| RTX 4090 | 1.34 | 5.56 | 4.1× |
| H100 80GB | 0.59 | 4.19 | 7.1× |
| Jetson AGX Orin | 0.175 | 1.57 | 9.0× |
| DGX Spark (GB10) | 1.19 | 2.26 | 1.9× |

Per-component on Jetson 0.6B: **talker 75 ms → 12 ms per step**. Nari independently
reaches for the same primitive — "capture the entire frame-generation loop as a single
CUDA graph" over the Code Predictor's fixed 15-step structure.

**Why it matters to us specifically.** Our 18.4 evidence file states the causal chain
outright: *"first chunk has to come out of the talker's autoregressive loop, and
`compile_talker=False` … keeps the talker eager. So first-chunk latency reflects talker
speed (unchanged)."* We disabled talker compilation because Story 16.8's forward-hook
(capturing multi-codebook `codec_ids` off `model.model.talker.forward`) does not survive
`torch.compile` wrapping.

`faster-qwen3-tts` sidesteps that collision by construction: it does not use HF
`generate()` + `BaseStreamer` hooks at all. It hand-rolls the decode loop in
`talker_graph.py` / `predictor_graph.py` / `streaming.py`, which exposes codec IDs
directly — no hook to break.

**Confidence:** HIGH. Two independent sources; our own measurement corroborates the
negative half (torch.compile null on TTFA).

**Caveat — this is not a tweak.** Adopting it means replacing or forking the Story 16.8
dispatch chain, which is audited, pinned by a trip-wire test, and load-bearing. It also
depends on `qwen-tts-hf` (a Transformers 5 compatible build), which will collide with our
Story 16.1 `qwen-tts` pin + import-attribute test. Scope this as an **architecture pass**,
not a story.

---

## Finding 2 — Chunk sizing is a first-order TTFA lever we have never tuned (HIGH — cheap)

**Our current state.** `codec_token_streamer.py:46-47` — `DEFAULT_CHUNK_SIZE = 25`,
`DEFAULT_LOOKAHEAD = 5`, fixed for every chunk including the first. Our model is
**Qwen3-TTS-12Hz** (`service_enums.py:82-84`), so one codec frame = **83.3 ms** of audio.

> **No PCM can be emitted until the talker has produced 30 frames = 2.5 seconds of audio.**

Those constants were inherited verbatim from a research example
(`01-streaming-tts-research.md:184`) and the file's own docstring notes Story 16.7's
harness "may revise" them. It never did.

**Evidence they matter.** `faster-qwen3-tts` (Jetson, 0.6B):

| chunk_size | TTFA | RTF |
|---|---|---|
| 2 | **266 ms** | 1.042 |
| 8 | 556 ms | 1.384 |

A 4× chunk increase doubled TTFA. Nari productizes the same trade-off as three named
profiles — `ttfa` ("smaller initial Codec chunks"), `balanced`, `throughput` ("larger
Codec chunks") — and describes a **dual chunk strategy**: *"Smaller chunks let playback
begin quickly, while larger chunks improve batching and GPU efficiency during sustained
playback."*

**Recommendation.** A **ramped** chunk schedule — small first chunk for TTFA, growing to
the current 25 for steady-state efficiency. This is the single highest
value-per-engineering-hour item in this document.

**Known obstacle.** `torch_runtime.py:628-646` enforces D-25 as a hard `AssertionError`:
`decode_window_frames` must equal `streamer_chunk_size + streamer_lookahead`, because
CUDA-graph replay captures one window shape. A variable chunk size therefore needs either
a small set of pre-captured window shapes or the D-25 invariant renegotiated. Flag for
Winston.

---

## Finding 3 — Leading-silence trim (MEDIUM — free, no inference change)

Nari implements a "dynamic trim" that detects speech onset from short RMS windows and
drops samples before it, reporting **~80 ms TTFA reduction without accelerating
inference** — pure post-processing on the first chunk.

We do **no** trimming anywhere on the TTS output path (verified: no silence/trim logic in
`qwen_tts_service.py` or `streaming_chunk_buffer.py`). Small, self-contained, testable,
and it composes with everything else here.

---

## Finding 4 — Reference-audio silence padding for cloned voices (MEDIUM — quality, not speed)

`faster-qwen3-tts` appends **0.5 s of silence to the reference audio before encoding**,
because the reference's final phoneme conditions the first output token and produces an
audible artifact. Applied automatically before `create_voice_clone_prompt()`.

This lands directly on our CLONED-voice path (Story 17.2's `voice_clone_prompt`
precompute). No performance cost; a pure quality win for a mode we ship to users. Note it
would invalidate cached `<voice>.pt` embeddings — needs a cache-version bump.

---

## Finding 5 — CPU↔GPU synchronization in the decode loop (MEDIUM)

Both sources independently attack sync points:

- **`faster-qwen3-tts`:** the per-token **repetition-penalty Python loop** forced CPU↔GPU
  syncs; vectorizing it with `torch.where` over unique tokens removed them. **This is live
  for us** — `emotion_profile.py` ships `repetition_penalty` values of 1.2–1.5 on every
  emotion preset, so we take that path on every generation.
- **Nari:** *"Defers termination checks until EOS enabled to avoid CPU-GPU synchronization
  overhead."*

Each sync point costs single-digit milliseconds but fires **once per decode step** — at
12 Hz over a multi-second utterance that compounds. Also a prerequisite for clean graph
capture (Finding 1), so treat it as enabling work rather than a standalone win.

---

## Finding 6 — Two of our open optimization bets are contradicted (MEDIUM — this one *saves* effort)

`faster-qwen3-tts` tested and **rejected**:

- **Attention backends (SDPA, Flash Attention 2):** *"No RTF difference; attention not
  bottleneck."* We carry a conditional FA2 attempt at `model_registry.py:488-512` and the
  streaming-acceleration architecture lists an FA2 runtime-verification story. Two
  independent sources say the ceiling here is ~zero.
- **Custom CUDA kernels:** fused RMSNorm 8.4×, fused SiLU 2.2× in isolation — **1.25×
  end-to-end**. This corroborates the existing decision to skip megakernel fusion
  (research P2.B) and extends it to hand-written kernels generally.

Recommend explicitly **deprioritizing the FA2 verification story** and citing this
document as the rationale.

---

## Explicitly Not Transferable

| Nari technique | Why not |
|---|---|
| Unified scheduler across Talker / Predictor / Codec | Requires concurrent requests; we are batch=1 |
| Anchor-request batch filling | Same |
| Per-batch-size CUDA graph pre-capture | Same |
| Speech-aware deadline scheduling | Meaningful only with competing requests |
| The `nari-qwen3-tts` engine wholesale | **H100 SXM only, Linux x86_64 only, CUDA 13.0+, Docker.** We are Windows + consumer RTX 30xx/40xx/50xx |

Nari's value to us is the **published technique list**, not the artifact.

---

## Verification Status

| Claim | Status |
|---|---|
| Discussion #358 content, both linked URLs | Verified — fetched directly |
| Nari technique list, profiles, benchmark table | Verified — blog + README, two fetches agreeing |
| `faster-qwen3-tts` numbers, rejected approaches | Verified — BLOG.md fetched directly |
| `faster-qwen3-tts` license = MIT, Windows-native, 1.7B + cloning | Reported by README fetch — **confirm before committing engineering** |
| `nari-qwen3-tts` license | **Unverified** — one fetch said Apache-2.0, a second said unspecified. Irrelevant unless we vendor code |
| MyVoice TTFA ~5.5 s, compile −7.46 %, `compile_talker=False`, chunk_size=25 @ 12 Hz | Verified — our own repo and Story 18.4 evidence file |

Not yet verified, and worth a spike before scoping: whether `faster-qwen3-tts` 1.7B
numbers match its 0.6B numbers (benchmark tables published are 0.6B), and whether the
`qwen-tts-hf` / Transformers 5 dependency can coexist with our pinned tree.

---

## Recommended Sequencing

1. **Finding 3** (silence trim) and **Finding 4** (reference padding) — small, independent,
   shippable now, no architectural entanglement.
2. **Finding 2** (ramped chunk sizing) — largest value per hour, but needs a D-25 ruling
   from Winston first.
3. **Finding 5** (sync removal) — enabling work for step 4.
4. **Finding 1** (talker CUDA Graph + StaticCache) — the real prize and the real cost.
   Warrants a spike (benchmark `faster-qwen3-tts` against our RTX 5090 baseline on the
   1.7B model) *before* an architecture pass, because it collides with Story 16.8.
5. **Finding 6** — a deprioritization decision, not build work. Free.

**Framing for the next epic:** Epic 18 was "close the producer bottleneck" and it
succeeded. The successor question is "close the TTFA gap" — and unlike Epic 18, we now
have third-party evidence of what works, what does not, and roughly how much each is
worth, before writing a line of code.
