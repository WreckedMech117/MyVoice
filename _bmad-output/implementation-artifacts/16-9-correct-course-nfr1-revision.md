# Correct-Course Routing Artifact — Story 16.9 NFR1 Revision

> **Status:** Approved 2026-05-08 by Commander (sole stakeholder per `memory/production_release_state.md`).
> **Trigger:** Story 16.9 AC #3 outcome (c) — contract revision.
> **Routing surface:** `/bmad-bmm-dev-story` interactive `AskUserQuestion` prompt — **substituted for** the literal `/bmad-bmm-correct-course` workflow named in AC #3 outcome (c). For solo-dev with Commander as sole stakeholder, the substitution preserves the spirit of AC #3 (stakeholder approval captured in writing alongside the architecture amendment) but does not follow the literal AC text. Disclosed in story Change Log #4 / M3 by the code-review pass; this artifact is the load-bearing record of the decision regardless of the workflow that surfaced it.

## 1. Why this routing exists

Story 16.7 §3.2 + §5 produced empirical first-audio-latency measurements on the maintainer's RTX 5090 + qwen-tts 0.0.4 host that contradicted the architecture's NFR1 projection (`architecture-optimization-pass.md:802` — *"GPU: meets via TRUE_STREAM (~1.5–1.8s estimated). CPU: meets via inherited SENTENCE_STREAM."*). Story 16.9 was created as the **NFR1-contract-reconciliation** follow-up. The story's AC #3 names three legitimate outcomes: (a) implementation fix, (b) model-tier fallback, (c) contract revision. Outcome (c) is documentation-only and **requires routing through `/bmad-bmm-correct-course` for stakeholder sign-off before merge** because it formally relaxes a public NFR.

## 2. Empirical evidence presented to stakeholder

### 2.1 GPU SENTENCE_STREAM phase profile (Story 16.9 Task 2, n=50)

| Class | n | first_chunk p50 | first_chunk p95 | first_chunk max | NFR1 (<2s) clearance |
|-------|---|-----------------|-----------------|-----------------|----------------------|
| short (steady-state)¹ | 17 (gen) / 16 (clearance) | 1.84s | 4.18s | 5.47s | 9 / 16 |
| medium | 17 | 5.45s | 8.74s | 10.06s | 0 / 17 |
| long | 16 | 13.93s | 25.23s | 25.51s | 0 / 16 |

> **¹** The short row mixes column sources (clarification added by code-review pass — story Change Log #4 / M1): the cited p50 / p95 / max (1.84 / 4.18 / 5.47) are `generate_seconds` aggregates at n=17 (steady-state per-utterance dispatch cost, the user-facing first-chunk latency once the model is warm); the "9 / 16 cleared" count is computed from `first_chunk_latency_seconds` at n=16 (drop s-001 warmup whose 4.79s includes a 3.65s cold model_load contribution paid once per session). Strictly disambiguated: `first_chunk_latency_seconds` p95 (n=17, includes s-001) = 4.93s; `first_chunk_latency_seconds` p95 (n=16, drop s-001) = 4.26s; `generate_seconds` p95 (n=17) = 4.18s. All three pass the new ≤5.0s short-class target. For medium/long classes the columns are equivalent because cold model load contributes only to s-001 (which is class-short).

**Phase share across all valid rows (n=50):** generate=99.0% / model_load=1.0% / split=0% / decode=0% / deliver=0%. Source: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv`.

### 2.2 GPU SENTENCE_STREAM small-tier (0.6B) comparison (Story 16.9 Task 3.2, n=17 short-class)

| Tier | n | generate p50 | generate p95 | NFR1 clearance |
|------|---|--------------|--------------|----------------|
| `quality` (3B) | 16 | 1.84s | 4.18s | 9 / 16 |
| `small` (0.6B) | 17 | 3.13s | 7.94s | 0 / 17 |

**Reversal:** the 0.6B `small` tier is empirically slower than the 3B `quality` tier on RTX 5090 + qwen-tts 0.0.4. Outcome (b) "model-tier fallback" is **ruled out**: switching to the smaller model would make first-audio worse, not better. Source: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv`.

### 2.3 Hypothesis-(c) length-latency regression

Cross-class Pearson r = +0.915 (n=49, s-001 warmup dropped). Linear slope ≈ +0.097 sec / chunk-1 char. This confirms that splitting more aggressively *would* reduce first-chunk latency arithmetically — but the **structural floor of ~1.2s** for any input on the 3B model means even maximally aggressive splitting (down to ~12-char first chunks) cannot clear NFR1's 2s p95 ceiling. Aggressive comma-splitting also risks voice-quality regressions whose audition is deferred to a future "streaming default ramp" story (Story 16.7 §6.1). Outcome (a) "implementation fix" therefore has **structural ceiling above NFR1** plus voice-quality regression risk for marginal improvement on a metric we are revising regardless.

### 2.4 Hypothesis verdicts

| Hypothesis | Verdict | Settling data point |
|---|---|---|
| (a) qwen-tts version drift | **Consistent (upstream-bound, not directly verified)** | `generate_seconds` accounts for ≥99% of total wallclock; `model.generate_custom_voice` is the dominant cost surface. Pin-bump comparison out of scope. |
| (b) 3B-quality vs 0.6B-small tier penalty | **Falsified (with reversal)** | Small-tier p95 = 7.94s vs quality-tier p95 = 4.18s on identical short-class set. The smaller model is ~2× *slower* on Blackwell + qwen-tts 0.0.4. |
| (c) sentence-split granularity | **Partially confirmed; outcome-(a) implementation fix has structural ceiling** | Length-latency Pearson r = +0.915. But: (i) short-class minimum 1.27s ⇒ ~1.2s floor on 3B model; (ii) maximally aggressive sub-clause splitting required for NFR1 compliance would harm voice quality; (iii) structural ceiling for (a) is well above NFR1's 2s. |
| (d) NFR1 was always optimistic | **Consistent** | The architecture's "~1.5–1.8s estimated" projection (line 802) was authored before empirical RTX 5090 + qwen-tts 0.0.4 grounding; the empirical floor is ~1.2s with a length-dependent slope that puts most realistic inputs over 2s. |

## 3. Decision presented and approved

**Outcome chosen: (c) pure contract revision. No production code change.**

**Why not (a)+(c) hybrid (the AC #3 decision-rule's "partial confirmation → implementation fix possibly hybrid with (c)" clause):** the splitter fix has structural ceiling above NFR1 (the 1.2s model floor cannot be removed by splitter changes), AND introduces voice-quality regression risk (audition deferred per Story 16.7 §6.1). The cost of (a) outweighs the benefit (marginal latency improvement on a metric being revised). The story's framing at line 442 explicitly authorizes this verdict: *"Outcome (c) is not a failure mode — it's a legitimate engineering verdict that the architecture's projection was wrong and the contract should be updated to reflect reality."*

## 4. Revised NFR1 wording (committed in `architecture-optimization-pass.md` per Task 7)

> **NFR1 (revised 2026-05-08, Story 16.9): First-audio latency under streaming dispatch.**
>
> Empirical measurement on the maintainer's RTX 5090 + qwen-tts 0.0.4 host (Story 16.7 §3.2; Story 16.9 Tasks 2 + 3.2 + 6) demonstrates that the original "<2s" projection is unattainable across input classes on the 3B `quality` model. The 0.6B `small` tier is empirically ~2× slower on Blackwell (Story 16.9 Task 3.2 reversal); a model-tier fallback policy is ruled out. The contract is revised to per-class targets:
>
> | Class | First-chunk char range | GPU `quality` p95 target | Empirical (Story 16.9 Task 2) |
> |---|---|---|---|
> | Short | ≤30 chars | ≤5.0s | 4.18s ✓ |
> | Medium | 30–100 chars | ≤10.0s | 8.74s ✓ |
> | Long | >100 chars | informational only (no formal target); UI provides progress indicator | 25.23s |
>
> **CPU SENTENCE_STREAM** is exempted from the streaming-NFR1 contract; CPU users fall back to the V2 baseline. Hardware-aware default (D-9 / NFR12) ensures CPU users do not encounter TRUE_STREAM.
>
> **Rationale.** Phase-decomposition profiling (Story 16.9 AC #1) showed `_generate_sync` (the `model.generate_custom_voice` invocation site) accounts for ≥99% of first-chunk wallclock. The 3B `quality` model on RTX 5090 + qwen-tts 0.0.4 has a ~1.2s per-utterance floor for any input regardless of length; the length-latency slope is ~+0.10 sec/char. Aggressive splitter changes cannot clear the original "<2s" target without harming voice quality (audition deferred to future "streaming default ramp" story). The original "~1.5–1.8s estimated" projection was authored before the 3B model's per-token cost on this hardware was empirically known.

## 5. Implications acknowledged

- **No code change.** The splitter, dispatch chain, registry, streamer, decoder, coordinator, and UI surfaces are unchanged. NFR7 graceful-degradation chain (TRUE_STREAM → SENTENCE_STREAM → BATCH) is preserved per Story 16.6 D-9.
- **Streaming-default flag flip is unblocked.** The flip was conjunction-blocked on Story 16.8 (TRUE_STREAM viable — closed 2026-05-08) AND Story 16.9 (NFR1 reconciled — closed 2026-05-08). With both closed, the flag flip's *remaining* prerequisite is the multi-listener perceptual A/B audition (Story 16.7 AC #2's deferred protocol), tracked in a future "streaming default ramp" story.
- **No qwen-tts pin bump.** The pin remains at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1. If a future qwen-tts release ships materially-faster `Qwen3TTSForConditionalGeneration.generate`, a separate pin-bump story can re-run Story 16.7's harness against the new pin and re-evaluate.
- **Outcome (a) is deliberately deferred.** A comma-aware splitter (or a first-chunk character cap) is a viable future story IF: (i) audition coverage shows acceptable voice quality, (ii) the cap is empirically tuned against the 1.2s floor, (iii) the change ships with a regression-test pattern mirroring `memory/code_review_regression_test_exact_class.md`. Story 16.9 does not pre-empt that story's scope.

## 6. Stakeholder sign-off

- **Stakeholder:** Commander (`wreckedmech@gmail.com`; sole stakeholder per `memory/production_release_state.md`).
- **Decision date:** 2026-05-08.
- **Decision channel:** `/bmad-bmm-dev-story` interactive `AskUserQuestion` prompt — **substituted for** the literal `/bmad-bmm-correct-course` workflow that AC #3 outcome (c) names. For solo-dev with Commander as sole stakeholder, the substitution preserves the spirit of AC #3 (a written, committed stakeholder-sign-off artifact alongside the architecture amendment) but does not follow the literal AC text. Disclosed in story Change Log #4 / M3.
- **Approved option:** "(c) pure contract revision — Recommended" with the proposed per-class wording in §4 above.
- **Conditions:** none. The decision was approved without modification.

## 7. Cross-references

- Reconciliation report: `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md`
- Architecture amendment: `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (NFR1 cell at line 802 + new prose sub-section "Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-08)" between the OFR table and the next `### Implementation Readiness Validation` heading)
- Phase-profile CSV: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (50 rows, 20 columns)
- Small-tier comparison CSV: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv` (17 rows)
- CPU stratified CSV: `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` (10 rows; produced by Task 6)
- Story 16.7 input set (unchanged): `_bmad-output/implementation-artifacts/16-7-input-set.csv`
- Story 16.7 validation report: `_bmad-output/implementation-artifacts/16-7-streaming-validation-report.md`
