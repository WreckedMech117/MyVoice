---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: complete
completedAt: '2026-05-10'
inputDocuments:
  - _bmad-output/planning-artifacts/research/technical-tts-streaming-fast-clone-research-2026-05-10.md
  - _bmad-output/planning-artifacts/architecture-optimization-pass.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/epics-optimization-pass.md
  - docs/QWEN3_TTS_INTEGRATION.md
  - docs/dual_service_audio_architecture_design.md
  - docs/CONFIGURATION.md
workflowType: 'architecture'
scope: 'streaming-acceleration-and-lightning-tier'
projectContext: brownfield
parentArchitecture: _bmad-output/planning-artifacts/architecture-optimization-pass.md
grandparentArchitecture: _bmad-output/planning-artifacts/architecture.md
relatedScopes:
  - epic_18_completion              # Story 18.4 refinement + new 18.5/18.6/18.7
  - lightning_tier_introduction     # New Epic 19 — Chatterbox-Turbo as a second engine
  - producer_bottleneck_close       # Story 18.1's emit/drain ratio 3.23× → < 1.0×
keyResearchFindings:
  myVoiceAlreadyImplements:
    - flashAttention2_attempt          # model_registry.py:488-512 (conditional)
    - voicePromptEncodingCache         # qwen_tts_service.py:649-1250 (Story 17.2)
    - slidingWindowTalkerCoupling      # qwen_tts_service.py:3404-3571 (Story 16.8)
    - boundedQueueBackpressure         # codec_token_streamer.py:116-119 (D-10)
  myVoiceMissing:
    - torchCompileCudaGraph            # Story 18.4 (deferred) — largest unrealized gain
  knownGotchas:
    - fa2SilentNoOp                    # HF Qwen3-TTS-12Hz-0.6B-Base discussion #5
    - dynamicVsFixedDecodeWindow       # Existing Story 18.4 stub conflict with CUDA Graph capture
project_name: 'MyVoice V2 (Streaming Acceleration + Lightning Tier)'
user_name: 'Commander'
date: '2026-05-10'
---

# Architecture Decision Document — Streaming Acceleration and Lightning Tier

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

**Scope:** Two cohesive concerns continuing from the sealed `architecture-optimization-pass.md`:

1. **Epic 18 completion — close the producer bottleneck.** Convert Story 18.1's measured emit/drain ratio of 3.23× (talker @ 31% real-time) into ratio < 1.0× sustained, using the official `qwen_tts.enable_streaming_optimizations()` API plus a small set of derivative refinements (FlashAttention-2 runtime verification, two-phase emission scheduler, Hann crossfade chunk-stitching). Refines and supersedes the existing Story 18.4 stub.
2. **Epic 19 introduction — Lightning Tier (new).** Add Chatterbox-Turbo (Resemble AI, MIT) as a second TTS engine alongside Qwen3-TTS, providing a hyper-fast first-audio path with native voice cloning from a 5-second reference. Pattern B (side-by-side native clone, both engines on PyTorch).

**Relationship to parent architecture:** This document refines and extends `architecture-optimization-pass.md` (sealed 2026-04-27), which itself extends `architecture.md` (sealed 2026-01-31). Where the parent is silent (e.g., compile-cache lifecycle, multi-engine selector, Lightning-tier voice-prompt compatibility), this document fills the gap. Where the parent spoke (D-1 SessionRegistry ownership, D-2 Qt main-thread mutations, D-9 hardware-aware streaming default, D-10 bounded-queue backpressure, D-19 telemetry stream, P-1 through P-9), this document inherits unchanged.

**Inheritance map:**

- **From `architecture-optimization-pass.md`:** D-1 through D-20 stand. P-1 through P-9 stand. The renegotiated NFR1 per-class targets (Story 16.9 reconciliation, 2026-05-08: ≤5s short / ≤10s medium / informational long for GPU `quality`) stand. D-9 hardware-aware streaming default stands.
- **From `architecture.md` (V2 baseline):** PyQt6 + PyTorch + asyncio threading model, snake_case naming, dual-service audio architecture, lazy single-model loading discipline, JSON-based persistence, `%LOCALAPPDATA%/MyVoice/` data root.
- **What this scope changes:** the existing Story 18.4 stub's `dynamic=True` compile choice (mutually exclusive with the CUDA Graph replay on which the largest reported speedup depends — see Step 4 D-21 when written).

**Out-of-scope (explicit):**

- DirectML cross-vendor expansion — gated on AMD/Intel-user demand (research P2.C).
- Qwen3-TTS GPTQ-Int8 quantization — research-grade, no public Qwen3-TTS-specific quant exists yet (research P2.A).
- Megakernel / single-CUDA-kernel-fusion optimization — engineering cost outweighs gain (research P2.B). Explicitly skip.
- Re-litigating NFR1 — Story 16.9's per-class targets stand. This scope adds *headroom*, not a new contract.
- Re-running the NFR3 perceptual A/B audition for the streaming-mode A/B — Story 17.1 cleared it for TRUE_STREAM vs SENTENCE_STREAM, 2026-05-08. New audition required only if perceptual-difficult fixtures change shape (Story 18.4's compiled-output audition is a separate, narrower audition).

**Workflow status:** in progress (Step 1 of 8 complete).

<!-- Subsequent sections will be appended as we work through each step:
     2. Project Context Analysis
     3. Starter / Scaffolding Inheritance
     4. Core Architectural Decisions
     5. Implementation Patterns & Consistency Rules
     6. Project Structure & Boundaries
     7. Architecture Validation Results
     8. Implementation Handoff
-->

## Project Context Analysis

### Scope of This Pass

Two cohesive concerns surfaced by post-Epic-17 production observation and Mary's 2026-05-10 technical research:

1. **Producer-bottleneck closure (Epic 18 completion).** Story 18.1 measured the talker model running at ~31% real-time on RTX 5090 + qwen-tts 0.0.4 (emit/drain ratio 3.23×), causing audible inter-chunk gaps when streaming sentence-by-sentence on RTX 3060-class hardware. Stories 18.2 (TF32 + cuDNN benchmark) and 18.3 (bf16 precision) closed part of the gap but ran into diminishing returns; the final compounding layer is `torch.compile` + CUDA Graph capture, which the existing Story 18.4 stub names but specifies in a way (`dynamic=True`) that is mutually exclusive with the technique's largest reported speedup. This scope refines Story 18.4's design, adds derivative stories for FA2 runtime verification, two-phase emission scheduling, and Hann crossfade chunk-stitching, and brings the producer ratio under 1.0× sustained.

2. **Lightning Tier introduction (Epic 19, new).** Mary's research identified Chatterbox-Turbo (Resemble AI, MIT license) as a hyper-fast TTS engine with native 5-second voice cloning, suitable as a second engine alongside Qwen3-TTS for users who prioritize latency over quality. The architectural pattern is **Pattern B — side-by-side native clone**, where both engines run on the existing PyTorch runtime and each handles its own voice cloning. This is the first multi-engine architecture introduction in MyVoice's history; the V2 baseline and optimization-pass treat TTS as a single-engine concern.

### Inherited Requirements

This scope is brownfield by deepest possible measure: it modifies a single sealed story (18.4) within a sealed pass (optimization-pass) that itself extends a sealed baseline (V2). Every requirement from the parent architectures applies unchanged unless this section explicitly modifies it.

#### Functional Requirements (from V2 PRD; reaffirmed by parent's Step 7 validation)

| Inherited FR | Status under this scope | Reason |
|---|---|---|
| FR1–FR5 (TTS Generation) | **Continues to satisfy** | Story 18.4-refined retains the TRUE_STREAM dispatch path; no API surface changes |
| FR4 (User can cancel) | **Continues to satisfy** | P-7 cancellation chain unchanged; compile/CUDA-Graph state cleared on cancel |
| FR28–FR32 (Playback Last) | **Continues to satisfy** | D-3, D-4 saveable slot semantics unchanged |
| FR42 (Status indicator) | **Extended (Epic 19)** | New session source variant `GENERATED + engine=lightning` flows through existing OFR-D indicator wiring |

#### Non-Functional Requirements (the load-bearing ones for this scope)

| Inherited NFR | Original wording | This scope's relationship |
|---|---|---|
| NFR1 (first audio) | **Per Story 16.9 reconciliation (2026-05-08):** Short ≤30 chars → ≤5.0s p95 GPU `quality`; Medium 30–100 chars → ≤10.0s p95; Long >100 chars → informational only. CPU exempt (D-9). | Adds *headroom* — Story 18.4 closes the producer ratio, which extends headroom. Lightning Tier adds a **new** NFR1-class target for the lightning engine specifically (target: ≤500 ms TTFA on RTX 3060, per research). |
| NFR3 (no audio stuttering) | Inherits Story 17.1 perceptual A/B verdict (PASS, 2026-05-08, 30/30 zero `audible_seam` flags) | **Conditional re-audition.** Compile + fixed-window may shift chunk boundaries; if Story 18.4 re-tunes window from 30 → 80 (D-21 alternative), a perceptual re-audition fires. Hann crossfade (Story 18.6) is a *quality preservation* mechanism, not a re-audition trigger. |
| NFR4 (UI responsiveness <200 ms) | D-2 Qt main-thread mutations | Compile cold-start (10–30s) **must not block UI thread.** Drives D-23 warmup-strategy decision in Step 4. |
| NFR6 (no crashes) | D-12 import-attribute test on `model.model.generate`, `speech_tokenizer.decode` | New private symbols accessed by `enable_streaming_optimizations()` need same import-attribute coverage. New decision needed: D-22 pin and verify the official optimization API. |
| NFR7 (graceful degradation) | TRUE_STREAM ← SENTENCE_STREAM ← BATCH fallback chain | **Extended.** Lightning Tier adds an engine-level fallback: if Chatterbox-Turbo fails to load (download missing, CUDA unavailable, OOM), engine selector falls back to Qwen3-TTS. New decision: D-31 lightning-tier failure-mode behavior. |
| NFR11 (<4GB RAM with model) | Single-model lazy load discipline | **Lightning Tier challenges this.** Both engines coexisting means two models potentially loaded. New decision: D-32 memory coexistence policy (unload-on-switch vs keep-warm). |
| NFR12 (CPU-only support) | D-9 hardware-aware streaming default; CPU stays SENTENCE_STREAM | **Compile path is CUDA-only initially.** Story 18.4 refinement preserves D-9 — CPU users skip compile path entirely. Future P2.A (GGUF quants) could open a CPU lightning path; out of scope for this pass. |

### New Requirements Introduced by This Scope

These do not yet have FR/NFR codes in the V2 PRD; they function as scope-bound requirements (mirroring the OFR-A through OFR-D pattern from `architecture-optimization-pass.md`). They are tracked here for Step 4 decision binding and back-propagated to the PRD as a follow-up administrative task (see Out-of-Scope).

**Epic 18 completion (refines Story 18.4 + adds 18.5/18.6/18.7):**

- **OFR-E** Producer-side acceleration via `enable_streaming_optimizations()` + CUDA Graph capture; emit/drain ratio < 1.0× sustained on RTX 5090; ≥ 100% real-time on RTX 3060.
- **OFR-F** FlashAttention-2 *runtime verification* (not just request) — distinguish "FA2 silently no-op" from "FA2 active" via probe + log + telemetry.
- **OFR-G** Two-phase emission scheduler — Phase 1 aggressive small-window for first chunk, Phase 2 larger-window for stability. Targets ≤ 250 ms TTFA on RTX 3060 0.6B (research P1.A).
- **OFR-H** Hann-windowed crossfade chunk-stitching at decoder output boundaries; eliminates click/pop artifacts under Story 18.6's tighter chunk schedule.

**Epic 19 — Lightning Tier (new):**

- **OFR-I** Engine selector: per-utterance choice between Qwen3-TTS (quality tier) and Chatterbox-Turbo (lightning tier); default tier choice persisted in app settings.
- **OFR-J** Chatterbox-Turbo native voice cloning from 5-second reference audio; per-engine voice-reference encoding cache (mirroring Story 17.2's pattern, separate cache namespace).
- **OFR-K** Optional Lightning-tier model download — installer ships *without* Chatterbox-Turbo by default; user opts in via Settings UI (HPE-style "Build with it" pattern). Mitigates the ~1–2 GB bundle delta documented in `memory/production_release_state.md` as a known pain point.
- **OFR-L** Engine-level NFR7 fallback — if Chatterbox-Turbo selected but unavailable (download missing, load failure, hardware mismatch), engine selector emits a user-visible message and falls back to Qwen3-TTS for this utterance.

### Scale & Complexity

| Indicator | Assessment |
|---|---|
| New / refactored components | ~5 (Story 18.4 refinement: `torch_runtime.py` modifications + new `compile_cache.py`; Story 18.5: probe/log added inline to `model_registry.py`; Story 18.6: two-phase scheduler in `codec_token_streamer.py`; Story 18.7: Hann crossfade in `streaming_decoder.py`. Lightning Tier: new `chatterbox_tts_service.py`, new `tts_engine_selector.py`, new `chatterbox_runtime.py`, modifications to `qwen_tts_service.py` + `app_settings.py` + `service_status_indicator.py`) |
| New external dependencies | ✅ Zero for Epic 18 completion (uses existing `torch.compile` + `qwen_tts` API). ⚠️ Two for Lightning Tier: `chatterbox-tts` PyPI package + `chatterbox-streaming` (community-maintained streaming wrapper). |
| Cross-service touchpoints | Engine selector → both TTS services; both services → existing `SessionRegistry`; settings UI → engine-selector state; download manager → Chatterbox model storage |
| Concurrency surface | **Unchanged** for Epic 18 completion (same threading model as parent). **New** for Lightning Tier: engine load/unload transitions if memory-coexistence policy is unload-on-switch (D-32). |
| Risk profile | **Low** for OFR-E/F (well-understood techniques, official upstream API). **Low-medium** for OFR-G/H (two-phase boundary handling and crossfade DSP — both well-precedented). **Medium-high** for Lightning Tier: new model integration, new dependency surface, new UX, new failure modes. |

**Primary domain:** Continues V2's "real-time interactive desktop audio pipeline" framing.
**Complexity level:** Medium for Epic 18 completion (refinement + adjacent stories). Medium-high for Lightning Tier (first multi-engine introduction).

### Technical Constraints & Dependencies

**Inherited constraints (non-negotiable):**

- All V2 baseline constraints (PyQt6, Python 3.11+, PyAudio, file-based JSON config, dual-service audio).
- All optimization-pass constraints (signal contracts P-1 to P-9, qwen-tts pin discipline D-12, telemetry stream D-19/P-9).
- D-9 hardware-aware streaming default — CPU stays SENTENCE_STREAM.
- Single-chokepoint patterns: `_transition_to`, `post_mutation`, `metrics.record`.
- Renegotiated NFR1 per-class targets stand.

**New constraints introduced by this scope:**

- **Compile cache invalidation discipline.** Cache key must include all dimensions that affect compilation correctness; getting this wrong is a Story 17.2-class bug pattern (see `memory/code_review_regression_test_exact_class.md`). Need to enumerate cache-key dimensions in Step 4 (D-24).
- **Decode-window invariant.** Once `enable_streaming_optimizations(decode_window_frames=N)` is called, the streamer's chunk-emit window must match N for the lifetime of the loaded model. Drives D-21 architectural choice (window = 30 vs 80) and D-25 invariant assertion at startup.
- **Chatterbox-Turbo dependency surface.** `chatterbox-tts` (Resemble) for the model + `chatterbox-streaming` (community) for the streaming wrapper. The community wrapper introduces a P-X-style maintenance risk analogous to D-12. New decision: D-33 chatterbox-streaming pin discipline.
- **Engine-selector signal surface.** New `current_engine_changed` signal. Drives D-30 decision on signal contract uniformity with existing engine state.
- **Optional-download installer pattern.** First time MyVoice ships an opt-in model; precedent comes from V2 baseline's tiered Voice Design / Voice Cloning installer options, but the *runtime* download (post-install) is new. Drives D-34 download UX and integrity verification.

### Cross-Cutting Concerns (require Step 4 decisions before tech-spec)

These are surfaced now so Step 4 binds them as decisions and Step 5 codifies them as patterns:

1. **Decode window: dynamic vs fixed.** The single most important architectural choice in this pass. `dynamic=True` is the existing Story 18.4 stub's design but forfeits CUDA Graph replay (most of the speedup). Fixed window enables CUDA Graph but constrains the streamer. Three sub-options for the fixed-window path: (a) align official API to MyVoice's existing 30 (assuming the API accepts arbitrary values — verify before locking), (b) refactor MyVoice's streamer to 80 + perceptual re-audition, (c) keep `dynamic=True` and forfeit gain. **Decision target: D-21.**

2. **Compile cold-start placement.** First call after `enable_streaming_optimizations()` triggers a 10–30s compile. NFR4 (UI <200ms) cannot be violated. Three options: background warmup at app start (UX-best, defers first-utterance success), lazy-with-progress-UI on first call (transparent), shipped pre-baked compile cache (fastest UX, brittle to driver/torch updates). **Decision target: D-23.**

3. **FA2 verification gate.** When FA2 is requested but silently no-ops (per HF Qwen3-TTS-12Hz-0.6B-Base discussion #5), what does MyVoice do? Hard-fail at startup (safe, hostile UX), soft-warn + continue (current implicit behavior), telemetry-only + persist a one-time recommendation banner (recommended). **Decision target: D-26.**

4. **Compile cache key.** Story 18.4's existing stub names `(qwen_tts_pin_hash, model_id, precision_str, torch_version)`. The research says `decode_window_frames` should also be in the key. Other candidates: `attn_implementation`, hardware-capability tier, `compile_mode`. **Decision target: D-24.**

5. **bf16 + compile interaction.** Story 18.3's bf16 audition was deferred pending Story 18.4 results. With compile + CUDA Graph engaged, bf16's tensor-core advantage may finally materialize (the deferred-audition reason). Need explicit decision on whether Story 18.4 takes bf16 as input, fp32 as input, or both as configurable. **Decision target: D-27.**

6. **Engine boundary placement (Lightning Tier).** Where does the engine selector sit in the call stack? Inside `tts_streaming/` (shared infrastructure for both engines), or as a separate module that owns two parallel `tts_*` modules? The shared-infrastructure path is more architecturally consistent but requires more upfront refactoring. **Decision target: D-30.**

7. **Voice-reference cross-engine compatibility.** Qwen3-TTS uses ICL prefix; Chatterbox-Turbo uses speaker-encoder + projection. Single voice profile usable across engines (with per-engine encoded artifacts cached) or per-engine voice profiles? **Decision target: D-31.**

8. **Memory coexistence (Lightning Tier).** RTX 3060 8GB headroom is tight. Both engines resident simultaneously, or unload-on-switch? **Decision target: D-32.**

9. **Lightning-tier optional download UX.** Installer ships without Chatterbox-Turbo. Discoverability path: Settings panel with size disclosure (calm), first-use trigger when user selects Lightning tier (just-in-time), or both? Integrity verification (hash check on download). **Decision target: D-34.**

10. **Streaming-decoder uniformity.** Chatterbox-Turbo yields PCM directly from `generate_stream()`; Qwen3-TTS yields code tokens needing decode. Where do the two streams converge? At `codec_token_streamer` layer (Chatterbox-Turbo bypasses a stage), at one layer up (audio output sink), or via two parallel pipelines that merge at `SessionRegistry`? **Decision target: D-35.**

### Out of Scope (explicit, with reasons)

Mirroring the parent's pattern of explicit exclusion to bound the document:

- **DirectML cross-vendor expansion** — deferred per research P2.C; AMD/Intel-user demand has not surfaced. Memory-pin: `memory/hardware_setup.md` confirms 30xx+ NVIDIA target.
- **Qwen3-TTS GPTQ-Int8 weight quantization** — research-grade; no public Qwen3-TTS-specific GPTQ release exists. Deferred per research P2.A.
- **Megakernel / single-CUDA-kernel-fusion** — explicitly skipped per research P2.B; engineering cost (6+ months CUDA expertise) outweighs gain over P0+P1.
- **NFR1 re-litigation** — Story 16.9's per-class targets stand. This scope adds *headroom*, not a new contract.
- **NeuTTS Air integration as a third engine** — research-strong second-place candidate, but adds llama.cpp runtime; out of scope for this pass to keep the multi-engine introduction focused on a single new engine. Tracked as future enhancement.
- **OmniVoice integration** — too recent (released March 31, 2026, ~6 weeks ago); production-deployment durability unverified. Tracked as future enhancement.
- **F5-TTS** — non-AR architecture inherently blocks true streaming per upstream caveat; disqualified.
- **XTTS-v2** — CPML license blocks commercial Windows-installer redistribution; disqualified.
- **Multi-listener perceptual A/B re-audition for the streaming-mode A/B** — Story 17.1's verdict (PASS, 2026-05-08, 30/30 `none` flags) stands. Story 18.4 + 18.7 trigger a *narrower* compiled-output audition (single fixture, focused on click/pop at chunk boundaries), not the full Story 17.1-grade gate.
- **PRD back-propagation of OFR-E through OFR-L** — administrative follow-up; owner: PM/Commander; not blocking this architecture pass. Tracked here so it doesn't go cold.

## Starter Template Evaluation

**Not applicable — brownfield project, brownfield-deep.** The existing MyVoice V2 codebase, governed by `architecture.md` (sealed 2026-01-31) and `architecture-optimization-pass.md` (sealed 2026-04-27), is the foundation. No greenfield starter selection is needed; this section instead documents the new dependencies this scope adds, the upstream-version status of those dependencies, and the pin discipline that will govern them.

### Inherited Baseline (no change)

The complete baseline is documented in the parent and grandparent architecture documents and is not restated here. The parent's "Inherited Codebase Baseline" section enumerates all retained services, UI components, and the tech stack (PyTorch ≥2.0, PyQt6 ≥6.6, qasync ≥0.27, soundfile ≥0.12, PyAudioWPatch ≥0.2.12.6, numpy ≥1.24). Per `memory/hardware_setup.md`, the dev/ship platform is RTX 5090 Blackwell on Windows 11 with `torch 2.10+cu128`; ship-target also covers RTX 30xx/40xx hosts.

### New Dependencies Introduced by This Scope

This scope adds three new dependency surfaces. Each is paired with the pin-discipline decision that will govern it (target IDs cited; full decisions land in Step 4).

#### 1. The official `qwen-tts` optimization API (Epic 18 completion)

The compile + CUDA Graph path uses `model.enable_streaming_optimizations(decode_window_frames=N, use_compile=True, compile_mode="reduce-overhead")`, which is part of the official `qwen-tts` package's `Qwen3TTSModel` class. _Sources: [QwenLM/Qwen3-TTS (GitHub)](https://github.com/QwenLM/Qwen3-TTS), [qwen-tts (PyPI)](https://pypi.org/project/qwen-tts/)._

| Aspect | Status | Action |
|---|---|---|
| MyVoice's current pin | `git+https://github.com/QwenLM/Qwen3-TTS.git@1ab0dd75` (= qwen-tts 0.0.4 per Story 16.1) | Verify `enable_streaming_optimizations` exists at this commit before locking design (Step 4 D-22 has a "verify-or-bump" branch) |
| If method exists at `1ab0dd75` | Use existing pin | Add import-attribute test per D-12 pattern, asserting `Qwen3TTSModel.enable_streaming_optimizations` is callable |
| If method doesn't exist at `1ab0dd75` | Pin-bump required | Re-audit upstream HEAD for known-good commit; full perceptual A/B re-audition required per parent's NFR3 row line 803 (Story 17.1 protocol) |

**Pin-discipline policy (D-22 placeholder):** any change to the `qwen-tts` pin triggers (a) updating `tests/test_qwen_tts_internals.py` to cover the new symbols touched, and (b) the perceptual A/B re-audition gate. No silent pin-bumps.

#### 2. `chatterbox-tts` — Resemble AI (Lightning Tier, Epic 19)

The model package and inference API for Chatterbox-Turbo. Resemble AI maintains it directly. _Sources: [resemble-ai/chatterbox (GitHub)](https://github.com/resemble-ai/chatterbox), [chatterbox-tts (PyPI)](https://pypi.org/project/chatterbox-tts/), [Chatterbox Turbo (Resemble AI)](https://www.resemble.ai/chatterbox-turbo/), [ResembleAI/chatterbox-turbo (HF)](https://huggingface.co/ResembleAI/chatterbox-turbo)._

| Aspect | Value |
|---|---|
| Current PyPI version | **0.1.7** (with 0.1.6 also published) |
| License | **MIT** ✅ — clean for commercial Windows-installer redistribution |
| Variants | Original Chatterbox (English, emotion control), Chatterbox-Turbo (350M, distilled one-step decoder, 75 ms latency), Chatterbox Multilingual (23 languages) — **Lightning tier targets Chatterbox-Turbo specifically** |
| Maintenance | Upstream-maintained by Resemble AI; actively developed |
| Voice clone reference | 5 s minimum (Resemble docs) |

**Pin-discipline policy:** standard PyPI version pin (e.g., `chatterbox-tts==0.1.7`) plus an import-attribute test mirroring D-12's pattern for the symbols MyVoice imports (`ChatterboxTurboTTS.from_pretrained`, `model.generate`, and the streaming-wrapper integration points).

#### 3. `chatterbox-streaming` — community wrapper (Lightning Tier, Epic 19)

Community-maintained streaming wrapper providing the `generate_stream(chunk_size=N)` generator API on top of Resemble's base model. _Sources: [davidbrowne17/chatterbox-streaming (GitHub)](https://github.com/davidbrowne17/chatterbox-streaming), [chatterbox-streaming (PyPI)](https://pypi.org/project/chatterbox-streaming/), [DeepWiki — Getting Started](https://deepwiki.com/davidbrowne17/chatterbox-streaming/2-getting-started)._

| Aspect | Value |
|---|---|
| Current PyPI version | **0.1.2** (released June 5) |
| License | **MIT** (inherited from Chatterbox) |
| Reported performance | RTF 0.499 on RTX 4090, ~472 ms first-chunk latency |
| Maintainer | `davidbrowne17` — community, not Resemble AI canonical |
| Maintenance risk | **Medium-low.** Not blessed by Resemble; could lag upstream Chatterbox releases. Two community alternatives exist (`MeiyuJ/chatterbox-streaming`, `stl314159/chatterbox-streaming-api`) — same upstream repo lineage. |

**Pin-discipline policy (D-33 placeholder):** stricter than `chatterbox-tts` — pin to specific commit hash (not just version), add import-attribute test for the streaming-wrapper symbols (`ChatterboxTTS.generate_stream` signature + `chunk_size` parameter), and include a quarterly check on whether Resemble has shipped an official streaming API that would let us drop the community dependency.

#### 4. `flash-attn` — version compatibility note (Epic 18 completion, OFR-F)

This isn't a new dependency (already part of MyVoice's runtime via the `flash_attn` import probe in `model_registry.py:488–512`), but the version compatibility is load-bearing for OFR-F (FA2 runtime verification). _Sources: [Definitive Guide to PyTorch, CUDA, Flash Attention compatibility (Medium)](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10), [HuggingFace transformers issue #44559](https://github.com/huggingface/transformers/issues/44559)._

| Aspect | Status |
|---|---|
| MyVoice probes for | `flash_attn` import; if available, sets `attn_impl="flash_attention_2"` |
| FA2 requirements | fp16/bf16 precision (Story 18.3 already aligned bf16 on Ampere+), CUDA ≥12.0, PyTorch ≥2.2 |
| FA4 status | **Not supported via `attn_implementation="flash_attention_2"`** — FA4 lives at `flash_attn.cute` with a different integration path |
| Recommended pin range | `flash-attn>=2.7.0,<3.0.0` (FA2 family); avoid FA4 |
| Risk | Without explicit pin, a future environment upgrade to FA4 silently no-ops the FA2 path |

**Pin-discipline policy (relates to D-22 + D-26):** add `flash-attn` pin range to `requirements.txt`; OFR-F's runtime probe asserts FA2 was applied (not just requested) and emits the `attn_implementation_active` telemetry field per D-19/P-9.

### What This Scope Does NOT Add

For clarity and to bound dependency creep:

- **No new model frameworks.** PyTorch + transformers continue to be the runtime; ONNX Runtime is *not* introduced (HPE's pattern; out of scope per research synthesis).
- **No new audio libraries.** PyAudioWPatch + soundfile + numpy continue to handle the audio side. Hann crossfade (OFR-H) uses `scipy.signal` (already a transitive dep via numpy ecosystem; verify in Step 6).
- **No new packaging tooling.** PyInstaller continues; Lightning Tier optional download adds a runtime download manager (decision D-34) but does not change build/install tooling.
- **No new test frameworks.** pytest + pytest-asyncio + pytest-qt continue from V2.

### Dependency Manifest (target state for this pass)

To be appended to `requirements.txt` in the Phase that introduces each dependency (per D-20 phasing analog):

```diff
  # Inherited (unchanged)
  qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@1ab0dd75
  PyQt6>=6.6.0
  qasync>=0.27.0
  torch>=2.0
  numpy>=1.24
  soundfile>=0.12.1
  PyAudioWPatch>=0.2.12.6
  flash-attn>=2.7.0,<3.0.0    # NEW pin range — Epic 18.5 OFR-F
  scipy>=1.10                  # verify already in tree; required for Hann window — Epic 18.7 OFR-H

  # Lightning Tier (Epic 19) — added when Phase L1 ships
+ chatterbox-tts==0.1.7
+ chatterbox-streaming @ git+https://github.com/davidbrowne17/chatterbox-streaming.git@<pinned-commit>
```

The exact `chatterbox-streaming` commit hash is captured at Phase L1 implementation time; the `<pinned-commit>` placeholder is intentional — it's not chosen by the architecture, but the pin-discipline policy is.

### Initialization Command

**Not applicable** — no project bootstrap. The architectural equivalent is captured as a recommended **dependency-pin update** (per Step 4 / D-22 / D-33 decisions to follow), executed in the phases defined in Step 6's migration map.

### Web Search Note

Step 3's protocol normally calls for web-search version verification of starter templates. That doesn't apply for a brownfield project, but the dependency-version verification *does* — performed and recorded above. One open verification item explicitly deferred to Step 4:

- **Does qwen-tts at commit `1ab0dd75` (= 0.0.4) ship `enable_streaming_optimizations()`?** If yes: D-22 keeps existing pin + adds import-attribute test. If no: D-22 specifies a pin-bump to a verified-good commit + triggers the parent's NFR3 perceptual A/B re-audition gate. This is verified empirically in the implementation phase, not as part of this architecture pass.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical decisions (block implementation):** D-21, D-22, D-23, D-24, D-26, D-30, D-31, D-32, D-34, D-38

**Important decisions (shape architecture):** D-25, D-27, D-28, D-29, D-33, D-35, D-36, D-37

**Inherited from parent (not re-decided):** D-1 through D-20 stand unchanged. P-1 through P-9 stand unchanged. Story 16.9 NFR1 per-class targets stand unchanged.

### Cluster E — Compile + CUDA Graph (Epic 18 completion)

**D-21 Decode-window strategy: fixed at 30, matching MyVoice's existing streamer.**
The official `enable_streaming_optimizations(decode_window_frames=N)` API requires a fixed `N` for CUDA Graph replay; `dynamic=True` (the existing Story 18.4 stub's choice) forfeits the technique's largest reported speedup. MyVoice's streamer (per Story 16.8 design at `qwen_tts_service.py:3404-3571`) uses `chunk_size=25 + lookahead=5 = 30`. This decision passes `decode_window_frames=30` to the official API.
- **Three sub-options were considered:** (a) align official API to MyVoice's 30 [chosen]; (b) refactor MyVoice's streamer to 80 + full Story-17.1-grade NFR3 perceptual re-audition; (c) keep `dynamic=True` and forfeit the gain. Option (a) wins because the 30 value was validated by Story 17.1 audition (PASS, 30/30 zero `audible_seam`) and option (b) costs an N=3-listener re-audition cycle without commensurate gain.
- **Verification gate (impl-time, not architecture-time):** confirm `enable_streaming_optimizations` accepts arbitrary `decode_window_frames` values (not only 80). If the API hard-codes 80, fall back to option (b) with explicit re-audition trigger captured at that point. **Architecture defines the fall-back path; this is not a silent failure mode.**
- **Rationale:** preserves Story 17.1's perceptual verdict; preserves the entire community-fork speedup envelope; matches the upstream-blessed pattern.

**D-22 `enable_streaming_optimizations` API pin discipline.**
The empirical question — does qwen-tts at commit `1ab0dd75` (= 0.0.4) ship `enable_streaming_optimizations`? — has two architecturally-designed branches:
- **Branch A — method exists at `1ab0dd75`:** keep existing pin. Add `Qwen3TTSModel.enable_streaming_optimizations` to `tests/test_qwen_tts_internals.py` per D-12 pattern. Assert callable + verify required kwargs (`decode_window_frames`, `use_compile`, `compile_mode`).
- **Branch B — method does not exist at `1ab0dd75`:** pin-bump to verified-good upstream commit. **Triggers full Story-17.1-grade NFR3 perceptual re-audition** before the bump merges, per parent doc's NFR3 row policy line 803.
- **Rationale:** brittle integration is acceptable if loud (D-12 lineage). Both branches are designed; the verification picks the branch.

**EXECUTED 2026-05-11 per Story 18.4 — Branch B fired.** The 2026-05-10 research subagent + dev-agent Task 1.1 verification both confirmed `enable_streaming_optimizations` does NOT exist at `1ab0dd75` (zero grep matches across `python310/Lib/site-packages/qwen_tts/`). Pin bumped from `QwenLM/Qwen3-TTS@1ab0dd75353392f28a0d05d9ca960c9954b13c83` to `dffdeeq/Qwen3-TTS-streaming@3fdb468233d73fa537202b94a1cc7c4e7a6160b8` (community fork, "compile and fast codebook" commit 2026-02-03; same `qwen-tts 0.0.4` package name; +50/-6 lines additive diff; drop-in replacement). Joint NFR3 audition **FULL PASS** — 3 listeners × 10 utterances × A/B = 60 defect observations; zero `audible_seam` flags on bf16+compile (and zero on fp32_eager); 1/30 trials non-equivalent preference (favors bf16_compile). See `architecture-optimization-pass.md` §"Story 18.4 Follow-up Note (Joint bf16 + Compile + Pin-Bump Audition — FULL PASS, 2026-05-11)".

**D-23 Compile cold-start: background warmup at app start + persistent compile cache.**
First call after `enable_streaming_optimizations()` triggers a 10–30s `torch.compile` invocation; this MUST NOT block the Qt UI thread (NFR4).
- **Cold-start strategy:** at app startup, after model load completes, dispatch a background warmup worker thread that runs one synthetic compile-priming generation (mocked text, default voice). UI shows "Preparing TTS engine…" indicator (mirrors Story 17.2's "Preparing voice…" pattern).
- **Persistent cache:** at `%LOCALAPPDATA%/MyVoice/torch_compile_cache/`, populated by the cold compile, reused on subsequent runs (effectively turning option (a) "background warmup" into option (c) "pre-baked cache" after first run).
- **Lazy fallback path:** if for any reason the background warmup is skipped (e.g., test mode, env-var override, headless mode), the first user-facing generation triggers compile inline, with a "Compiling…" indicator shown for 10–30s. **Acceptable degraded path; not the steady-state UX.**
- **Three options were considered:** (a) background-on-startup [chosen — primary path]; (b) lazy-with-progress-UI [chosen — fallback path]; (c) shipped pre-baked compile cache [rejected — brittle to driver/torch updates; the persistent cache built at runtime achieves the same UX without the brittleness].

**D-24 Compile cache key dimensions: 7 dimensions.**
Story 18.4's existing stub names 4 dimensions; research and downstream decisions add 3 more. Final cache key:

```python
cache_key = sha256("|".join([
    qwen_tts_pin_hash,           # existing
    model_id,                     # existing — e.g., "Qwen3-TTS-12Hz-1.7B-Base"
    precision_str,                # existing — "bf16" | "fp32"
    torch_version,                # existing — e.g., "2.10.0+cu128"
    str(decode_window_frames),    # NEW — affects compiled graph shape (D-21)
    f"{cuda_capability[0]}.{cuda_capability[1]}",  # NEW — graph specialization per HW (e.g., "8.9", "12.0")
    compile_mode,                 # NEW — "reduce-overhead" today; could change
]).encode()).hexdigest()
```

- **Skipped on purpose:** `attn_implementation` and `flash_attn_version` — these affect *which kernels* run but not the compiled graph's *shape*. They invalidate via D-26's verification gate, not via cache key.
- **Cache eviction:** if any dimension changes, cache miss triggers re-compile. No size cap on the cache directory in v1; future enhancement if disk usage becomes a concern.
- **Anti-pattern guardrail:** the cache-key tuple is constructed in exactly one place (a `compile_cache.compute_key()` helper). Story 17.2's H1/H2 lessons (per `memory/code_review_regression_test_exact_class.md`) inform this — multiple call sites would let dimensions drift silently.

**D-25 Decode-window invariant assertion at startup.**
Adding the `decode_window_frames` cache-key dimension surfaces a config-drift risk: if the streamer's `chunk_size + lookahead` ever drifts from the cached compile graph's `decode_window_frames`, CUDA Graph replay produces silently wrong audio.
- **Assertion at app startup, immediately after `enable_streaming_optimizations()` returns:**

```python
expected_window = streamer.chunk_size + streamer.lookahead  # 25 + 5 = 30
assert expected_window == compile_cache.decode_window_frames, (
    f"Decode-window drift: streamer expects {expected_window}, "
    f"compile cache expects {compile_cache.decode_window_frames}. "
    f"This is a config bug; CUDA Graph replay would produce wrong audio."
)
```

- **Failure mode:** raise loud `AssertionError` at startup. Do not silently proceed. UI shows error dialog with text recommending `tools/clear_compile_cache.py` (a developer tool we'll ship in Phase 18.4).
- **Rationale:** Story 17.2-class bug class — silent cache misuse is the failure pattern this scope must prevent.

**D-26 FA2 verification gate behavior: telemetry-only in prod, hard-fail in dev/CI.**
The HF Qwen3-TTS-12Hz-0.6B-Base discussion #5 documents that `attn_implementation="flash_attention_2"` may silently no-op on Qwen3-TTS in some `transformers` versions. MyVoice's current `model_registry.py:488–512` requests FA2 conditionally but does not verify it took effect.
- **Production policy (default):** runtime probe at model load checks `model.config._attn_implementation` (or equivalent — confirm the actual attribute name at impl time; HF's API surface is unstable). Emits `attn_implementation_active` telemetry tag per D-19/P-9. If user requested FA2 but got `eager` or `sdpa`: persist a one-time settings banner ("FlashAttention-2 is not active — see Settings → Performance for diagnostics"). **No hard fail in production.**
- **Dev/CI policy (env-var-gated):** `MYVOICE_REQUIRE_FA2=1` triggers a hard `AssertionError` on FA2 not being applied. CI sets this env var to catch silent regressions. Maintainer dev workflow can also enable it for local debugging.
- **Three options were considered:** (a) hard-fail at startup in prod [rejected — hostile UX for users on environments where FA2 install fails]; (b) soft-warn + continue [rejected — hides the issue from the user, leaves them confused about latency]; (c) telemetry + persistent-banner-on-mismatch + dev-gate hard-fail [chosen — production-friendly, dev-strict, observable].

**D-27 bf16 + compile interaction: precision is Story 18.3's call; Story 18.4 inherits it.**
Story 18.3 deferred its perceptual audition pending Story 18.4 results. With compile + CUDA Graph engaged, bf16's tensor-core advantage may finally materialize on the V2 inference workload (the deferred audition's reason).
- **D-27.1 Precision policy at compile layer:** Story 18.4 takes whatever precision Story 18.3's `AppSettings.tts_precision` resolves to (bf16 on Ampere+ default, fp32 fallback, user-overridable). The compile path traces whatever precision is loaded; no precision policy at the compile layer.
- **D-27.2 Cache invalidation on precision switch:** D-24's cache key includes `precision_str`, so cache invalidates cleanly on switch — first generation after switch is a cold compile.
- **D-27.3 Story 18.3 deferred audition resumes when this scope ships:** post-Story-18.4 retrospective re-runs Story 18.3's NFR1 measurement under compile-engaged. If bf16 finally outperforms fp32+TF32, the deferred Task 8 audition (≥3-listener perceptual A/B) fires per Story 18.3 OQ #3 option (b). If not, the recommended default flips from bf16 to fp32 on Ampere+ and `tts_precision="auto"` resolution updates.
- **Rationale:** doesn't pre-commit Story 18.3's outcome; integrates naturally with the deferred-audition framing.

**D-28 Two-phase emission scheduler: vary `emit_every_frames`, hold `decode_window_frames` fixed.**
Research's `rekuenkdr/Qwen3-TTS-streaming` two-phase scheduler varies *both* `emit_every_frames` and `decode_window_frames`. D-21's fixed-window choice forecloses the latter. Architectural compromise:
- **Phase 1 — aggressive small-emit for first chunk:** `emit_every_frames=2` (~160 ms intervals at 12.5 Hz). Triggers from generation start until first 30 frames buffered (≈ first chunk emitted).
- **Phase 2 — steady-state moderate-emit:** `emit_every_frames=5` (~400 ms intervals). Activates after Phase 1 transition; matches MyVoice's existing default rhythm.
- **`decode_window_frames=30` throughout** (per D-21).
- **Performance trade:** rekuenkdr's fork reports 2.75× TTFA reduction (208 ms vs 570 ms baseline) using both knobs. Holding decode_window fixed gives ~1.5× TTFA reduction, which is still meaningful but less dramatic. **Acceptable trade for graph-cache validity.**
- **Phase transition smoothness:** D-29's Hann crossfade applies *to the chunk emission boundary*, masking the rhythm change between Phase 1 and Phase 2 chunks.

**D-29 Hann crossfade: 1024-sample window with 512-sample overlap, applied in `streaming_decoder.py`.**
- **Window:** `numpy.hanning(1024)` — first 512 samples are fade-in, last 512 are fade-out.
- **Overlap:** each emitted chunk shares its tail-512-samples with the next chunk's head-512-samples; the producer emits slightly more than the consumer drains.
- **Math:** at 24 kHz, 512 samples ≈ 21.3 ms. `decode_window_frames=30` produces ~30 × 80 ms = 2400 ms per chunk; overlap is < 1% of chunk. Imperceptible duplication, click/pop-free transitions.
- **Bounded queue accounting:** the maxsize=100 backpressure (parent D-10) is sized in *tokens*, not PCM samples; overlap is post-decode, post-queue. Producer-side queue accounting unaffected.
- **Application:** in `streaming_decoder.py`'s post-decode path, before posting via `registry.post_mutation('append_chunk', session_id, pcm)` (parent P-3, P-6).
- **Test:** click/pop detector — peak-derivative magnitude at chunk boundaries should be < 2σ above intra-chunk peak-derivative. Regression test added per D-19/P-9 telemetry stream tag `chunk_boundary_peak_derivative`.

### Cluster F — Lightning Tier (Epic 19, new)

**D-30 Engine boundary placement: shared infrastructure with engine-specific adapters.**
A new `services/tts_engines/` namespace introduces a thin abstraction layer above the existing TTS service. Engine selector routes to one of two adapters:

```
src/myvoice/services/
├── tts_engines/                          ← NEW (Phase L1)
│   ├── base_tts_engine.py                ← Abstract base — engine contract
│   ├── qwen3_tts_engine.py               ← Refactor of qwen_tts_service.py
│   └── chatterbox_turbo_engine.py        ← New engine
├── tts_engine_selector.py                ← NEW (Phase L1)
└── tts_streaming/                        ← Existing — extended
    ├── codec_token_streamer.py           ← Existing — Qwen3-specific
    ├── pcm_chunk_streamer.py             ← NEW (Phase L2) — Chatterbox pass-through
    └── streaming_decoder.py              ← Existing — extended
```

- **`BaseTTSEngine` contract** (sketch — Step 5 makes it pattern-rigorous):
  - `async def generate(text, voice, session_id, settings) -> AsyncIterator[PCM_chunk]`
  - `def warmup() -> None` — pre-priming hook for engine-specific compile/cache
  - `def get_supported_voices() -> List[VoiceProfile]`
  - `def encode_voice_reference(voice: VoiceProfile) -> EngineEncodingArtifact`
- **Single `SessionRegistry`** continues to own all sessions regardless of engine (D-1 stands).
- **Sessions don't know which engine produced them** — engine identity is metadata, not lifecycle.
- **Two options considered:** (a) shared infrastructure under `tts_streaming/` [chosen]; (b) parallel module trees per engine [rejected — would force Lightning Tier to duplicate `SessionRegistry` integration, audio-coordinator wiring, telemetry plumbing].
- **Cost:** Phase L1 includes a refactor of `qwen_tts_service.py` to implement `BaseTTSEngine`. Mechanical, low-risk; isolated from the producer-bottleneck work.

**D-31 Voice-reference cross-engine compatibility: single profile + per-engine encoded artifact cache.**
- **`VoiceProfile`** (existing V2 baseline data class) gains an optional `engine_encodings: Dict[str, EngineEncodingArtifact]` field. Empty by default; populated lazily on first use of a voice with an engine.
- **Per-engine encoder runs once** (mirroring Story 17.2's `voice_clone_prompt` cache pattern); artifact cached on disk at:
  ```
  %LOCALAPPDATA%/MyVoice/voices/<voice_id>/engine_encodings/<engine_id>.bin
  ```
- **Cache invalidation:** D-33's pin-bump policy invalidates an engine's cache (delete `<engine_id>.bin` if engine pin changes).
- **UI compatibility messaging:** voice library shows per-engine status: "Voice X — Qwen3 ✓ • Chatterbox-Turbo ⏳ encoding…". Encoding runs in background on first selection.
- **Reference-clip duration policy:**
  - Qwen3 ICL works with ≥3 s reference clips.
  - Chatterbox-Turbo requires ≥5 s clean reference (per Resemble docs).
  - If user uploads a 4 s clip and selects Chatterbox-Turbo: UX surface "This voice clip is too short for Chatterbox-Turbo (≥5 s required); falling back to Qwen3-TTS for this voice on this engine. Re-upload a longer clip to enable Lightning Tier for this voice."
- **Rationale:** preserves user mental model (one voice = one identity); per-engine details are infrastructure concerns hidden from primary workflow.

**D-32 Memory coexistence: hybrid auto policy with explicit override.**
RTX 3060 8GB headroom is tight: Qwen3-TTS 1.7B = ~3.4 GB; Chatterbox-Turbo 350M = ~700 MB–1.4 GB depending on dtype. Both warm = ~5 GB combined; tight with overhead.

- **Default policy `tts_engine_coexistence="auto"`:**
  - At engine-load time, query `torch.cuda.mem_get_info()` for free + total VRAM.
  - If `(loaded_engine_size + new_engine_size) / total_vram < 0.70`: keep both warm (instant switching).
  - Else: unload current engine before loading new one. Switch cost ≈ 2–5 s with a "Switching engine…" indicator.
- **Manual override `tts_engine_coexistence: "auto" | "always-warm" | "unload-on-switch"`** in `AppSettings`.
  - `"always-warm"` for users who prioritize switching speed and have headroom.
  - `"unload-on-switch"` for users who want predictable VRAM behavior.
- **VRAM accounting on switch:** explicit `torch.cuda.empty_cache()` after unload before next load; assert `mem_get_info()` reports expected reclamation; raise telemetry warning if VRAM doesn't free as expected (signals memory leak).
- **Inheritance from V2 baseline:** parent's "lazy single-model loading discipline" is preserved as a steady-state policy when memory is tight (`unload-on-switch` is the default V2 behavior extended to multi-engine).
- **NFR11 (<4GB RAM) compliance:** RTX 3060 8GB users default to `unload-on-switch`, which keeps NFR11 satisfied (only one engine resident at a time). RTX 4090 users get `always-warm` automatically (5 GB combined < 70% × 24 GB = 16.8 GB threshold).

**D-33 `chatterbox-streaming` pin discipline: commit-hash pin + import-attribute test + quarterly check.**
The community wrapper introduces a maintenance risk analogous to D-12's qwen-tts policy.
- **Pin to specific commit hash** in `requirements.txt`:
  ```
  chatterbox-streaming @ git+https://github.com/davidbrowne17/chatterbox-streaming.git@<pinned-commit>
  ```
- **Add `tests/test_chatterbox_streaming_internals.py`** asserting:
  - `from chatterbox.tts import ChatterboxTTS` succeeds
  - `hasattr(ChatterboxTTS, 'generate_stream')` is True
  - `inspect.signature(ChatterboxTTS.generate_stream)` includes `chunk_size`, `audio_prompt_path`
- **Quarterly task** (added to `memory/` as a project-memory entry): check whether Resemble has shipped an official `generate_stream()` API in the canonical `chatterbox-tts` package. If yes: drop the community dependency, swap import paths, re-pin.
- **Rationale:** brittle integration is acceptable if loud (D-12 lineage); silent breakage is not.

**D-34 Optional download UX: settings panel + first-use trigger + SHA-256 integrity verification.**
Lightning Tier (Chatterbox-Turbo) ~1.5 GB model bundle ships as an optional download to manage installer-size impact (per `memory/production_release_state.md` known pain point).

- **Discoverability — both surfaces:**
  - **Settings panel "TTS Engines" section** displays Lightning Tier with size disclosure ("~1.5 GB") and explicit Download button. Status shown: not downloaded / downloading X% / downloaded / verification failed.
  - **First-use trigger from main UI:** if user picks Lightning Tier from main interface without prior download, just-in-time prompt: "Download Lightning Tier model (~1.5 GB)? [Download] [Cancel]". Clicking Download opens settings panel scrolled to that section (single download surface, not duplicated).

- **Download mechanism — new `services/tts_model_download_manager.py`:**
  - Downloads to `%LOCALAPPDATA%/MyVoice/lightning_tier_models/`.
  - Runs in dedicated `QThread` (per V2 baseline threading model).
  - Progress reported via `download_progress_changed` signal.
  - Resumes on interruption via HTTP Range requests.
  - **SHA-256 hash check** on completion (hash bundled in app source as a constant, not downloaded — supply-chain protection).
  - On hash mismatch: alert user, do not load model, request re-download. Delete corrupt artifact.
- **Failure mode integration with D-36 NFR7 fallback:** if user selects Lightning but model not downloaded, engine selector emits a one-time prompt; subsequent generations fall back silently to Qwen3 with a "Lightning Tier unavailable — using Quality Tier" toast.
- **HPE-style "Build with it" precedent:** mirrors HPE's optional model profiles. Different from V2 baseline's *install-time* tiered model selection; this is a *runtime* download. New pattern; documented in Step 5 (P-13 — "Optional model download lifecycle").

**D-35 Streaming-decoder uniformity: converge at the registry-bound `append_chunk(pcm)` callback.**
Both engines emit PCM chunks via the same registry callback; the difference is *what produces the PCM*:

| Engine | Pipeline |
|---|---|
| Qwen3-TTS | talker → CodecTokenStreamer → StreamingDecoder → PCM → registry |
| Chatterbox-Turbo | ChatterboxTTS.generate_stream → PCM (already decoded) → registry |

- **New `pcm_chunk_streamer.py` (Phase L2):** thin pass-through wrapping Chatterbox's `generate_stream()` generator. Reads PCM chunks, normalizes format (sample rate matching to MyVoice's expected 24 kHz; dtype normalization to int16 for the audio coordinator), posts to registry via the same `append_chunk` callback shape.
- **`StreamingDecoder` (existing) remains Qwen3-specific** but gets a small refactor: separates the codec-token-decode logic (Qwen3) from the chunk-stitching/Hann-crossfade logic (engine-agnostic). The crossfade path can apply to both engines if Chatterbox ever exhibits chunk-boundary artifacts (currently not reported; reactive measure).
- **Rationale:** the registry's `append_chunk(session_id, pcm)` is the existing convergence point (parent P-6); pulling Chatterbox into it costs less than building a parallel sink.

**D-36 Engine-level NFR7 fallback: silent fallback with one-time UI notification.**
When Lightning Tier is selected but unavailable:

- **Failure modes triggering fallback:**
  1. Chatterbox model not downloaded
  2. Chatterbox model load failure (OOM, file corruption)
  3. Chatterbox generation failure mid-utterance
  4. CUDA unavailable when Lightning Tier explicitly requested

- **Fallback policy:**
  - Engine selector wraps Lightning Tier load in try/except.
  - On exception: log error, emit `engine_fallback_occurred(requested_engine, fallback_engine, reason)` signal.
  - **Case 1 (model not downloaded) — actionable toast:** "Lightning Tier not installed — using Quality Tier. [Install Lightning Tier]". Click navigates to Settings.
  - **Case 4 (CUDA unavailable + Lightning requested) — silent fallback** to Qwen3 SENTENCE_STREAM (already covered by parent D-9 hardware-aware default). Lightning Tier requires CUDA per Chatterbox's published platform support.
  - **Cases 2 and 3 — ephemeral toast:** "Lightning Tier failed for this generation — using Quality Tier. [Details]". Click opens log viewer.
- **Composition with parent NFR7:** existing TRUE_STREAM ← SENTENCE_STREAM ← BATCH chain is preserved for Qwen3; Lightning ← Quality is the new outermost fallback layer.

**D-37 Per-engine telemetry tags: every metric record includes `engine: str`.**
Extends parent D-19/P-9 metric format:

```python
metrics.record(
    'first_chunk_latency_ms',
    250,
    session_id=session_id,
    engine='qwen3',                # NEW field — required for engine-stratified analysis
    tags={'model_type': '...', 'hardware': 'gpu', ...},
)
```

- **Backward-compatible:** `engine` defaults to `'qwen3'` if not specified, so existing telemetry call sites continue to work without modification during the migration.
- **Required for Lightning Tier:** Phase L1 adds explicit `engine='chatterbox-turbo'` to all Chatterbox-emitted metrics.
- **Enables per-engine performance comparison** in retrospectives (e.g., "TTFA p95 per engine" splits cleanly).

### Cluster G — Migration Plan

**D-38 Phased migration plan.**
Mirrors parent D-20's phased structure. Each phase is a self-contained PR-able unit; reverts cleanly.

| Phase | Story | Deliverable | Reverts cleanly? | Dependencies |
|---|---|---|---|---|
| 18.4 | Story 18.4 (refined) | `enable_streaming_optimizations()` wired in `torch_runtime.py`; `compile_cache.py` (7-dim cache key D-24); background warmup at app start (D-23); decode-window invariant assertion (D-25); test_qwen_tts_internals updates (D-22) | Yes (cache + setting toggle) | None within scope |
| 18.5 | Story 18.5 (NEW) | FA2 runtime verification probe + `attn_implementation_active` telemetry + dev-gate hard-fail env var (D-26) | Yes (probe-only feature) | 18.4 |
| 18.6 | Story 18.6 (NEW) | Two-phase emission scheduler in `codec_token_streamer.py` (D-28) | Yes (revert to single-phase emit_every) | 18.4 |
| 18.7 | Story 18.7 (NEW) | Hann crossfade chunk-stitching in `streaming_decoder.py` (D-29) | Yes (bypass crossfade flag) | 18.4 |
| **(gate)** | — | Story 18.1 metrics re-run: target ratio < 1.0× sustained, RTF ≥ 1.0 | — | 18.4 + 18.5 + 18.6 + 18.7 |
| L1 | Story 19.1 (NEW) | `BaseTTSEngine` contract + `qwen3_tts_engine.py` refactor (no UX yet) + `tts_engine_selector.py` skeleton + `engine` telemetry tag (D-37) | Yes (router defaults to qwen3) | Stories 18.4–18.7 (cleaner diff isolation; producer ratio confirmed first) |
| L2 | Story 19.2 (NEW) | `chatterbox_turbo_engine.py` + `pcm_chunk_streamer.py` + voice-reference re-encoding cache (D-31) + memory-coexistence policy (D-32) | Yes (engine selectable but defaulted off) | L1 |
| L3 | Story 19.3 (NEW) | Settings UI for engine selection + optional download manager (D-34) + main-window engine switcher + UX polish + NFR7 engine-level fallback (D-36) | Yes (UI hidden behind setting flag) | L2 |

**Sequencing rationale:**
- Stories 18.4–18.7 land in Epic 18 first; each delivers user-visible value (latency reduction) independently.
- Lightning Tier Phase L1 starts only after Stories 18.4–18.7 close, so the refactor (`qwen3_tts_engine.py`) operates on the producer-bottleneck-closed code, not on the in-flux code. Cleaner diff; safer.
- L1 is *no user-visible change* (the router defaults to Qwen3, the only existing engine). Validates the abstraction.
- L2 is *engine selectable but off by default* (gated behind a hidden `AppSettings` flag for testing).
- L3 is *user-visible Lightning Tier* with full UX. Optional download lands here.

### Cross-Component Dependencies (this scope)

- **D-21 + D-25 are tightly coupled.** Decode-window strategy and invariant assertion are two halves of the same architectural commitment.
- **D-23 + D-24 are coupled.** Cache key dimensions are hand-in-glove with cache lookup at warmup time.
- **D-28 depends on D-21.** Two-phase scheduler design is constrained by the fixed-window choice.
- **D-29 depends on D-28.** Hann crossfade is the smoothness mechanism that makes the two-phase rhythm-shift imperceptible.
- **D-30 enables D-31, D-32, D-35.** The engine boundary placement is the structural decision; the others are operational policies on top of it.
- **D-34 depends on D-32.** The download UX must understand the memory-coexistence policy to make sensible recommendations during install.
- **D-36 depends on D-30 (the engine-selector exists) and D-32 (knows how to unload safely).**

### Decisions Explicitly *Not* Made Here (delegated to per-feature tech-spec)

- Story-level acceptance criteria for each Story 18.4/5/6/7 + Story 19.1/2/3 — owned by `/bmad-bmm-create-story` runs (Bob/SM).
- Specific UI layout for the engine selector in main window — UX review territory.
- Specific UI layout for Lightning Tier settings panel — UX review territory.
- Exact cache directory size cap — defer until disk usage becomes a measured concern.
- Whether Chatterbox-Turbo emotion control / paralinguistic tags surface in MyVoice's UI — UX scope decision; out of architectural scope.
- Whether the persistent compile cache survives a `qwen-tts` pin-bump (it doesn't — D-24 cache key invalidates) — but the *user notification* of "first run after upgrade will be slow" is a UX call.
- Whether to expose `tts_engine_coexistence` as an explicit setting in the v1 UI or only via JSON config edit — UX call.

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Inherited from parent unchanged:** P-1 (state-bound method validity), P-2 (single transition helper), P-3 (Qt main-thread signal emission), P-4 (signals carry IDs not objects), P-5 (CodecTokenStreamer contract), P-6 (decoder worker contract), P-7 (cancellation propagation), P-8 (PlaybackQueue invariants), P-9 (telemetry log format). V2 baseline conventions (snake_case, `{property}_changed` signals, structured logging, dual-service audio) inherited unchanged from `architecture.md`.

**New for this scope:** P-10 through P-15 — six patterns that operationalize Cluster E (compile/CUDA Graph) and Cluster F (Lightning Tier).

**Critical conflict points addressed in this pass:**
- Cache-key construction across multiple call sites (Story 17.2 H1/H2 bug class)
- Silent fallthrough on capability-not-applied (FA2 silent no-op class)
- Invariant drift between decode-window and compile cache (D-25 assertion class)
- Engine adapter contract violations (Lightning Tier consistency)
- Optional download integrity gaps (supply-chain protection)
- Telemetry stream that doesn't distinguish engines (per-engine analysis blocked)

### P-10 — Compile cache key constructed in exactly one place

D-24 names 7 cache-key dimensions. Multiple AI agents writing different code paths would each construct the key independently, drift over time, and produce silent cache-key collisions (the Story 17.2 H1/H2 bug class — see `memory/code_review_regression_test_exact_class.md`).

**Rule:** the cache-key tuple is constructed by exactly one helper, `compile_cache.compute_key()`. All callers (warmup worker, generation path, settings-change invalidator, test fixtures) call this helper. No code outside `compile_cache.py` constructs cache keys directly.

**Anti-pattern:**

```python
# Multiple call sites, drift waiting to happen
key = f"{qwen_pin}/{model_id}/{precision}/{torch_ver}"  # somewhere in startup
key = hashlib.sha256(f"{qwen_pin}|{model_id}|{precision}|{torch_ver}".encode())  # somewhere else
```

**Pattern:**

```python
from myvoice.services.tts_streaming.compile_cache import compute_key

key = compute_key(
    qwen_tts_pin_hash=...,
    model_id=...,
    precision_str=...,
    torch_version=...,
    decode_window_frames=...,
    cuda_capability=...,
    compile_mode=...,
)
```

If a future story adds a cache-key dimension (D-24 enumerates 7 today), the dimension is added in `compute_key()` and **every cache key is re-computed** on next run — no cache-key version drift.

**Test obligation:** `tests/unit/services/tts_streaming/test_compile_cache.py` includes a "key stability" test — given fixed inputs, `compute_key()` returns the same value across runs and processes. Plus a "key uniqueness" test — varying any single dimension changes the output.

### P-11 — Invariant assertions at startup, not silent fallthroughs

D-25 (decode-window invariant), D-32 (VRAM reclamation invariant), D-26 (FA2 verification in dev/CI mode), and the parent's D-12 (qwen-tts symbol availability) all share a pattern: **at startup or load-completion time, assert architectural invariants loudly; never silently proceed past a violation**.

**Rule:** any invariant whose violation could cause silent audio corruption, silent performance loss, or silent state divergence MUST be asserted at the earliest detection point with a raised `AssertionError` (production) or strong-warning telemetry (production-soft, dev-strict).

**Example assertions in this scope:**

```python
# D-25 decode-window invariant (hard-fail)
assert (streamer.chunk_size + streamer.lookahead) == compile_cache.decode_window_frames, (
    f"Decode-window drift: streamer expects {streamer.chunk_size + streamer.lookahead}, "
    f"compile cache expects {compile_cache.decode_window_frames}. "
    "CUDA Graph replay would produce wrong audio."
)

# D-32 VRAM reclamation check (warning telemetry)
free_before = torch.cuda.mem_get_info()[0]
del current_engine
torch.cuda.empty_cache()
free_after = torch.cuda.mem_get_info()[0]
freed = free_after - free_before
expected = current_engine_size_bytes * 0.9  # tolerate 10% slack
if freed < expected:
    metrics.record('vram_reclamation_short', freed, tags={
        'expected': expected, 'engine': current_engine.id,
    })
    log.warning(f"[EngineSelector] VRAM reclamation shorter than expected: freed {freed} vs expected {expected}")

# D-22 Branch A — qwen-tts API method availability (hard-fail)
assert callable(getattr(Qwen3TTSModel, 'enable_streaming_optimizations', None)), (
    "qwen-tts pin lacks enable_streaming_optimizations(); D-22 Branch B (pin-bump) is required."
)
```

**Anti-pattern:**

```python
# Silent fallthrough — config drift produces wrong audio with no error surfaced
if streamer.chunk_size + streamer.lookahead != compile_cache.decode_window_frames:
    log.warning("Window mismatch")  # nobody reads this; audio sounds wrong
    # ... continues anyway
```

**When to use telemetry-only soft-warning instead of hard-fail:** when the invariant violation is *correctness-preserving but performance-degrading* (e.g., FA2 silently fell back to eager — produces correct audio at lower throughput). Gate with environment variable for dev/CI hard-fail (D-26 pattern).

### P-12 — Runtime verification of requested capability

D-26 (FA2 verification) generalizes: **when MyVoice requests a capability, verify it was actually applied; do not assume the request succeeded silently**.

**Rule:** every capability request — FA2, torch.compile mode, CUDA stream allocation, GGUF backend selection, engine warmup completion — is followed by a probe that confirms the capability is active. The probe result is emitted as telemetry per P-9. Mismatch handling follows P-11.

**Example — FA2 verification (D-26 implementation):**

```python
# In ModelRegistry, after model.from_pretrained()
effective_attn = _probe_effective_attn_implementation(model)
# returns one of: "flash_attention_2", "sdpa", "eager", or "unknown"

metrics.record(
    'attn_implementation_active',
    effective_attn,
    tags={'requested': requested_attn, 'engine': 'qwen3'},
)

if requested_attn == 'flash_attention_2' and effective_attn != 'flash_attention_2':
    log.warning(
        f"[ModelRegistry] Requested flash_attention_2 but got {effective_attn}; "
        "FlashAttention-2 is not active."
    )
    if os.environ.get('MYVOICE_REQUIRE_FA2') == '1':
        raise AssertionError("MYVOICE_REQUIRE_FA2 enforced; FA2 not active.")
    # else: persist user-facing one-time settings banner (D-26 production policy)
    settings_banner.show_once('fa2_inactive')
```

**Anti-pattern:**

```python
# Silent assumption — bug class
model = AutoModel.from_pretrained(..., attn_implementation="flash_attention_2")
# Code proceeds assuming FA2 is active; it may not be.
# Story 18.1's 31% real-time talker is consistent with this assumption.
```

**Test obligation:** for each requested capability, `tests/unit/services/test_model_registry.py` includes an integration test that exercises both branches (capability applied vs not applied) using mocking and asserts the correct telemetry + raise behavior fires.

### P-13 — Optional model download lifecycle

D-34 introduces MyVoice's first runtime model download. The pattern below codifies the lifecycle so future engines (NeuTTS Air, OmniVoice if either ever adopted) follow the same shape.

**Rule:** every optional model download follows this lifecycle:

1. **Bundle-time hash constant.** SHA-256 of the canonical model bundle is hard-coded in app source as a constant. **The hash is never downloaded** — supply-chain protection.
2. **Download to staging path.** Bytes go to `%LOCALAPPDATA%/MyVoice/lightning_tier_models/staging/<engine_id>.bin.partial` first; not to the load path.
3. **Resume support.** HTTP Range requests; partial file detected on resume; download continues from offset.
4. **Hash verification on completion.** Compute SHA-256 of `<engine_id>.bin.partial`; compare to bundle-time constant. Mismatch → delete staging file, surface error to user, do not load.
5. **Atomic move to load path.** On hash success, rename `<engine_id>.bin.partial` → `<engine_id>.bin`. The atomic move is the "download complete" signal.
6. **Threading.** Download runs in dedicated `QThread` (V2 baseline pattern); progress reported via `download_progress_changed` signal (V2 signal naming).
7. **Cancellation.** User-cancellable. Cancellation deletes staging file (or preserves it for resume — UX-dependent; default delete in v1).
8. **Failure modes.**
   - Network failure → "Download interrupted. Retry?" UI prompt.
   - Hash mismatch → "Download corrupted. Retrying with fresh download." (auto-retry once).
   - Disk full → "Insufficient disk space. Need ~1.5 GB; ~X GB available."
   - Permission denied → "Cannot write to model directory. Check folder permissions."
9. **Telemetry.** Each phase emits `download_lifecycle_event` per P-9 with `phase`, `bytes`, `engine_id`, `duration_ms`.

**Anti-pattern:**

```python
# Single-shot, no integrity check, no resume, no atomic write
urllib.request.urlretrieve(MODEL_URL, model_path)  # hangs forever on flaky networks; no hash check
```

**Test obligation:** `tests/integration/test_optional_download.py` exercises:
- Happy path (download → verify → atomic move)
- Resume after partial download
- Hash mismatch (corrupted bytes)
- Cancellation mid-download
- Insufficient disk space
- Permission denied

### P-14 — Engine adapter contract

D-30 defines `BaseTTSEngine` as the boundary between the engine selector and engine implementations. **The contract must be sharp** — agents implementing a new engine should fail any contract method definition rather than silently produce a misshapen engine.

**Rule:** every TTS engine implementation MUST:

1. Subclass `BaseTTSEngine` (`services/tts_engines/base_tts_engine.py`).
2. Implement all abstract methods of the contract — no engine ships with `pass` or `raise NotImplementedError`.
3. Pass `tests/integration/test_tts_engine_contract.py` — a contract-conformance suite that runs against every registered engine.
4. Emit telemetry with the engine's identifier per P-15.

**`BaseTTSEngine` contract sketch (final shape lands in Phase L1):**

```python
class BaseTTSEngine(ABC):
    """Contract for any TTS engine pluggable into MyVoice.

    Implementers MUST satisfy all abstract methods; tests/integration/
    test_tts_engine_contract.py verifies contract conformance.
    """

    engine_id: ClassVar[str]                    # e.g., "qwen3", "chatterbox-turbo"
    requires_cuda: ClassVar[bool]               # True for CUDA-only engines
    voice_clone_min_seconds: ClassVar[float]    # ICL threshold (Qwen3=3.0, Chatterbox=5.0)

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: VoiceProfile,
        session_id: str,
        settings: AppSettings,
    ) -> AsyncIterator[bytes]:
        """Yield PCM int16 chunks @ 24 kHz mono.
        Posts state transitions via SessionRegistry per P-3.
        Honors cancellation via session.cancel() per P-7.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Pre-prime engine state. Called from background warmup worker per D-23."""

    @abstractmethod
    def encode_voice_reference(self, voice: VoiceProfile) -> EngineEncodingArtifact:
        """Compute per-engine encoding artifact per D-31. Cached on disk."""

    @abstractmethod
    def get_memory_footprint_bytes(self) -> int:
        """Reported in-VRAM size when loaded; used by D-32 coexistence policy."""
```

**Anti-pattern:**

```python
class HalfBakedEngine(BaseTTSEngine):
    engine_id = "halfbaked"

    def generate(self, *args, **kwargs):
        # missing async, missing return type, ignores session_id
        return self._old_synchronous_api(text)
```

**Pattern:**

```python
class Qwen3TTSEngine(BaseTTSEngine):
    engine_id = "qwen3"
    requires_cuda = False  # CPU SENTENCE_STREAM mode supported per D-9
    voice_clone_min_seconds = 3.0

    async def generate(self, text, voice, session_id, settings):
        async for pcm_chunk in self._stream_dispatch(text, voice, session_id, settings):
            yield pcm_chunk

    def warmup(self):
        self._compile_warmup_priming()

    # ... etc, all abstract methods implemented
```

### P-15 — Per-engine telemetry tagging

D-37 mandates every metric record carries an `engine` tag. The pattern below ensures backward compat during migration and forward consistency afterward.

**Rule:** every call to `metrics.record()` MUST pass `engine` as a keyword argument. The helper signature is updated:

```python
def record(
    name: str,
    value: Any,
    *,
    engine: str = 'qwen3',          # NEW — defaults to qwen3 for backward compat
    session_id: str | None = None,
    tags: dict | None = None,
) -> None:
    ...
```

- During the migration (Phases L1+), the default `engine='qwen3'` means existing call sites continue to work without modification.
- Phase L1's PR also includes a static-analysis check (a custom flake8 rule or grep-CI step) that fails the build if a `metrics.record()` call site is found *without* `engine=` as a kwarg in any newly-added code under `services/tts_engines/`.
- After Phase L3 closes, a follow-up retrospective re-PR removes the default value (forces all call sites to be explicit). Tracked as a follow-up item, not blocking.

**Anti-pattern:**

```python
# Old call site without engine tag — ambiguous in mixed-engine retrospective analysis
metrics.record('first_chunk_latency_ms', 250, session_id=sid)
```

**Pattern:**

```python
metrics.record('first_chunk_latency_ms', 250, engine='qwen3', session_id=sid)
metrics.record('first_chunk_latency_ms', 75, engine='chatterbox-turbo', session_id=sid)
```

### Enforcement Guidelines

**All AI agents implementing this pass MUST:**

1. Use `compile_cache.compute_key()` for every cache-key lookup or invalidation (P-10).
2. Assert architectural invariants at startup; never proceed silently past a violation (P-11).
3. Probe for capability application after every capability request; emit telemetry; honor `MYVOICE_REQUIRE_*` env-var hard-fails in dev/CI (P-12).
4. Follow the optional-download lifecycle exactly — bundle-time hash, staging path, resume, hash verify, atomic move (P-13).
5. Subclass `BaseTTSEngine` and pass the contract test suite for any new engine (P-14).
6. Pass `engine=` as a kwarg to every `metrics.record()` call site in newly-added code (P-15).

**All AI agents implementing this pass MUST NOT:**

1. Construct compile cache keys outside `compile_cache.compute_key()`.
2. Add a `try: ... except: pass` around an invariant check (substitutes silent corruption for loud failure).
3. Assume a capability request succeeded without verifying it.
4. Skip hash verification on a downloaded model "just for testing" (test paths use mock downloads, not real downloads with verification disabled).
5. Implement an engine without subclassing `BaseTTSEngine` — even "temporary" or "experimental" engines.
6. Call `metrics.record()` without `engine=` in `services/tts_engines/`-rooted call sites.

### Anti-pattern Catalog

| Anti-pattern | Why it's wrong | Correct pattern |
|---|---|---|
| Constructing a cache key inline (`f"{a}/{b}/{c}"`) | Drift across call sites; Story 17.2 H1/H2 bug class | `compile_cache.compute_key(**kwargs)` |
| `log.warning(mismatch)` followed by silent continuation | Loses the user's audio quality; nobody reads warnings | `assert` at startup or telemetry + persistent banner |
| `model.from_pretrained(..., attn_implementation="flash_attention_2")` then assuming FA2 is active | Silent no-op produces the Story 18.1 bottleneck class | Probe `model.config._attn_implementation` after load; emit telemetry |
| `urllib.request.urlretrieve(URL, path)` | No hash, no resume, no atomic write, no retry | Use the optional-download manager service (P-13) |
| `class MyEngine: def generate(self, text): ...` (not subclassing BaseTTSEngine) | Bypasses contract test suite; engine-selector can't route to it | Subclass `BaseTTSEngine`; implement all abstract methods |
| `metrics.record('ttfa_ms', 250)` (no engine tag) | Per-engine retrospective analysis breaks (mixed-engine data) | `metrics.record('ttfa_ms', 250, engine='qwen3', session_id=sid)` |

### Pattern Examples

**Correct compile-cache lookup at warmup time:**

```python
# In services/tts_streaming/torch_runtime.py warmup worker
from myvoice.services.tts_streaming import compile_cache

key = compile_cache.compute_key(
    qwen_tts_pin_hash=qwen_tts_pin_hash(),
    model_id=settings.qwen3_model_id,
    precision_str=resolve_tts_precision(settings.tts_precision),
    torch_version=torch.__version__,
    decode_window_frames=streamer.chunk_size + streamer.lookahead,  # 30 today
    cuda_capability=torch.cuda.get_device_capability(),
    compile_mode="reduce-overhead",
)

if compile_cache.is_warm(key):
    metrics.record('compile_cache_hit', 1, engine='qwen3', tags={'key': key})
    model.enable_streaming_optimizations(
        decode_window_frames=streamer.chunk_size + streamer.lookahead,
        use_compile=True,
        compile_mode="reduce-overhead",
    )
else:
    # cold compile — show indicator, run priming generation
    metrics.record('compile_cache_miss', 1, engine='qwen3', tags={'key': key})
    indicator.show("Preparing TTS engine…")
    _prime_compile(model, streamer, key)
    indicator.hide()
```

**Correct engine fallback handling (D-36 / P-12):**

```python
# In services/tts_engine_selector.py
async def generate_with_engine(self, requested_engine_id, text, voice, session_id, settings):
    try:
        engine = self._load_or_get(requested_engine_id)
    except (ModelNotDownloadedError, EngineLoadError) as e:
        log.warning(f"[EngineSelector] {requested_engine_id} unavailable: {e}; falling back to qwen3")
        metrics.record(
            'engine_fallback',
            1,
            engine=requested_engine_id,                # the *requested* engine
            tags={'fallback_to': 'qwen3', 'reason': type(e).__name__},
        )
        self._notify_one_time_fallback(requested_engine_id, reason=str(e))
        engine = self._load_or_get('qwen3')
    async for pcm in engine.generate(text, voice, session_id, settings):
        yield pcm
```

## Project Structure & Boundaries

### Approach

Brownfield-deep. The complete V2 project structure is documented in `architecture.md` (sealed); the optimization-pass module additions (`services/sessions/`, `services/tts_streaming/`, `observability/`) are documented in `architecture-optimization-pass.md` (sealed). Neither is repeated here. This section covers only:

1. **New modules and files** added by this pass.
2. **Existing files modified** by this pass and the nature of the modification.
3. **Module boundaries** — what new modules may and may not import.
4. **Test additions** mapped to each new module.
5. **Migration map** showing which phase from D-38 introduces each file.

### New & Modified File Map

```
src/myvoice/
├── services/
│   ├── tts_engines/                              ← NEW (Phase L1)
│   │   ├── __init__.py
│   │   ├── base_tts_engine.py                    ← BaseTTSEngine ABC, EngineEncodingArtifact,
│   │   │                                            engine contract test fixtures (P-14)
│   │   ├── qwen3_tts_engine.py                   ← Refactor of qwen_tts_service.py;
│   │   │                                            implements BaseTTSEngine; Qwen3-specific
│   │   │                                            generation path delegated here
│   │   └── chatterbox_turbo_engine.py            ← NEW (Phase L2) — Chatterbox-Turbo
│   │                                                implementation of BaseTTSEngine
│   │
│   ├── tts_streaming/                            ← Existing (parent doc); extended
│   │   ├── __init__.py                           ← unchanged
│   │   ├── streaming_mode.py                     ← unchanged from parent
│   │   ├── torch_runtime.py                      ← MODIFIED (Phase 18.4) — adds
│   │   │                                            enable_streaming_optimizations() call after
│   │   │                                            model load; reads compile_cache for warmup
│   │   ├── compile_cache.py                      ← NEW (Phase 18.4) — compute_key() (P-10),
│   │   │                                            is_warm(), prime_warmup(), persistent
│   │   │                                            cache at %LOCALAPPDATA%/MyVoice/torch_compile_cache/
│   │   ├── codec_token_streamer.py               ← MODIFIED (Phase 18.6) — two-phase emission
│   │   │                                            scheduler (D-28); decode_window_frames=30
│   │   │                                            invariant assertion (D-25, P-11)
│   │   ├── streaming_decoder.py                  ← MODIFIED (Phase 18.7) — Hann crossfade
│   │   │                                            chunk-stitching post-decode (D-29)
│   │   └── pcm_chunk_streamer.py                 ← NEW (Phase L2) — thin pass-through wrapping
│   │                                                Chatterbox.generate_stream(); normalizes
│   │                                                format → 24kHz int16 mono; posts via
│   │                                                registry.append_chunk() (D-35)
│   │
│   ├── tts_engine_selector.py                    ← NEW (Phase L1) — engine selector (D-30);
│   │                                                NFR7 fallback (D-36); coexistence policy
│   │                                                wiring (D-32)
│   │
│   ├── tts_model_download_manager.py             ← NEW (Phase L3) — optional model download
│   │                                                lifecycle (D-34, P-13); QThread-based;
│   │                                                SHA-256 verification; resume support
│   │
│   ├── qwen_tts_service.py                       ← MODIFIED (Phase L1) — slimmed to a thin
│   │                                                delegation layer that constructs Qwen3TTSEngine
│   │                                                and routes through BaseTTSEngine contract;
│   │                                                wire-compatible during transition
│   │
│   ├── model_registry.py                         ← MODIFIED (Phase 18.5) — FA2 verification
│   │                                                probe + telemetry tag (D-26, P-12);
│   │                                                MYVOICE_REQUIRE_FA2 env-var hard-fail
│   │
│   └── ...                                       (unchanged: voice_profile_service, audio_*,
│                                                   whisper_*, transcription_*, quick_speak_*,
│                                                   sessions/, etc.)
│
├── models/
│   ├── app_settings.py                           ← MODIFIED (Phases L1, L3) — adds:
│   │                                              - tts_engine: Literal["qwen3", "chatterbox-turbo"]
│   │                                                = "qwen3"  (default unchanged for existing users)
│   │                                              - tts_engine_coexistence: Literal["auto",
│   │                                                "always-warm", "unload-on-switch"] = "auto"
│   ├── voice_profile.py                          ← MODIFIED (Phase L2) — adds:
│   │                                              - engine_encodings: Dict[str, EngineEncodingArtifact] = {}
│   │                                                (cached per-engine encodings, lazy-populated;
│   │                                                disk cache at voices/<id>/engine_encodings/<engine>.bin)
│   └── ...                                       (unchanged)
│
├── ui/
│   ├── components/
│   │   ├── service_status_indicator.py           ← MODIFIED (Phases 18.4, L3) —
│   │   │                                            18.4: "Preparing TTS engine…" cold-compile
│   │   │                                                indicator (D-23; mirrors Story 17.2 pattern)
│   │   │                                            L3: engine identifier in status string
│   │   ├── engine_selector_widget.py             ← NEW (Phase L3) — main-window engine switcher;
│   │   │                                            reads current_engine from AppSettings;
│   │   │                                            triggers JIT download prompt if Lightning
│   │   │                                            selected without prior download (D-34)
│   │   └── ...                                   (unchanged)
│   │
│   └── dialogs/
│       ├── settings/
│       │   └── tts_engines_settings_panel.py     ← NEW (Phase L3) — Settings tab section;
│       │                                            displays installed engines + Lightning Tier
│       │                                            download status; download progress bar;
│       │                                            tts_engine_coexistence dropdown
│       └── lightning_tier_download_dialog.py     ← NEW (Phase L3) — JIT download prompt
│                                                    surfaced from main UI when user selects
│                                                    Lightning without prior download
│
└── ...

tests/
├── unit/
│   ├── services/
│   │   ├── tts_engines/                          ← NEW (Phase L1)
│   │   │   ├── __init__.py
│   │   │   ├── test_base_tts_engine.py           ← contract definitions, ABC enforcement
│   │   │   ├── test_qwen3_tts_engine.py          ← Qwen3 engine refactor regression coverage
│   │   │   └── test_chatterbox_turbo_engine.py   ← Phase L2 — Chatterbox engine unit tests
│   │   │
│   │   ├── tts_streaming/                        ← Existing (parent); extended
│   │   │   ├── test_compile_cache.py             ← NEW (Phase 18.4) — P-10 key stability +
│   │   │   │                                        uniqueness; cache warmup/lookup; D-25
│   │   │   │                                        invariant assertion regression test
│   │   │   ├── test_codec_token_streamer.py      ← MODIFIED (Phase 18.6) — two-phase scheduler
│   │   │   │                                        boundary tests; emit_every_frames=2 → 5
│   │   │   │                                        transition test
│   │   │   ├── test_streaming_decoder.py         ← MODIFIED (Phase 18.7) — Hann crossfade
│   │   │   │                                        DSP regression test; chunk-boundary
│   │   │   │                                        click/pop detector (peak-derivative threshold)
│   │   │   └── test_pcm_chunk_streamer.py        ← NEW (Phase L2) — Chatterbox PCM
│   │   │                                            normalization test; 24kHz int16 contract
│   │   │
│   │   ├── test_tts_engine_selector.py           ← NEW (Phase L1) — engine routing;
│   │   │                                            NFR7 fallback (D-36); coexistence policy
│   │   ├── test_tts_model_download_manager.py    ← NEW (Phase L3) — lifecycle phases (P-13);
│   │   │                                            mock HTTP responses; hash mismatch
│   │   ├── test_model_registry.py                ← MODIFIED (Phase 18.5) — FA2 probe both
│   │   │                                            branches (applied + silently no-op)
│   │   └── ...                                   (unchanged)
│   │
│   └── models/
│       ├── test_app_settings_tts_engine.py       ← NEW (Phases L1, L3) — tts_engine setting
│       │                                            validation; tts_engine_coexistence enum
│       └── test_voice_profile_engine_encodings.py ← NEW (Phase L2) — engine_encodings field;
│                                                    serialization round-trip
│
├── integration/
│   ├── test_tts_engine_contract.py               ← NEW (Phase L1) — P-14 contract conformance
│   │                                                suite; runs against every registered engine;
│   │                                                must pass for any new engine to ship
│   ├── test_optional_download.py                 ← NEW (Phase L3) — P-13 happy path + 5 failure
│   │                                                modes (network failure, hash mismatch, cancel,
│   │                                                disk full, permission denied)
│   ├── test_compile_warmup_lifecycle.py          ← NEW (Phase 18.4) — D-23 background warmup
│   │                                                with mock indicator wiring; cold compile
│   │                                                completes within budget (target <30s)
│   └── test_engine_fallback_lifecycle.py         ← NEW (Phase L3) — D-36 fallback behavior;
│                                                    NFR7 chain composition (engine fallback +
│                                                    streaming-mode fallback)
│
├── test_qwen_tts_internals.py                    ← MODIFIED (Phase 18.4) — add
│                                                    Qwen3TTSModel.enable_streaming_optimizations
│                                                    symbol availability assertion (D-22 Branch A)
└── test_chatterbox_streaming_internals.py        ← NEW (Phase L2) — D-33 import-attribute test
                                                     for ChatterboxTTS.generate_stream signature

requirements.txt                                  ← MODIFIED:
                                                     - Phase 18.5 adds: flash-attn>=2.7.0,<3.0.0
                                                     - Phase 18.7 verifies: scipy>=1.10 (Hann window)
                                                     - Phase L2 adds: chatterbox-tts==0.1.7,
                                                                      chatterbox-streaming pinned
                                                                      to commit hash (D-33)
```

### Module Boundaries (import rules)

```
tts_engines/
├─ base_tts_engine.py          may import: stdlib, abc, typing, dataclasses
│                              may NOT import: PyQt6, services.*, tts_streaming.*, ui.*
│                              (the contract definition has no implementation dependencies)
│
├─ qwen3_tts_engine.py         may import: tts_engines.base_tts_engine,
│                                          tts_streaming.codec_token_streamer,
│                                          tts_streaming.streaming_decoder,
│                                          tts_streaming.compile_cache,
│                                          tts_streaming.torch_runtime,
│                                          sessions.session_registry,
│                                          observability.metrics,
│                                          qwen_tts internals (per D-12 pattern)
│                              may NOT import: tts_engines.chatterbox_turbo_engine
│                                              (peer engines do not import each other)
│
└─ chatterbox_turbo_engine.py  may import: tts_engines.base_tts_engine,
                                           tts_streaming.pcm_chunk_streamer,
                                           sessions.session_registry,
                                           observability.metrics,
                                           chatterbox.tts (Resemble),
                                           chatterbox-streaming wrapper (per D-33)
                               may NOT import: tts_streaming.codec_token_streamer
                                               (Qwen3-specific; Chatterbox produces PCM directly,
                                                bypasses code-token decoding)
                                               tts_engines.qwen3_tts_engine
                                               (peer engines do not import each other)

tts_engine_selector.py         may import: tts_engines.* (router needs to instantiate engines),
                                           tts_model_download_manager (status check before load),
                                           sessions.session_registry,
                                           observability.metrics,
                                           models.app_settings (read tts_engine + coexistence)
                               may NOT import: ui.*
                                               (UI talks to selector through service-locator)

tts_streaming/
├─ codec_token_streamer.py     EXISTING — no boundary change beyond two-phase scheduler logic
├─ streaming_decoder.py        EXISTING — no boundary change beyond crossfade post-processing
├─ compile_cache.py            may import: hashlib, pathlib, torch (cuda capability probe), stdlib
│                              may NOT import: services.*, tts_engines.*, ui.*
│                              (cache is a primitive; everything imports it; it imports nothing)
└─ pcm_chunk_streamer.py       may import: numpy, scipy.signal (resampling if needed),
                                           threading, observability.metrics
                               may NOT import: tts_engines.* (it's a primitive consumed by Chatterbox engine)

tts_model_download_manager.py  may import: PyQt6 (QThread + signals), urllib, hashlib,
                                           pathlib, observability.metrics
                               may NOT import: tts_engines.*, ui.*
                                               (download manager is consumed by selector and UI;
                                                it does not depend on either)

ui/
└─ engine_selector_widget.py + tts_engines_settings_panel.py + lightning_tier_download_dialog.py
                               may import: tts_engine_selector (read current state),
                                           tts_model_download_manager (subscribe to progress),
                                           models.app_settings (write tts_engine choice)
                               may NOT import: tts_engines.* directly
                                               (UI does not instantiate engines; only selector does)
```

**Forbidden import directions (additive to parent's rules):**

1. **`tts_streaming/*` does not import `tts_engines/*`.** Streaming primitives (codec streamer, PCM streamer, decoder, compile cache) are consumed by engines, not the other way around. This preserves unit-testability of the streaming primitives without an engine stack.
2. **Peer engines do not import each other.** `qwen3_tts_engine.py` does not import `chatterbox_turbo_engine.py`, and vice versa. Engine isolation is enforced by the import-rules table.
3. **UI does not import engines directly.** UI subscribes to selector + download manager; selector instantiates engines. Single seam for engine lifecycle.

### Requirements → Structure Mapping

| Requirement | Implementation home |
|---|---|
| **OFR-E** Producer acceleration via `enable_streaming_optimizations()` | `tts_streaming/torch_runtime.py` (model-load wiring) + `tts_streaming/compile_cache.py` (key + warmup) |
| **OFR-F** FA2 runtime verification | `services/model_registry.py` (probe + telemetry + dev-gate) |
| **OFR-G** Two-phase emission scheduler | `tts_streaming/codec_token_streamer.py` (Phase 1/Phase 2 logic) |
| **OFR-H** Hann crossfade chunk-stitching | `tts_streaming/streaming_decoder.py` (post-decode DSP) |
| **OFR-I** Engine selector (per-utterance choice + persisted default) | `services/tts_engine_selector.py` + `models/app_settings.py` (`tts_engine` field) |
| **OFR-J** Chatterbox-Turbo native voice clone + per-engine encoding cache | `services/tts_engines/chatterbox_turbo_engine.py` + `models/voice_profile.py` (`engine_encodings` field) |
| **OFR-K** Optional Lightning Tier download | `services/tts_model_download_manager.py` + `ui/dialogs/lightning_tier_download_dialog.py` + `ui/dialogs/settings/tts_engines_settings_panel.py` |
| **OFR-L** Engine-level NFR7 fallback | `services/tts_engine_selector.py` (try/except wrapper around engine load + generate; emits `engine_fallback_occurred`) |

### Cross-Cutting Concerns Mapping (where each pattern lives)

- **P-10 single-helper compile cache key:** centralized in `tts_streaming/compile_cache.compute_key()`. Every cache lookup and invalidation goes through this single function.
- **P-11 invariant assertions at startup:** primary call sites are `tts_streaming/torch_runtime.py` (decode-window invariant per D-25), `services/tts_engine_selector.py` (VRAM reclamation telemetry warning per D-32), `services/model_registry.py` (D-22 Branch A assertion).
- **P-12 runtime capability verification:** primary call sites are `services/model_registry.py` (FA2 probe per D-26), `tts_streaming/torch_runtime.py` (compile-applied probe), `services/tts_engine_selector.py` (engine-availability probe).
- **P-13 optional download lifecycle:** centralized in `services/tts_model_download_manager.py`. UI subscribes to its signals; everywhere else just calls `is_downloaded(engine_id)`.
- **P-14 engine adapter contract:** enforced in `services/tts_engines/base_tts_engine.py` (ABC) + `tests/integration/test_tts_engine_contract.py` (conformance suite).
- **P-15 per-engine telemetry tagging:** updated `observability/metrics.record()` signature carries `engine` kwarg; static-analysis check fails build if call sites under `services/tts_engines/` lack the kwarg.

### Integration Boundaries

#### Internal communication (within this scope)

```
                                         User UI Action
                                              │
                                              ▼
                                  ┌──────────────────────┐
                                  │ engine_selector_     │
                                  │ widget.py            │
                                  └──────────┬───────────┘
                                              │ reads/writes app_settings.tts_engine
                                              ▼
              ┌────────────────────────────────────────────────────────────┐
              │              tts_engine_selector.py                        │
              │   - chooses engine per AppSettings.tts_engine             │
              │   - applies coexistence policy (D-32)                     │
              │   - NFR7 fallback wrapper (D-36)                          │
              │   - emits current_engine_changed, engine_fallback_occurred│
              └─────┬─────────────────────────┬───────────────────────────┘
                    │                          │
            instantiates                instantiates
                    ▼                          ▼
          ┌────────────────────┐    ┌──────────────────────────┐
          │ qwen3_tts_engine   │    │ chatterbox_turbo_engine   │
          │ (BaseTTSEngine)    │    │ (BaseTTSEngine)           │
          └────────┬───────────┘    └────────────┬──────────────┘
                   │ uses                          │ uses
                   ▼                                ▼
        ┌─────────────────────┐         ┌──────────────────────┐
        │ codec_token_        │         │ pcm_chunk_streamer    │
        │ streamer +          │         │ (PCM pass-through)    │
        │ streaming_decoder + │         └──────────┬────────────┘
        │ compile_cache       │                    │
        └────────┬────────────┘                    │
                 │                                  │
                 │   both yield PCM chunks          │
                 ▼                                  ▼
              ┌─────────────────────────────────────────┐
              │       SessionRegistry (parent D-1)       │
              │   append_chunk(session_id, pcm_bytes)    │
              └──────────────────┬──────────────────────┘
                                  │
                                  ▼
                         AudioCoordinator (parent V2)
                                  │
                  ┌────────────────┴───────────────┐
                  ▼                                ▼
          MonitorAudioService              VirtualMicrophoneService
```

#### External integration (existing V2 plumbing)

- **`SessionRegistry` (parent D-1)** is the convergence point for both engines. Engine identity is metadata on the session, not lifecycle.
- **`AudioCoordinator` + dual-service audio** unchanged from V2 baseline. Engines emit PCM chunks through the registry callback; coordinator dispatches to monitor + virtual mic in parallel (parent P-8).
- **`qwen_tts` library** continues to be reached via private symbols per D-12; new symbol added to import-attribute test for D-22 Branch A.
- **`chatterbox-tts` + `chatterbox-streaming`** new external deps; private-symbol pattern mirrored via D-33.
- **HuggingFace download endpoint** for Chatterbox-Turbo model bundle — accessed via `tts_model_download_manager.py`; URL + expected SHA-256 hash bundled as constants.

### Migration Map (per D-38, augmented with file references)

| Phase | Story | New files | Modified files | Test additions |
|---|---|---|---|---|
| **18.4** | Story 18.4 (refined) | `tts_streaming/compile_cache.py` | `tts_streaming/torch_runtime.py`, `ui/components/service_status_indicator.py`, `tests/test_qwen_tts_internals.py` | `tests/unit/services/tts_streaming/test_compile_cache.py`, `tests/integration/test_compile_warmup_lifecycle.py` |
| **18.5** | Story 18.5 (NEW) | — | `services/model_registry.py`, `requirements.txt` (flash-attn pin) | `tests/unit/services/test_model_registry.py` (FA2 probe coverage) |
| **18.6** | Story 18.6 (NEW) | — | `tts_streaming/codec_token_streamer.py` | `tests/unit/services/tts_streaming/test_codec_token_streamer.py` |
| **18.7** | Story 18.7 (NEW) | — | `tts_streaming/streaming_decoder.py`, `requirements.txt` (verify scipy) | `tests/unit/services/tts_streaming/test_streaming_decoder.py` (Hann + click/pop detector) |
| **(gate)** | — | (Story 18.1 metrics re-run) | — | — |
| **L1** | Story 19.1 (NEW) | `tts_engines/__init__.py`, `tts_engines/base_tts_engine.py`, `tts_engines/qwen3_tts_engine.py`, `tts_engine_selector.py` | `services/qwen_tts_service.py` (delegation refactor), `models/app_settings.py` (tts_engine field), `observability/metrics.py` (engine kwarg) | `tests/unit/services/tts_engines/test_base_tts_engine.py`, `tests/unit/services/tts_engines/test_qwen3_tts_engine.py`, `tests/unit/services/test_tts_engine_selector.py`, `tests/integration/test_tts_engine_contract.py` |
| **L2** | Story 19.2 (NEW) | `tts_engines/chatterbox_turbo_engine.py`, `tts_streaming/pcm_chunk_streamer.py`, `tests/test_chatterbox_streaming_internals.py` | `models/voice_profile.py` (engine_encodings), `requirements.txt` (chatterbox pins) | `tests/unit/services/tts_engines/test_chatterbox_turbo_engine.py`, `tests/unit/services/tts_streaming/test_pcm_chunk_streamer.py`, `tests/unit/models/test_voice_profile_engine_encodings.py` |
| **L3** | Story 19.3 (NEW) | `services/tts_model_download_manager.py`, `ui/components/engine_selector_widget.py`, `ui/dialogs/settings/tts_engines_settings_panel.py`, `ui/dialogs/lightning_tier_download_dialog.py` | `models/app_settings.py` (tts_engine_coexistence), `ui/components/service_status_indicator.py` (engine identifier in status) | `tests/unit/services/test_tts_model_download_manager.py`, `tests/integration/test_optional_download.py`, `tests/integration/test_engine_fallback_lifecycle.py` |

### File Organization Patterns

- **New module namespaces follow existing convention.** `services/tts_engines/` mirrors `services/sessions/` and `services/tts_streaming/` (parent's added namespaces). `services/tts_engine_selector.py` is a top-level service file (not a namespace) following the existing `services/audio_coordinator.py`, `services/voice_profile_service.py` precedent.
- **Tests mirror source layout** under `tests/unit/services/` and `tests/integration/`. Matches V2 baseline + parent's pattern.
- **No module-level circular dependencies** allowed. The import-rule table above is the enforcement.
- **No top-level new directories outside `src/myvoice/`.** Storage paths under `%LOCALAPPDATA%/MyVoice/` are added (`torch_compile_cache/`, `lightning_tier_models/`, per-voice `engine_encodings/`) but the source tree stays flat-ish.

### Development Workflow Integration

- **Existing dev launch flow unchanged.** `python -m myvoice` continues to work; new services initialize in the existing service-locator at startup.
- **Build process unchanged for Phases 18.4–18.7.** Phase L2 adds two new pip dependencies; no changes to PyInstaller spec required (the new deps are pure-Python wrappers around existing model frameworks). Phase L3 ships the optional-download plumbing; the bundled installer remains installer-size-stable.
- **CI additions per phase:**
  - 18.4: compile-cache regression suite + warmup-lifecycle integration test.
  - 18.5: FA2 runtime probe test; CI sets `MYVOICE_REQUIRE_FA2=1` to catch silent regressions.
  - 18.6/18.7: codec-streamer two-phase boundary test; click/pop detector regression.
  - L1: contract conformance suite (`test_tts_engine_contract.py`) — must pass for *every* registered engine.
  - L2: chatterbox-streaming import-attribute test.
  - L3: optional-download lifecycle test (mock HTTP); engine-fallback lifecycle test.
  - **Static-analysis check (P-15):** custom rule fails build if a `metrics.record()` call site under `services/tts_engines/` lacks `engine=` kwarg.

### Out of Structural Scope

- **Repackaging of V2 modules** (`whisper_*`, `transcription_*`, `voice_profile_*`, etc.). Untouched.
- **Repackaging of optimization-pass modules** (`services/sessions/`, `observability/`). Untouched — those are sealed primitives this scope consumes.
- **New top-level directories outside `src/myvoice/`.** None proposed.
- **Changes to PyInstaller spec or installer NSIS script.** None required for this scope; the optional Lightning Tier download is a runtime mechanism that doesn't change build/install tooling.
- **Changes to `requirements-production.txt`** beyond mirroring the new pins from `requirements.txt`.

## Architecture Validation Results

### Coherence Validation ✅

**Decision compatibility:** All 18 new decisions (D-21 through D-38) compose without contradicting each other or the inherited 20 (parent) + V2 baseline. Three compositions warrant explicit explanation:

1. **D-21 (fixed window=30) constrains D-28 (two-phase scheduler).** The two-phase scheduler can only vary `emit_every_frames` (chunk emission cadence), not `decode_window_frames` (compile graph shape). This is intentional — the fixed-window choice is the architectural ground truth; the two-phase scheduler operates within it. **Trade quantified:** rekuenkdr's both-knobs design reports 2.75× TTFA improvement; the emit-cadence-only design reports ~1.5× TTFA improvement. Acceptable trade for graph-cache validity. Documented at D-28.

2. **D-22 has two branches (verify-or-bump); only Branch A is non-disrupting.** If qwen-tts at commit `1ab0dd75` (= 0.0.4) does NOT ship `enable_streaming_optimizations()`, Branch B requires a pin-bump that **triggers a full Story-17.1-grade NFR3 perceptual re-audition** (parent's NFR3 row policy at line 803). The architecture frames both branches; the empirical answer is verified at implementation time, not architecture time. **This is the single highest-risk gate in the scope.**

3. **D-32 (memory coexistence) composes with NFR11 (<4 GB RAM with model) per hardware tier.** RTX 3060 8GB users default to `unload-on-switch` (single model resident at any time, NFR11-compliant). RTX 4090 24GB users default to `always-warm` (5 GB combined < 70% × 24 GB threshold). The auto-detection mechanism in D-32 picks the right policy per host without user configuration; explicit override (`AppSettings.tts_engine_coexistence`) escapes the default for edge cases.

**Pattern consistency:** P-10 through P-15 align with D-21 through D-38 without override:
- P-10 (single-helper cache key) directly implements D-24 (7-dim key).
- P-11 (invariant assertions) operationalizes D-25 (decode-window), D-32 (VRAM reclamation), D-22 Branch A (API symbol availability).
- P-12 (capability verification) operationalizes D-22, D-23, D-26, D-32 — every "request a capability, then verify it took effect" call site.
- P-13 (download lifecycle) operationalizes D-34.
- P-14 (engine adapter contract) operationalizes D-30.
- P-15 (per-engine telemetry) operationalizes D-37, extends parent D-19 / P-9.

The single-chokepoint patterns from parent (`_transition_to`, `post_mutation`, `metrics.record`) are augmented by P-10's `compile_cache.compute_key()` — same architectural principle (single source of truth) extended to the new domain.

**Structure alignment:** Module boundaries support every pattern. The forbidden import directions (added in this scope) — `tts_streaming/*` ↛ `tts_engines/*`, peer engines ↛ each other, UI ↛ engines directly — preserve unit-testability and engine isolation. The new `services/tts_engines/` namespace mirrors parent's `services/sessions/` and `services/tts_streaming/` precedents.

### Requirements Coverage Validation ✅

**Inherited Functional Requirements (V2 PRD):**

| FR | Coverage |
|---|---|
| FR1–FR5 (TTS Generation) | ✅ Both engines satisfy via BaseTTSEngine contract; producer-bottleneck closure (Stories 18.4–18.7) tightens FR2 streaming latency margin |
| FR4 (User can cancel) | ✅ Parent P-7 cancellation chain unchanged; new engines inherit via P-14 contract |
| FR28–FR32 (Playback Last) | ✅ Saveable slot semantics unchanged (parent D-3, D-4); engine-agnostic at the SessionRegistry layer |
| FR42 (Status indicator) | ✅ Extended via OFR-D wiring + Phase 18.4 "Preparing TTS engine…" + Phase L3 engine identifier in status string |

**Inherited Non-Functional Requirements:**

| NFR | Coverage |
|---|---|
| NFR1 (renegotiated per-class targets, Story 16.9) | ✅ This scope adds *headroom*. Story 18.4 closes producer ratio < 1.0× sustained → existing per-class targets get easier; Lightning Tier introduces new ≤500 ms TTFA target on RTX 3060 |
| NFR3 (no audio stuttering — Story 17.1 PASS) | ✅ D-21's window=30 preserves Story 17.1's verdict; D-29 Hann crossfade prevents new artifacts at chunk boundaries; click/pop detector regression test added per OFR-H |
| NFR4 (UI <200 ms) | ✅ D-23 background warmup keeps cold compile off the UI thread; P-11 invariant checks at startup are bounded to startup window, not runtime |
| NFR6 (no crashes) | ✅ D-22 import-attribute test; D-33 chatterbox-streaming pin policy; P-7 cancellation propagation extends naturally to new engines |
| NFR7 (graceful degradation) | ✅ D-36 engine-level fallback (Lightning ← Quality) layered atop parent's TRUE_STREAM ← SENTENCE_STREAM ← BATCH chain |
| NFR11 (<4 GB RAM with model) | ✅ D-32 hybrid coexistence; RTX 3060 8GB → unload-on-switch (single model resident); explicit user override available |
| NFR12 (CPU-only support) | ✅ D-9 (parent) stands; CPU users skip compile path; Lightning is CUDA-only (silent fallback to Qwen3 SENTENCE_STREAM via D-36) |

**Local FR-equivalents (introduced by this scope):**

| OFR | Coverage |
|---|---|
| **OFR-E** Producer acceleration | ✅ D-21, D-22, D-23, D-24, D-25 + P-10, P-11 + `tts_streaming/compile_cache.py` + `tts_streaming/torch_runtime.py` (Story 18.4) |
| **OFR-F** FA2 runtime verification | ✅ D-26 + P-12 + `services/model_registry.py` (Story 18.5) |
| **OFR-G** Two-phase emission scheduler | ✅ D-28 + `tts_streaming/codec_token_streamer.py` (Story 18.6) |
| **OFR-H** Hann crossfade chunk-stitching | ✅ D-29 + `tts_streaming/streaming_decoder.py` (Story 18.7) |
| **OFR-I** Engine selector | ✅ D-30 + `services/tts_engine_selector.py` + `models/app_settings.py` (Phase L1, exposed in L3 UI) |
| **OFR-J** Native voice clone (Chatterbox-Turbo) | ✅ D-31 + `services/tts_engines/chatterbox_turbo_engine.py` + `models/voice_profile.py` engine_encodings (Phase L2) |
| **OFR-K** Optional Lightning Tier download | ✅ D-34 + P-13 + `services/tts_model_download_manager.py` + `ui/dialogs/lightning_tier_download_dialog.py` + Settings panel (Phase L3) |
| **OFR-L** Engine-level NFR7 fallback | ✅ D-36 + `services/tts_engine_selector.py` try/except wrapper (Phase L3 UI surface) |

### Implementation Readiness Validation — with surfaced gaps

Below are issues found during the validation that warrant explicit resolution before tech-spec.

**Critical (block tech-spec without resolution):**

None. All blocking decisions made in Step 4. The D-22 verify-or-bump branch IS architecturally critical, but the architecture frames both branches; the verification is an *implementation-time* gate, not an architecture-time blocker.

**Important (would cause AI-agent inconsistency or risk if left unresolved):**

1. **D-22 verification is empirical and not yet performed.** Whether `enable_streaming_optimizations()` exists at qwen-tts commit `1ab0dd75` is not verified at architecture time. Story 18.4's first implementation task should be the verification probe; if Branch B (pin-bump) fires, the story splits into "verify + plan" and "execute + re-audition". *Resolution:* Story 18.4's first AC is the verification step; the story splits if Branch B fires. **Not a hidden risk; named.**

2. **`scipy>=1.10` assumed already in tree (Step 3 said "verify in Step 6").** I did not actually grep `requirements.txt` to confirm scipy is present. Step 3's statement should be verified before Phase 18.7 ships. *Resolution:* Story 18.7's first AC is to verify `scipy.signal.windows.hann` is importable; if not, add scipy to requirements.txt. Low-risk verification, but should not be implicit.

3. **Chatterbox-Turbo TTFA on RTX 3060 is research-derived, not empirically MyVoice-measured.** The "75 ms latency, 6× real-time" claim from Resemble is on "modern GPU" — assumed to scale to RTX 3060 but not verified by MyVoice. Phase L1's first task should empirically measure TTFA on the project's reference RTX 3060 host before the Lightning Tier UX promises a specific latency to users. *Resolution:* Phase L1 includes a benchmark task; if RTX 3060 TTFA exceeds (e.g.) 200 ms, Phase L3's UX copy adjusts before shipping ("real-time" instead of specific ms numbers).

4. **Memory swap latency (`unload-on-switch` path) is research-derived.** D-32 says "2–5 s switch cost" — not empirically MyVoice-measured. Phase L2's first task should measure on RTX 3060 8GB; if swap exceeds ~5 s, the indicator UX (D-32 "Switching engine…" message) needs duration-aware framing. *Resolution:* Phase L2 includes a benchmark; UX copy adjusts.

5. **`pcm_chunk_streamer.py` format normalization is unspecified.** Step 6 says it "normalizes format → 24 kHz int16 mono" but doesn't specify what Chatterbox-Turbo actually emits by default. If Chatterbox emits 24 kHz mono natively, no transformation; if 16 kHz or 48 kHz, resampling required (scipy.signal.resample_poly is the natural choice). *Resolution:* Phase L2's first AC is to verify Chatterbox-Turbo's native output format; if mismatched, the streamer's resampling logic is specified in the per-feature tech-spec.

**Nice-to-have (deferable):**

1. **Compile cache directory size cap.** D-24 / Step 4 says "no size cap in v1; future enhancement if disk usage becomes a measured concern." The cache could grow unbounded across precision toggles, qwen-tts pin bumps, torch upgrades. Adding a 5 GB LRU cap would prevent surprise disk usage. **Defer; capture in retro.**

2. **`torch.cuda.mem_get_info()` API stability under PyTorch 2.10.** D-32 relies on this API; it's stable in 2.x but worth a CI-time smoke test that asserts the API returns sensible values on the dev host. **Defer; captured as an integration-test addition.**

3. **Engine telemetry retro-rename eligibility.** P-15 default `engine='qwen3'` is a backward-compat hack; after Phase L3 closes, a follow-up PR could remove the default, forcing all call sites to be explicit. **Tracked in P-15 text already; not blocking.**

### Gap Analysis Summary

| Severity | Count | Resolution status |
|---|---|---|
| Critical | 0 | — |
| Important | 5 | Resolved inline above; integrated into Story-AC obligations and per-feature tech-spec scope |
| Nice-to-have | 3 | Documented; not blocking |

### Architecture Completeness Checklist

**✅ Requirements analysis**
- [x] Project context thoroughly analyzed (Step 2)
- [x] Scale and complexity assessed (Medium for Epic 18 completion; Medium-high for Lightning Tier)
- [x] Technical constraints identified (V2 + optimization-pass inheritance + 4 new constraints in Step 2)
- [x] Cross-cutting concerns mapped (10 items in Step 2)

**✅ Architectural decisions**
- [x] Critical decisions documented (D-21 through D-38, with sub-options + chosen path + rationale)
- [x] Versions verified (Step 3 — chatterbox-tts 0.1.7, chatterbox-streaming 0.1.2 community, flash-attn 2.7.x range pin needed; qwen-tts pin verification deferred to D-22)
- [x] Integration patterns defined (D-35 streaming-decoder uniformity, D-30 engine boundary)
- [x] Performance considerations addressed (NFR1 headroom, NFR3 conditional re-audition path, NFR4 UI-thread protection, NFR11 hardware-aware coexistence)

**✅ Implementation patterns**
- [x] Patterns inherited from parent unchanged (P-1 through P-9)
- [x] New patterns defined (P-10 single-helper cache key, P-11 invariant assertions, P-12 capability verification, P-13 download lifecycle, P-14 engine adapter, P-15 per-engine telemetry)
- [x] Naming conventions inherited from V2 (no override)
- [x] Anti-pattern catalog complete with 6 pairs (anti-pattern → correct pattern)
- [x] Test obligations enumerated per pattern

**✅ Project structure**
- [x] New & modified file map complete (12 new source files, 12 new test files, 12 modified source files, 1 modified internals test)
- [x] Module boundaries with explicit import rules (3 new forbidden directions)
- [x] Requirements → structure mapping complete (OFR-E through OFR-L)
- [x] Cross-cutting concerns mapping (P-10 through P-15 each with primary + secondary call sites)
- [x] Test additions mapped to phases (per D-38)
- [x] Migration order matches D-38

**✅ Validation resolutions integrated**
- [x] D-22 verification empiricism named as Story 18.4 first AC
- [x] scipy verification named as Story 18.7 first AC
- [x] Chatterbox TTFA empirical measurement named as Phase L1 first task
- [x] Memory swap latency empirical measurement named as Phase L2 first task
- [x] PCM format normalization specification named as Phase L2 first AC

### Architecture Readiness Assessment

**Overall status:** READY FOR TECH-SPEC.

**Confidence level:**
- **High** for Epic 18 completion (Stories 18.4–18.7). The producer-bottleneck closure path is well-bounded, well-precedented (4 community Qwen3-TTS streaming forks demonstrate the technique), audit-grounded (4 of 5 most-leveraged moves already in place), and uses official upstream API. The single empirical risk (D-22 Branch B) is named and bounded.
- **Medium-High** for Lightning Tier (Epic 19, Phases L1–L3). First multi-engine introduction; new dependency surface (`chatterbox-tts`, `chatterbox-streaming`); new UX (engine selector, optional download). Pattern is well-established in the ecosystem (TTS-Audio-Suite, Voicebox precedents); the architectural decisions are conservative (Pattern B side-by-side native clone, both PyTorch, no runtime pivot). Empirical verification items in Phase L1/L2 first tasks bound the remaining unknowns.

**Key strengths:**

1. **Single-chokepoint discipline preserved and extended.** Parent's `_transition_to`, `post_mutation`, `metrics.record` are augmented by `compile_cache.compute_key()`; same architectural principle.
2. **Phased migration (D-38) — every phase ships independently, reverts cleanly.** Stories 18.4–18.7 each deliver user-visible value (latency reduction); Lightning Tier phases L1/L2 are no-user-visible-change until L3 ships UX.
3. **Backward-compatible at every boundary.** P-15 telemetry defaults to `'qwen3'`; D-14-style "existing signals stay wire-compatible" preserved (the new `current_engine_changed` is additive).
4. **Producer-bottleneck closure is upstream-blessed.** Using the official `qwen_tts.enable_streaming_optimizations()` API is not a community-fork hack; it's the path the upstream project has codified.
5. **Lightning Tier installer-size impact is near-zero.** Optional download keeps the Windows installer flat; users opt in post-install. Mitigates the documented installer-size pain (`memory/production_release_state.md`).
6. **Empirical verification items named, not hidden.** Five "important" gaps have explicit Story-AC homes; the architecture doesn't pretend they're already resolved.

**Areas for future enhancement:**

1. **Lightning Tier expansion to NeuTTS Air or OmniVoice.** Both are research-strong second-place candidates, deferred from this pass to keep the multi-engine introduction focused on a single engine. The `BaseTTSEngine` contract (D-30, P-14) is designed to accept additional engines without architectural change.
2. **Quantization (P2.A from research).** Once the producer ratio closes, Qwen3-TTS GPTQ-Int8 transferred from base Qwen3 weights becomes a memory-headroom enhancement worth investigating. Research-grade today.
3. **DirectML cross-vendor expansion (P2.C).** Gated on AMD/Intel-user demand; research P2.C path.
4. **Compile cache LRU eviction.** Nice-to-have if disk usage grows.
5. **CPU-tier Lightning path via NeuTTS Air's GGUF.** Would unlock "no GPU at all" Lightning tier; out of scope for this pass but architecturally enabled by the engine-selector abstraction.

### Implementation Handoff (preview — finalized in Step 8)

**AI-agent guidelines:**

- Start from this document. Read alongside the parent `architecture-optimization-pass.md` (sealed) and the grandparent `architecture.md` (sealed).
- Implementation order is **D-38 phased migration** (18.4 → 18.5 → 18.6 → 18.7 → gate → L1 → L2 → L3), not "all at once."
- Every cache-key construction flows through `compile_cache.compute_key()` (P-10). Every architectural invariant assertion is loud at startup (P-11). Every requested capability is probed for actual application (P-12). Every new engine subclasses `BaseTTSEngine` and passes the contract test suite (P-14). Every metric records `engine` kwarg (P-15).
- Five empirical verification items are first-AC obligations on their respective stories — they are not optional.

**First implementation priority:**

Story 18.4 — `enable_streaming_optimizations()` wiring + `compile_cache.py`. First AC: verify `enable_streaming_optimizations` exists at qwen-tts commit `1ab0dd75` (D-22 Branch A check). If Branch B fires, the story splits and a separate "pin-bump + re-audition" track engages.

## Implementation Handoff

### Companion Documents

This document is one of three layered architecture artifacts. Read in this order for full context:

1. **`architecture.md`** (V2 baseline, sealed 2026-01-31) — read first if onboarding fresh; provides the framework, language, audio backend, voice library, dual-service audio architecture, and naming conventions.
2. **`architecture-optimization-pass.md`** (sealed 2026-04-27) — read second; provides D-1 through D-20 (sessions, streaming, queue, save, clear-comms, indicator) and P-1 through P-9 (state-bound methods, signal contracts, cancellation). This is the *direct parent* of the current scope.
3. **`architecture-streaming-acceleration-and-lightning-tier.md`** (this document, 2026-05-10) — read last; provides D-21 through D-38 and P-10 through P-15 specific to producer-bottleneck closure (Stories 18.4–18.7) and Lightning Tier (Phases L1–L3).

Plus the source research that informed this architecture:
- **`research/technical-tts-streaming-fast-clone-research-2026-05-10.md`** — Mary's research with audit results, source citations, and the concrete recommendations this architecture refines into shippable decisions.

### AI-Agent Guidelines (consolidated)

For any AI agent implementing a story under this scope, the rules below are mandatory:

**MUST do:**

1. **Construct compile cache keys via `compile_cache.compute_key()` only** (P-10). Never inline-build cache keys.
2. **Assert architectural invariants loudly at startup** (P-11) — D-25 decode-window invariant, D-32 VRAM reclamation, D-22 Branch A symbol availability. Hard-fail in production for correctness invariants; telemetry + persistent banner for performance invariants.
3. **Probe for capability application after every capability request** (P-12). FA2, torch.compile, CUDA Graph, engine availability — verify the request took effect.
4. **Honor the optional-download lifecycle precisely** (P-13). Bundle-time hash, staging path, resume support, hash verify, atomic move. No shortcuts.
5. **Subclass `BaseTTSEngine` and pass the contract test suite** for any new engine (P-14). Even "experimental" or "test" engines.
6. **Pass `engine=` kwarg to every `metrics.record()` call site** in newly-added code (P-15). Static-analysis check fails CI if missed.
7. **Use `_transition_to`, `post_mutation`, `metrics.record`** (parent P-2, P-3, P-9) in every state mutation, cross-thread call, and metric emission.
8. **Update `tests/test_qwen_tts_internals.py`** when touching new private `qwen_tts` symbols (D-12 + D-22). Update `tests/test_chatterbox_streaming_internals.py` when touching `chatterbox-streaming` symbols (D-33).

**MUST NOT do:**

1. Construct compile cache keys outside `compile_cache.compute_key()`.
2. Add `try: ... except: pass` around invariant checks (substitutes silent corruption for loud failure).
3. Assume a capability request succeeded without verifying it.
4. Skip hash verification on a model download "for testing" — use mock downloads, not real downloads with verification disabled.
5. Implement a TTS engine without subclassing `BaseTTSEngine`.
6. Call `metrics.record()` without `engine=` in `services/tts_engines/`-rooted call sites.
7. Reach into private `qwen_tts` or `chatterbox-streaming` symbols without updating the corresponding internals test.
8. Import across forbidden directions (`tts_streaming/*` ↛ `tts_engines/*`, peer engines ↛ each other, UI ↛ engines directly).

### Recommended Story-Creation Order

```
Phase 18.4 — Story 18.4 (refines existing stub)
  ↓ Story 18.4 first AC: verify D-22 Branch A or fire Branch B
  ↓ If Branch B: pause Stories 18.5–18.7 until pin-bump + re-audition complete
Phase 18.5 — Story 18.5 (NEW, parallel-eligible with 18.6 / 18.7)
Phase 18.6 — Story 18.6 (NEW, parallel-eligible with 18.5 / 18.7)
Phase 18.7 — Story 18.7 (NEW, parallel-eligible with 18.5 / 18.6)
   ↓
   GATE: Story 18.1 metrics re-run; ratio < 1.0× sustained, RTF ≥ 1.0
   ↓
Phase L1 — Story 19.1 (NEW, Epic 19 introduction)
Phase L2 — Story 19.2 (NEW)
Phase L3 — Story 19.3 (NEW)
   ↓
   Lightning Tier user-visible release
```

**Parallel-eligibility:** Stories 18.5/18.6/18.7 each modify a different file (`model_registry.py`, `codec_token_streamer.py`, `streaming_decoder.py`) and are mechanically independent of each other. They can be authored / reviewed in any order or in parallel after Story 18.4 lands. The (gate) milestone integrates all four before Phase L1 starts.

### Story-Creation Inputs

For `/bmad-bmm-create-story` runs, supply this document as the architecture input. Each story template should reference:

- **D-IDs and P-IDs** the story implements (e.g., Story 18.4 implements D-21, D-22, D-23, D-24, D-25, P-10, P-11, P-12)
- **Files modified / created** per the migration map (D-38 + Step 6 file map table)
- **First-AC empirical verification items** where applicable (Stories 18.4, 18.7, 19.2 each have one)
- **Test obligations** per the Step 6 test additions table

### Open Operational Items (post-architecture)

These are administrative follow-ups, not blocking the architecture handoff:

- **PRD back-propagation of OFR-E through OFR-L.** Owner: PM/Commander. Tracked here so it doesn't go cold (mirrors parent's pattern of explicit OFR back-propagation tracking).
- **Memory-entry update.** Add a `production_release_state.md` entry capturing the post-Story-18.4 expected user experience (compile cold-start once, then warm cache; Lightning Tier opt-in download). Owner: Commander, post-shipment.
- **Quarterly task.** Per D-33: check whether Resemble has shipped an official `chatterbox-tts` streaming API; if yes, drop the community `chatterbox-streaming` dependency.
- **Story 18.3 deferred audition.** Per D-27, post-Story-18.4 retrospective: re-run the bf16 NFR1 measurement under compile-engaged. If bf16 finally outperforms fp32+TF32, the deferred Task 8 audition fires. If not, `tts_precision="auto"` recommendation flips to fp32-default on Ampere+.

### Workflow Status

**Workflow:** `create-architecture`
**Scope:** `streaming-acceleration-and-lightning-tier`
**Steps:** 1 through 8 complete
**Status:** **COMPLETE**
**Completion date:** 2026-05-10
**Parent architecture:** `architecture-optimization-pass.md` (sealed 2026-04-27)
**Grandparent architecture:** `architecture.md` (sealed 2026-01-31)
**Inherited governance:** D-1 through D-20 stand; P-1 through P-9 stand; Story 16.9 NFR1 per-class targets stand; Story 17.1 NFR3 PASS stands.
**New decisions added by this scope:** D-21 through D-38 (18 decisions across Cluster E, Cluster F, Cluster G).
**New patterns added by this scope:** P-10 through P-15.
**Migration map:** Phase 18.4 → 18.5 → 18.6 → 18.7 → (gate) → L1 → L2 → L3.
**Empirical verification items:** 5, named with concrete first-AC / first-task homes (Story 18.4, Story 18.7, Phase L1, Phase L2 [×2]).

---

_Architecture document complete. Ready for handoff to `/bmad-bmm-create-story` for per-story tech-spec authoring, or `/bmad-bmm-check-implementation-readiness` for cross-document alignment validation against the PRD and existing epics file._

