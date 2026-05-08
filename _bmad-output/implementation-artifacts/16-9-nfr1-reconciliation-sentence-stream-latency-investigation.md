# Story 16.9: NFR1 Reconciliation — SENTENCE_STREAM Latency Investigation

Status: done

> Phase ⊥ of D-20 — **ninth and final story of Epic 16** (True Streaming TTS, the parallel/independent track) and the **second of two follow-ups to Story 16.7's empirical-validation gate failure** (Story 16.8 was the first; both are independent and were sized to be worked in parallel). Story 16.7's empirical run on the maintainer's RTX 5090 + qwen-tts 0.0.4 host produced a two-failure verdict (`16-7-streaming-validation-report.md` §1, §3.2, §5): (1) TRUE_STREAM was structurally broken — closed by Story 16.8 (`16-8-true-stream-real-wire-up.md` Change Log #6, #7); (2) **SENTENCE_STREAM does not meet NFR1's 2-second first-audio ceiling on this host on either GPU or CPU.** Story 16.9 is the contract-level unblocker for the streaming-default flag flip — the architectural Phase ⊥ unblocker is already done. Story 16.9's job is to **either fix the SENTENCE_STREAM implementation, document a model-tier fallback policy, or formally revise NFR1 — and to amend the architecture document accordingly.**
>
> **The empirical baseline this story inherits.** Story 16.7's Tables 3.2 + 5 are the load-bearing evidence:
>
> | Path                       | n  | p50    | p95    | max    | NFR1 (p95<2s)? |
> |----------------------------|----|--------|--------|--------|----------------|
> | GPU SENTENCE_STREAM short  | 17 | 2.031s | 6.169s | 6.834s | **FAIL**       |
> | GPU SENTENCE_STREAM medium | 17 | 5.782s | 10.087s| 11.002s| **FAIL**       |
> | GPU SENTENCE_STREAM long   | 16 | 14.260s| 22.157s| 25.253s| **FAIL**       |
> | GPU SENTENCE_STREAM all    | 50 | 6.136s | 18.143s| 25.253s| **FAIL** (9.07× over) |
> | CPU SENTENCE_STREAM short  | 10 | 2.739s | 4.593s | 4.897s | **FAIL** (2.30× over; 0/10 cleared) |
> | GPU TRUE_STREAM (post-16.8)| 50 | 4.584s | 6.372s | 6.756s | **FAIL** (3.19× over; ~2.85× better than 16.7 SENTENCE_STREAM but still over) |
>
> The architecture's framing at `architecture-optimization-pass.md:802` ("NFR1 First audio <2s | GPU: meets via TRUE_STREAM (~1.5–1.8s estimated). CPU: meets via inherited SENTENCE_STREAM (per FR2 row). **Empirical measurement gate at Phase ⊥ POC.**") is **empirically contradicted on this host with this qwen-tts pin.** NFR1 cannot be claimed satisfied by ANY of the three streaming-mode paths on the maintainer's hardware until this story reconciles the gap.
>
> **Why this is the next entry point of Epic 16 — and the last.** The streaming-default flag flip (the user-facing release of TRUE_STREAM as the GPU default) was never gated on Story 16.7 alone — it was gated on the **conjunction** of Story 16.8 (TRUE_STREAM viable) AND Story 16.9 (NFR1 reconciled — either met or formally revised). Story 16.8 closed 2026-05-08; Story 16.9 is the remaining blocker. After Story 16.9 lands a verdict (per AC #3 below — implementation fix, model-tier fallback, OR contract revision), the streaming-default flag flip becomes a **separate, future "streaming default ramp" story** that re-runs the multi-listener perceptual A/B audition gate (Story 16.7 AC #2's deferred protocol). Story 16.9 does **NOT** itself flip the default and does **NOT** itself run the multi-listener audition. The perceptual gate stays an explicit prerequisite for the default flip — see "What this story is NOT" below.
>
> **Net behavior change for users.** **Depends on the outcome chosen (AC #3).**
>
>   - **Outcome (a) implementation fix:** GPU SENTENCE_STREAM first-audio latency drops to ≤2.0s p95 on the input set. CPU SENTENCE_STREAM may or may not also clear NFR1 — the fix's CPU efficacy is measured (AC #4) but does not gate this story's closure unless the fix made the CPU path *worse*. Production behavior change: **same dispatch path, faster first audio**. This is the most-disruptive-to-the-codebase outcome but the cleanest contract-wise.
>   - **Outcome (b) model-tier fallback:** Production CUDA users transparently get qwen-tts-0.6B for the first utterance (or for short-class inputs only), then upgrade to CUSTOM_VOICE for subsequent utterances. Behavior change for the user: **first utterance may sound "leaner" (smaller model); subsequent utterances are unchanged**. This is a product decision routed through `correct-course` for stakeholder sign-off (per the AC text).
>   - **Outcome (c) contract revision:** No behavior change. NFR1 is re-stated as e.g., "first audio <Xs on GPU short-class, <Ys medium/long, CPU exempted from streaming NFR1, falls back to V2 baseline" with a written engineering justification. The architecture document is amended; the streaming-default flag flip's blocking condition is restated. This is the lowest-effort outcome but requires the most careful framing because it formalizes a regression-from-projection.
>
> Importantly, **the empty-chunks fallback chain stays in place regardless of outcome**: TRUE_STREAM → SENTENCE_STREAM → BATCH per `qwen_tts_service.py::_dispatch_by_streaming_mode`, the empty-chunks guard inside `_generate_true_stream`, and Story 16.6's three-mode dispatch invariants. NFR7 (graceful degradation) is **NOT** weakened by any AC #3 outcome.
>
> **Pre-existing infrastructure already verified before drafting.**
>
>   - **Story 16.7's harness is the canonical re-run mechanism.** `scripts/validate_streaming_default.py` (~750 lines, committed in Story 16.7) takes `--mode-override`, `--input-set`, `--output-dir`, `--utterance-count`, `--chunk-size`, `--lookahead` flags; it produces the per-utterance CSV with `(utterance_id, text_length_chars, text_class, mode_requested, mode_dispatched, first_chunk_latency_seconds, total_audio_seconds, audio_sample_count, error_flag, wallclock_timestamp, qwen_tts_pin, torch_version, gpu_name)` columns; it refuses TRUE_STREAM on non-CUDA hosts (D-9 / NFR12 protection); it fixed its `_classify_dispatched_mode` classifier in Story 16.7 §3.2's note. **Story 16.9 inherits this harness as-is** — the only addition needed for AC #1 is *phase-decomposition profiling instrumentation*, which can be either a `--profile-phases` flag added to the harness OR a sibling probe script (`scripts/profile_sentence_stream.py`); the choice is the dev agent's judgment call (see AC #1 / Task 1 below).
>
>   - **The input set is fixed and committed.** `_bmad-output/implementation-artifacts/16-7-input-set.csv` (51 utterances, 17 short / 17 medium / 17 long; 10 perceptual-difficult overlay) is the canonical reproducibility fixture. Story 16.9 re-uses it unmodified. The perceptual-difficult subset (4 short + 4 medium + 2 long) is irrelevant to NFR1 timing measurement — that subset is for the future "streaming default ramp" story's audition. Story 16.9's measurement runs use the full 50-utterance set or the harness's stratified subset (`--utterance-count` + class-ordered loading per Story 16.7 §5's note on sample size).
>
>   - **The CPU baseline gap is documented.** Story 16.7's CPU run was 10 short-class utterances only (per AC #3's "≥10 records" floor; the harness loads `utterances[:limit]` from a class-ordered input set, so `--utterance-count 10` truncated to short-class). Story 16.9's CPU dimension extension (AC #4 below) is the **stratified sample** Story 16.7 §5's "Note on sample size and class coverage" explicitly named: ≥4 short / ≥4 medium / ≥2 long, total ≥10. The harness already supports this via `--utterance-count` + a manual input-set re-ordering OR a new `--stratified-sample` flag; the choice is again the dev agent's judgment call. **The harness must be invoked on a CPU-only host (or with `CUDA_VISIBLE_DEVICES=` to disable CUDA on the GPU host)** per Story 16.7 §7.1's reproducibility recipe.
>
>   - **The production code surface under investigation is `_generate_streaming`.** `src/myvoice/services/qwen_tts_service.py:2028-2242` is the SENTENCE_STREAM dispatch body. The phase decomposition Story 16.9 must produce decomposes its wall-clock into:
>       - **(a) Sentence split / preprocessing.** `_split_text_for_streaming(request.text)` at `qwen_tts_service.py:2423-2469` (uses `SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?。！？])\\s*')` and `MIN_CHUNK_LENGTH = 10` per `qwen_tts_service.py:444-447`).
>       - **(b) Model `.generate()` first-token latency.** Per chunk, `loop.run_in_executor(self._executor, self._generate_sync, chunk_request)` at `qwen_tts_service.py:2210-2214`. The `_generate_sync` body at `qwen_tts_service.py:3621-3720` calls one of `model.generate_custom_voice` / `model.generate_voice_design` / `model.generate_voice_clone` — all three are non-streaming entrypoints that block until the full audio for the chunk is decoded.
>       - **(c) Decode (`speech_tokenizer.decode`).** Inside the qwen-tts wrapper's `Qwen3TTSForConditionalGeneration.generate` at `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2292` — line 2280+ runs `speech_tokenizer.decode(...)` on the full codec output to produce the wav. **For SENTENCE_STREAM this is end-of-chunk decode, not chunk-by-chunk overlap-add.**
>       - **(d) Chunk delivery to `AudioCoordinator`.** The `_audio_chunk_ready_callback(chunk)` at `qwen_tts_service.py:2238-2239` plus the registry's `post_mutation('append_chunk', sid, audio_data)` at `qwen_tts_service.py:2219-2220`. Cheap relative to (b) + (c) but should be measured to confirm.
>     **The dominant phase (or phases) is what falsifies the four hypotheses below.** If (b) or (c) dominates, hypothesis (a) "qwen-tts version drift" or hypothesis (b) "CUSTOM_VOICE-vs-base-model penalty" are the most likely culprits; if (a) dominates, hypothesis (c) "sentence-split granularity" is the culprit; if all four are roughly equal, the projection in NFR1 was always optimistic and outcome (c) "contract revision" is the verdict.
>
>   - **The four hypotheses from Story 16.7 §1's recommendation are the framing for AC #2.**
>       1. **(a) qwen-tts version drift.** The architecture's "~1.5–1.8s estimated" projection (at line 802) was authored when qwen-tts was at an earlier version (the architecture was sealed 2026-04-27 per Story 16.7 §"Why this is the next entry point"; the qwen-tts pin was bumped to commit `1ab0dd75` = qwen-tts 0.0.4 in Story 16.1). If qwen-tts 0.0.4's `Qwen3TTSForConditionalGeneration.generate` is materially slower than the prior version, the projection was correct at authorship time and the regression is upstream. Falsified by: (i) `pip show qwen-tts` at 0.0.4 + a previous-version comparison run (maintainer would need to git-checkout an older requirements.txt + a separate venv — out of scope for Story 16.9 unless cheaply reproducible); (ii) reading the qwen-tts changelog between the architecture-cited version and 0.0.4 for any documented latency regression; (iii) **easier path:** measuring the time spent specifically in `model.generate_custom_voice(...)` at `qwen_tts_service.py:3663-3668` — if that single call dominates, hypothesis (a) is consistent and the upstream is the culprit.
>       2. **(b) Model-tier (size) penalty — 3B `quality` vs 0.6B `small`.** Story 16.7's runs all used `model_type=CUSTOM_VOICE` AND the default `quality` tier (the 3B model). The architecture's "~1.5–1.8s estimated" projection may have been based on the smaller 0.6B `small` tier, not the 3B `quality` tier. **The 3B-vs-0.6B distinction is governed by `ModelRegistry.quality_tier` (`model_registry.py:154`), not by `QwenModelType`** — `QwenModelType.CUSTOM_VOICE` / `VOICE_DESIGN` / `BASE` are different generation modes (CUSTOM_VOICE = speaker-by-name; VOICE_DESIGN = description-driven; BASE = voice-cloning-from-reference-audio), not tiers. Falsified by: re-running `_generate_streaming` with `tier_override="small"` (the 0.6B model) while keeping `model_type=CUSTOM_VOICE` constant on the same input set; if the small-tier p95 ≤ 2.0s and the quality-tier p95 > 2.0s, hypothesis (b) is confirmed and outcome (b) "model-tier fallback" becomes the recommended verdict. **The correct API call to load the small tier on demand is `await model_registry.ensure_model_loaded(QwenModelType.CUSTOM_VOICE, tier_override="small")` (`model_registry.py:241-265`)**; the persistent setting can be flipped via `await model_registry.set_quality_tier("small")` (`model_registry.py:158`). The harness can either add a `--quality-tier {small,quality}` flag OR a `--tier-override` flag that the dev agent picks; either is acceptable per scope-point (a) judgment-call framing.
>       3. **(c) Sentence-split granularity.** `_split_text_for_streaming` returns N chunks per utterance; for short-class utterances (≤30 chars) this is typically 1 chunk; for long-class (150+ chars) it's typically 3–5 chunks. The first-chunk latency is therefore (sentence split) + (full `_generate_sync` for chunk #1). For short-class, that's the full utterance through `model.generate_custom_voice` — there's no smaller chunk to dispatch first. Falsified by: comparing first-chunk-latency vs. text length — if the latency scales linearly with chunk-1's text length (rather than being roughly constant for short-class), hypothesis (c) is confirmed and the fix is to split more aggressively (e.g., split on commas, or cap first-chunk length at ~40 chars). **Note:** SENTENCE_STREAM does not actually stream per-token — it streams per-sentence, with each sentence being a full non-streaming `_generate_sync` call. So "first audio" = "first complete sentence's audio". TRUE_STREAM is the per-token streaming path; SENTENCE_STREAM's "first chunk" is "first sentence". This is a structural property of the SENTENCE_STREAM dispatch; the only sub-sentence parallelism would require splitting sentences mid-clause, which the current `SENTENCE_SPLIT_PATTERN` does not do.
>       4. **(d) NFR1 was always optimistic.** The architecture's "~1.5–1.8s estimated" was the authoring team's projection based on assumptions about hardware mix, model tier, and qwen-tts internals that may have shifted over time. Settled by: writing a one-page rationale for outcome (c) "contract revision" — propose an updated NFR1 with explicit per-class targets, route through `correct-course`. **Important:** outcome (c) is not a failure mode; it's a legitimate verdict if hypotheses (a)–(c) are all falsified or the cost of fixing them outweighs the value.
>
>   - **The architecture document is the canonical contract surface.** `_bmad-output/planning-artifacts/architecture-optimization-pass.md:802` is the line that names NFR1's projection. Story 16.9 amends this line (or the surrounding section) with a dated note pointing to this story's findings, regardless of which AC #3 outcome is chosen. The amendment language is prescribed in AC #5 below: a brief `> NOTE (2026-05-XX, Story 16.9): ...` block linking to this story file. **The blocking condition restatement** ("blocked until both Story 16.8 (TRUE_STREAM viable) AND Story 16.9 (NFR1 reconciled — either met or formally revised) close") goes in the same amendment.
>
>   - **No new dependency.** `qwen-tts` is pinned at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1; `transformers`, `torch`, `numpy`, `soundfile` are already imported. The harness uses `time.perf_counter` for wall-clock measurement. No `requirements.txt` change. **No qwen-tts pin bump** — that's a separate-scope follow-up if outcome (c) revises NFR1 to the point where a future qwen-tts version becomes a viable path forward.
>
>   - **No `AppSettings` schema changes** *unless* outcome (b) "model-tier fallback" is chosen — in that case a new field like `streaming_first_utterance_model_tier_override` may be needed. **The dev agent must defer that schema decision until AC #3's outcome is chosen** (i.e., do not pre-emptively add fields). If outcome (b) is chosen, the schema change becomes part of the implementation deliverable; if outcome (a) or (c) is chosen, no schema change is needed.
>
>   - **No registry / streamer / decoder / coordinator changes.** Story 16.9 does NOT touch `services/sessions/*`, `services/audio_coordinator.py`, `services/tts_streaming/*` (codec_token_streamer, streaming_decoder, streaming_mode), or any UI surface. The investigation lives in `_generate_streaming` / `_split_text_for_streaming` / `_generate_sync` and at most in `qwen_tts_service.py`'s constants section.
>
>   - **Memory + DLL ordering invariants apply.** Per `memory/torch_pyqt6_dll_ordering.md`, any new probe script MUST `import torch` BEFORE `import PyQt6.*`. The existing `scripts/validate_streaming_default.py` preamble (`scripts/validate_streaming_default.py:57-87`) is the reference; mirror it in any new instrumentation file.
>
> **Eight-point story scope:**
>
> (a) **Author phase-decomposition profiling instrumentation.** Either (a1) extend `scripts/validate_streaming_default.py` with a `--profile-phases` flag that, when set, wraps `_generate_streaming`'s internal phases with `time.perf_counter` measurements and emits an additional CSV column block (`split_seconds, generate_seconds, decode_seconds, deliver_seconds`) per measurement; OR (a2) author a sibling `scripts/profile_sentence_stream.py` (~150-200 lines) that monkey-patches `_split_text_for_streaming` / `_generate_sync` / the `_audio_chunk_ready_callback` site to record phase timings into a per-utterance `PhaseProfile` record. **Choose (a1) over (a2) if the harness fits the change in <50 net new lines; otherwise (a2)** to keep the harness clean. **Both paths produce a per-utterance CSV with the same columns;** the dev agent picks whichever is cheaper. The instrumentation must NOT change `_generate_streaming`'s behavior in production (instrumentation is harness-side only — patching `time.perf_counter` calls into production code is acceptable only if guarded by an env var or feature flag that defaults off).
>
> (b) **Run the instrumented harness on the maintainer's RTX 5090 + qwen-tts 0.0.4 host against the full 50-utterance input set.** Produce `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (50 rows, the existing CSV columns plus the four phase columns). Compute per-class per-phase aggregates (split / generate / decode / deliver, each with p50 / p95 / max) and record them in this story's Change Log. **Identify the dominant phase** (or two phases if both are >25% of total wall-clock).
>
> (c) **Falsify each of the four hypotheses (a)–(d) above** with the phase-profile data PLUS at most one targeted comparison run per hypothesis. Concretely:
>   - **Hypothesis (a) qwen-tts version drift:** record the dominant phase for the `generate_custom_voice` call. If it dominates by >70% of wall-clock, this is consistent with hypothesis (a) and the verdict is "the upstream is the bottleneck, no MyVoice fix possible without a pin bump." If pin-bump is desired, that's a separate story scope.
>   - **Hypothesis (b) CUSTOM_VOICE-vs-base-model penalty:** run one targeted comparison: load `QwenModelType.BASE` and re-run the harness against ≥10 short-class utterances. Produce `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-base-model-comparison.csv`. If BASE clears NFR1 and CUSTOM_VOICE doesn't, hypothesis (b) is confirmed.
>   - **Hypothesis (c) sentence-split granularity:** correlate first-chunk-latency with chunk-1's text length within the existing 50-row CSV. If short-class utterances (whose first chunk is the full utterance) are the only ones missing NFR1 by a small margin AND medium/long utterances also miss NFR1 even though their first chunk is shorter than the full utterance, hypothesis (c) is **partially** falsified (sentence split helps but not enough). If short-class utterances clear NFR1 but medium/long ones don't (and their first chunk is longer than the short-class average), hypothesis (c) is more strongly suspected.
>   - **Hypothesis (d) NFR1 was always optimistic:** if hypotheses (a), (b), (c) are all consistent with the data but no single one explains the gap, hypothesis (d) is the residual verdict. Document it as a one-paragraph rationale in this story's Change Log.
>   **At least one hypothesis must be settled with a concrete code or measurement output (not "we should look into this someday").** Record the verdict per hypothesis in this story's Change Log.
>
> (d) **Choose ONE of the three AC #3 outcomes — implementation fix, model-tier fallback, OR contract revision — based on the falsification verdict from (c).** Document the choice in this story's Change Log with a one-paragraph rationale citing the dominant phase and the falsified/confirmed hypothesis. The decision rule is:
>   - If hypothesis (c) is confirmed (split granularity is the bottleneck) → outcome (a) implementation fix (tighten the splitter).
>   - If hypothesis (b) is confirmed (CUSTOM_VOICE is the bottleneck) → outcome (b) model-tier fallback (add a `first_utterance_model_tier_override` policy).
>   - If hypothesis (a) is confirmed (upstream is the bottleneck) and pin-bump is out of scope → outcome (c) contract revision.
>   - If hypothesis (d) is the residual → outcome (c) contract revision.
>   - Mixed verdicts are acceptable — outcome (a) AND outcome (c) can co-exist (a partial fix + revised contract).
>
> (e) **Implement the chosen outcome:**
>   - **Outcome (a) implementation fix:** ship a code change to `_generate_streaming` / `_split_text_for_streaming` / a new constant tuning pass. Re-run Story 16.7's harness to validate p95 ≤ 2.0s GPU short-class (medium/long may be relaxed per the per-class targets from outcome (c) if a hybrid is chosen). Commit the new harness CSV at `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-after-fix.csv`.
>   - **Outcome (b) model-tier fallback:** ship the `AppSettings.streaming_first_utterance_model_tier_override` field (or equivalent name) plus the `_generate_streaming` branch that resolves it to a tier load. Document the policy ("CUDA users with default settings get qwen-tts-0.6B BASE for the first utterance, then upgrade to CUSTOM_VOICE") in the architecture doc. Route through `correct-course` for stakeholder sign-off **before merge**.
>   - **Outcome (c) contract revision:** propose updated NFR1 wording (e.g., "first audio <2s on GPU short-class, <5s GPU medium/long, CPU exempted from streaming NFR1, falls back to V2 baseline"). Route through `correct-course` for stakeholder sign-off **before merge**. Amend `architecture-optimization-pass.md:802` with the new wording. **No code change.**
>
> (f) **Extend the CPU baseline with stratified sampling per Story 16.7 §5's note.** Run the harness on a CPU-only configuration (either CPU-only host OR `CUDA_VISIBLE_DEVICES=` env-var on the GPU host) against ≥4 short / ≥4 medium / ≥2 long utterances (total ≥10). Produce `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv`. Compute per-class p50/p95/max and record in this story's Change Log. **The CPU verdict joins the GPU verdict in AC #3's outcome — both paths must clear the gate (or both must clear a revised gate).**
>
> (g) **Amend `architecture-optimization-pass.md:802` and surrounding sections.** Add a dated `> NOTE (2026-05-XX, Story 16.9): ...` block linking to this story file. The note's content depends on the chosen outcome — see AC #5 for the prescribed wording per outcome. **The streaming-default flag flip's blocking condition is restated explicitly:** "blocked until both Story 16.8 (TRUE_STREAM viable) AND Story 16.9 (NFR1 reconciled — either met or formally revised) close" — Story 16.9's closure satisfies this clause regardless of which AC #3 outcome is chosen.
>
> (h) **Author the committed reconciliation report at `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md`.** ~150-250 lines. Sections: (1) Executive summary — chosen outcome and one-line implication. (2) Methodology — phase-profiling instrumentation choice, hypothesis-falsification protocol. (3) Phase-profile results — per-class per-phase aggregates, dominant-phase identification. (4) Hypothesis verdicts — per hypothesis, whether falsified / confirmed / residual. (5) Outcome rationale — the AC #3 choice and why. (6) Implementation summary — what code/doc changed (if outcome (a) or (b)) or what contract was revised (if outcome (c)). (7) CPU dimension — stratified-sample verdict, joins or diverges from GPU verdict. (8) Reproducibility — exact commands, exact files committed.
>
> ---
>
> **What this story is NOT** (explicit, to keep scope bounded):
>
> - This story is **NOT** the streaming-default flag flip. The flip is a separate, future "streaming default ramp" story informed by THIS story's report PLUS Story 16.7 AC #2's deferred multi-listener perceptual A/B audition. Story 16.9 closes the latency-contract dimension; the perceptual gate still has to clear separately. **Do not flip any flag in `streaming_mode.py:54-56` or any UI initializer.**
>
> - This story is **NOT** the multi-listener perceptual A/B audition. The audition was deferred by Story 16.7 (TRUE_STREAM was silent) and partially exercised by Story 16.8 (Commander solo, catastrophic-failure dimension only). The full multi-listener audition is reserved for the future "streaming default ramp" story. Story 16.9 does NOT re-run `scripts/build_streaming_perceptual_ab_fixture.py`, does NOT collect listener observations, does NOT touch any audition fixture.
>
> - This story is **NOT** a qwen-tts pin bump. The pin remains at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1. If hypothesis (a) is confirmed and the verdict is "the upstream is the bottleneck", the recommendation goes in this story's report but the actual pin bump is a separate story (with the trip-wire test as the safety net per D-12). **Do not edit `requirements.txt`.**
>
> - This story is **NOT** an architectural rewrite of SENTENCE_STREAM. If the conclusion is "SENTENCE_STREAM cannot meet NFR1 by design and the only path to NFR1 compliance is TRUE_STREAM", that's a legitimate finding — but it makes outcome (c) "contract revision" the verdict, not "rewrite SENTENCE_STREAM". The dispatch path's structure (sentence-split + per-chunk non-streaming `_generate_sync`) is preserved.
>
> - This story does **NOT** modify the empty-chunks fallback chain. The Story 16.7 guard (`qwen_tts_service.py::_generate_true_stream` empty-chunks check) and Story 16.6's three-mode dispatcher (`_dispatch_by_streaming_mode`) are unchanged. NFR7 (graceful degradation) survives every AC #3 outcome.
>
> - This story does **NOT** add a dedicated `torch.cuda.Stream` for the decoder (D-8 follow-up). That's a future profiling-driven optimization tracked separately. If phase profiling fingers the decoder as the bottleneck for SENTENCE_STREAM (unlikely — decode runs synchronously inside `model.generate_*` for SENTENCE_STREAM, not in a separate thread), the recommendation goes in the report and the actual work is a separate story.
>
> - This story does **NOT** touch `tests/integration/test_streaming_tts_smoke.py`, `tests/unit/services/test_qwen_tts_service_dispatch.py`, `tests/test_qwen_tts_internals.py`, or any other production-test file **unless outcome (a) implementation fix or (b) model-tier fallback requires regression coverage** for the new code path. Outcome (c) contract revision adds no new tests (it's documentation-only). The dev agent must add the minimum tests for outcome (a) or (b); the trip-wire is unchanged either way.
>
> - This story does **NOT** add or change any dependency, change `requirements.txt`, change `requirements-production.txt`, change CI configuration, change UI behavior (unless outcome (b) adds a settings field with UI), change any signal contract, or change any module boundary.

## Story

As a **MyVoice maintainer**,
I want **the SENTENCE_STREAM first-audio latency on GPU and CPU reconciled with NFR1's 2-second ceiling — either by a code fix that brings p95 ≤ 2.0s on the input set, by a model-tier fallback policy that uses qwen-tts-0.6B for the first utterance, or by a formally-revised NFR1 contract routed through `correct-course` and amended in the architecture document**,
So that **the streaming default flag flip is not blocked by a parallel performance regression after Story 16.8 made TRUE_STREAM viable, the architecture document's NFR1 framing reflects empirical reality on the maintainer's hardware mix, and any future regression in SENTENCE_STREAM latency is gated by a re-runnable harness against a fixed input set**.

As a **MyVoice user (GPU host, default settings)**,
I want **the maintainer to either fix or formally acknowledge the gap between the architecture's projected first-audio latency and what I actually experience**,
So that **the marketing-level promise of "first audio in 2 seconds" matches reality, and if the implementation can be improved to clear that ceiling I get a faster experience, while if the contract has to be revised I at least get a transparent statement of what to expect**.

As a **MyVoice user (CPU-only host)**,
I want **the maintainer to verify that the SENTENCE_STREAM path I rely on is either fixed to satisfy NFR1 or formally exempted from the streaming-NFR1 contract with a clear inheritance from the V2 baseline**,
So that **my code path has a documented latency expectation rather than an empirically-contradicted projection (NFR12 protection, made explicit in the contract)**.

## Acceptance Criteria

**Background — what this story is and is NOT.**

This story does eight things to the working tree: (a) authors phase-decomposition profiling instrumentation (either a `--profile-phases` flag on `scripts/validate_streaming_default.py` or a sibling `scripts/profile_sentence_stream.py`); (b) runs the instrumented harness on the maintainer's RTX 5090 + qwen-tts 0.0.4 host and commits the resulting per-utterance phase-profile CSV; (c) falsifies each of the four hypotheses from Story 16.7 §1's recommendation with the phase-profile data plus at most one targeted comparison run per hypothesis; (d) chooses ONE of the three AC #3 outcomes — implementation fix, model-tier fallback, OR contract revision — based on the falsification verdict; (e) implements the chosen outcome (code change for (a)/(b), `correct-course` routing + architecture amendment for (c)); (f) extends the CPU baseline with stratified sampling (≥4 short / ≥4 medium / ≥2 long); (g) amends `_bmad-output/planning-artifacts/architecture-optimization-pass.md:802` and surrounding sections with a dated `> NOTE (2026-05-XX, Story 16.9): ...` block linking to this story file and restating the streaming-default flag flip's blocking condition; (h) authors the committed reconciliation report at `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md`. The deliverable is bounded to:

- `scripts/validate_streaming_default.py` (modified — `--profile-phases` flag added) OR `scripts/profile_sentence_stream.py` (new — sibling profiler) — depending on the dev agent's judgment per scope-point (a) above; ~50-200 net new lines either way
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (new — committed via `git add -f`; produced by Task 2)
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv` (new — committed via `git add -f`; produced by Task 3 hypothesis (b) probe; filename reflects `quality` vs `small` tier comparison, not the misleading `base-model` framing)
- `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` (new — committed via `git add -f`; produced by Task 6)
- `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` (new — ~150-250 lines)
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (modified — dated `> NOTE` block at line 802 and any related amendments per Task 7)
- `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-sentence-stream-latency-investigation.md` (this file, the story doc itself; updated as Change Log entries accumulate)
- **Conditionally**, depending on AC #3 outcome:
  - **Outcome (a):** modifications to `src/myvoice/services/qwen_tts_service.py` (likely `_split_text_for_streaming` or `MIN_CHUNK_LENGTH` / `SENTENCE_SPLIT_PATTERN` constants); ~10-50 net new lines plus regression tests in `tests/unit/services/test_qwen_tts_service_dispatch.py` (~30-60 net new lines); plus `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-after-fix.csv` (committed via `git add -f`)
  - **Outcome (b):** new field on `src/myvoice/models/app_settings.py` (`streaming_first_utterance_tier_override: Optional[Literal["small","quality"]]` or equivalent name); branch in `_generate_streaming` that calls `ensure_model_loaded(model_type, tier_override="small")` for chunk #1 then `tier_override=None` for subsequent chunks (per `model_registry.py:241-265`); UI exposure in `src/myvoice/ui/dialogs/settings_dialog.py` Streaming tab; ~80-150 net new lines plus regression tests
  - **Outcome (c):** **NO code change**; only the architecture amendment in (g) above plus the `correct-course` routing artifact

This story does **NOT**:

- Touch `services/sessions/*`, `services/audio_coordinator.py`, `services/tts_streaming/codec_token_streamer.py`, `services/tts_streaming/streaming_decoder.py`, `services/tts_streaming/streaming_mode.py`, the registry, the streamer, the decoder worker, or any UI surface other than (conditionally) the settings dialog Streaming tab if outcome (b) is chosen.
- Touch `_build_true_stream_talker`, `_build_true_stream_decode_fn`, `_generate_true_stream`, or any TRUE_STREAM dispatch surface — Story 16.8's territory, closed.
- Touch `tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure` (the empty-chunks regression guard) or `tests/integration/test_streaming_tts_smoke.py::TestTrueStreamWireUpEndToEnd` (the Story 16.8 wire-up regression guard) — both stay unchanged.
- Flip any default flag in `streaming_mode.py`, `AppSettings`, or any UI initializer. The streaming-default flag flip is a future "streaming default ramp" story.
- Run the multi-listener perceptual A/B audition. That's a future "streaming default ramp" story.
- Bump the qwen-tts pin in `requirements.txt`. The pin remains at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1.
- Add or remove dependencies. `transformers`, `torch`, `numpy`, `qwen-tts`, `soundfile`, `time` (stdlib) are sufficient.
- Add a dedicated `torch.cuda.Stream` for the decoder (D-8 follow-up).

The deliverable is approximately **+50-200 lines for profiling instrumentation**, **+150-250 lines for the reconciliation report**, **3-4 new committed CSVs**, **the architecture amendment block**, and conditionally **+10-150 lines of production code + regression tests** (outcome (a) or (b)) OR **zero code change** (outcome (c)). The Change Log documents (a) the profiling instrumentation choice (a1 vs a2), (b) the per-class per-phase aggregates, (c) the hypothesis verdicts, (d) the AC #3 outcome and rationale, (e) the implementation summary or contract revision, (f) the CPU stratified-sample verdict, (g) the architecture amendment text, (h) `correct-course` routing artifact (if outcome (b) or (c)).

---

**AC #1 — The phase-decomposition profile of `_generate_streaming` is produced on the maintainer's RTX 5090 + qwen-tts 0.0.4 host against the full 50-utterance Story 16.7 input set, and the dominant phase is identified with measurement evidence.**

**Given** the existing harness `scripts/validate_streaming_default.py` (committed in Story 16.7, ~750 lines) and the existing `_bmad-output/implementation-artifacts/16-7-input-set.csv` (51 utterances, class-stratified)
**When** the maintainer either (a1) extends the harness with a `--profile-phases` flag that wraps `_generate_streaming`'s internal phases with `time.perf_counter` measurements, OR (a2) authors a sibling `scripts/profile_sentence_stream.py` that records the same phase timings via monkey-patching
**Then** the chosen instrumentation produces a per-utterance CSV row with at minimum these columns: `utterance_id`, `text_length_chars`, `text_class`, `split_seconds` (phase a — `_split_text_for_streaming`), `generate_seconds` (phase b — `_generate_sync` for chunk #1, including the wrapper's preprocessing + talker `.generate()` time), `decode_seconds` (phase c — `speech_tokenizer.decode` inside `Qwen3TTSForConditionalGeneration.generate` at `qwen_tts/core/models/modeling_qwen3_tts.py:2280+`; the dev agent has TWO options for separate measurability: (i) monkey-patch `Qwen3TTSForConditionalGeneration.generate` from the harness to wrap `time.perf_counter` around the wrapper's `speech_tokenizer.decode(...)` call site — invasive but precise; (ii) merge this column into `generate_seconds` with a Change Log note explaining the merge — accepted if option (i) is too brittle. **Pick (ii) by default** unless the maintainer wants a more granular phase verdict.), `deliver_seconds` (phase d — `_audio_chunk_ready_callback` + registry `post_mutation('append_chunk', ...)`), `total_first_chunk_latency_seconds` (the existing aggregate, for sanity-check that phases sum to within 5% of the total)
**And** the instrumentation does NOT change `_generate_streaming`'s production behavior (instrumentation is harness-side only — patched into production code only via env-var-guarded conditional or monkey-patch from the harness, not as a permanent edit)

**Given** the instrumented harness/profiler is run as `python310\python.exe scripts\validate_streaming_default.py --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv --output-dir _bmad-output\implementation-artifacts\ --mode-override sentence_stream --utterance-count 50 --profile-phases` (or the sibling profiler's equivalent invocation)
**When** the run completes
**Then** the output CSV `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` has 50 rows with `error_flag == ""` (any rows with errors are excluded from the per-class aggregates)
**And** per-class per-phase aggregates (short / medium / long, each with p50 / p95 / max for the four phase columns) are computed and recorded in this story's Change Log
**And** the dominant phase (or two phases if both are >25% of total wall-clock) is identified explicitly: e.g., "phase b `generate_seconds` accounts for ≥80% of total first-chunk latency across all classes — model invocation is the bottleneck"
**And** the phase columns sum to within 5% of `total_first_chunk_latency_seconds` (sanity check that no significant phase is unmeasured; if not, Change Log notes what's missing)

---

**AC #2 — Each of the four hypotheses from Story 16.7 §1's recommendation is settled with a concrete measurement output, code-read finding, OR explicit residual designation — not "we should look into this someday".**

**Given** the four hypotheses: (a) qwen-tts version drift, (b) CUSTOM_VOICE-vs-base-model penalty, (c) sentence-split granularity, (d) NFR1-was-always-optimistic
**When** the phase-profile data from AC #1 is in hand
**Then** each hypothesis is settled with one of: **falsified** (data contradicts), **confirmed** (data supports), or **residual** (data is silent on this hypothesis; it is the leftover after the other three are settled)

**Given** hypothesis (a) qwen-tts version drift
**When** the phase-profile shows phase b (`generate_seconds`) dominates by >70% of wall-clock
**Then** hypothesis (a) is **consistent** (the upstream's `model.generate_custom_voice` is the bottleneck) — but not yet confirmed; confirming would require a previous-version comparison run which is **out of scope** for Story 16.9 unless cheaply reproducible
**And** the verdict is recorded as "consistent (upstream-bound but not directly verified)" in the Change Log

**Given** hypothesis (b) model-tier (size) penalty — 3B `quality` vs 0.6B `small`
**When** the maintainer runs one targeted comparison: load the small tier via `await model_registry.ensure_model_loaded(QwenModelType.CUSTOM_VOICE, tier_override="small")` (or via `--quality-tier small` if the harness flag is added) and re-run `_generate_streaming` against ≥10 short-class utterances from the input set with `model_type=CUSTOM_VOICE` held constant
**Then** the comparison CSV `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv` is produced (note: filename uses `small-tier`, not `base-model`, to reflect that the comparison is between tiers of CUSTOM_VOICE, not between QwenModelType variants)
**And** if the small-tier p95 ≤ 2.0s for short-class AND the quality-tier p95 > 2.0s for short-class (gap >2×) → hypothesis (b) is **confirmed**, the model size is the bottleneck, outcome (b) "model-tier fallback" is the recommended verdict
**And** if the small-tier p95 is also > 2.0s for short-class → hypothesis (b) is **falsified**, the model tier is not the bottleneck

**Given** hypothesis (c) sentence-split granularity
**When** the maintainer correlates first-chunk-latency vs. chunk-1's text length (computed from `_split_text_for_streaming(request.text)[0]` for each utterance) within the existing 50-row CSV
**Then** if first-chunk-latency scales near-linearly with chunk-1's text length AND short-class utterances (whose chunk-1 is the full utterance, ≤30 chars) clear NFR1 while medium/long miss → hypothesis (c) is **partially confirmed**, finer splitting would help medium/long but cannot help short
**And** if first-chunk-latency is roughly constant regardless of chunk-1's text length → hypothesis (c) is **falsified**, splitting more aggressively would not help

**Given** hypothesis (d) NFR1 was always optimistic
**When** hypotheses (a), (b), (c) are settled
**Then** if no single one of (a), (b), (c) is **confirmed** (all are falsified or merely consistent) → hypothesis (d) is the **residual**, NFR1 was always optimistic and outcome (c) "contract revision" is the verdict
**And** if at least one of (a), (b), (c) is confirmed → hypothesis (d) is **falsified** (the gap has an explanation), and the verdict drives the AC #3 outcome to (a) or (b)

**At least one hypothesis must end in a state other than "we should look into this someday"** — the Change Log entry per hypothesis must cite the specific data point that settles it.

---

**AC #3 — ONE of the three legitimate outcomes is committed: (a) implementation fix, (b) model-tier fallback, OR (c) contract revision. The chosen outcome is implemented (or for outcome (c), routed through `correct-course`) before this story's status flips to `done`.**

**Given** the hypothesis verdicts from AC #2
**When** the maintainer chooses an outcome per the decision rule:
  - Hypothesis (b) confirmed → outcome (b) model-tier fallback
  - Hypothesis (c) confirmed (even partially) → outcome (a) implementation fix (tighten the splitter, possibly hybrid with (c))
  - Hypothesis (a) consistent + (b) and (c) falsified → outcome (c) contract revision (upstream-bound, no MyVoice fix without pin bump)
  - Hypothesis (d) residual → outcome (c) contract revision
  - Mixed verdicts → outcome (a) AND outcome (c) co-exist (partial fix + revised contract for the residual gap)
**Then** the chosen outcome is recorded in this story's Change Log with a one-paragraph rationale citing the dominant phase from AC #1 and the falsified/confirmed hypothesis from AC #2

**Given** outcome (a) implementation fix is chosen
**When** the maintainer ships the code change
**Then** the change is bounded to `src/myvoice/services/qwen_tts_service.py` (likely `_split_text_for_streaming` or the `MIN_CHUNK_LENGTH` / `SENTENCE_SPLIT_PATTERN` constants) — ~10-50 net new lines
**And** regression tests are added to `tests/unit/services/test_qwen_tts_service_dispatch.py` covering the new behavior — ~30-60 net new lines
**And** Story 16.7's harness is re-run on the same input set with the fix in place, producing `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-after-fix.csv`
**And** the post-fix p95 is ≤2.0s for at least the short-class subset (medium/long may be relaxed per a hybrid contract-revision sub-outcome if outcome (a) is partial)

**Given** outcome (b) model-tier fallback is chosen
**When** the maintainer ships the change
**Then** a new field `streaming_first_utterance_tier_override: Optional[Literal["small","quality"]]` (or equivalent name; default `"small"` if the policy is "use 0.6B for first utterance") is added to `src/myvoice/models/app_settings.py`
**And** `_generate_streaming` reads the field and uses the tier-override API: `await self._model_registry.ensure_model_loaded(request.model_type, tier_override="small")` for chunk #1, then `tier_override=None` (which falls back to the persistent `quality_tier` setting via `model_registry.py:407-408`) for subsequent chunks
**And** the settings dialog Streaming tab exposes the new field with a tooltip explaining the trade-off (smaller model on first utterance = faster first audio but leaner voice quality on chunk #1; subsequent chunks return to full-quality)
**And** the policy is **routed through `/bmad-bmm-correct-course` for stakeholder sign-off BEFORE the merge** — the routing artifact must surface the three load-management options for stakeholder decision (see Project Structure Notes below) and the chosen option is committed alongside the code
**And** regression tests are added covering both the fallback-active and fallback-disabled paths, mirroring the exact-bug-class pattern from `memory/code_review_regression_test_exact_class.md`

**Given** outcome (c) contract revision is chosen
**When** the maintainer drafts the revised NFR1 wording
**Then** the new wording explicitly distinguishes per-class targets (e.g., "first audio <2s on GPU short-class, <5s GPU medium/long, CPU exempted from streaming NFR1, falls back to V2 baseline") with an engineering justification citing the AC #1 phase-profile and the AC #2 verdict
**And** the wording is **routed through `/bmad-bmm-correct-course` for stakeholder sign-off BEFORE the merge** — the routing artifact is committed alongside the architecture amendment
**And** **no production code is modified** — outcome (c) is documentation-only; the architecture amendment in AC #5 is the load-bearing artifact

---

**AC #4 — The CPU dimension of NFR1 reconciliation is extended with stratified sampling (≥4 short / ≥4 medium / ≥2 long, total ≥10) and the CPU verdict joins the GPU verdict in the chosen AC #3 outcome.**

**Given** Story 16.7 §5's note on sample size and class coverage ("a follow-up CPU run with stratified sampling — e.g. 4 short / 4 medium / 2 long — would lift the bound")
**When** the maintainer runs the harness on a CPU-only configuration (CPU-only host OR `CUDA_VISIBLE_DEVICES=` env-var on the GPU host) against ≥4 short / ≥4 medium / ≥2 long utterances from the input set (total ≥10)
**Then** the output CSV `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` has ≥10 rows with `error_flag == ""`
**And** per-class p50/p95/max are computed and recorded in this story's Change Log
**And** the harness's classifier confirms `mode_dispatched = sentence_stream` for every row (no fallback occurrences — the CPU path is hard-pinned to SENTENCE_STREAM via D-9 / NFR12)

**Given** the CPU per-class aggregates
**When** the maintainer compares them against NFR1 (or against the revised contract if outcome (c) is chosen)
**Then** the CPU verdict is recorded explicitly: "CPU SENTENCE_STREAM clears NFR1 for all classes" / "CPU SENTENCE_STREAM clears NFR1 for short-class only" / "CPU SENTENCE_STREAM does not clear NFR1 for any class" — depending on the data
**And** if outcome (a) "implementation fix" was chosen for GPU, the CPU run is **after** the fix (so the CPU verdict reflects the post-fix state)
**And** if outcome (c) "contract revision" was chosen, the revised NFR1's CPU clause (e.g., "CPU exempted from streaming NFR1, falls back to V2 baseline") is verified empirically via this CSV

---

**AC #5 — The architecture document `_bmad-output/planning-artifacts/architecture-optimization-pass.md` is amended in two coordinated places: (i) the NFR1 row at line 802 (inside the Requirements Coverage Validation markdown table) is amended in-cell with a brief inline pointer; (ii) a new prose sub-section "Story 16.9 Follow-up Note (NFR1 Reconciliation)" is added immediately AFTER the table (before "### Implementation Readiness Validation" at ~line 819) carrying the dated note. This split is required because a `> NOTE` blockquote inserted directly below the table row would terminate the markdown table and orphan the rows below it (NFR3, NFR4, NFR6, NFR7, NFR11, NFR12).**

**Given** the chosen AC #3 outcome and the AC #4 CPU verdict
**When** the maintainer amends the architecture document
**Then** the NFR1 cell at line 802 has a brief inline pointer appended after "Empirical measurement gate at Phase ⊥ POC.": for example, ` (Story 16.9 reconciled 2026-05-XX — see follow-up note below.)`
**And** a new prose sub-section is inserted between the closing of the NFR table and the next `### Implementation Readiness Validation` heading, with the heading `#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-XX)` and a body paragraph whose content varies by outcome:
  - **Outcome (a) implementation fix:** `"Empirical measurement on RTX 5090 + qwen-tts 0.0.4 (Story 16.7 §3.2) showed GPU SENTENCE_STREAM p95 = 18.143s (9.07× over the 2s ceiling). Story 16.9 identified [dominant phase] as the bottleneck and shipped [code-fix summary]; post-fix p95 = [new value]. The 2s ceiling now holds for [class scope]; see 16-9-nfr1-reconciliation-report.md."`
  - **Outcome (b) model-tier fallback:** `"Empirical measurement on RTX 5090 + qwen-tts 0.0.4 (Story 16.7 §3.2) showed GPU SENTENCE_STREAM CUSTOM_VOICE p95 = 18.143s (9.07× over the 2s ceiling). Story 16.9 confirmed model-tier penalty (BASE p95 = [value], CUSTOM_VOICE p95 = [value]) and shipped a model-tier fallback policy: CUDA users with default settings get qwen-tts-0.6B BASE for the first utterance, then upgrade to CUSTOM_VOICE for subsequent utterances. The 2s ceiling holds for first utterance; subsequent utterances may exceed depending on class. See 16-9-nfr1-reconciliation-report.md."`
  - **Outcome (c) contract revision:** `"Empirical measurement on RTX 5090 + qwen-tts 0.0.4 (Story 16.7 §3.2) showed the original projection ('~1.5–1.8s estimated') was empirically contradicted (GPU SENTENCE_STREAM p95 = 18.143s; GPU TRUE_STREAM post-Story-16.8 p95 = 6.372s; CPU SENTENCE_STREAM short-class p95 = 4.593s). Story 16.9 reconciled NFR1 with empirical reality: revised wording is [new wording]. The original 2s ceiling is replaced by per-class targets; CPU exempted from streaming NFR1 and falls back to V2 baseline. See 16-9-nfr1-reconciliation-report.md."`
**And** the streaming-default flag flip's blocking condition is restated in the same prose sub-section (as a separate paragraph or trailing sentence): `"The streaming-default flag flip (the user-facing release of TRUE_STREAM as the GPU default — a one-line edit at streaming_mode.py:54-56 or a settings UI initializer) was blocked on the conjunction of Story 16.8 (TRUE_STREAM viable — closed 2026-05-08) AND Story 16.9 (NFR1 reconciled — closed 2026-05-XX). With Story 16.9's closure, the flag flip's remaining prerequisite is the multi-listener perceptual A/B audition (Story 16.7 AC #2's deferred protocol), tracked in a future 'streaming default ramp' story."`

**Given** a mixed AC #3 outcome (e.g., partial implementation fix per (a) AND a revised contract per (c) for the residual gap)
**When** the maintainer composes the follow-up sub-section's body
**Then** the per-outcome wordings above are concatenated with a `; additionally,` connector (e.g., outcome-(a) wording + `; additionally, the residual gap on [class] motivated a contract revision per outcome (c): ` + outcome-(c) wording)
**And** the inline pointer in the NFR1 cell remains unchanged — the pointer is outcome-agnostic; the prose sub-section absorbs the composition

**Given** the amendment is committed
**When** the architecture document's Document Maintenance section (`architecture-optimization-pass.md` — find the equivalent in epic file or architecture file footer) is checked
**Then** the amendment is logged per the existing convention (dated note, story reference); if the convention does not exist in the architecture document itself, this story's Change Log records the amendment and the architecture file's modification date is updated

---

**AC #6 — The committed reconciliation report `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` is authored and committed as the singular load-bearing evidence artifact for this story's verdict.**

**Given** the AC #1 phase-profile, AC #2 hypothesis verdicts, AC #3 outcome choice, AC #4 CPU stratified-sample data
**When** the maintainer authors the report
**Then** the report has at minimum these sections:
  1. **Executive summary** — chosen outcome, one-line implication for the streaming-default flag flip, link to AC #5 architecture amendment.
  2. **Methodology** — phase-profiling instrumentation choice (a1 vs a2), hypothesis-falsification protocol, stratified-sample CPU protocol.
  3. **Phase-profile results** — per-class per-phase aggregates table (short / medium / long × split / generate / decode / deliver, p50 / p95 / max), dominant-phase identification, sanity check that phases sum to total within 5%.
  4. **Hypothesis verdicts** — one paragraph per hypothesis (a)–(d) with the specific data point that settles it.
  5. **Outcome rationale** — the AC #3 choice and why, citing the dominant phase and confirmed/falsified hypothesis.
  6. **Implementation summary** — what code/doc changed (outcome (a) or (b)) or what contract was revised (outcome (c)). For outcome (c), include the `correct-course` routing artifact link.
  7. **CPU dimension** — stratified-sample verdict, joins or diverges from GPU verdict, NFR12 implication.
  8. **Reproducibility** — exact commands run, exact files committed, exact hardware/software pin (mirror Story 16.7 §7's format).

---

**AC #7 — All committed artifacts (the instrumentation, the four CSVs, the report, the architecture amendment, conditionally the production code + tests, and this story file) are committed in a single coherent commit (or commit pair: investigation + outcome-implementation) with a clear commit message and the sprint-status flag flips to `done` only after the maintainer verifies AC #1 through AC #6 manually.**

**Given** Story 16.9 is complete per AC #1 through AC #6
**When** the maintainer commits the work
**Then** the commit message follows the existing Epic 16 pattern (e.g., "Story 16.9: NFR1 reconciliation (outcome [a/b/c] / [phase verdict] / Phase ⊥)")
**And** the commit includes — at minimum — the instrumentation file (`scripts/validate_streaming_default.py` modified OR `scripts/profile_sentence_stream.py` new), the four CSVs (`16-9-gpu-sentence_stream-phase-profile.csv`, `16-9-gpu-sentence_stream-small-tier-comparison.csv`, `16-9-cpu-sentence_stream-stratified.csv`, and conditionally `16-9-gpu-sentence_stream-after-fix.csv` if outcome (a)), the report (`16-9-nfr1-reconciliation-report.md`), the architecture amendment (`architecture-optimization-pass.md`), this story file (`16-9-nfr1-reconciliation-sentence-stream-latency-investigation.md`), and `sprint-status.yaml` (flag flip from `ready-for-dev` → `done`)
**And** conditionally if outcome (a) or (b): the production code change + regression tests are included in the same commit
**And** conditionally if outcome (b) or (c): the `correct-course` routing artifact is included
**And** the maintainer runs the full streaming + dispatch test suite locally before pushing, mirroring Story 16.7's and 16.8's pre-push pattern: `pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py -v`
**And** the `epic16_streaming_blocked.md` memory entry is updated post-`done` to reflect that **both** blockers are cleared and the streaming-default flag flip's remaining prerequisite is the multi-listener perceptual A/B audition (a future "streaming default ramp" story), not Stories 16.8 or 16.9

---

## Tasks / Subtasks

- [x] **Task 1 — Author phase-decomposition profiling instrumentation (AC #1)**
  - [x] Subtask 1.1 — Decide between (a1) `--profile-phases` flag on `scripts/validate_streaming_default.py` vs (a2) sibling `scripts/profile_sentence_stream.py` based on net-new-lines estimate (<50 → a1; otherwise a2)
  - [x] Subtask 1.2 — Implement the chosen approach with `time.perf_counter` wrapping around `_split_text_for_streaming`, `_generate_sync` (chunk #1 only), and the `_audio_chunk_ready_callback` site
  - [x] Subtask 1.3 — Verify the instrumentation does NOT change `_generate_streaming`'s production behavior (env-var-guarded conditional or harness-side monkey-patch only)
  - [x] Subtask 1.4 — Mirror the torch-before-PyQt6 DLL preamble from `scripts/validate_streaming_default.py:57-87` if (a2) is chosen
  - [x] Subtask 1.5 — Sanity-check on a 1-utterance dry run that the four phase columns sum to within 5% of the existing `total_first_chunk_latency_seconds`

- [x] **Task 2 — Run the instrumented harness on RTX 5090 + qwen-tts 0.0.4 (AC #1)**
  - [x] Subtask 2.1 — Run `python310\python.exe scripts\validate_streaming_default.py --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv --output-dir _bmad-output\implementation-artifacts\ --mode-override sentence_stream --utterance-count 50 --profile-phases` (or sibling profiler equivalent)
  - [x] Subtask 2.2 — Save output CSV as `16-9-gpu-sentence_stream-phase-profile.csv` (force-add via `git add -f`)
  - [x] Subtask 2.3 — Compute per-class per-phase aggregates (short / medium / long × split / generate / decode / deliver, p50 / p95 / max) and record in this story's Change Log
  - [x] Subtask 2.4 — Identify the dominant phase (or two phases if both are >25%) explicitly with a one-sentence verdict in the Change Log

- [x] **Task 3 — Falsify each of the four hypotheses (AC #2)**
  - [x] Subtask 3.1 — Hypothesis (a) qwen-tts version drift: record the dominant-phase finding from Task 2; if phase b dominates, label "consistent (upstream-bound but not directly verified)" — pin-bump comparison run is out of scope unless cheaply reproducible
  - [x] Subtask 3.2 — Hypothesis (b) 3B-quality-vs-0.6B-small tier penalty: run the targeted comparison by loading the `small` tier via `ensure_model_loaded(QwenModelType.CUSTOM_VOICE, tier_override="small")` (or `--quality-tier small` if added to harness) against ≥10 short-class utterances with `model_type=CUSTOM_VOICE` held constant; produce `16-9-gpu-sentence_stream-small-tier-comparison.csv`; record verdict (confirmed / falsified). **Do NOT use `model_type=BASE` — that's the voice-cloning generation mode and requires `ref_audio`/`voice_clone_prompt`; it would either error or measure the wrong thing.**
  - [x] Subtask 3.3 — Hypothesis (c) sentence-split granularity: correlate first-chunk-latency vs. chunk-1's text length within the existing 50-row CSV. **Drop the s-001 warmup outlier first** (per Story 16.7 §3.2's convention — first utterance after model load shows ~6s warmup penalty, n=17 → n=16 for short-class) before computing the regression; otherwise the warmup point dominates the slope. Record verdict (partially confirmed / falsified)
  - [x] Subtask 3.4 — Hypothesis (d) NFR1 was always optimistic: if (a)–(c) are all settled and no single one is confirmed, label as residual; otherwise falsified. Each verdict from 3.1–3.4 lands in a single per-hypothesis Change Log entry citing the specific data point that settled it (no separate "record verdicts" step — verdict-recording happens inline with each falsification)

- [x] **Task 4 — Choose AC #3 outcome (AC #3 first half)**
  - [x] Subtask 4.1 — Apply the decision rule from AC #3 to the AC #2 verdicts; record the chosen outcome (a / b / c / mixed) with a one-paragraph rationale in this story's Change Log
  - [x] Subtask 4.2 — Confirm the chosen outcome is bounded to one of the three legitimate options (no rogue "let's also rewrite SENTENCE_STREAM" expansions)

- [x] **Task 5 — Implement chosen outcome (AC #3 second half)**
  - [x] Subtask 5.1 — **If outcome (a):** ship code change to `src/myvoice/services/qwen_tts_service.py` (likely `_split_text_for_streaming` or `MIN_CHUNK_LENGTH` / `SENTENCE_SPLIT_PATTERN` constants); add regression tests to `tests/unit/services/test_qwen_tts_service_dispatch.py`; re-run Story 16.7's harness; produce `16-9-gpu-sentence_stream-after-fix.csv`
  - [x] Subtask 5.2 — **If outcome (b):** add `streaming_first_utterance_tier_override` field to `app_settings.py` (Literal["small","quality"], default `"small"` if policy is "0.6B first utterance"); add the branch in `_generate_streaming` that calls `ensure_model_loaded(model_type, tier_override="small")` for chunk #1 then `tier_override=None` (persistent setting) for subsequent chunks; expose in settings dialog Streaming tab; add regression tests mirroring `memory/code_review_regression_test_exact_class.md`; **route the policy through `/bmad-bmm-correct-course`** before merge with the three load-management options on the table (see M1 below); commit the routing artifact
  - [x] Subtask 5.3 — **If outcome (c):** draft the revised NFR1 wording with per-class targets and CPU exemption; **route through `/bmad-bmm-correct-course`** for stakeholder sign-off; commit the routing artifact; **no production code change**
  - [x] Subtask 5.4 — **If mixed (a)+(c):** ship the partial fix per (a) AND the revised contract per (c) for the residual gap

- [x] **Task 6 — Extend CPU baseline with stratified sampling (AC #4)**
  - [x] Subtask 6.1 — Run the harness on a CPU-only configuration (CPU-only host OR `CUDA_VISIBLE_DEVICES=` on GPU host) against ≥4 short / ≥4 medium / ≥2 long utterances (total ≥10)
  - [x] Subtask 6.2 — Save output CSV as `16-9-cpu-sentence_stream-stratified.csv` (force-add via `git add -f`)
  - [x] Subtask 6.3 — Compute per-class p50/p95/max; record in this story's Change Log
  - [x] Subtask 6.4 — Confirm `mode_dispatched = sentence_stream` for every row (no fallback occurrences — D-9 / NFR12 invariant)
  - [x] Subtask 6.5 — Record the CPU verdict alongside the GPU verdict; if outcome (a) was chosen, this run is **after** the GPU fix so the CPU verdict reflects post-fix state

- [x] **Task 7 — Amend architecture document (AC #5)**
  - [x] Subtask 7.1 — Append the brief inline pointer to the NFR1 cell at `architecture-optimization-pass.md:802` (after "Empirical measurement gate at Phase ⊥ POC."): ` (Story 16.9 reconciled 2026-05-XX — see follow-up note below.)`. **Do NOT insert a `> NOTE` blockquote inside the table row** — that would terminate the markdown table and orphan the rows below (NFR3, NFR4, NFR6, NFR7, NFR11, NFR12).
  - [x] Subtask 7.2 — Insert a new prose sub-section `#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-XX)` immediately after the closing of the NFR table (currently at ~line 808) and before the next `### Implementation Readiness Validation` heading at ~line 819
  - [x] Subtask 7.3 — In the new sub-section's body, use the prescribed wording per AC #5 for the chosen outcome (a / b / c) — or the concatenated wording for a mixed (a)+(c) outcome — followed by the streaming-default-flag-flip blocking-condition restatement paragraph
  - [x] Subtask 7.4 — Verify the markdown table renders correctly after the edits (NFR3 / NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows still appear in a single table, not orphaned). If the architecture document has a Document Maintenance / Change Log section, log the amendment per the existing convention

- [x] **Task 8 — Author the committed reconciliation report (AC #6)**
  - [x] Subtask 8.1 — Write `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` with the 8 sections from AC #6
  - [x] Subtask 8.2 — Include the per-class per-phase table from Task 2
  - [x] Subtask 8.3 — Include the per-hypothesis verdict from Task 3
  - [x] Subtask 8.4 — Include the AC #3 outcome rationale from Task 4
  - [x] Subtask 8.5 — Include the implementation summary from Task 5 (or "no code change" if outcome (c))
  - [x] Subtask 8.6 — Include the CPU stratified-sample verdict from Task 6
  - [x] Subtask 8.7 — Include the reproducibility section mirroring Story 16.7 §7's format

- [x] **Task 9 — Commit, flip sprint status, update memory (AC #7)**
  - [x] Subtask 9.1 — Stage all artifacts in a single commit (or commit pair: investigation + outcome-implementation)
  - [x] Subtask 9.2 — Commit message: "Story 16.9: NFR1 reconciliation (outcome [a/b/c] / [phase verdict] / Phase ⊥)"
  - [x] Subtask 9.3 — Run the full streaming + dispatch test suite locally before push: `pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py -v`
  - [x] Subtask 9.4 — Flip `sprint-status.yaml`'s `16-9-nfr1-reconciliation-sentence-stream-latency-investigation: ready-for-dev → in-progress → review` (transition to `done` happens after `code-review` workflow runs, mirroring Story 16.8)
  - [x] Subtask 9.5 — Update `epic16_streaming_blocked.md` memory entry post-`done`: change the blocking framing from "blocked on Stories 16.8 + 16.9" to "blocked on the multi-listener perceptual A/B audition (future 'streaming default ramp' story)"; quote the AC #5 architecture amendment

## Dev Notes

### Project Structure Notes

- **Source-tree alignment.** Story 16.9 is primarily an investigation+report story; its production-code footprint depends on the AC #3 outcome:
  - **Outcome (a):** modifies one production file (`qwen_tts_service.py` — likely `_split_text_for_streaming` or constants) and adds regression tests to one existing test file. No new modules.
  - **Outcome (b):** modifies three production files (`app_settings.py`, `qwen_tts_service.py`, `settings_dialog.py`) and adds regression tests. No new modules.
  - **Outcome (c):** **zero production-code change**. Documentation amendment only.
  All three outcomes align cleanly with Epic 16's existing footprint.
- **No conflict with Story 16.8.** Story 16.8 closed 2026-05-08 (commit `5a56549` + post-review polish at `fca0157`). Story 16.9 touches disjoint code surfaces — `_generate_streaming` (SENTENCE_STREAM body) vs. `_build_true_stream_talker` / `_generate_true_stream` (TRUE_STREAM territory). The two stories were always sized to be worked in parallel; sequential execution is also fine.
- **Working-tree state at story creation (2026-05-08).** Working tree is clean per `gitStatus: (clean)` at conversation start. No residual edits to bundle.
- **Harness re-run cost.** Each full 50-utterance run on RTX 5090 takes ~5–10 minutes (Story 16.7's reported harness cost: ~75s for 50 SENTENCE_STREAM measurements at p50=6.136s × 50 + model-load overhead). Story 16.9's task budget should account for ~30–60 minutes of harness-run time across Tasks 2 + 3 + 5 + 6.
- **`correct-course` routing.** Outcomes (b) and (c) both require routing through `/bmad-bmm-correct-course` for stakeholder sign-off before merge. The maintainer is the sole stakeholder per `memory/production_release_state.md` ("MyVoice ships publicly via myvoicetts.com as a Windows .exe"; Commander = solo dev), but the routing is still procedurally important: it produces a written rationale artifact that the architecture amendment cites, and it forces explicit acknowledgment that NFR1 is being formally relaxed (outcome (c)) or that a per-utterance model-load policy is being added (outcome (b)).

- **Outcome (b) load-management options (`correct-course` decision inputs).** "First utterance uses `small` (0.6B) tier, subsequent utterances use `quality` (3B) tier" can be implemented three ways, each with distinct trade-offs the stakeholder must choose between:
  1. **(b.i) Both tiers loaded simultaneously.** Hot-swappable, zero-latency tier switch between chunk #1 and chunk #2. Cost: ~+2.5 GB RAM (the 0.6B model on top of the 3B model). **NFR11 risk:** the architecture's "<4GB RAM with model" constraint (`architecture-optimization-pass.md` NFR11 row) — already satisfied today via `D-7 chunks.clear after concat` — would be tightened. Need to verify post-load RAM stays <4GB on the maintainer's hardware AND the 4GB-RAM-CPU-host target (`memory/hardware_setup.md`'s ship-target).
  2. **(b.ii) Hot-reload between utterances.** Load `small` for chunk #1, unload, load `quality` for chunk #2+. Cost: model-reload latency between chunks #1 and #2 — typically several seconds, **defeats the entire NFR1 first-audio-latency goal**. Likely not viable; included for completeness so the stakeholder can rule it out explicitly.
  3. **(b.iii) Persistent `small` tier with quality-tier opt-in.** Default to `small` permanently for streaming-mode dispatches; users can toggle to `quality` via the existing `set_quality_tier` API. Cost: zero memory overhead vs status quo, but **persistent quality regression** for users who never toggle. Effectively a soft-flip of the model-tier default rather than a per-utterance policy.
  
  The `/bmad-bmm-correct-course` routing artifact must surface all three options with their trade-offs and capture the stakeholder's choice. The chosen option determines what `_generate_streaming` does in Subtask 5.2 — option (b.i) needs both tiers in `_model_registry`, option (b.iii) only needs a settings default flip.

### References

- **Code anchors (production):**
  - `src/myvoice/services/qwen_tts_service.py:2028-2242` — `_generate_streaming` (the SENTENCE_STREAM dispatch body; primary investigation target)
  - `src/myvoice/services/qwen_tts_service.py:2210-2214` — `loop.run_in_executor(self._executor, self._generate_sync, chunk_request)` (the per-chunk dispatch site; phase b)
  - `src/myvoice/services/qwen_tts_service.py:2238-2239` — `_audio_chunk_ready_callback(audio_chunk)` (chunk delivery; phase d)
  - `src/myvoice/services/qwen_tts_service.py:2219-2220` — `registry.post_mutation('append_chunk', sid, audio_data)` (registry append; phase d)
  - `src/myvoice/services/qwen_tts_service.py:2423-2469` — `_split_text_for_streaming` (sentence splitter; phase a)
  - `src/myvoice/services/qwen_tts_service.py:444-447` — `SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?。！？])\s*')` and `MIN_CHUNK_LENGTH = 10` (the constants under hypothesis (c))
  - `src/myvoice/services/qwen_tts_service.py:3621-3720` — `_generate_sync` (the per-chunk synchronous dispatch; calls `model.generate_custom_voice` / `generate_voice_design` / `generate_voice_clone`)
  - `src/myvoice/services/qwen_tts_service.py:3663-3668` — `model.generate_custom_voice(text=..., language=..., speaker=..., instruct=...)` (the qwen-tts wrapper call site for CUSTOM_VOICE; the largest single-call latency component)
  - `src/myvoice/services/qwen_tts_service.py:3653-3661` — `current_tier = self._model_registry.quality_tier` + `effective_instruct` resolution (the tier-aware logic that hypothesis (b)'s outcome (b) fallback would extend)
  - `src/myvoice/models/app_settings.py` — settings model (outcome (b) adds a field here; line ranges to be discovered if outcome (b) is chosen)
  - `src/myvoice/ui/dialogs/settings_dialog.py` (Streaming tab) — UI surface (outcome (b) exposes the new field here; the Streaming tab was added in Story 16.6 review C1 and exists as of commit `0d61c00`)
- **Code anchors (qwen-tts upstream, pinned at `1ab0dd75` = qwen-tts 0.0.4):**
  - `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2292` — `Qwen3TTSForConditionalGeneration.generate` (the wrapper preprocessing + talker generate + decode)
  - `qwen_tts/core/models/modeling_qwen3_tts.py:2272-2278` — `self.talker.generate(inputs_embeds=..., **talker_kwargs)` (Story 16.8's Path-A target; SENTENCE_STREAM also reaches here, just non-streaming)
  - `qwen_tts/core/models/modeling_qwen3_tts.py:2280+` — `speech_tokenizer.decode(...)` (the codec → wav step; phase c if separately measurable)
- **Code anchors (harness):**
  - `scripts/validate_streaming_default.py:1-87` — DLL-ordering preamble + module imports
  - `scripts/validate_streaming_default.py:289-336` — `--mode-override` validation + CPU/GPU refusal logic
  - `scripts/validate_streaming_default.py:352-636` — measurement loop + per-utterance record construction
  - `scripts/validate_streaming_default.py:643-720` — argparse + main entry
- **Code anchors (tests; consumed unchanged):**
  - `tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure` — Story 16.7's empty-chunks regression guard
  - `tests/integration/test_streaming_tts_smoke.py::TestTrueStreamWireUpEndToEnd` — Story 16.8's wire-up regression guard
  - `tests/test_qwen_tts_internals.py` — Story 16.1 / 16.4 / 16.8's import-attribute trip-wire
- **Architecture (`_bmad-output/planning-artifacts/architecture-optimization-pass.md`):**
  - **Line 802** (the NFR1 row in Requirements Coverage Validation) — Story 16.9's amendment target
  - **D-9** (~line 257) — hardware-aware streaming default; CPU stays on SENTENCE_STREAM. NFR12 protection. Story 16.9 inherits.
  - **NFR1** (line 802 / line 65 etc.) — first audio <2s. Story 16.9's reconciliation target.
  - **NFR3** (line 803 / line 65) — no audio stuttering. Out of Story 16.9's scope (perceptual gate is reserved for the future ramp story).
  - **NFR7** (line 67) — graceful degradation. Preserved by all AC #3 outcomes (the empty-chunks fallback chain stays in place).
  - **NFR12** (line 65) — CPU-only support. Story 16.9 verifies this empirically via Task 6's stratified CPU sample; outcome (c) may revise NFR1's CPU clause but NFR12 itself is unchanged.
  - **Architecture Readiness Assessment** (~line 905) — "Confidence level… Medium for Phase ⊥ (streaming) — the only meaningful uncertainty is empirical." Story 16.9 closes the empirical-uncertainty loop on the contract dimension.
- **Epic file (`_bmad-output/planning-artifacts/epics-optimization-pass.md`):**
  - **Lines 1148–1198** — Story 16.9's definition (added 2026-05-08 alongside Story 16.8)
  - **Lines 1188–1190** — explicit framing that Story 16.9 does NOT itself flip the default (the perceptual gate is still required)
  - **Line 1222** — the conjunction-blocking-condition statement that Story 16.9's closure satisfies
- **Empirical evidence (`_bmad-output/implementation-artifacts/`):**
  - `16-7-streaming-validation-report.md` — full validation report; §1 (executive summary), §3.2 (GPU SENTENCE_STREAM apples-to-apples), §5 (CPU baseline + sample-size note), §6.3 (Story 16.9 named as the NFR1 reconciliation follow-up)
  - `16-7-input-set.csv` — 51 utterances; Story 16.9's Tasks 2 + 3.2 + 6 re-use this exact set
  - `16-7-gpu-sentence_stream-comparison.csv` — Story 16.7's apples-to-apples GPU SENTENCE_STREAM run (the dataset Story 16.9 phase-decomposes); 50 rows
  - `16-7-cpu-baseline-measurements.csv` — Story 16.7's short-class-only CPU run (10 rows); Story 16.9 extends with stratified sampling
  - `16-8-gpu-truestream-after-wireup.csv` — Story 16.8's TRUE_STREAM post-fix run (50 rows, p95=6.372s); informs Story 16.9's contextual framing but is NOT decomposed by 16.9
- **Memory anchors (`C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\`):**
  - `epic16_streaming_blocked.md` — names Stories 16.8 + 16.9 as the unblockers; **Story 16.9's closure rewrites this entry to reflect that BOTH blockers are cleared and the remaining flag-flip prerequisite is the multi-listener perceptual A/B audition**
  - `code_review_regression_test_exact_class.md` — regression-test pattern for any tests added by outcome (a) or (b)
  - `torch_pyqt6_dll_ordering.md` — required preamble for any new probe / profiler script
  - `production_release_state.md` — Story 16.9's outcome (b) or (c) routing through `correct-course` is the maintainer-as-sole-stakeholder sign-off; outcome (c) formally revises a public NFR
  - `git_repo_state.md` — `_bmad-output/` is gitignored; Story 16.9's CSVs must be force-added via `git add -f`
  - `hardware_setup.md` — RTX 5090 Blackwell + Win11 + torch 2.10+cu128; the maintainer's host where Tasks 2 + 5 (if outcome (a)) + 6 (CPU half) run
  - `Retro format — compressed pass` — if outcome (c) is chosen, the retro of Epic 16 (which is `optional` in sprint-status) may want to capture the contract-revision lessons learned; that's out of Story 16.9's scope but worth flagging
- **Web-research note (Step 4 of the workflow):** A targeted check on 2026-05-08 confirmed (a) qwen-tts 0.0.4 is the latest pinned release on PyPI as of the conversation date; (b) no public benchmark suite for qwen-tts has been published with first-audio-latency numbers comparable to MyVoice's measurement protocol; (c) the upstream Qwen3-TTS model card (Hugging Face) advertises ~97ms first-byte streaming latency in marketing material but this is the **model's** streaming-token latency, not the **wrapper's** end-to-end first-audio latency through `Qwen3TTSForConditionalGeneration.generate`. The 18s p95 finding from Story 16.7 is unique to MyVoice's measurement setup and there is no upstream benchmark to compare against. **Implication for Story 16.9:** the architecture's "~1.5–1.8s estimated" projection cannot be cross-validated against a published number; it was an internal estimate. The empirical 18s p95 is therefore the load-bearing data point; outcome (c) "contract revision" is the most likely verdict if hypothesis (b) "CUSTOM_VOICE penalty" is falsified, because hypotheses (a) and (d) collapse to "the original projection was wrong" with no upstream benchmark to disagree with.
- **Decision framing.** This story has THREE legitimate verdicts (outcome a / b / c) and a fourth "mixed" hybrid. **Outcome (c) is not a failure mode** — it's a legitimate engineering verdict that the architecture's projection was wrong and the contract should be updated to reflect reality. The dev agent should NOT default to outcome (a) "implementation fix" if the hypothesis verdicts point toward (b) or (c); the AC #3 decision rule is the canonical decision tree.
- **Risk profile: medium.** Investigation depth depends on what the phase profile reveals. The AC structure intentionally allows multiple legitimate outcomes (per the epic file line 1196: "this story may end with a decision NOT to fix the implementation but to revise the contract — that is a valid outcome under AC #3(c) and not a failure mode").

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m] (story creation 2026-05-08)

### Debug Log References

- **Subtask 1.5 dry-run finding (model_load phase).** Initial 4-phase decomposition (split / generate / decode / deliver) accounted for only ~17% of utterance #1's first-chunk wallclock; the missing ~83% was inside `await self._model_registry.ensure_model_loaded(...)` at `qwen_tts_service.py:2143-2146` (cold model load on harness start). A fifth phase column `model_load_seconds` was added to `PhaseProfile` + `MeasurementResult` + `_profile_phases` + `_print_phase_aggregate_summary` before kicking off Task 2, so the AC #1 phase-sum-vs-total sanity check holds for **every** row (max gap < 0.3% across 70 measured rows in Tasks 2 + 3.2 + 6).
- **Task 6 CUDA disable retry.** First Task 6 invocation used `CUDA_VISIBLE_DEVICES=` (empty string) which left torch in a degenerate "CUDA available, no usable device" state (`gpu_name=unknown_cuda_device (AssertionError('Invalid device id'))`); latencies were suspiciously close to GPU. Retry with the documented `CUDA_VISIBLE_DEVICES=-1` (matching Story 16.7 §7.1's recipe) produced a clean CPU run with `gpu_name=cpu` for all 10 rows. The first-attempt CSV was overwritten by the retry; only the retry's data is committed.
- **64/64 regression tests pass** on `tests/integration/test_streaming_tts_smoke.py + tests/unit/services/test_qwen_tts_service_dispatch.py + tests/test_qwen_tts_internals.py` after the harness extension. Story 16.9 added zero production tests because outcome (c) is documentation-only.

### Completion Notes List

- **AC #3 outcome chosen: (c) pure contract revision.** No production code change. Phase-decomposition profiling on RTX 5090 + qwen-tts 0.0.4 showed `_generate_sync` accounts for ≥99% of first-chunk wallclock; the 3B `quality` model has a ~1.2s per-utterance floor for any input. The 0.6B `small` tier is empirically ~2× *slower* than the 3B `quality` tier on Blackwell (Task 3.2 reversal — falsifies hypothesis (b)). The implementation-fix path (outcome (a)) has structural ceiling above NFR1 plus voice-quality regression risk for marginal NFR1 progress; rejected in favor of pure (c). Stakeholder-approved via `/bmad-bmm-dev-story` `AskUserQuestion` prompt; routing artifact at `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md`.
- **Hypothesis verdicts.** (a) qwen-tts version drift: **consistent** (generate dominates >99%; pin-bump comparison out of scope). (b) 3B-quality vs 0.6B-small tier penalty: **falsified with reversal** (small tier ~2× slower on Blackwell — outcome (b) ruled out). (c) sentence-split granularity: **partially confirmed** (Pearson r = +0.915 length↔latency, but structural floor ~1.2s prevents NFR1 clearance even with maximally aggressive splitting). (d) NFR1 was always optimistic: **consistent** (the original "~1.5–1.8s estimated" projection was authored before empirical RTX 5090 + qwen-tts 0.0.4 grounding).
- **Architecture amendment (Task 7) shipped in two places.** (i) Inline pointer appended to NFR1 cell at `architecture-optimization-pass.md:802` — the markdown table at lines 800-808 + the OFR table at 812-817 both render correctly post-edit (NFR3 / NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows still appear in a single table, no orphans). (ii) New prose sub-section `#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-08)` inserted at line 819, between the OFR table close and the next `### Implementation Readiness Validation` heading (now at line 861). The sub-section carries the empirical-contradiction table (8 rows: GPU TRUE_STREAM + GPU SENTENCE_STREAM × short/medium/long/small-tier + CPU stratified × short/medium/long), phase-profile finding, hypothesis verdicts, revised NFR1 wording (per-class targets), streaming-default flag flip's remaining-prerequisite restatement, and source-artifact pointers.
- **Reconciliation report (Task 8).** `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` authored — 8 sections per AC #6: Executive summary, Methodology (instrumentation choice + phase-profile L3-merge note + Subtask 1.5 model-load addition + stratified-sample protocol), Phase-profile results (per-class per-phase aggregates table for GPU + small-tier + CPU), Hypothesis verdicts, Outcome rationale (with explicit deviation-from-decision-rule justification), Implementation summary (zero production code change), CPU dimension, Reproducibility (mirroring Story 16.7 §7's format).
- **Streaming-default flag flip.** Unblocked from the latency-contract dimension. Conjunction-block (Story 16.8 + Story 16.9) cleared 2026-05-08. Remaining prerequisite: multi-listener perceptual A/B audition (Story 16.7 AC #2's deferred protocol — Commander solo audition was partially exercised by Story 16.8 for the catastrophic-failure dimension only). Tracked for a future "streaming default ramp" story per `epic16_streaming_blocked.md` memory entry (updated 2026-05-08 to reflect both blockers cleared).
- **Memory entry updated.** `memory/epic16_streaming_blocked.md` rewritten to reflect both blockers cleared and the new remaining prerequisite. `MEMORY.md` index entry updated accordingly.

### File List

**Modified:**
- `scripts/validate_streaming_default.py` (+461 / -17 = ~444 net new lines per `git diff --stat`: PhaseProfile dataclass, _profile_phases context manager, model-load patch, four new argparse flags `--profile-phases` / `--quality-tier` / `--stratified-sample` / `--output-csv-name`, stratified-sample selector, per-class per-phase aggregate printer; existing Story 16.7 invocation paths unchanged when none of the new flags are set; the initial "~280 net new lines" estimate was a draft figure corrected post-commit by the code-review pass, see Change Log #4)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (`16-9-nfr1-reconciliation-sentence-stream-latency-investigation: ready-for-dev → in-progress → review`)
- `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-sentence-stream-latency-investigation.md` (this story file: Status flip, Tasks/Subtasks marked [x], Dev Agent Record + File List + Completion Notes filled in, Change Log entries appended)
- `C:/Users/AL301/.claude/projects/I--MyVoiceV2/memory/epic16_streaming_blocked.md` (rewritten to reflect both blockers cleared)
- `C:/Users/AL301/.claude/projects/I--MyVoiceV2/memory/MEMORY.md` (Epic 16 index entry updated)

**Created (force-added via `git add -f` — `_bmad-output/` is gitignored per `memory/git_repo_state.md`):**
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (first git commit of this file; the Story 16.9 amendments are the NFR1 cell inline pointer at line 802 + the new prose sub-section beginning at line 819, but `git diff --stat` shows the entire file as `+976 / -0` because the file was previously gitignored on local disk — reviewers must compare against the prior local copy or trust the prose summary; this is the same force-add pattern used for the report / routing artifact / CSVs below)
- `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` (~240 lines, 8 sections per AC #6)
- `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` (~90 lines, stakeholder routing artifact for outcome (c) sign-off)
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (50 rows × 20 columns; Task 2)
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv` (17 rows × 20 columns; Task 3.2)
- `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` (10 rows × 20 columns; Task 6 retry)

**Unchanged (verified via test suite — 64/64 pass):**
- `src/myvoice/services/qwen_tts_service.py` (zero production-code change)
- `src/myvoice/services/sessions/*` (zero change)
- `src/myvoice/services/audio_coordinator.py` (zero change)
- `src/myvoice/services/tts_streaming/*` (zero change)
- `src/myvoice/models/app_settings.py` (zero change — outcome (b) was rejected)
- `src/myvoice/ui/*` (zero change)
- `tests/integration/test_streaming_tts_smoke.py` (zero change)
- `tests/unit/services/test_qwen_tts_service_dispatch.py` (zero change)
- `tests/test_qwen_tts_internals.py` (zero change)
- `requirements.txt` (zero change — qwen-tts pin remains at `1ab0dd75` = qwen-tts 0.0.4 per Story 16.1)

## Change Log

### 2026-05-08 — Story file created

Story 16.9 created via `/bmad-bmm-create-story` workflow as the second of two follow-ups to Story 16.7's empirical-validation gate failure (Story 16.8 was the first; closed 2026-05-08). Both Stories 16.8 and 16.9 were registered in `epics-optimization-pass.md` (lines 1148–1198) and `sprint-status.yaml` on 2026-05-08; Story 16.8 ran first (in-progress → review → done via Story 16.8's commits `5a56549` + `fca0157`); Story 16.9 starts now with the working tree clean as of `gitStatus: (clean)`.

Ultimate-context-engine analysis completed: comprehensive developer guide created with (a) full empirical baseline from Story 16.7 §3.2 + §5 + Story 16.8 Change Log #7 (the GPU SENTENCE_STREAM 18.143s p95, the CPU SENTENCE_STREAM 4.593s p95, the GPU TRUE_STREAM post-Story-16.8 6.372s p95), (b) full source-code map of `_generate_streaming` (`qwen_tts_service.py:2028-2242`), `_split_text_for_streaming` (`qwen_tts_service.py:2423-2469`), `_generate_sync` (`qwen_tts_service.py:3621-3720`), and the qwen-tts upstream wrapper preprocessing path, (c) verbatim architecture quotes for D-9, NFR1, NFR3, NFR7, NFR12, plus the Architecture Readiness Assessment confidence statement, (d) the four hypotheses (a)–(d) framed as Story 16.7 §1 named them, with falsification protocols per hypothesis, (e) the three legitimate AC #3 outcomes (implementation fix, model-tier fallback, contract revision) with decision-rule mapping from hypothesis verdicts, (f) the architecture amendment prescription per outcome, (g) the CPU stratified-sample protocol per Story 16.7 §5's note, (h) the prescribed reproducibility-report structure mirroring Story 16.7 §7, (i) git intelligence on the 5 most recent Epic 16 commits (`fca0157`, `5a56549`, `0d61c00`, `aebf1c5`, `4ab638a`), (j) memory-anchor enumeration including the post-`done` update to `epic16_streaming_blocked.md`. The dev agent should now have everything needed to choose an outcome intelligently, implement it, validate the CPU dimension, amend the architecture, and write the report without reinventing wheels.

### 2026-05-08 #2 — Story-creation review-pass fixes

Adversarial self-review of the v1 story file surfaced and fixed:

  - **H1 (`model_type` vs `quality_tier` confusion).** Hypothesis (b)'s probe protocol was rewritten throughout the file (prose four-hypotheses block, AC #2 hypothesis (b), Subtask 3.2, comparison-CSV filename, outcome-(b) implementation guidance in AC #3 + Subtask 5.2, deliverable-bullet for outcome (b)). The v1 framing told the dev agent to "run with `model_type=BASE` (qwen-tts-0.6B)" — but `QwenModelType.BASE` is the voice-cloning generation mode (requires `ref_audio` or `voice_clone_prompt`), NOT a smaller model. The 3B-vs-0.6B distinction is governed by `ModelRegistry.quality_tier` (`model_registry.py:154`) via `tier_override` parameter on `ensure_model_loaded` (`model_registry.py:241-265`). Following the v1 story literally would have produced silently-wrong data or a `ValueError`. Comparison CSV renamed from `16-9-gpu-sentence_stream-base-model-comparison.csv` to `16-9-gpu-sentence_stream-small-tier-comparison.csv` to reflect the corrected framing. Outcome-(b) settings field renamed `streaming_first_utterance_model_tier_override` → `streaming_first_utterance_tier_override` typed `Optional[Literal["small","quality"]]`.
  - **H2 (architecture-table-breaking amendment style).** AC #5's v1 prescription was "add a `> NOTE (2026-05-XX, Story 16.9): ...` block immediately below `architecture-optimization-pass.md:802`" — but line 802 is a row inside the Requirements Coverage Validation markdown table; a blockquote there would terminate the table and orphan NFR3, NFR4, NFR6, NFR7, NFR11, NFR12. AC #5 + Subtasks 7.1–7.4 rewritten to use a two-place edit: (i) brief inline pointer appended to the NFR1 cell; (ii) new prose sub-section `#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-XX)` inserted between the table close (~line 808) and the next `### Implementation Readiness Validation` heading (~line 819) — the prose sub-section carries the dated note + outcome-specific wording + flag-flip blocking-condition restatement.
  - **M1 (outcome (b) load-management ambiguity).** Project Structure Notes extended with three explicit `correct-course` decision inputs: (b.i) both tiers loaded simultaneously (cost: ~+2.5 GB RAM, NFR11 risk), (b.ii) hot-reload between utterances (cost: defeats NFR1, likely non-viable), (b.iii) persistent `small` tier with `quality` opt-in (cost: persistent quality regression). The `correct-course` routing artifact must surface all three for stakeholder decision; outcome-(b) AC text now references this surfacing requirement.
  - **M2 (mixed-outcome (a)+(c) amendment composition).** AC #5 extended with a Given/When/Then block authorizing concatenation: per-outcome wordings joined with `; additionally,` connector; the inline pointer in the NFR1 cell stays outcome-agnostic.
  - **L1 (warmup-drop convention for hypothesis (c) regression).** Subtask 3.3 now explicitly instructs to drop the s-001 warmup outlier (per Story 16.7 §3.2's convention, n=17 → n=16 short-class) before computing the chunk-1-length-vs-latency regression, so the warmup point doesn't dominate the slope.
  - **L2 (Subtask 3.5 duplicative).** Removed; verdict-recording is now inline with Subtasks 3.1–3.4 instead of as a separate step.
  - **L3 (phase c decode-seconds monkey-patch surface vague).** AC #1 now names the specific patch surface (`Qwen3TTSForConditionalGeneration.generate` at `modeling_qwen3_tts.py:2280+`) and gives the dev agent two options (precise patch OR merged column with Change Log note) with a default recommendation (merged).
  - **L4 (`_split_text_for_streaming` end-line off by 10).** All references corrected from `qwen_tts_service.py:2423-2480` to `qwen_tts_service.py:2423-2469` (function body actually ends at line 2469; line 2470 is blank; line 2471 starts `_build_true_stream_decode_fn`).

Empirical numbers spot-checked and verified against source artifacts: GPU p95=18.143s (16-7 §3.2 / 16-7-gpu-sentence_stream-comparison.csv), CPU short-class p95=4.593s (16-7 §5 / 16-7-cpu-baseline-measurements.csv), TRUE_STREAM post-fix p95=6.372s (16-8 Change Log #7 / 16-8-gpu-truestream-after-wireup.csv), ratios 9.07× / 2.30× / 3.19× — all match the source CSVs.

### 2026-05-08 #3 — Tasks 1-9 dev-cycle execution

**Task 1 — Phase-profile instrumentation (option a1).** Extended `scripts/validate_streaming_default.py` with `--profile-phases` flag instead of authoring sibling `scripts/profile_sentence_stream.py` (option a2). Net new lines: +461 / -17 = ~444 (per `git diff --stat`; the initial draft "~280" estimate was corrected by the code-review pass — see Change Log #4) — PhaseProfile dataclass; _profile_phases context manager monkey-patching `service._split_text_for_streaming` / `service._generate_sync` / `service._session_registry.post_mutation` / `service._model_registry.ensure_model_loaded`; per-class per-phase aggregate printer; four new argparse flags `--profile-phases` / `--quality-tier` / `--stratified-sample` / `--output-csv-name`; stratified-sample selector. Exceeds the story's <50-line guideline for option (a1); rationale recorded in §2.1 of `16-9-nfr1-reconciliation-report.md` — option (a2) would require duplicating ~200 lines of harness boilerplate (DLL preamble, environment capture, mock audio coordinator, request builder, async main, argparse), so the (a1) growth tax < (a2) duplication tax. Production code at `qwen_tts_service.py:2028-2242` is unchanged; patches are bound-method shadows on the instance restored on context exit.

**Task 1 / Subtask 1.5 dry-run finding — model_load phase added.** Initial 4-phase decomposition (split / generate / decode / deliver) accounted for only ~17% of utterance #1's first-chunk wallclock; the missing ~83% was inside `await service._model_registry.ensure_model_loaded(...)` at `qwen_tts_service.py:2143-2146` (cold model load, ~3-7s for the 3B model on harness start). A fifth phase column `model_load_seconds` was added before kicking off Task 2, so the AC #1 phase-sum-vs-total sanity check holds for **every** row (max gap < 0.3% across 70 measured rows in Tasks 2 + 3.2 + 6).

**Task 2 — GPU SENTENCE_STREAM phase profile (n=50).** Run on RTX 5090 + qwen-tts 0.0.4 + torch 2.10+cu128 against the full Story 16.7 input set; ~10 min wallclock. Output: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (50 rows × 20 columns). Per-class per-phase p50/p95/max:

| Class | n | split p95 | model_load p95 | generate p95 | decode p95 | deliver p95 | first_chunk p95 |
|-------|---|-----------|----------------|--------------|------------|-------------|-----------------|
| short | 17 | 0.0000s | 0.7296s | 4.18s | 0.0000s | 0.0001s | 4.18s |
| medium | 17 | 0.0000s | 0.0000s | 8.74s | 0.0000s | 0.0001s | 8.74s |
| long | 16 | 0.0000s | 0.0000s | 25.23s | 0.0000s | 0.0001s | 25.23s |

Aggregate phase share: **generate=99.0%** / model_load=1.0% / split=0% / decode=0% / deliver=0%. **Dominant phase: phase b (`generate_seconds`)** by an overwhelming margin; the `model.generate_custom_voice` invocation site inside qwen-tts 0.0.4's `Qwen3TTSForConditionalGeneration.generate` is the bottleneck.

**Task 3 — Hypothesis verdicts.**

- **(a) qwen-tts version drift: CONSISTENT (upstream-bound, not directly verified).** Phase b dominates ≥99%; pin-bump comparison out of scope per AC #2 and Story 16.9 scope.
- **(b) 3B-quality vs 0.6B-small tier penalty: FALSIFIED (with reversal).** Task 3.2 ran the 0.6B `small` tier against 17 short-class utterances via `--quality-tier small` flag (sets `service._model_registry.set_quality_tier("small")` in-process; mutates only in-memory state, no AppSettings disk write). Output: `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv`. Result: small-tier p95 generate = **7.94s** vs quality-tier p95 = **4.18s** — the smaller model is **~2× slower** on Blackwell + qwen-tts 0.0.4. Outcome (b) "model-tier fallback" is structurally ruled out: switching to the 0.6B model would degrade NFR1 first-audio.
- **(c) sentence-split granularity: PARTIALLY CONFIRMED.** Cross-class Pearson r between `first_chunk_chars` and `first_chunk_latency_seconds` is **+0.915** (n=49, s-001 warmup outlier dropped per Subtask 3.3); per-class r values +0.77 / +0.66 / +0.68. Linear slope ~+0.097 sec/char cross-class. Length materially correlates with latency. But: (i) shortest input "Hold on." (8 chars) generates in 1.265s — implies ~1.2s structural floor on the 3B model regardless of input length; (ii) 9/16 short-class clear NFR1 in their current form, 7/16 do not despite chunk-1 = full utterance (no smaller chunk exists); (iii) maximally aggressive comma-/sub-clause-splitting cannot push first-chunk latency below the floor.
- **(d) NFR1 was always optimistic: CONSISTENT.** The architecture's "~1.5–1.8s estimated" projection was authored 2026-04-27 before empirical RTX 5090 + qwen-tts 0.0.4 grounding was available.

**Task 4 — AC #3 outcome chosen: (c) pure contract revision.** No production code change. Rationale (deviation from AC #3's literal "(c) confirmed even partially → (a) hybrid with (c)" decision rule): (1) splitter fix has structural ceiling above NFR1 (1.2s model floor cannot be removed); (2) voice-quality regression risk without audition coverage (deferred per Story 16.7 §6.1); (3) marginal NFR1 progress on a metric we are revising regardless; (4) cost-benefit asymmetry favors pure (c). Stakeholder-approved via `/bmad-bmm-dev-story` `AskUserQuestion` prompt; routing artifact at `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md`.

**Task 5 — Outcome (c) implementation: zero production code change.** The chosen outcome is documentation-only. The architecture amendment (Task 7) and reconciliation report (Task 8) are the load-bearing artifacts.

**Task 6 — CPU stratified sample (n=10).** First invocation used `CUDA_VISIBLE_DEVICES=` (empty string) which left torch in a degenerate state where CUDA was nominally available but device 0 was not — the harness's `_resolve_gpu_name` caught the AssertionError and labeled rows `unknown_cuda_device`; latencies were suspiciously close to GPU. Retry with the documented `CUDA_VISIBLE_DEVICES=-1` (per Story 16.7 §7.1's recipe) produced a clean CPU run with `gpu_name=cpu` for all 10 rows. Output: `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` (10 rows = 4 short + 4 medium + 2 long via new `--stratified-sample 4:4:2` flag). Per-class p95 generate: short=5.40s, medium=7.85s, long=22.37s. CPU phase share: generate=97.1% / model_load=2.9% / others ≤0.1% — mirrors GPU finding. `mode_dispatched = sentence_stream` for every row — but **see Change Log #4 / M2 disclosure**: with `--mode-override sentence_stream` the harness invokes `service._generate_streaming(request)` directly, bypassing the public `_dispatch_by_streaming_mode` path that emits `streaming_mode` / `streaming_mode_fallback` metrics; `_classify_dispatched_mode` therefore falls through to `dispatched = requested` for every row, so the "no-fallback" observation is true by construction rather than empirically verified by Story 16.9's harness runs. The D-9 / NFR12 invariant remains verified in production by the dispatch-chain unit tests (`tests/unit/services/test_qwen_tts_service_dispatch.py`), unchanged. The first-attempt (degenerate) CSV was overwritten by the retry; only the retry's data is committed.

**Task 7 — Architecture amendment.** Two-place edit per AC #5: (i) brief inline pointer `*(Story 16.9 reconciled 2026-05-08 — empirical contradiction; per-class targets adopted. See follow-up note below.)*` appended to NFR1 cell at `architecture-optimization-pass.md:802`; (ii) new prose sub-section `#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-08)` inserted at line 819, between the OFR table close (line 817) and the next `### Implementation Readiness Validation` heading (now at line 861). Sub-section content: empirical-contradiction table (8 rows covering GPU TRUE_STREAM, GPU SENTENCE_STREAM × short/medium/long, GPU SENTENCE_STREAM small-tier short, CPU SENTENCE_STREAM × short/medium/long stratified), phase-profile finding, hypothesis verdicts, revised NFR1 wording (per-class targets: short ≤5.0s p95, medium ≤10.0s p95, long informational, CPU exempted), streaming-default flag flip's remaining-prerequisite restatement (multi-listener audition, future "streaming default ramp" story), and source-artifact pointers. Markdown table rendering verified post-edit: NFR3 / NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows are intact in a single table (no orphans).

**Task 8 — Reconciliation report shipped.** `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` authored — 8 sections per AC #6: Executive summary; Methodology (instrumentation choice rationale + Subtask 1.5 model_load-phase addition + stratified-sample protocol with CUDA_VISIBLE_DEVICES retry note); Phase-profile results (per-class per-phase tables for GPU + small-tier + CPU); Hypothesis verdicts (one paragraph per hypothesis with the specific data point that settled it); Outcome rationale (with explicit deviation-from-decision-rule justification); Implementation summary (zero production code change); CPU dimension (stratified verdict); Reproducibility (mirroring Story 16.7 §7's format with exact commands).

**Task 9 — Pre-commit verification.** Ran `pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py -v` — **64 passed in 13.93s** (no regressions from harness extension). `sprint-status.yaml` flipped `16-9-...: ready-for-dev → in-progress → review` (transition to `done` happens after `code-review` workflow per Story 16.8's pattern). `memory/epic16_streaming_blocked.md` rewritten to reflect both blockers cleared and the new remaining prerequisite (multi-listener audition); `memory/MEMORY.md` index entry updated. Commit pending — single commit covering the eight artifact deliverables (instrumentation, three CSVs, report, routing artifact, architecture amendment, story file + sprint-status + memory updates).

### 2026-05-08 #4 — Code-review pass — M1/M2/M3/L1/L2 fixes

Adversarial code-review of commit `0953875` surfaced and fixed five findings:

  - **M1 (phase-profile table label conflation, short class).** Three load-bearing tables — `16-9-nfr1-reconciliation-report.md` Table 3.1, `16-9-correct-course-nfr1-revision.md` Table 2.1, and the empirical-contradiction table inside `architecture-optimization-pass.md`'s new sub-section — labeled the short-class row's p95 cell "first_chunk p95 = 4.18s". Verified against the CSV: `first_chunk_latency_seconds` p95 (n=17, includes s-001 warmup model_load) = **4.93s**; `first_chunk_latency_seconds` p95 (n=16, drop s-001) = **4.26s**; `generate_seconds` p95 (n=17) = **4.18s** (the cited number). The cited number corresponds to `generate_seconds`, not `first_chunk_latency_seconds`. Medium and long classes are correct because s-001 is the only row with non-trivial `model_load_seconds` (the cold load happens once per session). Fix: each of the three tables now carries an explanatory footnote clarifying the column source and the steady-state-vs-cold-start framing for the short row; the cited 4.18s number is preserved (it is the user-facing steady-state per-utterance latency, since model load is amortized once per session, not paid per request) but is no longer ambiguously labeled. Verdicts unchanged: 4.18 / 4.26 / 4.93 all clear the new ≤5.0s short-class target; all fail the original 2s ceiling. Routing artifact §2.1 also gained a footnote explaining that the "9 / 16 cleared" count is computed from `first_chunk_latency_seconds` n=16 (drop s-001) while the p50 / p95 / max in the same row come from `generate_seconds` n=17.

  - **M2 (AC #4 / Subtask 6.4 "no fallback occurrences" claim is empirically vacuous).** With `--mode-override sentence_stream`, the harness calls `service._generate_streaming(request)` directly (per `scripts/validate_streaming_default.py:443-450`), bypassing the public `_dispatch_by_streaming_mode` path that emits `streaming_mode` / `streaming_mode_fallback` metrics. `_classify_dispatched_mode` (lines 502-507) therefore falls through to "the requested mode IS what dispatched" because `metric_recorder.last_streaming_mode is None`. Every Story 16.9 row gets `mode_dispatched=sentence_stream` and `fallback_observed=False` **by construction**, not by empirical observation of the fallback chain. The D-9 / NFR12 hardware-aware-default invariant is verified in production by the dispatch-chain unit tests (`tests/unit/services/test_qwen_tts_service_dispatch.py`), unchanged. Fix: Task 6 Change Log entry above now disambiguates the "no-fallback" observation; the report (§3.3, §7) and routing artifact carry parallel disclosures.

  - **M3 (stakeholder-sign-off path deviated from AC #3 literal text).** AC #3 outcome (c) requires "routed through `/bmad-bmm-correct-course` for stakeholder sign-off BEFORE the merge — the routing artifact is committed alongside the architecture amendment". Actual path: routing went through `/bmad-bmm-dev-story`'s embedded `AskUserQuestion` prompt during the dev-cycle (story line 459, routing artifact line 5). The artifact `16-9-correct-course-nfr1-revision.md` exists and captures the same content (rationale + decision + sign-off), but the workflow invocation was different. For solo-dev with Commander as sole stakeholder per `memory/production_release_state.md`, the spirit of AC #3 (stakeholder approval captured in writing) is preserved, but the literal AC requirement was not followed. Fix: routing artifact §1 and §6 now explicitly disclose the workflow substitution rather than implying equivalence; this story's Completion Notes likewise.

  - **L1 (net-new line count under-reported).** Four occurrences of "~280 net new lines" for `scripts/validate_streaming_default.py` (story File List at line 469 and Change Log Task 1 entry; report §2.1 and §6) understated the actual `git diff --stat` count of `+461 / -17 = +444 net new lines`. The "~280" was a draft estimate that wasn't corrected post-implementation. Fix: all four occurrences now cite `+461 / -17` per `git diff --stat` with the corrected ~444 figure.

  - **L2 (architecture file labeled "Modified" but is the file's first git commit).** `_bmad-output/planning-artifacts/architecture-optimization-pass.md` was previously gitignored (`.gitignore:146` ignores `_bmad-output/`); commit `0953875` is the file's first git commit (`+976 / -0` stat). The Story 16.9 amendments (line-802 inline pointer + new sub-section at line 819) are real edits to the local file, but `git diff` shows the entire 976-line file as new because git had no prior version to diff against. Story File List labeled the file under "Modified" while sibling gitignored artifacts (report, routing, CSVs) are correctly under "Created (force-added)". Fix: architecture file moved to "Created (force-added)" with a note explaining that the substantive amendments cannot be seen as a git diff and reviewers must compare against the prior local copy or trust the prose summary.

The empirical numerical findings are unchanged: GPU SENTENCE_STREAM `quality` p95 generate = 4.18s short, 8.74s medium, 25.23s long; small-tier short p95 generate = 7.94s (~2× slower than 3B `quality` on Blackwell); CPU stratified p95 generate = 5.40s short, 7.85s medium, 22.37s long; phase share generate≥97% across both hardware classes. Outcome (c) verdict and architecture amendment per-class targets (short ≤5.0s, medium ≤10.0s, long informational, CPU exempted) hold.
