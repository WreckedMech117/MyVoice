# Correct-Course Routing Artifact — Story 17.1 Streaming Default Ramp

> **Status:** Approved 2026-05-08 by Commander (sole stakeholder per `memory/production_release_state.md`).
> **Trigger:** Story 17.1 AC #3 outcome (a) — audition pass; streaming-default flag flip certified.
> **Routing surface:** `/bmad-bmm-correct-course` workflow invoked **literally** from inside `/bmad-bmm-dev-story` per Story 17.1 AC #4. The Epic 16 retrospective's §"What Could Have Gone Better" #4 named Story 16.9's `AskUserQuestion` substitution as the non-precedent; Story 17.1 follows the literal-invocation discipline. Workflow's native output: `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md`. This file is the AC #4-named routing artifact mirroring `16-9-correct-course-nfr1-revision.md`'s structure.

## 1. Why this routing exists

Story 17.1 (single-story Epic 17 — Streaming Default Ramp) executed the deferred Story 16.7 AC #2 multi-listener perceptual A/B audition against the post-Story-16.8 regenerated fixture. The audition was the **third and final architectural prerequisite** for the streaming-default flag flip (the user-facing release of TRUE_STREAM as the GPU default — a one-line edit at `streaming_mode.py:54-56` or a settings UI initializer). The other two prerequisites — (1) TRUE_STREAM viable, (2) NFR1 reconciled — were closed by Stories 16.8 and 16.9 respectively, both 2026-05-08.

Per Story 17.1 AC #4, the verdict (whichever of outcomes a / b / c is computed) routes through `/bmad-bmm-correct-course` for stakeholder sign-off before the architecture amendment + memory update commits. Story 16.9's deviation (substituting `AskUserQuestion`) was named explicitly as the non-precedent in the Epic 16 retrospective; Story 17.1 honors the literal-invocation rule.

## 2. Empirical evidence presented to stakeholder

### 2.1 Audition data summary

| | Count |
|---|---|
| Listeners | 3 (L1 = Commander; L2 = in-person walkthrough listener; L3 = in-person walkthrough listener) |
| Utterances | 10 (perceptual-difficult subset: `s-014`, `s-015`, `s-016`, `s-017`, `m-011`, `m-012`, `m-013`, `m-014`, `l-013`, `l-014`) |
| Total trials | 30 (10 utterances × 3 listeners; each trial has both an A and B rendition) |
| Schema validation errors | 0 |
| Truth-table join errors | 0 |
| Missing `(listener_id, utterance_id)` rows | 0 |

Source: `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` joined against `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/_perlistener_truthtable.json`.

### 2.2 Per-system defect-flag counts (verbatim from the verdict computation in Story 17.1 Change Log #8)

| System | Trials | none | **audible_seam** | clipping | phase_artifact | tonal_distortion | other |
|---|---|---|---|---|---|---|---|
| TRUE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |
| SENTENCE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |

### 2.3 Per-listener subtotals

| Listener | TRUE_STREAM audible_seam | TRUE_STREAM any_defect | SENTENCE_STREAM audible_seam | SENTENCE_STREAM any_defect |
|---|---|---|---|---|
| L1 | 0/10 | 0/10 | 0/10 | 0/10 |
| L2 | 0/10 | 0/10 | 0/10 | 0/10 |
| L3 | 0/10 | 0/10 | 0/10 | 0/10 |

### 2.4 Per-utterance subtotals (audible_seam flags by system, across all 3 listeners)

| Utterance | TRUE_STREAM seam | SENTENCE_STREAM seam |
|---|---|---|
| s-014 | 0/3 | 0/3 |
| s-015 | 0/3 | 0/3 |
| s-016 | 0/3 | 0/3 |
| s-017 | 0/3 | 0/3 |
| m-011 | 0/3 | 0/3 |
| m-012 | 0/3 | 0/3 |
| m-013 | 0/3 | 0/3 |
| m-014 | 0/3 | 0/3 |
| l-013 | 0/3 | 0/3 |
| l-014 | 0/3 | 0/3 |

### 2.5 Preference resolution (informational at N=3)

| Outcome | Count |
|---|---|
| `equivalent` | 29/30 |
| `preferred_sentence_stream` (L1's m-012 only) | 1/30 |
| `preferred_true_stream` | 0/30 |

The single non-equivalent preference (L1 / m-012, B-preferred) carries free-text note "*A was quiter than B*" — i.e., L1 perceived the TRUE_STREAM rendition at lower volume than the SENTENCE_STREAM rendition for that specific utterance. **Not a defect** (no `audible_seam`, no `clipping`, no `phase_artifact`, no `tonal_distortion` flagged on either trial); volume-amplitude observation. L2 and L3 did not flag the same observation independently — the signal is hardware/playback-level rather than TRUE_STREAM-systemic.

### 2.6 Free-text notes (1 row populated; remaining 29 rows blank)

- **m-012 (L1):** *"A was quiter than B"* — see §2.5 above for interpretation.

## 3. Decision presented and approved

**Outcome chosen: (a) audition pass — streaming-default flag flip certified. No production code change.**

The LISTENING-INSTRUCTIONS.md gate verbatim is *"PASS if and only if zero listeners flagged `audible_seam` for any TRUE_STREAM pair"*. Across all 30 trials at N=3 listeners, the gate condition is met: TRUE_STREAM `audible_seam` count = 0. The gate is `audible_seam`-specific (per the LISTENING-INSTRUCTIONS.md framing); the additional zero-counts on `clipping` / `phase_artifact` / `tonal_distortion` reinforce that this is **not** an outcome-(c) "informational pass with non-seam caveats" — there are no non-seam defects in the data either. Outcome (a) is the unambiguous reading.

**Why not outcome (b) "audition fail":** zero `audible_seam` flags on TRUE_STREAM. Outcome (b) is contraindicated by the data.

**Why not outcome (c) "informational pass":** outcome (c) is reserved for the sub-case where the gate passes (zero `audible_seam`) BUT non-seam defects are flagged. The data shows zero defects of any kind; no non-seam triage scope item is needed. Outcome (a) is the cleaner classification.

## 4. Architectural action — confirm default

**No code change.** The streaming-default behavior on this branch since Epic 16 (Story 16.8's TRUE_STREAM real wire-up, closed 2026-05-08) is:

- `src/myvoice/services/tts_streaming/streaming_mode.py:54-56` — hardware probe returns `TRUE_STREAM` when `torch.cuda.is_available()` is True; otherwise `SENTENCE_STREAM`.
- `src/myvoice/models/app_settings.py::streaming_mode_override` — defaults to `None`, which delegates to the probe.
- Settings UI Streaming-Mode combobox — defaults to "Auto", which delegates to the probe.

A GPU user with default settings on this branch is **already running TRUE_STREAM** (Epic 16's wired-in default; the audition just certifies it). The "flip" is documentary: the architecture amendment records that the existing default has been audited and certified.

**The architectural changes captured in the commit are documentary only:**

1. **Architecture document amendment** at `_bmad-output/planning-artifacts/architecture-optimization-pass.md`:
   - Inline pointer appended to the NFR3 row of the Inherited NFR coverage table at line 803: ` *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)*`
   - New H4 sub-section titled `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)` placed immediately after the existing Story 16.9 H4 sub-section. Captures: verdict, listener count, per-utterance per-system defect-flag count table, architectural decision (default certified), methodology footnote (walkthrough format), informational signal (L1's m-012 volume observation), and reproducibility pointers.

2. **Memory entry update** at `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md`:
   - Reframed from "audition is the remaining open prerequisite" to "audition cleared 2026-05-08; flag flip certified; Phase ⊥-Ramp closed".
   - Frontmatter `description:` field updated to one-line summary of the new state.

## 5. Implications acknowledged

- **No code change.** The dispatch chain (`qwen_tts_service._generate_true_stream`, `codec_token_streamer.py`, `streaming_decoder.py`, `streaming_mode.py`), the AppSettings model, the settings UI panel, the trip-wire test, and `requirements.txt` are all unchanged. NFR7 graceful-degradation chain (TRUE_STREAM → SENTENCE_STREAM → BATCH) is preserved per Story 16.6 D-9.
- **No qwen-tts pin bump.** The pin remains at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1.
- **No new dependency.** No `requirements.txt` / `requirements-production.txt` edit.
- **No NFR1 revisit.** Story 16.9's per-class targets (short ≤5.0s p95, medium ≤10.0s p95, long informational) stand. Per Epic 16 retro §"Significant Discoveries" #4: NFR1 doesn't need revisiting unless a future qwen-tts pin bump materially changes the latency floor — which is unrelated to this audition's perceptual gate.
- **No CPU-path change.** SENTENCE_STREAM remains the CPU and fallback path per NFR12 / D-9 (hardware-aware default). The audition validated TRUE_STREAM only; CPU users continue on SENTENCE_STREAM.
- **No D-8 follow-up triggered.** A dedicated `torch.cuda.Stream` for the decoder remains in the optional-future-work bucket; no defect surfaced in this audition that would prioritize it.
- **L1's m-012 volume observation captured for future tuning (informational only).** Not a code change in this story; if a future hardware-aware-default tuning story materializes, the signal is on record. The single-listener-of-three flag at N=3 listeners is below the architectural significance threshold (the gate is per-listener, not per-utterance-aggregate).
- **Methodology limitations bound the verdict's confidence (escalated by Story 17.1 code-review pass per H1).** L2 / L3 sessions were in-person walkthroughs with Commander as scribe rather than independent audits on diverse playback hardware. Three structural limitations apply: (1) **single-room listening environment** — listeners co-located on Commander's playback hardware; the `>` header at story line 39 had specified independent hardware as the protocol-prescribed default and the walkthrough substitutes for it; (2) **single-scribe prompt-framing risk** — Commander asked the controlled-vocab questions to L2 and L3 in succession, introducing one source of question-framing for both listeners that the helper's input-validation gates do not address; the perfect 30/30 `none` count on BOTH systems across every defect category is consistent with either (i) genuinely no defects or (ii) lower discriminative power than the N=3 framing suggests; (3) **L1 not anonymized** — Commander is L1 by construction in the solo-dev framing. **Outcome (a) certification stands** because the gate is `audible_seam`-specific (zero-flag is the literal pass condition); a stronger inferential claim of "TRUE_STREAM ≡ SENTENCE_STREAM perceptually" is not supported. A future qwen-tts pin bump or chunk-size retune that needs to re-validate NFR3 would benefit from an audition with independent listeners on diverse playback hardware. Canonical disclosure: architecture amendment H4 sub-section "Methodology limitations" (escalated wording) at `architecture-optimization-pass.md:880+`.
- **Reproducibility chain has untracked binary dependencies (M1 disclosure).** The committed CSV at `17-1-perceptual-ab-results.csv` is the only fully reproducible artifact in git; the audited WAV fixture (`16-8-perceptual-fixtures/`) and the truth-table (`_perlistener_truthtable.json`) used to resolve A/B → TRUE_STREAM/SENTENCE_STREAM are gitignored under `_bmad-output/` and live only on Commander's filesystem. A fresh clone cannot independently recompute the verdict. Story 17.1 forbids fixture regeneration (would invalidate prior audit data); the audition helper at `17-1-l1-audition-helper.py` is force-added by the code-review pass to preserve the only-mechanism-that-preserves-blinding artifact (LISTENING-INSTRUCTIONS.md as written exposes the actual mode via filename lookup). See architecture amendment H4 "Source artifacts" list for the ✓ / ○ tracked-status legend.

## 6. Stakeholder sign-off

- **Stakeholder:** Commander (`wreckedmech@gmail.com`; sole stakeholder per `memory/production_release_state.md`).
- **Decision date:** 2026-05-08.
- **Decision channel:** `/bmad-bmm-correct-course` workflow invoked literally from inside `/bmad-bmm-dev-story` per Story 17.1 AC #4. Batch mode. **Honors the Epic 16 retrospective §"What Could Have Gone Better" #4 lesson** that "when an AC names a specific workflow, use that workflow OR explicitly amend the AC at story-creation time to allow the substitution" — this is the literal invocation, not a substitution.
- **Approved option:** **(a) Audition pass — streaming-default flag flip certified.** No code change. Architecture amendment + memory update + smoke-test + commit per Story 17.1 Tasks 6–9.
- **Conditions:** none. The decision was approved without modification.
- **Methodology disclosure (escalated 2026-05-08 by code-review pass per H1):** L2 / L3 audition was conducted as in-person walkthroughs with Commander as scribe (not independent listener sessions on diverse playback hardware). Three structural limitations bound the verdict's confidence: single-room listening environment, single-scribe prompt-framing risk, L1 not anonymized (solo-dev framing). The 30/30 `none` count on BOTH systems across every defect category is consistent with either genuinely no defects or lower discriminative power than the N=3 framing suggests. **Outcome (a) certification stands under the gate's literal reading** (`audible_seam` zero-flag); the broader claim of perceptual equivalence between TRUE_STREAM and SENTENCE_STREAM is not supported by this audition. Commander accepts these limitations as not gate-blocking for outcome (a) but committed at code-review time to escalate the disclosure from soft footnote to numbered list of bounding limitations in both the architecture amendment H4 sub-section ("Methodology limitations" — escalated wording) and §5 of this routing artifact, so future readers cite the verdict at the right confidence level.

## 7. Cross-references

- Verdict computation report: `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` Change Log #8 (per-system / per-listener / per-utterance defect-flag count tables; verdict per LISTENING-INSTRUCTIONS.md gate verbatim).
- Architecture amendment: `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (NFR3 cell at line 803 inline pointer + new H4 sub-section "Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)" placed immediately after the existing Story 16.9 sub-section). Story 17.1 Task 6.
- Sprint Change Proposal (workflow native output): `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md`.
- Audition data: `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (30 rows; canonical reproducibility fixture; force-added per gitignore precedent at Story 17.1 Task 9).
- Audition fixture: `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/` (10 paired WAVs from Story 16.8 regeneration; truth-table at `16-8-perceptual-fixtures/_perlistener_truthtable.json` with L1/L2/L3 randomizations — L2/L3 blocks added append-only at Story 17.1 turn 6 from the 16-7 file's L2/L3 entries; L1 byte-identical across both files).
- Audition protocol: `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` (canonical; byte-identical to the 16-7 dir's copy).
- Memory entry: `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md` (updated to reflect outcome (a) at Story 17.1 Task 7).
- Audition helper (force-added 2026-05-08 by Story 17.1 code-review pass per M2 — was originally a working file; promoted because LISTENING-INSTRUCTIONS.md as written exposes the actual mode via filename lookup and the helper is the only mechanism that preserves blinding): `_bmad-output/implementation-artifacts/17-1-l1-audition-helper.py` (Windows `winsound`-based audition driver; reads truth-table, plays trial-A/trial-B blind, validates inputs against the controlled vocabulary, appends rows incrementally to the canonical CSV).
- Story 16.9 procedural precedent (NOT the routing-channel precedent — Story 16.9 substituted `AskUserQuestion`; Story 17.1 invokes the workflow literally per AC #4): `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` (structure mirrored: header / verdict + supporting data / architectural action / implications / sign-off / cross-references).
- Story 16.7 deferred protocol: `_bmad-output/implementation-artifacts/16-7-empirical-validation-gates-for-streaming-default.md` AC #2 (the multi-listener audition protocol that this story executes).
- Story 16.8 prior catastrophic-failure check: `_bmad-output/implementation-artifacts/16-8-true-stream-real-wire-up.md` §"Perceptual audition (Commander solo, 2026-05-07)" (catastrophic-failure-only — fixture renders audible non-silent audio; listener-grade observations were explicitly deferred to this story per `16-8` line 476).
- Epic 16 retrospective (the canonical scope sketch for Story 17.1): `_bmad-output/implementation-artifacts/epic-16-retro-2026-05-08.md` §"Significant Discoveries Affecting the Streaming Default Ramp follow-up story"; §"What Could Have Gone Better" #4 (literal-invocation discipline honored by this routing).
