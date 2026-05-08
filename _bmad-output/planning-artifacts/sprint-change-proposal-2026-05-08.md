# Sprint Change Proposal — 2026-05-08

**Trigger:** Story 17.1 Streaming Default Ramp — perceptual A/B audition completed; verdict outcome (a) PASS per Story 17.1 AC #3.
**Routing pass:** `/bmad-bmm-correct-course` workflow invoked literally per Story 17.1 AC #4 (Epic 16 retrospective §"What Could Have Gone Better" #4 named Story 16.9's `AskUserQuestion` substitution as the non-precedent; Story 17.1 follows the literal-invocation discipline).
**Mode:** Batch (specified at invocation).
**Stakeholder:** Commander (sole maintainer per `memory/production_release_state.md`).

---

## Section 1 — Issue Summary

This is **not a problem-driven course correction**. It is a **planned story-defined milestone**: Story 17.1 conducted the deferred multi-listener perceptual A/B audition (Story 16.7 AC #2's protocol) against the post-Story-16.8 fixture, computed the verdict against the architecturally-named gate ("zero `audible_seam` flags on TRUE_STREAM pairs across all listeners"), and is routing the result through this workflow for documentary certification.

**Discovery context.** Epic 16 closed 2026-05-08 with the streaming-default flag flip's three architectural prerequisites identified: (1) TRUE_STREAM viable (Story 16.8, closed); (2) NFR1 reconciled (Story 16.9, closed); (3) perceptual gate cleared (this story). Epic 16's retrospective (`epic-16-retro-2026-05-08.md`) explicitly handed off the deferred audition to a single-story Epic 17 (Streaming Default Ramp). Story 17.1 = Epic 17 = this story.

**Evidence.** Audition data captured in `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (30 rows; 3 listeners × 10 utterances; controlled-vocabulary defect labels validated at entry time by `17-1-l1-audition-helper.py`'s input gates). Per-system defect-flag aggregates: TRUE_STREAM audible_seam = 0 across all 30 trials; SENTENCE_STREAM audible_seam = 0 across all 30 trials; both systems 30/30 `none` on every defect category. Preference: 29/30 equivalent; 1/30 SENTENCE_STREAM-preferred (L1's m-012 with note "A was quiter than B" — a volume-amplitude observation, not a defect; L2/L3 did not flag the same; informational signal only).

**Verdict per the LISTENING-INSTRUCTIONS.md gate verbatim** (`PASS if and only if zero listeners flagged audible_seam for any TRUE_STREAM pair`): **PASS — outcome (a).**

---

## Section 2 — Impact Analysis

### Epic Impact

| Epic | Status before | Status after | Change |
|---|---|---|---|
| Epic 17 (single-story Streaming Default Ramp) | `in-progress` | `done` | Single-story epic closes when Story 17.1 reaches `review`. No epic scope or AC modification. |
| Epics 11–16 | `done` | `done` | Unchanged — already closed. |

No new epics. No epic resequencing. No future-epic invalidation. The optimization-pass track (per `memory/git_repo_state.md` and the `epics-optimization-pass.md` document) closes when Story 17.1 reaches `done`.

### Story Impact

| Story | Status before | Status after | Change |
|---|---|---|---|
| Story 17.1 | `in-progress` | `review` (via Task 9) → `done` (after code-review) | Closes outcome (a) per AC #3. |

No other stories affected. No new stories created. No story scope/AC modification beyond Story 17.1's own Tasks/Subtasks marked complete during execution.

### Artifact Conflicts

| Artifact | Conflict? | Action |
|---|---|---|
| PRD (`prd.md`) | No | NFR3 ("no audio stuttering / no audible seams") is **confirmed** by the audition, not modified. No PRD edit. |
| Architecture (`architecture-optimization-pass.md`) | No conflict; documentary amendment | Two-place edit at line 803 (NFR3 row inline pointer) + new H4 sub-section after the existing Story 16.9 sub-section. Mirrors Story 16.9's pattern exactly. Performed by Story 17.1 Task 6. |
| UX (`ux-design-specification.md`) | No | The settings UI's "Streaming Mode" combobox already defaults to "Auto" (delegates to the hardware probe). No safety net pinning users to SENTENCE_STREAM exists; nothing to remove. No UX edit. |
| Tech-spec | N/A | Not present in optimization-pass artifacts (per `_bmad-output/planning-artifacts/` listing). |
| Production source (`src/myvoice/services/tts_streaming/streaming_mode.py:54-56`, `app_settings.py::streaming_mode_override`, dispatch path) | No | The TRUE_STREAM-on-CUDA hardware probe is **already** the live default since Epic 16. Outcome (a) certifies it; the "flip" is empty. No code change. |

### Technical Impact

- **No code change.** Production source-tree footprint is exactly zero. Smoke-test gate at Story 17.1 Task 8 verifies no regression in `streaming_mode.py` / `qwen_tts_service.py` dispatch / `codec_token_streamer.py` / `streaming_decoder.py` / TTS streaming unit tests.
- **No dependency change.** qwen-tts pin remains at commit `1ab0dd75` per Story 16.1.
- **No deployment / CI / IaC change.**
- **Memory artifact:** `memory/epic16_streaming_blocked.md` updated to reflect outcome (a) — flag flip certified; entry retained as historical pointer to the three-prerequisite conjunction (Stories 16.8 + 16.9 + 17.1).

---

## Section 3 — Recommended Approach

**Option 1 — Direct Adjustment** is the only viable path; Options 2 (Rollback) and 3 (MVP Review) are N/A for outcome (a):

| Option | Viable? | Rationale |
|---|---|---|
| 1. Direct Adjustment | **Yes (selected)** | Story 17.1's Tasks 6 (architecture amendment) + 7 (memory update) + 8 (smoke tests) + 9 (commit + sprint flip) execute the documentary certification. Effort: **Low**. Risk: **Low**. Timeline: same session. |
| 2. Potential Rollback | N/A | No recent stories require reverting. The audition CONFIRMS Epic 16's wired-in default, not contradicts it. |
| 3. PRD MVP Review | N/A | MVP unaffected. NFR3 is met under TRUE_STREAM (the audition is the proof). The original PRD goals stand. |

**Justification for Direct Adjustment:**
- Implementation effort is minimal (two-place markdown edit + one-paragraph memory update + a pytest run).
- Technical risk is zero (no code change shipped).
- Stakeholder alignment is total (Commander = sole stakeholder = audition listener L1 = decision maker; in-line sign-off captured in the routing artifact).
- Long-term sustainability: the audition data + routing artifact + architecture amendment together form the canonical certification record; reproducible from `17-1-perceptual-ab-results.csv` + `_perlistener_truthtable.json`.

---

## Section 4 — Detailed Change Proposals

### Change 4.1 — Architecture document amendment

**Artifact:** `_bmad-output/planning-artifacts/architecture-optimization-pass.md`

**Edit 4.1.a — NFR3 row inline pointer (line 803).**

OLD:
```
| NFR3 No audio stuttering | D-8 chunk + overlap-add with seam-quality A/B testing before flipping streaming default; sentence-stream/batch unchanged from V2 |
```

NEW:
```
| NFR3 No audio stuttering | D-8 chunk + overlap-add with seam-quality A/B testing before flipping streaming default; sentence-stream/batch unchanged from V2 *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)* |
```

**Rationale:** Mirrors Story 16.9's NFR1 pattern at line 802. Parenthetical italic appended to the cell's primary text (does NOT terminate the markdown table — verified against the rendering of the Story 16.9 amendment which used the same approach).

**Edit 4.1.b — New H4 sub-section.**

INSERTION POINT: immediately after the closing of the existing Story 16.9 sub-section (around line 861 / before the `### Implementation Readiness Validation` heading at line 863).

NEW SECTION:
```markdown
#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)

Story 17.1 (Streaming Default Ramp) executed the deferred Story 16.7 AC #2 multi-listener perceptual A/B audition against the post-Story-16.8 regenerated fixture (`16-8-perceptual-fixtures/`). Three listeners (L1 = Commander; L2, L3 = co-located in-person walkthrough listeners) each labeled all 10 utterances of the perceptual-difficult subset (`s-014/15/16/17`, `m-011/12/13/14`, `l-013/14`) per the controlled defect vocabulary in `LISTENING-INSTRUCTIONS.md`.

Per-system defect-flag count (verbatim from `17-1-perceptual-ab-results.csv` joined against `16-8-perceptual-fixtures/_perlistener_truthtable.json`):

| System | Trials | none | audible_seam | clipping | phase_artifact | tonal_distortion | other |
|---|---|---|---|---|---|---|---|
| TRUE_STREAM  | 30 | 30 | **0** | 0 | 0 | 0 | 0 |
| SENTENCE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |

Per-listener subtotals: L1 = 0/10 audible_seam (TRUE_STREAM); L2 = 0/10; L3 = 0/10. Per-utterance subtotals: 0/3 audible_seam on every utterance for every system.

**Verdict per the LISTENING-INSTRUCTIONS.md gate verbatim** (PASS iff zero listeners flagged `audible_seam` on any TRUE_STREAM pair): **PASS — outcome (a).**

**Architectural decision:** the streaming-default flag flip is **certified**. The existing `streaming_mode.py:54-56` hardware probe's TRUE_STREAM-on-CUDA default is the audited release default; no code change required (Epic 16 wired this in at Story 16.8 and the dispatch path has been live on this branch since). NFR7 graceful-degradation chain (TRUE_STREAM → SENTENCE_STREAM → BATCH) is preserved unchanged. D-9 / NFR12 hardware-aware default (CPU stays on SENTENCE_STREAM) is preserved unchanged.

**Methodology footnote:** the L2 and L3 sessions were conducted as in-person walkthroughs with Commander as scribe (listener-id arg passed to the audition helper; helper's per-utterance forced playback ensured per-pair attention before labeling). Listeners were co-located on Commander's playback hardware rather than independent setups; this is a real but lesser caveat — the architectural defense the story's N≥3 rule was designed to provide (independent per-utterance per-listener structured defect labeling) was preserved by the helper's controlled-vocabulary input gates and the listeners' independent-of-each-other answer recording.

**Informational signal:** L1 noted on m-012 that trial A (TRUE_STREAM) was perceived as quieter than trial B (SENTENCE_STREAM) — a volume-amplitude observation, not a defect (zero `audible_seam` flagged on either trial). L2 and L3 did not flag the same observation. Captured here for future hardware-aware default-tuning consideration; not actionable in this story.

Source artifacts:
- `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (30 rows; reproducibility fixture)
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (stakeholder routing artifact)
- `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/` (audible WAVs + truth-table; pre-existing from Story 16.8 regeneration; truth-table append-only edit at Story 17.1 turn 6 added L2/L3 blocks)
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (story document with Change Log #8 verdict computation table)
```

**Rationale:** Mirrors Story 16.9's H4 sub-section structure (lines 819–861). Captures the verdict, supporting data, architectural decision, methodology footnote, informational signal, and reproducibility pointers per Story 17.1 AC #5.

### Change 4.2 — Memory entry update

**Artifact:** `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md`

Updated to reflect outcome (a): "audition cleared 2026-05-08; flag flip certified; Phase ⊥-Ramp closed". Frontmatter `description:` field updated to one-line summary of the new state. Performed by Story 17.1 Task 7.

### Change 4.3 — Sprint-status updates

**Artifact:** `_bmad-output/implementation-artifacts/sprint-status.yaml`

`development_status[17-1-streaming-default-ramp]: in-progress → review` (Task 9, after smoke tests). `development_status[epic-17]: in-progress → done` (Task 9, single-story-epic closure). Final transition to `done` for Story 17.1 happens after the `code-review` workflow runs (mirrors Stories 16.8 / 16.9 procedural pattern).

---

## Section 5 — Implementation Handoff

**Scope classification: Minor.** Two-place markdown amendment + memory update + smoke test + commit. No PO/SM/PM/Architect coordination required.

**Handoff recipient:** Story 17.1 dev agent (the same dev-story execution that invoked this workflow). Resumes at Story 17.1 Task 6 after this proposal is approved.

**Deliverables expected:**
- Architecture document amendment per Change 4.1 (Task 6).
- Memory entry update per Change 4.2 (Task 7).
- Smoke-test pass (Task 8).
- Commit + sprint-status flip per Change 4.3 (Task 9).
- Story 17.1 routing artifact at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (mirroring Story 16.9's `16-9-correct-course-nfr1-revision.md` structure; produced alongside this proposal as the AC-named routing artifact).

**Success criteria:**
- All Story 17.1 ACs satisfied per its acceptance-criteria block (specifically AC #5 architecture amendment with the verbatim per-listener / per-utterance / per-system defect-flag count tables; AC #6 memory entry update; AC #7 clean smoke-test + commit).
- Story 17.1 reaches sprint-status `review`; epic-17 reaches `done`.
- No code-change leak in the commit (Subtask 9.5 verifies `git diff --stat` shows only the 4 artifact files + sprint-status.yaml).

---

## Stakeholder approval (in-line)

**Commander (sole stakeholder per `memory/production_release_state.md`):** approved 2026-05-08. Outcome (a) PASS verdict accepted; Direct Adjustment path approved for the documentary certification; no rollback or MVP review needed; methodology footnote (in-person walkthrough format) accepted as a documented limitation worth disclosing in the architecture amendment but not gate-blocking.

**Approval channel:** `/bmad-bmm-correct-course` workflow invoked literally from inside `/bmad-bmm-dev-story` (Story 17.1 AC #4 compliance — Epic 16 retro §"What Could Have Gone Better" #4 lesson honored). Batch mode per Commander's invocation args.

---

## Workflow completion log

- **Issue addressed:** Story 17.1 Streaming Default Ramp — audition outcome (a) certification.
- **Change scope classification:** Minor.
- **Artifacts modified (committed):** `architecture-optimization-pass.md` (Task 6), this proposal (`sprint-change-proposal-2026-05-08.md` — workflow native output), the AC #4 routing artifact (`17-1-correct-course-streaming-default-ramp.md`), `17-1-streaming-default-ramp.md` (the story document).
- **Artifacts modified (working / not committed):** `epic16_streaming_blocked.md` memory entry (Task 7); `sprint-status.yaml` (Task 9; tracked in git per existing precedent).
- **Routed to:** Story 17.1 dev agent for direct implementation.
- **Workflow status:** Complete.
