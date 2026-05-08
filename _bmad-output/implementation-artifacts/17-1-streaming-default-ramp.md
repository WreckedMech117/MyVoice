# Story 17.1: Multi-Listener Perceptual A/B Audition and Streaming-Default Confirmation

Status: done

> Phase ⊥-Ramp of D-20 — **single-story Epic 17** (Streaming Default Ramp), the audition-gated follow-up to Epic 16's nine stories. This story closes the only remaining open follow-up named by the Epic 16 retrospective (`epic-16-retro-2026-05-08.md` §"Significant Discoveries Affecting the Streaming Default Ramp follow-up story"): **the multi-listener perceptual A/B audition** (Story 16.7 AC #2's deferred protocol). After this story closes, the streaming-default flag flip's three architectural prerequisites — (1) TRUE_STREAM viable (Story 16.8, closed 2026-05-08), (2) NFR1 reconciled (Story 16.9, closed 2026-05-08), (3) perceptual gate cleared (this story) — are fully resolved. The Phase ⊥ track is then complete and the V2 optimization pass closes.
>
> **Why this is the right (and only) entry point of Epic 17.** The Epic 16 retro names this story explicitly at lines 167-171 with a six-bullet scope sketch — author the audition-protocol invocation against the existing fixture, collect listener observations from ≥3 listeners, compute pass/fail per the architecturally-named gate ("zero `audible_seam` flags on TRUE_STREAM pairs"), route through `/bmad-bmm-correct-course`, and either confirm the streaming default in the architecture doc or defer the ramp + name the next-trigger condition. The fixture is already built and on disk (see "Pre-existing infrastructure already verified" below); the per-listener truth-table already has L1 / L2 / L3 randomizations pre-computed; the LISTENING-INSTRUCTIONS.md is already in place. This story is fundamentally a **measurement-and-framing exercise**, not a code change — mirroring Story 16.9's outcome (c) discipline.
>
> **Net behavior change for users (zero — this story does not flip any flag).** The `streaming_mode.py:54-56` hardware probe already returns `TRUE_STREAM` when `torch.cuda.is_available()` is True, and `AppSettings.streaming_mode_override` already defaults to `None` (which delegates to the probe). A GPU user with default settings on this branch is **already running TRUE_STREAM** — Epic 16 shipped the dispatch path live. What this story produces is **certification** (pass) or **deferral framing** (fail / inconclusive) — not a code change. The retro's §1 makes this explicit: "*the actual change for the ramp story is removing any UI-level safety net that pins users to SENTENCE_STREAM by default — none exists currently.*" The "ramp" is documentary.
>
> **Pre-existing infrastructure already verified before drafting.**
>
>   - **The perceptual A/B fixture is fully built and on disk.** `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` contains 10 paired WAV files spanning the perceptual-difficult subset of the canonical input set: `s-014`, `s-015`, `s-016`, `s-017` (4 short — sibilants, articulation), `m-011`, `m-012`, `m-013`, `m-014` (4 medium — sibilants + clarity + bell tones), `l-013`, `l-014` (2 long — alliteration / sustained sibilants). Each utterance has both `{utterance_id}-A-true_stream.wav` and `{utterance_id}-B-sentence_stream.wav` files (canonical disk naming preserves the truth-table for analysis). The fixture was regenerated post-Story-16.8 with the working TRUE_STREAM forward-hook, so `*_A-true_stream.wav` files contain real audible TRUE_STREAM speech (Story 16.8 §"Perceptual audition (Commander solo, 2026-05-07)" verified non-silent rendition). **Do NOT regenerate** — that would invalidate the L1 (Commander) solo audition data Story 16.8 already collected.
>
>   - **The per-listener truth-table is pre-computed for L1 / L2 / L3.** `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/_perlistener_truthtable.json` already contains randomized A/B-vs-system mappings for all three listeners across all ten utterances (verified at story-creation time: `listeners: ['L1', 'L2', 'L3']` × 10 utterances each = 30 rows). The randomization is stable: each listener sees a different per-utterance A/B-vs-true_stream/sentence_stream mapping, so listeners cannot pattern-match on "A is always TRUE_STREAM". The maintainer joins audition CSV results against this manifest at verdict-computation time to compute per-system defect counts.
>
>   - **The audition protocol is canonical.** `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` codifies the protocol: trial A end-to-end, trial B end-to-end, headphones if available, comfortable Discord-call volume, one CSV row per utterance with `(utterance_id, listener_id, a_or_b_preferred, a_defects_observed, b_defects_observed, free_text_notes)`. The controlled defect vocabulary is fixed: `none` / `audible_seam` / `clipping` / `phase_artifact` / `tonal_distortion` / `other_describe_in_notes`. The pass/fail gate is named verbatim in the instructions: "**PASS if and only if zero listeners flagged `audible_seam` for any TRUE_STREAM pair**". Preference is informational at N=3; defect detection is the architectural concern.
>
>   - **L1's prior audition was scoped to catastrophic-failure detection, not listener-grade defect labeling.** Story 16.8 §"Perceptual audition (Commander solo, 2026-05-07)" recorded only that the fixture renders non-silent audio on both A and B for all 10 pairs (line 472: "*Both `*_A.wav` (TRUE_STREAM) and `*_B.wav` (SENTENCE_STREAM) files render audible, non-silent audio.*") and the catastrophic-failure check at line 474 ("*PASS — no silence, no full-second dropouts, no distortion observed*"). The listener-grade observations (preference, defect detection per the controlled vocabulary in `LISTENING-INSTRUCTIONS.md`) were **explicitly deferred** to this ramp story (16-8 line 476: "*Sibilant / cadence / preference observations: deferred to the future streaming-default ramp story.*"). For Story 17.1, **Commander conducts a fresh listener-grade L1 audition** following the canonical protocol — there is no "retroactive normalization" of partial data; L1 audits all 10 pairs at listener-grade detail just as L2 and L3 do.
>
>   - **The architecture amendment pattern is established.** Story 16.9's NFR1 reconciliation amendment at `architecture-optimization-pass.md:802` (inline pointer) + line 819 (new H4 sub-section) is the canonical two-place edit pattern. Story 17.1's NFR3 amendment follows the same shape: an inline pointer in the NFR3 row of the Inherited NFR coverage table at line 803 + a new H4 sub-section titled `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, YYYY-MM-DD)` placed immediately after the existing Story 16.9 sub-section (currently ending around line 853). The architecture document is gitignored at the directory level (`_bmad-output/` per `.gitignore:146`) but tracked individually since Story 16.9's `git add -f`; this story's amendment is committed via `git add -f` per the established precedent.
>
>   - **The `correct-course` routing artifact pattern is established.** `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` is the canonical structure: header (story / date / context), the verdict + supporting data table, the architectural action, Commander sign-off line. Story 17.1's routing artifact at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` mirrors this structure verbatim. **The retro's §"What Could Have Gone Better" #4 names this explicitly: Story 16.9 deviated from AC #3 by routing through `AskUserQuestion` instead of `/bmad-bmm-correct-course`; Story 17.1 must use the literal `/bmad-bmm-correct-course` workflow** (the retro's lesson: "when an AC names a specific workflow, use that workflow OR explicitly amend the AC at story-creation time to allow the substitution"). This story's AC #4 names `/bmad-bmm-correct-course` — use it literally.
>
>   - **The memory entry exists and is the canonical handoff target.** `memory/epic16_streaming_blocked.md` is currently framed as "Epic 16 streaming-default flag flip prerequisite — Stories 16.8 + 16.9 closed both blockers; flag flip's remaining prerequisite is the multi-listener perceptual A/B audition". After this story closes, the entry is updated to reflect the audition outcome (pass / fail / deferred-again). The retro's §"Action Items" line names this update verbatim: "*Update memory entry `epic16_streaming_blocked.md` post-close to reflect the audition outcome.*" Format the update as a one-paragraph append-or-replace, following the existing entry's style.
>
>   - **No production code change expected in any outcome.** The retro's scope sketch at line 169 names this verbatim: "**Do NOT add any production code change beyond what (e) requires** — the ramp is primarily an audition + framing exercise, mirroring Story 16.9's outcome (c) discipline." The `streaming_mode.py:54-56` hardware probe is already returning `TRUE_STREAM` on CUDA. The `AppSettings.streaming_mode_override` field already defaults to `None`. The settings UI's "Streaming Mode" combobox already defaults to "Auto" which delegates to the probe. There is no UI safety net pinning users to SENTENCE_STREAM. The retro's §1 confirms: "*the 'flip' is an empty operation because the resolver already returns TRUE_STREAM.*" Story 17.1's deliverable is **audition data + routing artifact + architecture amendment + memory entry update** — total ≈400-700 lines of new content across CSV + markdown + memory edits, zero source-tree edits.
>
>   - **The Story 16.9 outcome (c) discipline is the precedent for outcome-(b) here.** Story 16.9 explicitly named contract revision as a peer outcome, not a fallback. Story 17.1's AC #3 mirrors this: outcome (b) "audition fail" is a legitimate verdict that closes the story by amending NFR3's framing to defer the user-facing default-confirmation + naming a follow-up trigger. The dev agent should NOT default to "scramble for a code fix" if the audition fails. The retro's §"Key Insights" #2 articulates this: "*the 'outcome (c) is not a failure mode' framing is a critical structural escape valve.*"
>
>   - **The empirical baseline is fixed and reproducible.** `_bmad-output/implementation-artifacts/16-7-input-set.csv` (51 utterances; 10 perceptual-difficult subset) is the canonical reproducibility fixture. **Do NOT modify** — the retro's §"Significant Discoveries" #3 names this verbatim: "*Any future re-run uses this exact set.*" If the audition surfaces an observation suggesting the input set is missing a category (e.g., a tongue-twister class not represented), capture that in the routing artifact as a follow-up scope item — NOT as an in-story input-set edit.
>
>   - **NFR1's per-class targets stand unchanged.** Story 16.9's reconciliation (`architecture-optimization-pass.md:819` H4 sub-section + lines 838-850 revised wording) is the load-bearing reference. Story 17.1 does NOT revisit NFR1 — only NFR3 (the perceptual-quality dimension). The retro's §"Significant Discoveries" #4 names this verbatim: "*The ramp story does NOT need to revise NFR1 again unless a future qwen-tts pin bump materially changes the latency floor.*" If NFR1 needs revisiting, that's a separate follow-up story.
>
>   - **No qwen-tts pin bump in this story.** The pin remains at commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83` per Story 16.1. The retro's §"Significant Discoveries" #7 names the pin-bump trigger explicitly: only if the audition surfaces a need to bump the pin AND the new pin's diff cleared all five existing trip-wires + the talker forward-hook. If the audition flags a pin-bump-conditional defect, capture it in the routing artifact as a follow-up scope item.
>
> **Five-point story scope:**
>
> (a) **Recruit ≥2 additional listeners (L2, L3) and prepare audition packets.** Listener selection is friends / family / collaborators per the LISTENING-INSTRUCTIONS.md framing — the audition is not a corporate UAT. Each listener receives the fixture directory + LISTENING-INSTRUCTIONS.md + their listener ID (L2 or L3) + a results-CSV template (or instructions to send observations as plain text / spreadsheet for the maintainer to fold into the canonical CSV). Listener identity is anonymized in the committed artifact for L2/L3; Commander retains a private mapping for follow-up correspondence if a defect needs reproducing. (L1 is Commander by construction — solo-dev framing — and is not anonymized in the committed artifact.) Listeners may either (i) conduct the audition on their own playback hardware (headphones if available) — the LISTENING-INSTRUCTIONS.md protocol verbatim, reflecting realistic Discord-call usage — OR (ii) audition co-located with Commander via the `17-1-l1-audition-helper.py` walkthrough format, with Commander as scribe and the helper's controlled-vocabulary input gates enforcing label discipline. Variant (i) is the prose-named default; variant (ii) is a permitted substitution **but its three structural limitations** (single-room listening environment / single-scribe prompt-framing risk / L1 non-anonymization) **must be disclosed numerically in the architecture amendment H4 sub-section and the routing artifact §5/§6** so readers can cite the verdict at the right confidence level. **Amendment 2026-05-08 (Story 17.1 code-review pass per M3):** this header originally named only variant (i) and the walkthrough substitution at Change Log #7 was selected via runtime `AskUserQuestion` without amending the prose; the M3 finding called this out and the prose now matches what was actually done.
>
> (b) **Collect, normalize, and commit audition observations.** Compile L1 (Commander, retroactively normalized into the canonical schema from Story 16.8 §"Perceptual audition (Commander solo, 2026-05-07)"), L2, and L3 observations into a single CSV at `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` with header `utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes`. Verify ≥30 rows total (≥10 utterances × ≥3 listeners; missing rows are not allowed — if a listener skipped an utterance, that's a process failure that gates the verdict). Verify every defect-vocabulary value is from the controlled list (`none` / `audible_seam` / `clipping` / `phase_artifact` / `tonal_distortion` / `other_describe_in_notes`); reject and re-collect any out-of-vocabulary observations. Force-add the CSV via `git add -f`.
>
> (c) **Compute the verdict.** Join the audition CSV against `_perlistener_truthtable.json` to resolve each row's A/B label into the actual system rendition (TRUE_STREAM or SENTENCE_STREAM). Aggregate per-system defect counts (TRUE_STREAM defect-flag count: N out of NN; SENTENCE_STREAM defect-flag count: M out of MM, broken down per controlled-vocabulary value). Apply the LISTENING-INSTRUCTIONS.md gate verbatim: PASS if and only if zero listeners flagged `audible_seam` for any TRUE_STREAM pair across all utterances. Record the verdict in the report; record per-utterance per-listener defect detail. Choose ONE of three outcomes per AC #3.
>
> (d) **Route through `/bmad-bmm-correct-course` and commit the routing artifact.** Use the `/bmad-bmm-correct-course` workflow literally (NOT via `AskUserQuestion` substitution — the retro's §"What Could Have Gone Better" #4 names this explicitly). The routing artifact at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` mirrors `16-9-correct-course-nfr1-revision.md`'s structure: header (story / date / context), verdict + supporting data table, architectural action (confirm default / defer + name next-trigger / informational pass), Commander sign-off line. Force-add the artifact via `git add -f`.
>
> (e) **Amend the architecture document and update memory.** Two-place edit on `architecture-optimization-pass.md` per Story 16.9's pattern: inline pointer in the NFR3 row at line 803 of the Inherited NFR coverage table reading something like "*(Story 17.1 audition cleared YYYY-MM-DD — see follow-up note below.)*" + new H4 sub-section `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, YYYY-MM-DD)` immediately after the Story 16.9 sub-section. The new sub-section captures: verdict, listener count + per-utterance defect counts table, decision (default certified / deferred / inconclusive), reproducibility pointer to `17-1-perceptual-ab-results.csv`. Commit via `git add -f`. Update `memory/epic16_streaming_blocked.md` to reflect the audition outcome — change the framing from "blocked on the multi-listener perceptual A/B audition" to "audition cleared YYYY-MM-DD; flag flip certified" (outcome a) OR "audition deferred YYYY-MM-DD on grounds of <named defect>; next trigger: <named follow-up>" (outcome b) OR "audition inconclusive YYYY-MM-DD; non-seam defects logged for triage" (outcome c).
>
> ---
>
> **What this story is NOT** (explicit, to keep scope bounded):
>
> - This story is NOT a code change. The `streaming_mode.py:54-56` hardware probe, the `AppSettings.streaming_mode_override` field, the dispatch path in `qwen_tts_service.py`, the streaming settings panel, the trip-wire test, and `requirements.txt` are all untouched. If the audition surfaces a defect that mandates a code fix (e.g., a chunk-size / lookahead retune), that fix is a separate follow-up story scoped via the routing artifact.
>
> - This story does NOT regenerate the perceptual fixture. The 10 paired WAV files are on disk from Story 16.7 (initial build) + Story 16.8 (post-talker-fix regeneration). Regeneration would invalidate Commander's solo L1 audition from Story 16.8 and force a fresh L1 pass — a wasted hour with no architectural benefit.
>
> - This story does NOT extend the input set. `16-7-input-set.csv` is the canonical reproducibility fixture per the retro's §"Significant Discoveries" #3. If the audition surfaces a missing input class, capture it in the routing artifact as a follow-up scope item.
>
> - This story does NOT revisit NFR1. Story 16.9's per-class targets stand. Per the retro's §"Significant Discoveries" #4: "*the ramp story does NOT need to revise NFR1 again unless a future qwen-tts pin bump materially changes the latency floor.*"
>
> - This story does NOT bump the qwen-tts pin. The pin remains at `1ab0dd75353392f28a0d05d9ca960c9954b13c83` per Story 16.1.
>
> - This story does NOT add a dedicated `torch.cuda.Stream` for the decoder (D-8 follow-up). If the audition flags a phase artifact or tonal distortion that profiling would attribute to decoder-talker contention, that's a separate follow-up story.
>
> - This story does NOT touch any test file in `tests/`. The audition is a human-judgment exercise; there is no automated audition test (and there couldn't be — the architectural gate is human listener perception of audible seams, not a programmatic spectral-analysis check).
>
> - This story does NOT add or change any dependency. No `requirements.txt` edit, no `requirements-production.txt` edit, no qwen-tts pin change.
>
> - This story does NOT run on cloud infrastructure or hosted CI. The audition is conducted by Commander + 2 friends / family on their own playback hardware. No GPU is required for the audition itself — the WAV files are pre-rendered.

## Story

As a **MyVoice maintainer**,
I want **the deferred multi-listener perceptual A/B audition (Story 16.7 AC #2) executed against the post-Story-16.8 fixture, the verdict computed against the architecturally-named gate ("zero `audible_seam` flags on TRUE_STREAM pairs"), routed through `/bmad-bmm-correct-course` for stakeholder sign-off, and the architecture document amended to record the verdict**,
So that **the streaming-default flag flip is either certified (and the existing `streaming_mode.py:54-56` TRUE_STREAM-on-CUDA default is documented as audited) or formally deferred with a named next-trigger condition — either outcome closes Phase ⊥-Ramp and the Epic 16 retrospective's only remaining open follow-up**.

As a **MyVoice user (GPU host, default settings)**,
I want **the maintainer to confirm via independent listeners that the TRUE_STREAM rendition Epic 16 wired into production doesn't introduce audible seams on inputs my voice tends to produce (sibilants, tonal peaks, alliteration)**,
So that **the streaming-default I'm already running on this branch is one the maintainer has audited rather than one shipped on faith — and if the audition flags audible seams, the maintainer responds with a documented next step rather than ignoring the signal**.

As a **MyVoice user (CPU-only host)**,
I want **the maintainer to NOT change anything about my SENTENCE_STREAM code path while the GPU audition runs**,
So that **the audition's outcome (positive or negative) doesn't accidentally degrade the CPU code path I depend on (NFR12 protection unchanged)**.

## Acceptance Criteria

**Background — what this story is and is NOT.**

This story does five things to the working tree: recruit L2 + L3 listeners; collect fresh listener-grade L1 (Commander) + L2 + L3 audition observations into a committed CSV; route the verdict through `/bmad-bmm-correct-course` and commit the routing artifact; amend the architecture document with the verdict; update the memory entry. No source-tree edits expected. The deliverable is bounded to:

- `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (NEW — committed audition records, anonymized listener IDs, ≥30 rows = ≥10 utterances × ≥3 listeners; force-added via `git add -f` per the gitignore precedent established by Story 16.9)
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (NEW — routing artifact mirroring Story 16.9's `16-9-correct-course-nfr1-revision.md` structure; force-added)
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (MODIFIED — two-place edit per Story 16.9's pattern: inline pointer in the NFR3 row at line 803 + new H4 sub-section after line ~853; force-added per the gitignore precedent — file was first git-tracked by Story 16.9)
- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md` (MODIFIED — one-paragraph update reflecting the audition outcome)
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (this file, the story doc itself; updated as Change Log entries accumulate during dev)

This story does **NOT**:

- Touch `src/myvoice/services/tts_streaming/streaming_mode.py`, `src/myvoice/services/qwen_tts_service.py`, `src/myvoice/services/tts_streaming/codec_token_streamer.py`, `src/myvoice/services/tts_streaming/streaming_decoder.py`, `src/myvoice/models/app_settings.py`, `src/myvoice/ui/dialogs/settings/streaming_settings_panel.py`, or any other production source file. The audition is a documentary / certification pass, not a behavior change.

- Touch `tests/integration/test_streaming_tts_smoke.py`, `tests/unit/services/test_qwen_tts_service_dispatch.py`, `tests/unit/services/tts_streaming/*`, `tests/test_qwen_tts_internals.py`, or any other test file. The architectural gate is human listener perception; there is no automated audition test.

- Regenerate the perceptual fixture at `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/`. The fixture is on disk from Story 16.8's regeneration; regenerating would invalidate Commander's L1 solo audition.

- Edit `_bmad-output/implementation-artifacts/16-7-input-set.csv`. The input set is the canonical reproducibility fixture per the retro's §"Significant Discoveries" #3.

- Modify the existing `L1` / `L2` / `L3` entries in `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/_perlistener_truthtable.json`. The pre-computed randomizations for L1/L2/L3 must remain byte-identical (modifying them would invalidate any audition observations already collected against them). **Appending** new top-level keys (`L4`, `L5`, ...) for organically-volunteered listeners is the one permitted truth-table edit per AC #1 second-half.

- Edit `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/LISTENING-INSTRUCTIONS.md`. The protocol is canonical; if a listener has a question, capture it in the routing artifact as a follow-up scope item.

- Bump the qwen-tts pin or any dependency. No `requirements.txt` / `requirements-production.txt` change.

- Revisit NFR1. Story 16.9's per-class targets stand.

- Add a dedicated `torch.cuda.Stream` for the decoder (D-8 follow-up). If the audition flags a phase artifact, that's a separate follow-up story scoped via the routing artifact.

- Run on cloud infrastructure. Listeners audition on their own playback hardware.

The deliverable is approximately **+30-50 CSV rows** (the audition results — typically ~30 rows for the minimum 3 listeners × 10 utterances, may grow to ~40 if a 4th listener volunteers), **+150-300 lines** for the routing artifact + architecture amendment + memory entry update, and this story's Change Log documenting the audition runs. No production-code lines.

---

**AC #1 — Commander conducts a fresh listener-grade L1 audition AND L2 / L3 listeners are recruited and audition the fixture per the LISTENING-INSTRUCTIONS.md protocol.**

**Given** Story 16.8's prior audition (`16-8` line 472) verified only that the fixture renders audible non-silent audio (catastrophic-failure dimension) and explicitly deferred listener-grade observations to this story (`16-8` line 476)
**And** the deferred dimension is exactly what AC #2's controlled-vocabulary CSV requires (`audible_seam` / `clipping` / `phase_artifact` / `tonal_distortion` / `none`)
**When** Commander conducts the L1 audition fresh at listener-grade detail — following the canonical LISTENING-INSTRUCTIONS.md protocol verbatim, identical to what L2 and L3 will do
**Then** L1's observations are recorded as 10 rows in the canonical CSV schema with `listener_id = 'L1'`
**And** each row uses the controlled defect vocabulary (`none` / `audible_seam` / `clipping` / `phase_artifact` / `tonal_distortion` / `other_describe_in_notes`)
**And** the L1 trial-A / trial-B-vs-system mapping is resolved by joining against `_perlistener_truthtable.json` at verdict-computation time, NOT at audition time (the audition is blind in the protocol — the listener responds to A/B labels, the maintainer resolves to TRUE_STREAM/SENTENCE_STREAM during analysis)

**Given** Commander recruits ≥2 additional listeners (L2, L3) — friends / family / collaborators per the LISTENING-INSTRUCTIONS.md framing
**When** each listener accepts the audition packet (fixture directory + LISTENING-INSTRUCTIONS.md + listener-ID assignment + results-CSV template or plain-text instructions) **OR** (per AC #1 amendment 2026-05-08, Story 17.1 code-review pass per M3) audits co-located with Commander via the `17-1-l1-audition-helper.py` walkthrough format with Commander as scribe — the walkthrough variant is permitted; its three structural limitations (single-room listening environment / single-scribe prompt-framing risk / L1 non-anonymization) **must** be disclosed numerically in the architecture amendment and routing artifact
**Then** each listener audits all 10 utterances using the protocol verbatim (trial A end-to-end, trial B end-to-end, headphones if available, comfortable Discord-call volume)
**And** each listener records 10 rows in the canonical CSV schema (whether self-recorded in the remote-packet variant or scribed by Commander in the walkthrough variant; the helper's controlled-vocabulary input gates enforce label discipline either way)
**And** each listener's identity is anonymized in the committed artifact for L2 / L3 (use `L2`, `L3`); Commander retains a private mapping for follow-up correspondence if a defect needs reproducing. **L1 is Commander by construction (solo-dev framing) and is not anonymized — disclosed in the methodology limitations.**
**And** if a 4th or 5th listener volunteers organically (e.g., a partner or roommate sits in on the audition), their observations are accepted and recorded with sequentially-assigned listener IDs (`L4`, `L5`); each new listener requires APPENDING a new top-level entry to `_perlistener_truthtable.json` with their per-utterance A/B-vs-system randomization (mirror the L1/L2/L3 entries' shape; pre-existing L1/L2/L3 entries are NOT modified — append-only is the only permitted truth-table edit in this story); the gate at AC #3 still applies (zero `audible_seam` flags on TRUE_STREAM across ALL listeners)

---

**AC #2 — A canonical audition results CSV is committed with all listener observations, schema-validated and joinable against the truth-table.**

**Given** L1 observations from AC #1 first half + L2 + L3 observations (and any organic L4+) from AC #1 second half
**When** Commander compiles the results
**Then** a CSV at `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` exists with header `utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes`
**And** the CSV contains ≥30 rows (≥10 utterances × ≥3 listeners; if more listeners contributed, more rows — but no listener × utterance combination is missing; missing rows are a process failure that gates the verdict)
**And** every `a_defects_observed` and `b_defects_observed` value is from the controlled vocabulary (no out-of-vocabulary categories invented at the audition site; if a listener reports a defect not on the list, they used `other_describe_in_notes` and the maintainer captures the description in `free_text_notes`)
**And** every `a_or_b_preferred` value is one of `A`, `B`, or `equivalent`
**And** the CSV is force-added via `git add -f` per the gitignore precedent (`_bmad-output/` is gitignored at the directory level; individual files are tracked via `git add -f`)

**Given** the CSV needs to be joined against `_perlistener_truthtable.json` at verdict-computation time
**When** the join script (or manual lookup) is run
**Then** every `(listener_id, utterance_id)` pair in the CSV has a matching entry in the truth-table's listener block (i.e., L1's row for s-014 has a corresponding `s-014` block under the L1 key in the truth-table)
**And** the resolved per-row mapping yields a `trial_A_actual_mode` and `trial_B_actual_mode` value of either `true_stream` or `sentence_stream` (not anything else — the truth-table only contains these two systems)
**And** if any join failure occurs (mismatched listener_id, mismatched utterance_id, missing truth-table entry), the maintainer halts the verdict computation and resolves the mismatch before proceeding

---

**AC #3 — The pass/fail/inconclusive verdict is computed verbatim per the LISTENING-INSTRUCTIONS.md gate and ONE of three outcomes is committed.**

**Given** the resolved per-system defect counts from AC #2
**When** the verdict is computed
**Then** the verdict follows the LISTENING-INSTRUCTIONS.md gate verbatim: **PASS if and only if zero listeners flagged `audible_seam` for any TRUE_STREAM pair across all utterances**
**And** ONE of the following outcomes is committed:
  - **(a) Audition pass:** zero `audible_seam` flags on TRUE_STREAM pairs across all listeners and all utterances. The streaming-default flag flip is certified — the existing `streaming_mode.py:54-56` hardware probe's TRUE_STREAM-on-CUDA default is the audited release default. Architecture document amended at NFR3 cell (line 803) with a dated pointer + new H4 sub-section confirming the gate cleared. Routing artifact records "audition cleared, default certified". Memory entry updated to "audition cleared YYYY-MM-DD; flag flip certified".
  - **(b) Audition fail (audible seams flagged):** one or more `audible_seam` flags on TRUE_STREAM pairs from one or more listeners. The user-facing default in `streaming_mode.py:54-56` is NOT changed in this story (it was already TRUE_STREAM on CUDA from Epic 16; reverting that here would be a separate code-change story). Architecture document amended at NFR3 cell with the failure verdict + a named next-trigger condition for a follow-up story (chosen from: chunk-size / lookahead retune; dedicated `torch.cuda.Stream` for the decoder; qwen-tts pin bump; or "deferred indefinitely pending a different signal"). Routing artifact records "audition failed, default deferred, next trigger: <named follow-up>". Memory entry updated accordingly.
  - **(c) Audition inconclusive (non-seam defects flagged but no `audible_seam`):** the gate as written is PASS (the gate is specifically `audible_seam`-only), but the maintainer chooses to commit an "informational pass" that captures the non-seam defects (e.g., `tonal_distortion`, `phase_artifact`) for triage in a separate follow-up story. The streaming default is treated as in outcome (a) — certified per the LISTENING-INSTRUCTIONS.md gate as written — but the architecture amendment notes the non-seam defect ambiguity. Routing artifact records "audition pass with informational caveats, default certified, follow-up scoped for non-seam defects". Memory entry updated to "audition cleared YYYY-MM-DD; flag flip certified; non-seam defects logged for triage".
**And** the chosen outcome is implemented (or documented for outcome b) in this story's deliverable
**And** the verdict computation explicitly rejects "majority preference" as the gate — preference is informational at N=3 listeners; defect detection is the architectural concern (this is the LISTENING-INSTRUCTIONS.md gate verbatim)

**Given** the verdict computation requires a per-system defect-flag count
**When** the maintainer compiles the routing-artifact's supporting data table
**Then** the table contains at minimum: TRUE_STREAM (`A` in the canonical naming) defect-flag count by category (`audible_seam`, `clipping`, `phase_artifact`, `tonal_distortion`, `other`); SENTENCE_STREAM (`B`) defect-flag count by category; per-listener subtotals; per-utterance subtotals
**And** the table is included verbatim in both the routing artifact and the architecture amendment's H4 sub-section

---

**AC #4 — The verdict is routed through `/bmad-bmm-correct-course` literally and the routing artifact is committed.**

**Given** the verdict from AC #3 and Story 16.9's procedural precedent (`16-9-correct-course-nfr1-revision.md` is the canonical structure)
**And** the Epic 16 retrospective's §"What Could Have Gone Better" #4 explicitly names Story 16.9's deviation (using `AskUserQuestion` instead of `/bmad-bmm-correct-course`) as the **non-precedent** — Story 17.1 must use the literal `/bmad-bmm-correct-course` workflow
**When** the routing pass is executed
**Then** the `/bmad-bmm-correct-course` workflow is invoked (e.g., via `/bmad-bmm-correct-course "Streaming Default Ramp audition verdict — outcome (a/b/c) per AC #3"` or equivalent)
**And** the routing artifact at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` is generated and committed
**And** the artifact mirrors `16-9-correct-course-nfr1-revision.md`'s structure: header (story / date / context); the verdict + per-system defect-flag count table from AC #3; the architectural action (confirm default / defer + name next-trigger / informational pass); Commander sign-off line
**And** the artifact is force-added via `git add -f`

**Given** the solo-dev framing (Commander is sole stakeholder)
**When** the workflow asks for stakeholder sign-off
**Then** Commander signs off in-line in the routing artifact (a written, committed sign-off statement — not a verbal acknowledgment)
**And** the sign-off statement names the chosen outcome explicitly (a / b / c)
**And** if outcome (b) is chosen, the sign-off names the next-trigger condition explicitly (chunk-size retune / dedicated CUDA stream / pin bump / deferred indefinitely) — vague "we should look into this someday" framings are rejected

---

**AC #5 — The architecture document is amended via the two-place edit pattern established by Story 16.9.**

**Given** Story 16.9's amendment at `architecture-optimization-pass.md:802` (inline pointer) + line 819 (new H4 sub-section) is the canonical two-place edit pattern
**And** the architecture document is gitignored at the directory level but tracked individually since Story 16.9's `git add -f`
**When** the amendment is authored
**Then** the inline pointer is appended to the NFR3 row of the Inherited NFR coverage table at line 803, after the existing text "D-8 chunk + overlap-add with seam-quality A/B testing before flipping streaming default; sentence-stream/batch unchanged from V2"
**And** the inline pointer reads (exact wording for outcome (a)): ` *(Story 17.1 audition cleared YYYY-MM-DD — see follow-up note below.)*` — substituting the actual closure date
**And** the inline pointer for outcome (b) reads: ` *(Story 17.1 audition deferred YYYY-MM-DD — audible-seam defect on <utterance(s)>, next trigger <named>; see follow-up note below.)*`
**And** the inline pointer for outcome (c) reads: ` *(Story 17.1 audition cleared YYYY-MM-DD with informational caveats — see follow-up note below.)*`
**And** the inline pointer is OUTSIDE the table cell's primary text — appended as a parenthetical italic — so the markdown table renders cleanly (no table-row breakage; verify the NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows still render in a single table after the edit)

**Given** the new H4 sub-section is placed immediately after the existing Story 16.9 sub-section (currently ending around line 853)
**When** the new sub-section is authored
**Then** the new sub-section's heading is `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, YYYY-MM-DD)`
**And** the body captures: (1) the verdict (pass / fail / inconclusive); (2) the listener count; (3) the per-utterance per-system defect-flag count table from AC #3 (verbatim); (4) the architectural decision (default certified / deferred + named next-trigger / informational pass); (5) the reproducibility pointer to `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv`; (6) the routing-artifact pointer to `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md`
**And** the sub-section explicitly does NOT propose a code change — it's a documentary amendment confirming or deferring the existing default
**And** the architecture document's edits are committed via `git add -f` per the gitignore precedent

**Given** Story 16.9's Document Maintenance section convention (if present) for logging amendments
**When** the architecture amendment is committed
**Then** if the Document Maintenance section exists in `architecture-optimization-pass.md`, the amendment is logged per the existing convention (e.g., a dated bullet "*2026-05-XX: NFR3 follow-up note appended (Story 17.1 audition outcome).*"); if no Document Maintenance section exists, no log entry is added (don't invent one)

---

**AC #6 — The memory entry `epic16_streaming_blocked.md` is updated to reflect the audition outcome.**

**Given** the existing memory entry at `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md` is currently framed as "Stories 16.8 + 16.9 closed both blockers (TRUE_STREAM wire-up + NFR1 reconciled to per-class targets); flag flip's remaining prerequisite is the multi-listener perceptual A/B audition"
**When** the audition outcome is recorded
**Then** the memory entry is updated (NOT replaced wholesale — preserve the existing context that names the conjunction of three prerequisites)
**And** the update reflects the chosen outcome verbatim:
  - **Outcome (a):** "audition cleared YYYY-MM-DD; flag flip certified; Phase ⊥-Ramp closed"
  - **Outcome (b):** "audition deferred YYYY-MM-DD on grounds of <named defect>; next trigger: <named follow-up>; default unchanged on this branch (already TRUE_STREAM on CUDA from Epic 16)"
  - **Outcome (c):** "audition cleared YYYY-MM-DD with informational caveats (non-seam defects: <named>); flag flip certified; non-seam triage in <follow-up story / not yet scoped>"
**And** the entry's frontmatter `description:` field is updated to reflect the new state (one-line summary)
**And** if the entry no longer represents an open block, its `description:` should reflect that — e.g., "Streaming default flag flip certified after Story 17.1 audition; entry retained as a historical pointer to the three-prerequisite conjunction"

---

**AC #7 — The story closes with a clean test suite, sprint-status flip, and final commit.**

**Given** the audition + routing + amendment artifacts are committed
**When** the story is marked complete
**Then** the full streaming + dispatch test suite passes locally: `python310\python.exe -m pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py tests/unit/services/tts_streaming/ -v` (verifying no regressions even though no source-tree edits were made — the test run is the architectural smoke gate)
**And** `sprint-status.yaml`'s `17-1-streaming-default-ramp` is flipped from `ready-for-dev` → `in-progress` → `review` (transition to `done` happens after `code-review` workflow runs, mirroring Stories 16.8 / 16.9)
**And** the commit message names the outcome explicitly: `Story 17.1: streaming default ramp — audition outcome (a/b/c) — Phase ⊥-Ramp closed` (substituting the actual outcome letter)
**And** if outcome (b) was chosen, the commit message also names the next-trigger condition (e.g., `Story 17.1: streaming default ramp — audition deferred — next trigger: chunk-size retune`)

## Tasks / Subtasks

- [ ] **Task 1 — Recruit L2 + L3 listeners and prepare audition packets (AC #1 second half)**
  - [ ] Subtask 1.1 — Identify ≥2 candidate listeners from friends / family / collaborators per the LISTENING-INSTRUCTIONS.md framing (the audition is not a corporate UAT — informal recruit is correct)
  - [ ] Subtask 1.2 — For each listener, prepare a packet: zip or share-link the `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` directory (10 paired WAV files + LISTENING-INSTRUCTIONS.md + `_perlistener_truthtable.json` — share the truth-table file too, since the LISTENING-INSTRUCTIONS.md instructs the listener to look up their own A/B filenames in it)
  - [ ] Subtask 1.3 — Assign listener IDs (`L2`, `L3`) and instruct each listener to use their assigned ID when recording observations; record the private listener-ID-to-human mapping in a non-committed location (Commander's notes; this mapping is NOT committed)
  - [ ] Subtask 1.4 — Provide each listener a results-CSV template with the canonical header (`utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes`) and the controlled defect vocabulary (copy from LISTENING-INSTRUCTIONS.md); accept observations as plain text or spreadsheet if a listener prefers (the maintainer normalizes during compilation in Task 2)
  - [ ] Subtask 1.5 — If a listener has questions about the protocol, answer them OUT-OF-BAND (do NOT edit LISTENING-INSTRUCTIONS.md to clarify — the protocol is canonical); capture any non-trivial questions in the routing artifact as a follow-up scope item

- [x] **Task 1 — Recruit L2 + L3 listeners and prepare audition packets (AC #1 second half)** *(Closed via the in-person walkthrough format chosen 2026-05-08 — see Change Log #7. Subtasks 1.2 / 1.4 substituted by the helper which serves as the audition driver directly; subtasks 1.1 / 1.3 satisfied by L2/L3 being co-located with Commander.)*

- [x] **Task 2 — Conduct fresh L1 audition AND compile L1 + L2 + L3 observations into the canonical CSV (AC #1 first half + AC #2)**
  - [x] Subtask 2.1 — L1 audition conducted via `17-1-l1-audition-helper.py`; 10 L1 rows recorded; clean PASS profile under the gate (9× `equivalent / none / none`, 1× `m-012 B-preferred / none / none / "A was quiter than B"`); zero `audible_seam` on TRUE_STREAM
  - [x] Subtask 2.2 — L2 and L3 in-person walkthrough sessions conducted via `...helper.py L2` and `...helper.py L3` with Commander as scribe; 10 rows each recorded in the canonical CSV with controlled-vocabulary discipline preserved by the helper's input validation
  - [x] Subtask 2.3 — Out-of-vocabulary rejection enforced continuously by the helper's input gates at entry time; no separate normalization pass needed; CSV passes schema validation with 0 errors
  - [x] Subtask 2.4 — 30 rows verified (10 utterances × 3 listeners); no `(listener_id, utterance_id)` combination missing; truth-table join in Task 3 resolved 30/30 rows with 0 join errors
  - [x] Subtask 2.5 — Canonical CSV saved at `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv`; force-added via `git add -f` at Task 9 commit pass

- [x] **Task 3 — Resolve A/B-vs-system mapping and compute per-system defect-flag counts (AC #2 second half + AC #3 second half)**
  - [x] Subtask 3.1 — Inline Python join script (one-off, executed via `python310\python.exe -c "..."`) joined the 30-row canonical CSV against `16-8-perceptual-fixtures/_perlistener_truthtable.json`; 30/30 rows resolved with 0 join errors; working table captured in Change Log #8 verdict-computation tables
  - [x] Subtask 3.2 — Per-system defect-flag counts computed: TRUE_STREAM = 30/30 `none`, 0/30 of every other category; SENTENCE_STREAM = 30/30 `none`, 0/30 of every other category. Tables in Change Log #8 + routing artifact §2.2
  - [x] Subtask 3.3 — Per-listener (L1/L2/L3 each: 0/10 audible_seam on each system) and per-utterance (10 utterances × 0/3 audible_seam on each system) subtotals computed. Tables in Change Log #8 + routing artifact §2.3 / §2.4
  - [x] Subtask 3.4 — Working table + aggregates recorded in Change Log #8 ("Verdict computation 2026-05-08")

- [x] **Task 4 — Choose AC #3 outcome (AC #3 first half)**
  - [x] Subtask 4.1 — Gate applied verbatim per LISTENING-INSTRUCTIONS.md: TRUE_STREAM `audible_seam` count = 0 across all 30 trials. **Outcome (a) — audition PASS, default certified.**
  - [x] Subtask 4.2 — Outcome (a) is the unambiguous reading; not (b) (zero `audible_seam` contradicts fail), not (c) (zero non-seam defects either, so no informational caveats apply). Bounded to AC #3's legitimate options; no rogue scope expansion
  - [x] Subtask 4.3 — N/A for outcome (a); no next-trigger or non-seam triage scope item required

- [x] **Task 5 — Route the verdict through `/bmad-bmm-correct-course` and commit the routing artifact (AC #4)**
  - [x] Subtask 5.1 — `/bmad-bmm-correct-course` invoked **literally** via the `Skill` tool from inside `/bmad-bmm-dev-story` with verdict context + Mode=Batch (honors Epic 16 retro §"What Could Have Gone Better" #4 lesson — Story 16.9's `AskUserQuestion` substitution was the non-precedent; this turn uses the literal-invocation discipline)
  - [x] Subtask 5.2 — Routing artifact authored at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` mirroring `16-9-correct-course-nfr1-revision.md`'s structure (header / §1 routing rationale / §2 empirical evidence with 4 sub-tables / §3 decision approved / §4 architectural action / §5 implications acknowledged / §6 stakeholder sign-off / §7 cross-references). The workflow's native output (Sprint Change Proposal at `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md`) is committed alongside as the procedural-compliance record
  - [x] Subtask 5.3 — Full supporting data table included in §2 of the routing artifact (per-system, per-listener, per-utterance subtotals + preference resolution + free-text notes); also in the Sprint Change Proposal §1 + §4
  - [x] Subtask 5.4 — Commander sign-off captured in-line in §6 of the routing artifact (approval via `AskUserQuestion` at workflow Step 5; recorded with date, channel, approved option, conditions, and methodology disclosure)
  - [x] Subtask 5.5 — Routing artifact + Sprint Change Proposal force-added via `git add -f` at Task 9 commit pass

- [x] **Task 6 — Amend the architecture document (AC #5)**
  - [x] Subtask 6.1 — Located Inherited NFR coverage table at lines 800-808; NFR3 row at line 803
  - [x] Subtask 6.2 — Inline pointer appended to NFR3 row at line 803: ` *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)*` (parenthetical italic outside the cell's primary text, mirroring Story 16.9 line 802 pattern)
  - [x] Subtask 6.3 — New H4 sub-section `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)` inserted immediately after the existing Story 16.9 sub-section's last source-artifact bullet; before the `### Implementation Readiness Validation — with surfaced gaps` heading
  - [x] Subtask 6.4 — H4 body authored per AC #5: verdict (PASS); listener count (3); per-utterance per-system defect-flag count table (verbatim from Change Log #8); architectural decision (default certified, no code change); methodology footnote (walkthrough format disclosure); informational signal (m-012 volume observation); reproducibility pointers to `17-1-perceptual-ab-results.csv`, routing artifact, Sprint Change Proposal, fixture dir, story file
  - [x] Subtask 6.5 — Table integrity verified post-edit by reading lines 800-808: NFR3 row's parenthetical italic does NOT break the table; NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows still render in a single table; closing at line 808 → blank → "Local FR-equivalent coverage:" intact at line 810
  - [x] Subtask 6.6 — Grep for "Document Maintenance" / "Maintenance Log" / "Amendment Log" / "Revision History" returned no matches in the architecture document; no log entry invented (per AC #5 final clause)
  - [x] Subtask 6.7 — Architecture document staged via regular `git add` at Task 9 commit pass (file is already tracked via Story 16.9's prior `git add -f`; this amendment lands as a regular Modified change, NOT a force-add — per the retro's §"What Could Have Gone Better" #6 lesson on labeling)

- [x] **Task 7 — Update the memory entry `epic16_streaming_blocked.md` (AC #6)**
  - [x] Subtask 7.1 — Read the existing memory entry (preserved the existing conjunction-of-three framing as historical context; numbered items 1 and 2 for Stories 16.8 and 16.9 retained verbatim; new numbered item 3 added for Story 17.1)
  - [x] Subtask 7.2 — Entry updated per AC #6 outcome (a) wording: "audition cleared 2026-05-08; flag flip certified; Phase ⊥-Ramp closed". One paragraph for Story 17.1 + reframed "Streaming-default flag flip status" line + reframed "How to apply" guidance to remove the now-obsolete "Do NOT recommend the flip without the audition" rule
  - [x] Subtask 7.3 — Frontmatter `description:` updated to "Stories 16.8 + 16.9 + 17.1 closed all three prerequisites (TRUE_STREAM wire-up + NFR1 reconciliation + multi-listener audition); streaming-default flag flip certified 2026-05-08; entry retained as historical pointer." Frontmatter `name:` also updated to "(closed)" framing for clarity
  - [x] Subtask 7.4 — File saved. Also updated `MEMORY.md` index line to reflect the new title + hook (per CLAUDE.md auto-memory guidance: "Keep the name, description, and type fields in memory files up-to-date with the content"). Memory directory is outside the repo; no git commit

- [x] **Task 8 — Smoke-test the streaming + dispatch test suite (AC #7 first half)**
  - [x] Subtask 8.1 — Ran `python310\python.exe -m pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py tests/unit/services/tts_streaming/ -v`
  - [x] Subtask 8.2 — **154 tests passed in 26.18s; 0 failures, 0 skips, 0 errors.** Confirms no regressions (correct — no source-tree edits were made; documentary smoke gate clean)
  - [x] Subtask 8.3 — N/A; no test failure to investigate

- [ ] **Task 9 — Commit, flip sprint status, and finalize (AC #7 second half)**
  - [ ] Subtask 9.1 — Stage all artifacts in a single commit (or commit pair: audition data + routing + amendment in one commit, sprint-status flip in a second). Files: `17-1-perceptual-ab-results.csv` (force-added), `17-1-correct-course-streaming-default-ramp.md` (force-added), `architecture-optimization-pass.md` (force-added), `17-1-streaming-default-ramp.md` (this file, with Change Log updates)
  - [ ] Subtask 9.2 — Commit message: `Story 17.1: streaming default ramp — audition outcome [a/b/c] — Phase ⊥-Ramp closed` (substituting the actual outcome letter; if outcome (b), append the next-trigger condition)
  - [ ] Subtask 9.3 — Flip `sprint-status.yaml`'s `17-1-streaming-default-ramp: ready-for-dev → in-progress → review` (transition to `done` happens after `code-review` workflow runs, mirroring Stories 16.8 / 16.9). Also flip `epic-17: in-progress → done` once the story reaches `review` (single-story epic)
  - [ ] Subtask 9.4 — Verify `git status` shows a clean working tree after the commit
  - [ ] Subtask 9.5 — Verify the commit's `git diff --stat` shows only the 4 artifact files + sprint-status.yaml — no source-tree edits leaked in

## Dev Notes

### Project Structure Notes

- **Source-tree alignment.** Story 17.1 is purely a documentary / certification story; its production-code footprint is **zero**. The deliverable is audition data (CSV) + routing artifact (markdown) + architecture amendment (two-place edit on existing markdown) + memory entry update (single file in the memory directory). All artifacts land in the `_bmad-output/` tree (gitignored at the directory level; tracked individually via `git add -f` per the precedent established by Story 16.9) or in the memory directory (outside the repo; persisted via Claude's auto-memory system).
- **No conflict with prior stories.** Stories 16.8 + 16.9 closed 2026-05-08; this story builds on their artifacts (the regenerated fixture from 16.8; the architecture amendment pattern from 16.9; the routing artifact format from 16.9; the gitignore precedent from 16.9). The dispatch path is unchanged from Epic 16; the latency contract is unchanged from Story 16.9.
- **Working-tree state at story creation (2026-05-08).** Working tree is clean per `gitStatus: (clean)` at conversation start. No residual edits to bundle.
- **Audition cost.** L2 / L3 audition runtime is approximately 5-10 minutes per listener (10 utterances × ~30 seconds per utterance × 2 trials per utterance + recording time). Listener recruit-to-results turnaround is the rate-limiting factor; expect 1-3 days from story-creation to compiled CSV depending on listener availability.
- **`correct-course` routing.** AC #4 requires `/bmad-bmm-correct-course` literally. The retro's §"What Could Have Gone Better" #4 explicitly names Story 16.9's deviation (using `AskUserQuestion`) as the non-precedent. Use the literal workflow.
- **Architecture document handling.** `architecture-optimization-pass.md` is gitignored at the directory level (`_bmad-output/` per `.gitignore:146`) but tracked individually since Story 16.9's `git add -f`. This story's amendment is committed via `git add -f` per the established precedent. The retro's §"What Could Have Gone Better" #6 names the "force-added file labeled as Modified vs Created" issue from Story 16.9 — for this story, the file is already tracked, so the amendment is a regular `Modified` (label correctly in the File List).

### Pre-existing Infrastructure That This Story Uses (Read-Only)

The fixture, truth-table, instructions, and architecture-amendment scaffolding are all in place from prior stories. This story consumes them as-is:

- `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` (10 paired WAV files; do NOT regenerate)
- `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/_perlistener_truthtable.json` (L1/L2/L3 randomizations pre-computed; pre-existing entries do NOT change; append-only edits to add `L4`+ entries are permitted if organic listeners join — see AC #1 second-half)
- `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` (canonical protocol; do NOT edit)
- `_bmad-output/implementation-artifacts/16-7-input-set.csv` (canonical input set; do NOT edit)
- `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` (routing-artifact format precedent; mirror the structure)
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:802` + line 819 (two-place amendment pattern; mirror the structure)
- `src/myvoice/services/tts_streaming/streaming_mode.py:54-56` (the existing TRUE_STREAM-on-CUDA hardware probe; do NOT edit)
- `src/myvoice/models/app_settings.py::streaming_mode_override` (default `None` defers to the probe; do NOT edit)
- Story 16.8 §"Perceptual audition (Commander solo, 2026-05-07)" (catastrophic-failure-only record; informational only — Task 2 collects fresh L1 listener-grade observations, NOT a normalization of this prior record)

### Solo-Dev Framing (Listener Recruit and Sign-Off)

The project is solo-dev (Commander is sole maintainer per `memory/production_release_state.md`). The "stakeholder" in `/bmad-bmm-correct-course` is Commander acting in a release-decision capacity. The "listeners" L2 + L3 are recruited from friends / family / collaborators per the LISTENING-INSTRUCTIONS.md framing — this is informal recruit, not corporate UAT. Listeners do not need MyVoice expertise; the audition protocol is designed for naive listeners (the gate is "did you hear an audible click / gap / discontinuity", not "is this stream artifact-free in a spectral analysis").

If recruiting two listeners proves slow (e.g., 1+ week without responses), the story remains in `ready-for-dev` until ≥2 additional listeners are recruited — AC #1 + AC #2 hard-require ≥3 total listeners (≥30 rows). N<3 is not a permitted closure path; the architectural cost of an under-powered audition (false-PASS risk) outweighs the schedule cost of waiting for a third listener. If recruitment stalls indefinitely, the story is paused (sprint-status flips back to `backlog` with a recruitment-stall note), not closed at reduced N.

### Outcome (a) — Audition Pass Path (Most Likely)

If zero listeners flag `audible_seam` on TRUE_STREAM pairs, the story closes via outcome (a):

1. The streaming default is certified — `streaming_mode.py:54-56`'s TRUE_STREAM-on-CUDA branch is documented as audited.
2. Architecture amendment captures: verdict (PASS), listener count (≥3), per-utterance per-system defect-flag table (likely sparse — possibly all `none`), decision (default certified), reproducibility pointer.
3. Routing artifact captures: verdict, supporting data table, architectural action ("default certified"), Commander sign-off.
4. Memory entry updated to "audition cleared YYYY-MM-DD; flag flip certified; Phase ⊥-Ramp closed".
5. Commit message: `Story 17.1: streaming default ramp — audition outcome (a) — Phase ⊥-Ramp closed`.
6. Sprint status flips epic-17 to `done` and 17-1 to `review`.

Outcome (a) is plausible — but only weakly so before the listener-grade audition runs. Story 16.8's prior catastrophic-failure check (`16-8` line 474: "*no silence, no full-second dropouts, no distortion observed*") would have flagged a *severe* seam (e.g., a half-second click between chunks), but the LISTENING-INSTRUCTIONS.md `audible_seam` category covers any audible click / gap / discontinuity at listener-grade attention — a finer threshold than catastrophic-failure detection. The fixture's perceptual-difficult subset is intentionally adversarial (sibilants, alliteration, tonal peaks); the N≥3 listener-grade audition is the actual gate. **Do NOT skip Subtask 2.1 (fresh L1 audition) on the assumption that Story 16.8's prior pass already covered listener-grade detection — it did not.**

### Outcome (b) — Audition Fail Path

If one or more listeners flag `audible_seam` on TRUE_STREAM pairs, the story closes via outcome (b):

1. The user-facing default in `streaming_mode.py:54-56` is **NOT changed** in this story (Epic 16 already shipped TRUE_STREAM as the live default; reverting that here would be a separate code-change story scoped via the routing artifact's named next-trigger).
2. Architecture amendment captures: verdict (FAIL), listener count (≥3), per-utterance per-system defect-flag table, decision (default deferred + named next-trigger), reproducibility pointer.
3. Routing artifact captures: verdict, supporting data table, architectural action ("default deferred, next trigger: <named>"), Commander sign-off naming the next-trigger explicitly.
4. Memory entry updated to "audition deferred YYYY-MM-DD on grounds of <named defect>; next trigger: <named follow-up>; default unchanged on this branch (already TRUE_STREAM on CUDA from Epic 16)".
5. Commit message: `Story 17.1: streaming default ramp — audition outcome (b) — next trigger: <named>`.
6. Sprint status flips epic-17 to `done` (single-story epic; the deferral is a closure, not a re-open) and 17-1 to `review`.

The next-trigger options Commander chooses from in outcome (b):
- **Chunk-size / lookahead retune:** if the seam pattern suggests overlap-add boundary artifacts, scope a follow-up story that sweeps `codec_token_streamer.py:DEFAULT_CHUNK_SIZE` / `DEFAULT_LOOKAHEAD` and re-runs the perceptual A/B audition. **Trigger:** if seam pattern is regular (every chunk boundary).
- **Dedicated `torch.cuda.Stream` for the decoder:** if the seam pattern suggests decoder-talker contention (e.g., periodic dropouts under load), scope a D-8 follow-up story that adds a dedicated stream and re-runs the latency harness + audition. **Trigger:** if seam pattern is irregular and correlates with concurrent dispatches.
- **qwen-tts pin bump:** if the audition surfaces defects that match a known upstream issue (e.g., a streaming-related bug fix in a newer qwen-tts commit), scope a pin-bump follow-up story per the retro's §"Significant Discoveries" #7. **Trigger:** if upstream has a relevant patch.
- **Deferred indefinitely with a named signal:** if the audition surfaces defects but no clear path to fix, defer the ramp until a future signal (e.g., user complaint volume, model upstream improvement). **Trigger:** documented user-impact threshold or upstream-version threshold.

### Outcome (c) — Audition Inconclusive Path

If listeners flag non-`audible_seam` defects (e.g., `tonal_distortion`, `phase_artifact`) but no `audible_seam`, the gate-as-written (LISTENING-INSTRUCTIONS.md "PASS if and only if zero `audible_seam` flags on TRUE_STREAM") is technically PASS, but Commander chooses to commit an "informational pass":

1. The streaming default is certified per the gate as-written.
2. Architecture amendment captures: verdict (PASS with informational caveats), listener count (≥3), per-utterance per-system defect-flag table (showing the non-seam defects), decision (default certified, follow-up scoped for non-seam triage), reproducibility pointer.
3. Routing artifact captures: verdict, supporting data table including non-seam defects, architectural action ("default certified, follow-up scoped for non-seam defects"), Commander sign-off naming the follow-up scope item.
4. Memory entry updated to "audition cleared YYYY-MM-DD; flag flip certified; non-seam defects logged for triage".
5. Commit message: `Story 17.1: streaming default ramp — audition outcome (c) — informational pass`.
6. Sprint status flips epic-17 to `done` and 17-1 to `review`.

Outcome (c) is the structurally "messiest" path because the gate as-written passes but the audition produced signal-of-concern. Commander has discretion: if the non-seam defects are minor and well-understood (e.g., one listener flagged `tonal_distortion` on one utterance and the others didn't), outcome (c) is appropriate. If multiple listeners flag the same non-seam defect on the same utterance, that's evidence for outcome (b) framing — Commander reserves the right to escalate from outcome (c) to outcome (b) at sign-off time.

### Reviewer Mental-Model Additions From the Epic 16 Retrospective

Three reviewer mental-model additions named in the Epic 16 retrospective's §"Action Items / Follow-ups" apply to this story:

1. **"Regression tests must mirror the *exact* bug class, not the nearest adjacent class"** (`memory/code_review_regression_test_exact_class.md`). For Story 17.1, this means: if the audition surfaces a defect and outcome (b) is chosen with a chunk-size retune as the next-trigger, the follow-up story's regression test must exercise the EXACT defect type the audition surfaced (e.g., `audible_seam` on `m-013`), not a nearest-adjacent test (e.g., a generic chunk-size sweep test).

2. **"Claim that an AC is verified by empirical measurement → trace the verification mechanism through the dispatch path before drafting the AC"** (Epic 16 retro §"What Could Have Gone Better" #3 — Story 16.9's vacuous AC #4). For Story 17.1's AC #3, the verification mechanism is "human listener flagged `audible_seam` per the LISTENING-INSTRUCTIONS.md gate"; the dispatch path is the listener auditing the actual TRUE_STREAM rendition (resolved via the truth-table join). This trace is documented above; reviewers should verify the ACs match the actual mechanism.

3. **"Marked-`[x]` subtask is a CLAIM, not a deliverable"** (Epic 16 retro §"What Could Have Gone Better" #1 — Story 16.6's C2). For Story 17.1's Task 2 in particular, marking subtask 2.1 (`Re-listen Commander's L1 audition`) as `[x]` requires the L1 rows to actually exist in the canonical CSV; reviewers should `grep` the CSV for `listener_id = 'L1'` rows before accepting the subtask as done.

### References

- Architecture: `_bmad-output/planning-artifacts/architecture-optimization-pass.md:803` (NFR3), `:255` (D-8 — overlap-add), `:257` (D-9 — hardware-aware default), `:802` + `:819` (Story 16.9 amendment pattern — mirror this)
- Epic 16 retrospective: `_bmad-output/implementation-artifacts/epic-16-retro-2026-05-08.md` §"Significant Discoveries Affecting the Streaming Default Ramp follow-up story" (the canonical scope sketch); §"What Could Have Gone Better" #4 (use `/bmad-bmm-correct-course` literally); §"Action Items / Follow-ups" "Streaming-default ramp story scope (sketch for the SM workflow)" (this story's de-facto template)
- Story 16.7: `_bmad-output/implementation-artifacts/16-7-empirical-validation-gates-for-streaming-default.md` AC #2 (the deferred multi-listener audition protocol)
- Story 16.7 report: `_bmad-output/implementation-artifacts/16-7-streaming-validation-report.md` (the empirical baseline)
- Story 16.8: `_bmad-output/implementation-artifacts/16-8-true-stream-real-wire-up.md` AC #6 + §"Perceptual audition (Commander solo, 2026-05-07)" (catastrophic-failure-only audition; explicitly deferred listener-grade observations to this story per `16-8` line 476 — Story 17.1's L1 audition is FRESH at listener-grade detail, not a normalization of the 16.8 record)
- Story 16.9: `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-sentence-stream-latency-investigation.md` (procedural precedent for outcome (c) framing); `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` (routing-artifact format precedent — mirror this)
- Memory: `memory/epic16_streaming_blocked.md` (the entry to update in Task 7); `memory/code_review_regression_test_exact_class.md` (reviewer mental model #1); `memory/production_release_state.md` (solo-dev framing)
- Code anchors (read-only — no edits expected): `src/myvoice/services/tts_streaming/streaming_mode.py:54-56` (the TRUE_STREAM-on-CUDA hardware probe); `src/myvoice/models/app_settings.py::streaming_mode_override` (defaults to None); `src/myvoice/services/qwen_tts_service.py::_generate_true_stream` (Epic 16 dispatch path; no change)
- Fixture: `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` (10 paired WAV files); `_perlistener_truthtable.json` (L1/L2/L3 randomizations); `LISTENING-INSTRUCTIONS.md` (canonical protocol)
- Decision framing: this story may end with a decision NOT to certify the default but to defer the ramp + name a follow-up trigger — that is a valid outcome under AC #3(b) and not a failure mode (mirror of Story 16.9's "outcome (c) is a peer outcome" framing — Epic 16 retro §"Key Insights" #2)
- Risk profile: low — no production source-tree edits expected; the deliverable is audition data + a routing artifact + an architecture amendment + a memory-entry update
- Sequencing: independent of any future Epic — closes the Epic 16 retrospective's only remaining open follow-up; the streaming-default flag flip's blocking condition (conjunction of Story 16.8 + Story 16.9 + this story's audition gate) is fully resolved when this story closes

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m]

### Debug Log References

### Completion Notes List

- **2026-05-08 — Story load + Task 1.4 working-file preparation.** `/bmad-bmm-dev-story` started; sprint-status flipped `ready-for-dev → in-progress`. Verified pre-existing infrastructure: 10 paired WAV files in `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` (20 WAVs total: `s-014/15/16/17`, `m-011/12/13/14`, `l-013/14`); `LISTENING-INSTRUCTIONS.md` present and unchanged; `_perlistener_truthtable.json` present with L1/L2/L3 keys + 10 utterances each (verified byte-identical to story-creation state). Confirmed scope boundary: AC #1 + AC #2 hard-require ≥3 listeners' real audition observations (≥30 CSV rows from human listening), so Tasks 2–9 cannot run until Commander has L1 fresh listener-grade audition data + L2/L3 recruited+returned data in hand. Per Commander's selection ("No data yet — prep packets, then I HALT"), prepared two working-only (NOT committed) helper files for Subtask 1.4: a results-CSV template (`17-1-perceptual-ab-results-TEMPLATE.csv`) with header + 10 pre-populated `utterance_id` rows in short→medium→long order; and a listener-packet cover note (`17-1-listener-packet-note.md`) duplicating the controlled defect vocabulary from LISTENING-INSTRUCTIONS.md and pointing the listener at the truth-table for A/B file lookup. Subtasks 1.1 (recruit), 1.2 (zip+share), 1.3 (assign IDs), 1.4 ("provide each listener"), 1.5 (answer protocol Qs out-of-band) all remain unchecked — Commander still needs to dispatch the packets and collect results. HALT pending audition data.

### File List

**Committed deliverables (force-added new files + regular-modified tracked files):**

- `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (NEW — canonical audition CSV, 30 rows, schema-validated, force-added via `git add -f` per gitignore precedent; AC #2 deliverable)
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (NEW — AC #4 routing artifact mirroring `16-9-correct-course-nfr1-revision.md` structure; force-added; Commander sign-off captured in §6)
- `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md` (NEW — `/bmad-bmm-correct-course` workflow native output; force-added; procedural-compliance record paired with the routing artifact)
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (MODIFIED — NFR3 row inline pointer at line 803 + new H4 sub-section "Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)" inserted after the existing Story 16.9 sub-section; AC #5 deliverable; regular `git add` since the file is already tracked from Story 16.9's prior force-add)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (MODIFIED — `17-1-streaming-default-ramp: ready-for-dev → in-progress → review` and `epic-17: in-progress → done`; tracked file)
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (MODIFIED — this story file: subtask checkmarks across Tasks 1-8, Status: ready-for-dev → review, Change Log entries #3-#9, Completion Notes, File List; force-added per gitignore)

**Modified outside the repo (not in commit; persisted via Claude auto-memory):**

- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md` (MODIFIED — Story 17.1 numbered item 3 added; "Streaming-default flag flip status" line reframed to "certified 2026-05-08"; "How to apply" guidance reframed to remove the obsolete "do not recommend the flip without the audition" rule; frontmatter `name:` and `description:` updated to reflect closure; AC #6 deliverable)
- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\MEMORY.md` (MODIFIED — index line updated to match the entry's new title + closure framing per CLAUDE.md auto-memory guidance)

**Promoted to committed by Story 17.1 code-review pass (M2):**

- `_bmad-output/implementation-artifacts/17-1-l1-audition-helper.py` (NEW — Windows-native Python audition driver using `winsound.PlaySound`; reads truth-table, plays trial-A/trial-B blind, validates inputs against the controlled vocabulary, appends rows incrementally to the canonical CSV with `flush()` after every row; resume-safe via per-listener `(utterance_id)` skip; defaults to `L1` with optional CLI listener-id arg for in-person L2/L3 use; targets `16-8-perceptual-fixtures/` per the post-fix fixture directory). **Force-added by code-review pass per M2** — the helper is the only mechanism that preserves blinding (LISTENING-INSTRUCTIONS.md as written exposes the actual mode via filename lookup); leaving it as a working file would have meant the audit cannot be re-run in a blind-preserving way from a fresh clone.

**Working files (NOT committed; gitignored under `_bmad-output/`):**
- `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results-TEMPLATE.csv` (NEW — helper template for hypothetical remote L2/L3 packet dispatch; superseded by the in-person walkthrough format Commander selected; retained on disk as a working reference)
- `_bmad-output/implementation-artifacts/17-1-listener-packet-note.md` (NEW — per-packet listener orientation note; same supersedence as the template CSV; retained on disk as a working reference)
- `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/_perlistener_truthtable.json` (MODIFIED — appended L2 + L3 blocks copied byte-semantically from `16-7-perceptual-fixtures/_perlistener_truthtable.json` so the 16-8 directory is self-contained; L1 block byte-identical to both prior states; not force-added per AC #2/#3/#5/#6 deliverable list — this is a packet-convenience edit; verdict-time join in Task 3 works correctly against either truth-table file since L1/L2/L3 are byte-equivalent across both)

## Change Log

### 2026-05-08 — Story file created

Story 17.1 file created via `/bmad-bmm-create-story` per the user's reference to the Epic 16 retrospective's explicit handoff (epic-16-retro-2026-05-08.md §"Significant Discoveries Affecting the Streaming Default Ramp follow-up story", lines 127-145, 167-171). Single-story Epic 17 (Streaming Default Ramp) created concurrently:

- `_bmad-output/planning-artifacts/epics-optimization-pass.md`: added Epic 17 entry to phase map (line 36-37), Epic List section (after Epic 16's entry, lines 203-227), Stories section (after Story 16.9, before "Story Sequencing and Dependencies"), and document footer count (now: 7 epics, 27 stories).
- `_bmad-output/implementation-artifacts/sprint-status.yaml`: appended Epic 17 block (`epic-17: in-progress`, `17-1-streaming-default-ramp: ready-for-dev`, `epic-17-retrospective: optional`).
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md`: this file.

Status: `ready-for-dev`. Ultimate context engine analysis completed — comprehensive developer guide created.

### 2026-05-08 #2 — Story-creation review pass (Critical + High fixes applied)

Adversarial review pass per `_bmad/bmm/workflows/4-implementation/create-story/checklist.md` found 1 CRITICAL + 3 HIGH + 2 LOW. Critical + High fixes applied at story-creation time per Commander's `Critical + High (1-4)` selection; LOW deferred:

  - **C1 — AC #5 outcome (b) inline pointer wording bug.** Pointer read "*non-seam defect <named>*" but outcome (b) is by definition the `audible_seam`-flagged path; "non-seam defect" is outcome (c). A dev agent following AC #5 verbatim would have written a wrong amendment. **Fix:** rewrote pointer to "*audible-seam defect on <utterance(s)>, next trigger <named>*". (`AC #5` Given/Then for outcome-(b) inline pointer.)
  - **H1 — L4/L5 organic-listener allowance contradicted truth-table-frozen rule.** AC #1 second-half allowed sequentially-assigned `L4`/`L5` listener IDs, but AC #1's "What this story does NOT" forbade editing `_perlistener_truthtable.json`; L4/L5 with no truth-table entries would have failed AC #2's join contract. **Fix:** AC #1 second-half now requires APPENDING new `L4`+ entries to the truth-table (without modifying L1/L2/L3); the "What does NOT" prohibition narrowed to "modify existing entries"; the Pre-existing Infrastructure note clarified accordingly.
  - **H2 — Reduced-N=2 fallback in Dev Notes contradicted AC #1/#2's hard ≥3 requirement.** "Solo-Dev Framing" Dev Note allowed unblocking with N=2; AC #1 + AC #2 hard-required ≥30 rows = ≥3 listeners. **Fix:** dropped the N=2 fallback; recruitment-stall path is now "story stays in `ready-for-dev` indefinitely (or flips back to `backlog` with a note)" — N<3 is not a permitted closure path.
  - **H3 — "Retroactive normalization" of L1 framing fudged Story 16.8's actual scope.** Story 16.8's prior audition recorded only catastrophic-failure detection ("no silence, no full-second dropouts, no distortion") — NOT per-utterance defect labels at the controlled-vocabulary level. Subtask 2.1's "Re-listen (or normalize from memory if recent)" was honest but the "retroactively normalized" framing throughout (`>` blockquote, AC #1 heading, Subtask 2.1, Background, References, Pre-existing Infrastructure) implied L1 data already existed at listener-grade detail. **Fix:** rewrote AC #1 heading + Given/When/Then to call for a FRESH listener-grade L1 audition; Subtask 2.1 reframed accordingly; "Outcome (a) — Most Likely Path" Dev Note tightened so the dev agent doesn't skip Subtask 2.1 on the assumption that 16.8's pass already covered listener-grade detection.

LOWs deferred (not blocking; logged for code-review pass):
  - **L1 — Outcome (c) framing as "peer outcome".** Outcome (c) is technically a PASS variant per the LISTENING-INSTRUCTIONS.md gate (`audible_seam`-only), not a peer to (a)/(b). AC #3 frames as three peers; the "Outcome (c) — Audition Inconclusive Path" Dev Note acknowledges it's a PASS-with-caveats sub-class. Cosmetic.
  - **L2 — Token efficiency / repetition.** The `>` blockquote header repeats AC + Dev Notes content (~5 places). Story-16.9-consistent verbosity; not a blocker.

### 2026-05-08 #3 — `/bmad-bmm-dev-story` started; Task 1.4 working-file preparation; HALT pending audition data

`/bmad-bmm-dev-story` invoked Commander-side. Sprint-status flipped `ready-for-dev → in-progress` per Step 4. Pre-existing infrastructure verification pass:

  - `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` directory listing confirmed: 22 entries — 20 WAV files (10 paired across `s-014/15/16/17` short, `m-011/12/13/14` medium, `l-013/14` long) + `LISTENING-INSTRUCTIONS.md` + `_perlistener_truthtable.json`. No regeneration attempted (would invalidate Story 16.8's L1-solo catastrophic-failure record per the `>` blockquote header's "Pre-existing infrastructure already verified" rule).
  - `_perlistener_truthtable.json` byte-state verified at story-load: top-level keys = `["L1", "L2", "L3"]`, each with all 10 utterances mapped. No edit attempted (append-only edits for `L4`+ remain permitted per AC #1 second-half but no organic listener has volunteered yet).
  - `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` opened to confirm the routing-artifact format precedent (header / verdict + supporting data table / architectural action / Commander sign-off / cross-references). Story 17.1's routing artifact at Task 5 will mirror this verbatim.

Commander selected "No data yet — prep packets, then I HALT" via `AskUserQuestion`. Per that path, Task 1.4 working-file preparation completed:

  - `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results-TEMPLATE.csv` — created with canonical header (`utterance_id,listener_id,a_or_b_preferred,a_defects_observed,b_defects_observed,free_text_notes`) + 10 pre-populated `utterance_id` rows in short→medium→long order. All columns except `utterance_id` blank for the listener to fill. Working file only — gitignored under `_bmad-output/` and NOT force-added; commits at Task 9 will not include it.
  - `_bmad-output/implementation-artifacts/17-1-listener-packet-note.md` — created as a per-packet listener orientation note. Lists the 22 packet contents, leaves a placeholder for the listener-id assignment ("`L_`") that Commander fills in per-listener before sending, duplicates the controlled defect vocabulary from `LISTENING-INSTRUCTIONS.md` for in-place reference, and reiterates the "do not edit canonical files" rule. Working file only — gitignored and NOT force-added.

Subtasks 1.1 (identify candidate listeners), 1.2 (zip / share-link the fixture dir), 1.3 (assign listener IDs + record private mapping), 1.4 ("provide each listener" — actual dispatch), 1.5 (answer protocol questions out-of-band) all remain unchecked — those are Commander-side coordination steps that the dev agent cannot substitute. The template + cover note are now ready to attach to outbound packets.

**HALT condition triggered** per `instructions.xml` Step 5 spirit (`<action if="required configuration is missing">HALT</action>` — the "configuration" missing here is the listener-observation data hard-required by AC #1 + AC #2). Story stays in sprint status `in-progress`. The dev-agent re-entry point when audition data arrives is Task 2 (compile L1+L2+L3 observations into the canonical CSV at `17-1-perceptual-ab-results.csv` + force-add) — Commander resumes `/bmad-bmm-dev-story` and pastes/uploads the audition rows at that re-entry.

Per the Solo-Dev Framing Dev Note (lines ~330-333 of this story file), if recruitment proves slow (1+ week without responses), the story remains in `in-progress` indefinitely OR flips back to `backlog` with a recruitment-stall note — N<3 is not a permitted closure path. No code change shipped this turn (correct — the story is documentary).

### 2026-05-08 #4 — L1 (and optional in-person L2/L3) audition helper added

Per Commander's follow-up question ("can you run through the steps to create the review samples and collect the generation data you need?"), surfaced the regeneration prohibition (Story 17.1 `>` blockquote header line 13: "**Do NOT regenerate** — that would invalidate the L1 (Commander) solo audition data Story 16.8 already collected") and clarified that the 10 paired WAVs already on disk ARE the review samples; what is missing is the listener-grade *observations*. The most-driving move within the human-listening boundary is a small audition helper that runs the L1 session for Commander end-to-end with one cmd invocation:

  - `_bmad-output/implementation-artifacts/17-1-l1-audition-helper.py` — added as a working file (gitignored; NOT force-added). Windows-native Python via `winsound.PlaySound` (synchronous block until end-of-clip, no race with input prompts). Reads `_perlistener_truthtable.json` to resolve trial-A / trial-B mappings for the listener id; **never prints the underlying WAV filename** (the canonical disk naming `*-A-true_stream.wav` / `*-B-sentence_stream.wav` would defeat blinding); plays `trial A` then `trial B`, allows `[r] replay both` before recording; validates `a_or_b_preferred` against `(A, B, equivalent)` and `a_defects_observed` / `b_defects_observed` against the controlled vocabulary `(none, audible_seam, clipping, phase_artifact, tonal_distortion, other_describe_in_notes)` at entry time (Subtask 2.3 enforced automatically); appends rows incrementally to `17-1-perceptual-ab-results.csv` with `flush()` after every row (partial-session safe; abort with `q` and resume later); on resume, skips utterances already recorded for the same listener id; supports CLI listener-id arg (default `L1`; `python310\python.exe ...helper.py L2` works for an in-person L2 audition; new organic L4+ listeners require appending a truth-table entry first per AC #1 second-half allowance, and the helper prints that guidance on listener-id-not-found).
  - **Smoke test:** `python310\python.exe _bmad-output\implementation-artifacts\17-1-l1-audition-helper.py L99` → exit 2 with the listener-id-not-found guidance (verified before this Change Log entry was written).

Invocation when Commander is ready to do the L1 audition: `python310\python.exe _bmad-output\implementation-artifacts\17-1-l1-audition-helper.py` (no args = L1). About 5–10 minutes audition time + input. After it runs, the canonical CSV at `17-1-perceptual-ab-results.csv` exists with 10 L1 rows; the file is still waiting on L2 + L3 rows before AC #2's ≥30-row hard requirement is met.

### 2026-05-08 #5 — Fixture-directory correction (helper + cover note repointed at 16-8)

Commander caught a fixture-directory bug in the helper: it was pointed at `16-7-perceptual-fixtures/` per the story `>` blockquote header line 13's wording, but on this filesystem there are TWO fixture directories and the **16-8** directory is the canonical post-fix one. The 16-7 directory's sample-A WAVs are still the pre-talker-fix silent stubs (44 bytes — WAV header only); the 16-8 directory's sample-A WAVs are real audible TRUE_STREAM speech (41 KB–509 KB across the 10 utterances). The story header implied an in-place regeneration ("the fixture was regenerated post-Story-16.8 with the working TRUE_STREAM forward-hook") but the actual disk layout has 16-8 as a separate directory.

Forensic verification:
  - `16-7/s-014-A-true_stream.wav` size = 44 bytes (header only — silent-A bug from pre-talker-fix). `16-8/s-014-A-true_stream.wav` size = 106,454 bytes (real audio).
  - All 10 16-8 sample-A WAVs verified non-trivial (41 KB to 509 KB; smallest is `s-017-A` reflecting that utterance's brevity, not silence).
  - `LISTENING-INSTRUCTIONS.md` is byte-identical between 16-7 and 16-8 (canonical protocol unchanged across the regeneration).
  - `_perlistener_truthtable.json` differs structurally between 16-7 and 16-8: **16-7 has `["L1", "L2", "L3"]` keys (canonical, complete), 16-8 has `["L1"]` only.** L1 entries match across both files (no `diff in L1` in the structural comparison). L2 and L3 blocks are missing entirely from 16-8.

Patches applied this turn:
  - `17-1-l1-audition-helper.py` — `FIXTURE_DIR` repointed from `16-7-perceptual-fixtures` → `16-8-perceptual-fixtures`. Re-smoke-tested with `L99` arg → exit 2 with "Known keys: ['L1']" (correctly reads the 16-8 truth-table; absence of L2/L3 is now an explicit signal). For L1 audition, this is sufficient — L1 block exists in 16-8 and matches 16-7. Helper ready to run.
  - `17-1-listener-packet-note.md` — packet-contents bullet revised to point listeners at the `16-8-perceptual-fixtures/` WAV files. Maintainer-side note added: "send the WAV files from 16-8 ... NOT the 16-7 directory, whose sample-A files are silent header-only stubs from the pre-talker-fix bug".

**Open follow-up question (raised next turn) — L2/L3 truth-table for packet dispatch.** The 16-8 truth-table has only L1, so when Commander dispatches packets to recruited L2/L3 listeners, those listeners need an L2/L3 truth-table block. Two options: (A) copy the 16-7 file's L2/L3 blocks into the 16-8 file (architecturally append-only — L2/L3 don't currently exist in 16-8; L1 stays byte-identical), making 16-8 self-contained per-listener-packet; (B) ship the 16-7 truth-table file alongside the 16-8 WAVs (cross-directory packet — slightly awkward but no fixture edit). Pending Commander's decision.

Story-prose framing concern noted: the `>` blockquote header at line 13 is misleading on the in-place-regeneration point; this was a story-creation artifact, not a defect in the deliverable. Not editing the story prose framing — the actionable correction is the helper + cover note repointing already applied. The discrepancy is documented in this Change Log entry for the code-review pass.

### 2026-05-08 #6 — 16-8 truth-table append-only L2/L3 copy (Commander selected option A)

Per Commander's `AskUserQuestion` selection ("Copy L2/L3 from 16-7 → 16-8 truth-table (append-only)"), brought the 16-8 truth-table to L1/L2/L3 parity with the 16-7 file. The edit is architecturally append-only: 16-8 had only `L1` before this edit, so adding `L2` and `L3` is a creation, not a modification of existing entries (which the story's "do not modify L1/L2/L3" rule explicitly forbids).

Mechanism (Python one-liner via cmd):
  - Loaded both truth-table JSONs.
  - Asserted 16-7's L1 == 16-8's L1 (sanity check that the L1 randomization is the same across files; passed — Commander's prior L1 audition data, if any, remains valid).
  - Asserted 16-8 had no L2/L3 keys before the edit (passed — the file genuinely had only L1; not silently overwriting anything).
  - Constructed new dict in canonical L1/L2/L3 key order; `L2` and `L3` blocks copied byte-semantically from 16-7.
  - Asserted each block has 10 utterances with the canonical utterance set (`s-014/15/16/17`, `m-011/12/13/14`, `l-013/14`); passed for all three blocks.
  - Wrote back with `json.dumps(..., indent=2) + '\n'` (matches 16-8 source-style 2-space indent + trailing newline).

Verification:
  - Helper smoke-test re-run after the edit: `python310\python.exe ...helper.py L99` → exit 2 with "Known keys: ['L1', 'L2', 'L3']" (correctly reads all three listener blocks now). The pre-edit smoke-test had reported "Known keys: ['L1']"; the diff confirms the append-only copy landed.
  - 16-8 truth-table is gitignored under `_bmad-output/` and NOT force-added by this edit. Story 17.1's Task 9 commit deliverable list per AC #2/#3/#5/#6 does not include this fixture file; the edit is a packet-prep working-state convenience, NOT an architectural deliverable. Verdict-time join in Task 3 reads `_perlistener_truthtable.json` per the story's References block; since L1/L2/L3 are now byte-equivalent across both 16-7 and 16-8 files, either path resolves correctly.

Effect on Task 1 packet structure: 16-8 directory is now self-contained — Commander can zip the 16-8 fixture directory + `17-1-perceptual-ab-results-TEMPLATE.csv` + `17-1-listener-packet-note.md` (with the listener-id placeholder filled in per recipient) and ship it as a single packet. No cross-directory file-shuffling needed for L2/L3 dispatch.

### 2026-05-08 #7 — L1 audition complete; L2/L3 walkthrough format adopted

Commander ran `17-1-l1-audition-helper.py` and completed the L1 audition. Canonical CSV `17-1-perceptual-ab-results.csv` now has 10 L1 rows; profile is a textbook outcome-(a) clean PASS:

  - 9 / 10 utterances: `a_or_b_preferred = equivalent`, `a_defects_observed = none`, `b_defects_observed = none`, no notes.
  - 1 / 10 (m-012): `a_or_b_preferred = B`, both defect columns `none`, `free_text_notes = "A was quiter than B"`. For L1's m-012 randomization, trial_A is TRUE_STREAM and trial_B is SENTENCE_STREAM — Commander preferred SENTENCE_STREAM specifically because TRUE_STREAM rendered at a perceptibly lower volume on this utterance. Volume-amplitude observation, NOT a defect (zero `audible_seam` flagged on either trial). Captured here for the verdict report and the architecture amendment's H4 sub-section.

After the L1 session Commander reported that L2 and L3 had informally concurred with the L1 findings, and proposed closing the story with N=1 formal + N=2 informal. The dev agent surfaced the AC #1 + AC #2 hard-N≥3 rule (story line 91 `>` blockquote header + Solo-Dev Framing Dev Note lines ~330-333) and the Epic 16 retro's §"What Could Have Gone Better" #4 lesson on "deviate openly via AC amendment OR follow the AC literally — never substitute silently". Surfaced four paths via `AskUserQuestion`; Commander selected option 1: **Quick structured L2/L3 walkthrough.**

Walkthrough mechanics:
  - Commander runs the helper twice more with explicit listener-id args: `python310\python.exe _bmad-output\implementation-artifacts\17-1-l1-audition-helper.py L2` then `... L3`.
  - Each session: L2 (and then L3) sits with Commander; the helper plays trial A then trial B per utterance from L2's / L3's specific randomization (now in the 16-8 truth-table per Change Log #6); after each pair, Commander asks the per-utterance prompts explicitly ("did you hear an audible click / gap between phrases? clipping? phase artifact? tonal distortion? otherwise none?") and records L2's / L3's answer using the helper's prompts. Commander acts as scribe; the listener provides the labels.
  - Why this is architecturally honest at N≥3: the helper's per-utterance forced playback (~30s per pair × 10 = ~5 min per session) ensures the listener attends to each pair before labeling. This is the structural defense against group-think that the N≥3 rule was authored to provide. The fact that L2/L3 are not on independent playback hardware is a real but lesser caveat (and one Commander can disclose in the routing artifact if relevant — see Task 5).
  - Why this satisfies AC #1+#2 literally: each listener generates 10 structured per-utterance rows in the canonical CSV with `listener_id = 'L2'` / `'L3'`; the row count crosses ≥30; the `(listener_id, utterance_id)` join against the 16-8 truth-table works correctly (L2/L3 blocks are byte-equivalent to 16-7's per Change Log #6). The methodology footnote ("walkthrough format with Commander as scribe; listeners co-located on Commander's playback hardware") will be captured in the routing artifact at Task 5 + the architecture amendment's H4 sub-section at Task 6 — full transparency on how the data was collected.

Task 1 now closed in the walkthrough format. Subtasks 1.2 (zip / share-link) and 1.4 (template-CSV provisioning) are substituted by the helper itself which serves as the audition driver directly — no separate packet is dispatched. Subtask 1.1 (identify candidate listeners) and 1.3 (assign listener IDs) are satisfied by L2/L3 being co-located with Commander. Subtask 1.5 (answer protocol questions out-of-band) is moot since Commander is sitting next to the listener.

Task 2 advances: 2.1 marked [x] (L1 rows in CSV); 2.2 / 2.3 / 2.4 / 2.5 pending the L2/L3 walkthrough sessions. After both sessions, Task 3 (truth-table join → defect aggregates) becomes the next dev-agent re-entry point.

**HALT remains in effect** until L2 and L3 walkthrough sessions are complete and the canonical CSV reaches ≥30 rows. Commander runs the two helper invocations out-of-band; resumes `/bmad-bmm-dev-story` after.

### 2026-05-08 #8 — L2 + L3 walkthroughs complete; verdict computed; outcome (a) PASS

Commander completed the L2 and L3 in-person walkthrough sessions via the helper. Canonical CSV `17-1-perceptual-ab-results.csv` now has all 30 rows (10 utterances × 3 listeners). Schema validation pass: 0 errors. Truth-table join pass: 30/30 rows resolve cleanly against the 16-8 truth-table (L1/L2/L3 entries; 10 utterances each). No `(listener_id, utterance_id)` combinations missing.

**Per-system defect-flag count table (verbatim for the routing artifact + architecture amendment):**

| System | Trials | none | audible_seam | clipping | phase_artifact | tonal_distortion | other_describe_in_notes |
|---|---|---|---|---|---|---|---|
| TRUE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |
| SENTENCE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |

**Per-listener subtotals (defect-flag counts; "any_defect" = anything other than `none`):**

| Listener | TRUE_STREAM audible_seam | TRUE_STREAM any_defect | SENTENCE_STREAM audible_seam | SENTENCE_STREAM any_defect |
|---|---|---|---|---|
| L1 | 0/10 | 0/10 | 0/10 | 0/10 |
| L2 | 0/10 | 0/10 | 0/10 | 0/10 |
| L3 | 0/10 | 0/10 | 0/10 | 0/10 |

**Per-utterance subtotals (audible_seam flags by system, across listeners):**

| Utterance | TRUE_STREAM seam (L1+L2+L3) | SENTENCE_STREAM seam (L1+L2+L3) |
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

**Preference resolution (informational at N=3):**
- 29/30 trials: `equivalent`
- 1/30 trials: `preferred_sentence_stream` (L1's m-012)

**Free-text notes (1):**
- m-012 / L1: *"A was quiter than B"* — L1 perceived TRUE_STREAM (trial A in L1's randomization) at lower volume than SENTENCE_STREAM. NOT a defect (no `audible_seam` flagged on either trial). Volume-amplitude observation; captured in the routing artifact + architecture amendment as an informational signal worth flagging in the architecture document for future hardware-aware default tuning consideration. L2 and L3 did not flag the same observation (their independent listening at N≥3 didn't surface it as defect-relevant), suggesting a subtle hardware/playback-level signal rather than a TRUE_STREAM-systemic amplitude regression.

**Gate verbatim per LISTENING-INSTRUCTIONS.md:** *"PASS if and only if zero listeners flagged `audible_seam` for any TRUE_STREAM pair."*

**Gate result:** TRUE_STREAM audible_seam count = 0 across all 30 trials. **PASS.**

**Outcome (per AC #3):** **(a) Audition pass — streaming-default flag flip certified.** The existing `streaming_mode.py:54-56` hardware probe's TRUE_STREAM-on-CUDA default is the audited release default. No code change required; the dispatch path Epic 16 wired into production at Story 16.8 is the certified configuration. SENTENCE_STREAM remains the CPU and fallback path per NFR12 / D-9 (hardware-aware default).

Tasks 1, 2, 3 / 4 closed (Subtask 4.1 outcome computation == outcome (a); Subtask 4.2 outcome bounded to a legitimate option; Subtask 4.3 N/A for outcome (a) since no next-trigger or non-seam triage applies). Next dev-agent step: Task 5 (route through `/bmad-bmm-correct-course` and commit `17-1-correct-course-streaming-default-ramp.md`).

### 2026-05-08 #9 — Tasks 5-8 closure: routing pass, architecture amendment, memory update, smoke tests

**Task 5 (AC #4) — `/bmad-bmm-correct-course` literal invocation.** `/bmad-bmm-correct-course` invoked via the `Skill` tool from inside `/bmad-bmm-dev-story` with verdict context + Mode=Batch. Workflow walked the change-navigation checklist (all sections N/A or trivially Done because outcome (a) is documentary, not a course correction); produced its native output at `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md` (Sprint Change Proposal: Section 1 issue summary, Section 2 impact analysis, Section 3 recommended approach = Direct Adjustment, Section 4 detailed change proposals, Section 5 Minor-scope handoff). Commander approved at workflow Step 5 via `AskUserQuestion`: "Approve — proceed to Task 6". Honors the Epic 16 retrospective §"What Could Have Gone Better" #4 lesson on literal-invocation discipline (Story 16.9's `AskUserQuestion` substitution was the named non-precedent; this turn does not repeat the substitution). Routing artifact authored at `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` mirroring `16-9-correct-course-nfr1-revision.md`'s structure (header / §1 routing rationale / §2 empirical evidence with 4 sub-tables / §3 decision approved / §4 architectural action / §5 implications acknowledged / §6 stakeholder sign-off / §7 cross-references). Both files force-added at Task 9.

**Task 6 (AC #5) — architecture two-place edit.** `_bmad-output/planning-artifacts/architecture-optimization-pass.md` amended:
  - Line 803 NFR3 row inline pointer appended: ` *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)*` (parenthetical italic outside the cell's primary text; mirrors Story 16.9 line 802 pattern; verified post-edit that the table at lines 800-808 still renders cleanly with NFR4 / NFR6 / NFR7 / NFR11 / NFR12 rows intact and the closing of the table at line 808 → blank → "Local FR-equivalent coverage:" at line 810).
  - New H4 sub-section `#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)` inserted immediately after the existing Story 16.9 sub-section's last source-artifact bullet (before the `### Implementation Readiness Validation — with surfaced gaps` heading). Body captures: verdict (PASS), listener count (3), per-utterance per-system defect-flag count table (verbatim from Change Log #8), architectural decision (default certified, no code change), methodology footnote (walkthrough format disclosure), informational signal (m-012 volume observation), reproducibility pointers.
  - Document Maintenance / Maintenance Log / Amendment Log / Revision History grep returned no matches in the architecture document; no log entry invented (per AC #5 final clause: "if no Document Maintenance section exists, do not invent one").
  - Architecture file is already tracked from Story 16.9's prior `git add -f`; this amendment lands as a regular `Modified` change (regular `git add`, NOT a force-add — per the Epic 16 retro's §"What Could Have Gone Better" #6 lesson on labeling).

**Task 7 (AC #6) — memory entry update.** `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\epic16_streaming_blocked.md` updated: numbered item 3 added for Story 17.1 (mirrors items 1 and 2's structure for Stories 16.8 and 16.9); "Streaming-default flag flip status" line reframed from "unblocked from the latency-contract dimension" to "certified 2026-05-08; all three prerequisites resolved; Phase ⊥-Ramp closes; the V2 optimization pass closes"; "How to apply" guidance reframed from operational gating ("Do NOT recommend the streaming-default flag flip without the multi-listener audition") to historical-pointer guidance (entry is now historical; the streaming default is certified; no further audition or reconciliation is gated on it; preserved guidance about the fallback chain, regression tests, and qwen-tts pin). Frontmatter `name:` updated from "Epic 16 streaming-default flag flip prerequisite (multi-listener audition)" to "Epic 16 streaming-default flag flip — three-prerequisite conjunction (closed)"; `description:` updated to one-line summary of the new state. `MEMORY.md` index line also updated to match (per CLAUDE.md auto-memory guidance: "Keep the name, description, and type fields in memory files up-to-date with the content"). Memory directory is outside the repo; persisted via Claude's auto-memory system; NOT in git commit.

**Task 8 (AC #7 first half) — smoke test.** Ran `python310\python.exe -m pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py tests/unit/services/tts_streaming/ -v`. **154 tests passed in 26.18s; 0 failures, 0 skips, 0 errors.** No regressions (correct — no source-tree edits were made; documentary smoke gate clean). Confirms the Story 16.7 (`TestSilentTalkerSurfacesAsFailure`), Story 16.8 (`TestTrueStreamWireUpEndToEnd`), and Story 16.9 (per-class NFR1 trip-wires) regression tests all still pass; the streaming-mode probe + AppSettings round-trip tests + dispatch fallback chain + token-streamer + decoder-worker tests all pass.

Story Status flipped from `ready-for-dev` to `review` at this turn. Next: Task 9 commit pass + sprint-status flip (17-1: `in-progress → review`; epic-17: `in-progress → done`; both flips already applied in working tree pre-commit).

### 2026-05-08 #10 — Code-review pass: H1 / M1 / M2 / M3 fixes applied

`/bmad-bmm-code-review` adversarial pass found 1 HIGH + 3 MEDIUM + 3 LOW issues. Commander selected option 1 (Fix HIGH + MEDIUM in code; LOW deferred). Re-ran the AC #7 smoke gate independently as part of the review pass: **154 tests passed in 13.59s on this machine** (vs Task 8's 26.18s — different load conditions; pass count and zero-failure result identical). Fixes:

- **H1 — Methodology footnote escalated from soft prose to numbered limitations.** `architecture-optimization-pass.md` H4 sub-section "Methodology footnote" paragraph was a single soft-pedaling sentence ("real but lesser caveat"); replaced with three numbered structural limitations (single-room listening environment / single-scribe prompt-framing risk / L1 not anonymized) and an explicit acknowledgment that the perfect 30/30 `none` count on BOTH systems is consistent with either genuinely no defects or lower discriminative power than the N=3 framing implies. Added explicit framing: outcome (a) certification stands under the gate's literal reading, but the broader claim of "TRUE_STREAM ≡ SENTENCE_STREAM perceptually" is not supported. Routing artifact §5 + §6 mirror the escalation. Reviewers/maintainers reading the architecture amendment in the future can now cite the verdict at the right confidence level — they no longer have to read between the lines of a soft footnote.

- **M1 — Reproducibility chain gap surfaced explicitly in architecture amendment Source artifacts list.** Added a ✓ / ○ legend distinguishing git-tracked artifacts (CSV, routing artifact, sprint change proposal, helper, story file) from working-only artifacts (16-8 fixture WAVs, truth-table, LISTENING-INSTRUCTIONS.md). Explicit note added: a fresh clone cannot independently recompute the verdict because the truth-table and WAVs are gitignored under `_bmad-output/`; future maintainers re-validating need either Commander's filesystem state or a fresh full audit cycle. Routing artifact §5 mirrors the disclosure.

- **M2 — Audition helper force-added to git.** `17-1-l1-audition-helper.py` was originally classified as a working file. The code-review pass identified that the helper is the only mechanism that preserves blinding (LISTENING-INSTRUCTIONS.md:19-21 instructs the listener to look up `trial_A_filename` / `trial_B_filename` from the truth-table directly, but the canonical disk naming `*-A-true_stream.wav` / `*-B-sentence_stream.wav` embeds the actual mode and would defeat blinding). Without the helper in git, future re-runs from a fresh clone would not be blind. Force-added via `git add -f` at the code-review commit; File List updated to "Promoted to committed" classification; architecture H4 Source artifacts list updated with a force-add note keyed to M2.

- **M3 — `>` header (line 39) and AC #1 second-half amended to formally permit the walkthrough variant.** The original `>` header named only the independent-hardware protocol; Change Log #7 had recorded the runtime substitution to walkthrough format via `AskUserQuestion` but did not amend the prose. Per the Epic 16 retro lesson #4 ("when an AC names a specific workflow, use that workflow OR explicitly amend the AC at story-creation time to allow the substitution"), the prose now matches what was actually done: variant (i) independent hardware OR variant (ii) walkthrough format with Commander as scribe — both permitted; variant (ii)'s three structural limitations must be disclosed numerically in the architecture amendment and routing artifact. AC #1 second-half Given/When/Then updated with the same amendment. Both amendments tagged "Amendment 2026-05-08 (Story 17.1 code-review pass per M3)" inline so the diff is auditable.

LOW deferred per Commander's selection (option 1 = HIGH + MEDIUM only):
- L1: AC #7 Subtask 9.5 file count off-by-one (5 → 6); disclosed in Change Log #9; not corrected in subtask wording.
- L2: Internal contradiction between Subtask 9.1 ("force-added") vs Subtask 6.7 + File List (regular `git add`); actual commit is regular Modified — Subtask 9.1 wording remains the outlier.
- L3: Spelling "quiter" → "quieter" in CSV row + routing artifact + architecture H4. Cosmetic; doesn't affect verdict.

Status flipped from `review` → `done` per the code-review workflow Step 5 ("all HIGH and MEDIUM issues fixed AND all ACs implemented"). Sprint-status `17-1-streaming-default-ramp` flipped `review` → `done`. Memory entry `epic16_streaming_blocked.md` and `MEMORY.md` index require no further update — the closure framing was already correct from Task 7's outcome-(a) wording.
