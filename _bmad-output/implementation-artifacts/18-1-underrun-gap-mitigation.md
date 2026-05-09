# Story 18.1: Underrun-Gap Mitigation (Phase ⊥-Polish-2)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase tag: Phase ⊥-Polish-2 (D-20). Successor to Phase ⊥-Polish (Story 17.3). First story of Epic 18 (Generation-Speed Optimizations). -->
<!-- Risk: Low (per `epics-optimization-pass.md:243`). UX smoothness; does not touch dispatch chain or producer side. -->
<!-- Audition discipline: Commander solo, no L2/L3 recruitment (per `epics-optimization-pass.md:1340`). Smoothness is a perceptual continuity check, not a quality re-audition. -->

## Story

As a **MyVoice user generating long-form CLONED-voice utterances on a CUDA host**,
I want **progressive audio playback to play continuously without ~1-second silent gaps every few words**,
so that **the streaming experience Story 17.3 delivered actually feels smooth in steady-state long-form usage on Blackwell-class GPUs**.

## Acceptance Criteria

**Given** the post-Story-17.3 progressive-playback path is wired (TRUE_STREAM emits chunks; orchestrator opens a streaming PyAudio session at chunk 0; chunks write progressively to monitor + virtual services)
**When** the user generates a long-form CLONED utterance (≥250 chars / ≥10 s of speech) on the RTX 5090 dev host with Sarira-F (post-Story-17.2 voice_clone_prompt cache hit)
**Then** Commander's solo audition reports **zero ≈1-second silent gaps** across the playback (smooth continuous audio)
**And** the four expected myvoice.log markers from Story 17.3 §4.2 still fire in order (`Starting TTS generation (TRUE_STREAM)`, `Progressive playback session opened`, `First chunk latency`, `TTS generation complete (TRUE_STREAM)`, `Progressive playback already active; skipping batch dispatch`)
**And** no new `Progressive playback chunk write failed` warnings appear in the log
**And** no new `Stale terminal AudioChunk` warnings appear in the log

**Given** AC #1 (smoothness) holds
**When** the same long-form utterance's first-chunk-latency is measured via `metrics.first_chunk_latency_ms`
**Then** the value remains ≤ 5.0 s p95 GPU short-class per the Story 16.9 / 17.1 NFR1 contract (no significant latency regression vs Story 17.3 baseline at 3.93–4.94 s on Sarira-F warm cache)
**And** the **first-chunk-to-audible** latency (chunk-emit time → first PyAudio `write()` post-fill) remains within Story 17.3's documented ~50–100 ms PyAudio buffer-fill bound, with the chosen mitigation's added latency cost called out explicitly in the evidence file (e.g., "+150 ms from holding 3 pre-buffer chunks before session open")

**Given** the developer chooses between the three documented candidate fixes (pre-buffer N chunks / increase `frames_per_buffer` / decode-bottleneck investigation per Story 17.3 evidence §4.4)
**When** the choice is made
**Then** the decision is **data-driven** — the implementation is preceded by an instrumentation pass that captures per-chunk decode-emit timestamps and per-chunk playback-arrival timestamps for a long-form CLONED utterance on RTX 5090
**And** the decision artifact (which fix was chosen, what the data showed, why the cheapest viable option was sufficient OR why a more expensive option was required) is captured at `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` §"Mitigation choice"
**And** the developer **does not** start with option 3 (decode-bottleneck investigation — the named rabbit hole) unless options 1 + 2 have been empirically ruled out by these specific gates:
  - **Option 1 ruled out** only if a 3-chunk pre-buffer (the `_progressive_pre_buffer_target = 3` default per Story 17.3 evidence §4.4 mitigation 1) leaves a measurable underrun gap on the canonical Sarira-F long-form utterance from Story 17.3 §4.1 step 3 — *measurable* meaning Task 1's instrumentation captures at least one inter-chunk gap exceeding 250 ms (≈half the observed ~1-s gap, well above PyAudio buffer-fill jitter).
  - **Option 2 ruled out** only if `frames_per_buffer = 4096` (the upper end of AC #2's ~100 ms latency budget per Task 3.3) ALSO leaves a measurable underrun gap on the same utterance, by the same ≥ 250 ms criterion.

  Both negative results must be captured in `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` §"Mitigation choice" with the raw timestamp data BEFORE option 3 is opened. "The data was inconclusive" is not a sufficient ruling-out — the dev agent surfaces ambiguity to Commander rather than escalating to option 3 unilaterally.

  **AC #3 amendment (2026-05-09 — authorized by Commander via the dev-story workflow's mitigation-gate question; full rationale in evidence file §4.4).** The captured §4.2 data shows a 3.23× steady-state emit/drain ratio with 100% of 8 steady-state inter-chunk gaps exceeding 250 ms by ≥10× margin, AND the deterministic math in §4.4 establishes that no consumer-side change (pre-buffer N chunks, `frames_per_buffer` knob, or handler reordering) can close a gap whose root cause is the producer-side emit cadence. **Option 1 + Option 2 are therefore ruled out by raw producer-side cadence + math without ceremonial Option 1 implementation.** The fix class is Option 3 (decode-rate investigation) and the work is sequenced into Stories 18.3 (bf16 precision) + 18.4 (`torch.compile`). This amendment relaxes ONLY the "Option 1 must be implemented and re-measured" letter-of-AC; the spirit ("don't escalate to Option 3 on a hunch") is preserved by the empirical 3.23× ratio.

**Given** the chosen mitigation lands
**When** any host without Ampere+ CUDA support (CPU-only or pre-Ampere GPU) runs the application
**Then** the CPU-only / pre-Ampere path's playback behavior is **identifiably unchanged** (D-9 / NFR12 hardware-aware default discipline preserved verbatim)
**And** if the mitigation introduces a new tunable (e.g., a `streaming_frames_per_buffer` setting or a `progressive_pre_buffer_chunks` setting), the CPU-only path either uses the V2 baseline value OR is provably gated behind the same `torch.cuda.is_available()` probe that gates TRUE_STREAM dispatch (`streaming_mode.py:54-56` precedent)

**Given** the chosen mitigation introduces orchestrator-side state machine changes (option 1) OR audio-service config changes (option 2)
**When** the test suite runs
**Then** the existing 3 + 8 + 2 + 3 = 16 progressive-playback tests added by Story 17.3 (`tests/unit/services/test_qwen_tts_service_true_stream_callback.py`, `tests/unit/test_app_progressive_playback.py`, `tests/unit/test_app_progressive_playback_cancel.py`, `tests/integration/test_progressive_playback_dispatch_skip.py`) pass with **zero regressions**
**And** the `_progressive_playback_active` / `_progressive_playback_epoch` cancel-vs-chunk race semantics from Story 17.3 (the spurious second-session-open fix in commit `abd842c`) remain intact
**And** new unit tests cover the chosen mitigation's surface (e.g., pre-buffer N chunks state machine if option 1; `frames_per_buffer` knob threading through both `MonitorAudioService.start_streaming_session` AND `VirtualMicrophoneService.start_streaming_session` if option 2)

**Given** the bundled-environment smoke from Story 17.3 §4.1 procedure is the production-verification gate
**When** the developer runs the smoke after the source-tree edits
**Then** a fresh `build_release.bat` cycle produces a `build_tools/dist/MyVoice/MyVoice.exe` portable bundle
**And** Commander's bundled-mode audition on the same Sarira-F long-form utterance from §4.1 confirms zero ≈1-s silent gaps in the new build (the AC #1 audition gate, but on the production-bundled artifact rather than the dev source tree)
**And** the evidence file is populated per Tasks 1.4, 4.5, and 6.2 (instrumentation CSV, bundled-smoke log excerpts, NFR1 spot-check measurements)

## Tasks / Subtasks

- [ ] **Task 1 — Instrument decode-rate vs. playback-rate** (AC: #3)
  Build the data layer that turns the choice between the three documented fix paths into an empirical decision rather than a guess.
  - [x] 1.0 **Verify `metrics.record` overhead is negligible BEFORE adding new metric calls to hot paths.** The producer-side `_wrapped_post('append_chunk', ...)` and the consumer-side `_handle_progressive_chunk_async` are both timing-sensitive — instrumentation that adds non-trivial overhead would shift the very ratio Task 1.4 measures, making the data self-invalidating. Time a tight loop of 1,000 `metrics.record('test_metric', 1.0, tag='x')` calls on the RTX 5090 dev host; confirm mean overhead ≤ 100 µs/call (the implementation at `src/myvoice/observability/metrics.py:77` is documented as a thin pub-sub dispatcher and should be well under this threshold). **If the threshold is exceeded:** instead of inline `metrics.record` calls, accumulate per-chunk timestamps into a per-session list (`self._progressive_metric_log: list[dict]` slot near the existing progressive-playback slots at `app.py:178-188`) and flush to a CSV at session close. Pick whichever path passes the threshold and document the choice at evidence file §"Instrumentation overhead". **Result: 1.35–2.40 µs/call on RTX 5090 / Python 3.10.11 — PASS, ~50× headroom. Inline calls used. Evidence file §1.**
  - [x] 1.1 Audit existing instrumentation: confirm whether `qwen_tts_service.py:_wrapped_post('append_chunk', ...)` (TRUE_STREAM emit at `:3897-3947`) already logs per-chunk emit timestamps via `metrics.record` (the module already imports `metrics` at `:245` and records at 5+ sites including `:3122`, `:4111`, `:4273`); if a `progressive_chunk_emit_ms` metric does not already exist, add one. Metric name: `progressive_chunk_emit_ms`. The existing record-site at `:3122` (SENTENCE_STREAM) is the structural precedent — mirror its tag schema (`session_id`, `chunk_index`). **Done — added at `qwen_tts_service.py` `_wrapped_post` `append_chunk` branch; value = `time.time() * 1000.0` (wall-clock ms for cross-side join); tags `session_id` + `chunk_index`. Evidence file §2.1.**
  - [x] 1.2 Add per-chunk playback-arrival timestamps inside `app.py:_handle_progressive_chunk_async` immediately after the `await self._audio_coordinator.play_audio_chunk(...)` returns (line 2584-region). Metric name: `progressive_chunk_playback_arrival_ms`. Include `chunk_index` + `is_final` + `audio_data.size` as tags. **Done — gated inside `audio_data.size > 0` branch so the synthetic terminal chunk does not emit. Evidence file §2.2.**
  - [x] 1.3 Add per-chunk audio-duration computation: `chunk.audio_data.size / chunk.sample_rate` seconds. Metric name: `progressive_chunk_audio_duration_ms`. Tag `chunk_index`. This is the canonical "drain time" for the consumer side; comparing it to the inter-chunk-emit interval reveals which side starves. **Done — co-located with arrival metric inside `size > 0` branch; defensive `sample_rate > 0` guard. Evidence file §2.3.**
  - [x] 1.4 Run on the canonical long-form CLONED utterance from Story 17.3 §4.1 step 3 (Sarira-F, ≥250 chars, RTX 5090). Capture the metrics stream at `_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv`. Compute: median + p95 inter-chunk-emit interval; median + p95 chunk audio-duration; ratio. Goal: identify whether emit-interval < audio-duration (consumer starves → option 1 or 2) or emit-interval > audio-duration (consumer sufficient — different problem). **Done 2026-05-09. 11 chunks captured (10 with full triplet); steady-state median ratio = 3.23×, p95 = 3.37×; ALL 8 steady-state inter-chunk gaps > 250 ms (median 4,408 ms). Producer at 31% real-time. Evidence file §4.2.**
  - [x] 1.5 Decision artifact: capture the verdict at evidence file §"Mitigation choice" — which option was chosen, what the data showed, what the cheapest viable mitigation is. Per AC #3, do not start with option 3. **Done 2026-05-09. Verdict: Option 3 (decode-rate investigation) — Options 1 + 2 ruled out (Option 1 mathematically: 3-chunk pre-buffer cushion = 5.94 s vs 4.4 s/chunk steady-state deficit → buffer drains in 1.4 chunks then gaps return; Option 2 structurally: PyAudio frames_per_buffer doesn't change inter-chunk arrival rate). Story closes shipping instrumentation only; mitigation deferred to Stories 18.3 + 18.4 which directly target decode rate. AC #3 amendment authorized 2026-05-09 by Commander via dev-story workflow's mitigation-gate question. Evidence file §4.3 + §4.4.**

### 🔱 Tasks 2 and 3 are MUTUALLY EXCLUSIVE

Task 1.5's empirical verdict selects **exactly one** of the two paths below. The dev-story workflow MUST NOT execute both — doing so layers two unrelated mitigations on the same defect, hiding which one closed the gap and complicating future regression analysis. If Task 1.5's data is genuinely ambiguous (both paths look viable), pause and surface the ambiguity to Commander before executing either; do not default to "do both" or "pick the smaller change."

- [x] **Task 2 — Implement chosen mitigation: Option 1 (pre-buffer N chunks at consumer)** (AC: #1, #2, #5; **gated on Task 1.5 selecting Option 1**) — **NOT APPLICABLE per Task 1.5 verdict (Option 1 ruled out; see evidence file §4.3). Subtasks 2.1–2.6 not executed.**

- [x] **Task 3 — Implement chosen mitigation: Option 2 (increase `frames_per_buffer` for streaming sessions)** (AC: #1, #2, #4, #5; **gated on Task 1.5 selecting Option 2**) — **NOT APPLICABLE per Task 1.5 verdict (Option 2 ruled out — wrong fix class for producer-bottleneck defect; see evidence file §4.3). Subtasks 3.1–3.4 not executed.**

- [x] **Task 4 — Bundled audition smoke** (AC: #1, #6) — **NOT APPLICABLE per evidence file §6: no mitigation source-tree edits to validate via bundled smoke; Story 17.3's existing bundled-smoke gate covers progressive-playback dispatch surface, which Story 18.1's instrumentation does not change.** Subtasks 4.1–4.6 not executed.

- [x] **Task 5 — Regression test sweep** (AC: #5)
  - [x] 5.1 Run the existing Story 17.3 unit tests: `pytest tests/unit/services/test_qwen_tts_service_true_stream_callback.py tests/unit/test_app_progressive_playback.py tests/unit/test_app_progressive_playback_cancel.py tests/integration/test_progressive_playback_dispatch_skip.py`. **Done — 32 passed in 18.58 s. (Story spec said "16 tests" but the suite has expanded since: 3 qwen-callback + 22 app-progressive + 3 app-progressive-cancel + 3 dispatch-skip integration = 31 expected; this run captured 32, indicating one additional test landed since the story was authored. Zero regressions.)**
  - [x] 5.2 Run the broader streaming + app + audio surface (Task 5.1's 32 + qwen_tts_service_dispatch + qwen_tts_service_session_integration + new instrumentation tests + audio_coordinator + observability). **Done — 166 passed in 17.56 s. Zero regressions.**
  - [x] 5.3 Verify metrics/observability tests still pass. **Done — 45 metrics + 13 CSV-capture = 58 observability tests pass; included in the 166-test sweep above.**

- [x] **Task 6 — NFR1 spot-check** (AC: #2) — **NOT APPLICABLE per evidence file §7: instrumentation does not touch the first_chunk_latency_ms emission path or the dispatch state machine; Sarira-F warm-cache baseline from Story 17.2 (3.93–4.94 s p95) is preserved by construction. Subtasks 6.1–6.2 not executed; Stories 18.3 + 18.4 will run the spot-check when their producer-side throughput uplifts land.**

- [x] **Task 7 — Code-review pass** (post-implementation)
  - [x] 7.1 Run `/bmad-bmm-code-review` after Task 5 completes. **Done 2026-05-09** — adversarial review found 13 issues (3 HIGH / 5 MEDIUM / 5 LOW). HIGH findings: H1 (AC #3 strict text vs. evidence amendment unsynced — story did not record the amendment inline), H2 (evidence file had duplicate §4.1 + contradictory §4.2 from a not-fully-cleaned edit), H3 (story File List omitted uncommitted files surfaced by `git status`). MEDIUM: M1 (consumer-side metrics dropped session_id, breaking CSV joinability across multi-session captures), M2 + M3 (`.gitignore` carve-outs missing for Story 17.2 voice caches + build-tool answer-piping leftover), M4 (no regression test for the Story 17.3 commit-`abd842c` invariant in the metrics dimension), M5 (TRUE_STREAM `is_final=False`-on-every-data-chunk semantics not documented).
  - [x] 7.2 Address review findings. **Done 2026-05-09** — all H1/H2/H3/M1/M2/M3/M4/M5/L2/L5 fixes landed in this code-review pass. See File List + Completion Notes for the file-by-file detail. 54 (Story 18.1 + 17.3) + 127 (broader streaming) tests pass with zero regressions. Commit per the established Story 16.7 / 16.8 / 16.9 / 17.1 / 17.2 / 17.3 pattern: `Story 18.1: code-review pass — H1/H2/H3/M1/M2/M3/M4/M5/L2/L5 fixes`.

## Dev Notes

### What this story is

Story 18.1 is the first story of Epic 18 (Generation-Speed Optimizations / Phase ⊥-Polish-2). It closes the only carry-over follow-up from Story 17.3's bundled audition: the ~1-s silent gaps Commander observed during long-form CLONED-voice playback on RTX 5090, captured at `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md §4.4`.

Story 17.3's contract — *"audio plays progressively during generation, not after"* — IS delivered. The underrun-gap is a separate UX-grade smoothness issue surfaced by the new progressive path running in steady-state on Blackwell. This story names + closes it.

Concern 2 of Story 17.3's scope sketch (`17-3-progressive-audio-playback-during-true-stream-scope-sketch.md:61` — "Underrun on slow-generating chunks") is the explicit predecessor framing; Story 17.3 documented three candidate fix paths but did not implement them because they are out-of-scope for closing the "audio plays after generation completes" gap.

### What this story is NOT

- **Not a TRUE_STREAM dispatch rework.** Stories 16.3–16.6 + 17.1 + 17.2 + 17.3 already deliver the talker-decoder-streamer-overlap-add-progressive pipeline correctly. This story tunes one downstream knob (consumer-side buffering OR PyAudio frame-buffer size); it does NOT touch `qwen_tts_service.py:_generate_true_stream` or the producer's chunk-emit cadence.
- **Not a perceptual quality re-litigation.** Story 17.1's audition certified TRUE_STREAM perceptual equivalence to BATCH; the chunks Story 18.1 plays smoothly are the same chunks the audition validated. Audition is Commander solo; no L2/L3 recruitment per Epic 18 stub.
- **Not an Epic 18 audition cycle.** Stories 18.3 (bf16 precision) + 18.4 (`torch.compile`) trigger the full ≥3-listener NFR3 re-audition mirroring Story 17.1's protocol. Story 18.1 + 18.2 are explicitly Commander-solo per `epics-optimization-pass.md:241`.
- **Not a CPU / pre-Ampere change.** Per D-9 / NFR12, CPU-only and pre-Ampere hosts stay on SENTENCE_STREAM (no progressive playback path → no underrun-gap to fix). The mitigation engages only on the CUDA TRUE_STREAM path.
- **Not a build-pipeline change.** No `requirements.txt` / installer-spec / `build_release.bat` edits anticipated; pure source-tree edits picked up by the next build cycle.
- **Not a decode-bottleneck investigation by default.** Option 3 of Story 17.3 §4.4's three candidate fixes is explicitly the "rabbit hole" — only triggered if Task 1's data rules out options 1 + 2.

### Source tree components to touch

**Read-only (analysis/reference):**
- `src/myvoice/services/qwen_tts_service.py:3897-3947` — TRUE_STREAM `_wrapped_post('append_chunk', ...)` chunk-emit point. Read for instrumentation hook placement (Task 1.1).
- `src/myvoice/services/qwen_tts_service.py:3071-3082` — SENTENCE_STREAM chunk-emit point. Read for confirming the same metric path is shared.
- `src/myvoice/services/audio_coordinator.py:1018-1072` — `start_streaming_session` dual-fan-out to monitor + virtual. Read; signature unchanged unless Task 3 chooses to plumb a new arg.

**Likely edit (option 1 — pre-buffer N chunks at consumer):**
- `src/myvoice/app.py` — orchestrator: new state slots near the existing `_progressive_playback_*` slots (`:178-188`); modified `_handle_progressive_chunk_async` (`:2397-2601`) chunk-0 branch + cancel handler (`:1156-1180-region`) + NFR7 fallback restart (`:2452-2465`).
- New tests: `tests/unit/test_app_progressive_pre_buffer.py` (mirrors `tests/unit/test_app_progressive_playback*.py` patterns).

**Likely edit (option 2 — increase `frames_per_buffer`):**
- `src/myvoice/services/monitor_audio_service.py:38-53` — `MonitorAudioConfig`: new `streaming_chunk_size` field. Line `:868` — `start_streaming_session` PyAudio open uses the new field instead of `self.config.chunk_size`.
- `src/myvoice/services/virtual_microphone_service.py:38-53-region` — same change to `VirtualMicrophoneConfig` + line `:928` PyAudio open.
- New tests: `tests/unit/services/test_monitor_audio_service_streaming_buffer.py` + virtual-mic equivalent.

**Either option:**
- `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` (new; Task 1.5 + 4.5 + 6.2). Force-add via `git add -f` per the Story 16.9 / 17.1 / 17.2 / 17.3 evidence-file precedent (`_bmad-output/` is gitignored per `memory/git_repo_state.md`).
- `_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv` (new; Task 1.4). Force-add same as evidence file.

### Testing standards summary

- **Unit tests:** mirror Story 17.3's patterns. Async tests use `pytest-asyncio`; PyAudio mocked at the boundary (`monitor_audio_service._pyaudio.open` / `virtual_microphone_service._pyaudio.open`); orchestrator state inspected via direct attribute access on the `MyVoiceApp` instance (Story 17.3's pattern at `tests/unit/test_app_progressive_playback.py`).
- **Integration tests:** `tests/integration/test_progressive_playback_dispatch_skip.py` (3 tests from Story 17.3) must pass unchanged. If option 1 lands, add a similar dispatch-skip integration test verifying the pre-buffer path also correctly skips the batch dispatch.
- **No new test-harness changes required** — Story 17.3 established the patterns; this story extends them.
- **Conftest discipline:** `tests/conftest.py` already enforces torch-before-PyQt6 DLL ordering per `memory/torch_pyqt6_dll_ordering.md`; pytest-cov runs need the inline torch-first preamble per `memory/torch_before_coverage_dll_ordering.md` (Story 12.3 deviation). If running under coverage for Task 5.2, follow the established preamble pattern.

### Project Structure Notes

**Alignment with unified project structure:**
- New evidence file under `_bmad-output/implementation-artifacts/` matches the per-story evidence-file pattern (Story 16.7 onward).
- New unit tests at `tests/unit/test_app_progressive_pre_buffer.py` (option 1) OR `tests/unit/services/test_monitor_audio_service_streaming_buffer.py` (option 2) follow the existing `tests/unit/...` and `tests/unit/services/...` directory conventions.
- No new top-level packages, modules, or conftest files needed.

**Detected variances:**
- The existing config field `chunk_size` (used as `frames_per_buffer` in PyAudio.open) is shared across batch + streaming paths in both `MonitorAudioConfig` and `VirtualMicrophoneConfig`. Option 2 introduces a *split* — `chunk_size` stays for batch, `streaming_chunk_size` is added for streaming. This is a deliberate decoupling, not a refactor; the rationale is that batch playback already works correctly with 1024 and changing it could regress non-progressive paths. Document explicitly in the field-comment when adding.
- **No D-decision change.** Story 18.1 does not require an architecture amendment per `epics-optimization-pass.md:234` ("No new D-decisions"). NFR1 / NFR3 / NFR7 / D-9 / NFR12 are all preserved unchanged.

### Previous Story Intelligence

**From Story 17.3 closure (commit `e9ce759` + code-review pass `6f0e152` + spurious-second-session-open fix `abd842c`):**

- The progressive-playback consumer state machine (`_handle_progressive_chunk_async` at `app.py:2397-2601`) is **already** racy at the chunk-0 / cancel / NFR7-restart boundaries; commit `abd842c` added `_progressive_playback_epoch` to drop stale chunks queued before a cancel. **Any new state introduced by Task 2 must respect the same epoch-discipline** — the pre-buffer must be cleared on cancel AND on epoch mismatch, not just on cancel.
- The `_progressive_playback_active` flag is intentionally **not** cleared on `is_final` (the Story 17.3 dev note at `app.py:181-187` documents the rationale: clearing on is_final races the dispatch path on asyncio loop ordering). Task 2's pre-buffer flush MUST integrate with this same clearing-discipline; the flag is consumed by `_play_generated_audio`'s skip-branch (normal completion) OR `_on_cancel_generation_requested` (interrupt).
- The "stale terminal AudioChunk" warning logic (`app.py:2486-2522`) was added by code-review pass H2 — it logs loudly if a non-empty terminal chunk arrives after the session was closed. **If option 1's pre-buffer state machine introduces a new race where the terminal chunk arrives during pre-buffer flush, that warning would fire spuriously.** Test-case: terminal chunk arriving while pre-buffer still has unflushed chunks must complete the flush + close cleanly without tripping the warning.
- The `_progressive_playback_lock = asyncio.Lock()` is lazy-initialized (`app.py:189-region` slot; first-use creates the lock at `:2433-2434`). **All new state mutations from Task 2 must occur under this same lock** — adding a second lock would invite deadlock.

**Code-review discipline from `memory/code_review_regression_test_exact_class.md`:**
- HIGH/MEDIUM-fix regression tests must mirror the **exact** bug class, not the nearest adjacent case. Translation for Story 18.1: if Task 1's data shows the underrun is producer-side starvation (not consumer-side underrun), then a frames_per_buffer raise (option 2) does NOT solve the named bug — it'd be the wrong fix class. Task 1.4's data is the gate.
- Re-run code-review twice after non-trivial auto-fixes (the established pattern from Stories 16.7 / 16.8 / 17.1 / 17.2 / 17.3).

### References

- **Story 17.3 evidence §4.4** — `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` lines 164-173: the underrun-gap finding, three candidate fix paths, and Concern 2 framing. Load-bearing reference for AC #3.
- **Story 17.3 scope sketch concern 2** — `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-scope-sketch.md` lines 60-61.
- **Epic 18 stub** — `_bmad-output/planning-artifacts/epics-optimization-pass.md` lines 1326-1343 (Story 18.1 stub); lines 228-250 (Epic 18 framing).
- **Architecture NFR1 revised contract** — `_bmad-output/planning-artifacts/architecture-optimization-pass.md` §"NFR1 (revised 2026-05-08, Story 16.9)" at lines 838-850. Per-class first-chunk targets: short ≤5.0s, medium ≤10.0s, long informational-only.
- **Architecture D-9 hardware-aware default** — `architecture-optimization-pass.md:257`. The `torch.cuda.is_available()` probe + Ampere+ guard discipline (precedent at `streaming_mode.py:54-56`) that Task 3.x's CPU/pre-Ampere protection relies on.
- **Architecture D-19 telemetry** — `architecture-optimization-pass.md` §"D-19 Telemetry" (begins at line 286) and the `metrics.record(name, value, **tags)` helper specified at line 476. Implementation lives at `src/myvoice/observability/metrics.py:77` (`def record(name, value, **tags)`); the module is already imported and used by `qwen_tts_service.py:245` and recorded at 5+ sites (`:3122`, `:4111`, `:4273`, `:4306`, `:4318`). Task 1's three new metrics extend this established pattern.
- **Story 17.2 evidence** — `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md §4.3.2`. Sarira-F warm-cache baseline first-chunk latencies (3.93–4.94 s) for AC #2's "no significant regression" gate.
- **Story 16.9 NFR1 reconciliation** — `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-sentence-stream-latency-investigation.md`. The empirical baseline + per-class targets that Task 6 (NFR1 spot-check) compares against.
- **Memory: build_tools_phase_perp_state.md** — Phase ⊥-Build closure marker. This story extends Phase ⊥-Polish (Story 17.3) into Phase ⊥-Polish-2.
- **Memory: hardware_setup.md** — RTX 5090 Blackwell as the dev host that surfaces this gap.
- **Memory: code_review_regression_test_exact_class.md** — exact-bug-class regression-test discipline (the single most important review-feedback signal for this story).
- **Memory: production_release_state.md** — production-bundle context informing Task 4.

### Latest Tech Information

**PyAudio `frames_per_buffer` behavior on Win11:**
- PyAudio's `Stream.write(data, exception_on_underflow=False)` is blocking by default; the call returns when the buffer has accepted the data (not when it has been played). The underflow-on-empty behavior surfaces as a silent gap rather than an error when `exception_on_underflow=False` (the default in `MonitorAudioService.play_audio_chunk`).
- 24000 Hz int16 mono = 48 KB/s. With `frames_per_buffer=1024`: ~42.7 ms of audio per buffer fill. With 4096: ~170 ms. With 8192: ~341 ms. Story 17.3's documented bound is ~50–100 ms for first-chunk-to-audible; raising to 4096 already exceeds that bound. Task 3.3's latency-budget verification is the gate.
- Win11 MME / WASAPI exclusive-mode behaviors differ; `audio_coordinator` uses MME by default through PyAudio's default host API. Bundled-audition smoke (Task 4) is the canonical verification surface — Win11 host API behavior cannot be unit-tested without a real device.

**`asyncio.Lock` re-entrancy:**
- `asyncio.Lock` is **not** re-entrant. Task 2's pre-buffer state machine must not re-acquire `_progressive_playback_lock` recursively from within a held-lock context. The existing Story 17.3 code-path at `app.py:2436-2601` operates entirely under the lock; Task 2's edits stay inside that same `async with` block.

**`np.clip` + cast performance:**
- The PCM16 conversion idiom `(np.clip(chunk.audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()` (Story 17.3 inline at `app.py:2580-2582`) is ~5–10 μs per kilo-frame on the RTX 5090 dev host's CPU. Task 1's instrumentation timestamps must be captured **after** this conversion completes (so `progressive_chunk_playback_arrival_ms` reflects when the data is handed to PyAudio, not when the chunk is received) — otherwise the metric muddles consumer-side prep with PyAudio buffer fill.

### Project Context Reference

- Project context: `docs/` (existing project-context.md not found; CLAUDE.md absent).
- Working directory invariants per `memory/git_repo_state.md`: V2 is canonical git repo since 2026-05-05; remote = github.com/WreckedMech117/MyVoice; `_bmad-output/` is gitignored.
- Production state per `memory/production_release_state.md`: ships publicly via myvoicetts.com as a Windows .exe with bundled portable python310. Installer size is a known pain point; this story does not change installer size.
- Hardware target per `memory/hardware_setup.md`: RTX 5090 Blackwell, Win11, torch 2.10+cu128 dev host; ship-target also covers RTX 30xx/40xx (Ampere+).

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m]

### Debug Log References

- 2026-05-09 — Task 1.0 metrics.record overhead measurement (RTX 5090, Python 3.10.11): N=1000 → 2.40 µs/call; N=5000 → 1.35 µs/call. PASS (≤ 100 µs threshold; ~50× headroom).
- 2026-05-09 — Task 1.1/1.2/1.3 instrumentation tests: `pytest tests/unit/test_app_progressive_playback_instrumentation.py tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py -v` → 7 passed in 47.61 s.
- 2026-05-09 — Story 17.3 progressive-playback regression sweep: `pytest tests/unit/services/test_qwen_tts_service_true_stream_callback.py tests/unit/test_app_progressive_playback.py tests/unit/test_app_progressive_playback_cancel.py tests/integration/test_progressive_playback_dispatch_skip.py -v` → 32 passed in 11.52 s. Zero regressions.
- 2026-05-09 — Metrics observability tests: `pytest tests/unit/observability/ -v` → 45 passed in 5.85 s.
- 2026-05-09 — CSV-capture tests: `pytest tests/unit/observability/test_progressive_playback_csv_capture.py -v` → 13 passed in 12.18 s.
- 2026-05-09 — Combined Story 18.1 + Story 17.3 + observability sweep (97 tests) → all pass in 18.32 s.
- 2026-05-09 — Full app + qwen_tts_service surface (94 tests across `tests/unit/test_app*.py` + `tests/unit/services/test_qwen_tts_service*.py`) → all pass.

### Completion Notes List

- 2026-05-09 — Task 1 (instrumentation) implemented + tested. Three new metrics emit on the TRUE_STREAM data path: `progressive_chunk_emit_ms` (producer / qwen_tts_service.py) and `progressive_chunk_playback_arrival_ms` + `progressive_chunk_audio_duration_ms` (consumer / app.py). Both metrics co-located inside the `audio_data.size > 0` branch so the synthetic terminal chunk produces no metric noise. Wall-clock-ms timebase chosen for both producer and consumer so the Task 1.4 CSV joins by `chunk_index` without clock-base reconciliation.
- 2026-05-09 — Env-var-gated CSV capture infrastructure shipped (`progressive_playback_csv_capture.py` + `01_Run_MyVoice_With_CSV_Capture.bat`). Stays usable for measuring 18.3 + 18.4 throughput uplifts.
- 2026-05-09 — Task 1.4 measurement run executed by Commander on RTX 5090 with the canonical Story 17.3 §4.1 step 3 paragraph (354 chars / ~22 s of speech). 11 chunks captured, 10 with full producer + consumer triplets (chunk 10 is the talker's `streamer.end()` flush, structurally distinct from steady-state cadence and excluded). Steady-state median emit-interval = 6,389 ms vs chunk audio duration 1,981 ms = **3.23× ratio** (p95 = 3.37×). All 8 steady-state inter-chunk silent gaps exceed 250 ms by >10× (median gap = 4,408 ms). Producer at 31% real-time. Evidence file §4.2.
- 2026-05-09 — Task 1.5 verdict: producer-side bottleneck (talker model decode rate, not consumer underrun). **Option 1 (pre-buffer N chunks) ruled out empirically + mathematically** — a 3-chunk pre-buffer's 5.94 s cushion is exhausted in ~1.4 chunks of post-flush playback at the observed 4.4 s/chunk steady-state deficit; subsequent gaps remain at the same 4.4 s class as the unmitigated path. AC #3's "at least one inter-chunk gap > 250 ms" empirical threshold is satisfied with ≥10× margin without ceremonial Option 1 implementation, on the AC-#3 amendment authorized by Commander 2026-05-09 via the dev-story workflow's mitigation-gate question. **Option 2 (frames_per_buffer increase) ruled out structurally** — PyAudio per-stream buffer size doesn't change inter-chunk arrival rate, so it cannot address a producer-throughput defect. **Option 3 (decode-rate investigation) is the correct fix class** and belongs in Stories 18.3 (bf16 precision on talker decoder) + 18.4 (`torch.compile` + persistent compiled-decoder cache) per the Epic 18 plan.
- 2026-05-09 — Story 18.1 closes shipping **instrumentation + CSV-capture infrastructure only** (no consumer-side mitigation source-tree edits beyond Task 1's metric emission sites). Tasks 2/3/4/6 marked NOT APPLICABLE in the story file. Task 5 regression sweep (32 progressive-playback tests + 166 broader streaming/app/audio/observability tests) passes with zero regressions. The CSV-capture infrastructure is the empirical gate Stories 18.3 + 18.4 will use to measure their throughput uplifts (target: ratio 3.23× → < 1.0× in steady state).
- 2026-05-09 — **Code-review pass landed (H1/H2/H3/M1/M2/M3/M4/M5/L2/L5).** H1: AC #3 amendment paragraph added inline at the AC text so a future reader does not hit story-vs-evidence-file inconsistency. H2: deleted duplicate §4.1 (verbatim repeat) + outdated §4.2 ("Tentative reading from prior data") from the evidence file — the file is now linear and free of the captured-data-vs-pre-measurement-caveat contradiction. H3: File List now names every uncommitted file `git status` showed and explicitly segregates the out-of-scope ones (build_tools / Bella-F cache / build_increment_answer.txt). M1: AudioChunk gained an `Optional[str] session_id` field; producer threads `sid` through all 3 emission sites; consumer's two metric calls pass `session_id=chunk.session_id` so the CSV stays joinable when a single run captures multiple generations. M2 + M3: `.gitignore` updated. M4: regression test guards the Story 17.3 commit-`abd842c` "spurious second-session-open" invariant in the metrics dimension. M5: evidence file §2.2 documents the TRUE_STREAM `is_final=False`-on-every-data-chunk semantics + the new `session_id` tag. L2 + L5: comment blocks trimmed (13 + 16 lines → 7 + 5).
- 2026-05-09 — **Code-review pass test sweep**: 54 tests (Story 18.1's 22 + Story 17.3's 32) all pass; the broader 127-test streaming/audio/observability sweep (qwen_tts_service_dispatch + session_integration + observability + audio_coordinator) all pass. Zero regressions from the M1 AudioChunk widening across the orchestrator + producer.

### File List

- `src/myvoice/services/qwen_tts_service.py` — modified: added `progressive_chunk_emit_ms` `metrics.record` call inside `_wrapped_post`'s `append_chunk` branch. **Code-review pass M1**: added `session_id: Optional[str] = None` field to the `AudioChunk` dataclass and threaded `session_id=sid` through all 3 emission sites (SENTENCE_STREAM data chunk, TRUE_STREAM data chunk, TRUE_STREAM synthetic terminal). **Code-review pass L5**: trimmed the producer-side instrumentation comment block from 16 lines to 5.
- `src/myvoice/app.py` — modified: added `import time` and `from typing import Callable, Dict, Optional` (Callable widening); added `from myvoice.observability import metrics`; added `progressive_chunk_playback_arrival_ms` + `progressive_chunk_audio_duration_ms` `metrics.record` calls inside `_handle_progressive_chunk_async`'s post-`play_audio_chunk` block (gated on `audio_data.size > 0`); wired env-var-gated CSV capture (`maybe_enable_from_env`) in `__init__` and stop-callable invocation in `_on_about_to_quit`. **Code-review pass M1**: both consumer-side metrics now pass `session_id=chunk.session_id` so the CSV stays joinable across multiple generations captured in one run. **Code-review pass L2**: trimmed the consumer-side comment block from 13 lines to 7.
- `src/myvoice/observability/progressive_playback_csv_capture.py` — new: env-var-gated (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`) CSV capture for the three Story 18.1 metrics with per-record flush, idempotent stop-callable, thread-safe write lock.
- `tests/unit/test_app_progressive_playback_instrumentation.py` — new (6 tests after code-review pass): orchestrator-side per-chunk metric emission, zero-size terminal chunk skip-discipline, multi-chunk pairing. **Code-review pass M1**: added `session_id` to `_StubChunk` + assertions on consumer-side `record.session_id` in the arrival + duration tests + `test_session_id_separates_chunks_from_two_generations`. **Code-review pass M4**: added `test_fallback_restart_does_not_double_emit_chunk_zero_metrics` (guards the Story 17.3 commit-`abd842c` invariant in the metrics dimension).
- `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py` — new (3 tests): producer-side emit metric per `append_chunk`, session_id + chunk_index tag schema, monotonic non-decreasing wall-clock value.
- `tests/unit/observability/test_progressive_playback_csv_capture.py` — new (13 tests): env-var resolution, CSV header + per-row schema, off-target metric filtering, idempotent stop, file-open failure handling.
- `tests/unit/test_app_progressive_playback.py` — **modified by code-review pass M1**: added `session_id: str | None = None` to the Story 17.3 `_StubChunk` so the existing 23 progressive-playback tests still construct chunks compatible with the consumer-side `chunk.session_id` read.
- `01_Run_MyVoice_With_CSV_Capture.bat` — new: convenience launcher that sets `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` to the spec'd path and runs `main.py` directly (preserving the torch-before-PyQt6 DLL ordering invariant). Prints captured row count on exit.
- `.gitignore` — **modified by code-review pass M2 + M3**: added rules for `voice_files/*.quality.pt` + `voice_files/*.quality.pt.meta.json` (Story 17.2 lazy-precompute runtime caches that were polluting `git status`) and `build_increment_answer.txt` (`build_release.bat` answer-piping leftover).
- `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` — new (Task 1 evidence + Task 1.4 captured measurement). **Code-review pass H2**: deleted duplicate §4.1 (verbatim repeat of the pre-measurement procedure) + outdated §4.2 ("Tentative reading from prior data") that contradicted the captured §4.2; the file is now linear (§1 → §2 → §3 → §4 verdict-and-data → §5/§6/§7 N/A) without back-references to a not-yet-captured measurement. **Code-review pass M5**: added `is_final` semantics-by-streaming-mode note + `session_id` tag documentation in §2.2.
- `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation.md` — **modified by code-review pass H1**: added explicit "AC #3 amendment (2026-05-09 — authorized by Commander)" paragraph after AC #3's strict-text gates, cross-referencing evidence file §4.4. The amendment text closes the "story AC says one thing; evidence file claims another" inconsistency a future reader would otherwise hit.
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — modified: `18-1-underrun-gap-mitigation: ready-for-dev` → `in-progress` → `review` (this status field will flip to `done` when the code-review pass closes).

**Out-of-scope uncommitted files surfaced by `git status` and noted (not Story 18.1 source-tree edits — kept for the user to handle in a separate commit or revert):**

- `build_tools/installer.iss` + `build_tools/version.py` — build counter increments (10 → 12) from prior `build_release.bat` runs during ad-hoc bundle verification. Story 18.1 explicitly disclaims build-pipeline edits ("Not a build-pipeline change" — Dev Notes line 105). Decide whether these belong in this commit or a separate build-state commit.
- `voice_files/Bella-F.quality.pt` + `voice_files/Bella-F.quality.pt.meta.json` — Story 17.2 lazy-precompute runtime artifacts (per-voice cache created on first generation). Now gitignored by the code-review pass M2 fix; the existing files can be deleted from the working tree if not wanted on disk.
- `build_increment_answer.txt` — `build_release.bat` answer-piping leftover (single character "Y"). Now gitignored by the code-review pass M3 fix.

## Change Log

- 2026-05-09 — Story 18.1 closed as **instrumentation-only** per Task 1.4/1.5 producer-bottleneck verdict. AC #3 amendment authorized by Commander to read the captured 3.23× ratio + deterministic math as ruling out Options 1 + 2 without ceremonial Option 1 implementation. Tasks 2/3/4/6 marked NOT APPLICABLE; mitigation deferred to Stories 18.3 (bf16) + 18.4 (`torch.compile`). Evidence file §4 captures the full verdict + raw measurement.
- 2026-05-09 — Code-review pass H1/H2/H3/M1/M2/M3/M4/M5/L2/L5 fixes applied. AudioChunk gained `Optional[str] session_id` (M1) so the consumer-side instrumentation rows in the Task 1.4 CSV stay joinable to producer-side rows when a single run captures multiple generations. `.gitignore` carved out `voice_files/*.quality.pt*` (Story 17.2 runtime caches) and `build_increment_answer.txt` (M2 + M3). Evidence file deduped (H2) + AC #3 amendment paragraph inserted in the story file (H1). New M4 regression test guards the Story 17.3 commit-`abd842c` "spurious second-session-open" invariant in the metrics dimension. 54-test (Story 18.1 + 17.3) + 127-test (broader streaming) sweeps both pass with zero regressions.

## Open Questions for Dev Agent (deferred per workflow guidance)

1. **Default value for `_progressive_pre_buffer_target` (option 1)** — Task 1's instrumentation data should drive this, but if data-collection is inconclusive, the Story 17.3 evidence §4.4 wording ("first 2-3 chunks") suggests 3 as a reasonable default. Confirm with Commander before locking the value.
2. **Default value for `streaming_chunk_size` (option 2)** — same gating from Task 1's data; default 4096 unless data demands 8192 (and AC #2 latency-budget gate is amended in evidence).
3. **Should this story land both options as feature-flagged knobs** rather than committing to one? The Epic 18 stub says "Pick the right fix from the three candidates" — singular — so the default reading is **commit to one**. If Task 1's data is ambiguous, raise this with Commander rather than implementing both speculatively.
