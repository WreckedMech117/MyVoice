# Story 17.2: Lazy + Persistent voice_clone_prompt Precompute for CLONED Voices on TRUE_STREAM

Status: in-progress

> **Phase tag:** Phase ⊥-Ramp completion (closes the gap between Story 17.1's certified TRUE_STREAM path and the user-facing reality that CLONED voices reach it). On closure, Epic 17 transitions back to `done`.
> **Re-opens:** Epic 17 (was `done` per `sprint-status.yaml:103`). Sprint-status edited at workflow step 6 of `/bmad-bmm-create-story`: `epic-17 → in-progress`; `17-2-cloned-voice-truestream-prompt-precompute → ready-for-dev`. Authorized by `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-scope-sketch.md`.
> **Authored:** 2026-05-08 by `/bmad-bmm-create-story` from the scope sketch above; revised same day after fresh-context adversarial review.
> **Why:** Story tooling-2 closure (2026-05-08) discovered the bundled MyVoice.exe probes CUDA → TRUE_STREAM correctly, but every UI-initiated CLONED-voice generation raises `ValueError: TRUE_STREAM voice-clone path requires request.voice_clone_prompt` and falls through to SENTENCE_STREAM via the NFR7 chain. Users get audio, but never the certified-by-Story-17.1 TRUE_STREAM path. This story makes the certified path actually reach users.

## Story

As a **MyVoice end-user with a CLONED voice profile (e.g., `Sarira-F`, `Base (Clone)`) on a CUDA-capable host**,
I want **TRUE_STREAM dispatch to succeed end-to-end on my voice without falling through to SENTENCE_STREAM**,
so that **I receive the lower first-audio latency the streaming-default ramp (Story 17.1) certified for me, instead of silently degrading to SENTENCE_STREAM on every generation**.

## Acceptance Criteria

### AC #1 — Cache wired into `generate_voice_clone`; tight precompute gate; per-voice serialization

**Cache key:** `str(voice_profile.file_path.resolve())` — the absolute, symlink-resolved ref_audio path. **NOT** voice profile name (names collide across folders). **NOT** raw `file_path` string (varies portable-vs-installed and pre/post symlink). The `.resolve()` form is stable across reinstalls when the bundled `voice_files/` directory layout is preserved (it is, per `build_release.bat`).

**If Task 7 confirms tier-locked embeddings (it will — see "Pre-resolved decisions" below), key becomes `(resolved_path, tier)`** where `tier ∈ {"quality", "small"}` per `qwen_tts_service.py:1253`.

**Given** a CLONED-voice generation request enters `QwenTTSService.generate_voice_clone(text, ref_audio, ref_text, ...)` (`src/myvoice/services/qwen_tts_service.py:1082`),
**When** the constructed `QwenTTSRequest` is built and dispatched via `_dispatch_by_streaming_mode`,
**Then** the precompute pipeline runs IF AND ONLY IF ALL of these hold:
1. `request.streaming is True` (NOT False — Voice Design Studio sets `streaming=False` at four sites: `voice_design_studio_dialog.py:555, 827, 1287, 1560`; the BATCH-forcing override at `qwen_tts_service.py:3344-3346` must remain authoritative);
2. The resolved streaming mode (per `_resolve_streaming_mode()`) is `StreamingMode.TRUE_STREAM` (CPU paths resolve to SENTENCE_STREAM/BATCH and DO NOT need precompute);
3. `request.model_type == QwenModelType.BASE` (CLONED voices only; OPTIMIZED/EMBEDDING/CUSTOM_VOICE paths do not reach line 2793-2798);
4. `request.x_vector_only_mode is False` (x-vector mode skips ICL transcription entirely; calling Whisper for an empty ref_text path is meaningless — `app.py:920` and `app.py:1010` both set `x_vector_only_mode=True` for designed-as-clone and no-transcription CLONED voices respectively).

**On all four conditions true:** check `self._voice_clone_prompts[<cache_key>]`; cache miss → acquire per-voice lock → re-check (double-checked locking) → run AC #2 + AC #3 → cache-set → release lock; cache hit → set `request.voice_clone_prompt = self._voice_clone_prompts[<cache_key>]` BEFORE invoking `_dispatch_by_streaming_mode`.
**On any condition false:** skip precompute entirely; the existing dispatch chain handles the request unchanged (BATCH for `streaming=False`; SENTENCE_STREAM for non-GPU; existing fallback for x-vector mode if it later raises in TRUE_STREAM).

**Per-voice `asyncio.Lock` registry** uses `weakref.WeakValueDictionary` so locks for deleted/unloaded voices are garbage-collected (closes the unbounded-growth concern). Concurrent calls for the same voice serialize → precompute fires once. Concurrent calls for different voices proceed in parallel. The lock-registry mutation itself is guarded by a single `asyncio.Lock` (or `setdefault`-style atomic op).

### AC #2 — Lazy transcription via existing `WhisperSubprocessService.transcribe_file`

**Given** AC #1 reached the lazy-precompute branch (cache miss + four-condition gate passed),
**When** the precompute resolves a transcription source,
**Then** priority order is: (1) `voice_profile.transcription` if non-empty → use as-is; (2) `<voice_dir>/<voice_name>.txt` sidecar (per existing auto-detect at `voice_profile.py:348-355`) → read + populate `voice_profile.transcription` + use; (3) only on miss of both, invoke `self._whisper_service.transcribe_file(ref_audio_path)` in async-thread (`whisper_subprocess.py:46`),
**And** when Whisper succeeds, the transcription text is written to `<voice_dir>/<voice_name>.txt` (UTF-8, no BOM) AND `voice_profile.transcription` is updated in-memory AND status transitions through `QUEUED → PROCESSING → COMPLETED` per `TranscriptionStatus` enum (`voice_profile.py:22-29`) using existing methods (`update_transcription_status`/`set_transcription_result` at `voice_profile.py:1101, 1135`),
**And** when Whisper fails, the precompute retries up to 3 times with progressive backoff (1s, 3s; document chosen values in Change Log); on the third failure `mark_transcription_failed(error_str)` (`voice_profile.py:1151`) is called and an exception is raised that the dispatch chain catches → falls through to SENTENCE_STREAM (preserves NFR7),
**And** Whisper is NOT invoked when transcription is already available from sources (1) or (2) — bundled voices that ship with `.txt` sidecars skip Whisper entirely.

**Whisper service is the subprocess variant.** The runtime instance is `WhisperSubprocessService` (`whisper_subprocess.py:29`), NOT `WhisperService` (`whisper_service.py:86`). Both expose `transcribe_file(file_path, language=None, ...)` returning a `.text`-bearing result; only the subprocess form is wired into the orchestrator (`app.py:1883`) due to the PyQt6/Whisper DLL conflict the subprocess isolates.

### AC #3 — Persistent embedding via `create_voice_clone_prompt_for_tier` + `torch.save`

**Pre-resolved decision:** embeddings are **tier-locked**. Justification: `_create_voice_clone_prompt_sync` at `qwen_tts_service.py:1286` is the shared sync helper called by both `create_voice_clone_prompt` (`:1179`) and `create_voice_clone_prompt_for_tier` (`:1230`); the only difference is which model tier `ensure_model_loaded` has loaded. The 1.7B (`quality`) and 0.6B (`small`) Qwen3 models have different hidden dimensions, so `model.create_voice_clone_prompt(...)` produces tensors with different shapes per tier. Persisted file naming: `<voice_dir>/<voice_name>.<tier>.pt` + `<voice_dir>/<voice_name>.<tier>.pt.meta.json` (e.g., `Sarira-F.quality.pt`, `Sarira-F.small.pt`).

**Given** a transcription is available (per AC #2 chain),
**And** AC #1's per-voice lock is held,
**When** the precompute computes the embedding,
**Then** `QwenTTSService.create_voice_clone_prompt_for_tier(ref_audio, ref_text, tier)` (`qwen_tts_service.py:1230-1284`) is awaited with `tier` resolved from the model registry's currently-loaded tier (`self._model_registry.quality_tier` per the precedent at `qwen_tts_service.py:1784`),
**And** the returned `VoiceClonePromptItem`'s tensors are moved to CPU before persistence — mirrors the canonical pattern at `voice_design_studio_dialog.py:1154-1158`,
**And** the prompt is persisted via `torch.save(prompt, str("<voice_dir>/<voice_name>.<tier>.pt"))` — mirrors `voice_design_studio_dialog.py:1162` (the only existing project precedent for this file format),
**And** an adjacent metadata file `<voice_dir>/<voice_name>.<tier>.pt.meta.json` records `{"schema_version": "1.0", "ref_audio_mtime": <float>, "ref_audio_size": <int>, "tier": <str>, "qwen_tts_pin": "1ab0dd75"}` for cache-invalidation detection,
**And** `self._voice_clone_prompts[(<resolved_path>, <tier>)] = prompt` is written in-memory,
**And** verification: reload via `torch.load(persist_path, map_location='cpu', weights_only=False)` (PyTorch 2.6+ requirement for `VoiceClonePromptItem`, per `voice_design_studio_dialog.py:1172` and `scripts/validate_embedding_api.py:219`); pass through `_normalize_voice_clone_prompt` (`qwen_tts_service.py:1343`); on verification failure, delete `.pt` + `.pt.meta.json` and raise.

**Cache invalidation:** when ref_audio's `mtime` or `size` differs from the stored meta, OR when `qwen_tts_pin` differs from the current pin (defensive guard against future pin bumps invalidating the embedding format), delete both files + treat as miss.

**Startup hydration:** at voice library scan time (or first `start()` post-model-registration), iterate CLONED voices in the active library; for each, attempt `<voice>.<tier>.pt` load for the currently-loaded tier per the invalidation rules above; populate `_voice_clone_prompts[(resolved_path, tier)]` on hits; log summary `"Voice clone prompt cache: hydrated N/M CLONED voices for tier <tier> from disk"`. App restarts MUST NOT recompute when meta matches.

### AC #4 — UI feedback during first-run precompute (custom-message variant)

**Given** a TRUE_STREAM dispatch hits a cache-miss precompute that will take ~1–3s (Whisper + embedding compute on cold cache),
**When** the precompute is in flight,
**Then** the user sees visible feedback via `ServiceStatusIndicator.update_status(...)` (`service_status_indicator.py:189`) carrying a custom message reading "Preparing voice for streaming…" — DO NOT reuse `set_loading(True)` (`:215`), which already signals model-loading state and would conflate two distinct transient phases,
**And** the indicator returns to its idle/healthy state once the cache is populated (success path) OR transitions to the existing failure state (precompute exhausted retries) so the user understands why audio will arrive via SENTENCE_STREAM,
**And** the emission is paired (start-on-entry / end-on-exit including failure path) using `try/finally` or async-context-manager — no stuck indicator,
**And** subsequent generations on the same voice (cache hit) MUST NOT emit any indicator change — cache hits are instant and invisible to the user.

If `ServiceStatusInfo` does not currently carry a free-form message field, extend it with `preparing_voice_message: Optional[str] = None`; the indicator's tooltip / label renders the message when set. Document the schema delta in Change Log.

### AC #5 — NFR7 graceful-degradation chain preserved (regression guard)

**Given** a TRUE_STREAM dispatch on a CLONED voice with a populated cache (post-AC #1),
**And** `_generate_true_stream` raises a non-cancel exception for a reason OTHER than missing `voice_clone_prompt` (e.g., synthetic CUDA-OOM, qwen-tts library exception),
**When** `_dispatch_by_streaming_mode` (`qwen_tts_service.py:3320-3399`) catches it,
**Then** the fallback chain progresses to SENTENCE_STREAM (Story 16.6 D-9 / NFR7) exactly as today — no behavior change to the dispatch chain itself,
**And** if AC #2's Whisper retries are exhausted (persistent FAILED), the exception bubbles out of the precompute, `_dispatch_by_streaming_mode` catches it, and the chain falls through to SENTENCE_STREAM (audio still serves; user is not left hanging — preserves NFR7 in the "transcription unavailable" failure mode too).

The voice_clone_prompt-missing case stops triggering fallback only when AC #1's four-condition gate is satisfied AND transcription is computable. **Explicit unit tests required (Task 6) so future refactors don't accidentally short-circuit the dispatch chain.**

### AC #6 — Bundled-environment smoke verifies TRUE_STREAM end-to-end

**Given** AC #1–#5 land on `epic-16` and `build_release.bat` produces a new bundle,
**When** the Story tooling-2 §4 portable smoke flow re-runs on a fresh bundle (clean `voice_files/` — no `.pt` cached), generating utterance `s-014` on `Base (Clone)` voice,
**Then** `myvoice.log` shows: (i) first attempt invokes Whisper IF `.txt` absent OR uses `.txt` short-circuit; (ii) first attempt invokes `create_voice_clone_prompt_for_tier` and writes `<voice>.<tier>.pt` + `.pt.meta.json`; (iii) first attempt completes via TRUE_STREAM end-to-end with NO log line containing "TRUE_STREAM voice-clone path requires" and NO `streaming_mode_fallback` metric for this generation; (iv) **second attempt** (same voice, same or different utterance) is a cache-hit — no Whisper, no `create_voice_clone_prompt_for_tier`, no file writes, TRUE_STREAM completes; second attempt's first-audio latency satisfies NFR1 GPU short-class target (≤5.0s p95 per `architecture-optimization-pass.md:836+`),
**And** the same flow re-runs on installer-mode bundle (Story tooling-2 §6 path) with identical outcomes,
**And** a separate clean-install run with no transcription sidecar exercises the Whisper path end-to-end (UI feedback per AC #4 visible during the ~1–3s window),
**And** evidence is captured in `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` mirroring tooling-2's §4 + §6 + §7 structure (verbatim log excerpts for each scenario).

**Closure note:** Story tooling-2's HIGH §7.2 follow-up resolves on AC #6 evidence pass.

## Tasks / Subtasks

- [x] **Task 1 — Wire `_voice_clone_prompts` cache + four-condition gate + per-voice serialization (AC: #1)**
  - [x] 1.1 Added `_voice_clone_prompt_locks` (WeakValueDictionary) + `_voice_clone_prompt_locks_guard` (lazy-init `asyncio.Lock`) next to the cache. Tuple-keyed `Dict[(resolved_path, tier), Any]`.
  - [x] 1.2 `_get_voice_clone_prompt_lock` returns the same Lock for repeated keys (DCL inside the guard); covered by `test_concurrent_same_voice_serializes`.
  - [x] 1.3 Cache key = `(str(ref_audio.resolve()), self._model_registry.quality_tier.value)` (verified `quality_tier.value` returns `"quality"` / `"small"` per service_enums.py:42-43). Four-condition gate evaluated inline.
  - [x] 1.4 Gate-pass branch: hit → set `request.voice_clone_prompt`; miss → DCL → Tasks 2+3 helpers → cache-set; gate-fail → straight dispatch.
  - [x] 1.5 No new parameter on `generate_voice_clone` — the cache key is derivable from `ref_audio`. Three call sites verified per scope sketch (designed-as-clone, CLONED, internal embedding-fallback).
  - [x] 1.6 Six unit-test scenarios in `tests/unit/services/test_voice_clone_prompt_cache.py::TestFourConditionGate` + `TestCacheKeyAndSerialization`.

- [x] **Task 2 — Lazy transcription helper invoking `WhisperSubprocessService` (AC: #2)**
  - [x] 2.1 `_ensure_transcription_for_clone_voice(voice_profile, ref_audio) -> str`.
  - [x] 2.2 Resolution priority chain implemented (in-memory → .txt sidecar → Whisper).
  - [x] 2.3 `set_whisper_service` setter + orchestrator wiring in `app.py:_initialize_whisper_service_on_demand` (after `_main_window.set_whisper_service(...)`). Lazy-fail-safe raises when service is None. Second on-demand-init trigger wired via `set_whisper_init_callback(self._on_whisper_init_requested)` so cache-miss with no Whisper requests init in the background.
  - [x] 2.4 Whisper success writes UTF-8 sidecar then updates profile via `set_transcription_result(text, confidence, "whisper-base")`.
  - [x] 2.5 Retry policy: backoffs `(1.0, 3.0)` seconds (constant `_WHISPER_RETRY_BACKOFFS_SECONDS`); three attempts total (initial + 2 retries); `mark_transcription_failed` on exhaustion; raises `RuntimeError`.
  - [x] 2.6 Status transitions QUEUED → PROCESSING → COMPLETED/FAILED via existing `update_transcription_status` / `set_transcription_result` / `mark_transcription_failed` helpers.
  - [x] 2.7 Six unit-test scenarios in `TestEnsureTranscriptionForCloneVoice`.

- [x] **Task 3 — Persistent embedding helper + cache invalidation + startup hydration (AC: #3)**
  - [x] 3.1 `_ensure_voice_clone_prompt_for_voice(voice_profile, ref_audio, transcription, tier)`.
  - [x] 3.2 Tier-locked file naming via `_voice_clone_prompt_persist_paths`: `<stem>.<tier>.pt` + `<stem>.<tier>.pt.meta.json`.
  - [x] 3.3 Calls existing `create_voice_clone_prompt_for_tier` (which calls the inner sync helper at line 1286 — the test-mock target). Returns the prompt unchanged on success.
  - [x] 3.4 CPU move on `ref_code` + `ref_spk_embedding` (defensive try/except for non-tensor mocks); `torch.save(prompt, str(pt_path))`; meta JSON; verification reload with `weights_only=False`; on verification fail delete both files + raise.
  - [x] 3.5 `_voice_clone_prompt_meta_is_valid` checks `tier`, `qwen_tts_pin`, `ref_audio_mtime` (1ms tolerance), `ref_audio_size`. Stale files purged. Pin held in `_QWEN_TTS_PIN_HASH = "1ab0dd75"` constant — bump alongside `requirements.txt:23` if pin changes.
  - [x] 3.6 Hydration moved from `start()` to a separate `hydrate_voice_clone_prompt_cache()` method (the orchestrator constructs the VoiceProfileManager **after** `tts.start()` returns; in-place start hook would always see no manager). Orchestrator runs hydration as a fire-and-forget `_run_async_task` after `await self._voice_manager.start()` (app.py).
  - [x] 3.7 Eight unit-test scenarios in `TestEnsureVoiceClonePromptForVoice` + `TestHydrateVoiceClonePromptCache`.

- [x] **Task 4 — Integration: precompute pipeline calls Tasks 2 + 3 from Task 1's miss path (AC: #1, #2, #3)**
  - [x] 4.1 Cache-miss branch under per-voice lock calls Task 2 (transcription) → Task 3 (prompt) → cache-set; structured log lines fire at each step.
  - [x] 4.2 Exception propagation goes through `try/finally` so the lock is always released and the indicator is always cleared (Task 5). The exception is logged + swallowed at the precompute layer; the request reaches dispatch with `voice_clone_prompt is None` so the existing TRUE_STREAM-contract check at qwen_tts_service.py:2793-2798 raises and the dispatch chain handles fallback (NFR7).
  - [x] 4.3 INFO log lines at miss-on-disk, miss-on-memory, Whisper start, Whisper completion, persisted-to-disk.
  - [x] 4.4 Integration-style test `test_persisted_pt_loads_on_second_call` — first call computes + persists, second call hits the on-disk fast path.

- [x] **Task 5 — UI indicator state for first-run precompute via custom message (AC: #4)**
  - [x] 5.1 `ServiceStatusInfo.preparing_voice_message: Optional[str] = None` added to `models/ui_state.py`.
  - [x] 5.2 `service_status_indicator.py::_update_tooltip` surfaces the message as an italic line in the indicator's HTML tooltip. Visual style: tooltip-only (no separate label widget) — keeps the indicator footprint identical to its 16x16 emoji form, avoids layout reflow, and the tooltip is the existing accessible surface for service status.
  - [x] 5.3 Emission wired via `_emit_preparing_voice` in the cache-miss branch — entry inside `try`, clear in `finally`. Cache-hit path never invokes the helper.
  - [x] 5.4 Cache-hit path explicitly does NOT emit (verified by `test_callback_silent_on_cache_hit`).
  - [x] 5.5 Three indicator unit tests in `TestPreparingVoiceIndicator` — emission on miss, silent on hit, clears on precompute failure.

- [x] **Task 6 — NFR7 preservation regression tests (AC: #5)**
  - [x] 6.1 `test_cache_hit_then_oom_falls_back_to_sentence_stream` — cache HIT + RuntimeError("CUDA OOM") → SENTENCE_STREAM response.
  - [x] 6.2 `test_persistent_whisper_failure_falls_back` — cache MISS + persistent Whisper failure → SENTENCE_STREAM.
  - [x] 6.3 `test_embedding_compute_failure_falls_back` — cache MISS + embedding compute raise → SENTENCE_STREAM.
  - [x] 6.4 `test_precompute_succeeds_request_carries_voice_clone_prompt` — captures the request reaching `_generate_true_stream` and asserts `voice_clone_prompt is not None`.
  - [x] 6.5 `test_streaming_false_skips_precompute` — `streaming=False` skip + dispatch unchanged.
  - [x] 6.6 `test_x_vector_only_mode_skips_precompute` — gate condition #4 covered.

- [ ] **Task 7 — Bundled-environment smoke verification (AC: #6)**
  - [ ] 7.1 After source-tree edits land + tests pass + commit lands, run `build_release.bat` (per Story tooling-2 §3 build pipeline).
  - [ ] 7.2 Execute Story tooling-2 §4 portable-smoke flow on fresh bundle: launch `dist/MyVoice.exe`; navigate to Voice Library; select `Base (Clone)`; generate `s-014`; observe UI indicator (AC #4); verify audio plays end-to-end.
  - [ ] 7.3 Inspect `myvoice.log` for: (i) first-attempt log markers per AC #6 (Whisper invocation OR `.txt` short-circuit; `torch.save` line; TRUE_STREAM completion with no fallback metric); (ii) second-attempt cache-hit markers (no Whisper, no `torch.save`, TRUE_STREAM completion).
  - [ ] 7.4 Repeat 7.2–7.3 on installer-mode bundle (Story tooling-2 §6 flow): run installer; launch installed `MyVoice.exe`; same generation flow; verify identical log markers.
  - [ ] 7.5 Capture both runs' log excerpts + observations in `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` (mirror tooling-2's evidence file structure: §1 summary, §2 source-tree changes, §3 build pipeline, §4 portable smoke, §5 lazy-precompute timing measurements, §6 installer smoke, §7 follow-ups). Include first-audio latency on second-attempt cache hit (target: NFR1 GPU short-class ≤5.0s p95).

- [ ] **Task 8 — Sprint-status finalization + epic-17 closure evaluation**
  - [ ] 8.1 Sprint-status edit (`_bmad-output/implementation-artifacts/sprint-status.yaml`): `epic-17 → in-progress` and `17-2-cloned-voice-truestream-prompt-precompute → ready-for-dev` are written by `/bmad-bmm-create-story` workflow step 6 (already applied 2026-05-08).
  - [ ] 8.2 On story closure: code-review pass via `/bmad-bmm-code-review` (per Epic 16 / Story 17.1 closure pattern); on review pass, `17-2 → done`.
  - [ ] 8.3 Evaluate `epic-17` transition: when `17-2 → done`, set `epic-17 → done`; consider whether to flip `epic-17-retrospective: optional → done` (Epic 17 grew from 1 to 2 stories post-retrospective-marking — Commander's call at story close).
  - [ ] 8.4 Memory update: `memory/build_tools_phase_perp_state.md`'s "HIGH follow-up = TRUE_STREAM voice_clone_prompt regression in bundled UI flow" line is the canonical pointer to this story; on closure, update or remove the line per the file's existing format.

## Dev Notes

### Architecture compliance (the developer MUST follow)

- **D-9 hardware-aware streaming default** (`architecture-optimization-pass.md:257`): preserved unchanged. TRUE_STREAM path only when `torch.cuda.is_available()` is True. CPU users continue on SENTENCE_STREAM and DO NOT trigger precompute (AC #1 condition #2 gate).
- **NFR7 graceful degradation** (`architecture-optimization-pass.md:73, 806`): the three-mode fallback chain (TRUE_STREAM → SENTENCE_STREAM → BATCH) at `qwen_tts_service.py:3320-3399` is the safety net. AC #1–#3 must NOT modify the dispatch chain itself; they only populate the request the chain consumes. AC #5 is the explicit regression guard.
- **NFR3 audition** (`architecture-optimization-pass.md:803, 863`): cleared by Story 17.1 outcome (a). The embedding tensor produced by precompute is the same one the audition's fixture would have used — perceptual equivalence preserved. **Story 17.2 does NOT re-litigate Story 17.1's certification.**
- **NFR1 first-audio latency** (`architecture-optimization-pass.md:802, 825-848`): per-class targets (short ≤5.0s p95, medium ≤10.0s p95, long informational) per Story 16.9. AC #6's second-attempt cache-hit measurement should land within short-class target for `s-014`. **First-attempt cold-cache latency includes Whisper + embedding compute and is exempt from NFR1** (one-time cost per voice).
- **Phase ⊥-Ramp**: Story 17.1 closed Phase ⊥-Ramp's certification dimension; Story 17.2 closes the user-reach dimension.

### Library / framework requirements (DO NOT change without explicit approval)

- **qwen-tts pin: `1ab0dd75` (qwen-tts 0.0.4)** per Story 16.1; verified at `requirements.txt:23` and `build_tools/requirements-production.txt`. **NO pin bump in this story.** The `qwen_tts_pin` field in `.pt.meta.json` (AC #3) is a defensive guard against future pin bumps invalidating cached embeddings.
- **PyTorch**: 2.10+cu128 per `memory/hardware_setup.md` and Story tooling-2 closure. The `weights_only=False` argument is required on `torch.load` for `VoiceClonePromptItem` deserialization in PyTorch 2.6+ (per `voice_design_studio_dialog.py:1172` and `scripts/validate_embedding_api.py:219` precedents). DO NOT use `weights_only=True`.
- **Whisper integration**: use the existing `WhisperSubprocessService.transcribe_file(file_path: Path, language=None, word_timestamps=False, temperature=0.0)` (`whisper_subprocess.py:46`) — async; runs the bundled subprocess; returns `TranscriptionResult` with `.text` field. **DO NOT introduce a new transcription dependency.** **DO NOT use the `WhisperService` class** (`whisper_service.py:86`) directly — the runtime instance is the subprocess variant due to PyQt6/Whisper DLL conflict isolation.
- **Persistence format**: `torch.save(VoiceClonePromptItem, str(path))` — exactly the pattern at `voice_design_studio_dialog.py:1162` and `:1542`. **DO NOT use a different serialization format.**
- **`asyncio.Lock` + `weakref.WeakValueDictionary`**: standard library; no new dependency. The dict-mutation race when lazy-creating per-voice locks is guarded by a single `asyncio.Lock` (or `setdefault` semantics).

### File structure requirements

- **Cache files location: next to the voice's `.wav`**. For voice at `<voice_dir>/<voice_name>.wav`:
  - Transcription sidecar: `<voice_dir>/<voice_name>.txt` (existing convention per `voice_profile.py:348-355`).
  - Embedding (tier-locked): `<voice_dir>/<voice_name>.<tier>.pt` and `<voice_dir>/<voice_name>.<tier>.pt.meta.json` where `<tier> ∈ {quality, small}`.
- **DO NOT relocate cache files** to a separate cache directory. Sidecar placement keeps voice + transcription + embedding together.
- **Bundled `voice_files/` directory**: bundled installer ships voices under `dist/MyVoice/_internal/voice_files/<voice_name>/<voice_name>.wav`. Some bundled voices may already have `<voice_name>.txt` — AC #2 priority-2 handles this. **DO NOT pre-bundle `.pt` files** — the embedding is hardware/PyTorch-version-specific; first launch on user's hardware computes locally.
- **No new module / no new directory.** Source-tree changes localized to existing files.

### Testing requirements

- **Unit-test framework**: pytest + pytest-asyncio. Existing `tests/conftest.py` torch-before-PyQt6 preamble per `memory/torch_pyqt6_dll_ordering.md`.
- **Coverage runs**: `tests/conftest.py` preamble does NOT fire under coverage — use the inline torch-first preamble per `memory/torch_before_coverage_dll_ordering.md`. Add inline preamble to any new test module that imports torch directly.
- **Mocking strategy**:
  - Mock `WhisperSubprocessService.transcribe_file` to control transcription outcomes.
  - Mock `_create_voice_clone_prompt_sync` (`qwen_tts_service.py:1286` — the inner sync helper) to fast-return a synthetic `VoiceClonePromptItem`. NOT `create_voice_clone_prompt` (the outer async wrapper that calls `ensure_model_loaded`).
  - Mock `_generate_true_stream` to verify `request.voice_clone_prompt is not None` on entry (AC #5 / Task 6.4).
  - Use `tmp_path` pytest fixture for `.pt` / `.txt` persistence tests.
  - Qt indicator tests (Task 5.5): consult existing `tests/test_main_window*.py` for `_force_quit=True` pattern (`memory/main_window_close_confirm_dialog_in_tests.md`).
- **Integration test (Task 4.4)** mocks `_create_voice_clone_prompt_sync` and exercises the full Task 1+2+3+4 path; assert two-call sequence (miss → hit).
- **Smoke verification (Task 7) is NOT a unit test** — manual / scripted run on the bundled .exe per Story tooling-2's evidence-file pattern.

### Pre-existing infrastructure (REUSE — do not reinvent)

| Component | Path | Use |
|---|---|---|
| Dead cache dict | `qwen_tts_service.py:631` | **Wire it up.** Primary integration point. |
| `generate_voice_clone` entry | `qwen_tts_service.py:1082-1121` | Modify to gate precompute on four-condition AC #1. |
| `create_voice_clone_prompt_for_tier` | `qwen_tts_service.py:1230-1284` | Call from precompute (tier-locked persistence). |
| `_create_voice_clone_prompt_sync` (mock target) | `qwen_tts_service.py:1286` | Inner sync helper; mock here for tests. |
| `_normalize_voice_clone_prompt` | `qwen_tts_service.py:1343-1471` | Pass loaded `.pt` through this for library-form normalization. |
| Embedding-fallback caller | `qwen_tts_service.py:1797` | `streaming=streaming` passes through; AC #1 gate respects it. |
| `_dispatch_by_streaming_mode` (NFR7 chain) | `qwen_tts_service.py:3320-3399` | **DO NOT MODIFY.** `streaming=False` BATCH-force at `:3344-3346` remains authoritative. |
| TRUE_STREAM contract check (the "bug") | `qwen_tts_service.py:2793-2798` | **DO NOT MODIFY.** AC #1 ensures `request.voice_clone_prompt` is set before reaching this. |
| `VoiceProfile.transcription` + `.txt` auto-detect | `voice_profile.py:219, 348-355` | AC #2 priorities 1 + 2. |
| `TranscriptionStatus` enum | `voice_profile.py:22-29` | AC #2 status transitions. |
| `update_transcription_status` / `set_transcription_result` / `mark_transcription_failed` | `voice_profile.py:1101, 1135, 1151` | AC #2 status updates. |
| `WhisperSubprocessService.transcribe_file` | `whisper_subprocess.py:46` | AC #2 priority 3. |
| Whisper on-demand init flow | `app.py:1828-1899` | Mirror pattern; add parallel trigger for cache-miss precompute. |
| Whisper service propagation | `app.py:1897` | Add parallel `_tts_service.set_whisper_service(...)` call. |
| `torch.save(prompt, str(path))` precedent | `voice_design_studio_dialog.py:1141-1162, 1526-1542` | AC #3 persistence pattern. |
| `torch.load(... weights_only=False)` precedent | `voice_design_studio_dialog.py:1172` | AC #3 verification reload pattern. |
| `VoiceClonePromptItem` import for deserialization | `voice_design_studio_dialog.py:64-65` | Required for `torch.load` to work. |
| `ServiceStatusIndicator.update_status` (custom message) | `service_status_indicator.py:189` | AC #4 wiring (NOT `set_loading` at `:215`). |

### Project Structure Notes

- **All source-tree changes localized to existing files.** No new modules. Files likely to edit (in priority order):
  1. `src/myvoice/services/qwen_tts_service.py` — primary; cache wiring + precompute helpers + integration in `generate_voice_clone`. Bulk of LOC delta.
  2. `src/myvoice/app.py` — orchestrator: add `_tts_service.set_whisper_service(...)` call at `_on_whisper_init_completed` (`:1848`); optionally add a second on-demand-init trigger from the cache-miss path.
  3. `src/myvoice/models/voice_profile.py` — minor; possibly extend `TranscriptionStatus` usage paths.
  4. `src/myvoice/ui/components/service_status_indicator.py` — minor; extend `ServiceStatusInfo` with `preparing_voice_message` field if absent.
- **No `requirements.txt` edits.** No `build_tools/requirements-production.txt` edits. No `build_release.bat` edits. No installer-spec edits.
- **Test files**: new test cases added to `tests/test_qwen_tts_service*.py` (or new `tests/test_voice_clone_prompt_cache.py` if cleaner). UI tests in `tests/test_service_status_indicator*.py` if extant.
- **Cache key safety**: `voice_profile.file_path.resolve()` produces an absolute `Path`; `str(...)` of it is the cache key. No filesystem-unsafe-character risk on the key (it's a string identifier, not a filename). The persisted-file name uses `voice_profile.file_path.stem` + `.<tier>.pt` — `.stem` strips the `.wav` extension only and preserves spaces/parens (e.g., `Base (Clone).quality.pt`); Windows + Linux both accept these characters in filenames.

### Previous-story intelligence (Story 17.1 carryover)

- **Code-review pattern: H/M/L severity discipline.** Story 17.1's review produced H1 (methodology disclosure escalation) + M1/M2/M3. Story 17.2's review will likely surface: cache-key edge cases (very long paths, network paths), per-voice lock cleanup edge cases, `.pt.meta.json` schema versioning corner cases. Anticipate; tag in Change Log.
- **Force-add discipline for non-test artifacts** (per Story 17.1 M2): `_bmad-output/` is gitignored; if any tooling script becomes the only-mechanism-that-preserves-something, force-add on closure. Most likely Task 7's evidence file IS committed.
- **Reproducibility framing**: `.pt.meta.json` includes `schema_version` for forward compatibility — Story 17.1's M2 surfaced reproducibility gaps that schema-versioning helps avoid.
- **Code-review regression-test discipline** (per `memory/code_review_regression_test_exact_class.md`): if review surfaces a HIGH/MEDIUM bug class, the regression test must mirror the EXACT bug class (not the nearest adjacent case); re-run review after non-trivial auto-fixes.

### Git intelligence

```
e83bc6d Story 17.2 scope sketch: lazy + persistent voice_clone_prompt precompute for CLONED voices on TRUE_STREAM
6da003b Story tooling-2: Phase ⊥-Build closed — outcome (b) CUDA-enabled bundle verified
19f1d54 Story 17.1: code-review pass — H1/M1/M2/M3 fixes
d13b78f Story 17.1: streaming default ramp — audition outcome (a) — Phase ⊥-Ramp closed
dc9e9bd Epic 16 retrospective (compressed pass) — Phase ⊥ closed
f4fc31a Story 16.9: code-review pass — M1/M2/M3/L1/L2 fixes
0953875 Story 16.9: NFR1 reconciliation (outcome (c) / generate=99% phase verdict / Phase ⊥)
fca0157 Story 16.8: code-review pass — H1/H2/M1/M2/M3/M4 fixes
5a56549 Story 16.8: TRUE_STREAM real wire-up (Path A forward-hook / FR2 / NFR7 / Phase ⊥)
```

**Patterns to mirror:** each Story 16.x landed a single feature commit + a code-review pass commit. Story 17.2 should follow: single feature commit on closure, then code-review pass commit.

**Pin context:** last `requirements.txt` / `build_tools/requirements-production.txt` edit was Story 16.1's qwen-tts pin (`1ab0dd75` = 0.0.4). Story tooling-2 verified the pin survives the production build. Story 17.2 inherits unchanged.

### What this story is NOT

See scope sketch §"What this story is NOT" for the full list (verbatim). Key cuts: NOT a Story 17.1 audition re-litigation; NOT a Voice Design Studio change; NOT a transcription quality story; NOT a build-pipeline change; NOT a re-run of the production release; NOT a Whisper integration overhaul.

### References

**Source tree (touched files — all citations verified 2026-05-08):**
- `src/myvoice/services/qwen_tts_service.py:631` — dead `_voice_clone_prompts` cache (Task 1)
- `src/myvoice/services/qwen_tts_service.py:1082-1121` — `generate_voice_clone` entry (Task 1)
- `src/myvoice/services/qwen_tts_service.py:1230-1284` — `create_voice_clone_prompt_for_tier` (Task 3)
- `src/myvoice/services/qwen_tts_service.py:1286` — `_create_voice_clone_prompt_sync` (mock target, Task 4.4)
- `src/myvoice/services/qwen_tts_service.py:1343-1471` — `_normalize_voice_clone_prompt` (Task 3 verification)
- `src/myvoice/services/qwen_tts_service.py:1797` — embedding-fallback caller (Task 1.5 third call site)
- `src/myvoice/services/qwen_tts_service.py:2793-2798` — TRUE_STREAM contract (read-only; AC #1 must populate before)
- `src/myvoice/services/qwen_tts_service.py:3320-3399` — `_dispatch_by_streaming_mode` (read-only; NFR7 chain)
- `src/myvoice/services/qwen_tts_service.py:3344-3346` — `streaming=False` BATCH-force (AC #1 condition #1)
- `src/myvoice/models/voice_profile.py:22-29, 219, 348-355` — `TranscriptionStatus` + `transcription` + `.txt` auto-detect
- `src/myvoice/models/voice_profile.py:1101, 1135, 1151` — transcription status update methods
- `src/myvoice/services/whisper_subprocess.py:29, 46` — `WhisperSubprocessService` + `transcribe_file`
- `src/myvoice/app.py:916, 1006` — UI call sites of `generate_voice_clone`
- `src/myvoice/app.py:1828-1899` — Whisper on-demand init flow
- `src/myvoice/app.py:1897` — Whisper service propagation pattern
- `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:64-65` — `VoiceClonePromptItem` import for `torch.load`
- `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:555, 827, 1287, 1560` — `streaming=False` sites (AC #1 condition #1 motivation)
- `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:1141-1162, 1170-1178` — `torch.save` + verification reload pattern
- `src/myvoice/ui/components/service_status_indicator.py:189` — `update_status` (AC #4)
- `scripts/validate_embedding_api.py:219` — `weights_only=False` precedent
- `requirements.txt:23` — qwen-tts pin `1ab0dd75`
- `build_tools/requirements-production.txt` — production pin set (read-only; no edits)

**Architecture references:**
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:73, 806` — NFR7 graceful degradation
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:257` — D-9 hardware-aware streaming default
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:802, 825-848` — NFR1 per-class targets
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:803, 863+` — NFR3 + Story 17.1 H4 follow-up note

**Memory references:**
- `memory/build_tools_phase_perp_state.md` — names this story as the HIGH follow-up gating Phase ⊥-Ramp's user-facing deliverable
- `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical pointer)
- `memory/hardware_setup.md` — RTX 5090 CUDA dev host
- `memory/torch_pyqt6_dll_ordering.md` + `memory/torch_before_coverage_dll_ordering.md` — testing preamble requirements
- `memory/main_window_close_confirm_dialog_in_tests.md` — UI test pattern
- `memory/code_review_regression_test_exact_class.md` — code-review fix-test discipline
- `memory/production_release_state.md` — installer-size pain point (informs "no new dependency" rule)

**Precedent stories:**
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-scope-sketch.md` — **THIS STORY'S CANONICAL SCOPE** (authored 2026-05-08)
- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` §4.3.2, §6.2, §7.2 — verbatim regression evidence; §7.2's HIGH follow-up resolves on closure
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` — Story 17.1 closure
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` — routing artifact
- `_bmad-output/implementation-artifacts/16-7-empirical-validation-gates-for-streaming-default.md` — empirical-validation harness (informs Task 7 smoke design)
- `_bmad-output/implementation-artifacts/16-3-codectokenstreamer-with-bounded-queue.md` — TRUE_STREAM bounded-queue lock-step constraint (why on-the-fly compute is infeasible)
- `_bmad-output/implementation-artifacts/16-6-true-stream-dispatch-and-three-mode-fallback-chain.md` — `_dispatch_by_streaming_mode` design rationale (NFR7 chain preserved by AC #5)

## Open questions for the dev-story to resolve (saved per workflow's "save questions for the end" rule)

1. **Trigger for second on-demand Whisper init.** `app.py:1828` triggers Whisper init when Voice Design Studio opens. Should the cache-miss precompute also trigger init? Probably yes (otherwise CLONED-voice users who never open Voice Design Studio never get TRUE_STREAM). Confirm during Task 2.3.
2. **Whisper retry backoff durations.** Story currently picks 1s/3s. Confirm reasonable for the bundled environment; re-tune if AC #6 evidence shows pathologically slow retries.
3. **Bundled-voice `.pt` exclusion in build pipeline.** Confirm `build_release.bat` does NOT pre-bundle `.pt` files (per scope sketch). If a dev computes `.pt` on the build host and it leaks into the bundle, the user's hardware may invalidate it via meta-mismatch (correct behavior) — but this generates user-visible Whisper-and-recompute on first launch that could be misinterpreted as a bug. Add a build-pipeline check OR a `.gitignore` entry under `voice_files/` if needed. Surface during Task 7 evidence run.

## Dev Agent Record

### Agent Model Used

claude-opus-4-7 (1M context) via `/bmad-bmm-dev-story` workflow on 2026-05-08.

### Debug Log References

- `tests/unit/services/test_voice_clone_prompt_cache.py` — 26 tests, all passing.
- `tests/unit/services/test_qwen_tts_service_dispatch.py` — 39 tests passing (no regressions).
- `tests/unit/services/test_qwen_tts_service_session_integration.py` — 19 tests passing (no regressions).
- `tests/unit/services/test_qwen_tts_metric_migration_static.py` — 7 tests passing (no regressions).
- Targeted regression on `tests/unit/services/`, `tests/unit/models/`, `tests/unit/observability/`: 687 tests passing.
- Full suite shows 45 pre-existing failures (verified via `git stash` — paths in error messages reference `G:\MyVoicePublicInst`, an unrelated checkout layout) plus 2295 passes; no failures originate from Story 17.2 changes.

### Completion Notes List

Source-tree implementation (Tasks 1–6) complete and unit-tested. Key decisions and deviations from the as-written story spec:

1. **Cache key shape — `(resolved_path, tier)`** as the story specified. The cache typing was changed from `Dict[str, Any]` to `Dict[Tuple[str, str], Any]`. Tier resolved via `self._model_registry.quality_tier.value` (returns `"quality"` / `"small"` per service_enums.py:42-43).

2. **Hydration trigger — moved from `start()` to a dedicated `hydrate_voice_clone_prompt_cache()` method.** Story Task 3.6 named `start()` post-model-registration, but the orchestrator constructs `VoiceProfileManager` AFTER `await self._tts_service.start()` returns (app.py:406 vs :429). A start()-internal hydration would always see `_voice_profile_manager is None`. The orchestrator now invokes hydration as a fire-and-forget `_run_async_task` after `_voice_manager.start()` lands. Behavior identical to spec; trigger relocated.

3. **Whisper retry backoffs — `(1.0, 3.0)` seconds** as proposed in Open Question 2. Three attempts total (initial + two backed-off retries) before declaring FAILED.

4. **Whisper-init second trigger — `set_whisper_init_callback(...)` setter** with the orchestrator wiring `_on_whisper_init_requested` so a cache-miss with no Whisper service in flight requests on-demand init in the background. The precompute itself raises so the dispatch chain falls through to SENTENCE_STREAM for the in-flight request (NFR7); the next generation on the same voice (post-init) hits the cache. Closes Open Question 1.

5. **Precompute exception handling at the gate — caught + logged, NOT re-raised.** Story Task 4.2 says "any exception bubbles up; the dispatch chain catches and falls through". In practice, the dispatch chain catches non-cancel exceptions only when raised from `_generate_true_stream` itself — not from preceding code. The cleaner equivalent (and what we ship): the precompute logs + swallows; the request reaches dispatch with `voice_clone_prompt is None`; the existing TRUE_STREAM contract check at qwen_tts_service.py:2793-2798 raises ValueError; the dispatch chain catches that and falls through to SENTENCE_STREAM. Same end result, no asymmetric exception flow. Verified by `test_persistent_whisper_failure_falls_back` and `test_embedding_compute_failure_falls_back`.

6. **UI indicator visual style — tooltip-only.** Task 5.2 left the visual style to dev-story discretion. Tooltip-only avoids any layout reflow on the 16x16 emoji indicator footprint; the existing tooltip is the accessible canonical surface for service status messaging.

7. **Test mock target — outer `create_voice_clone_prompt_for_tier` (not the inner sync helper).** Story Dev Notes named `_create_voice_clone_prompt_sync` (line 1286) as the canonical mock target. In practice, the outer async wrapper itself calls `ensure_model_loaded` first; mocking the outer wrapper short-circuits the model load entirely with no behavioral difference. Both surfaces are validated by Story 16.1's test_qwen_tts_internals.py pin.

8. **Build-pipeline `.pt` exclusion (Open Question 3) — punt to Task 7 evidence run.** No `voice_files/` `.gitignore` change applied in this commit; if Task 7's smoke run reveals a pre-bundled `.pt` leaking into the installer, address with a follow-up.

**Task 7 (bundled-environment smoke verification) and Task 8 (sprint-status finalization) are deferred** — Task 7 requires a Commander-led `build_release.bat` run + interactive UI smoke on `dist/MyVoice.exe` (cannot be driven from a coding session). Task 8 lands at code-review pass / closure.

### File List

**Modified:**
- `src/myvoice/services/qwen_tts_service.py` — bulk of LOC delta. Added imports (`json`, `weakref`); changed `_voice_clone_prompts` cache typing to tuple-keyed; added `_voice_clone_prompt_locks` registry + guard, `_whisper_service` / `_whisper_init_callback` / `_voice_profile_manager` / `_preparing_voice_callback` slots; constants `_QWEN_TTS_PIN_HASH` + `_WHISPER_RETRY_BACKOFFS_SECONDS`; setters `set_whisper_service` / `set_whisper_init_callback` / `set_voice_profile_manager` / `set_preparing_voice_callback`; helpers `_get_voice_clone_prompt_lock` / `_ensure_transcription_for_clone_voice` / `_voice_clone_prompt_persist_paths` / `_voice_clone_prompt_meta_is_valid` / `_delete_voice_clone_prompt_files` / `_ensure_voice_clone_prompt_for_voice` / `_emit_preparing_voice` / `_lookup_voice_profile`; public `hydrate_voice_clone_prompt_cache`; modified `generate_voice_clone` body to gate-and-precompute before dispatch.
- `src/myvoice/app.py` — added `set_preparing_voice_callback` / `set_whisper_init_callback` wiring after TTS service construction; added `set_voice_profile_manager` + fire-and-forget hydration trigger after `_voice_manager.start()`; added `set_whisper_service` propagation in `_initialize_whisper_service_on_demand` after MainWindow propagation; added `_on_tts_preparing_voice_message` handler that re-emits `ServiceStatusInfo` with the message field set.
- `src/myvoice/models/ui_state.py` — added `preparing_voice_message: Optional[str] = None` field to `ServiceStatusInfo`.
- `src/myvoice/ui/components/service_status_indicator.py` — `_update_tooltip` renders `preparing_voice_message` as an italic line above the existing tooltip fields.
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — `17-2-cloned-voice-truestream-prompt-precompute: ready-for-dev → in-progress`.
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md` — story file (this document) updated with Tasks 1-6 marked complete + Dev Agent Record populated.

**Added:**
- `tests/unit/services/test_voice_clone_prompt_cache.py` — 26 unit tests covering Tasks 1, 2, 3, 5, 6 (Task 4 integration coverage delivered by the second-call-cache-hit test in `TestEnsureVoiceClonePromptForVoice::test_persisted_pt_loads_on_second_call`).

### Change Log

- 2026-05-08 — Source-tree implementation complete. Cache wiring, lazy transcription, persistent embedding, NFR7 regression tests all landed. 26/26 new tests pass; 65/65 existing dispatch + session-integration tests pass with no regressions.
- Whisper retry backoffs chosen as `(1.0, 3.0)` seconds (Task 2.5 / Open Question 2).
- UI indicator visual style: tooltip-only italic line (Task 5.2 dev-story decision).
- Orchestrator wiring delta (Task 2.3): `set_whisper_service` propagation lives in `app.py:_initialize_whisper_service_on_demand` after the existing `MainWindow.set_whisper_service(...)` call (line 1897-region) — equivalent to the story-named line 1848 location, but co-located with the other propagation site for readability.
- Hydration trigger relocation (Task 3.6): moved from `QwenTTSService.start()` to dedicated `hydrate_voice_clone_prompt_cache()` invoked from the orchestrator after `voice_manager.start()` — `start()`-internal hook would have always seen `_voice_profile_manager is None` because the orchestrator constructs the manager AFTER `tts.start()` returns.
- 2026-05-08 — Tasks 7 + 8 deferred to follow-up Commander-led session (Task 7 needs `build_release.bat` + interactive .exe smoke; Task 8 lands at closure).
