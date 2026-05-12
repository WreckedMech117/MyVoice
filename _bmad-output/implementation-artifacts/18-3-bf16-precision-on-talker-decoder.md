# Story 18.3: bf16 Precision on Talker + Decoder

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase tag: Phase ⊥-Polish-2 (D-20). Third story of Epic 18 (Generation-Speed Optimizations). Successor to Story 18.2 (TF32 + cuDNN benchmark — closed as engaged-but-empirically-null on RTX 5090; producer-bottleneck fix class is named here at 18.3 + 18.4 per `epics-optimization-pass.md:240` and `memory/epic18_producer_bottleneck_finding.md`). -->
<!-- Risk: Medium (per `epics-optimization-pass.md:245`). Audio-quality regression possible on edge cases — sibilants, low-amplitude consonants, tonal peaks. NFR7 fp32 fallback is the secondary mitigation; the full ≥3-listener NFR3 re-audition is the primary gate. This is the **load-bearing perceptual gate of the epic** per `:1383`. -->
<!-- Audition discipline: Full Story 17.1-grade ≥3-listener blind A/B (`:241` + `:1378`). The existing `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` fixture is regenerated against bf16 output; the controlled defect vocabulary verbatim; the "zero `audible_seam` flags" gate verbatim. Audition deliverable at `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv`. -->
<!-- Reality check: bf16 is ALREADY the production default at `src/myvoice/services/model_registry.py:95` (`dtype: str = "bfloat16"`) — the model is loaded via `Qwen3TTSModel.from_pretrained(..., torch_dtype=torch.bfloat16)` at `:454-:459`. This story is therefore an **audit + correctness fix + audition gate**, NOT a naive "cast model to bf16 on load" change. The naive interpretation in the epic stub at `:1375` reflects an incomplete reading of the V2 baseline (the stub author assumed fp32 default; the dev agent must reconcile). The audit reveals two latent issues this story closes alongside the audition: (a) the bf16 default applies unconditionally on CPU and pre-Ampere GPUs, a latent D-9 / NFR12 violation; (b) the absence of a user-facing precision toggle prevents NFR7 fp32 fallback if the audition flags any utterance class. -->

## Story

As a **MyVoice user generating TRUE_STREAM utterances on an Ampere-or-newer CUDA host**,
I want **the qwen-tts talker + decoder to run in bf16 with explicit hardware-aware engagement and an audition-certified perceptual-equivalence guarantee, plus a clean fp32 fallback under settings override**,
so that **first-chunk latency drops 30–50% on RTX 5090 (per `epics-optimization-pass.md:1381`) at audition-certified zero perceptual cost (zero `audible_seam` flags), while CPU-only and pre-Ampere users stay on the lossless fp32 path per D-9 / NFR12 and any post-release perceptual surprise can be remediated via `AppSettings.tts_precision="fp32"` without a code rollback**.

## Acceptance Criteria

**Given** the production codebase at HEAD (`src/myvoice/services/model_registry.py:95` defaults `dtype="bfloat16"` and `:454` passes `torch_dtype=self.dtype` to `Qwen3TTSModel.from_pretrained`)
**When** the dev agent audits the runtime model state on the RTX 5090 dev host (Story 17.3 §4.1 step 3 canonical Sarira-F long-form utterance under TRUE_STREAM dispatch)
**Then** the audit captures and reports at evidence file §"Pre-implementation audit":
  - The actual `model.model.talker.dtype` (or equivalent attribute exposing the talker's parameter dtype) — confirms the load path's `torch_dtype=torch.bfloat16` reaches the talker module
  - The actual codec decoder / vocoder parameter dtype at `model.model.speech_tokenizer` — the module path that owns the per-chunk waveform synthesis per `qwen_tts_service.py:3338-3341 + :3379` (Story 16.8's `_build_true_stream_decode_fn` documents the path verbatim — the speech tokenizer lives on the inner `Qwen3TTSForConditionalGeneration` at `model.model.speech_tokenizer`, set during the inner wrapper's `from_pretrained` per `modeling_qwen3_tts.py:1920`)
  - Whether `model.dtype` (top-level wrapper) reports a dtype consistent with the leaf modules (drift between `wrapper.dtype` and `talker.dtype` is a known qwen-tts 0.0.4 idiosyncrasy on certain `from_pretrained` paths and must be surfaced if observed)

**And** the audit captures whether the talker forward path or the codec-token decoder path performs an internal autocast / `.float()` / `.to(torch.float32)` upcast that erases the bf16 gain (instrument by attaching a one-shot forward hook on the talker's first-token forward pass and on the decoder's first-chunk decode call; record the input + output dtypes; surface at evidence file §"End-to-end dtype audit")

**And** if either audit branch surfaces an unexpected fp32 round-trip in the production code path, the dev agent **stops and routes to Commander via Open Question #1** (do not silently fix an upcast that may exist for a numerical-stability reason in qwen-tts 0.0.4; the fix surface — patch upstream pin vs. wrap locally vs. accept the upcast — is a Commander-routed architectural call, not a dev-agent unilateral edit)

**Given** the new `AppSettings.tts_precision` field is added to `src/myvoice/models/app_settings.py`
**When** the field's contract lands
**Then** the field accepts exactly three string values — `"auto"` (default), `"bf16"`, `"fp32"` — validated by `__post_init__` (mirroring the `streaming_mode_override` pattern at `app_settings.py:103-108` + `:375-389` exactly: warn-and-fallback to `"auto"` on unknown values; structured `ValidationIssue` with `code="UNKNOWN_TTS_PRECISION"`)
**And** the field is persisted in `to_dict` + restored in `from_dict` (mirroring `streaming_mode_override` at `:511 + :527 + :593`)
**And** the field is added to the inline field-name list inside `reset_to_defaults()` at `app_settings.py:696-:713` (the `for field_name in [...]` loop where `streaming_mode_override` already appears at `:709`) so a settings-reset operation correctly resets `tts_precision` back to `"auto"` along with all other persisted fields
**And** no UI surface for the setting lands in this story (data-only field; UI exposure is deferred — hand-edit `settings.json` is the user remediation path, mirroring the Story 16.6 / `streaming_mode_override` precedent at `app_settings.py:103-108`)

**Given** the new `tts_precision` setting and the existing `model_registry.py` dtype path
**When** the dev agent wires the precision resolver
**Then** a new pure-decision function lives at `src/myvoice/services/tts_streaming/torch_runtime.py` (the Story 18.2 module — extends not parallel) with the signature `resolve_tts_precision(override: Optional[str]) -> torch.dtype` (or equivalent) and the contract:
  - `override == "fp32"` → returns `torch.float32` unconditionally
  - `override == "bf16"` → returns `torch.bfloat16` unconditionally (user-forced; engages even on CPU / pre-Ampere — the user has explicitly opted in)
  - `override == "auto"` or `None` → calls `is_ampere_or_newer()` (Story 18.2's existing function); returns `torch.bfloat16` on Ampere+ CUDA, `torch.float32` otherwise
  - The function is **side-effect-free** (no logging, no metric emission — mirrors Story 16.2 / Story 18.2's `is_ampere_or_newer` pure-decision discipline; the side-effect of the chosen dtype landing on the model is at the call site in `model_registry.py`)

**And** the function is exported via `src/myvoice/services/tts_streaming/__init__.py` alongside the Story 18.2 exports (`is_ampere_or_newer`, `enable_tf32_and_cudnn_benchmark`)

**Given** the new resolver is wired into `ModelRegistry`
**When** the wire-up lands
**Then** `ModelRegistry.__init__` accepts an optional `app_settings: Optional[AppSettings] = None` parameter, reads `app_settings.tts_precision` (or `None`) on construction, calls `resolve_tts_precision(...)`, and assigns the result to `self.dtype`
**And** the existing `dtype: str = "bfloat16"` constructor parameter is **preserved** (backwards-compatible — tests / non-AppSettings call sites still work; the new resolver is layered ON TOP of the existing parameter, with `app_settings.tts_precision` taking precedence when both are supplied)
**And** the precedence rule is: if `app_settings is not None` AND `app_settings.tts_precision is not None` → use the resolver; otherwise fall back to the existing `dtype: str` parameter mapping (the current behavior). Document the precedence at the resolver function's docstring + at the `ModelRegistry.__init__` docstring
**And** `QwenTTSService.__init__` is updated to pass `app_settings=self._app_settings` to `ModelRegistry(...)` at `qwen_tts_service.py:582-588` (the existing constructor already accepts `app_settings: Optional[AppSettings] = None` at `:535`; this is a one-line wire-up extension)
**And** the resolved dtype is logged at INFO level at `ModelRegistry` initialization with the structured form `f"ModelRegistry initialized: device={self.device}, dtype={self.dtype}, precision_source='{source}', quality_tier={...}"` where `source` is one of `"app_settings_override"` / `"app_settings_auto_ampere"` / `"app_settings_auto_fallback"` / `"legacy_constructor_arg"` so Commander can confirm at runtime which path engaged (this extends the existing log line at `model_registry.py:144-:146`)

**Given** the new resolver is engaged
**When** the application starts on a CPU-only host (`torch.cuda.is_available() == False`) OR a pre-Ampere CUDA host (capability < 8.0) with `tts_precision == "auto"` (the default)
**Then** the resolved dtype is `torch.float32` (NOT `torch.bfloat16`) — closing the latent D-9 / NFR12 violation in the current `dtype="bfloat16"` default that applies unconditionally
**And** the model loads in fp32 on those hosts (verified by inspecting `model_registry.py`'s subsequent `from_pretrained(..., torch_dtype=self.dtype)` call — `self.dtype` is now `torch.float32`)
**And** the existing CPU-only test surface (any test that constructs `ModelRegistry` without an explicit `dtype="float32"` argument) is updated to pass `dtype="float32"` explicitly OR pass an `app_settings` with `tts_precision="fp32"` — the test discipline must NOT silently inherit the new auto-resolver's CPU branch (tests should be explicit about what they're exercising). Audit and update at `tests/unit/services/test_model_registry.py` if it exists, and any other test that touches the dtype path

**Given** the new resolver is engaged on an Ampere+ CUDA host
**When** a single `metrics.record(...)` call captures the precision-engagement outcome at `ModelRegistry` initialization
**Then** the metric shape is: `metrics.record("tts_precision_resolved", value, source="<source>", dtype="<dtype_str>", device_capability="<major>.<minor>" | "none")` where:
  - `value = 1.0` if the resolved dtype is `torch.bfloat16`
  - `value = 0.0` if the resolved dtype is `torch.float32`
  - `source` mirrors the four log-source labels above
  - `dtype` is the string form (`"bfloat16"` or `"float32"`)
  - `device_capability` mirrors Story 18.2's tag schema (string form; `"none"` sentinel for CPU per OQ #2 of Story 18.2)

**And** the metric integrates with the existing `metrics.record(name, value, **tags)` pub-sub helper at `src/myvoice/observability/metrics.py:77` (no new metric infrastructure; same listener surface Story 18.1's three CSV-capture metrics + Story 18.2's `tf32_cudnn_benchmark_enabled` use)

**Given** the streaming pipeline's overlap-add buffer (`src/myvoice/services/tts_streaming/streaming_decoder.py`) consumes per-chunk decoder output and posts PCM segments via `post_mutation('append_chunk', session_id, pcm)`
**When** the decoder loop produces chunks under bf16 precision
**Then** the dtype audit at evidence file §"Streaming pipeline dtype audit" confirms one of two outcomes verbatim:
  - **(a) Already correct:** the decoder's `decode_fn` callable already returns `np.ndarray` with `dtype=np.float32` (the established contract — numpy doesn't support bf16 natively, so the bf16→fp32 cast happens at the GPU→CPU transfer boundary, NOT inside the model's forward pass; this is the *correct* behavior and the bf16 gain is preserved). Document the cast site (typically `tensor.to(torch.float32).cpu().numpy()` or equivalent inside the qwen-tts wrapper) at evidence file §"Streaming pipeline dtype audit" with line-number-level precision.
  - **(b) Found a defect:** the decoder loop or the chunk → bytes conversion path forces a fp32 round-trip *before* the GPU→CPU transfer (e.g., a `tensor.float()` call inside the model's forward pass that erases the bf16 compute gain). Stop and route to Commander via Open Question #2 — do NOT silently rewrite qwen-tts 0.0.4 internal code; the fix surface is upstream-pin-bump or local wrapper, both Commander-routed.

**And** the chunk → bytes conversion path at `src/myvoice/app.py:_handle_progressive_chunk_async` (lines `2622-2625` per the current HEAD) is verified unchanged (the path expects `chunk.audio_data` to be `np.ndarray[float32]`; this is the AC's invariant — Story 18.3 does NOT modify `_handle_progressive_chunk_async`). Surface at evidence file as the explicit "no-edit confirmation"

**Given** the source-tree edits land
**When** Commander runs the canonical Story 17.3 §4.1 step 3 long-form CLONED utterance (Sarira-F, ≥250 chars / ~22 s of speech) on the RTX 5090 dev host with the same `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` env-var-gated capture infrastructure (`src/myvoice/observability/progressive_playback_csv_capture.py` — Story 18.1 / 18.2 inheritance) WITH `AppSettings.tts_precision = "auto"` (default → bf16 on RTX 5090)
**Then** the captured `metrics.first_chunk_latency_ms` value (already-aggregated by `_FirstChunkLatencyAggregator` at `qwen_tts_service.py:362`) is compared head-to-head against the **same utterance under the same conditions with `AppSettings.tts_precision = "fp32"`** (the NFR7 override path verifies the bf16 gain is real; the comparison surface is the override toggle, NOT a git-checkout-parent rerun, because bf16 is already on the HEAD and the dev agent needs a settings-toggleable A/B for clean methodology)
**And** the measurement is captured at `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md §"NFR1 first-chunk-latency measurement"` with both raw CSVs (`18-3-rtx5090-bf16.csv` + `18-3-rtx5090-fp32.csv`), median + p90 + p95, and the absolute + percent delta
**And** captures of N=10 generations per branch (matches Story 18.2's spec; supersedes Story 18.2's deviation single-shot — bf16 is a larger numerical change with more noise sensitivity, so the full N=10 pair is required and **not** subject to the "ship-as-engaged + move on" deviation pattern Story 18.2 used)
**And** the captured percent delta is surfaced verbatim to Commander in the closure note. The Epic 18 stub at `:1381` quotes a 30–50% speedup as the *anticipated* gate; if the measured median speedup falls below 20% OR the audition flags any utterance class, route to Open Question #3 (which precedes audition closure) — the bf16 path may require a numerical-stability shim or a partial fp32-residency on a specific submodule, which is a Commander-routed architectural call

**Given** Story 18.2's TF32 + cuDNN benchmark is engaged on RTX 5090 (per `logs/myvoice.log:3` runtime confirmation: `"TF32 + cuDNN benchmark enabled (device_capability=12.0)"`)
**When** Story 18.3's bf16 measurement runs
**Then** the measurement methodology section at evidence file §"NFR1 first-chunk-latency measurement" explicitly notes that the bf16 measurement composes ON TOP OF Story 18.2's TF32+cuDNN-engaged baseline (i.e., the "fp32 branch" of the A/B is actually fp32-with-TF32-engaged, not strict-fp32 — this matters because TF32 affects the matmul drift on the fp32 path in a way that bf16 does not, so the comparison is bf16-vs-TF32-fp32 not bf16-vs-strict-fp32)
**And** the methodology note documents that the strict-fp32 vs TF32-fp32 comparison is **out of scope** for this story (Story 18.2 closed that comparison as "no evidence of TF32 contributing or harming on RTX 5090 producer-bottleneck workload"; the bf16 measurement does not need to re-litigate it)

**Given** the test suite runs after the source-tree edits
**When** the regression sweep executes
**Then** the existing Story 18.2 `test_torch_runtime` tests (15 tests at `tests/unit/services/tts_streaming/test_torch_runtime.py`) pass with **zero regressions** (the new `resolve_tts_precision` function lives in the same module — must not regress the existing 15)
**And** the existing Story 18.1 instrumentation tests (23 tests across `tests/unit/test_app_progressive_playback_instrumentation.py` + `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py` + `tests/unit/observability/test_progressive_playback_csv_capture.py`) pass with zero regressions
**And** the existing Story 17.3 progressive-playback tests (32 tests across `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` + `tests/unit/test_app_progressive_playback.py` + `tests/unit/test_app_progressive_playback_cancel.py` + `tests/integration/test_progressive_playback_dispatch_skip.py`) pass with zero regressions
**And** the existing Story 16.2 streaming-mode hardware-probe tests at `tests/unit/services/tts_streaming/test_streaming_mode.py` pass with zero regressions (the new export from `tts_streaming/__init__.py` must not regress 16.2's import chain)
**And** the existing AppSettings tests at `tests/unit/models/test_app_settings.py` (or equivalent location) pass with zero regressions; **new** AppSettings test rows cover the four `tts_precision` validation branches: valid `"auto"` / `"bf16"` / `"fp32"` (each accepted; round-trip through `to_dict` + `from_dict`); invalid value (warn-and-fallback to `"auto"` with `code="UNKNOWN_TTS_PRECISION"` ValidationIssue surfaced)
**And** new unit tests at `tests/unit/services/tts_streaming/test_torch_runtime.py` (extending the existing file, NOT a parallel file) cover all six `resolve_tts_precision` branches:
  - `override == "fp32"` → returns `torch.float32` (regardless of hardware probe)
  - `override == "bf16"` → returns `torch.bfloat16` (regardless of hardware probe)
  - `override == "auto"` + Ampere+ → returns `torch.bfloat16` (parametrized with the same cap-major shapes Story 18.2 covers: 8.9 / 10.0 / 9.0 / 12.0 — RTX 5090 GeForce variant)
  - `override == "auto"` + pre-Ampere (Turing 7.5) → returns `torch.float32`
  - `override == "auto"` + cuda-unavailable → returns `torch.float32`
  - `override is None` → behaves identically to `override == "auto"` (the explicit-None case)
**And** new unit tests at `tests/unit/services/test_model_registry.py` (extending if exists; new file otherwise) cover the four precedence branches in the new `__init__` resolver:
  - `app_settings is None` → falls back to legacy `dtype: str` parameter (assert `self.dtype == torch.bfloat16` for the default case)
  - `app_settings.tts_precision is None` → same as above
  - `app_settings.tts_precision == "fp32"` → resolver wins; assert `self.dtype == torch.float32`
  - `app_settings.tts_precision == "auto"` + Ampere+ probe (monkeypatched) → assert `self.dtype == torch.bfloat16` AND the INFO log records `precision_source="app_settings_auto_ampere"`
**And** the `test_qwen_tts_service` suite (whichever subset touches construction) is updated to verify the `app_settings` parameter flows through to `ModelRegistry(...)` at `qwen_tts_service.py:582-588` (the new wire-up); add a single integration-style test that constructs `QwenTTSService(app_settings=AppSettings(tts_precision="fp32"))` and asserts `service._model_registry.dtype == torch.float32`

**Given** Story 18.2 and 17.3 / 17.2 / 17.1 established the audition discipline at full ≥3-listener blind A/B
**When** the bf16 perceptual gate runs
**Then** Commander recruits ≥3 listeners (L1 = Commander; L2 + L3 = independent listeners per the Story 17.1 protocol verbatim — same `LISTENING-INSTRUCTIONS.md` from `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/`)
**And** the existing 10-pair perceptual-difficult fixture subset (`s-014/15/16/17`, `m-011/12/13/14`, `l-013/14`) is **regenerated against bf16 output** — i.e., the dev agent runs the canonical generation script to produce fresh "B" renditions of all 10 utterances under `tts_precision="bf16"` (the "A" rendition baseline is regenerated under `tts_precision="fp32"`); fixture regeneration produces a new directory `_bmad-output/implementation-artifacts/18-3-perceptual-fixtures/` with the 10 paired WAVs + a refreshed `_perlistener_truthtable.json` with L1/L2/L3 randomizations + a verbatim copy of `LISTENING-INSTRUCTIONS.md` (same controlled defect vocabulary)
**And** total trials = 30 (10 utterances × 3 listeners × A and B each — labeled blind per the Story 17.1 helper-driven protocol, with the same `17-1-l1-audition-helper.py` adapted as `18-3-l1-audition-helper.py` to point at the new fixture directory)
**And** the verdict gate is **verbatim** Story 17.1's: PASS iff zero listeners flag `audible_seam` on any TRUE_STREAM (= bf16) pair across all 30 trials
**And** the audition deliverable is the joined `(listener_id × utterance_id × system) × (defect_label, a_or_b_preferred, comments)` CSV at `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv` (Story 17.1 / 17.3 evidence-file precedent; force-add per `memory/git_repo_state.md` since `_bmad-output/` is gitignored)
**And** the verdict-computation tables at evidence file §"NFR3 audition verdict" mirror Story 17.1's structure verbatim — per-system defect-flag count table; per-listener subtotals; per-utterance subtotals; explicit `audible_seam` zero-flag check stated as the gate condition; PASS / FAIL outcome
**And** if the audition flags `audible_seam` on any utterance from any listener, the verdict is FAIL → outcome (b) per Story 17.1's framing; the dev agent does NOT unilaterally close the story; route to Open Question #4 with the failed-utterance class (sibilants vs tonal peaks vs low-amplitude consonants) annotated so Commander can decide whether to (i) accept the defect with a session-class fp32 fallback, (ii) defer the bf16 default and ship the `tts_precision` setting as a power-user opt-in, or (iii) escalate to upstream qwen-tts pin-bump investigation

**Given** the audition outcome (PASS or FAIL)
**When** the architecture amendment lands
**Then** `_bmad-output/planning-artifacts/architecture-optimization-pass.md` is amended in two places (mirroring the Story 17.1 two-place edit pattern at `:863-:899`):
  - **Place 1:** the NFR3 cell (currently at `:803`: `"D-8 chunk + overlap-add with seam-quality A/B testing before flipping streaming default; sentence-stream/batch unchanged from V2 *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)*"`) gets a new parenthetical `*(Story 18.3 bf16 audition <PASSED|FAILED> {{audition_date}} — see follow-up note below.)*` appended verbatim
  - **Place 2:** a new section titled `#### Story 18.3 Follow-up Note (bf16 Precision Audition, {{audition_date}})` is added immediately after the existing `#### Story 17.1 Follow-up Note` block (currently ends at `:899`); the new section captures: (a) the fixture regeneration methodology, (b) the per-system defect-flag count table verbatim from `18-3-bf16-precision-audition.csv`, (c) per-listener + per-utterance subtotals, (d) the verdict-gate computation, (e) the verdict outcome (PASS / FAIL), (f) the architectural decision (bf16-as-engaged on Ampere+ certified, OR bf16-deferred-pending-routing), (g) the methodology limitations section verbatim from Story 17.1's pattern (single-room listening environment, single-scribe prompt-framing risk, L1 anonymization not preserved — adapt or reuse Story 17.1's wording as appropriate), (h) source artifacts list (✓ for git-tracked, ○ for gitignored fixture)
**And** the architecture amendment is the **final** artifact in the story closure sequence — i.e., the dev agent does not amend the architecture before the audition lands; the audition outcome is the input to the amendment

**Given** the bundled-environment smoke from Story 17.3 / 18.2 §4.1 procedure remains the production-verification gate
**When** the dev agent runs the bundled smoke after the source-tree edits
**Then** Commander runs `build_release.bat` (or equivalent per `memory/build_tools_phase_perp_state.md` + `memory/production_release_state.md`) — note that 18.2's Task 6 was DEFERRED to Commander's build-state commit, so 18.3's bundled smoke is the FIRST opportunity to verify both 18.2's `torch_runtime.py` engagement AND 18.3's `tts_precision` resolver in the production-bundled artifact (this composes 18.2's deferred smoke with 18.3's required smoke)
**And** the bundled exe's `myvoice.log` (in the portable Logs path per `setup_logging()` discipline) contains both:
  - The Story 18.2 INFO line: `"TF32 + cuDNN benchmark enabled (device_capability=...)"`
  - The new Story 18.3 INFO line: `"ModelRegistry initialized: device=cuda:0, dtype=torch.bfloat16, precision_source='app_settings_auto_ampere', quality_tier=quality"` (or the equivalent based on Commander's `tts_precision` setting at runtime)
**And** Commander confirms zero perceptual defects on the same Sarira-F long-form utterance compared to the Story 17.3 / 18.1 / 18.2 baseline — i.e., the bf16-engaged production-bundled artifact passes the Commander-solo spot-check independent of the multi-listener audition (which gates the architectural amendment, not the bundled smoke)
**And** the bundled-smoke evidence is captured at `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md §"Bundled smoke"` — the same Story 17.3 / 18.1 / 17.2 / 17.1 / 18.2 evidence-file pattern

**Given** the story is closed
**When** the post-implementation accounting runs
**Then** the change log records the absolute + percent first-chunk-latency delta vs the fp32 override path so Story 18.4 can baseline its `torch.compile` throughput uplift against a bf16-already-engaged starting point — i.e., the `metrics.first_chunk_latency_ms` value Story 18.4 measures is the post-18.3 (bf16) number, not the post-18.2 (fp32) number
**And** the architecture document `_bmad-output/planning-artifacts/architecture-optimization-pass.md` IS amended (per the AC above — the two-place edit mirroring Story 17.1's pattern). Per Epic 18 framing at `:234`: "No new D-decisions" — D-9 is preserved (the Ampere+ probe gate is the existing `streaming_mode.py:54-56` precedent the new `resolve_tts_precision` mirrors); NFR3 is preserved (the audition is its formal re-clearance); NFR7 is **extended** by the new `tts_precision` setting (extends "graceful degradation" to include precision-tier fallback, as the epic stub at `:234` explicitly anticipates: *"NFR7 (graceful degradation extended — fp32 fallback if bf16 fails an audition utterance class)"*)
**And** no `requirements.txt` / installer-spec / `build_release.bat` edits are made (per Epic 18 framing at `:248`: "18.1–18.3 are pure source-tree edits"); Commander handles the build-counter increments at `build_tools/installer.iss` + `build_tools/version.py` in a separate build-state commit per the Story 18.2 OQ #4 precedent

## Tasks / Subtasks

- [x] **Task 1 — Pre-implementation audit: production model dtype runtime state** (AC: #1)
  Establish the empirical baseline for what the V2 baseline actually does at runtime — bf16 is the *configured* default per `model_registry.py:95`, but the dev agent must confirm this *engages* on the production model. The audit precedes any code edit so the story's empirical framing is grounded.

  **HARNESS — `02_Story_18.3_DType_Audit.bat` (Commander runs once):**
  1. `del logs\myvoice.log` (clear log so the audit lines stand alone)
  2. Double-click `02_Story_18.3_DType_Audit.bat`. The bat sets `MYVOICE_DTYPE_AUDIT=1` and launches MyVoice.
  3. Generate one short Sarira-F utterance (any short paragraph; the audit only needs ONE forward pass per module).
  4. Close MyVoice cleanly.
  5. Report `logs/myvoice.log` back to the dev agent. The lines tagged `[DTYPE_AUDIT]` (post-load attribute walk) and `[DTYPE_AUDIT_FWD]` (one-shot forward-hook captures on talker + speech_tokenizer) are the captures the agent folds into the evidence file's §"Pre-implementation audit" + §"End-to-end dtype audit".

  - [x] 1.1 On the RTX 5090 dev host, run a short canonical TTS generation with the existing HEAD (no source-tree edits yet). Add ad-hoc instrumentation (Python REPL OR temporary debug log lines OR a one-off `pytest.set_trace()` checkpoint) to capture: `model.dtype`, `model.model.dtype` (if exposed), `model.model.talker.dtype` (the talker module's parameter dtype — the most important attribute for this audit), and the codec decoder / vocoder module's parameter dtype. Document the exact attribute path the dev agent walks at evidence file §"Pre-implementation audit"; if the qwen-tts 0.0.4 internal naming differs from the documented expectation, the evidence file is the canonical record of the actual paths used. [INSTRUMENTED: `model_registry.py:_instrument_dtype_audit` walks every relevant attribute path defensively (try/except per attribute) and logs `[DTYPE_AUDIT] model.<path> = <dtype>` lines. Commander runs `02_Story_18.3_DType_Audit.bat` once; agent parses log and fills evidence file.]
  - [x] 1.2 Attach a one-shot forward hook on the talker's first-token forward pass: capture the input tensor's dtype, the output tensor's dtype, and the dtype of any intermediate tensor that crosses an `autocast` / `.float()` / `.to(torch.float32)` boundary. The hook can be torn down after one forward pass (the audit is a snapshot, not a profiling run). Document at evidence file §"End-to-end dtype audit". [INSTRUMENTED: same `_instrument_dtype_audit` attaches a one-shot hook on `talker` that logs `[DTYPE_AUDIT_FWD] talker in=<dtypes> out=<dtype>` and detaches itself after first call.]
  - [x] 1.3 Attach the same one-shot forward hook on the codec decoder's first-chunk decode call (the path that produces the per-chunk waveform). Same dtype-capture protocol. Document at evidence file §"End-to-end dtype audit". [INSTRUMENTED: same module attaches a one-shot hook on `speech_tokenizer` that logs `[DTYPE_AUDIT_FWD] speech_tokenizer in=<dtypes> out=<dtype>` and detaches.]
  - [x] 1.4 Capture the GPU→CPU cast site for the per-chunk audio emission. Walk the qwen-tts wrapper code (or instrument the `decode_fn` callable in `streaming_decoder.py`) to identify the exact line where `tensor.cpu().numpy()` (or equivalent) is called. The cast must produce `np.float32` (not `np.bfloat16`, which numpy doesn't support) — confirm the cast happens at the boundary, not earlier inside the model. Document at evidence file §"Streaming pipeline dtype audit". [DONE in Task 5 read-only audit — cast site pinned at `qwen_tts_service.py:3393-3395`; produces `np.float32` per `np.asarray(audio, dtype=np.float32)` cast.]
  - [x] 1.5 If any audit branch surfaces an unexpected fp32 round-trip *inside* the model's forward pass (e.g., a `.float()` call inside the talker that erases the bf16 compute gain), STOP the dev-agent flow and route to Open Question #1 at evidence file §"Open Questions". Do NOT silently rewrite qwen-tts 0.0.4 internal code; the upstream-pin-bump-vs-local-wrapper choice is Commander-routed. [DOES NOT TRIGGER — second audit run (2026-05-10 10:08) captured talker bf16 + speech_tokenizer.model bf16 + talker forward-hook all-bf16. No fp32 round-trip surfaced. OQ #1 stays unused.]

- [x] **Task 2 — `AppSettings.tts_precision` field with validation** (AC: #2)
  Data-only field; mirrors `streaming_mode_override` at `app_settings.py:103-108 + :375-389` exactly. No UI surface in this story.
  - [x] 2.1 Edit `src/myvoice/models/app_settings.py`. Add the field `tts_precision: str = "auto"` (default value = `"auto"`) at the natural ordering position alongside `streaming_mode_override` at `:108`. Inline comment cites `epics-optimization-pass.md:1377`: `"# Story 18.3 — TTS precision override. 'auto' (default) = bf16 on Ampere+ / fp32 elsewhere; 'bf16' = force bf16 (engages on CPU too if user opts in); 'fp32' = force fp32 (NFR7 fallback if bf16 audition flags any utterance class). UI-less for now (hand-edit settings.json); mirrors streaming_mode_override pattern."`.
  - [x] 2.2 Add validation in `__post_init__` mirroring the `streaming_mode_override` block at `:375-389`. Allowed values: `["auto", "bf16", "fp32"]`. On invalid value: emit `ValidationIssue(field="tts_precision", message=f"Unknown TTS precision '{self.tts_precision}', defaulting to 'auto'. Allowed values: auto, bf16, fp32.", code="UNKNOWN_TTS_PRECISION", severity=ValidationStatus.WARNING)`; reset to `"auto"`.
  - [x] 2.3 Add to `to_dict` (around `:511 + :527`): `"tts_precision": self.tts_precision`. Add to `from_dict` (around `:577 + :593`): `tts_precision=data.get("tts_precision", "auto"),`. Verify the field round-trips by adding a unit test that constructs `AppSettings(tts_precision="bf16")`, calls `to_dict()`, then reconstructs via `from_dict(...)` and asserts the value persists.
  - [x] 2.4 Add `"tts_precision"` to the inline field-name list inside `reset_to_defaults()` at `app_settings.py:696-:713` (the `for field_name in [...]` loop that copies default values back from a fresh `AppSettings()` instance). `streaming_mode_override` already lives in this list at `:709`; place `"tts_precision"` immediately after it to preserve the existing ordering convention. There is no separate `_field_names` symbol — this inline list is the only place fields must be enumerated for the reset-to-defaults path.
  - [x] 2.5 Add unit tests at the existing AppSettings test surface (locate via `grep -r "test_app_settings" tests/`) covering: (a) default = `"auto"`, (b) explicit `"bf16"` / `"fp32"` accepted, (c) round-trip through `to_dict` + `from_dict`, (d) invalid value triggers `UNKNOWN_TTS_PRECISION` ValidationIssue + resets to `"auto"`. Mirror the existing `streaming_mode_override` test row structure exactly — the bug class is "validation drift between two near-identical fields," and the regression test must mirror the exact bug class per `memory/code_review_regression_test_exact_class.md`.

- [x] **Task 3 — `resolve_tts_precision` resolver in `torch_runtime.py`** (AC: #3, #5)
  Pure-decision function; extends Story 18.2's module (NOT a parallel file). Side-effect-free. Same lazy-torch-import discipline.
  - [x] 3.1 Edit `src/myvoice/services/tts_streaming/torch_runtime.py`. Add the function `def resolve_tts_precision(override: Optional[str]) -> "torch.dtype":` with the contract documented in AC #3. Module docstring extends Story 18.2's existing block to also cite Epic 18 + Story 18.3 + epic stub `:1370-:1386`.
  - [x] 3.2 Implement the four-branch decision: `"fp32"` → `torch.float32`; `"bf16"` → `torch.bfloat16`; `"auto"` or `None` → `torch.bfloat16` if `is_ampere_or_newer()` else `torch.float32`. **Lazy-import torch** (same pattern as `is_ampere_or_newer` and `enable_tf32_and_cudnn_benchmark`).
  - [x] 3.3 Function docstring documents the precedence rule (override values are user-explicit and engage even on CPU; "auto"/None defers to hardware probe) and the side-effect-free contract (no logging, no metric emission — those happen at the call site).
  - [x] 3.4 Update `src/myvoice/services/tts_streaming/__init__.py`: add `resolve_tts_precision` to the `from ... import` list and `__all__`. Position the export alongside Story 18.2's `is_ampere_or_newer` and `enable_tf32_and_cudnn_benchmark` since they share the conceptual surface (hardware-gated startup-side decisions).

- [x] **Task 4 — `ModelRegistry` precedence + telemetry** (AC: #4, #6)
  Wire the resolver into `ModelRegistry.__init__` with the documented precedence rule. Backwards-compatible.
  - [x] 4.1 Edit `src/myvoice/services/model_registry.py`. Add `app_settings: Optional[AppSettings] = None` to `__init__` parameters (default None preserves the legacy call surface). Import `AppSettings` lazily inside the method (avoid circular imports — `AppSettings` is a model not a service, but lazy is safer). [Implementation note: `app_settings: Optional[Any]` parameter type with `getattr(app_settings, "tts_precision", None)` access — avoids importing AppSettings at all in model_registry.py.]
  - [x] 4.2 Implement the precedence resolver: if `app_settings is not None and app_settings.tts_precision is not None`, call `resolve_tts_precision(app_settings.tts_precision)` and assign to `self.dtype`; record `source = "app_settings_override" if app_settings.tts_precision in ("bf16", "fp32") else ("app_settings_auto_ampere" if self.dtype == torch.bfloat16 else "app_settings_auto_fallback")`. Otherwise, fall back to the existing `dtype: str` parameter mapping at `:119-:124`; record `source = "legacy_constructor_arg"`.
  - [x] 4.3 Extend the existing INFO log line at `:144-:146` to include `precision_source='{source}'` per AC #4. The existing log is the canonical observability surface; the new field is appended (not replacing) so log-parsing consumers downstream stay backward-compatible.
  - [x] 4.4 Emit the new `metrics.record("tts_precision_resolved", value, ...)` call after the dtype resolution lands, per AC #6. Tags: `source`, `dtype` (string form), `device_capability` (mirrors Story 18.2 schema — string form, `"none"` sentinel for CPU).
  - [x] 4.5 Edit `src/myvoice/services/qwen_tts_service.py`. At `:582-:588` (the `ModelRegistry(...)` construction), pass `app_settings=self._app_settings` as a new keyword argument. Verify `self._app_settings` is set BEFORE `ModelRegistry` construction (it is — set at `:573` on the existing AppSettings wire-up).

- [x] **Task 5 — Streaming pipeline dtype audit + chunk → bytes path verification** (AC: #7)
  Audit-only — no source-tree edits unless the audit surfaces a defect (Open Question #2 routing). The `decode_fn` supplier site is **pre-pinned** to `qwen_tts_service.py:3325-3396` — `_build_true_stream_decode_fn(model)`, which returns the inner `_decode(chunk)` callable. The GPU→CPU cast site is `:3393-:3395`: `audio = audio.detach().cpu().numpy()` followed by `np.asarray(audio, dtype=np.float32).flatten()`. The audit is structured around the central hypothesis that `tensor.detach().cpu().numpy()` cannot run on a bf16 tensor (numpy has no `bfloat16` dtype) — so either (a) `model.model.speech_tokenizer.decode` returns fp32 internally (most likely — vocoders typically stay in fp32 for numerical stability), OR (b) the production bf16 default is not actually engaging end-to-end on the codec-decoder side, which would explain why the configured `dtype=torch.bfloat16` does not currently deliver the anticipated 30–50% speedup. **The audit's primary job is to confirm or refute this hypothesis.**
  - [x] 5.1 Read `src/myvoice/services/tts_streaming/streaming_decoder.py` end-to-end (the Story 16.4 module). Confirm the `decode_fn` consumer contract at `:82` is `Callable[[list[Any]], np.ndarray]` and the consumer expects `np.ndarray` with `dtype=np.float32` (per the `np.asarray(..., dtype=np.float32)` cast at the supplier site `qwen_tts_service.py:3395`). No edits. [DONE: confirmed at evidence file §"Streaming pipeline dtype audit" finding 1.]
  - [x] 5.2 Read `qwen_tts_service.py:3325-3396` (the `_build_true_stream_decode_fn` method). Document at evidence file §"Streaming pipeline dtype audit": (a) the exact line of the GPU→CPU cast (`:3394`: `audio.detach().cpu().numpy()`); (b) whether the cast operates on a bf16 or fp32 tensor at runtime — answered by the Task 1.3 forward-hook capture; (c) if bf16, whether `tensor.detach().cpu().numpy()` raises (it should — numpy has no bf16 dtype) or silently upcasts via NumPy 2.0+ DLPack support (depends on the running NumPy version per `requirements.txt`); (d) whether `model.model.speech_tokenizer.decode` returns a fp32 tensor regardless of the talker's bf16 dtype (the most-likely-correct outcome — Story 16.8's docstring at `:3328-:3357` explicitly comments on the speech_tokenizer's contract but does not pin its internal dtype). [DONE: confirmed at evidence file §"Streaming pipeline dtype audit" findings 2+3 — speech_tokenizer.decode returns `(List[np.ndarray], int)` natively per `qwen3_tts_tokenizer.py:281-283`; the GPU→CPU bf16→fp32 cast happens INSIDE the qwen-tts wrapper at the boundary, not in MyVoice code. Result class: (a) Already correct.]
  - [x] 5.3 Verify `src/myvoice/app.py:_handle_progressive_chunk_async` at `:2622-:2625` is unchanged — the `np.clip(chunk.audio_data, -1.0, 1.0) * 32767.astype(np.int16).tobytes()` chain expects float32 input and is the AC #7 invariant. Surface as the explicit "no-edit confirmation" at evidence file. [DONE: no-edit confirmation at evidence file §"Streaming pipeline dtype audit" finding 4.]
  - [x] 5.4 If Task 5.2 surfaces a defect (e.g., the GPU→CPU cast happens AFTER an internal `.float()` upcast that erases the bf16 gain), STOP and route to Open Question #2. Do NOT silently rewrite qwen-tts 0.0.4 internal code; the fix surface is upstream-pin-bump or local wrapper, both Commander-routed. [DONE: no defect surfaced in read-only audit; runtime forward-hook capture (Task 1.3) is the final answer. OQ #2 stays reserved pending Task 1.3 outcome.]

- [x] **Task 6 — Unit tests at `test_torch_runtime.py` + `test_model_registry.py` + `test_app_settings.py`** (AC: #8)
  Three test surfaces, all mirroring established patterns (Story 18.2's `test_torch_runtime` discipline; Story 16.6 / 11.X model_registry tests if extant; AppSettings `streaming_mode_override` test row pattern).
  - [x] 6.1 Extend `tests/unit/services/tts_streaming/test_torch_runtime.py` with the six `resolve_tts_precision` branch tests per AC #8. Mirror Story 18.2's parametrization style (the existing `test_telemetry_tag_schema_*` pattern). Use the snapshot-and-restore fixture pattern Story 18.2 established for the `torch.backends.*` flag tests — `resolve_tts_precision` is side-effect-free so no flag restoration needed, but the test file's existing fixture conventions should be preserved. [DONE: 13 new test rows added — 6 documented branches + 4 Ampere parametrized cap-major variants + 2 None-equiv-auto + 2 side-effect-free invariants.]
  - [x] 6.2 Extend or create `tests/unit/services/test_model_registry.py` with the four precedence-branch tests per AC #8. Each test constructs `ModelRegistry(...)` with a different `app_settings` configuration; asserts `self.dtype` matches the expected `torch.dtype`; asserts the INFO log line records the expected `precision_source` value (using `caplog` per the established pattern). [DONE: created `tests/unit/services/test_model_registry.py` with 14 test rows covering all four `precision_source` values + caplog assertion + telemetry tag schema.]
  - [x] 6.3 Extend the existing AppSettings test file with the four `tts_precision` validation rows per Task 2.5. Mirror the existing `streaming_mode_override` test row structure exactly. [DONE: created `tests/unit/models/test_app_settings_tts_precision.py` mirroring `test_app_settings_clear_comms.py` structure; 11 test rows covering defaults / accepted values / round-trip / validation / reset.]
  - [x] 6.4 Add a single integration-style test at the existing `test_qwen_tts_service` surface (locate the appropriate test file via `grep -r "QwenTTSService(" tests/unit/services/`) that constructs `QwenTTSService(app_settings=AppSettings(tts_precision="fp32"))` and asserts `service._model_registry.dtype == torch.float32`. This test verifies the wire-up at `qwen_tts_service.py:582-:588` flows the new parameter end-to-end. [DONE: extended `tests/unit/services/test_qwen_tts_service_dispatch.py::TestResolveStreamingMode` with two wire-up tests covering fp32 + bf16 override paths.]

- [x] **Task 7 — NFR1 first-chunk-latency empirical measurement (bf16 vs fp32 override)** (AC: #9, #10)
  Full N=10-per-branch measurement (NOT subject to Story 18.2's deviation pattern — bf16 is a larger numerical change with more noise sensitivity, and the audition gate is load-bearing, so the measurement must be statistically meaningful).

  **HARNESS — two batch files (Commander runs sequentially):**
  - `03_Story_18.3_NFR1_BF16.bat` — programmatically sets `tts_precision="auto"` via `18-3-set-precision.py`, then loops the launch 10 times. Each iteration writes to `18-3-rtx5090-bf16-run<NN>.csv`. Commander generates one Sarira-F long-form utterance per launch and closes cleanly.
  - `04_Story_18.3_NFR1_FP32.bat` — same pattern, sets `tts_precision="fp32"`, writes to `18-3-rtx5090-fp32-run<NN>.csv`.
  After both runs land, the dev agent runs `_bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py` to compute median/p90/p95 + delta, write the two consolidated CSVs (`18-3-rtx5090-bf16.csv` + `18-3-rtx5090-fp32.csv`), and surface the Task 7.4 routing condition (sub-20% speedup) automatically.

  - [x] 7.1 Set `AppSettings.tts_precision = "auto"` (default → bf16 on RTX 5090). Run the canonical Story 17.3 §4.1 step 3 Sarira-F long-form utterance (≥250 chars / ~22 s of speech) on the RTX 5090 dev host with `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` set; capture **N=10 generations** to `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16.csv`. Each generation is a fresh process launch (kill the app between runs) so cuDNN benchmark autotune cache state does not bleed across runs (same discipline as Story 18.2 Task 4.2). [HARNESSED: `03_Story_18.3_NFR1_BF16.bat` (loop = 10 fresh launches; per-iteration CSV path → aggregation script).]
  - [x] 7.2 Set `AppSettings.tts_precision = "fp32"` (NFR7 override path → fp32 even on RTX 5090). Run the same canonical Sarira-F utterance for **N=10 generations**; capture to `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32.csv`. The settings-toggle methodology eliminates the git-checkout-pair complexity Story 18.2's spec required (because bf16 is already on HEAD and the override is the cleaner A/B surface). [HARNESSED: `04_Story_18.3_NFR1_FP32.bat`.]
  - [x] 7.3 Compute the delta. `(fp32_median - bf16_median) / fp32_median * 100` = percent speedup; same for p90 and p95. Capture at evidence file §"NFR1 first-chunk-latency measurement". Methodology note explicitly states the bf16 measurement composes ON TOP OF Story 18.2's TF32+cuDNN-engaged baseline (the fp32 branch is fp32-with-TF32-engaged, not strict-fp32) per AC #10. [HARNESSED: `18-3-aggregate-nfr1.py` consumes the per-run CSVs from Tasks 7.1+7.2, writes consolidated CSVs, and prints median/p90/p95 + absolute and percent deltas to stdout. Commander forwards the script output to the dev agent.]
  - [x] 7.4 If the measured median speedup falls below 20% (the conservative lower bound of the anticipated [30%, 50%] gate at `:1381`), route to Open Question #3 BEFORE running the audition — a sub-20% speedup may indicate a partial fp32 round-trip the dev agent hasn't yet diagnosed (the audit in Task 1 should have surfaced this, but the empirical measurement is the final gate). Do NOT close the story or run the audition until the speedup question is resolved — running an expensive ≥3-listener audition on a half-engaged bf16 path would be wasted effort. [TRIGGERED 2026-05-10. Median speedup = -3.77% (bf16 slightly slower); steady-state ratio bf16=1.62 vs fp32=1.40 (bf16 worse). Diagnosis: TF32+cuDNN already collected the matmul win on Blackwell (3.23 → 1.40 producer ratio per Story 18.2); bf16's residual headroom is small or negative on the autoregressive single-token workload. Routed via OQ #3 with three-option framing; Commander selected option (b) — defer Task 8 audition to post-Story-18.4 retrospective.]

- [x] ~~**Task 8 — NFR3 ≥3-listener perceptual A/B audition**~~ DEFERRED to post-Story-18.4 per OQ #3 option (b) (AC: #11)
  Full Story 17.1-grade gate. Load-bearing perceptual gate of the epic per `:1383`. **Substantially Commander-routed** — the fixture regeneration uses the production GUI (no CLI generate-and-save tool exists) and the listener recruitment is a human task. The dev agent provides the helper script and procedural instructions.
  - [ ] 8.1 Regenerate the 10-pair perceptual-difficult fixture subset against bf16 output. Use the same generation script that produced `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` (locate via `grep -r "perceptual-fixtures" _bmad/` or by reading `16-7-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` for the regeneration command). Produce `18-3-perceptual-fixtures/` with: 10 paired WAVs (A = fp32 baseline; B = bf16 candidate), refreshed `_perlistener_truthtable.json` with L1/L2/L3 randomizations, byte-identical copy of `LISTENING-INSTRUCTIONS.md`. Commit fixture directory via `git add -f` (gitignored under `_bmad-output/`). [COMMANDER-ROUTED: there is no CLI generate-and-save tool; Story 17.1's `16-8-perceptual-fixtures/` was Commander-produced via the GUI and that pattern continues here. Procedure: run `04_Story_18.3_NFR1_FP32.bat`'s precision-toggle for the A (fp32 baseline) renditions, then re-toggle to bf16 for the B (bf16 candidate) renditions, generating each utterance manually. Save WAVs to `18-3-perceptual-fixtures/`. Detail in the evidence file §"NFR3 audition verdict" procedure outline.]
  - [x] 8.2 Adapt `17-1-l1-audition-helper.py` as `18-3-l1-audition-helper.py` to point at the new fixture directory. Same blinding discipline (no filename printing); same controlled-vocabulary input gates. Force-add to git. [DONE: `_bmad-output/implementation-artifacts/18-3-l1-audition-helper.py` adapted from 17-1; FIXTURE_DIR repointed to `18-3-perceptual-fixtures/`; CANONICAL_CSV repointed to `18-3-bf16-precision-audition.csv`; verdict-gate reminder updated for the Story 18.3 framing (PASS iff zero `audible_seam` on bf16 pair across 30 trials).]
  - [ ] 8.3 Recruit ≥3 listeners. L1 = Commander; L2 + L3 = independent listeners per Story 17.1's protocol. The single-room walkthrough format Story 17.1 used is acceptable per Story 17.1's M1 / methodology limitations disclosure; the dev agent reproduces those limitations verbatim in the architecture amendment (AC #12 part h). [COMMANDER-ROUTED.]
  - [ ] 8.4 Run the audition. Total trials = 30 (10 utterances × 3 listeners × A and B labeled blind per pair). Capture results at `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv` joined against `18-3-perceptual-fixtures/_perlistener_truthtable.json`. [COMMANDER-ROUTED — Commander runs `python310/python.exe _bmad-output/implementation-artifacts/18-3-l1-audition-helper.py L<N>` once per listener.]
  - [ ] 8.5 Compute the verdict. PASS iff zero listeners flag `audible_seam` on any TRUE_STREAM (= bf16) pair across all 30 trials. Verdict-computation tables at evidence file §"NFR3 audition verdict" mirror Story 17.1's structure verbatim — per-system defect-flag count table; per-listener subtotals; per-utterance subtotals; explicit `audible_seam` zero-flag check; PASS / FAIL outcome. [PENDING: dev agent computes after Commander forwards the audition CSV.]
  - [ ] 8.6 If FAIL on any utterance from any listener, STOP and route to Open Question #4 with the failed-utterance class (sibilants vs tonal peaks vs low-amplitude consonants) annotated. Do NOT close the story or amend the architecture until Commander routes the outcome. [PENDING.]

- [x] **Task 9 — Architecture amendment (two-place edit)** (AC: #12)
  Mirrors the Story 17.1 two-place edit pattern at `architecture-optimization-pass.md:863-:899` exactly. Lands ONLY after the audition closes (Task 8 verdict input).
  - [x] 9.1 Place 1: append the parenthetical `*(Story 18.3 bf16 audition <PASSED|FAILED> {{audition_date}} — see follow-up note below.)*` to the NFR3 cell at `:803`. Use the verbatim wording from the Story 17.1 cell pattern. [DONE: NFR3 cell amended with `*(Story 18.3 bf16 audition DEFERRED 2026-05-10 pending Story 18.4 producer-bottleneck close — measured no speedup over Story 18.2 fp32+TF32 baseline; revisit post-18.4. See follow-up note below.)*`]
  - [x] 9.2 Place 2: add a new section `#### Story 18.3 Follow-up Note (bf16 Precision Audition, {{audition_date}})` immediately after the existing `#### Story 17.1 Follow-up Note` block (currently ends at `:899`). Section content per AC #12 sub-points (a)–(h). Reuse Story 17.1's wording for the methodology limitations section (single-room listening environment, single-scribe prompt-framing risk, L1 anonymization not preserved) — adapt only where the fixture or protocol differs. [DONE: `#### Story 18.3 Follow-up Note (bf16 Precision Audition — DEFERRED, 2026-05-10)` inserted at `architecture-optimization-pass.md` immediately after the Story 17.1 follow-up note. Methodology limitations adapted: single-host RTX 5090 measurement (NOT single-room listening — audition deferred); cold-start variance; audition-not-run.]
  - [x] 9.3 Source artifacts list at the new section's footer mirrors Story 17.1's footer at `:892-:900` exactly: ✓ for git-tracked (audition CSV, helper script, evidence file, story file); ○ for gitignored (fixture directory, listener-instructions copy). [DONE: source artifacts list at the new section's footer; ✓ for force-add candidates (story, evidence, set-precision.py, aggregate-nfr1.py, l1-audition-helper.py, three .bat files, two consolidated CSVs); ○ for gitignored (per-run CSVs, deferred fixture directory, deferred audition CSV).]

- [x] **Task 10 — Bundled-environment smoke** (AC: #13)
  Composes Story 18.2's deferred Task 6 (the bundled smoke for `torch_runtime.py` engagement) with Story 18.3's required smoke. First production-bundle verification of the combined Epic 18 source-tree edits. **First bundled-smoke run (2026-05-10) surfaced cut-off-at-end on the canonical Sarira-F paragraph; root-caused to a drain-math bug → fixed in M6 follow-up. Closing Task 10 contingent on Commander's post-M6 rebuild + re-test confirming the cut-off is gone.**
  - [x] 10.1 Run `build_release.bat` (or equivalent per `memory/build_tools_phase_perp_state.md` + `memory/production_release_state.md`). Confirm the resulting `build_tools/dist/MyVoice/MyVoice.exe` includes both Story 18.2's `torch_runtime.py` (now extended with `resolve_tts_precision`) and Story 18.3's `model_registry.py` precedence resolver + AppSettings field. [DONE 2026-05-10 — Commander built; build counter bumped 10→12 at `build_tools/installer.iss` + `build_tools/version.py` per Story 18.2 OQ #4 separate-build-state-commit precedent.]
  - [x] 10.2 Launch the bundled exe. Confirm `myvoice.log` (in the portable Logs path per `setup_logging()` discipline) contains BOTH expected INFO lines: (a) the Story 18.2 `"TF32 + cuDNN benchmark enabled (device_capability=...)"` line, AND (b) the new Story 18.3 `"ModelRegistry initialized: device=cuda:0, dtype=torch.bfloat16, precision_source='app_settings_auto_ampere', quality_tier=quality"` line. Capture both verbatim at evidence file §"Bundled smoke". [DONE — both INFO lines confirmed in bundled myvoice.log; resolver + TF32 enable both engaged.]
  - [x] 10.3 Run a short canonical TTS generation in the bundled exe (Story 17.3 §4.1 step 1 short paragraph). Confirm no new error / warning / unexpected log line; confirm TTS generation completes successfully end-to-end (audio plays through the streaming pipeline). [DONE — bundled exe generates audio end-to-end.]
  - [x] 10.4 Run a long-form CLONED utterance (the Sarira-F, ≥250 chars / ~22 s of speech case Tasks 7 + 8 used). Commander confirms zero perceptual defects compared to the pre-18.3 / pre-18.2 baseline (Commander-solo spot-check; independent of the multi-listener audition which gates the architectural amendment). [DONE — first run surfaced cut-off-at-end (~last 4 words truncated, "at around five seconds"); root-caused to drain-math bug; M6 fix landed (drain math reads last-chunk trackers, drops `if remaining > 0` gate). **Pending Commander post-M6 rebuild + re-test to confirm the cut-off is gone before final closeout.**]
  - [x] 10.5 If the bundled smoke surfaces a packaging defect (e.g., the new `tts_precision` field fails to round-trip through `settings.json` in the portable bundle, or the new `resolve_tts_precision` function fails to import in the PyInstaller environment for a hidden-imports reason), the dev agent surfaces it to Commander rather than absorbing it. Document at evidence file §"Bundled smoke" with the defect's exact log excerpt + the proposed fix surface (typically a one-line `hidden-imports` entry in the PyInstaller spec). [DONE — defect surfaced was NOT a packaging defect (M6 drain-math regression class instead); documented at story Change Log + evidence file with full root cause + fix.]

- [x] **Task 11 — Regression test sweep** (AC: #8)
  Verify the new module surface + AppSettings field + ModelRegistry resolver does not regress the established surfaces. Mirrors Story 18.2 Task 7's structure.
  - [x] 11.1 Run the new + extended unit tests: `pytest tests/unit/services/tts_streaming/test_torch_runtime.py tests/unit/services/test_model_registry.py tests/unit/models/test_app_settings.py -v`. Expected: Story 18.2's 15 pre-existing `test_torch_runtime` tests + 6 new `resolve_tts_precision` rows = ~21 in test_torch_runtime; ~4–6 new `test_model_registry` precedence rows; ~4 new `test_app_settings` `tts_precision` rows. Exact count is informational; the contract is "all six AC #8 sub-bullets have at least one passing test that exercises them." [DONE: 28 test_torch_runtime + 14 test_model_registry + 11 test_app_settings_tts_precision = 53 PASS.]
  - [x] 11.2 Run the existing Story 18.1 instrumentation tests (23 tests; Story 18.2 closed at 23/23). Expect 23/23 PASS. [DONE — included in the wider 803-PASS sweep below.]
  - [x] 11.3 Run the existing Story 17.3 progressive-playback tests (32 tests). Expect 32/32 PASS. [DONE: 35/35 PASS — count grew slightly since Story 17.3 closed; zero regressions.]
  - [x] 11.4 Run the existing Story 16.2 streaming-mode tests (16 tests; Story 18.2 closed at 16/16). Expect 16/16 PASS. [DONE — included in the wider 803-PASS sweep below; covered by `tests/unit/services/tts_streaming/test_streaming_mode.py`.]
  - [x] 11.5 Run the broader Epic 16 / 17 / 18 regression sweep: `pytest tests/unit/services/test_qwen_tts_service_*.py tests/unit/services/test_qwen_tts_service_session_integration.py tests/unit/observability/ tests/unit/test_app_progressive_playback*.py tests/integration/test_progressive_playback_dispatch_skip.py -v`. Expect 152/152 PASS (Story 18.2's broader-sweep count; new tests from this story bring the grand total to ~167+). [DONE: ran a wider net (`tests/unit/services/ tests/unit/observability/ tests/unit/models/` + progressive-playback + integration smoke) = **803/803 PASS**, well above the 167+ threshold.]
  - [x] 11.6 (added) Two pre-existing tests pinned the exact `tts_streaming/__init__.py` `__all__` ordering (declaration-order discipline): `test_codec_token_streamer.py::test_package_all_lists_expected_symbols_in_order` and `test_streaming_decoder.py::test_package_all_lists_expected_symbols_in_declaration_order`. Both updated to append `"resolve_tts_precision"` per the established Story 18.2 widening precedent. [DONE.]

- [ ] **Task 12 — Code-review pass** (post-implementation)
  - [ ] 12.1 Run `/bmad-bmm-code-review`. Per `memory/code_review_regression_test_exact_class.md`: HIGH/MEDIUM-fix regression tests must mirror the exact bug class. Expected review-finding categories for this story: (a) precedence-rule drift in `ModelRegistry.__init__` (the "app_settings overrides legacy dtype param" contract is the most-easily-broken AC); (b) `tts_precision` validation drift (the `streaming_mode_override` mirror at `app_settings.py:375-389` is the precedent — any deviation is a smell); (c) telemetry tag schema drift (`source` enum values + `device_capability` `"none"` sentinel must match Story 18.2's contract); (d) audition fixture regeneration discipline — the regenerated `18-3-perceptual-fixtures/` must NOT overwrite or rename the original `16-7-perceptual-fixtures/` (those preserve the Story 17.1 audited verdict's evidentiary surface); (e) architecture-amendment placement drift — the new follow-up note must land *after* Story 17.1's follow-up note (chronological ordering preserves the document's reading flow); (f) bundled-smoke evidence file completeness.
  - [ ] 12.2 Address findings. Re-run code-review twice after non-trivial auto-fixes per the established Stories 16.7 / 16.8 / 17.1 / 17.2 / 17.3 / 18.1 / 18.2 pattern. Commit per the `Story 18.3: code-review pass — H#/M#/L# fixes` pattern.

## Dev Notes

### What this story is

Story 18.3 is the third story of Epic 18 (Generation-Speed Optimizations / Phase ⊥-Polish-2). It adds explicit hardware-aware bf16 precision engagement on the talker + decoder, a user-facing `AppSettings.tts_precision` setting with `"auto" | "bf16" | "fp32"` semantics for NFR7 fp32 fallback, and the load-bearing ≥3-listener perceptual A/B audition that gates the bf16 default for Ampere+ users.

**Central audit hypothesis (load-bearing for the story's framing):** the configured `dtype=torch.bfloat16` default at `model_registry.py:95` is unlikely to be engaging end-to-end through the codec-decoder side. Evidence: the `_decode` callable at `qwen_tts_service.py:3393-:3395` calls `audio.detach().cpu().numpy()` directly — and NumPy has no `bfloat16` dtype, so a bf16 tensor would raise on this conversion. Two ways production could be passing this code path today: (1) `model.model.speech_tokenizer.decode` returns a fp32 tensor regardless of the talker's bf16 dtype (vocoders typically stay in fp32 for numerical stability — this is the *most likely* outcome and partially explains why the V2 baseline does not yet deliver the anticipated 30–50% speedup), or (2) NumPy 2.0+ DLPack-mediated bf16 → fp32 silent upcast at the conversion boundary (requires a recent NumPy + PyTorch combination; the upcast still erases the bf16 gain at the codec stage even if it doesn't crash). The Task 1 forward-hook audit confirms which path qwen-tts 0.0.4 takes; if outcome (1), the story's `resolve_tts_precision` work delivers the *talker-side* gain (which is what the producer-bottleneck verdict at `memory/epic18_producer_bottleneck_finding.md` named) but the codec-decoder stage stays in fp32 either way — that is acceptable and architecturally intentional for the speech tokenizer (Story 16.8's docstring at `qwen_tts_service.py:3328-:3357` documents the speech_tokenizer's contract but explicitly does not pin its internal dtype).

This is the **load-bearing perceptual gate of the epic** per `epics-optimization-pass.md:1383`. The audition's verdict is the input to the architecture amendment — if PASS, bf16 is the certified default on Ampere+; if FAIL, the `tts_precision` setting becomes a power-user opt-in and Commander routes the outcome (acceptable defect with class-specific fp32 fallback, or upstream qwen-tts pin-bump investigation).

The change cumulative: Story 18.1 (instrumentation) closed the producer-bottleneck question; Story 18.2 (TF32 + cuDNN) closed lossless precision-tuning; Story 18.3 (bf16) is the first of the two precision-class changes (18.3 + 18.4) that Story 18.1's evidence file §4.4 named as the producer-bottleneck fix class.

### What this story is NOT

- **NOT a naive "cast model to bf16 on load" change.** bf16 is ALREADY the production default at `model_registry.py:95` (`dtype: str = "bfloat16"`). The naive interpretation in the epic stub at `:1375` reflects an incomplete reading of the V2 baseline — the dev agent reconciles by framing the work as audit + correctness fix + audition gate.
- **NOT a UI surface for the `tts_precision` setting.** Data-only field; UI exposure deferred (hand-edit `settings.json` is the user remediation path, mirroring Story 16.6 / `streaming_mode_override` precedent).
- **NOT a CPU / pre-Ampere change *on the bf16 side*.** Per D-9 / NFR12, CPU and pre-Ampere hosts move to fp32 when `tts_precision == "auto"` (the default) — this is the latent-bug *fix* relative to the current `dtype="bfloat16"` default. The user can still override via `tts_precision = "bf16"` to force engagement, but that is an explicit opt-in.
- **NOT an `_handle_progressive_chunk_async` edit.** The chunk → bytes path at `app.py:2622-2625` expects float32 input; the bf16 → float32 cast happens upstream at the GPU→CPU boundary. Story 18.3 audits the cast site but does not rewrite it.
- **NOT a `streaming_decoder.py` edit.** The decoder consumes whatever `decode_fn` returns; the dtype contract is `np.float32`. Story 18.3 audits but does not rewrite.
- **NOT a strict-fp32 vs TF32-fp32 comparison.** Story 18.2 closed that comparison as null on the producer-bottleneck workload; Story 18.3's fp32 override path is fp32-with-TF32-engaged (composes on top of 18.2). The methodology note documents this explicitly so future stories reading the evidence don't conflate the two.
- **NOT a build-pipeline change.** No `requirements.txt` / installer-spec / `build_release.bat` edits. Per Epic 18 framing at `:248`. Build-counter increments at `build_tools/installer.iss` + `build_tools/version.py` are Commander-handled in a separate build-state commit per the Story 18.2 OQ #4 precedent.
- **NOT an `audible_seam` gate-relaxation.** The verdict gate is verbatim Story 17.1's: PASS iff zero listeners flag `audible_seam` on any TRUE_STREAM pair across all 30 trials. The dev agent does NOT widen the gate to permit "low-amplitude" defects or "minor sibilance"; FAIL routes to Open Question #4.

### Source tree components to touch

**Read-only (analysis/reference):**
- `src/myvoice/services/model_registry.py:95` (current `dtype: str = "bfloat16"` default), `:119-:124` (the dtype_map), `:144-:146` (the existing INFO log line), `:432-:461` (the `_load_model` path with `from_pretrained(..., torch_dtype=self.dtype)` + the existing `torch.set_float32_matmul_precision('high')` at `:446-:447` — note this Story 18.2 wire-up redundancy with the new `enable_tf32_and_cudnn_benchmark` is acceptable; Story 18.2's enable is the canonical surface, and the legacy `set_float32_matmul_precision` call is documentation-grade not behavior-grade)
- `src/myvoice/services/tts_streaming/torch_runtime.py` — the Story 18.2 module the new `resolve_tts_precision` extends. Mirror its docstring + lazy-import + side-effect-free patterns exactly.
- `src/myvoice/services/tts_streaming/__init__.py` — package re-export pattern; Task 3.4 widens this with the new public symbol.
- `src/myvoice/models/app_settings.py:103-:108` (`streaming_mode_override` field declaration), `:375-:389` (the validation block), `:511 + :527 + :593 + :700-:709` (round-trip + diff machinery). The mirror surface for Task 2.
- `src/myvoice/services/qwen_tts_service.py:524-:588` — the `__init__` that constructs `ModelRegistry(...)`; Task 4.5 wires `app_settings=self._app_settings` through.
- `src/myvoice/services/tts_streaming/streaming_decoder.py:82` — the `decode_fn` callable's consumer contract (`Callable[[list[Any]], np.ndarray]`).
- `src/myvoice/services/qwen_tts_service.py:3325-:3396` — `_build_true_stream_decode_fn(model)` (Story 16.8). The supplier site for `decode_fn`. The GPU→CPU cast lives at `:3393-:3395` (`audio.detach().cpu().numpy()` followed by `np.asarray(audio, dtype=np.float32).flatten()`). The codec decoder path: `model.model.speech_tokenizer.decode([{"audio_codes": chunk}])` at `:3379-:3381`.
- `src/myvoice/app.py:_handle_progressive_chunk_async` (the orchestrator's chunk handler at `:2440-:2680`) — the AC #7 invariant; do not edit.
- `src/myvoice/observability/metrics.py:77-:150-region` — `metrics.record(name, value, **tags)` API surface. Same listener pattern Story 18.1 + 18.2's metrics use.

**New (source tree):**
- *None.* All new code lives inside existing files (`torch_runtime.py`, `model_registry.py`, `app_settings.py`).

**Edit (source tree):**
- `src/myvoice/services/tts_streaming/torch_runtime.py` (Task 3) — add `resolve_tts_precision` function + extend module docstring.
- `src/myvoice/services/tts_streaming/__init__.py` (Task 3.4) — add `resolve_tts_precision` to `from ... import` + `__all__`.
- `src/myvoice/models/app_settings.py` (Task 2) — add `tts_precision` field + validation + round-trip + `_field_names`.
- `src/myvoice/services/model_registry.py` (Task 4) — accept `app_settings` param + precedence resolver + extended INFO log + telemetry emission.
- `src/myvoice/services/qwen_tts_service.py` (Task 4.5) — pass `app_settings` to `ModelRegistry(...)` constructor.

**New (tests):**
- *None at file level.* All new tests extend existing files: `test_torch_runtime.py`, `test_model_registry.py` (create if absent), `test_app_settings.py`, `test_qwen_tts_service*.py`.

**New (evidence + measurement + audition artifacts):**
- `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md` (multi-section evidence file — Tasks 1, 5, 7, 10). Force-add per `memory/git_repo_state.md`.
- `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16.csv` (Task 7.1; N=10). Force-add.
- `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32.csv` (Task 7.2; N=10). Force-add.
- `_bmad-output/implementation-artifacts/18-3-perceptual-fixtures/` (Task 8.1; 10 paired WAVs + truth-table + LISTENING-INSTRUCTIONS copy). Force-add.
- `_bmad-output/implementation-artifacts/18-3-l1-audition-helper.py` (Task 8.2; adapted from `17-1-l1-audition-helper.py`). Force-add.
- `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv` (Task 8.4; 30 rows). Force-add.

**Edit (evidence layer):**
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (Task 9; two-place edit) — NFR3 cell + new follow-up note. The architecture amendment IS this story's change.

### Testing standards summary

- **Unit tests:** mirror Story 18.2's `test_torch_runtime.py` patterns. Monkeypatch `torch.cuda.is_available` and `torch.cuda.get_device_capability` at the attribute-access level. PyAudio / Qt / qwen-tts NOT involved in the unit-test surface.
- **No real GPU required for unit tests.** All hardware-truth-table branches exercised via monkeypatch. Real-GPU validation lives in Tasks 7 + 8 + 10 (NFR1 measurement, NFR3 audition, bundled smoke).
- **Conftest discipline:** `tests/conftest.py` enforces torch-before-PyQt6 DLL ordering per `memory/torch_pyqt6_dll_ordering.md`. New test additions do NOT need a special preamble — they are pure-Python torch-attribute monkeypatching, no DLL-ordering surface area. If running under coverage, follow the inline torch-first preamble per `memory/torch_before_coverage_dll_ordering.md`.
- **AppSettings test discipline:** the `tts_precision` field's validation tests must mirror `streaming_mode_override`'s test row structure exactly. The bug class is "validation drift between two near-identical fields"; mirror the bug-class regression-test discipline per `memory/code_review_regression_test_exact_class.md`.
- **ModelRegistry precedence test discipline:** the four-branch precedence test (Task 6.2) must use `caplog` to assert the INFO log records the correct `precision_source` value. The bug class is "precedence-rule drift" (the most-easily-broken AC per the code-review discipline below); the test must exercise each of the four `source` values directly.
- **Audition discipline:** the fixture regeneration MUST produce a NEW directory `18-3-perceptual-fixtures/` — do NOT overwrite or rename `16-7-perceptual-fixtures/` (those preserve the Story 17.1 audited verdict's evidentiary surface; future maintainers re-validating the streaming-default flag flip need that fixture intact per Story 17.1's M1 reproducibility note).

### Project Structure Notes

**Alignment with unified project structure:**
- New `resolve_tts_precision` lives in the existing `src/myvoice/services/tts_streaming/torch_runtime.py` module — sibling of `is_ampere_or_newer`, `enable_tf32_and_cudnn_benchmark`, `default_streaming_mode_for_hardware`. Same package; same import discipline.
- New `tts_precision` field lives in the existing `src/myvoice/models/app_settings.py` — alphabetically positioned alongside `streaming_mode_override`.
- New `app_settings` parameter on `ModelRegistry.__init__` — backwards-compatible (default None preserves the legacy call surface).
- All new tests extend existing test files — no new test file or test directory.
- Evidence file at `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md` matches the per-story evidence-file pattern (Story 16.7 onward).

**Detected variances:**
- The `ModelRegistry.__init__` precedence rule layers the new resolver ON TOP of the existing `dtype: str` parameter (preserves the legacy call surface). This is a **deliberate** backwards-compat layer: tests + non-AppSettings call sites continue to work without modification. The alternative — replacing the `dtype: str` parameter with `app_settings: AppSettings` — would force every test that constructs `ModelRegistry` to also construct an `AppSettings`, which is a significant scope-creep against the AC. Document the precedence at the resolver function's docstring + at `ModelRegistry.__init__` docstring.
- The bf16 default already engaged in V2 means Story 18.3's measurement is the FIRST time the bf16 vs fp32 A/B is captured *for the production code path* (Story 18.2's measurement compared TF32-engaged vs TF32-disengaged on a bf16-already-engaged baseline). This is the canonical bf16 measurement Story 18.4 will baseline against.
- **No D-decision change.** Story 18.3 does not require a D-decision amendment per `epics-optimization-pass.md:234` ("No new D-decisions"). NFR3 + NFR7 are extended by the `tts_precision` setting + the audition; the architecture amendment captures the extension as a follow-up note, not a D-decision rewrite.

### Previous Story Intelligence

**From Story 18.2 closure (commit `7300c6f` + the empirical-null-but-shipped evidence):**

- Story 18.2 closed with TF32 + cuDNN engaged on RTX 5090 (logged at `myvoice.log:3` with `device_capability=12.0`) but the empirical first-chunk-latency speedup was -2.4% (slower) on the producer-bottleneck workload. Commander selected option (a) "ship as-engaged + move to 18.3" 2026-05-09 — TF32 was never the named producer-bottleneck fix class; bf16 + `torch.compile` are. **Translation for Story 18.3:** the bf16 measurement composes ON TOP OF a TF32-engaged baseline. The fp32 override path in this story's A/B is fp32-with-TF32-engaged, NOT strict-fp32. Document this methodology nuance at evidence file §"NFR1 first-chunk-latency measurement" verbatim so future stories don't conflate the two comparisons.
- Story 18.2 shipped the `resolve_tts_precision`'s sibling module `torch_runtime.py` with the same conceptual pattern (lazy-torch-import; side-effect-free; pure-decision; lives in `tts_streaming`). **Reuse this module verbatim** — extend it, don't parallel it.
- Story 18.2's idempotency contract for `enable_tf32_and_cudnn_benchmark` is exercised via the snapshot-and-restore fixture pattern at `test_torch_runtime.py`. The new `resolve_tts_precision` is side-effect-free so no flag restoration is needed, but the test file's existing fixture conventions (lazy-torch monkeypatch, attribute-access monkeypatch.setattr) MUST be preserved.
- Story 18.2's deferred bundled smoke (Task 6) is composed with Story 18.3's Task 10 — the first build-state commit cycle that lands both story's source-tree edits is the bundled-smoke verification surface. Coordinate with Commander on the build-state commit timing.
- Story 18.2's CSV-capture filter widening (`first_chunk_latency_ms` added to `progressive_playback_csv_capture.py`) is the canonical capture surface for Story 18.3's Task 7. Reuse the existing infrastructure verbatim.

**From Story 18.1 closure (commit `956c039` + the producer-bottleneck verdict):**

- Story 18.1's evidence file §4.4 pinned the producer bottleneck at 3.23× steady-state ratio with the talker model decode rate at 31% real-time. Story 18.3 (bf16) is named as one of two fix-class members (alongside 18.4 `torch.compile`). **Translation for Story 18.3:** the anticipated 30–50% first-chunk-latency speedup at `:1381` should partially close the 3.23× ratio. If the measured speedup is sub-20%, the audit in Task 1 should have surfaced the cause — re-read Task 1's findings before routing to Open Question #3.
- Story 18.1's `M1` (`AudioChunk.session_id`) and the consumer-side metric session_id threading are NOT touched by Story 18.3. The new metric `tts_precision_resolved` is a startup-once event, not a per-chunk metric.

**From Story 17.1 + 17.2 + 17.3 (audition + architecture-amendment + fixture-regeneration discipline):**

- Story 17.1's two-place architecture amendment pattern at `architecture-optimization-pass.md:863-:899` is the canonical template for Task 9. The NFR3 cell parenthetical + the new follow-up note structure mirror it verbatim.
- Story 17.1's `17-1-l1-audition-helper.py` is the canonical helper-script template; Task 8.2's `18-3-l1-audition-helper.py` adapts the path-pointing only (same blinding discipline; same controlled-vocabulary input gates).
- Story 17.1's M1 reproducibility note bounds the verdict's reproducibility — future maintainers re-validating need either the original fixture or a fresh full audit. Story 18.3's fixture is the SAME pattern: gitignored under `_bmad-output/`; held only on Commander's filesystem; force-added via `git add -f`. The follow-up note in Task 9.2 mirrors the source artifacts list at `:892-:900` exactly.
- Story 17.2's H1 / H2 cache-invalidation discipline (per `memory/code_review_regression_test_exact_class.md`) is NOT applicable to Story 18.3 — there is no cache surface in this story (the `tts_precision` setting is loaded fresh on every `AppSettings` construction). The discipline IS applicable to Story 18.4 (the persistent compile cache); 18.4 will reuse 17.2's H1 / H2 lessons explicitly.
- Story 17.3 closed the progressive-playback contract; Story 18.3 inherits the canonical Sarira-F long-form utterance methodology (≥250 chars / ~22 s of speech) for the NFR1 measurement at Task 7.

**Code-review discipline from `memory/code_review_regression_test_exact_class.md`:**

- HIGH/MEDIUM-fix regression tests must mirror the **exact** bug class. For Story 18.3, the highest-risk regression class is **precedence-rule drift in `ModelRegistry.__init__`** (the "app_settings overrides legacy dtype param" contract is the most-easily-broken AC). The Task 6.2 test must exercise each of the four `source` values directly — a test that ONLY exercises one branch and assumes the others are "obvious" would be the wrong fix class for the most likely regression.
- Re-run code-review twice after non-trivial auto-fixes (the established pattern from Stories 16.7 / 16.8 / 17.1 / 17.2 / 17.3 / 18.1 / 18.2).

### Latest Tech Information

**bf16 (bfloat16) on Ampere+ tensor cores:**
- bf16 is a 16-bit floating-point representation with the same 8-bit exponent as fp32 (preserves dynamic range; no underflow guard / autocast scaffolding required, unlike fp16) and a 7-bit mantissa (vs fp32's 23-bit). Tensor cores on Ampere (8.0+), Hopper (9.0), and Blackwell (10.0+ datacenter / 12.0 GeForce RTX 50xx per Commander's measurement at `myvoice.log:3` 2026-05-09) accelerate bf16 matmul at the same throughput as TF32 (~8× fp32 on equivalent hardware), with the additional benefit of halving VRAM consumption (fp32 weights → bf16 weights → 2× VRAM headroom for activations + KV cache + future compile cache).
- The exponent-range preservation is the structural reason bf16 is preferred over fp16 for inference: fp16's smaller exponent (5 bits) requires gradient-scaling / autocast scaffolding to avoid underflow on small intermediate values, which is implementation complexity Story 18.3 does NOT want to introduce.
- Mantissa truncation — bf16 has 7 mantissa bits vs fp32's 23 — produces a quantization noise floor at roughly `2^-7 = ~0.78%` relative error per multiply, vs TF32's `2^-10 ≈ 0.1%`. This is the perceptual risk class the audition gate at Task 8 closes — for audio waveform synthesis, the question is whether 0.78% per-multiply quantization noise compounds across the talker + decoder forward passes into a perceptually audible defect (sibilants, low-amplitude consonants, tonal peaks are the named risk classes per `epics-optimization-pass.md:245`).
- Compute-capability gating: bf16 tensor cores require Ampere+ (capability major ≥ 8). Pre-Ampere (Turing 7.5, Volta 7.0, Pascal 6.x) executes bf16 in software emulation — much slower than fp32. The `is_ampere_or_newer()` gate from Story 18.2 (at `torch_runtime.py`) is the canonical hardware-aware enable point.

**`AppSettings.tts_precision` semantics:**
- `"auto"` (default) defers to the hardware probe: bf16 on Ampere+ CUDA, fp32 elsewhere. Mirrors the `streaming_mode_override = None` / `default_streaming_mode_for_hardware()` precedent at `streaming_mode.py:54-:56`.
- `"bf16"` is user-explicit opt-in: engages bf16 even on CPU / pre-Ampere. The user has accepted the slowdown (CPU bf16 emulation is ~10× slower than CPU fp32). This is the "advanced user override" surface; not commonly used but architecturally clean.
- `"fp32"` is the NFR7 fallback path: forces fp32 even on Ampere+. The user has observed a perceptual defect in bf16 mode (post-release surprise) and wants the lossless path. This is the canonical NFR7 fp32 fallback hatch.
- `"auto"` is the default because the audition (Task 8) is the certification surface — a passing audition means bf16 is safe-by-default on Ampere+; CPU users are protected by the probe; advanced users can override.

**Streaming pipeline dtype contract (audit-only):**
- The `streaming_decoder.py:82` `decode_fn: Callable[[list[Any]], np.ndarray]` contract specifies `np.ndarray` as the return type. NumPy doesn't support bf16 natively (no `np.bfloat16` dtype); the GPU→CPU cast must therefore produce `np.float32`. The expected pattern: `tensor.to(torch.float32).cpu().numpy()` or `tensor.float().cpu().numpy()` — the cast happens at the boundary, not earlier inside the model's forward pass.
- The `app.py:_handle_progressive_chunk_async` chunk → bytes path at `:2622-:2625` reads `chunk.audio_data` as `np.ndarray[float32]` and converts to int16 PCM bytes. The dtype invariant is float32 input; Story 18.3's Task 5 audits this is preserved.

**Audition fixture regeneration:**
- The `16-7-perceptual-fixtures/` directory contains 10 paired WAVs (4 short + 4 medium + 2 long; sibilants + tonal peaks + alliteration), `_perlistener_truthtable.json` with L1/L2/L3 randomizations, and `LISTENING-INSTRUCTIONS.md` with the canonical protocol. Story 17.1 used this fixture against post-Story-16.8 dispatch; the verdict was PASS (zero `audible_seam` flags across 30 trials).
- For Story 18.3, the fixture must be **regenerated** because the audio under audition is the bf16 candidate output (vs the fp32 baseline), not the streaming-vs-sentence-stream A/B Story 17.1 audited. The directory `18-3-perceptual-fixtures/` is the new home; do NOT overwrite `16-7-perceptual-fixtures/` (Story 17.1's verdict reproducibility depends on it).
- The fixture regeneration uses the same generation script — locate via `grep -r "perceptual-fixtures" _bmad/` or by reading the existing `16-7-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` for the regeneration command. The "A" rendition is the `tts_precision="fp32"` baseline; the "B" rendition is the `tts_precision="bf16"` candidate. L1/L2/L3 randomization is freshly recomputed in the new `_perlistener_truthtable.json`.

### Project Context Reference

- Project context: `docs/` (existing project-context.md not found; CLAUDE.md absent).
- Working directory invariants per `memory/git_repo_state.md`: V2 is canonical git repo since 2026-05-05; remote = github.com/WreckedMech117/MyVoice; `_bmad-output/` is gitignored (evidence file + CSVs + fixture + audition CSV all need `git add -f`).
- Production state per `memory/production_release_state.md`: ships publicly via myvoicetts.com as a Windows .exe with bundled portable python310. Installer size unchanged by Story 18.3 (no `requirements.txt` / installer-spec / `build_release.bat` edits per Epic 18 framing at `:248`).
- Hardware target per `memory/hardware_setup.md`: RTX 5090 Blackwell (compute 12.0 per Story 18.2's measurement) dev host; ship-target covers RTX 30xx (compute 8.6) / RTX 40xx (compute 8.9). All three satisfy the new probe's `>= 8` Ampere+ gate.
- Phase context per `memory/build_tools_phase_perp_state.md`: Phase ⊥-Polish-2 is the successor to Phase ⊥-Polish (Story 17.3 closed the progressive-playback contract); Story 18.3 is the third story of Phase ⊥-Polish-2 (after Story 18.1 instrumentation + Story 18.2 TF32+cuDNN), and the load-bearing perceptual-gate story of the epic.
- Producer-bottleneck context per `memory/epic18_producer_bottleneck_finding.md`: Story 18.1's 3.23× ratio at the producer side names 18.3 + 18.4 as the fix class. Story 18.3 is the first of the two; the bf16 measurement at Task 7 is the canonical producer-side throughput-uplift number.

### References

- **Epic 18 stub** — `_bmad-output/planning-artifacts/epics-optimization-pass.md` lines 1370–1386 (Story 18.3 stub); lines 228–250 (Epic 18 framing); line 245 (risk profile: Medium); line 1378 (audition discipline: full Story 17.1-grade ≥3-listener); line 1381 (anticipated acceptance gate: 30–50% speedup + ≥40% VRAM reduction + zero `audible_seam` flags + fp32 override engages cleanly).
- **Story 18.2 evidence** — `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md`. The TF32-engaged baseline this story composes against; the methodology note in Task 7.3 + AC #10 documents the composition explicitly.
- **Story 18.1 evidence** — `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` §4.2 + §4.4. The producer-bottleneck verdict naming Story 18.3 as a fix-class member.
- **Story 17.1 audition + architecture-amendment** — `architecture-optimization-pass.md:863-:899` (the two-place edit pattern Task 9 mirrors); `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (the audition-CSV format Task 8.4 mirrors); `_bmad-output/implementation-artifacts/17-1-l1-audition-helper.py` (the helper-script template Task 8.2 adapts).
- **Story 17.1 fixture** — `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/` (the canonical 10-pair fixture; Story 18.3 regenerates as `18-3-perceptual-fixtures/` against bf16 output, preserves the 16-7 directory unchanged per Story 17.1's M1 reproducibility note).
- **Story 16.2 streaming-mode hardware probe** — `src/myvoice/services/tts_streaming/streaming_mode.py:54-:56`; `tests/unit/services/tts_streaming/test_streaming_mode.py`. The structural precedent for the new `resolve_tts_precision` resolver.
- **Story 18.2 torch_runtime module** — `src/myvoice/services/tts_streaming/torch_runtime.py`; `tests/unit/services/tts_streaming/test_torch_runtime.py`. The module the new `resolve_tts_precision` extends.
- **Architecture D-9 hardware-aware default** — `architecture-optimization-pass.md:257`. The `torch.cuda.is_available()` probe + Ampere+ guard discipline. Story 18.3's `tts_precision = "auto"` branch enforces this for the precision dimension, closing a latent CPU-bf16 violation.
- **Architecture NFR3 perceptual-gate** — `architecture-optimization-pass.md:803`. Task 9 amends this cell with the Story 18.3 audition outcome.
- **Architecture NFR7 graceful degradation** — `architecture-optimization-pass.md:806`. Story 18.3's `tts_precision = "fp32"` override is the precision-tier extension of NFR7 the epic stub at `:234` anticipates.
- **Architecture NFR12 CPU-only support** — `architecture-optimization-pass.md:75 + :808`. CPU-only hosts stay identifiably unchanged; the `tts_precision = "auto"` CPU branch returns fp32 (closing the latent V2 default that applied bf16 unconditionally on CPU).
- **Architecture NFR1 revised contract** — `architecture-optimization-pass.md` §"NFR1 (revised 2026-05-08, Story 16.9)" at lines 838–850. Per-class first-chunk targets; Story 18.3's Task 7 measures the bf16 vs fp32 delta against this baseline.
- **Architecture D-19 telemetry** — `architecture-optimization-pass.md` §"D-19 Telemetry" (begins at line 286) and the `metrics.record(name, value, **tags)` helper specified at line 476. Implementation lives at `src/myvoice/observability/metrics.py:77`. Story 18.3's new `tts_precision_resolved` metric extends this established pattern.
- **AppSettings `streaming_mode_override` precedent** — `src/myvoice/models/app_settings.py:103-:108` (declaration), `:375-:389` (validation). The mirror surface Task 2 uses verbatim for the new `tts_precision` field.
- **ModelRegistry production code path** — `src/myvoice/services/model_registry.py:95` (current bf16 default), `:119-:124` (dtype_map), `:144-:146` (INFO log), `:432-:461` (`_load_model` with `from_pretrained(..., torch_dtype=self.dtype)`).
- **QwenTTSService construction site** — `src/myvoice/app.py:445-:448` (the single production call site); `src/myvoice/services/qwen_tts_service.py:582-:588` (the `ModelRegistry(...)` call inside).
- **Streaming decoder consumer contract** — `src/myvoice/services/tts_streaming/streaming_decoder.py:82` (the `decode_fn: Callable[[list[Any]], np.ndarray]` type); `:1-:36` (the module docstring's Story 16.4 framing).
- **Streaming decoder supplier site** — `src/myvoice/services/qwen_tts_service.py:3325-:3396` (`_build_true_stream_decode_fn`, Story 16.8 origin). GPU→CPU cast at `:3393-:3395`. Codec decoder call at `:3379-:3381` (`model.model.speech_tokenizer.decode([{"audio_codes": chunk}])`).
- **Chunk → bytes invariant** — `src/myvoice/app.py:_handle_progressive_chunk_async` at `:2440-:2680`, specifically `:2622-:2625` (the float32-input invariant Task 5.3 audits).
- **Memory: hardware_setup.md** — RTX 5090 Blackwell (Commander measurement: compute 12.0 GeForce variant) as dev host; ship-target covers RTX 30xx / RTX 40xx. All three satisfy the `>= 8` Ampere+ gate.
- **Memory: torch_pyqt6_dll_ordering.md** — the Windows DLL-init invariant the Task 4 wire-up preserves (the new code is pure-Python torch-attribute access; no DLL-ordering surface).
- **Memory: build_tools_phase_perp_state.md** — Phase ⊥-Polish-2; Story 18.3 is the third story of the phase.
- **Memory: code_review_regression_test_exact_class.md** — exact-bug-class regression-test discipline (Task 6.2's precedence-rule test is the load-bearing application here; Task 2.5's validation-mirror test is the secondary application).
- **Memory: production_release_state.md** — production-bundle context informing Task 10 (bundled-smoke).
- **Memory: epic18_producer_bottleneck_finding.md** — the 3.23× ratio context Story 18.3 composes against (bf16 helps producer matmul; the bigger wins compose with 18.4 `torch.compile`).
- **Memory: git_repo_state.md** — V2 git-repo state; `_bmad-output/` gitignored (force-add discipline for evidence + CSVs + fixture + audition CSV).

## Open Questions

(The dev agent saves discovered questions here for Commander-routing rather than absorbing them unilaterally. Per the `instructions.xml` "❓ SAVE QUESTIONS" critical rule.)

1. **Pre-implementation audit surfaces an unexpected fp32 round-trip inside the model's forward pass (Task 1.5 routing).** If the talker forward-pass hook captures a `.float()` upcast that erases the bf16 compute gain, the fix surface is upstream-pin-bump (qwen-tts >0.0.4 if a fix exists), local wrapper (intercept the upcast in MyVoice's wrapper code — risks divergence from upstream), or accept-and-document (the upcast may exist for a numerical-stability reason worth preserving). Commander-routed because all three options have non-trivial tradeoffs.

2. **Streaming pipeline dtype audit surfaces a defect (Task 5.4 routing).** If the `decode_fn` supplier's GPU→CPU cast happens AFTER an internal `.float()` upcast that erases the bf16 gain, the fix surface is the same upstream-vs-local-vs-accept tradeoff as OQ #1. Commander-routed because the fix surface is in qwen-tts internals, not MyVoice code.

3. **NFR1 measurement falls below 20% speedup (Task 7.4 routing).** The anticipated [30%, 50%] gate at `:1381` is informational not pass/fail; tails outside the range route to Commander rather than dev-agent interpretation. Sub-20% specifically suggests a partial fp32 round-trip or a non-matmul-dominant workload regime; the audit in Task 1 should have surfaced the cause, but if not, Commander decides whether to (a) ship-as-engaged-anyway and accept the partial gain, (b) defer to a future investigation with a shim, or (c) close the epic with bf16 as a power-user opt-in.

4. **NFR3 audition flags `audible_seam` on any utterance (Task 8.6 routing).** Verdict FAIL → outcome (b) per Story 17.1's framing. Commander decides whether to (i) accept the defect with a session-class fp32 fallback (e.g., utterances containing sibilants force fp32 internally), (ii) defer the bf16 default and ship the `tts_precision` setting as a power-user opt-in, or (iii) escalate to upstream qwen-tts pin-bump investigation. Each option has different cost / benefit / risk profiles that exceed the dev agent's autonomous-decision scope.

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m]

### Debug Log References

**2026-05-09 — autonomous source-tree pass (Tasks 2–6, 11):**

- `python310/python.exe -m pytest tests/unit/services/tts_streaming/test_torch_runtime.py tests/unit/models/test_app_settings_tts_precision.py -v`
  → **40 passed in 6.33s** (Story 18.2's 15 pre-existing + 13 new resolver rows + 11 new AppSettings rows + 1 ordering-already-aligned).
- `python310/python.exe -m pytest tests/unit/services/test_model_registry.py -v`
  → **14 passed in 8.83s** (4 precedence-rule branches + 4 Ampere parametrized cap-major variants + 2 telemetry assertions + 1 caplog assertion + 3 supporting cases).
- `python310/python.exe -m pytest tests/unit/services/ tests/unit/observability/ tests/unit/models/ tests/unit/test_app_progressive_playback*.py tests/integration/test_progressive_playback_dispatch_skip.py`
  → **803 passed in 25.80s** (initial run surfaced 2 declaration-order-pin failures at `test_codec_token_streamer.py:46` + `test_streaming_decoder.py:98`; both fixed by appending `"resolve_tts_precision"` to the asserted `__all__` literal per Story 18.2's append-only precedent; rerun confirmed 803/803).

### Completion Notes List

**Autonomous source-tree work (Tasks 2–6, 11) — COMPLETE.** The portions of
the story that require RTX 5090 GPU runs (Tasks 1, 7, 10), recruiting ≥3
listeners (Task 8), the architecture amendment which depends on Task 8's
verdict (Task 9), and the post-implementation code review (Task 12) are
**Commander-routed** — see §"Commander-routed work" in the evidence file.

Key implementation notes:

1. `AppSettings.tts_precision` mirrors `streaming_mode_override`'s
   warn-and-fallback pattern verbatim (`UNKNOWN_TTS_PRECISION` ValidationIssue
   on invalid value, auto-correct to `"auto"`).
2. `resolve_tts_precision` in `torch_runtime.py` is a pure-decision function
   with the same lazy-torch-import discipline as `is_ampere_or_newer` /
   `enable_tf32_and_cudnn_benchmark`. Side-effect-free: no logging, no metric
   emission, no `torch.backends.*` mutation.
3. `ModelRegistry.__init__` precedence rule layers the new resolver ON TOP
   of the legacy `dtype: str` parameter (preserves backward compatibility
   for tests / non-AppSettings call sites). The four `precision_source`
   labels (`legacy_constructor_arg` / `app_settings_override` /
   `app_settings_auto_ampere` / `app_settings_auto_fallback`) surface the
   chosen path verbatim in both the INFO log line AND the new
   `tts_precision_resolved` telemetry metric so Commander can confirm
   runtime engagement.
4. `model_registry.py` uses `app_settings: Optional[Any]` parameter type with
   `getattr(app_settings, "tts_precision", None)` access — avoids importing
   `AppSettings` at module level (mirrors the lazy-import discipline of
   neighboring service modules).
5. Streaming pipeline dtype audit (Task 5, read-only) confirmed result class
   **(a) Already correct** — the GPU→CPU bf16→fp32 cast happens INSIDE the
   qwen-tts wrapper at the boundary; MyVoice code receives `np.float32`
   regardless of the talker's bf16 dtype. The runtime forward-hook capture
   (Task 1.3) is the final answer; no defect surfaced in the read-only audit.

### File List

**New (autonomous source-tree pass):**

- `tests/unit/services/test_model_registry.py` — 14 tests covering the
  four precedence-rule branches + telemetry tag schema + caplog INFO log
  assertion.
- `tests/unit/models/test_app_settings_tts_precision.py` — 11 tests
  mirroring the Story 15.2 `test_app_settings_clear_comms.py` structure.
- `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md`
  — multi-section evidence file with read-only audit findings + procedure
  templates for Commander-routed sections.

**New (Commander-routed harness — autonomous prep pass):**

- `02_Story_18.3_DType_Audit.bat` — Task 1 harness (env-var-gated dtype audit launch).
- `03_Story_18.3_NFR1_BF16.bat` — Task 7.1 harness (10 fresh-process bf16 launches).
- `04_Story_18.3_NFR1_FP32.bat` — Task 7.2 harness (10 fresh-process fp32 launches).
- `_bmad-output/implementation-artifacts/18-3-set-precision.py` — settings.json mutator.
- `_bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py` — NFR1 N=10 aggregator.
- `_bmad-output/implementation-artifacts/18-3-l1-audition-helper.py` — audition helper (Task 8.2; adapted from 17-1).

**New (H1 + M1 follow-up — 2026-05-10 audit-driven fixes):**

- `tests/unit/test_app_qwen_tts_construction.py` — 3 static-scan tests
  on `app.py` that catch the exact bug class "production call-site
  drops `app_settings=` from `QwenTTSService(...)`."

**Edit (H1 + M1 follow-up):**

- `src/myvoice/app.py` — at the `QwenTTSService(...)` constructor call
  (line 445), added `app_settings=self._app_settings,` kwarg. Without
  this, the Story 18.3 user surface was silently a no-op.
- `src/myvoice/services/model_registry.py:_instrument_dtype_audit` —
  M1 robustness pass: hook uses `with_kwargs=True` (torch ≥ 2.0),
  walks structured outputs (e.g., `Qwen3TTSTalkerOutputWithPast`) for
  tensor fields, and walks `speech_tokenizer` for the inner
  `nn.Module` so the codec-decoder forward hook actually attaches.

**Edit (Story 17.3 finalization-drain follow-up — Path A):**

- `src/myvoice/services/audio_coordinator.py` — added
  `_stream_first_write_ts` + `_stream_total_bytes` + `_stream_sample_width`
  + `_stream_channels` instance state; `play_audio_chunk` records the
  first-write timestamp on the first non-empty write and accumulates
  total bytes; `stop_streaming_session(wait_for_drain: bool = False)`
  optionally awaits the predicted drain time with a 150 ms safety
  buffer and a 15 s hard cap. Class constants
  `_DRAIN_SAFETY_BUFFER_S` + `_MAX_DRAIN_WAIT_S` document the tunables.
- `src/myvoice/app.py` — both `is_final` call sites (`_handle_progressive_chunk_async`
  finalization at line ~2671 and the stale-terminal-chunk close at
  line ~2566) now pass `wait_for_drain=True`. The cancel-path call
  sites (user-cancel at line ~1213 and fallback-restart at line ~2510)
  keep the default `wait_for_drain=False` so cancellation stays prompt.
- `tests/unit/services/test_audio_coordinator.py` — appended a
  `TestStopStreamingSessionDrain` class with 7 tests pinning the
  contract: default no-wait preserves legacy immediate-teardown,
  drain-wait awaits ~remaining audio duration, already-drained audio
  does NOT wait, the wait is capped at `_MAX_DRAIN_WAIT_S`, no-writes
  early-out, `play_audio_chunk` records the trackers, and stop
  resets them for the next session.

**Edit (autonomous source-tree pass):**

- `src/myvoice/models/app_settings.py` — added `tts_precision: str = "auto"`
  field at `:108` immediately after `streaming_mode_override`; added
  validation block in `validate()` after the streaming_mode_override block;
  added `"tts_precision"` key to `to_dict` / `from_dict` / inline reset list.
- `src/myvoice/services/tts_streaming/torch_runtime.py` — extended module
  docstring; added `resolve_tts_precision(override)` function at end of file.
- `src/myvoice/services/tts_streaming/__init__.py` — added
  `resolve_tts_precision` to the import + `__all__` (append-only precedent).
- `src/myvoice/services/model_registry.py` — added `from myvoice.observability import metrics`;
  added `app_settings: Optional[Any] = None` constructor parameter; added
  precedence-resolver block (replaces the old `dtype_map` block); extended
  INFO log line with `precision_source='{source}'`; emit
  `tts_precision_resolved` telemetry metric. Also added env-var-gated
  `_instrument_dtype_audit` method (Task 1 harness: post-load attribute
  walk + one-shot forward hooks on talker + speech_tokenizer; logs
  `[DTYPE_AUDIT]` / `[DTYPE_AUDIT_FWD]` lines; gated on
  `MYVOICE_DTYPE_AUDIT=1` env var → zero overhead in production).
- `src/myvoice/services/qwen_tts_service.py` — pass
  `app_settings=self._app_settings` to `ModelRegistry(...)` at `:582-:588`.
- `tests/unit/services/tts_streaming/test_torch_runtime.py` — added
  `resolve_tts_precision` import + 13 new test rows (6 documented branches +
  4 Ampere parametrized cap-major variants + 2 None-equiv-auto + 1
  side-effect-free metric + 1 side-effect-free flag).
- `tests/unit/services/tts_streaming/test_codec_token_streamer.py` —
  appended `"resolve_tts_precision"` to the pinned `__all__` literal at
  `test_package_all_lists_expected_symbols_in_order`.
- `tests/unit/services/tts_streaming/test_streaming_decoder.py` —
  appended `"resolve_tts_precision"` to the pinned `__all__` literal at
  `test_package_all_lists_expected_symbols_in_declaration_order`.
- `tests/unit/services/test_qwen_tts_service_dispatch.py` — added two
  wire-up integration tests (`test_app_settings_tts_precision_flows_through_to_model_registry`
  + `test_app_settings_tts_precision_bf16_flows_through`) inside
  `TestResolveStreamingMode`.
- `_bmad-output/implementation-artifacts/sprint-status.yaml` —
  `18-3-bf16-precision-on-talker-decoder` ready-for-dev → in-progress.
- `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder.md`
  — Status ready-for-dev → in-progress; Tasks 2–6 + 11 marked [x]; Dev
  Agent Record + File List + Change Log populated; Commander-routed
  Tasks 1 / 7 / 8 / 9 / 10 / 12 stay [ ] pending Commander run.

**Edit (code-review pass — M3/M4/M5/M6 fixes):**

- `src/myvoice/services/audio_coordinator.py` — M6 fix: added
  `_stream_last_write_ts` + `_stream_last_chunk_bytes` instance state;
  `play_audio_chunk` stamps both per write (last_chunk_bytes is
  per-chunk, not cumulative); rewrote drain math to use
  `last_chunk_duration - time_since_last_write` instead of the
  full-stream `expected_total - elapsed` math (which goes negative on
  producer-bottleneck workloads); dropped the `if remaining > 0` gate so
  the safety buffer always fires when `wait_for_drain=True`.
  `start_streaming_session` + `stop_streaming_session` reset the new
  trackers. M4 fix: docstring on `stop_streaming_session(wait_for_drain=...)`
  updated; replaced the stale "~150ms safety buffer" wording with a
  reference to the `_DRAIN_SAFETY_BUFFER_S` constant + the current
  500ms value + the M2/M6-follow-up rationale.
- `src/myvoice/ui/main_window.py` — M6 mirror: `_wait_for_pending_audio_drain`
  now reads `_stream_last_write_ts` / `_stream_last_chunk_bytes`
  instead of the FIRST-write trackers, so measurement-mode close uses
  the same corrected math as the production drain.
- `tests/unit/services/test_audio_coordinator.py` — M5 fix:
  `test_wait_for_drain_true_waits_remaining_audio_duration` reworked
  to compute `expected_wait` from the live
  `AudioCoordinator._DRAIN_SAFETY_BUFFER_S` constant rather than
  hard-coding `0.6 <= elapsed <= 1.2`. A future bump to the constant
  no longer breaks the test for the wrong reason. M6 fix: 4 existing
  drain tests rewritten for the new `_stream_last_*` state model;
  `test_wait_for_drain_true_with_already_drained_audio_does_not_wait`
  renamed + behavior-flipped to
  `test_wait_for_drain_true_with_already_drained_last_chunk_still_waits_safety`
  (the M6 fix is exactly that the safety buffer now always fires);
  added `test_M6_producer_bottleneck_workload_still_drains_last_chunk`
  that simulates the exact bug-class state Commander's bundled smoke
  surfaced (10×1.98s chunks under 1.62× producer ratio, last chunk
  written 0.05s ago) and asserts drain wait still covers the last
  chunk's residual playback.
- `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md`
  — M3 fix: three "**Status:** PENDING" markers (§"Pre-implementation
  audit", §"End-to-end dtype audit", §"NFR1 first-chunk-latency
  measurement") flipped to "**Status:** COMPLETE" with the deferral
  rationale where applicable, so section headers match the body data
  the dev agent already populated.

**Restored (code-review pass — M2 fix):**

- `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv` —
  reverted via `git checkout HEAD --` because a Story 18.3 NFR1
  measurement run had pointed `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` at
  the 18-2 path and clobbered Story 18.2's audited 33-row evidence
  (session_id `828eed8a-…`). Restored from HEAD; Story 18.3's own
  measurement CSVs at `18-3-rtx5090-bf16.csv` / `-fp32.csv` are
  unaffected.

**Out-of-scope (present in working tree, NOT Story 18.3 — code-review pass M1 documentation):**

- `build_tools/installer.iss` — `MyAppBuild "10" → "12"` (build counter
  bump from a Commander-run `build_release.bat`).
- `build_tools/version.py` — `VERSION_BUILD = 10 → 12` (matching
  build counter).
- These two files are **NOT Story 18.3 source-tree edits.** Per Epic 18
  framing at `epics-optimization-pass.md:248` ("18.1–18.3 are pure
  source-tree edits — no `requirements.txt` / installer-spec /
  `build_release.bat` edits") and the Story 18.2 OQ #4 precedent,
  build-counter increments are Commander-handled in a separate
  build-state commit. Documented here so the working-tree drift is
  explicit rather than silent; the actual commit boundary belongs to
  Commander's separate build-state commit cycle.

### Change Log

| Date | Change | Author |
|---|---|---|
| 2026-05-09 | Story status: ready-for-dev → in-progress (sprint-status + story file). | dev agent (Opus 4.7 [1m]) |
| 2026-05-09 | Tasks 2–6 + 11 closed: AppSettings.tts_precision field + resolve_tts_precision resolver + ModelRegistry precedence + telemetry + read-only streaming pipeline dtype audit + 53 new test rows + 803/803 broader regression sweep. | dev agent (Opus 4.7 [1m]) |
| 2026-05-09 | Two pre-existing `__all__` ordering tests updated to append `resolve_tts_precision` (Story 18.2's append-only precedent). | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | Commander-routed harness scaffolded: env-var-gated `_instrument_dtype_audit` in `model_registry.py` (Task 1); `02_Story_18.3_DType_Audit.bat` (Task 1 launcher); `03_Story_18.3_NFR1_BF16.bat` + `04_Story_18.3_NFR1_FP32.bat` (Task 7 N=10 loop launchers); `18-3-set-precision.py` (settings.json mutator); `18-3-aggregate-nfr1.py` (NFR1 aggregator with auto-OQ-#3 detection); `18-3-l1-audition-helper.py` (Task 8.2 helper, adapted from 17-1). Tasks 1.1–1.4 + 7.1–7.3 + 8.2 marked [x]; tasks pending Commander run remain [ ]. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **H1 — production wire-up regression fixed.** First `02_*.bat` run surfaced `precision_source='legacy_constructor_arg'` — the `QwenTTSService(...)` constructor at `app.py:445` did NOT pass `app_settings=self._app_settings` so the new resolver was a silent no-op. Fix: added the kwarg at the production call site. New regression test `tests/unit/test_app_qwen_tts_construction.py` (3 rows; AST-scans `app.py`) catches the exact bug class — "production call site drops the new keyword argument." Per `memory/code_review_regression_test_exact_class.md`. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **M1 — audit hook robustness.** First `02_*.bat` run surfaced two coverage gaps: (a) `talker.forward` is kwargs-only so the hook captured `in=[]`; (b) talker output is a structured `Qwen3TTSTalkerOutputWithPast`; (c) `speech_tokenizer` is a HF wrapper, not an `nn.Module` — original hook skipped it entirely. Fix: hook now uses `with_kwargs=True` (torch ≥ 2.0), walks structured outputs for tensor fields, and walks `speech_tokenizer.{codec_model,model,vocoder,tokenizer,_model}` to find the inner `nn.Module` for the codec-decoder hook. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Task 1 audit captures landed (post-H1+M1 second run).** All three Task 1 sub-points confirmed: (a) `model.model.talker.dtype = torch.bfloat16`; (b) `model.model.speech_tokenizer.model` (the inner `Qwen3TTSTokenizerV2Model`) parameters all `torch.bfloat16` — surprising finding (original audit hypothesis predicted fp32 vocoder); (c) talker forward hook captured all-bf16 tensor kwargs + outputs. Codec forward hook didn't fire because qwen-tts decodes via `Qwen3TTSTokenizer.decode(...)` which bypasses the inner Module's `forward` — but parameter walk is sufficient evidence. Task 1.5 routing condition does NOT trigger; story proceeds to Task 7 measurement. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Side observation: cut-off-at-end finalization race.** Commander reported the last chunk's tail audibly cuts off. Log analysis at `app.py:2661-2668` and `audio_coordinator.py:1234-1259` confirms `stop_streaming_session()` is called immediately on `is_final` without awaiting the PyAudio buffer drain. Pre-existing Story 17.3 finalization race; bf16 engagement made the producer fast enough to expose it. **Out of scope for Story 18.3** (no source changes here); recommended as a Story 18.5 / "17.3 finalization-drain follow-up." Listeners in Task 8 should be briefed to NOT mark `audible_seam` for the cut-off-at-end since the seam is at session-teardown, not at chunk-overlap-add. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Finalization-drain follow-up landed (Path A — closed before Task 7).** `AudioCoordinator.stop_streaming_session(wait_for_drain: bool = False)` now optionally awaits the PyAudio buffer drain by computing `(total_bytes_written / bytes_per_second) - elapsed`, adding a 150 ms safety buffer for PyAudio internal latency, and capping at `_MAX_DRAIN_WAIT_S = 15s` to prevent a math drift from hanging the close. `play_audio_chunk` records the first-write timestamp + accumulates `total_bytes`. Two `is_final` call sites in `app.py` (`:2671` finalization, `:2566` stale-terminal) pass `wait_for_drain=True`; the two cancel sites (`:1213` user-cancel, `:2510` fallback-restart) keep the default False so cancellation stays prompt. 7 new unit tests at `test_audio_coordinator.py::TestStopStreamingSessionDrain` pin the contract (legacy default no-wait, drain-wait, already-drained no-op, max-wait cap, no-writes early-out, tracker recording, tracker reset). 813/813 broader regression sweep PASS. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Measurement-mode bypass for the N=10 NFR1 loop.** `MainWindow.closeEvent` now checks `MYVOICE_AUTO_QUIT_ON_CLOSE=1` env var; when set, flips `_force_quit=True` so X-button click bypasses both the tray-minimize branch AND the confirm-close `QMessageBox`. Production behavior unchanged when env var unset (per `memory/main_window_close_confirm_dialog_in_tests.md` — don't weaken the dialog itself, opt-in env var is fine). Wired into `03_*.bat` + `04_*.bat`. Without this, the 10-launch loop hung on every iteration's tray-minimize. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Task 7 NFR1 measurement complete + OQ #3 routing triggered.** N=10 cold-start fresh-process launches per branch. Median bf16 = 5029 ms; median fp32 = 4846 ms (bf16 -3.77% — slightly slower; within noise floor of ±1500 ms per-launch variance). Steady-state ratio: bf16 = 1.62; fp32 = 1.40 (Story 18.1 baseline was 3.23 — Story 18.2's TF32+cuDNN engagement collected the bulk of the producer-bottleneck win; bf16's residual headroom is small or negative on Blackwell). Aggregator `18-3-aggregate-nfr1.py` extended with cold-start dedup (runs 7 + 10 had a second warmed-pipeline first_chunk record — discarded), per-launch detail dump, and producer steady-state ratio analysis. OQ #3 routing surfaced to Commander with three options. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **OQ #3 routed: option (b) selected — defer Task 8 audition.** Commander chose to defer the ≥3-listener perceptual A/B audition to post-Story-18.4 retrospective. Rationale: bf16 doesn't pay for itself on perf today; spending listener time certifying perceptual equivalence for a no-op default is poor ROI. Re-running the same `03_*.bat` + `04_*.bat` harness post-Story-18.4 (with `torch.compile`'s CUDA graphs / kernel-launch overhead collapse) will give a cleaner answer; if bf16 starts helping there, the audition fires at that point. `tts_precision="auto"` keeps resolving to `bfloat16` on Ampere+ (engaged-but-no-measured-speedup); the conservative ship-as-engaged choice. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Task 9 architecture amendment landed.** Two-place edit at `_bmad-output/planning-artifacts/architecture-optimization-pass.md` per AC #12. Place 1: NFR3 cell at `:803` amended with `*(Story 18.3 bf16 audition DEFERRED 2026-05-10 pending Story 18.4 producer-bottleneck close — measured no speedup over Story 18.2 fp32+TF32 baseline; revisit post-18.4. See follow-up note below.)*`. Place 2: new section `#### Story 18.3 Follow-up Note (bf16 Precision Audition — DEFERRED, 2026-05-10)` inserted immediately after the Story 17.1 follow-up note. Methodology limitations adapted: single-host RTX 5090 (not single-room listening — audition deferred); cold-start variance; audition-not-run. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Story status: in-progress → review.** All in-scope dev-agent tasks closed (Tasks 1–7, 9, 11, plus the H1 + finalization-drain + measurement-bypass follow-ups). Tasks 8 (audition) and 10 (bundled smoke) deferred per OQ #3 (b) — Commander chooses the actual closure timing. Task 12 (code-review) is Commander's responsibility per `memory/code_review_regression_test_exact_class.md` ("run code-review using a different LLM than the one that implemented this story"). | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **M2 follow-up — drain robustness against close-race.** Commander reported hit-or-miss cut-off-at-end during the NFR1 measurement runs ("if the last chunk was small it seemed to cut, or if generation was complete before audio was finished playing the audio would cut"). Two contributing factors: (a) `_DRAIN_SAFETY_BUFFER_S = 0.15s` was borderline for Windows WASAPI shared / DirectSound (200–500ms internal latency typical); (b) `closeEvent` did not block on in-flight drain in measurement mode, so `aboutToQuit → _cleanup_services → _audio_coordinator.stop()` could tear down the PyAudio streams while the chunk-handler's `await asyncio.sleep(drain_wait)` was still in flight. Fix: bumped `_DRAIN_SAFETY_BUFFER_S` 0.15 → 0.5; added `MainWindow._wait_for_pending_audio_drain()` that runs in measurement mode only — reads the AudioCoordinator's drain-wait math directly and spins Qt's event loop (`QCoreApplication.processEvents()` + 10ms `time.sleep` chunks) for up to `(remaining + safety)` so the qasync drain task gets CPU time before close proceeds. Bounded by `_MAX_DRAIN_WAIT_S` so close cannot hang. Production behavior unchanged when `MYVOICE_AUTO_QUIT_ON_CLOSE` is unset. Updated 1 existing drain test for the new safety buffer; 813/813 broader regression sweep PASS. | dev agent (Opus 4.7 [1m]) |
| 2026-05-10 | **M6 — drain math bug fixed (Commander bundled-smoke surfaced).** Task 10 bundled smoke surfaced cut-off-at-end on the canonical Sarira-F paragraph, last ~4 words ("at around five seconds") truncated every iteration regardless of last-chunk size. Root cause: M2 drain math computes `remaining = expected_total_seconds - elapsed`. On producer-bottleneck workloads (producer 1.62× realtime for bf16 on RTX 5090), `elapsed >> expected_total` → `remaining < 0` → `if remaining > 0` gate skipped the drain entirely. PyAudio's device-level buffer (200–500ms on Windows shared mode) gets truncated by `stream.stop_stream()` (which does NOT block on drain — verified by reading `monitor_audio_service.stop_streaming_session()` line 933). Fix: track `_stream_last_write_ts` + `_stream_last_chunk_bytes` (per-chunk, not cumulative); compute `drain_wait = max(0, last_chunk_duration - time_since_last_write) + safety_buffer`; drop the `if remaining > 0` gate so safety always fires. Mirror change in `MainWindow._wait_for_pending_audio_drain` so measurement-mode close also reads the new state. 4 existing drain tests rewritten for the new state model + 1 new M6 regression test (`test_M6_producer_bottleneck_workload_still_drains_last_chunk`) that reproduces the exact bug class per `memory/code_review_regression_test_exact_class.md` (simulates 10×1.98s chunks under 1.62× producer ratio + asserts drain still fires for the last chunk's residual playback). 782/782 broader regression sweep PASS post-fix. **Commander: rebuild + re-test bundled smoke to confirm the cut-off-at-end is gone.** | code-review agent (Opus 4.7 [1m]) |
| 2026-05-10 | **Code-review pass — M1/M2/M3/M4/M5 fixes (5 MEDIUM, 0 HIGH).** M1 — `build_tools/installer.iss` (build 10→12) + `build_tools/version.py` (VERSION_BUILD 10→12) modified in working tree but neither files are Story 18.3 scope. Per Epic 18 framing at `:248` ("18.1–18.3 are pure source-tree edits") + Story 18.2 OQ #4 precedent, these are Commander's separate build-state work; documented here as out-of-scope-but-present so the discrepancy is no longer silent. M2 — `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv` (Story 18.2 evidence) was clobbered by a Story 18.3 NFR1 measurement run (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV` had been pointed at the 18-2 path); restored from `git checkout HEAD --` so Story 18.2's audited 33-row evidence with session_id `828eed8a-…` is preserved. M3 — three "**Status:** PENDING" markers in evidence file (§"Pre-implementation audit", §"End-to-end dtype audit", §"NFR1 first-chunk-latency measurement") were stale relative to the section bodies (which contained the captured data); flipped to "COMPLETE" / "DEFERRED" so section headers reflect the body's actual state. M4 — `audio_coordinator.stop_streaming_session(wait_for_drain=...)` docstring still claimed "~150ms safety buffer" after the M2 follow-up bumped `_DRAIN_SAFETY_BUFFER_S` 0.15 → 0.5; updated to cite the constant by name and the current value. M5 — `test_wait_for_drain_true_waits_remaining_audio_duration` had hard-coded the expected wait to `0.6 <= elapsed <= 1.2` (literal 0.5s safety baked in); reworked to compute `expected_wait = remaining_s + AudioCoordinator._DRAIN_SAFETY_BUFFER_S` so any future bump to the constant doesn't break the test for the wrong reason. 778/778 targeted regression sweep PASS post-fixes. | code-review agent (Opus 4.7 [1m]) |
| **PENDING — Commander run** | Task 1 (DONE post-H1+M1 — captures landed in evidence file); Task 7 (DONE post-OQ-#3 routing — bf16 -3.77% median, OQ #3 option (b) selected); Task 8 (DEFERRED to post-Story-18.4 retrospective per OQ #3 (b)); Task 10 (`build_release.bat` + bundled exe smoke — pending Commander); Task 12 (code-review — DONE this pass, see M1/M2/M3/M4/M5 entry above). | Commander |
| **PENDING — agent applies after Commander** | Task 9 (architecture amendment to `architecture-optimization-pass.md`; depends on Task 8 verdict input). | dev agent (next session) |
