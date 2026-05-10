# Story 18.2: TF32 + cuDNN Benchmark Enable

Status: in-progress

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase tag: Phase ⊥-Polish-2 (D-20). Second story of Epic 18 (Generation-Speed Optimizations). Successor to Story 18.1 (Underrun-Gap Mitigation), which closed as instrumentation-only after pinning the bottleneck to the producer side at 3.23× steady-state ratio. -->
<!-- Risk: Low (per `epics-optimization-pass.md:244`). Lossless toggle on the precision the model already uses; matmul TF32 drift ~1e-4, perceptually inaudible per public PyTorch guidance and prior Ampere+ deployments. -->
<!-- Audition discipline: Commander solo, no L2/L3 recruitment (per `epics-optimization-pass.md:241` + Story 18.2 stub at `:1363`). The drift is well-known to be sub-perceptual; the spot-check is documentary diligence, not a gate. -->

## Story

As a **MyVoice user generating TRUE_STREAM utterances on an Ampere-or-newer CUDA host**,
I want **the application to opt into PyTorch's lossless TF32 matmul + cuDNN benchmark autotune at startup**,
so that **first-chunk latency drops 10–30% on Ampere+ tensor cores at zero perceptual cost, composing with Stories 18.3 + 18.4 toward the cumulative ~25 s → ~10 s long-form-utterance target the Epic 18 plan promises**.

## Acceptance Criteria

**Given** the application starts on an Ampere-or-newer CUDA host (`torch.cuda.is_available() == True` AND `torch.cuda.get_device_capability()[0] >= 8` — RTX 30xx/40xx/50xx in the documented ship-target per `memory/hardware_setup.md`)
**When** the new startup-side opt-in fires before the QApplication / QwenTTSService initialization
**Then** all three of the following PyTorch global flags become `True`:
  - `torch.backends.cuda.matmul.allow_tf32`
  - `torch.backends.cudnn.allow_tf32`
  - `torch.backends.cudnn.benchmark`

**And** the opt-in emits exactly one INFO-level log line on the application logger (e.g., `"TF32 + cuDNN benchmark enabled (device_capability=8.9)"`) so Commander can confirm engagement at runtime by inspecting `myvoice.log`
**And** the opt-in is **idempotent** — calling the enable function twice in the same process produces the same final flag state and no exception. The second-call discipline is implemented by checking the three `torch.backends.*` values on entry: if all three are already True AND the host is Ampere+, short-circuit at DEBUG level (no second INFO log, no second metric record). No module-level mutable state (this preserves Story 16.2's `streaming_mode.py` pure-function discipline; the only departure is the deliberate side-effect set itself).

**Given** the application starts on a CPU-only host (`torch.cuda.is_available() == False`) OR a pre-Ampere CUDA host (compute capability < 8.0 — e.g., RTX 20xx Turing, GTX 10xx Pascal)
**When** the same startup-side opt-in fires
**Then** **none** of the three PyTorch global flags are mutated from their default values (CPU/older-GPU path is identifiably unchanged — `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` remain at their PyTorch-installed defaults; `torch.backends.cudnn.benchmark` remains `False`)
**And** the opt-in logs a single DEBUG-level explanation of the skip with a structured reason (`"cuda_unavailable"` for CPU, `"pre_ampere"` for compute < 8.0, with the actual `device_capability` value if cuda is available)
**And** D-9 / NFR12 (hardware-aware default discipline; CPU-only hosts stay on the V2 baseline) are preserved verbatim (the `streaming_mode.py:54-56` Ampere+ check is the precedent the new probe mirrors structurally)

**Given** the opt-in fires (either branch — engaged or skipped)
**When** the telemetry emission completes
**Then** a single `metrics.record(...)` call captures the outcome on the established D-19 telemetry surface:
  - **Ampere+ engaged branch:** `metrics.record("tf32_cudnn_benchmark_enabled", 1.0, device_capability="<major>.<minor>")` (e.g., `device_capability="8.9"`)
  - **Skipped branch:** `metrics.record("tf32_cudnn_benchmark_enabled", 0.0, reason="cuda_unavailable" | "pre_ampere", device_capability="<major>.<minor>" if cuda else "none")`

**And** the metric name (`tf32_cudnn_benchmark_enabled`) is documented at the new module's docstring + at the evidence file §"Telemetry breadcrumb" so Stories 18.3 + 18.4 can correlate their throughput uplifts against confirmed-engaged TF32+cuDNN baselines
**And** the metric emission integrates with the existing `metrics.record(name, value, **tags)` pub-sub helper at `src/myvoice/observability/metrics.py:77` (no new metric infrastructure; same listener surface that Story 18.1's three CSV-capture metrics use)

**Given** the new module wires the probe + enable + telemetry into `src/myvoice/main.py`
**When** the wire-up lands
**Then** the call fires **at startup, before `QwenTTSService` initialization**, so the flags are global to the process from the earliest correct moment (the exact placement — module-level after the `import torch` block vs. inside `main()` after `setup_logging()` returns — is deferred to Open Question #1 because the choice has runtime-observable consequences for where the INFO log line lands)
**And** the wire-up is **a single function call** — `enable_tf32_and_cudnn_benchmark()` — not inlined logic at the call site (so the call site stays readable and the unit-testable surface is the function, not `main.py`'s startup sequence)
**And** the wire-up is **unconditional from main.py's perspective** — `main.py` does NOT inspect hardware capability before calling; the probe lives inside the function. This keeps the conditional-logic surface in one place (the new module) instead of scattered across `main.py` + the new module.
**And** `main.py`'s existing torch-before-PyQt6 DLL ordering invariant per `memory/torch_pyqt6_dll_ordering.md` is preserved verbatim (the new call sits after the torch import, before any PyQt6 import) regardless of which placement Open Question #1 resolves to

**Given** the new module is wired and engaged
**When** Commander runs the canonical Story 17.3 §4.1 step 3 long-form CLONED utterance (Sarira-F, ≥250 chars / ~22 s of speech) on the RTX 5090 dev host with the same `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` env-var-gated capture infrastructure that Story 18.1 shipped (`src/myvoice/observability/progressive_playback_csv_capture.py`)
**Then** the captured `metrics.first_chunk_latency_ms` value (already-aggregated by the `_FirstChunkLatencyAggregator` at `qwen_tts_service.py:362`) is compared head-to-head against the **same utterance under the same conditions with the new opt-in disabled** (a one-line revert of the wire-up in `main.py`, captured to a second CSV)
**And** the measurement is captured at `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md §"NFR1 first-chunk-latency measurement"` with both raw CSVs (`18-2-rtx5090-tf32-on.csv` + `18-2-rtx5090-tf32-off.csv`), median + p90 + p95, and the absolute + percent delta
**And** the captured percent delta is surfaced verbatim to Commander in the closure note (Task 8) — the Epic 18 stub at `:1361` quotes a 10–30% speedup as the *anticipated* gate; the dev agent does NOT unilaterally interpret tails outside that range as pass/fail. If the measured median speedup falls outside [10%, 30%], the dev agent flags this in the closure note (Open Question #4 below) and waits for Commander's call rather than declaring the story closed or stalled.
**And** zero NFR3 perceptual defects are reported by Commander on the same utterance — Sarira-F long-form audition by Commander solo, no L2/L3 recruitment per Epic 18's Commander-solo discipline for 18.1 + 18.2

**Given** the chosen mitigation lands
**When** any host without Ampere+ CUDA support runs the application
**Then** the CPU-only / pre-Ampere path is **identifiably unchanged** (per AC #2 above, verified by absence of flag mutation + absence of telemetry-record emission with `value=1.0` on those hosts)
**And** no new tunables or settings are exposed (no `AppSettings` field, no UI toggle — the flags engage automatically and unconditionally on Ampere+; users who want to disable them edit the new module, but that is a developer operation, not a user-facing setting; Story 18.3's `tts_precision = "fp32" | "bf16" | "auto"` is the *first* Epic 18 user-facing tunable per the epic stub at `:1377`, not this story)

**Given** the test suite runs after the source-tree edits
**When** the regression sweep executes
**Then** the existing Story 18.1 instrumentation tests (22 tests across `tests/unit/test_app_progressive_playback_instrumentation.py` + `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py` + `tests/unit/observability/test_progressive_playback_csv_capture.py`) pass with **zero regressions**
**And** the existing Story 17.3 progressive-playback tests (32 tests across `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` + `tests/unit/test_app_progressive_playback.py` + `tests/unit/test_app_progressive_playback_cancel.py` + `tests/integration/test_progressive_playback_dispatch_skip.py`) pass with **zero regressions**
**And** the existing Story 16.2 streaming-mode hardware-probe tests at `tests/unit/services/tts_streaming/test_streaming_mode.py` pass with zero regressions (the new module is a sibling of `streaming_mode.py` — same package; touching that package's `__init__.py` for the new export must not regress 16.2's import chain)
**And** new unit tests at `tests/unit/services/tts_streaming/test_torch_runtime.py` (or equivalent path mirroring `test_streaming_mode.py`'s location) cover all four hardware truth-table branches:
  - cuda available + Ampere+ (capability ≥ 8.0) → all three flags True + INFO log + metric value 1.0
  - cuda available + pre-Ampere (capability 7.5 — Turing) → flags unchanged + DEBUG skip log + metric value 0.0 with `reason="pre_ampere"`
  - cuda unavailable (CPU-only) → flags unchanged + DEBUG skip log + metric value 0.0 with `reason="cuda_unavailable"`
  - second call within same process → idempotent (no error, no double-log, final flag state same as first call)

**Given** the bundled-environment smoke from Story 17.3 §4.1 procedure remains the production-verification gate
**When** the dev agent runs the bundled smoke after the source-tree edits
**Then** a fresh `build_release.bat` cycle produces a `build_tools/dist/MyVoice/MyVoice.exe` portable bundle
**And** Commander's bundled-mode runtime confirms the `myvoice.log` contains the expected single INFO-level "TF32 + cuDNN benchmark enabled (device_capability=...)" line on the RTX 5090 ship-target host
**And** Commander confirms zero perceptual defects on the same Sarira-F long-form utterance compared to the Story 17.3 / 18.1 baseline — i.e., the lossless-TF32 promise (sub-1e-4 matmul drift) holds on the actual production-bundled artifact, not just the dev source tree
**And** the bundled-smoke evidence is captured at `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md §"Bundled smoke"` — the same Story 17.3 / 18.1 / 17.2 / 17.1 evidence-file pattern

**Given** the story is closed
**When** the post-implementation accounting runs
**Then** the change log records the absolute + percent first-chunk-latency delta (Task 4) so Stories 18.3 + 18.4 can baseline their throughput uplifts against a TF32-already-engaged starting point — i.e., the `metrics.first_chunk_latency_ms` value Stories 18.3 + 18.4 measure is the post-18.2 number, not the pre-18.2 number
**And** the architecture document `_bmad-output/planning-artifacts/architecture-optimization-pass.md` is **NOT** amended (per Epic 18 framing at `:234`: "No new D-decisions"; D-9 / NFR1 / NFR3 / NFR7 / NFR12 are all preserved verbatim by Story 18.2)
**And** no `requirements.txt` / installer-spec / `build_release.bat` edits are made (per Epic 18 framing at `:248`: "18.1–18.3 are pure source-tree edits; no `requirements.txt` / installer-spec / `build_release.bat` changes anticipated")

## Tasks / Subtasks

- [x] **Task 1 — New `torch_runtime.py` module: hardware probe + flag enable + telemetry** (AC: #1, #2, #3, #6)
  Build the testable, side-effect-free probe + the thin idempotent enable side-effect. Mirror Story 16.2's `streaming_mode.py` pattern (signal-free, side-effect-free decision + a thin separate enable path), so the conditional-logic surface lives in exactly one module and is unit-testable without a real GPU.
  - [x] 1.1 Create `src/myvoice/services/tts_streaming/torch_runtime.py`. Module docstring cites Epic 18 + Story 18.2 + Epic 18 stub `:1347` + the Ampere+ guard precedent at `streaming_mode.py:54-56`. Same import discipline as `streaming_mode.py`: lazy-import torch inside the probe function (so monkeypatch of `torch.cuda.is_available` and `torch.cuda.get_device_capability` is honored without import-order gymnastics), no peer imports from `myvoice.*`.
  - [x] 1.2 Implement `is_ampere_or_newer() -> bool`. Returns False if `torch.cuda.is_available()` is False; otherwise returns `torch.cuda.get_device_capability()[0] >= 8`. Document the capability-major mapping inline (Ampere = 8.x including 8.9 RTX 40xx + 8.6 RTX 30xx; Hopper = 9.x; Blackwell = 10.x RTX 50xx; pre-Ampere = Turing 7.5 / Pascal 6.x). The `>= 8` test is forward-compatible (Hopper / Blackwell / future architectures all satisfy it).
  - [x] 1.3 Implement `enable_tf32_and_cudnn_benchmark() -> dict`. Returns a small status dict: `{"engaged": bool, "reason": str | None, "device_capability": tuple[int, int] | None}` so the wire-up at `main.py` (Task 2) can log a single line carrying the actionable detail without re-probing torch. Internally:
      - **Idempotency check first.** On entry, if `is_ampere_or_newer()` is True AND all three `torch.backends.*` flags are already True, log at DEBUG ("TF32 + cuDNN already enabled — no-op"), return the engaged-True status dict, do NOT re-emit the metric, do NOT re-log INFO. The first-call observable behavior is the contract; subsequent calls are equivalence-class no-ops. **No module-level mutable state** — the flag values themselves are the canonical "have we run" signal. This preserves Story 16.2's `streaming_mode.py` pure-function discipline; the only deliberate departure is the side-effect set itself, kept minimal.
      - If `is_ampere_or_newer()` is False, determine reason (`"cuda_unavailable"` if `not torch.cuda.is_available()`; else `"pre_ampere"`); log a single DEBUG line with the reason; emit `metrics.record("tf32_cudnn_benchmark_enabled", 0.0, reason=reason, device_capability=...)` telemetry; return the engaged-False status dict.
      - If `is_ampere_or_newer()` is True AND not all three flags are already set, set the three flags to True (`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.backends.cudnn.benchmark`); log a single INFO line; emit `metrics.record("tf32_cudnn_benchmark_enabled", 1.0, device_capability="<major>.<minor>")` telemetry; return the engaged-True status dict.
  - [x] 1.4 Update `src/myvoice/services/tts_streaming/__init__.py`: add `is_ampere_or_newer` and `enable_tf32_and_cudnn_benchmark` to the package's `from ... import` list and `__all__` (mirroring how `default_streaming_mode_for_hardware` is exported). The two new exports sit alongside `StreamingMode` / `default_streaming_mode_for_hardware` / `effective_streaming_mode` since they are conceptually the same surface (hardware-gated startup-side decisions).

- [x] **Task 2 — Wire `enable_tf32_and_cudnn_benchmark()` into `main.py` startup** (AC: #4)
  Single-call wire-up at the canonical earliest-correct startup point. No conditional logic at the call site (Task 1.3 owns the conditional surface). Preserves the torch-before-PyQt6 DLL ordering invariant.
  Per OQ #1 resolution: placed inside `main()` immediately after `setup_logging()` returns and before `setup_application()` is called (the recommended placement; INFO breadcrumb lands in `myvoice.log` via the file handler `setup_logging()` configures, not stderr-only).
  - [x] 2.1 Edit `src/myvoice/main.py`. Immediately after the existing torch-import `try: import torch / except (ImportError, OSError): pass` block (currently `:42-49`), add a guarded import + call to the new function. Guard the import with the same `try / except ImportError` discipline the file already uses for torch — if the import fails (e.g., during a partial install), the application MUST continue to start, with a single warning log; the absence of the speedup is non-fatal. Concretely:
      ```python
      # Story 18.2: enable lossless TF32 + cuDNN benchmark autotune on Ampere+
      # (no-op on CPU / pre-Ampere). Preserves D-9 / NFR12 hardware-aware
      # default discipline.
      try:
          from myvoice.services.tts_streaming.torch_runtime import enable_tf32_and_cudnn_benchmark
          enable_tf32_and_cudnn_benchmark()
      except Exception as _tf32_err:
          # Non-fatal: the speedup is opt-in; absence is the V2 baseline.
          logging.getLogger(__name__).warning(
              f"TF32 + cuDNN benchmark enable failed (continuing without speedup): {_tf32_err}"
          )
      ```
      The placement is BEFORE `setup_application()` runs at `:60-102` and BEFORE the qasync import / event-loop creation, so the flags are global to the process from the earliest possible moment.
  - [x] 2.2 Verify the placement does NOT violate the torch-before-PyQt6 DLL ordering invariant: the new code touches only `myvoice.services.tts_streaming.torch_runtime` (which lazy-imports torch inside the probe) and the standard library `logging` module; it does NOT import PyQt6. The PyQt6 import block at `:51-54` stays unchanged and runs after the new call.
  - [x] 2.3 Verify the placement does NOT change `setup_logging()` ordering: `setup_logging()` runs inside `main()` at `:334`, AFTER module-level code finishes. The new wire-up at module level (or inside `main()` BEFORE `setup_application()` but AFTER `setup_logging()`) — pick whichever placement keeps the new INFO/DEBUG log line landing in the expected `myvoice.log` file. **Recommended:** place the new call at the top of `main()` immediately after `setup_logging()` returns and BEFORE `setup_application()` is called, so the log line lands in `myvoice.log` rather than getting lost on stderr. Document the placement choice at evidence file §"Wire-up placement".

- [x] **Task 3 — Unit tests at `tests/unit/services/tts_streaming/test_torch_runtime.py`** (AC: #7)
  Mirror `tests/unit/services/tts_streaming/test_streaming_mode.py`'s structure; same monkeypatch-the-torch-API pattern; PyTest-only, no real GPU required. **15 tests landed, all passing on `python310` first run.**
  - [x] 3.1 Test: cuda available + Ampere → engaged. Monkeypatch `torch.cuda.is_available` to return True; monkeypatch `torch.cuda.get_device_capability` to return `(8, 9)`. Assert `is_ampere_or_newer()` returns True; assert `enable_tf32_and_cudnn_benchmark()` returns a dict with `engaged=True` + `device_capability=(8, 9)`; assert (via spy/listener on the metrics module) that exactly one `metrics.record("tf32_cudnn_benchmark_enabled", 1.0, ...)` call fires; assert all three `torch.backends.*` flags are True after the call.
  - [x] 3.2 Test: cuda available + pre-Ampere (Turing 7.5) → skipped. Monkeypatch `torch.cuda.is_available` → True; `torch.cuda.get_device_capability` → `(7, 5)`. Assert `is_ampere_or_newer()` returns False; assert `enable_tf32_and_cudnn_benchmark()` returns `engaged=False, reason="pre_ampere"`; assert metric records 0.0 with `reason="pre_ampere"`. **Critical:** assert the three `torch.backends.*` flags retain the values they had BEFORE the call (capture them in the test's `setUp` / fixture and compare). Do NOT assume any specific PyTorch default; the contract is "no mutation," not "specific value preserved."
  - [x] 3.3 Test: cuda unavailable (CPU-only) → skipped. Monkeypatch `torch.cuda.is_available` → False; `torch.cuda.get_device_capability` should NOT be called (defensive — pre-Ampere systems may not even have CUDA installed; the function must early-out on `is_available()=False` without touching `get_device_capability`). Assert `engaged=False, reason="cuda_unavailable"`; assert metric records 0.0 with `reason="cuda_unavailable"`; assert flags unchanged. **Implemented by patching `get_device_capability` to RAISE so any accidental invocation surfaces immediately.**
  - [x] 3.4 Test: idempotency — second call within same process. Set up Ampere+ monkeypatch; pre-set the three `torch.backends.*` flags to True (simulating a prior call's effect); call `enable_tf32_and_cudnn_benchmark()`; assert the function returns `engaged=True` with the correct `device_capability`; assert no INFO log fires (only DEBUG); assert no NEW `metrics.record` call fires (the function observes the already-engaged state and short-circuits at DEBUG level). Because the implementation reads the flag values directly (no module-level state per Task 1.3), this test does NOT need a module-level reset fixture — but DOES need to capture and restore the three `torch.backends.*` flags around the test (use a `@pytest.fixture` that snapshots-and-restores, mirroring how Task 3.2 captures flags before assertion). Document the snapshot-and-restore fixture at the test-file docstring. **Two idempotency tests landed: pre-set-True scenario + back-to-back-from-cold scenario.**
  - [x] 3.5 Test: telemetry tag schema. For both branches (engaged and skipped), assert the metric record's tags dict carries `device_capability` as a string (formatted `"<major>.<minor>"` for cuda-available paths; either `"none"` or omitted for the cuda-unavailable branch — pick one and document at the function docstring). The string-vs-tuple choice matters because Story 18.1's CSV-capture infrastructure stringifies tag values; the metric must be CSV-compatible from day one. **Resolved per OQ #2 → `"none"` sentinel. Parametrized across 5 hardware shapes (8.9, 10.0, 9.0, 7.5, cuda-unavailable).**
  - [x] 3.6 Test: D-19 listener-isolation discipline. If the new module's metric emission raises (e.g., a buggy listener), the function MUST still successfully set the flags; the broader application startup MUST NOT abort. This mirrors the metrics module's own AC #9 listener-exception isolation at `metrics.py:144-region`. Test by registering a listener that raises; assert the function returns engaged=True with all flags set; assert the exception was logged but did not propagate.

- [ ] **Task 4 — NFR1 first-chunk-latency empirical measurement** (AC: #5) — **Tasks 4.2–4.5 require Commander on RTX 5090 dev host; Task 4.1 closed by dev agent.**
  The Epic 18 stub at `:1361` quotes "10–30% on RTX 5090" as the anticipated acceptance gate. The dev agent's job is to capture the data and surface the percent-delta number to Commander; tails outside [10%, 30%] route to Open Question #4 rather than dev-agent interpretation.
  - [x] 4.1 Extend the existing Story 18.1 CSV-capture infrastructure (`src/myvoice/observability/progressive_playback_csv_capture.py`) to ALSO accept `first_chunk_latency_ms` in its metric-name filter list. This is a one-line backward-compatible change; the existing three Story 18.1 metrics keep capturing as before. The committed approach (NOT a separate ad-hoc listener) so Stories 18.3 + 18.4 inherit the same first-chunk capture surface for their own throughput uplifts. Document the filter-list extension at evidence file §"Measurement methodology" (one paragraph: "extended the Story 18.1 CSV-capture filter to include `first_chunk_latency_ms` for Story 18.2 measurement; reused for 18.3 + 18.4"). **Implemented + 2 new tests added at `test_progressive_playback_csv_capture.py` (`test_only_targeted_metrics_are_captured` updated; new `test_first_chunk_latency_row_columns_match_header`). 14/14 passing.**
  - [ ] 4.2 Capture the **after** measurement first (the wire-up is part of this story's HEAD commit, so it's already in place). Run the canonical Story 17.3 §4.1 step 3 paragraph (Sarira-F long-form, ≥250 chars / ~22 s of speech) on the RTX 5090 dev host with `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` set; capture **N=10 generations minimum** to `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv`. Each generation must be a fresh process launch (kill the app between runs) so cuDNN benchmark autotune cache state does not bleed across runs. Record `first_chunk_latency_ms` median + p90 + p95.
  - [ ] 4.3 Capture the **before** baseline by checking out the parent commit (the commit BEFORE Task 2's wire-up landed — i.e., HEAD~1 if the wire-up commit is the current HEAD; whichever git ref is the immediate ancestor of the wire-up). Run N=10 generations on the SAME Sarira-F utterance; capture to `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-off.csv`. Record median + p90 + p95. **Critical:** do NOT measure by source-tree-edits-then-revert; the git-commit-pair methodology eliminates the risk of forgetting the revert and ensures both runs are reproducibly identifiable. After the baseline is captured, `git checkout` back to the wire-up commit.
  - [ ] 4.4 Compute the delta. `(off_median - on_median) / off_median * 100` = percent speedup; same for p90 and p95. Capture absolute pre/post values (in ms) AND the three percent deltas at evidence file §"NFR1 first-chunk-latency measurement". Per AC #5: if the median speedup falls outside [10%, 30%], surface to Commander via Open Question #4 — do NOT declare the story done or stalled unilaterally.
  - [ ] 4.5 Sanity-check that the `myvoice.log` "TF32 + cuDNN benchmark enabled (device_capability=...)" INFO line appears in the **after** runs and does NOT appear in the **before** runs. This is the runtime-engagement breadcrumb AC #1's INFO log promises and the canonical confirmation that the two CSVs really do bracket the change Story 18.2 introduces.

- [ ] **Task 5 — NFR3 spot-check (Commander solo)** (AC: #5)
  Per Epic 18 stub at `:241` + `:1363`: Commander solo, no L2/L3 recruitment. The TF32 matmul drift is well-known to be sub-perceptual at ≤1e-4; the spot-check is documentary diligence to confirm "no surprise on this specific qwen_tts + Sarira-F + RTX 5090 combination," not a gate.
  - [ ] 5.1 On the same canonical Sarira-F long-form utterance, Commander listens to the **before** baseline (TF32 off) recording AND the **after** (TF32 on) recording back-to-back, A/B style. The two recordings already exist as the audio Tasks 4.2 + 4.3 generated; do not regenerate.
  - [ ] 5.2 Commander logs perceptual observation at evidence file §"NFR3 spot-check": expected verdict = "indistinguishable" (the public PyTorch guidance for TF32 on inference is a 1e-4 round-trip error, well below human perceptual sensitivity for waveform amplitude); the documented-diligence verdict is the literal phrase "no perceptual defect detected" or "perceptual defect detected: <description>". The latter is the only branch that triggers a re-think (likely a TF32-incompatibility surprise on this specific model — the Story 18.3 NFR7 fp32 fallback machinery would then need to extend to cover TF32 too, but that is explicitly NOT this story's surface and would be Commander-routed to the architecture layer rather than absorbed by the dev agent).

- [ ] **Task 6 — Bundled-environment smoke** (AC: #8)
  Production-verification gate. The source-tree wire-up MUST also engage in the bundled `MyVoice.exe` artifact, not just the dev source tree. The bundled artifact is what users actually run.
  - [ ] 6.1 Run `build_release.bat` (the standard production-bundle build cycle per `memory/build_tools_phase_perp_state.md` + `memory/production_release_state.md`). Confirm the resulting `build_tools/dist/MyVoice/MyVoice.exe` includes the new `src/myvoice/services/tts_streaming/torch_runtime.py` module + the `main.py` wire-up.
  - [ ] 6.2 Launch the bundled exe. Confirm `myvoice.log` (in the portable Logs path per `setup_logging()` discipline) contains the single INFO-level "TF32 + cuDNN benchmark enabled (device_capability=...)" line on Commander's RTX 5090 + Win11 ship-target host. Commander logs this at evidence file §"Bundled smoke" with a verbatim log excerpt.
  - [ ] 6.3 In the bundled exe, run a single short-class TTS generation (the Story 17.3 §4.1 step 1 short paragraph from the standard fixture). Confirm no new error, warning, or unexpected log line appears around the TF32 wire-up. Confirm a TTS generation completes successfully end-to-end (audio plays through the streaming pipeline).
  - [ ] 6.4 If the bundled smoke surfaces a defect (e.g., the new module fails to import in the PyInstaller / portable-bundle environment for a packaging reason), the dev agent surfaces it to Commander rather than absorbing it; the most likely failure mode is `MEIPASS`-path invisibility of the new module, which would be a one-line fix to PyInstaller's hidden-imports list at `build_tools/`. Document at evidence file §"Bundled smoke".

- [x] **Task 7 — Regression test sweep** (AC: #7)
  Verify the new module + wire-up does not regress the established surfaces.
  - [x] 7.1 Run the new Task 3 unit tests: `pytest tests/unit/services/tts_streaming/test_torch_runtime.py -v`. Expect 6–8 tests pass (one per Task 3.1 / 3.2 / 3.3 / 3.4 / 3.6; Task 3.5 covers tag schema for both engaged + skipped branches → 1–3 tests depending on parametrization). Exact count is informational; the contract is "all six Task 3 subtasks have at least one passing test that exercises them." **15/15 PASS** (parametrized tag-schema test expands the count above the 6–8 estimate).
  - [x] 7.2 Run the existing Story 18.1 instrumentation tests: `pytest tests/unit/test_app_progressive_playback_instrumentation.py tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py tests/unit/observability/test_progressive_playback_csv_capture.py -v`. Expect 22 tests pass (the count Story 18.1 closed at). **23/23 PASS** (6 + 3 + 14, the +1 vs Story 18.1's 22 is the new Task 4.1 row-columns test).
  - [x] 7.3 Run the existing Story 17.3 progressive-playback tests: `pytest tests/unit/services/test_qwen_tts_service_true_stream_callback.py tests/unit/test_app_progressive_playback.py tests/unit/test_app_progressive_playback_cancel.py tests/integration/test_progressive_playback_dispatch_skip.py -v`. Expect 32 tests pass (the count Story 18.1 closed at after the Story 17.3 suite expanded). **32/32 PASS**.
  - [x] 7.4 Run the Story 16.2 streaming-mode tests: `pytest tests/unit/services/tts_streaming/test_streaming_mode.py -v`. Expect zero regressions from the Task 1.4 `__init__.py` widening. **16/16 PASS**.
  - [x] 7.5 Run the broader streaming + app + audio + observability surface: `pytest tests/unit/services/test_qwen_tts_service_dispatch.py tests/unit/services/test_qwen_tts_service_session_integration.py tests/unit/observability/ tests/unit/test_app_progressive_playback*.py tests/integration/test_progressive_playback_dispatch_skip.py -v`. Expect ~166+ tests pass with zero regressions (Story 18.1's broader-sweep count, plus the new 6 from Task 7.1). **152/152 PASS** (the broader sweep without test_torch_runtime; 152 + 15 new = 167 grand total).

- [ ] **Task 8 — Code-review pass** (post-implementation)
  - [ ] 8.1 Run `/bmad-bmm-code-review`. Per `memory/code_review_regression_test_exact_class.md`: HIGH/MEDIUM-fix regression tests must mirror the exact bug class. Expected review-finding categories for this story: (a) idempotency contract drift (the second-call discipline is the most-easily-broken AC); (b) telemetry tag schema drift (string vs tuple `device_capability`); (c) wire-up placement drift (between `setup_logging()` and `setup_application()` is the contract; landing it elsewhere subtly breaks AC #4); (d) test discipline — Task 3.4 idempotency reset must use a fixture not module-level state; (e) bundled-smoke evidence file completeness.
  - [ ] 8.2 Address findings. Re-run code-review twice after non-trivial auto-fixes (the established Stories 16.7 / 16.8 / 17.1 / 17.2 / 17.3 / 18.1 pattern). Commit per the established Story-NNN: code-review pass — H#/M#/L# fixes pattern.

## Dev Notes

### What this story is

Story 18.2 is the second story of Epic 18 (Generation-Speed Optimizations / Phase ⊥-Polish-2). It enables PyTorch's lossless TF32 + cuDNN benchmark autotune at startup on Ampere-or-newer CUDA hosts, gated on a hardware probe that mirrors Story 16.2's `streaming_mode.py:54-56` precedent.

This is the **cheapest available speedup** in the Epic 18 plan (per `epics-optimization-pass.md:1349`). It composes with Story 18.3 (bf16 precision) and Story 18.4 (`torch.compile`) toward the cumulative Epic 18 throughput target: a 25-second long-form CLONED utterance generating in ~10s instead of ~40s on the RTX 5090, with no perceptual quality regression.

The change is structurally tiny (~5 LOC of substantive logic, plus a probe + telemetry wrapper) but the testable surface is a discrete module so the conditional logic is unit-testable without a real GPU. This is the same pattern Story 16.2 established for `streaming_mode.py`.

### What this story is NOT

- **Not a precision change.** TF32 is the same fp32 numerical path PyTorch already uses; the matmul rounding loses ~1e-4 precision on Ampere+ tensor cores. This is NOT bf16 (Story 18.3) and NOT fp16 (out of scope per Epic 18 framing at `:250`).
- **Not a producer-side throughput rework.** Story 18.1's evidence file §4.4 pinned the long-form-utterance bottleneck to producer cadence (talker model decode rate at 31% real-time, ratio 3.23×). TF32 helps the matmul-bound segments of the producer side, which is *part of* the bottleneck Story 18.1 identified, but the named "fix class" for the producer bottleneck is Stories 18.3 (bf16) + 18.4 (`torch.compile`) per the Story 18.1 evidence file §4.4 verdict. Story 18.2 is a complement, not a substitute.
- **Not an audition cycle.** Stories 18.3 + 18.4 trigger the full ≥3-listener NFR3 re-audition mirroring Story 17.1's protocol. Story 18.1 + 18.2 are explicitly Commander-solo per `epics-optimization-pass.md:241`. The 1e-4 matmul drift is sub-perceptual; the Task 5 spot-check is documentary diligence, not a gate.
- **Not a CPU / pre-Ampere change.** Per D-9 / NFR12, CPU-only and pre-Ampere hosts (Turing 7.5, Pascal 6.x) stay on their current behavior. The flags engage only on Ampere+ CUDA hosts. The hardware probe is the gate.
- **Not a build-pipeline change.** No `requirements.txt` / installer-spec / `build_release.bat` edits anticipated; pure source-tree edits picked up by the next build cycle. Per Epic 18 framing at `:248`.
- **Not a user-facing setting.** No `AppSettings` field, no UI toggle. The flags engage automatically and unconditionally on Ampere+. Story 18.3's `tts_precision = "fp32" | "bf16" | "auto"` is the *first* Epic 18 user-facing tunable per the epic stub at `:1377`, not this story.
- **Not an architecture amendment.** Per Epic 18 framing at `:234`: "No new D-decisions." D-9 / NFR1 / NFR3 / NFR7 / NFR12 are all preserved verbatim by Story 18.2.

### Source tree components to touch

**Read-only (analysis/reference):**
- `src/myvoice/services/tts_streaming/streaming_mode.py:54-56` — the canonical Ampere+ probe precedent (Story 16.2). Story 18.2's new probe mirrors this structurally (lazy torch import; no peer imports; pure decision function).
- `src/myvoice/services/tts_streaming/__init__.py` — package re-export pattern. Task 1.4 widens this with the two new public symbols.
- `src/myvoice/observability/metrics.py:77-150-region` — `metrics.record(name, value, **tags)` API surface. Task 1.3's telemetry uses this verbatim; same listener pattern Story 18.1's three CSV-capture metrics use.
- `src/myvoice/services/qwen_tts_service.py:362-region + :3134 + :4136` — existing `first_chunk_latency_ms` recording sites. Task 4's NFR1 measurement consumes the metric Story 18.1's CSV-capture infrastructure (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`) needs to extend to.
- `src/myvoice/observability/progressive_playback_csv_capture.py` — Story 18.1's env-var-gated CSV capture. Task 4.1 picks between extending it to also capture `first_chunk_latency_ms` (one-line filter list change) OR running a separate listener for this story.

**New (source tree):**
- `src/myvoice/services/tts_streaming/torch_runtime.py` (Task 1) — the new probe + enable + telemetry module. Single file, ~80 LOC including docstring + the two functions.

**Edit (source tree):**
- `src/myvoice/services/tts_streaming/__init__.py` (Task 1.4) — two new public symbols added to imports + `__all__`.
- `src/myvoice/main.py` (Task 2) — single guarded import + call; placement after `setup_logging()` and before `setup_application()` per Task 2.3.
- (optional, Task 4.1) `src/myvoice/observability/progressive_playback_csv_capture.py` — one-line filter-list extension if the CSV-capture path is the chosen measurement gate. Backwards-compatible.

**New (tests):**
- `tests/unit/services/tts_streaming/test_torch_runtime.py` (Task 3) — six tests mirroring `test_streaming_mode.py`'s structure. Same monkeypatch pattern; PyTest-only, no real GPU.

**New (evidence + measurement artifacts):**
- `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md` (Tasks 4.4, 5.2, 6.2). Force-add via `git add -f` per the Story 16.9 / 17.1 / 17.2 / 17.3 / 18.1 evidence-file precedent (`_bmad-output/` is gitignored per `memory/git_repo_state.md`).
- `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-off.csv` (Task 4.2) — N=5 baseline. Force-add same as evidence file.
- `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv` (Task 4.3) — N=5 post-enable. Force-add same as evidence file.

### Testing standards summary

- **Unit tests:** mirror Story 16.2's `test_streaming_mode.py` patterns. Monkeypatch `torch.cuda.is_available` and `torch.cuda.get_device_capability` at the attribute-access level (not by binding a function reference) so the lazy-import discipline in the new module is honored. PyAudio / Qt / qwen-tts NOT involved in this test surface — the new module is isolated to torch attribute-checks.
- **No real GPU required.** All hardware-truth-table branches (Ampere, pre-Ampere, CPU-only) are exercised via monkeypatch. The real-GPU-only validation lives in Tasks 4 + 5 + 6 (NFR1 measurement, NFR3 spot-check, bundled smoke) on Commander's RTX 5090.
- **Idempotency reset discipline:** Task 3.4 must reset module-level state in a fixture, not rely on test ordering. Document this at the test-file docstring (`pytest fixture: torch_runtime_reset` or equivalent).
- **Conftest discipline:** `tests/conftest.py` already enforces torch-before-PyQt6 DLL ordering per `memory/torch_pyqt6_dll_ordering.md`. The new test file does NOT need a special preamble — it is pure-Python torch-attribute monkeypatching, no DLL-ordering surface area. If running under coverage for Task 7.5, follow the inline torch-first preamble per `memory/torch_before_coverage_dll_ordering.md`.
- **No new test-harness changes.** Story 16.2 + Story 18.1 established the patterns; this story extends them.

### Project Structure Notes

**Alignment with unified project structure:**
- New module at `src/myvoice/services/tts_streaming/torch_runtime.py` — sibling of `streaming_mode.py`, `codec_token_streamer.py`, `streaming_decoder.py`. Same package; same import discipline.
- New tests at `tests/unit/services/tts_streaming/test_torch_runtime.py` — mirrors the `tests/unit/services/tts_streaming/test_streaming_mode.py` location.
- New evidence file at `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md` matches the per-story evidence-file pattern (Story 16.7 onward; reaffirmed for 17.1, 17.2, 17.3, 18.1).
- No new top-level packages, modules, or conftest files needed.

**Detected variances:**
- The new module exports a *function with side-effect* (`enable_tf32_and_cudnn_benchmark`) alongside Story 16.2's pure-decision functions. This is a deliberate departure: `streaming_mode.py`'s docstring explicitly says "signal-free, side-effect-free, no metrics emission" — the new module must NOT be added to the same docstring; instead, document at the new module's docstring that the side-effect is deliberate (the alternative — pure probe + caller-side flag set + caller-side metric — would scatter the conditional surface across `main.py` AND a probe module, defeating the testability goal). The conditional logic surface MUST live in exactly one place; the new module is that place.
- The new module is **stateless** — no module-level mutable state. Idempotency is implemented by reading the three `torch.backends.*` values on entry and short-circuiting if all three are already True on an Ampere+ host. This preserves Story 16.2's pure-function discipline (no class-instance state, no module-level booleans) — the only deliberate departure is the side-effect set itself, kept minimal.
- **No D-decision change.** Story 18.2 does not require an architecture amendment per `epics-optimization-pass.md:234` ("No new D-decisions"). NFR1 / NFR3 / NFR7 / D-9 / NFR12 are all preserved unchanged.

### Previous Story Intelligence

**From Story 18.1 closure (commit `956c039` + the producer-bottleneck verdict in the evidence file):**

- Story 18.1 closed as **instrumentation-only** after the Task 1.4 measurement showed a 3.23× steady-state emit-vs-drain ratio with the producer (talker) at 31% real-time. The data ruled out Options 1 + 2 (consumer-side fixes) and named Stories 18.3 (bf16) + 18.4 (`torch.compile`) as the correct fix class for the producer-side throughput defect. **Translation for Story 18.2:** TF32 is a *complementary* speedup that helps the producer-side matmul throughput, but does NOT close the 3.23× ratio by itself — the Epic 18 stub's "10–30%" speedup estimate is consistent with this (TF32 on its own would shrink the ratio to ~2.5–3.0× on a matmul-heavy workload; the rest of the gap closes after 18.3 + 18.4).
- Story 18.1 shipped the env-var-gated CSV capture infrastructure (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV` + `progressive_playback_csv_capture.py`) explicitly as the empirical gate for measuring 18.3 + 18.4 throughput uplifts. **Story 18.2's Task 4 reuses this infrastructure** as the cheapest path to a `first_chunk_latency_ms` measurement. If the existing capture filter doesn't include `first_chunk_latency_ms`, extend it (one-line change) — do NOT build a parallel measurement path.
- Story 18.1 shipped the `01_Run_MyVoice_With_CSV_Capture.bat` convenience launcher. **Reuse this for Task 4** rather than constructing a new launcher.
- Story 18.1's M1 (`AudioChunk.session_id`) and the consumer-side metric session_id threading are NOT touched by Story 18.2 — the new metric (`tf32_cudnn_benchmark_enabled`) is a startup-once event, not a per-chunk metric, so session_id is unused for this story's telemetry.

**From Story 17.3 + Story 16.2 discipline:**

- Story 16.2's `streaming_mode.py` is the canonical Ampere+ guard precedent. The lazy-import-torch-inside-the-probe pattern (so monkeypatch.setattr is honored) is non-negotiable: the test discipline in `test_streaming_mode.py` depends on this, and the new test file at `test_torch_runtime.py` will rely on the same property.
- The `__init__.py` re-export pattern is the "expand the public surface" idiom. Add the two new symbols; do NOT scatter imports across multiple call sites in `main.py`.

**Code-review discipline from `memory/code_review_regression_test_exact_class.md`:**
- HIGH/MEDIUM-fix regression tests must mirror the **exact** bug class, not the nearest adjacent case. Translation for Story 18.2: the highest-risk regression class is **idempotency contract drift** (the second-call discipline is the most-easily-broken AC). The Task 3.4 idempotency test must exercise the second-call no-op branch directly; a test that ONLY exercises the first-call branch and assumes the second is "obvious" would be the wrong fix class for the most likely regression.
- Re-run code-review twice after non-trivial auto-fixes (the established pattern from Stories 16.7 / 16.8 / 17.1 / 17.2 / 17.3 / 18.1).

### References

- **Epic 18 stub** — `_bmad-output/planning-artifacts/epics-optimization-pass.md` lines 1347–1366 (Story 18.2 stub); lines 228–250 (Epic 18 framing); line 244 (risk profile: Low); line 241 (audition discipline: Commander solo for 18.1 + 18.2).
- **Story 18.1 evidence** — `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation-evidence.md` §4.2 (the 3.23× steady-state ratio captured measurement) + §4.4 (the producer-bottleneck verdict naming Stories 18.3 + 18.4 as the fix class). The evidence Story 18.2 composes with.
- **Story 16.2 streaming-mode hardware probe** — `src/myvoice/services/tts_streaming/streaming_mode.py:54-56` and the test patterns at `tests/unit/services/tts_streaming/test_streaming_mode.py`. The structural precedent for the new module + tests.
- **Architecture D-9 hardware-aware default** — `_bmad-output/planning-artifacts/architecture-optimization-pass.md:257`. The `torch.cuda.is_available()` probe + Ampere+ guard discipline that Story 18.2's CPU/pre-Ampere protection relies on.
- **Architecture NFR12 CPU-only support** — `architecture-optimization-pass.md:75 + :808`. CPU-only hosts stay identifiably unchanged; the new module's CPU branch is the AC #2 vehicle for this contract.
- **Architecture NFR1 revised contract** — `architecture-optimization-pass.md` §"NFR1 (revised 2026-05-08, Story 16.9)" at lines 838–850. Per-class first-chunk targets; Story 18.2's Task 4 measures against this baseline.
- **Architecture D-19 telemetry** — `architecture-optimization-pass.md` §"D-19 Telemetry" (begins at line 286) and the `metrics.record(name, value, **tags)` helper specified at line 476. Implementation lives at `src/myvoice/observability/metrics.py:77`. Story 18.2's new `tf32_cudnn_benchmark_enabled` metric extends this established pattern.
- **Story 17.2 evidence** — `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md §4.3.2`. Sarira-F warm-cache baseline first-chunk latencies (3.93–4.94 s) for the AC #5 NFR1 spot-check baseline.
- **Memory: hardware_setup.md** — RTX 5090 Blackwell (compute 10.x) as the dev host; ship-target also covers RTX 30xx (compute 8.6) / RTX 40xx (compute 8.9). All three satisfy the `>= 8` Ampere+ gate.
- **Memory: torch_pyqt6_dll_ordering.md** — the Windows DLL-init invariant that the Task 2 wire-up placement preserves verbatim.
- **Memory: build_tools_phase_perp_state.md** — Phase ⊥-Build closure marker; Story 18.2 is Phase ⊥-Polish-2.
- **Memory: code_review_regression_test_exact_class.md** — exact-bug-class regression-test discipline (Task 3.4's idempotency-reset is the load-bearing application here).
- **Memory: production_release_state.md** — production-bundle context informing Task 6 (bundled-smoke).
- **Memory: epic18_producer_bottleneck_finding.md** — the 3.23× ratio context Story 18.2 composes against (TF32 helps producer matmul; the bigger wins are 18.3 + 18.4).

### Latest Tech Information

**TF32 (TensorFloat-32) on Ampere+ tensor cores:**
- TF32 is a 19-bit mantissa-truncated fp32 representation that PyTorch uses on Ampere+ tensor cores when `torch.backends.cuda.matmul.allow_tf32 = True`. It is **not** a separate dtype — the tensor stays fp32 in memory; the tensor-core matmul internally rounds to TF32. Numerical drift vs strict fp32 is bounded at ~1e-4 round-trip per matmul, which is sub-perceptual for any TTS workload (audio amplitude perception thresholds are several orders of magnitude coarser).
- Default in modern PyTorch (2.0+): `allow_tf32` is `False` for matmul to match strict NumPy fp32 behavior; `True` for cuDNN convolutions. Story 18.2 sets BOTH to `True` for explicit, documented intent and to cover both the matmul and convolution paths in the qwen-tts model.
- Compute-capability gating: TF32 tensor cores are present on Ampere (8.0 / 8.6 / 8.9) + Hopper (9.0) + Blackwell (10.0). Pre-Ampere (Turing 7.5, Volta 7.0, Pascal 6.x) does not have TF32 tensor cores; setting the flag is a no-op on those generations, but the explicit `>= 8` guard documents intent and avoids confusion in `myvoice.log` (the INFO line confirms engagement; the DEBUG line confirms skip).

**`torch.backends.cudnn.benchmark = True`:**
- This enables cuDNN's autotune for convolution algorithm selection: the first call with each unique input shape pays a one-shot autotune cost (typically tens of milliseconds); subsequent calls with the same shape use the cached fastest kernel. For inference workloads with stable input shapes (qwen-tts decode loop has input-shape variation per chunk but converges to a small set), the autotune cost amortizes after a handful of generations.
- **Cold-vs-warm asymmetry:** the first generation after enabling benchmark mode may be marginally slower than baseline (the autotune cost). Task 4.3's N=5 measurement averages this out; Task 4.4's median + p95 is the canonical comparison surface.
- Default: `False`. Story 18.2 sets to `True`.

**`torch.cuda.get_device_capability()` API surface:**
- Returns `tuple[int, int]` — `(major, minor)`. RTX 5090 Blackwell: `(10, 0)`. RTX 4090 Ada: `(8, 9)`. RTX 3090 Ampere: `(8, 6)`. RTX 2080 Turing: `(7, 5)`. The `>= 8` major check is forward-compatible — Hopper (9.0), Blackwell (10.0), and any future architecture with major ≥ 8 all satisfy it.
- Defensive ordering: do NOT call `get_device_capability()` if `cuda.is_available()` is False — on a CPU-only system, the device may not exist; the call may error or warn. The Task 1.3 ordering `if not is_available(): early-out with reason="cuda_unavailable"` is the contract, NOT a polish.

**Idempotent flag set in PyTorch:**
- Setting `torch.backends.cuda.matmul.allow_tf32 = True` twice is safe; the second set is a no-op at the C++ level. The idempotency contract in Story 18.2's AC #1 is about *observable side effects from the wrapper function* (one INFO log per process; one metric record per process), not about the flag-set primitive itself.

**`metrics.record` API for the telemetry breadcrumb:**
- The helper at `src/myvoice/observability/metrics.py:77` accepts `name: str`, `value: float | int | str`, optional keyword `session_id`, and arbitrary `**tags`. Story 18.2's call shape: `metrics.record("tf32_cudnn_benchmark_enabled", 1.0, device_capability="8.9")`. The `value` is `1.0` for engaged, `0.0` for skipped — keeping the metric's listener surface uniform with Story 18.1's three CSV-captured per-chunk metrics (which are all numeric values).
- Listener-exception isolation: per the metrics module's own AC #9 (`metrics.py:144-region`), a raising listener does not propagate. Task 3.6's regression test verifies this property holds for the new metric.

### Project Context Reference

- Project context: `docs/` (existing project-context.md not found; CLAUDE.md absent).
- Working directory invariants per `memory/git_repo_state.md`: V2 is canonical git repo since 2026-05-05; remote = github.com/WreckedMech117/MyVoice; `_bmad-output/` is gitignored (evidence file + CSVs need `git add -f`).
- Production state per `memory/production_release_state.md`: ships publicly via myvoicetts.com as a Windows .exe with bundled portable python310. Installer size unchanged by Story 18.2 (no `requirements.txt` / installer-spec / `build_release.bat` edits).
- Hardware target per `memory/hardware_setup.md`: RTX 5090 Blackwell (compute 10.0) dev host; ship-target covers RTX 30xx (compute 8.6) / RTX 40xx (compute 8.9). All three satisfy the new probe's `>= 8` Ampere+ gate.
- Phase context per `memory/build_tools_phase_perp_state.md`: Phase ⊥-Polish-2 is the successor to Phase ⊥-Polish (Story 17.3 closed the progressive-playback contract); Story 18.2 is the second story of Phase ⊥-Polish-2 (after Story 18.1 shipped the instrumentation that pinned the producer bottleneck).

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m]

### Debug Log References

- Test runs executed via `./python310/python.exe -m pytest ...` per
  `memory/torch_pyqt6_dll_ordering.md` (DLL ordering enforced by
  `tests/conftest.py`).
- 2026-05-09: Story 18.1 test `test_only_targeted_metrics_are_captured`
  asserted `first_chunk_latency_ms` was filtered OUT. Updated alongside
  Task 4.1 source change so the assertion matches the new contract; new
  `test_first_chunk_latency_row_columns_match_header` test pins the row
  layout for the newly-captured metric (chunk_index / is_final /
  audio_data_size columns are blank for `first_chunk_latency_ms` rows).

### Completion Notes List

**Source-tree implementation closed (Tasks 1, 2, 3, 4.1, 7).**
Tasks 4.2–4.5 (NFR1 measurement on RTX 5090), Task 5 (NFR3 spot-check),
and Task 6 (bundled smoke after `build_release.bat`) require Commander
on the RTX 5090 / Win11 ship-target host. Evidence-file scaffolding for
those tasks lives at
`_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md`
with templated tables ready for Commander's captured data.

**Locked decisions (per OQ resolution):**
- OQ #1 (wire-up placement) → **inside `main()` after `setup_logging()`**.
  INFO breadcrumb lands in `myvoice.log` via the file handler.
- OQ #2 (cuda-unavailable tag schema) → **`device_capability="none"`
  string sentinel**. Schema uniformity for downstream parsers.
- OQ #3 (out-of-range speedup) → **route to Commander** rather than
  dev-agent interpretation. Evidence-file §4.4 has the tabulated
  surface ready for Commander's call.
- OQ #4 (build-counter increment files) → **out-of-scope** to this
  story's commit per Story 18.1 precedent. Commander handles them in a
  separate build-state commit when bundled smoke completes.

**Why the 15 new test_torch_runtime tests instead of 6–8:** the spec
estimated 6–8 tests; the parametrized tag-schema test (Task 3.5)
expanded to 5 parametrized rows covering Ampere 8.9, Blackwell 10.0,
Hopper 9.0, pre-Ampere Turing 7.5, and cuda-unavailable. The
idempotency contract (Task 3.4) split into two scenarios
(pre-set-True + back-to-back-from-cold) because the bug class is
"second call observable side-effect," and both scenarios stress
distinct mutation paths through the function.

**Test sweep totals:**
- Task 7.1–7.4 sweep: 86/86 PASS
- Task 7.5 broader sweep: 152/152 PASS (159 unique tests; some overlap
  between sweeps — net new from this story = 15 + 2 = 17 tests)
- Combined unique new tests in this story: 17 (15 in test_torch_runtime
  + 2 in test_progressive_playback_csv_capture)

### File List

**New (source tree):**
- `src/myvoice/services/tts_streaming/torch_runtime.py`

**Edit (source tree):**
- `src/myvoice/services/tts_streaming/__init__.py` (re-export the two new symbols)
- `src/myvoice/main.py` (wire-up inside `main()` after `setup_logging()`)
- `src/myvoice/observability/progressive_playback_csv_capture.py`
  (Task 4.1: add `first_chunk_latency_ms` to `_CAPTURED_METRIC_NAMES`;
  docstring updated)

**New (tests):**
- `tests/unit/services/tts_streaming/test_torch_runtime.py` (15 tests)

**Edit (tests):**
- `tests/unit/observability/test_progressive_playback_csv_capture.py`
  (`test_only_targeted_metrics_are_captured` updated to capture
  `first_chunk_latency_ms`; new `test_first_chunk_latency_row_columns_match_header`)

**New (evidence + measurement artifacts):**
- `_bmad-output/implementation-artifacts/18-2-tf32-cudnn-benchmark-enable-evidence.md`
  (force-add per Story 16.9 / 17.1 / 17.2 / 17.3 / 18.1 evidence-file precedent)
- (pending Commander) `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv`
- (pending Commander) `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-off.csv`

**Sprint-status update:**
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
  (`18-2-tf32-cudnn-benchmark-enable: ready-for-dev` → `in-progress`;
  to be flipped to `review` at story closure after Commander's Tasks
  4.2–4.5 / 5 / 6 land)

**Out-of-scope uncommitted (per Story 18.1 precedent + OQ #4):**
- `build_tools/installer.iss` (build counter — pre-existing modification at conversation start)
- `build_tools/version.py` (build counter — pre-existing modification at conversation start)
  These will be re-incremented by Task 6's `build_release.bat`. Not
  Story 18.2 feature work; Commander handles them in a separate
  build-state commit.

<!--
Dev agent: when populating the File List on closure, follow the Story 18.1 precedent
of explicitly segregating in-scope edits from out-of-scope uncommitted files. At
conversation start, `git status` showed two pre-existing modified files left over
from prior bundle-verification runs:
  - `build_tools/installer.iss` (build counter increment)
  - `build_tools/version.py` (build counter increment)
Story 18.1's File List flagged these as "out-of-scope uncommitted files surfaced by
git status and noted (not Story 18.1 source-tree edits)". Task 6 (bundled smoke) will
likely re-increment these. Either roll the increments into this story's commit (with
a clear note that the increments are build-counter-only, not feature work) OR call
them out under the same "out-of-scope" header Story 18.1 used and let Commander
decide. Do NOT silently include them in the main commit body without a callout.
-->

## Open Questions for Dev Agent (deferred per workflow guidance)

> All four OQs resolved at story start; see "Locked decisions" in Dev
> Agent Record → Completion Notes List + §1 of the evidence file. Kept
> verbatim below for audit trail.

1. **Wire-up placement choice (Task 2 + AC #4):** module-level (immediately after the `import torch` block at `main.py:42-49`) vs. inside `main()` (immediately after `setup_logging()` returns at `main.py:334-region`). Module-level is structurally simpler but the INFO log from the enable function fires before `setup_logging()` configures the file handler — it would land on stderr only. Inside `main()` after `setup_logging()` is the cleaner placement for the log-line breadcrumb to land in `myvoice.log`. **Recommended:** inside `main()` immediately after `setup_logging()` returns. Confirm with Commander before locking the placement.
   **RESOLVED 2026-05-09**: inside `main()` after `setup_logging()` (recommended path confirmed by Commander).

2. **`device_capability` tag schema for the cuda-unavailable branch (AC #3 + Task 3.5):** Should the metric tag be `device_capability="none"` (string sentinel) or omit the tag entirely on the cuda-unavailable branch? **Recommended:** include the tag with value `"none"` for schema uniformity (downstream parsers see the same tag set on every record); document the sentinel at the function docstring. Confirm with Commander if a different sentinel (e.g., `"cpu"` or `null`) is preferred.
   **RESOLVED 2026-05-09**: include `device_capability="none"` (recommended sentinel confirmed by Commander).

3. **Out-of-range first-chunk-latency speedup (Task 4.4 + AC #5):** if the measured median speedup on the RTX 5090 falls outside the Epic 18 stub's anticipated [10%, 30%] range, route to Commander rather than dev-agent interpretation. Below 10%: probably "ship anyway because the cost is zero" but Commander confirms. Above 30%: probably "great, document and continue" but worth confirming the measurement isn't confounded by warm-vs-cold cache, GPU thermal state, qwen_tts model variant, or which Sarira-F quality cache state from Story 17.2 is in play. Surface the captured data + a confounder-checklist for Commander to act on.
   **POLICY LOCKED 2026-05-09**: route to Commander (no autonomous interpretation). Evidence-file §4.4 ready for Commander's number; will be revisited when Tasks 4.2–4.5 land.

4. **Build-counter increment files (`build_tools/installer.iss` + `build_tools/version.py`):** these were already modified at conversation start (leftover from prior `build_release.bat` runs); Task 6's bundled smoke will re-increment them. Roll into this story's commit with a "build-counter-only, not feature work" note, OR keep them out-of-scope per Story 18.1's precedent? **Recommended:** out-of-scope per the Story 18.1 precedent — they aren't Story 18.2 feature work; let Commander handle them in a separate build-state commit when the bundled smoke completes. Confirm before the closing commit.
   **POLICY LOCKED 2026-05-09**: out-of-scope per Story 18.1 precedent (recommended path). The two files stay un-touched by this story's commits; Commander rolls a separate build-state commit when bundled smoke (Task 6) completes.
