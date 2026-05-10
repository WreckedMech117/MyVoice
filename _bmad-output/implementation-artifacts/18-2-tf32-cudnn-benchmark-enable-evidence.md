# Story 18.2 Evidence — TF32 + cuDNN Benchmark Enable (Phase ⊥-Polish-2)

This file is the empirical record for Story 18.2's Tasks 4 (NFR1
first-chunk-latency measurement), 5 (NFR3 spot-check), and 6 (bundled
smoke). Force-added per the Story 16.9 / 17.1 / 17.2 / 17.3 / 18.1
evidence-file precedent (`_bmad-output/` is gitignored per
`memory/git_repo_state.md`).

## §1. Locked decisions

Captured at story start so a future reviewer can audit the contract
against the implementation without re-reading the conversation log.

### 1.1 Wire-up placement (OQ #1)

**Decision**: inside `main()` immediately after `setup_logging()` returns
and before `setup_application()` is called. This places the INFO
breadcrumb in `myvoice.log` via the file handler `setup_logging()`
configures, rather than stderr-only.

**Implementation**: `src/myvoice/main.py` — see the `# Story 18.2:`
guarded block immediately after `setup_logging()` in `main()`.

### 1.2 `device_capability` tag schema (OQ #2)

**Decision**: include the tag with value `"none"` (string sentinel) on
the cuda-unavailable branch for schema uniformity. Downstream parsers
(Story 18.1's CSV-capture infrastructure stringifies tag values) see the
same key set on every record.

**Implementation**: `src/myvoice/services/tts_streaming/torch_runtime.py`
— `_device_capability_str(None) == "none"`.

### 1.3 OQ #3 (out-of-range speedup) policy

If the captured median speedup (Task 4.4) falls outside the Epic 18
stub's anticipated [10%, 30%] range, the dev agent surfaces the data to
Commander rather than declaring pass/fail. See §4.4 below for the actual
captured number and the routing decision.

### 1.4 OQ #4 (build-counter increment files)

`build_tools/installer.iss` + `build_tools/version.py` were already
modified at conversation start (leftover from prior `build_release.bat`
runs). Per Story 18.1's precedent, treated as **out-of-scope** to this
story's commit. Task 6's bundled smoke will re-increment them; Commander
handles them in a separate build-state commit.

## §2. Module + wire-up + tests landed (Tasks 1, 2, 3, 7)

Source-tree implementation closed before Commander engaged the empirical
gates (Tasks 4, 5, 6). All test runs below executed via
`./python310/python.exe -m pytest`.

### 2.1 New module

- `src/myvoice/services/tts_streaming/torch_runtime.py` — ~95 LOC
  including docstring + module constants + `is_ampere_or_newer()` +
  `_device_capability_str()` + `_all_three_flags_already_true()` +
  `enable_tf32_and_cudnn_benchmark()`. Mirrors Story 16.2's
  `streaming_mode.py` lazy-torch-import + no-peer-imports discipline.
- `src/myvoice/services/tts_streaming/__init__.py` — widened to
  re-export `enable_tf32_and_cudnn_benchmark` + `is_ampere_or_newer`.

### 2.2 Wire-up

- `src/myvoice/main.py` — single guarded import + call inside `main()`
  immediately after `setup_logging()` returns and before
  `setup_application()`. Wrapped in `try / except Exception` so a
  partial-install missing the new module logs a WARNING and continues
  startup (the speedup is opt-in; absence is the V2 baseline).

### 2.3 CSV-capture filter extension (Task 4.1)

- `src/myvoice/observability/progressive_playback_csv_capture.py` —
  `_CAPTURED_METRIC_NAMES` widened to include `first_chunk_latency_ms`.
  Backwards-compatible: chunk-specific tag columns
  (`chunk_index` / `is_final` / `audio_data_size`) are blank for these
  rows; downstream analysis distinguishes by `metric_name`.
- Module docstring updated to enumerate the four captured metrics +
  cite Story 18.2 Task 4.1.

### 2.4 Test suites — all green

| Suite                                                                         | Tests | Result |
| ----------------------------------------------------------------------------- | ----: | :----: |
| `tests/unit/services/tts_streaming/test_torch_runtime.py` (NEW)               |    15 |  PASS  |
| `tests/unit/services/tts_streaming/test_streaming_mode.py` (16.2 regression)  |    16 |  PASS  |
| `tests/unit/observability/test_progressive_playback_csv_capture.py` (+2 new)  |    14 |  PASS  |
| `tests/unit/test_app_progressive_playback_instrumentation.py` (18.1)          |     6 |  PASS  |
| `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py`    |     3 |  PASS  |
| `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` (17.3)    |     3 |  PASS  |
| `tests/unit/test_app_progressive_playback.py` (17.3)                          |    23 |  PASS  |
| `tests/unit/test_app_progressive_playback_cancel.py` (17.3)                   |     3 |  PASS  |
| `tests/integration/test_progressive_playback_dispatch_skip.py` (17.3)         |     3 |  PASS  |
| **Story 18.2 + 18.1 + 17.3 + 16.2 sweep (Task 7.1–7.4)**                      |    86 |  PASS  |
| `tests/unit/services/test_qwen_tts_service_dispatch.py` (16.6)                |    39 |  PASS  |
| `tests/unit/services/test_qwen_tts_service_session_integration.py` (11.4)     |    19 |  PASS  |
| `tests/unit/observability/test_metrics.py` (11.3)                             |    45 |  PASS  |
| **Broader sweep (Task 7.5)**                                                  |   152 |  PASS  |

Zero regressions across either sweep. The Task 1.4 `__init__.py` widening
and the Task 4.1 CSV-capture filter extension did not touch any existing
test surface.

### 2.5 The 15 new test_torch_runtime.py tests

Hardware truth-table coverage (AC #1 + #2 + #7):
1. `is_ampere_or_newer` returns True on Ampere CUDA
2. enable engages on Ampere + sets all three flags + returns engaged dict
3. enable logs INFO breadcrumb on engagement (Blackwell 10.0)
4. `is_ampere_or_newer` returns False on Turing 7.5
5. enable skips on pre-Ampere and does NOT mutate flags
6. `is_ampere_or_newer` returns False when cuda unavailable
7. enable skips on cuda unavailable AND does NOT call `get_device_capability`
   (defensive ordering — pre-Ampere CPU systems may not even have CUDA installed)

Idempotency contract (Task 3.4 — load-bearing per
`memory/code_review_regression_test_exact_class.md`):
8. second call with all flags already True → no INFO, no metric re-emit, DEBUG only
9. two back-to-back calls (cold start) → exactly one metric record total

Telemetry tag schema (Task 3.5, parametrized):
10–14. Ampere 8.9, Blackwell 10.0, Hopper 9.0, pre-Ampere 7.5,
    cuda-unavailable — `device_capability` tag is always a string;
    `none` sentinel for cuda-unavailable; `reason` tag absent on
    engaged branch.

D-19 listener-isolation (Task 3.6):
15. enable still sets all flags + returns engaged when a registered
    metrics listener raises (the function MUST NOT abort startup).

## §3. Wire-up placement (Task 2.3)

Per §1.1 above. The `main.py` ordering is:

1. `setup_logging()` — file handler attached to `myvoice.log`
2. `logger = logging.getLogger(__name__)`
3. `logger.info("Starting MyVoice V2 application with qasync event loop")`
4. **(NEW)** `enable_tf32_and_cudnn_benchmark()` — guarded
5. `exception_handler.install()` — Story 7.6
6. `setup_application()` — QApplication construction
7. `QEventLoop(qt_app)` — qasync event loop
8. `loop.run_until_complete(async_main(qt_app, logger))`

The torch-before-PyQt6 DLL ordering invariant per
`memory/torch_pyqt6_dll_ordering.md` is preserved verbatim: the new code
touches only the new `myvoice.services.tts_streaming.torch_runtime`
module (which lazy-imports torch inside the probe) and the standard
library `logging` module. No PyQt6 import. The existing PyQt6 import
block at `main.py:51-54` runs at module-load time, before `main()` —
which means torch was already imported at module level (`main.py:42-49`)
before PyQt6, satisfying the DLL invariant. The new wire-up inside
`main()` is a no-op for that ordering.

## §4. NFR1 first-chunk-latency measurement (Task 4) — COMMANDER

> **Status: pending Commander run on RTX 5090 dev host.**

### 4.1 Method

Reuse the Story 18.1 env-var-gated CSV-capture infrastructure:

```bat
REM From the repo root:
set MYVOICE_PROGRESSIVE_PLAYBACK_CSV=18-2-rtx5090-tf32-on.csv
python310\python.exe -m myvoice.main
```

(Or use `01_Run_MyVoice_With_CSV_Capture.bat` from Story 18.1 with the
target filename overridden.)

For each capture, run the canonical Story 17.3 §4.1 step 3 long-form
CLONED utterance: **Sarira-F voice**, paragraph ≥ 250 characters /
~22 s of speech. Each generation must be a fresh process launch so
cuDNN benchmark autotune cache state does not bleed across runs. **N=10
generations per condition.**

### 4.2 Capture: TF32 ON (after / current HEAD)

CSV: `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv`

| Statistic | first_chunk_latency_ms |
| --------- | ---------------------: |
| median    |                        |
| p90       |                        |
| p95       |                        |

(Commander fills in after capture.)

### 4.3 Capture: TF32 OFF (before / parent commit)

```bash
git checkout <parent-of-tf32-wireup-commit>
```

Re-run the same N=10 capture; output to
`_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-off.csv`. Then
`git checkout` back to the wire-up commit. The git-commit-pair
methodology eliminates the risk of forgetting a source-tree revert.

| Statistic | first_chunk_latency_ms |
| --------- | ---------------------: |
| median    |                        |
| p90       |                        |
| p95       |                        |

### 4.4 Delta + speedup

| Statistic | TF32 OFF (ms) | TF32 ON (ms) | Δ (ms) | Speedup |
| --------- | ------------: | -----------: | -----: | ------: |
| median    |               |              |        |         |
| p90       |               |              |        |         |
| p95       |               |              |        |         |

**Anticipated gate (Epic 18 stub `:1361`)**: 10–30% median speedup on
RTX 5090. Per AC #5 + OQ #3: if the measured median falls outside that
range, surface the data here and route to Commander rather than declare
pass/fail unilaterally.

### 4.5 Engagement breadcrumb sanity-check

Confirm `myvoice.log` from the **TF32 ON** runs contains exactly one
`"TF32 + cuDNN benchmark enabled (device_capability=10.0)"` INFO line per
process launch, and the **TF32 OFF** runs contain none (the parent
commit predates the wire-up).

## §5. NFR3 spot-check (Task 5) — COMMANDER

> **Status: pending Commander A/B audition on RTX 5090.**

Per Epic 18 stub `:241` + `:1363`: Commander solo, no L2/L3 recruitment.
The TF32 matmul drift is well-known to be sub-perceptual at ≤ 1e-4; the
spot-check is documentary diligence, not a gate.

**Procedure**: A/B-listen back-to-back to one TF32-OFF and one TF32-ON
audio file from §4 above (do not regenerate; the audio Tasks 4.2 + 4.3
already produced is the audition material).

**Verdict**: _________________________________________________

(Expected literal: `"no perceptual defect detected"`. Any other verdict
triggers a Commander → architecture-layer routing per Task 5.2 — likely
a TF32-incompatibility surprise that Story 18.3's NFR7 fp32 fallback
machinery would need to extend to cover, which is explicitly out of
scope for this story.)

## §6. Bundled smoke (Task 6) — COMMANDER

> **Status: pending Commander run after `build_release.bat` cycle.**

### 6.1 Build cycle

```bat
build_tools\build_release.bat
```

Confirms `build_tools/dist/MyVoice/MyVoice.exe` includes the new module.
Per `memory/build_tools_phase_perp_state.md` + `memory/production_release_state.md`,
this is the standard production-bundle build cycle.

### 6.2 Bundled-mode INFO breadcrumb

Launch `build_tools/dist/MyVoice/MyVoice.exe` on the RTX 5090 ship-target
host. Confirm `myvoice.log` (in the portable Logs path per
`setup_logging()` discipline) contains the single INFO-level line:

```
TF32 + cuDNN benchmark enabled (device_capability=10.0)
```

**Verbatim log excerpt** (Commander pastes after capture):

```
<paste here>
```

### 6.3 Short-class TTS round-trip in bundled exe

Execute the Story 17.3 §4.1 step 1 short paragraph in the bundled exe.
Confirm:
- No new error / warning / unexpected log line around the TF32 wire-up.
- TTS generation completes successfully end-to-end (audio plays).

**Verdict**: _________________________________________________

### 6.4 PyInstaller hidden-imports check

If §6.2 reveals the new module fails to import in the
PyInstaller / portable-bundle environment, the most likely cause is
`MEIPASS`-path invisibility — a one-line fix to PyInstaller's
hidden-imports list at `build_tools/`. Surface to Commander; do NOT
absorb. (Expected: no such failure — the new module is reachable
through normal imports of `myvoice.services.tts_streaming` which is
already in the bundle.)

## §7. Post-implementation accounting (Task 8)

### 7.1 First-chunk-latency baseline note for Stories 18.3 + 18.4

Per AC #5: the captured `metrics.first_chunk_latency_ms` value Stories
18.3 + 18.4 baseline against is the **post-18.2 number** (TF32 ON), not
the pre-18.2 number. Stories 18.3 + 18.4 should reuse the same Task 4.1
CSV-capture filter extension (it already covers `first_chunk_latency_ms`)
rather than re-extend it.

### 7.2 No architecture amendment

Per Epic 18 framing `:234` ("No new D-decisions"): D-9, NFR1, NFR3,
NFR7, NFR12 all preserved verbatim by Story 18.2.
`_bmad-output/planning-artifacts/architecture-optimization-pass.md` is
**not** amended.

### 7.3 No build-pipeline change

Per Epic 18 framing `:248` ("18.1–18.3 are pure source-tree edits; no
`requirements.txt` / installer-spec / `build_release.bat` changes
anticipated"): no edits to `requirements.txt`, installer spec, or build
scripts. Pure source-tree edits picked up by the next build cycle.
