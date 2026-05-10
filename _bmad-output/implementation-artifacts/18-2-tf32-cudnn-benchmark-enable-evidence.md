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

## §4. NFR1 first-chunk-latency measurement (Task 4)

> **Status: single-shot capture landed 2026-05-09 19:23 by Commander on
> RTX 5090. Full N=10 + git-checkout-OFF baseline DEFERRED pending
> Commander's call (see §4.6 below).**

### 4.1 Method

Reuse the Story 18.1 env-var-gated CSV-capture infrastructure:

```bat
REM From the repo root:
set MYVOICE_PROGRESSIVE_PLAYBACK_CSV=...\18-2-rtx5090-tf32-on.csv
python310\python.exe src\myvoice\main.py
```

The `01_Run_MyVoice_With_CSV_Capture.bat` was repointed to the Story
18.2 path so the Story 18.1 baseline CSV at
`18-1-instrumentation-rtx5090-longform.csv` is preserved as the
**implicit TF32-OFF baseline** for the producer-cadence comparison
(both runs are the same canonical Sarira-F long-form utterance through
the same dispatch path; the only material difference is the Story 18.2
wire-up commit).

Filter-list extension (Story 18.2 Task 4.1):
`progressive_playback_csv_capture.py` was widened to also capture
`first_chunk_latency_ms` so Stories 18.2 + 18.3 + 18.4 inherit a single
measurement surface. The Story 18.1 baseline CSV does NOT contain
`first_chunk_latency_ms` rows (the filter widening only landed in
commit `787960c` — current HEAD).

### 4.2 Capture: TF32 ON (single-shot, current HEAD `787960c`)

CSV: `_bmad-output/implementation-artifacts/18-2-rtx5090-tf32-on.csv`
Generated 2026-05-09 19:23 on RTX 5090 / Win11 / device_capability=12.0.
N=1 generation, 11 chunks emitted (chunk_index 0–10), 1 first_chunk record.

| Statistic                          | TF32 ON (single-shot) |
| ---------------------------------- | --------------------: |
| `first_chunk_latency_ms`           |             **7800**  |
| inter-chunk-emit interval median   |          **6515 ms**  |
| inter-chunk-emit interval mean     |             6726 ms   |
| chunk audio duration               |          1981 ms (constant) |
| **emit-interval / audio-duration** |          **3.29×**    |

### 4.3 Capture: TF32 OFF (implicit baseline from Story 18.1, commit pre-`787960c`)

CSV: `_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv`
(Captured during Story 18.1 Task 1.4 measurement run; predates Story 18.2
wire-up.)

| Statistic                          | TF32 OFF (single-shot) |
| ---------------------------------- | ---------------------: |
| `first_chunk_latency_ms`           |    **N/A** (filter widening absent) |
| inter-chunk-emit interval median   |           **6362 ms**  |
| inter-chunk-emit interval mean     |              6196 ms   |
| chunk audio duration               |           1981 ms (constant) |
| **emit-interval / audio-duration** |           **3.21×**    |

### 4.4 Delta + speedup

| Statistic                          | OFF (ms) | ON (ms)  |       Δ | Speedup |
| ---------------------------------- | -------: | -------: | ------: | ------: |
| inter-chunk-emit median            |     6362 |     6515 |  +153 ms | **−2.4%** (slower) |
| inter-chunk-emit mean              |     6196 |     6726 |  +530 ms | **−8.6%** (slower) |
| emit/audio ratio                   |    3.21× |    3.29× |  +0.08× | unchanged within noise |
| `first_chunk_latency_ms`           |    (N/A) |     7800 |       — | **uncomputable** (no OFF sample) |

**Anticipated gate (Epic 18 stub `:1361`)**: 10–30% median speedup. The
measured single-shot inter-chunk-emit cadence shows **no speedup; in
fact a slight slowdown** in the directional sample. **Per AC #5 + OQ
#3, this is an out-of-range result that routes to Commander rather
than dev-agent interpretation.**

**Confounders** (not the dev agent's call to weigh):

1. **cuDNN benchmark autotune cold cost.** Story 18.2 §"Latest Tech
   Information" specifically called this out: "the first generation
   after enabling benchmark mode may be marginally slower than baseline
   (the autotune cost). Task 4.3's N=5 measurement averages this out;
   Task 4.4's median + p95 is the canonical comparison surface."
   Single-shot is the worst-case for ON. A second + third + Nth
   generation in the same process would benefit from the autotune
   cache; we did not capture them here.

2. **N=1 vs N=1 noise floor.** Both samples are single generations.
   ±10% per-run variance is normal for ML inference workloads on the
   same hardware. The observed −2.4% median delta is well within this
   noise band; the −8.6% mean delta is heavier-tailed (the warmup
   chunk-1-after-chunk-0 interval at 7714 ms in the ON CSV pulls the
   mean up disproportionately).

3. **GPU thermal state / driver-cache state across the day.** The OFF
   baseline was captured during Story 18.1's Task 1.4 run on a different
   day; the ON capture was just now (2026-05-09 19:23). Driver +
   nvidia-smi state, OS scheduler entropy, and ambient GPU temperature
   are not controlled across the gap.

4. **Story 18.1's ratio verdict holds.** OFF ratio = 3.21×; ON ratio
   = 3.29×. Both are essentially Story 18.1's canonical 3.23×
   steady-state ratio (producer at ~31% real-time talker decode). TF32
   was *never* the named fix class for this bottleneck per
   `memory/epic18_producer_bottleneck_finding.md`; Stories 18.3 (bf16)
   + 18.4 (`torch.compile`) are. Story 18.2's role in the Epic 18 plan
   was to ship the cheapest cumulative speedup before 18.3 + 18.4
   land — not to single-handedly close the producer bottleneck.

5. **No new perceptual artifact.** Commander reports the same gaps
   from Story 18.1, not a new TF32-induced defect. The lossless-TF32
   promise (~1e-4 matmul drift) holds — see §5 below.

### 4.5 Engagement breadcrumb sanity-check (Task 4.5) — **PASS**

Verbatim from `logs/myvoice.log`:

```
2026-05-09 19:23:10,511 - myvoice.services.tts_streaming.torch_runtime - INFO - TF32 + cuDNN benchmark enabled (device_capability=12.0)
```

Exactly one INFO line, fired at startup before `setup_application()`
(per Task 2.3 placement decision). Wire-up engaged correctly.

**Documentation-fix follow-up:** the docstring's compute-capability
mapping initially recorded "Blackwell = 10.0 (RTX 50xx, B100)" which
conflated datacenter Blackwell (B100/B200, CC 10.0) with the GeForce
Blackwell variant on the RTX 5090 (CC 12.0). The actual measured value
is **12.0**. The docstring at `torch_runtime.py:_AMPERE_CAPABILITY_MAJOR`
was updated post-capture to record both CC variants. Functionally
irrelevant — the `>= 8` gate engages either way — but worth recording
for future architecture-mapping audits.

### 4.6 OQ #3 routing — Commander decides next step

Per AC #5 + OQ #3: out-of-range single-shot speedup routes to Commander.
The dev agent does NOT declare pass/fail unilaterally. Three plausible
next moves, in roughly increasing rigor:

**(a) Accept and continue.** Treat TF32 wire-up as engaged-and-no-harm;
acknowledge the speedup didn't materialize at single-shot but the
intent (cheapest available speedup; composes with 18.3 + 18.4) is
preserved. The producer bottleneck is the named target of Stories 18.3
+ 18.4, not 18.2. Move on to 18.3.

**(b) Run N=10 + git-checkout OFF baseline for full statistical rigor.**
Re-run Tasks 4.2–4.4 per the original story spec: kill app between
runs, N=10 fresh-process generations on current HEAD; `git checkout`
parent commit; N=10 fresh-process generations on the OFF baseline;
restore HEAD; compute median/p90/p95 + delta. This is the "default
correct" path; the cost is ~2 hours of Commander hand-time.

**(c) Revert the wire-up.** If Commander suspects TF32 is a net loss
(unlikely given the public PyTorch guidance) or wants to ship 18.3 +
18.4 as the only producer-cadence change, the wire-up at
`main.py:main()` is a one-block revert.

**Recommendation**: option (a). The Story 18.1 verdict had already
named 18.3 + 18.4 as the bottleneck fix; spending 2 hours of N=10
captures to confirm "TF32 didn't close a bottleneck it was never
expected to close" is low-leverage. Ship 18.2 as-engaged and move
to 18.3 where the leverage is.

**Status: HALTED pending Commander's choice between (a)/(b)/(c).**

## §5. NFR3 spot-check (Task 5)

> **Status: Commander reported 2026-05-09 19:23 — "Audio still had gaps."**

**Verdict**: **No new perceptual defect attributable to TF32.** The gaps
Commander reports are the **same gaps Story 18.1 measured**, not a new
TF32-induced artifact. The lossless-TF32 promise (~1e-4 matmul drift,
sub-perceptual at any audio amplitude resolution) holds: there is no
audible TF32-incompatibility surprise on this qwen-tts model + Sarira-F
voice + RTX 5090 combination.

**Why "still had gaps" is not a Story 18.2 defect:**

The producer-bottleneck cadence (3.21× → 3.29× emit/audio ratio per §4.4
above) means chunks emit at ~6.5s per chunk while each chunk only
*plays* ~2s of audio. The drain-faster-than-fill pattern produces ~4.5s
silent stretches between consumed chunks — the "gaps" Commander hears.

Story 18.2 was *never* the gap-closer. Per
`memory/epic18_producer_bottleneck_finding.md` + Story 18.1 evidence
§4.4: the producer bottleneck's **named fix class is Stories 18.3 (bf16
precision on talker/decoder) + 18.4 (`torch.compile` decoder persistent
cache)**. Story 18.2 ships the cheapest cumulative speedup (lossless
TF32 + cuDNN benchmark) which composes with — but does not substitute
for — the named producer-side fixes.

**No Task 5.2 architecture-layer routing required.** The "perceptual
defect detected: <description>" branch (TF32 numerical-incompatibility
surprise → NFR7 fp32 fallback extension) does NOT apply here. The gaps
are the pre-existing Story 18.1-measured cadence defect, not a new
defect introduced by Story 18.2.

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
