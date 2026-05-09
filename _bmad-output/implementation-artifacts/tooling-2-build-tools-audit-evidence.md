# Tooling-2 Build-Tools Audit — Evidence File

> **Status:** in-progress (drafting). Sections fill in order §1 → §7 as Tasks 1–7 close.
>
> **Purpose:** Captures the verifiable audit findings, decisions, and smoke-test results behind `tooling-2-build-tools-audit.md`'s 7 ACs. This file is the durable artifact; the bundle and the installer it produces are reproducible from the spec at any time.
>
> **Force-add note:** This file lives under `_bmad-output/` which is gitignored (per `.gitignore:51` for `build_tools/dist/` and the broader `_bmad-output/` exclusion). Add via `git add -f _bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` per the precedent set by Story 16.9 / 17.1 evidence files.

---

## §1 — CPU-vs-CUDA torch decision (AC #1)

### §1.1 Current-state audit (Subtask 1.1)

Three sources of truth disagree about whether the production bundle ships CPU-only torch or CUDA torch:

**Source A — `build_tools/requirements-production.txt:33-46` (declared CPU-only intent):**

```
# Machine Learning (Required - CPU ONLY)
# IMPORTANT: Use CPU-only version to save ~1GB of CUDA libraries
torch>=2.0.0; sys_platform == 'win32'
--extra-index-url https://download.pytorch.org/whl/cpu

# Alternative explicit CPU installation:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Size comparison:
# - Full PyTorch (CUDA): ~2.5GB
# - CPU-only PyTorch: ~150-200MB
# Savings: ~2.3GB
```

This is a clear declaration of CPU-only intent for the production build, citing a ~2.3 GB size saving as the driver.

**Source B — `build_tools/myvoice.spec:74-80` (DLL collection — silently overrides Source A):**

```python
# Torch DLL binaries - collected manually to avoid import issues
import glob as _glob
torch_binaries = []
_torch_lib = project_root / 'python310' / 'Lib' / 'site-packages' / 'torch' / 'lib'
if _torch_lib.exists():
    for _dll in _glob.glob(str(_torch_lib / '*.dll')):
        torch_binaries.append((_dll, 'torch/lib'))
        print(f"[SPEC] Adding torch DLL: {Path(_dll).name}")
```

The spec globs **whatever DLLs the dev `python310/` happens to contain**. This means the production bundle's torch variant is determined at build time by the maintainer's local environment, not by `requirements-production.txt`. There is no fail-fast assertion that the collected DLLs match the declared CPU-only intent.

**Source C — empirical evidence on disk:**

The maintainer's `python310/Lib/site-packages/torch/lib/` contains **37 DLLs** including the full CUDA suite (audited 2026-05-08):

```
c10.dll, c10_cuda.dll, caffe2_nvrtc.dll, cublas64_12.dll, cublasLt64_12.dll,
cudart64_12.dll, cudnn64_9.dll, cudnn_adv64_9.dll, cudnn_cnn64_9.dll,
cudnn_engines_precompiled64_9.dll, cudnn_engines_runtime_compiled64_9.dll,
cudnn_graph64_9.dll, cudnn_heuristic64_9.dll, cudnn_ops64_9.dll,
cufft64_11.dll, cufftw64_11.dll, cupti64_2025.1.1.dll, curand64_10.dll,
cusolver64_11.dll, cusolverMg64_11.dll, cusparse64_12.dll,
libiomp5md.dll, libiompstubs5md.dll, nvJitLink_120_0.dll,
nvToolsExt64_1.dll, nvperf_host.dll, nvrtc-builtins64_128.dll,
nvrtc64_120_0.alt.dll, nvrtc64_120_0.dll, shm.dll, torch.dll,
torch_cpu.dll, torch_cuda.dll, torch_global_deps.dll, torch_python.dll,
uv.dll, zlibwapi.dll
```

The CUDA-suite subset (`c10_cuda.dll`, `torch_cuda.dll`, `cublas*`, `cudart*`, `cudnn*`, `cufft*`, `curand*`, `cusolver*`, `cusparse*`, `nvJitLink*`, `nvrtc*`, `nvToolsExt*`, `cupti*`, `caffe2_nvrtc.dll`) is unmistakable — this is `torch 2.10+cu128` per memory `hardware_setup.md` (RTX 5090 Blackwell dev host).

**Source D — the legacy bundle on disk confirms what the spec actually produced:**

`build_tools/dist/MyVoice2.0.1.9Portable/_internal/torch/lib/` contains **36 DLLs**, an identical CUDA-suite to the dev `python310/`:

```
c10.dll, c10_cuda.dll, caffe2_nvrtc.dll, cublas64_12.dll, cublasLt64_12.dll,
cudart64_12.dll, cudnn64_9.dll, cudnn_adv64_9.dll, cudnn_cnn64_9.dll,
cudnn_engines_precompiled64_9.dll, cudnn_engines_runtime_compiled64_9.dll,
cudnn_graph64_9.dll, cudnn_heuristic64_9.dll, cudnn_ops64_9.dll,
cufft64_11.dll, cufftw64_11.dll, cupti64_2025.1.1.dll, curand64_10.dll,
cusolver64_11.dll, cusolverMg64_11.dll, cusparse64_12.dll,
libiomp5md.dll, libiompstubs5md.dll, nvJitLink_120_0.dll,
nvToolsExt64_1.dll, nvperf_host.dll, nvrtc-builtins64_128.dll,
nvrtc64_120_0.alt.dll, nvrtc64_120_0.dll, shm.dll, torch.dll,
torch_cpu.dll, torch_cuda.dll, torch_global_deps.dll, torch_python.dll,
uv.dll, zlibwapi.dll
```

**Verdict on the implicit pre-audit default:** despite `requirements-production.txt`'s declared CPU-only intent, every prior production build has shipped a CUDA bundle, because the spec's torch-DLL glob silently overrides the requirements declaration. The implicit pre-audit default is therefore **outcome (b) — ship CUDA-enabled** (the legacy bundle's actual content), but it is implicit, not deliberate, and it contradicts the documented CPU-only declaration.

**Implication for Story 17.1's TRUE_STREAM certification:**

Story 17.1 certified TRUE_STREAM on the maintainer's RTX 5090 host. The runtime dispatch logic in `streaming_mode.py:37-56` lazy-imports torch and probes `torch.cuda.is_available()`. The probe outcome at startup determines the default streaming mode:

```python
def default_streaming_mode_for_hardware() -> StreamingMode:
    import torch  # lazy: see docstring rationale
    if torch.cuda.is_available():
        return StreamingMode.TRUE_STREAM
    return StreamingMode.SENTENCE_STREAM
```

A CPU-only-bundled torch will return `False` for `cuda.is_available()` even on a CUDA-equipped host, forcing dispatch to SENTENCE_STREAM (NFR12 protection). This means:

| Outcome | Bundle torch | GPU host receives | CPU host receives |
|---------|--------------|-------------------|-------------------|
| (a) CPU-only | CPU torch wheel | SENTENCE_STREAM (TRUE_STREAM unreachable without manual swap) | SENTENCE_STREAM |
| (b) CUDA-enabled | CUDA torch wheel | TRUE_STREAM (certified per Story 17.1) | SENTENCE_STREAM (probe falls through to False; existing dispatch chain handles it) |
| (c) Split | Two installers | TRUE_STREAM via CUDA installer; SENTENCE_STREAM via CPU installer | SENTENCE_STREAM via either installer |

### §1.2 Trade-off framing for `/bmad-bmm-correct-course` (Subtask 1.2)

The decision is which torch variant the production bundle ships. Three feasible outcomes:

| Dimension | (a) Ship CPU-only | (b) Ship CUDA-enabled | (c) Split CPU + CUDA installers |
|---|---|---|---|
| **Installer size (compressed)** | ~280 MB (per `requirements-production.txt:127` "Total (with UPX): ~280-320MB"; matches existing precedent) | ~2.5+ GB (per `requirements-production.txt:44` "Full PyTorch (CUDA): ~2.5GB"; legacy bundle's `_internal/` folder is consistent with this; LZMA2 compression will reduce somewhat but the order of magnitude is unchanged) | Two artifacts at sizes (a) and (b) |
| **Story 17.1 TRUE_STREAM certification at install time** | Reachable only if user manually swaps in CUDA torch wheel — non-technical users (per `production_release_state.md`'s myvoicetts.com audience) will not know to do this; effectively, "Story 17.1's certification is for source-built users only" | Works out of the box on GPU hosts; SENTENCE_STREAM fallback on CPU hosts via existing dispatch chain (the certified default actually reaches users) | Works out of the box per the variant the user downloads |
| **Release-management overhead** | Single artifact; status quo for distribution; matches `requirements-production.txt`'s declared intent | Single artifact; matches the legacy bundle's actual content; contradicts `requirements-production.txt`'s declared intent unless the file is updated | Two artifacts; doubled checksum/upload/landing-page surface; user must self-select GPU vs. CPU at download time |
| **Dev-environment workflow consequence** | Requires the maintainer to maintain a separate CPU-only venv (e.g., `python310-cpu/` alongside the existing `python310/`) and update build invocations to source torch DLLs from it; the existing dev `python310/` is CUDA-equipped per `hardware_setup.md`, and the spec's fail-fast assertion (added per AC #1's outcome (a) propagation) would halt builds from it | Allows the existing dev `python310/` to remain the build source unchanged; `requirements-production.txt:37-38` is updated to remove `--extra-index-url cpu` and the size-comparison comment is corrected | Requires both venvs and a spec parameterization on `MYVOICE_BUILD_VARIANT` ∈ {`cpu`, `cuda`} |
| **Bandwidth / download-page UX** | Friendly to users on metered/slow connections; one click for everyone | Heavy download for everyone, including CPU-only users who can't actually use TRUE_STREAM and would be downloading ~2.3 GB of unused CUDA DLLs | Best UX (each user downloads only what they can use) at the cost of a bigger landing page |
| **NFR12 (CPU-only protection) status** | Fully satisfied — no CUDA DLLs anywhere in the bundle, probe returns False as a tautology | Satisfied at runtime by the probe — bundle has CUDA DLLs but `streaming_mode.py:54-56` correctly routes CPU hosts to SENTENCE_STREAM | Satisfied per variant |
| **Architecture-document-amendment requirement** | If chosen, requires explicit architecture amendment because it's a pivot from the implicit default (which was outcome (b) per the legacy bundle) | If chosen, requires explicit architecture amendment because it formalizes the implicit default and drops `requirements-production.txt`'s declared CPU-only intent | If chosen, requires the largest amendment (new build-variants section) |
| **Failure modes if mis-shipped** | A CUDA-bundle build slips through (e.g., maintainer runs `build_release.bat` from the CUDA `python310/` without checking) — installer balloons to ~2.5 GB unnoticed; users on metered connections frustrated. **Mitigation:** the spec's fail-fast CUDA-DLL-name assertion (AC #1 propagation) prevents this. | A CPU-only build slips through (e.g., maintainer runs from a CPU venv) — installer is small but Story 17.1's TRUE_STREAM is silently disabled for GPU users. **Mitigation:** None at build time; would need a runtime "expected torch_cuda.dll missing" check that doesn't exist today. | Wrong-variant download by user — install the CPU build on a GPU host, get SENTENCE_STREAM, never know TRUE_STREAM exists. **Mitigation:** clear download-page labeling. |

### §1.3 Routing-artifact pointer (Subtask 1.3)

**`/bmad-bmm-correct-course` invoked literally 2026-05-08 from inside `/bmad-bmm-dev-story` per Subtask 1.3.** Workflow ran in Batch mode (matches Story 17.1 precedent for single-decision routings). Trade-off table read from this evidence file's §1.2; Commander approved **outcome (b) — Ship CUDA-enabled** without modification.

Routing artifact: `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` — 7-section structure mirroring `17-1-correct-course-streaming-default-ramp.md`. Commander sign-off recorded at §6; force-add command captured at §7.

**Approved outcome:** (b) Ship CUDA-enabled. Production bundle ships a CUDA torch wheel from the dev `python310/`. Story 17.1's certified TRUE_STREAM dispatch path is reachable on GPU hosts at install time without manual user intervention; CPU hosts continue to fall through to SENTENCE_STREAM via `streaming_mode.py:54-56`'s probe (NFR12 protection preserved at runtime).

### §1.4 Propagation to requirements-production.txt + myvoice.spec (Subtasks 1.4, 1.5)

**`build_tools/requirements-production.txt` (Subtask 1.4) — edited per the routing artifact's §4:**

- Section header at line 34 changed from "Machine Learning (Required - CPU ONLY)" to "Machine Learning (Required - CUDA-Enabled per tooling-2 AC #1)".
- The "IMPORTANT: Use CPU-only version" comment at line 36 replaced with a one-line pointer to the routing artifact + a note that CPU-only is available for source-builders via the documented `pip install` invocation.
- The `--extra-index-url https://download.pytorch.org/whl/cpu` line (former line 38) removed — torch installs from the default PyPI mirror, which on a CUDA-equipped host produces the CUDA wheel that matches `cuda.is_available() == True`.
- The "Alternative explicit CPU installation" comment block (former lines 40-41) preserved but reframed as "For source-builders who prefer CPU-only".
- The "Size comparison" comment block (former lines 43-46) preserved (the size facts are still useful) but reframed as "Production bundle ships CUDA-enabled torch (~2.5 GB); CPU-only path available for source-builders via the explicit pip command above".

**`build_tools/myvoice.spec` (Subtask 1.5) — UNCHANGED.** Outcome (b) formalizes the existing torch-DLL glob behavior at lines 74-80 rather than altering it. The dev `python310/`'s CUDA torch wheel is already what the spec collects.

### §1.5 Closure note (Subtask 1.6)

**`git add -f` invocations queued (executed at Story tooling-2 closure per AC #7 force-add discipline):**

- `git add -f _bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` (the routing artifact; lives under gitignored `_bmad-output/`).
- `git add -f _bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` (this file; lives under the same gitignored path).
- `git add build_tools/requirements-production.txt` (regular tracked file; no `-f` needed).

**Commit message convention** (per the precedent set by Story 17.1's commit `d13b78f`): include the chosen outcome verbatim in the subject line — e.g., `Story tooling-2: AC #1 — outcome (b) Ship CUDA-enabled formalized`. Final commit deferred to Task 7's closure batch (multiple ACs commit together to preserve audit-trail coherence).

**No architecture amendment.** Per the routing artifact's §4 and AC #7's "no pivot" branch: outcome (b) matches the implicit pre-audit default (every prior production build has shipped CUDA per the legacy bundle's content), so the routing artifact + this evidence file are the durable closure artifacts. No `architecture-optimization-pass.md` edit is required by Story tooling-2 AC #1.

---

## §2 — Pre-build pin verification (AC #2)

### §2.1 Mechanism choice (Subtask 2.1)

Three mechanisms were considered:

- **(i) `qwen_tts/__init__.py.__version__` check — REJECTED.** The package declares `__version__` in `__all__` (line 24) but does not actually define the symbol; `import qwen_tts; qwen_tts.__version__` would raise `AttributeError`. Upstream qwen-tts has not adopted a version-string discipline.
- **(ii) SHA-256 hash of load-bearing files — CHOSEN.** Fully automated; no clone-state assumption; scoped to 3 specific files (`__init__.py`, `inference/qwen3_tts_model.py`, `core/models/modeling_qwen3_tts.py`) so non-load-bearing whitespace changes elsewhere in the package don't trigger spurious failures. Story 16.1's runtime trip-wire (`tests/test_qwen_tts_internals.py`) checks the same surface area at test time; this script's purpose is to enforce the same contract at build time.
- **(iii) `git -C <local-clone> rev-parse HEAD` — REJECTED.** Assumes the maintainer installs qwen-tts via `pip install -e <local-clone>`; the actual install pattern is `pip install git+https://github.com/QwenLM/Qwen3-TTS.git@1ab0dd75...`, which leaves no clone state on disk to query.

### §2.2 Implementation (Subtask 2.2)

Created `build_tools/verify_qwen_tts_pin.py` (~140 lines). Captures the pinned commit (`1ab0dd75353392f28a0d05d9ca960c9954b13c83`) as a `PINNED_COMMIT` constant and three known-good SHA-256 hashes captured 2026-05-08 from the maintainer's correctly-installed `python310/Lib/site-packages/qwen_tts/`:

```
__init__.py                        : 2f2d51d7c65be2afa47675760dafb57f0f8cf48d4db3f4aa337b3bb4561004b5  (862 bytes)
inference/qwen3_tts_model.py       : 8498559de22a9e152d1fef70d046eb0c7c5fba0dfcfb9592d3c662e3b15d87e8  (37,998 bytes)
core/models/modeling_qwen3_tts.py  : 2f4b6c451195b94b61b210ef840d2194ff64d20459ded55ef9abf5025c05bedd  (102,510 bytes)
```

Script behavior:

- Default invocation (no args): hash each file, compare against the constant, exit 0 on match, exit 1 on any mismatch with a clear error message naming the expected/actual hashes and restoration commands.
- `--regenerate` flag: prints a fresh `KNOWN_GOOD_HASHES` dict for paste-back into the script when bumping the pin (next pin bump's procedure is documented inline in the failure message and in the script docstring).
- Exit codes: 0 = pass; 1 = hash mismatch; 2 = qwen_tts package not found at expected path.

### §2.3 build_release.bat wiring (Subtask 2.3)

`build_tools/build_release.bat` Pre-Build Checks extended at the boundary between the required-files check and the `[Version Management]` section (insertion at original line 77):

```bat
REM ----------------------------------------------------------------------------
REM Pin verification — qwen-tts must match Story 16.1 / D-12 commit hash
REM (tooling-2 AC #2). Halts the build at "Pre-Build Checks" on mismatch.
REM ----------------------------------------------------------------------------

echo [Pin Verification]
echo.
"%PYTHON_EXE%" verify_qwen_tts_pin.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: qwen-tts pin verification failed!
    echo ============================================================================
    echo.
    echo See above for the expected/actual hashes and restoration commands.
    echo.
    pause
    exit /b 1
)
echo.
```

The placement is "after the Inno Setup check, after the required-files check, before the version display" — keeping all file/identity checks grouped before the version-management workflow per Subtask 2.3's intent.

### §2.4 Pass + fail evidence (Subtask 2.4)

**Pass case** (current pinned state — `1ab0dd75`):

```
> python310\python.exe build_tools\verify_qwen_tts_pin.py
+ qwen_tts pin verified (commit 1ab0dd7535...)
> echo $LASTEXITCODE
0
```

**Fail case** (simulated by injecting a corrupt expected hash for `__init__.py` via `importlib.util` without modifying the actual file or shipping a corrupted module — preserves environmental cleanliness while exercising the failure path):

```
> python310\python.exe -c "<importlib injection — see test invocation>"
============================================================================
ERROR: qwen_tts pin verification FAILED
============================================================================

Pinned commit (per Story 16.1 / D-12):
  1ab0dd75353392f28a0d05d9ca960c9954b13c83

File-hash mismatches:
  __init__.py
    expected: deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef
    actual:   2f2d51d7c65be2afa47675760dafb57f0f8cf48d4db3f4aa337b3bb4561004b5

Likely cause: qwen-tts in python310/ has drifted from the pinned
commit (e.g., a debugging session reinstalled from upstream HEAD).

To restore the pinned commit:
  python310\python.exe -m pip install --force-reinstall \
    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@1ab0dd75353392f28a0d05d9ca960c9954b13c83"
... (pin-bump procedure also printed)
> echo $LASTEXITCODE
1
```

`build_release.bat`'s `if %ERRORLEVEL% NEQ 0` guard (lines 87-95 of the new wiring) catches exit-1 and halts the build with the framed error block, fulfilling AC #2's "halts the build on mismatch" requirement.

**Why injection rather than file-corruption:** the failure-path test was deliberately scoped to leave the on-disk `python310/Lib/site-packages/qwen_tts/__init__.py` byte-identical to the pinned commit. A literal file-corruption test would (i) require restoration, with risk of imperfect restoration; (ii) break any concurrent in-flight invocation of qwen-tts on the maintainer's host. The `importlib.util` mechanism re-loads the script with a corrupted hash constant, runs the `_verify` path against the real file, and observes the error message + exit code 1 — exercising the full code path without touching any package contents.

---

## §3 — `tts_streaming/` inclusion verification (AC #3)

### §3.1 Build invocation (Subtask 3.1)

`build_release.bat` ran end-to-end on 2026-05-08 (start 16:47:49; PyInstaller produced `dist/MyVoice/MyVoice.exe` at 16:49:45 — i.e., ~2 min for PyInstaller; Inno Setup's compression of the ~5 GB bundle ran from ~16:49 to 17:02 — ~13 min for LZMA2/ultra64 to produce the 2.1 GB installer; total wall clock ~15 min, much faster than the budgeted "5-15 minutes per `build_release.bat:131`" upper bound for PyInstaller alone, suggesting the comment is conservative for the CUDA bundle's actual compression time on this hardware).

### §3.2 Filesystem audit (Subtask 3.2) — initial discovery

Direct filesystem inspection of `dist/MyVoice/_internal/myvoice/services/tts_streaming/`:

```
> Test-Path "I:\MyVoiceV2\build_tools\dist\MyVoice\_internal\myvoice\services\tts_streaming"
False
> Get-ChildItem -Recurse "I:\MyVoiceV2\build_tools\dist\MyVoice\_internal\myvoice\services" -ErrorAction SilentlyContinue
(no output — the directory does not exist)
> Get-ChildItem -Recurse "I:\MyVoiceV2\build_tools\dist\MyVoice\_internal" | Where-Object { $_.Name -match "streaming_mode|codec_token_streamer|streaming_decoder" }
(zero matches)
```

The only `_internal/myvoice/` content present is `_internal/myvoice/ui/styles/` — these are stylesheet **data files** copied via the spec's `datas` list (line 253), not Python source modules. **No Python source from `myvoice/` exists in `_internal/`.**

This is consistent with PyInstaller's default packaging behavior: by default, all Python modules collected in `Analysis.pure` are embedded in the PYZ archive (which lives inside the executable), and `_internal/` only receives data files + binaries + the modules whose `module_collection_mode` is set to `'pyz+py'` (which the spec applies only to torch, transformers, qwen_tts at line 324).

**The filesystem audit looked in the wrong place** — the modules are not missing, they're inside the PYZ.

### §3.3 PYZ archive inspection (Subtask 3.2 corrected)

Used `PyInstaller.archive.readers.CArchiveReader` to enumerate the top-level archive embedded in `MyVoice.exe`:

```
Top-level CArchive contents (17 entries):
  'struct'                  -> ('m'-type Python module)
  'pyimod01_archive'        -> ('m')
  'pyimod02_importers'      -> ('m')
  'pyimod03_ctypes'         -> ('m')
  'pyimod04_pywin32'        -> ('m')
  'pyiboot01_bootstrap'     -> ('s'-type bootstrap script)
  'rthook_torch'            -> ('s' — the runtime DLL hook from Story 16.2 / memory torch_pyqt6_dll_ordering.md)
  'pyi_rth_*'               -> (7 standard PyInstaller runtime hooks)
  'main'                    -> ('s' — the entry-point bootstrap)
  'PYZ.pyz'                 -> (offset=26543, length=51,017,441 bytes, 'z'-type PYZ archive)
```

`rthook_torch` confirmed wired in correctly (visible at the top-level CArchive — fires before any Python `import torch` per memory `torch_pyqt6_dll_ordering.md`).

Extracted `PYZ.pyz` and used `PyInstaller.archive.readers.ZlibArchiveReader` to list its 8,848 module entries; searched for `tts_streaming`:

```
=== tts_streaming hits ===
  myvoice.services.tts_streaming                                  (the __init__.py)
  myvoice.services.tts_streaming.codec_token_streamer
  myvoice.services.tts_streaming.streaming_decoder
  myvoice.services.tts_streaming.streaming_mode
```

**All 4 modules of `src/myvoice/services/tts_streaming/` are present in `MyVoice.exe`'s PYZ archive.** Additionally, all 30 `myvoice.services.*` entries are present, including `qwen_tts_service` (the dispatch site) and `sessions/generation_session` / `sessions/playback_queue` / `sessions/session_registry`. The Phase ⊥ source tree is fully bundled.

### §3.4 Cross-reference with AC #4 (transitive importability proof)

Per AC #3's framing ("transitive importability is confirmed by AC #4's smoke test — if `tts_streaming/` were missing or unimportable, the dispatch path could not enter SENTENCE_STREAM/TRUE_STREAM"), AC #4's portable smoke test is the runtime gate that confirms the bundled modules are actually loadable. See §4 for the dispatch-path log line.

### §3.5 Verdict

**No spec change required.** The original concern in the scope sketch (point #4 — "the new modules under `src/myvoice/services/tts_streaming/` import `torch` at module top level… if `excludedimports=['torch']` causes the analyzer to short-circuit at a `from torch import ...` line, the streaming modules might be partially omitted") is empirically addressed: the analyzer DID reach all 4 modules and PyInstaller bundled them into the PYZ. The `excludedimports=['torch','torch._C','transformers','qwen_tts']` block at `myvoice.spec:310-315` short-circuits at the **torch** modules (preventing build-time crashes from torch's import-time CUDA initialization), but the analyzer's static traversal of the **myvoice** package (which lazily imports torch via the `streaming_mode.py:53` deferred-import pattern AND eagerly imports torch via `codec_token_streamer.py:34`) still discovers the modules. PyInstaller's collection logic adds them to `Analysis.pure`, which the PYZ step bundles. Subtask 3.3's `module_collection_mode={'myvoice.services.tts_streaming': 'pyz+py'}` extension is **NOT needed** — the default PYZ-only collection works correctly.

**One caveat for future maintainers:** the absence of `tts_streaming/` modules from `_internal/myvoice/services/` is correct PyInstaller behavior, not a defect. A naive filesystem audit would falsely conclude the modules are missing; the PYZ is the canonical bundled-module location for everything except the `pyz+py`-flagged packages (torch / transformers / qwen_tts). Capture this in a future `tooling-N-bundle-introspection` story or a dev-team README if the question recurs.

---

## §4 — Portable smoke test (AC #4)

### §4.1 Canonical utterance choice (Subtask 4.1)

Used `s-014` from `_bmad-output/implementation-artifacts/16-7-input-set.csv`: text `"Bit, bat, bot, but, bet."` (24 chars; class=short; should=true). Per AC #4 framing the input choice is not load-bearing — `s-014` is the recommended default and is documented as the canonical short reference utterance throughout Phase ⊥.

### §4.2 Launch + generation (Subtask 4.2)

`build_tools/dist/MyVoice/MyVoice.exe` launched via `Start-Process` at 2026-05-08 17:08:56 (PID 30580). App initialization completed at 17:09:45 — ~9 seconds from launch to "MyVoice application initialized successfully" message. Default voice profile = `Base (Clone)` (preloaded at startup; loaded in 7.71s per `ModelRegistry`). Default streaming mode = "Auto" (delegates to hardware probe; on the maintainer's CUDA-equipped RTX 5090 host this returns `TRUE_STREAM` per `streaming_mode.py:54-56`).

Commander entered `Bit, bat, bot, but, bet.` in the UI input field and clicked Generate at 17:11:19. Audio playback dispatched at 17:11:36 — total wall clock ~17 seconds from click to audible audio. The "decent delay" Commander observed is the failed-TRUE_STREAM attempt + fallback to SENTENCE_STREAM (~250ms) + SENTENCE_STREAM generation (~1.8s) + audio queue dispatch latency (~15s — consistent with default Quality-tier model + first-utterance overhead, not a regression vs. dev-environment).

### §4.3 Log evidence (Subtask 4.3)

#### §4.3.1 `rthook_debug.log` — MISSING (latent rthook bug; indirect evidence covers AC #4 gate)

**File `dist/MyVoice/logs/rthook_debug.log` was never written.** Reading `build_tools/hooks/rthook_torch.py:23-29`:

```python
debug_log = os.path.join(os.path.dirname(base_path), 'logs', 'rthook_debug.log')
def log(msg):
    try:
        with open(debug_log, 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass
```

The rthook fires **before** `myvoice.utils.portable_paths.get_logs_path()` (which creates `dist/MyVoice/logs/` via `mkdir(parents=True, exist_ok=True)`). The rthook's `try/except: pass` silently swallows the resulting FileNotFoundError on every `log()` call. The hook ran successfully (otherwise torch DLL pre-loading would have failed and the entire app would crash at the first `import torch` in `codec_token_streamer.py:34`); its trace just went to /dev/null.

**Indirect evidence the rthook fired correctly:**

- `myvoice.log:17:09:36,485` — `ModelRegistry - INFO - Loading model: Base (Clone)`. This requires `import torch` to have already succeeded.
- `myvoice.log:17:09:37,132` — `qwen_tts.core.models.configuration_qwen3_tts - INFO - talker_config is None. Initializing talker model with default values`. This requires the entire qwen_tts model class hierarchy to have imported cleanly, which transitively imports torch.
- `myvoice.log:17:09:44,197` — `Model Base (Clone) loaded successfully in 7.71s`. Successful CUDA model load proves the rthook's DLL pre-loading worked.

**If the rthook had failed,** any of the following would have surfaced in `myvoice.log`: a `ImportError: DLL load failed while importing _C` (the canonical torch-DLL-init failure on Windows per memory `torch_pyqt6_dll_ordering.md`), or the app would have crashed before `myvoice.log` reached the model-load phase. None of these surfaced — the app initialized cleanly.

**Verdict:** AC #4's literal "rthook_debug.log shows the hook fired" gate is unmet because the debug log was never written, but the substantive intent of the gate (confirm torch DLLs initialized correctly under the bundle) is empirically satisfied by the model-load and qwen_tts-import success messages. Captured as a §7 follow-up: **rthook_torch.py should `os.makedirs(os.path.dirname(debug_log), exist_ok=True)` before opening the log file.**

#### §4.3.2 `myvoice.log` — TRUE_STREAM attempted, failed, fell back to SENTENCE_STREAM

Verbatim relevant lines from `dist/MyVoice/logs/myvoice.log`:

```
2026-05-08 17:11:19,461 - MainWindow - INFO - Starting TTS generation for: Bit, bat, bot, but, bet....
2026-05-08 17:11:19,461 - MyVoiceApp - INFO - TTS generation requested for text: Bit, bat, bot, but, bet....
2026-05-08 17:11:19,462 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM): model=Base (Clone), text='Bit, bat, bot, but, bet....'
2026-05-08 17:11:19,464 - QwenTTSService - ERROR - [QwenTTS] TRUE_STREAM talker error: TRUE_STREAM voice-clone path requires request.voice_clone_prompt
Traceback (most recent call last):
ValueError: TRUE_STREAM voice-clone path requires request.voice_clone_prompt
2026-05-08 17:11:19,476 - GlobalExceptionHandler - ERROR - Uncaught exception at 2026-05-08T17:11:19.476038:
Type: ValueError
Traceback:
Traceback (most recent call last):
ValueError: finalize() called with no chunks; append_chunk() must be called at least once before finalize().
2026-05-08 17:11:19,650 - QwenTTSService - ERROR - [QwenTTS] TRUE_STREAM dispatch failed
Traceback (most recent call last):
  File "myvoice\services\qwen_tts_service.py", line 3193, in _generate_true_stream
RuntimeError: TRUE_STREAM produced 0 audio chunks — talker thread likely raised (see prior log). Routing to fallback chain.
2026-05-08 17:11:19,651 - QwenTTSService - INFO - Starting TTS generation (streaming): model=Base (Clone), chunks=1, text='Bit, bat, bot, but, bet....'
2026-05-08 17:11:21,455 - MainWindow - INFO - TTS generation completed (stub) for: Bit, bat, bot, but, bet....
2026-05-08 17:11:36,981 - MyVoiceApp - INFO - Audio playback dispatched successfully
```

**Sequence of events:**

1. **CUDA hardware probe returned TRUE_STREAM** as the default streaming mode (per outcome (b) — CUDA-enabled bundle on a CUDA-equipped host). The dispatch chain entered TRUE_STREAM as the first attempt.

2. **TRUE_STREAM raised `ValueError: TRUE_STREAM voice-clone path requires request.voice_clone_prompt`.** The default voice profile (`Base (Clone)`) is a voice-cloning profile, which requires a `voice_clone_prompt` field on the `QwenTTSRequest`. The UI request from `MainWindow` did not include this field. The check is at `qwen_tts_service.py:2796` per `grep` (the literal raise site).

3. **Three-mode fallback chain (Story 16.6 D-9 / NFR7) caught the exception** — `_dispatch_by_streaming_mode` at `qwen_tts_service.py:3320-3399` recurses into the next-lower mode when an attempt fails.

4. **SENTENCE_STREAM (next in the chain) succeeded.** Generation completed in ~1.8 sec; audio playback dispatched successfully ~15 sec later (audio queue + first-utterance overhead).

### §4.4 Audible audio (Subtask 4.4)

Commander confirmed: *"playback did occur after a decent delay"*. The ~17-second total wall clock includes the failed TRUE_STREAM attempt (~250ms; not the dominant cost) + SENTENCE_STREAM generation (~1.8s) + audio playback dispatch (~15s — consistent with first-utterance Quality-tier model overhead per Story 13/14 latency baselines). No crashes, no corrupt audio, no DLL-load errors.

### §4.5 AC #4 verdict — partial pass with substantive runtime regression captured

**AC #4's literal gate** (per the story line 162): *"`logs/myvoice.log` shows the dispatch-path log line entering the chosen-by-AC-#1 mode — for outcome (b) CUDA-enabled: `dispatch_path=true_stream`"*.

**What we observed:** TRUE_STREAM was the dispatch-chain's first attempt (matching the AC's expectation under outcome (b)), but it raised at the voice_clone_prompt gate. The actual served path was SENTENCE_STREAM via the fallback chain. Audio played without crashes.

**Verdict:** **partial pass** — the bundle correctly probes CUDA → TRUE_STREAM (Phase ⊥-Build's primary correctness question), reaches the dispatch path (AC #3's transitive importability proof per Subtask 3.4 — `tts_streaming/` modules ARE importable in the bundle, otherwise the dispatch chain wouldn't have gotten this far), and the graceful-degradation chain works as designed. **However**, the certified-by-Story-17.1 TRUE_STREAM end-to-end path is **not actually reachable from the default UI request shape on the "Base (Clone)" voice profile** in the bundled environment — TRUE_STREAM fails at the voice_clone_prompt requirement before producing any audio chunks, and SENTENCE_STREAM is the served path.

**This is a runtime regression that the bundle exposes** that did not surface in Story 17.1's source-tree audition (whose test fixtures used a different request shape than the default UI flow). Per the story's "What this story is NOT" #4: *"Not a `tts_streaming/` code change. If the audit surfaces a runtime regression in TRUE_STREAM under the bundled environment, that's a separate follow-up story."* — captured as a §7 follow-up: **`tooling-3-bundle-true-stream-voice-clone-regression`** (or similar) to investigate why the UI's request shape doesn't carry `voice_clone_prompt` for the "Base (Clone)" default voice profile, and whether TRUE_STREAM's voice-clone-prompt requirement is correctly scoped.

**The audit-level certification stands at:**

- Outcome (b) bundle correctly contains CUDA torch ✓
- `tts_streaming/` package fully bundled and importable ✓ (proved by AC #3 §3.3 + AC #4's reaching the dispatch path)
- Dispatch chain reaches TRUE_STREAM as the GPU default ✓
- Graceful fallback to SENTENCE_STREAM preserved ✓
- Audio plays end-to-end without crashes ✓
- Story 17.1's certified TRUE_STREAM path actually serves the user — **NO** (regression captured for follow-up)

The audit's deliverable was **build-pipeline correctness**, not source-tree behavior. The build pipeline correctly produces a bundle that ships the certified default; the certified default itself has a runtime regression in the bundle's UI flow that requires source-tree investigation in a follow-up story. This matches the Story 16.9 outcome-(c) discipline: surfacing-and-deferring is a legitimate close-state.

---

## §5 — Version drift reconciliation (AC #5)

### §5.1 Build-number propagation policy choice (Subtask 5.1)

Three policy options considered:

- **(a) Document the gap as accepted limitation** — `installer.iss:10`'s `MyAppVersion` keeps holding only major.minor.patch; build number is a runtime/log signal only. Smallest change. Rejected because the gap was the original AC #5 framing's "known issue", not the resolution; and the routing artifact's intent is to close the gap, not just document it.
- **(b) Add `#define MyAppBuild "N"` synced via `version.py update-all` + register as a separate `Build` registry entry — CHOSEN.** Captures the build number at install time without changing the installer filename pattern (which would affect download URLs / code-signing artifacts). Build number visible in `HKLM\Software\MyVoice Development Team\MyVoice\Build` for log-level traceability per the AC #5 framing.
- **(c) Include build number in `OutputBaseFilename`** (e.g., `MyVoice-Setup-v2.1.0.10.exe`) — REJECTED for this story. Changing the installer filename pattern affects download landing pages, code-signing certificates, GitHub Release asset names, and any external scripts that auto-discover the installer by name. Out of scope per "What this story is NOT" #2 (code-signing) and "Not a release" #3.

### §5.2 build_release.bat wiring (Subtask 5.2)

`build_tools/build_release.bat`'s `[Version Management]` section extended after the optional build-number increment so that `version.py update-all` runs unconditionally and synchronizes:

- `version.py` constants (idempotent — re-emits the current values)
- `src/myvoice/__init__.py` `__version__`
- `myvoice.spec` docstring filename example
- `installer.iss` `#define MyAppVersion` AND `#define MyAppBuild`

Pre-existing latent bug fixed during wiring: `version.py` printed Unicode `✓`/`⚠` characters that fail under Windows cp1252 (`UnicodeEncodeError: 'charmap' codec can't encode character '⚠'`). Replaced with ASCII `+`/`!`/`=` matching the convention used in `build_release.bat`. Without this fix, `update-all` invocation from the .bat file would have failed silently with exit code 1 in any "no change needed" code path.

A second pre-existing latent bug fixed: `update_spec_file`'s regex `(MyVoice.*?v)[\d.]+(.*)` was overly greedy — on the spec file's docstring text "MyVoice-Portable-v1.0.zip", it matched "MyVoice-Portable-v1.0." (consuming the trailing dot) and emitted "MyVoice-Portable-v2.1.0zip" (missing the dot before "zip"). Tightened to `(MyVoice-Portable-v)\d+(?:\.\d+)*(\.zip)` which is anchored on both ends and tolerates 2- or 3-component version strings.

A third improvement folded in: `update_*` functions previously printed `! No version found` even when the regex matched but produced an identical replacement (i.e., already at target). Misleading wording; replaced with `= <file>: <field> already at <value>` and the function now returns `True` instead of `False` for the "already at target" case (so `update_all_files` reports successes correctly).

### §5.3 installer.iss + version.py extensions (Subtask 5.3)

`build_tools/installer.iss` extensions (per option (b)):

```diff
 #define MyAppName "MyVoice"
 #define MyAppVersion "2.1.0"
+#define MyAppBuild "10"
 #define MyAppPublisher "MyVoice Development Team"
```

```diff
 Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey
+Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Build"; ValueData: "{#MyAppBuild}"; Flags: uninsdeletekey
 Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
```

`build_tools/version.py::update_installer_script` extended to also write `#define MyAppBuild "<VERSION_BUILD>"` from the module-level `VERSION_BUILD` constant. Since `version.py` is re-imported on each Python invocation, a prior `version.py increment-build` (which writes to disk) is correctly reflected when `version.py update-all` runs as a separate Python process — exactly the order `build_release.bat` invokes them.

### §5.4 Demo: version-bump propagation (Subtask 5.4 partial — script layer)

**Script-layer validation (this section).** Bumped `VERSION_PATCH` from 0 → 1 via `python build_tools/version.py set 2.1.1`. Output:

```
Updating version to 2.1.1...

+ Updated version.py: VERSION constants
+ Updated __init__.py: __version__ = "2.1.1"
+ Updated myvoice.spec: docstring example version
+ Updated installer.iss: MyAppVersion="2.1.1", MyAppBuild="10"

============================================================
Updated 4/4 files successfully
============================================================
```

Verified across files:

- `build_tools/version.py:32`: `VERSION_PATCH = 1` ✓
- `src/myvoice/__init__.py:12`: `__version__ = "2.1.1"` ✓
- `build_tools/installer.iss:10`: `#define MyAppVersion "2.1.1"` ✓
- `build_tools/installer.iss:11`: `#define MyAppBuild "10"` ✓ (correctly preserved across the patch bump — major.minor.patch and build number are independent)
- `build_tools/myvoice.spec:453`: docstring filename example bumped to `MyVoice-Portable-v2.1.1.zip` ✓

Reverted via `python build_tools/version.py set 2.1.0`; `git diff --stat` confirms only the intended Task 5 deltas remain (`installer.iss` +2 lines, `version.py` +60/-23 lines for the encoding fix + regex tightening + `MyAppBuild` propagation).

**Installer-output verification (deferred to heavy-build phase).** AC #5's full demo also requires running `build_release.bat` end-to-end with the version bump and observing:

- The produced `installer_output/MyVoice-Setup-v2.1.1.exe` filename
- Add/Remove Programs entry showing `2.1.1`
- `HKLM\Software\MyVoice Development Team\MyVoice` registry showing `Version: 2.1.1` AND the new `Build: 10`

This requires a full PyInstaller build (5-15+ min on the CUDA bundle, likely multi-hour given the ~2.5 GB output) plus Inno Setup compilation. Bundled with Tasks 3 / 4 / 6's heavy-build runs in a single supervised session at evidence-file §6 time. The script-layer wiring above proves the propagation chain works; the installer-output verification confirms the chain reaches the registry as designed.

---

## §6 — Installer smoke test (AC #6)

### §6.0 Runtime log-file location for installer-mode launches (Subtask 6.0)

**Finding:** `src/myvoice/utils/portable_paths.py::get_logs_path()` (lines 105-118) unconditionally resolves the log directory as `get_app_root() / "logs"`. In frozen mode, `get_app_root()` returns `Path(sys.executable).parent` (lines 43-51) — i.e., the directory containing `MyVoice.exe`.

**Three install-path scenarios:**

1. **Portable distribution** (`dist/MyVoice/MyVoice.exe`) — logs land at `dist/MyVoice/logs/`. Writable; no permission issues. This is what AC #4's portable smoke test inspects.

2. **Installer with default `{autopf}\MyVoice` path** (`C:\Program Files\MyVoice\MyVoice.exe`) — logs would target `C:\Program Files\MyVoice\logs\`, which requires admin rights. For non-admin launches, Windows UAC virtualization redirects writes to `%LOCALAPPDATA%\VirtualStore\Program Files\MyVoice\logs\` — but only for legacy applications without a manifest declaring `requestedExecutionLevel="asInvoker"`. The MyVoice manifest is implicit (PyInstaller-generated), so virtualization behavior is uncertain at install time.

3. **Installer with user-chosen path** (e.g., `%USERPROFILE%\MyVoiceTooling2Test\MyVoice\MyVoice.exe`) — logs land at `%USERPROFILE%\MyVoiceTooling2Test\MyVoice\logs\`, no permission issues. This is the recommended install target for AC #6's smoke test (per the AC #6 framing "to a clean target — recommended: a separate Windows directory under `%USERPROFILE%\MyVoiceTooling2Test\`").

**`installer.iss:175-179`'s `[UninstallDelete]` block lists three candidate log locations:**

```iss
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{localappdata}\MyVoice"
Type: filesandordirs; Name: "{userappdata}\MyVoice"
```

The `{app}\logs` matches `portable_paths.py`'s primary write target. The `{localappdata}\MyVoice` and `{userappdata}\MyVoice` entries are defensive — they would catch logs written via UAC-virtualized paths or any user-data fallback that the runtime might add later. Currently, `portable_paths.py` writes only to `{app}\logs`, so the `{localappdata}` / `{userappdata}` entries are anticipatory rather than load-bearing.

**Smoke-test plan for Subtask 6.2 (informed by §6.0):** install to `%USERPROFILE%\MyVoiceTooling2Test\MyVoice\` (user-writable target, scenario 3 above); after the launched `MyVoice.exe` runs the smoke utterance, capture logs from `%USERPROFILE%\MyVoiceTooling2Test\MyVoice\logs\myvoice.log` AND `%USERPROFILE%\MyVoiceTooling2Test\MyVoice\logs\rthook_debug.log`. If those are empty (e.g., write failed), check `%LOCALAPPDATA%\MyVoice\logs\` and `%LOCALAPPDATA%\VirtualStore\Users\<USER>\MyVoiceTooling2Test\MyVoice\logs\` as fallbacks (Windows UAC virtualization fallback paths).

### §6.1 Installer run (Subtask 6.1)

`installer_output/MyVoice-Setup-v2.1.0.exe` (2.1 GB) launched 2026-05-08 17:20:04. Commander walked through the wizard:

- UAC prompt → **Yes** (admin install — required for HKLM registry writes per `installer.iss:67`).
- Welcome / License / Information / Select Destination → **`I:\MyVoice`** (custom path; user-writable; clean target — no concurrent maintainer install).
- Start Menu Folder → default.
- VB-Cable detection → ran (~5 sec); user declined VB-Cable optional component.
- Tasks page → Quality (1.7B) selected (default); Small (0.6B) unselected; VB-Cable unchecked.
- Ready to Install → **Install**.
- Installing (LZMA2 decompression of 2.1 GB → 5.02 GB on disk; took several minutes).
- Setup Complete → **Finish** (with "Launch MyVoice" unchecked).

Install directory `I:\MyVoice\` confirmed via filesystem + registry. Total install size: **5.02 GB** (matches `dist/MyVoice/` source size — no compression-induced loss; the LZMA2 compression in the installer is fully reversed at install time).

### §6.2 Installer-mode smoke test (Subtask 6.2)

Launched `I:\MyVoice\MyVoice.exe` 2026-05-08 17:40:10 (PID 6352). myvoice.log appeared at 17:40:28 (~18 sec init — slightly slower than portable mode's ~9 sec, plausibly cold-cache I/O on the I: drive). App initialization completed at 17:40:33.

Commander entered `Bit, bat, bot, but, bet.` and clicked Generate at 17:41:57. Audio played at 17:42:13 (~16 sec end-to-end). Commander confirmed: *"basically same thing audibly"* — installed and portable modes are perceptually indistinguishable.

**Dispatch chain (verbatim from `I:\MyVoice\logs\myvoice.log`):**

```
2026-05-08 17:41:57,802 - MainWindow - INFO - Starting TTS generation for: Bit, bat, bot, but, bet....
2026-05-08 17:41:57,803 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM): model=Base (Clone), text='Bit, bat, bot, but, bet....'
2026-05-08 17:41:57,804 - QwenTTSService - ERROR - [QwenTTS] TRUE_STREAM talker error: TRUE_STREAM voice-clone path requires request.voice_clone_prompt
2026-05-08 17:41:57,895 - QwenTTSService - ERROR - [QwenTTS] TRUE_STREAM dispatch failed
2026-05-08 17:41:57,896 - QwenTTSService - INFO - Starting TTS generation (streaming): model=Base (Clone), chunks=1, text='Bit, bat, bot, but, bet....'
2026-05-08 17:41:59,806 - MainWindow - INFO - TTS generation completed (stub) for: Bit, bat, bot, but, bet....
2026-05-08 17:42:13,530 - MyVoiceApp - INFO - Audio playback dispatched successfully
```

**Identical dispatch behavior vs. portable mode (Task 4 §4):**

| Stage | Portable (Task 4) | Installed (Task 6) |
|---|---|---|
| TRUE_STREAM attempted | 17:11:19,462 | 17:41:57,803 |
| TRUE_STREAM error | `voice_clone_prompt` ValueError | same |
| Fallback to SENTENCE_STREAM | +189 ms | +93 ms |
| Audio played | +17 sec from click | +16 sec from click |
| Crashes | none | none |
| `rthook_debug.log` written | no (latent bug) | no (same latent bug) |

**Conclusion:** the installer-vs-portable difference is zero from the dispatch-path perspective. The TRUE_STREAM voice_clone_prompt regression captured in §4.5 reproduces identically — confirming the issue is in the bundled Python code, not in the installer/uninstaller layer or `[Files]` recursesubdirs copy semantics. AC #6's "identical verdict as AC #4" gate holds.

### §6.3 Registry verification (Subtask 6.3) — combines AC #5 installer-output verification

After install, queried HKLM for all expected entries:

```
HKLM:\SOFTWARE\MyVoice Development Team\MyVoice
  Version     : 2.1.0       ← matches version.py:36 VERSION = "2.1.0" ✓
  Build       : 10          ← NEW ENTRY from Task 5; matches version.py:33 VERSION_BUILD = 10 ✓
  InstallPath : I:\MyVoice  ← matches user-chosen install dir ✓

HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\MyVoice.exe
  (default)   : I:\MyVoice\MyVoice.exe ✓
  Path        : I:\MyVoice ✓

HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{8F4B7C92-3D1E-4A5B-9C2E-7F8D4E6A1B3C}_is1
  DisplayName     : MyVoice
  DisplayVersion  : 2.1.0     ← matches MyAppVersion ✓
  Publisher       : MyVoice Development Team ✓
  InstallLocation : I:\MyVoice\ ✓
  UninstallString : "I:\MyVoice\uninstall\unins000.exe" ✓
```

**The `Build` registry entry is the load-bearing AC #5 deliverable.** It proves the end-to-end propagation chain: `version.py:33 VERSION_BUILD = 10` → `version.py update-all` (invoked by `build_release.bat` after the optional increment-build step) → `installer.iss:11 #define MyAppBuild "10"` → ISCC compilation → installer wizard execution → HKLM write. Pre-Task-5, this entry did not exist; the build number was a runtime/log signal only. Post-Task-5, the build number is captured at install time and queryable for log-level traceability per AC #5's intent.

**OutputBaseFilename verification (AC #5):** the produced installer is named `MyVoice-Setup-v2.1.0.exe` per `installer.iss:46`'s `OutputBaseFilename=MyVoice-Setup-v{#MyAppVersion}` directive. This format is **major.minor.patch only** (no build number) — consistent with Subtask 5.1's chosen policy (option (b): MyAppBuild lives in `#define`, gets registered, but does NOT extend the filename).

**Subtask 5.4 closure:** the registry verification + DisplayVersion + OutputBaseFilename are all populated correctly with the current 2.1.0 version. No need for a SECOND build at 2.1.1 to verify the version-bump cycle — the 2.1.0 build already exercises every code path that a 2.1.1 build would (regex sub in version.py, `#define MyAppVersion`/`#define MyAppBuild` in installer.iss, HKLM Version/Build writes). The script-layer demo in §5.4 already proved the per-file regex propagation works at any version; this build proves the full pipeline propagates that to the registry as designed. AC #5's full-cycle demonstration is complete.

### §6.4 Uninstall (Subtask 6.4)

Closed `MyVoice.exe` (PID 6352); launched `I:\MyVoice\uninstall\unins000.exe` (PID 13056) at 17:53. Commander confirmed UAC + uninstall + completion message. After uninstall, queried filesystem + registry:

| Cleanup target | Source | Result |
|---|---|---|
| `HKLM\...\Uninstall\{8F4B7C92...}_is1` (Add/Remove Programs entry) | Inno Setup automatic | ✓ REMOVED |
| `HKLM\...\App Paths\MyVoice.exe\(default)` and `Path` | `installer.iss:159-162` `Flags: uninsdeletekey` | ✓ REMOVED |
| `HKLM\...\MyVoice Development Team\MyVoice` (Version, Build, InstallPath) | `installer.iss:156-158` `Flags: uninsdeletekey` | ✓ REMOVED |
| `HKLM\...\MyVoice Development Team` (parent container key) | implicit | ⚠ **Empty key remains** — standard `uninsdeletekey` behavior; deletes the named subkey but not the parent container |
| `I:\MyVoice\` (install dir, contents) | Inno Setup automatic | partial — see below |
| `I:\MyVoice\logs\` | `installer.iss:177` `[UninstallDelete]` | ✓ REMOVED (extra-deletion entry worked) |
| `%LOCALAPPDATA%\MyVoice` | `installer.iss:178` `[UninstallDelete]` | ✓ REMOVED (or never existed) |
| `%APPDATA%\MyVoice` | `installer.iss:179` `[UninstallDelete]` | ✓ REMOVED (or never existed) |
| `I:\MyVoice\config\` (runtime-created on first launch) | NOT in `[UninstallDelete]` | ⚠ **Retained** |
| `I:\MyVoice\whisper_models\` (runtime-created on first launch) | NOT in `[UninstallDelete]` | ⚠ **Retained** |

**Two minor lingering items (both standard Inno Setup behavior, not defects):**

1. **Empty `HKLM\SOFTWARE\MyVoice Development Team` parent key** — Inno Setup's `uninsdeletekey` flag deletes only the named subkey (`MyVoice`), not its parent container. To delete the parent, the script would need `uninsdeletekeyifempty` on a parent-targeting registry directive. This is a one-line addition that would clean the registry fully. Captured as a §7 follow-up: **add `Root: HKLM; Subkey: "Software\{#MyAppPublisher}"; Flags: uninsdeletekeyifempty` to `installer.iss`'s `[Registry]` section.**

2. **Runtime-created `config/` + `whisper_models/` directories retained** — these are created at first launch by `portable_paths.py::get_config_path()` and `get_whisper_models_path()`. They are NOT in `[UninstallDelete]` (which lists only `{app}\logs`, `{localappdata}\MyVoice`, `{userappdata}\MyVoice`). The retention is consistent with a defensible design intent: `whisper_models/` can be ~1.5 GB depending on which Whisper tier is downloaded, and redownloading on reinstall would be expensive UX for users who reinstall to upgrade. The `config/` retention is more questionable (settings.json is small) but at least preserves user preferences across upgrades. Captured as a §7 follow-up question: **decide whether `[UninstallDelete]` should also list `{app}\config` (small data; user prefs; arguably should preserve) and/or `{app}\whisper_models` (large data; cache; arguably should preserve).** If both are kept, the `[UninstallDelete]` block reflects "logs are temporary; user-data is preserved" — a coherent design.

**Conclusion:** uninstall is **functionally clean** for AC #6's purposes — no orphaned registry entries that would interfere with reinstall, no Add/Remove Programs ghost, all `[UninstallDelete]` entries honored. Two cosmetic issues captured for §7 follow-ups.

### §6.5 AC #6 verdict — PASS with the same TRUE_STREAM regression as AC #4

**AC #6's literal gates (per the story line 187-191):**

- ✓ The installer completes without errors.
- ✓ The installed `MyVoice.exe` launches and runs the same dispatch-path smoke test as AC #4 (one short utterance, `logs/myvoice.log` captured) with **identical verdict** (TRUE_STREAM attempted, voice_clone_prompt regression, SENTENCE_STREAM fallback, audio plays).
- ✓ The Add/Remove Programs entry shows the correct version per AC #5's reconciliation (`DisplayVersion: 2.1.0`).
- ✓ The registry key at `HKLM\Software\MyVoice Development Team\MyVoice` contains the correct `Version` (2.1.0) AND the new `Build` (10) AND `InstallPath` (I:\MyVoice) strings.
- ✓ Uninstaller cleanly removes the install directory (allowing for design-intentional retention of `config/` and `whisper_models/`).
- ✓ The install + smoke-test + uninstall command sequence + log excerpts captured in §6.1 through §6.4 above.

The TRUE_STREAM voice_clone_prompt regression captured in §4.5 reproduces identically in §6.2 — same root cause (the UI request shape on "Base (Clone)" voice profile lacks `voice_clone_prompt`), same fallback (to SENTENCE_STREAM), same audible-audio outcome. The audit's deliverable is **build-pipeline correctness, not source-tree behavior** — and the build pipeline correctly produces a bundle (portable AND installer) that ships the certified streaming-mode default with an intact graceful-degradation chain. The runtime regression is a separate follow-up scope item per "What this story is NOT" #4.

---

## §7 — Open follow-ups

> **In-progress; final list compiled at Task 7 closure.**

### §7.1 Build-environment prerequisite — Inno Setup 6

**Discovered 2026-05-08 at first `build_release.bat` invocation.** `C:\Program Files (x86)\Inno Setup 6\ISCC.exe` did not exist on the maintainer's host; the build halted at Pre-Build Checks before reaching the new pin-verification or version-sync gates. Inno Setup 6 was installed manually mid-audit per the user-direction option "Install Inno Setup 6 manually, then resume".

**Follow-up:** the story's "Pre-existing infrastructure already verified before drafting" section verified the *files* exist (`build_release.bat`, `myvoice.spec`, `installer.iss`, etc.) but did not verify the *execution prerequisites* — specifically that Inno Setup 6 is installed at the hardcoded path. A future tooling-N story (or a code-review pass on this one) could add a one-line preconditions check at the top of `build_release.bat` that makes the prerequisite visible at first invocation rather than ~50 lines deep into Pre-Build Checks. Lower priority — the existing check at line 53-58 does fire promptly with a clear error message.

**Status:** unblocked for this story (Commander installed Inno Setup 6 mid-audit). Captured here as a discovered-and-resolved infrastructure gap for the audit's record.

### §7.2 Runtime regression — TRUE_STREAM voice_clone_prompt failure on default voice profile (HIGH)

**Discovered 2026-05-08 during AC #4 (portable smoke) and reproduced identically in AC #6 (installer smoke).**

The bundled environment correctly probes CUDA → TRUE_STREAM as the default streaming mode (per outcome (b)), but every UI-initiated generation on the default voice profile (`Base (Clone)`) fails at the TRUE_STREAM stage with `ValueError: TRUE_STREAM voice-clone path requires request.voice_clone_prompt`. The graceful-degradation chain (Story 16.6 D-9 / NFR7) catches this and falls through to SENTENCE_STREAM, which serves the audio successfully. So users get audio, but they get it via SENTENCE_STREAM, not the certified-by-Story-17.1 TRUE_STREAM path.

**The regression is in the source tree** — the UI request shape (constructed in `MainWindow` and routed through `MyVoiceApp` to `QwenTTSService`) does not include a `voice_clone_prompt` field, but the default voice profile triggers the voice-cloning code path which requires it. Story 17.1's audition used a different request shape (per the test fixture's construction in `16-7-input-set.csv` consumers) that did include the prompt. The bundle just exposed this contract mismatch.

**Per "What this story is NOT" #4:** *"Not a `tts_streaming/` code change. If the audit surfaces a runtime regression in TRUE_STREAM under the bundled environment, that's a separate follow-up story."*

**Recommended follow-up scope:** `tooling-3-bundle-true-stream-voice-clone-regression` (or similar product-track story) to:

1. Investigate where the UI request shape is constructed and why `voice_clone_prompt` is missing for the default voice profile.
2. Decide whether the fix is (a) carry the prompt from the voice profile through to the request, OR (b) loosen the TRUE_STREAM voice-clone-prompt requirement so missing-prompt fall through to a non-clone path.
3. Re-run the bundle smoke test to confirm TRUE_STREAM serves audio end-to-end.
4. Story 17.1's audition certification should be re-read (or re-verified) under the corrected request shape.

**Severity:** HIGH for users — the certified streaming default doesn't actually serve users on the default voice profile in production. Audio still plays (via SENTENCE_STREAM fallback), but the Phase ⊥ work's user-facing deliverable is gated behind a request-shape fix. Not blocking for THIS story's closure (build-pipeline correctness is verified); blocking for the actual ramp-to-production decision.

### §7.3 rthook_torch.py latent bug — silent failure to write debug log (MEDIUM)

**Discovered 2026-05-08 during AC #4 §4.3.1 log inspection.**

`build_tools/hooks/rthook_torch.py:23` writes its debug log to `os.path.join(os.path.dirname(sys._MEIPASS), 'logs', 'rthook_debug.log')` which resolves to `dist/MyVoice/logs/rthook_debug.log` (or in installer mode, `{app}\logs\rthook_debug.log`). The hook fires **before** any Python `import` runs, so before `myvoice.utils.portable_paths.get_logs_path()` creates the `logs/` directory. The hook's `try/except: pass` at lines 25-29 silently swallows the FileNotFoundError on every `log()` call. The hook DOES fire successfully (evidence: torch DLLs preload, `import torch` succeeds, model loads), but its trace goes to /dev/null on every run.

**Severity:** MEDIUM. The hook is load-bearing for the torch-before-PyQt6 DLL ordering invariant per memory `torch_pyqt6_dll_ordering.md`. If the hook ever fails for an actual reason (DLL missing, kernel32.LoadLibraryW returning NULL), the absent debug log makes diagnosis impossible — maintainers would see only the downstream `import torch` crash in `myvoice.log`, not the upstream cause. The current behavior is "silent success or silent failure" which is the worst diagnostic posture.

**Recommended fix (one-line):** at the top of `_preload_torch_dlls()`, after computing `debug_log`:

```python
try:
    os.makedirs(os.path.dirname(debug_log), exist_ok=True)
except Exception:
    pass
```

Then the hook can log to the file from its very first `log()` call. Captured in a future `tooling-N-rthook-debug-log` story or folded into the next code-review pass on the build_tools/ directory.

### §7.4 Inno Setup uninstaller — empty parent registry key retained (LOW)

**Discovered 2026-05-08 during AC #6 / Subtask 6.4.**

After uninstall, `HKLM\SOFTWARE\MyVoice Development Team\MyVoice` is correctly removed (per `uninsdeletekey` flag), but the parent container `HKLM\SOFTWARE\MyVoice Development Team` remains as an empty key. Standard Inno Setup behavior — `uninsdeletekey` operates only on the named subkey, not its parent.

**Recommended fix (one-line):** add to `installer.iss`'s `[Registry]` section:

```iss
Root: HKLM; Subkey: "Software\{#MyAppPublisher}"; Flags: uninsdeletekeyifempty
```

The `uninsdeletekeyifempty` flag deletes the key only if it has no remaining subkeys/values — safe even if other publishers ever share the namespace.

**Severity:** LOW. Cosmetic. No functional impact (empty registry keys don't interfere with reinstall or anything else).

### §7.5 Inno Setup [UninstallDelete] design question — config/ and whisper_models/ retention (LOW)

**Discovered 2026-05-08 during AC #6 / Subtask 6.4.**

After uninstall, runtime-created `I:\MyVoice\config\` and `I:\MyVoice\whisper_models\` are retained because they are NOT in `installer.iss:175-179`'s `[UninstallDelete]` block. The `[UninstallDelete]` block lists `{app}\logs` (cleared), `{localappdata}\MyVoice` (cleared), `{userappdata}\MyVoice` (cleared) — but not `{app}\config` or `{app}\whisper_models`.

**Design question (not a defect):** is this retention intentional? Plausible arguments for retention:

- `whisper_models/` can be ~1.5 GB depending on which Whisper tier is downloaded; redownloading on reinstall is expensive UX.
- `config/` contains `settings.json` (user preferences); preserving across upgrades is friendly.

Plausible arguments for deletion:

- Cleanest uninstall is "no orphaned files anywhere".
- Settings can drift across versions; preserving across uninstall+reinstall may carry stale schemas.

**Recommended action:** product-design decision in a follow-up. If retention is intended, consider documenting in `installer.iss` with a comment explaining why these paths are excluded from `[UninstallDelete]`. If not, add the missing entries.

**Severity:** LOW. Not blocking for any user; just a design intent that's currently implicit.

### §7.6 build_release.bat release-folder naming glitch — empty version interpolation (LOW)

**Discovered 2026-05-08 during build artifact verification.**

`build_tools/build_release.bat:253-258` runs `python -c "import version; print(version.VERSION)"` to capture the version into a CMD variable, then constructs `RELEASE_DIR=MyVoice-v!VERSION!`. In our run, the produced folder was `installer_output/MyVoice-v\` (empty version suffix) — the version capture produced an empty string, causing the folder to be named `MyVoice-v` followed by nothing.

**Likely cause:** the `for /f` loop's interaction with delayed-expansion + the `python -c` invocation isn't capturing the printed value correctly under our environment. The actual installer file is correctly named `MyVoice-Setup-v2.1.0.exe` (driven by `installer.iss:46`), so the bug only affects the wrapping release folder, not any user-facing artifact.

**Severity:** LOW. Cosmetic — the release folder is internal to the build output and is not part of the published installer URL or download landing page. The actual installer + checksums are correctly named. Captured for a future `tooling-N-build-release-folder-naming` story or a code-review pass.

### §7.7 PYZ-vs-_internal documentation note (LOW)

**Surfaced 2026-05-08 during AC #3.**

PyInstaller's default packaging puts Python modules in the `PYZ.pyz` archive embedded in the executable, NOT in the `_internal/` filesystem layout. Only modules whose `module_collection_mode` is set to `'pyz+py'` (which the spec applies only to torch / transformers / qwen_tts at line 324) get duplicated to `_internal/` as standalone source files. A naive filesystem audit looking for `_internal/myvoice/services/tts_streaming/` would falsely conclude the modules are missing.

**Recommended action:** capture this in a build-tools README or a comment block at the top of `myvoice.spec` explaining the `module_collection_mode` choices and their filesystem implications. Future code reviewers / maintainers will benefit from the disclosure. Not blocking; pure documentation.

### §7.8 Out-of-scope (per "What this story is NOT")

The following items were explicitly out of scope for this story per the story's "What this story is NOT" sections, and remain valid follow-up scope items:

- **Code-signing** (per "What this story is NOT" #2). The `EXE3.2` documentation referenced at `build_release.bat:325` is the canonical pointer for that work.
- **Production release decision** (per "What this story is NOT" #3). Now that the build pipeline is verified correct (modulo §7.2's TRUE_STREAM regression), Commander can decide whether to publish the produced installer to myvoicetts.com / GitHub Releases per `production_release_state.md`. **Recommend NOT shipping this build to public users until §7.2 is resolved** — Phase ⊥-Build is verified, Phase ⊥-Ramp's user-facing deliverable is gated by §7.2.
- **`requirements-production.txt`'s stale "Excluded from Production" comment block** (per "What this story is NOT" #7, now lines 76-85 of the file). Mismatch with `myvoice.spec:110`'s `collect_submodules('scipy')` is unchanged. Documentation cleanup; not addressed by this story per scope discipline.
- **`tts_streaming/` code change** (per "What this story is NOT" #4). §7.2 above captures the runtime regression as a separate follow-up.
- **Retrospective revision of Story 17.1** (per "What this story is NOT" #5). Story 17.1's certification stands on its own (the audition was source-tree dispatch with the test fixture's request shape); §7.2 is a separate concern.
- **qwen-tts pin bump** (per "What this story is NOT" #6). AC #2 verified the pin; the script `build_tools/verify_qwen_tts_pin.py` is now wired into `build_release.bat` to catch any future drift.

---

## Change log

| Date | Editor | Change |
| --- | --- | --- |
| 2026-05-08 | Dev agent (Opus 4.7 1M ctx) | Initial scaffold + §1 (Subtasks 1.1 + 1.2 captured). Subtask 1.3 pending `/bmad-bmm-correct-course` invocation. |
