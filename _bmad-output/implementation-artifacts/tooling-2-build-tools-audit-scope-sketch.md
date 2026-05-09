# Scope Sketch — Tooling-2: Build-Tools Audit (Phase ⊥-Build)

> **Status:** scope sketch (input to `/bmad-bmm-create-story`); not yet a story file.
> **Authored:** 2026-05-08 by `/bmad-bmm-code-review` follow-up turn after Story 17.1 closure.
> **Purpose:** Capture concrete audit dimensions, known concerns, and a five-point scope sketch so `/bmad-bmm-create-story` can convert this into a real story (`tooling-2-build-tools-audit.md`) with full ACs / Tasks / Dev Notes. Mirrors the role the Epic 16 retro's §"Significant Discoveries Affecting the Streaming Default Ramp follow-up story" played for Story 17.1.

## Why this story exists

Phase ⊥ (Epic 16 + Story 17.1) added new production source modules and a new dependency to the V2 codebase:

- **New package:** `src/myvoice/services/tts_streaming/` (4 modules — `__init__.py`, `streaming_mode.py`, `codec_token_streamer.py`, `streaming_decoder.py`).
- **New dependency:** `qwen-tts` pinned to commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83` (Story 16.1 / D-12).
- **New dispatch path:** `qwen_tts_service.py::_generate_true_stream` plus the three-mode fallback chain (TRUE_STREAM → SENTENCE_STREAM → BATCH).

The latest build under `build_tools/dist/` is `MyVoice2.0.1.9Portable/` — **pre-Phase ⊥ track** (Epic 11–15 era). No production build has been produced since the Phase ⊥ work landed. The build pipeline (PyInstaller spec at `build_tools/myvoice.spec` + Inno Setup script at `build_tools/installer.iss` + the `build_release.bat` orchestrator) has not been exercised against the new modules, the new dependency, or the new dispatch path.

Several mismatches exist on the surface that warrant a deliberate audit pass before the next release:

1. **`requirements-production.txt:37-38` ships CPU-only PyTorch** (`--extra-index-url https://download.pytorch.org/whl/cpu`) to save ~2.3 GB. But Story 17.1 certified TRUE_STREAM, which is gated by `streaming_mode.py:54-56`'s `torch.cuda.is_available()` probe. **A CPU-only production bundle will never enter TRUE_STREAM at runtime** regardless of host hardware — the user gets SENTENCE_STREAM unless they manually swap in a CUDA torch wheel. This may be intentional (NFR12 hardware-aware default; CPU users on SENTENCE_STREAM is the safe path) or unintentional (size optimization that silently disables a certified feature). The audit must surface this and either (a) document it as a deliberate trade-off in the architecture document and the installer's user-facing docs, or (b) split the production build into two artifacts (CPU portable / CUDA portable) per existing precedent.

2. **`myvoice.spec:74-80` copies torch DLLs from `python310/Lib/site-packages/torch/lib/`** — i.e., whatever torch is actually installed in the dev `python310/` directory at build time. This is `torch 2.10+cu128` per memory `hardware_setup.md` (RTX 5090 Blackwell). So the spec packages CUDA DLLs, but `requirements-production.txt` says CPU-only. **The two sources disagree.** If the build is run from the dev python310 dir, the resulting bundle will include the CUDA DLLs; if a fresh production venv is used per `requirements-production.txt`'s instructions, it'll be CPU-only. The audit must reconcile.

3. **The qwen-tts pin is not verified at build time.** `requirements.txt:23` and `requirements-production.txt:56` both pin to commit `1ab0dd75`. `myvoice.spec:141-142` copies `python310/Lib/site-packages/qwen_tts/` wholesale — whatever's installed. There's no pre-build assertion that the installed commit matches the pinned commit. If the maintainer's `python310/` has drifted (e.g., a follow-up debugging session installed `git+...@HEAD`), the build silently ships a different qwen-tts. **The pin lives in the requirements file but is not enforced at build time.**

4. **`tts_streaming/` package — implicit inclusion vs. excludedimports.** `myvoice.spec:310-315` excludes `'torch'`, `'torch._C'`, `'transformers'`, `'qwen_tts'` from import analysis (the comment says "prevents crash during build"). The new modules under `src/myvoice/services/tts_streaming/` import `torch` at module top level (e.g., `streaming_mode.py:54-56` calls `torch.cuda.is_available()`). The Analysis step traces from `main.py`; if `tts_streaming` is reachable through the static import graph, PyInstaller will include the source — but if `excludedimports=['torch']` causes the analyzer to short-circuit at a `from torch import ...` line, the streaming modules might be partially omitted. Needs concrete verification: build the bundle, grep `_internal/myvoice/services/tts_streaming/` for the four source files plus their compiled bytecode.

5. **`rthook_torch.py` runtime DLL ordering.** Per memory `torch_pyqt6_dll_ordering.md`, torch must initialize before PyQt6 on Windows. The runtime hook is wired in `myvoice.spec:326`. The audit must verify the hook is still effective after the new tts_streaming imports — specifically whether the new dispatch path's `import torch` calls happen before or after the hook fires.

6. **Version drift between Inno Setup and `version.py`.** `installer.iss:11` hardcodes `MyAppVersion "2.1.0"`; `build_release.bat` reads from `version.py` and does increment-build prompts. If `version.py` reports `2.1.x` but `installer.iss` is hardcoded to `2.1.0`, the installer's Add/Remove Programs entry and the Registry key will lag the actual build version. Audit must verify the version source of truth.

7. **Installer size baseline.** Per memory `production_release_state.md`, "installer size is a known pain point". The new `tts_streaming/` modules add ~50 lines of Python (negligible); torch is the dominant size driver and is unchanged. But `requirements-production.txt:78-86` excludes `matplotlib / pandas / scipy / opencv / pillow` — and `myvoice.spec:111` actually `collect_submodules('scipy')` because `voice_design_studio_dialog.py` needs it. The "excluded from production" comment in `requirements-production.txt` is **stale** vs. the spec. May not be a Phase ⊥-induced issue but is worth flagging for the same audit pass.

8. **Smoke-test the produced binary against the certified TRUE_STREAM path.** The story's deliverable should include running the produced `MyVoice.exe` against one short utterance (e.g., `s-014` from the canonical input set) and verifying the dispatch path log shows TRUE_STREAM (on a CUDA host) or SENTENCE_STREAM (on CPU-only) without crashing. This is the architectural smoke gate equivalent to Story 17.1's pytest run.

## Five-point scope sketch (for the SM workflow to expand into ACs)

(a) **Reconcile CPU vs. CUDA torch.** Decide and document: does the production build ship CPU-only (Story 17.1's TRUE_STREAM certification is then "available only when the user supplies CUDA torch separately") or does it ship CUDA-enabled (installer size grows by ~2.3 GB; certified TRUE_STREAM works out of the box on GPU hosts)? Update `requirements-production.txt` AND `myvoice.spec` so they agree, and add a one-paragraph note to `architecture-optimization-pass.md` (probably a new "Build-time torch variant" sub-section). Recommend deciding via `/bmad-bmm-correct-course` if the trade-off is non-obvious.

(b) **Pin-verification at build time.** Add a pre-build check (probably in `build_release.bat` Step 0 or `build_portable.py` Step 1.5) that asserts `python310/Lib/site-packages/qwen_tts/__init__.py` corresponds to commit `1ab0dd75`. Mechanism: `git -C` against the cached source if pip installed it as editable, OR a `__version__` / file-hash check against a known-good. Failing the check halts the build. Mirrors Story 16.1's import-attribute trip-wire pattern but at build-time rather than runtime.

(c) **Verify `tts_streaming/` inclusion in the bundle.** Build the portable, grep for the 4 modules in `dist/MyVoice/_internal/`, and either (i) confirm they're present (no fix needed; document the verification step in the build README), or (ii) find them missing and add an explicit `module_collection_mode={'myvoice.services.tts_streaming': 'pyz+py'}` entry to `myvoice.spec`. Add a regression test (a `test_release.bat`-equivalent assertion or a Python smoke script that imports them from the bundled `_internal/` path).

(d) **Smoke-test the produced binary against the dispatch path.** Run the built `MyVoice.exe` once with a short canonical utterance (e.g., `s-014`); capture the application log; assert the dispatch path entered TRUE_STREAM (on the maintainer's RTX 5090 host) or SENTENCE_STREAM (if the audit decides CPU-only is the production default). The smoke result is captured as a build-artifact alongside the `.sha256.txt` checksums per `build_release.bat` Step 4.

(e) **Reconcile version drift.** Make `version.py` the single source of truth; update `installer.iss` to read the version via Inno Setup's `#expr` directive or a pre-build script that sed-replaces the hardcoded literal. Verify the resulting installer's Add/Remove Programs entry matches `version.py`.

## What this story is NOT

- **Not a redesign of the build pipeline.** PyInstaller + Inno Setup is the established stack; this audit accepts the existing tooling and verifies it handles the Phase ⊥ additions correctly.
- **Not a code-signing story.** The `EXE3.2` documentation referenced in `build_release.bat` line 325 is a separate concern; if the audit surfaces a code-signing issue, capture it as a follow-up scope item.
- **Not a release.** This story produces a verified-correct build pipeline; the actual release decision (publish .exe to myvoicetts.com, push to GitHub Releases) is a Commander decision after the audit closes.
- **Not a tts_streaming code change.** If the audit surfaces a runtime regression in TRUE_STREAM under the bundled environment, that's a separate follow-up story — this story's scope is the build pipeline, not the streaming code.
- **Not a retrospective revision of Story 17.1.** Story 17.1's audition certified the dev-environment dispatch path; if the production-bundle dispatch path differs (e.g., CPU-only forces SENTENCE_STREAM), that's an architectural disclosure for the build-tools story, not a re-litigation of Story 17.1's verdict.

## References

- Build pipeline:
  - `build_tools/myvoice.spec` (PyInstaller spec — 456 lines; the load-bearing artifact)
  - `build_tools/installer.iss` (Inno Setup script — 539 lines)
  - `build_tools/build.bat` (portable-only entry; calls PyInstaller, sets up dist dirs)
  - `build_tools/build_portable.py` (Python-driven portable build with cleanup + README generation)
  - `build_tools/build_release.bat` (full release pipeline — Pre-Build Checks → PyInstaller → Inno Setup → checksums → release folder)
  - `build_tools/hooks/rthook_torch.py` (runtime DLL ordering hook; per `memory/torch_pyqt6_dll_ordering.md`)
  - `build_tools/requirements-production.txt` vs. top-level `requirements.txt` (mismatch surface — see point #2)
- New production code added in Phase ⊥:
  - `src/myvoice/services/tts_streaming/__init__.py`
  - `src/myvoice/services/tts_streaming/streaming_mode.py` (the hardware probe)
  - `src/myvoice/services/tts_streaming/codec_token_streamer.py`
  - `src/myvoice/services/tts_streaming/streaming_decoder.py`
- Architecture: `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (NFR3 row at line 803 with Story 17.1 audition pointer; new H4 sub-section after Story 16.9 amendment captures the certified default)
- Memory:
  - `memory/torch_pyqt6_dll_ordering.md` (DLL-init invariant — load-bearing for the runtime hook)
  - `memory/torch_before_coverage_dll_ordering.md` (related dev-environment quirk; not directly relevant to release builds but worth knowing)
  - `memory/hardware_setup.md` (dev host RTX 5090 + cu128; ship-target also covers RTX 30xx/40xx)
  - `memory/production_release_state.md` (installer-size pain point context; ships via myvoicetts.com)
- Precedent stories for shape and tooling-namespace conventions:
  - `_bmad-output/implementation-artifacts/tooling-1-git-repo-integration.md` (the only prior `tooling-N` story; tooling-namespace pattern)
  - `_bmad-output/implementation-artifacts/16-1-qwen-tts-dependency-pin-and-import-attribute-test.md` (Story 16.1 — the runtime pin-verification trip-wire that point (b) of this scope sketch builds on for build-time)
  - `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (Story 17.1 — what gets dispatched in the bundle is what this audit must verify)

## Suggested story-file naming

`_bmad-output/implementation-artifacts/tooling-2-build-tools-audit.md` (continues the `tooling-N` namespace; sprint-status entry could be `tooling-2-build-tools-audit: ready-for-dev` outside any epic block, mirroring `tooling-1`'s placement).

## Suggested Phase tag

`Phase ⊥-Build` — the build-pipeline corollary to `Phase ⊥-Ramp` (Story 17.1). Closes the loop between "we have a certified default" (Story 17.1) and "the build artifact ships that certified default" (this story).
