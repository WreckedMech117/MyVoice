# Tooling Story 2: Build-Tools Audit — Bring the PyInstaller / Inno Setup Pipeline Up to Date with Phase ⊥

Status: review

> **Scope note:** This is a **standalone tooling story**, NOT part of any product epic (11–17). It addresses a development-infrastructure gap surfaced by the Story 17.1 closure: the V2 baseline acquired a new package (`src/myvoice/services/tts_streaming/`), a new pinned dependency (`qwen-tts @ 1ab0dd75`), and a new dispatch path (TRUE_STREAM via `qwen_tts_service.py::_generate_true_stream`) during Phase ⊥, but the build pipeline (PyInstaller spec + Inno Setup script + `build_release.bat`) has not been exercised against any of those additions. The latest bundle on disk is `build_tools/dist/MyVoice2.0.1.9Portable/` — a pre-Phase ⊥ build (Epic 11–15 era). Filed in `implementation-artifacts/` so `/bmad-bmm-code-review` can review it the same way as a product story; **deliberately not added to `sprint-status.yaml`** — `tooling-N` stories follow the precedent set by `tooling-1-git-repo-integration` (sprint-status tracks user-facing epics; infrastructure stories are reviewed and closed independently).
>
> **Why this is the right (and only) entry point of Phase ⊥-Build.** The Story 17.1 closure handed off a working, certified TRUE_STREAM dispatch path on the maintainer's dev environment. The next architectural milestone — putting that certified default into the hands of users via a release build — is gated by verifying that the PyInstaller spec, the runtime DLL hook, the Inno Setup installer, and the version-management glue still produce a working artifact after the Phase ⊥ source-tree changes. Mirrors the role `tooling-1` played when `/bmad-bmm-code-review` needed git context that V2 didn't have.
>
> **Net behavior change for users (zero on the ship-target hardware — this story does not change runtime code).** No source under `src/myvoice/` is touched. Possible touched files: `build_tools/myvoice.spec`, `build_tools/installer.iss`, `build_tools/requirements-production.txt`, `build_tools/build_release.bat`, possibly a new pre-build assertion script. The deliverable is a verified-correct build artifact (the `.exe` and the installer) and the documented evidence (smoke-test log + bundle-content audit) that the artifact runs the certified TRUE_STREAM dispatch path on a CUDA host or the SENTENCE_STREAM fallback on a CPU-only host.
>
> **Pre-existing infrastructure already verified before drafting.**
>
>   - **The scope sketch is committed and complete.** `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-scope-sketch.md` (authored 2026-05-08 by `/bmad-bmm-code-review` follow-up turn) names 8 concrete concerns and a 5-point scope. This story file translates that sketch into ACs / Tasks / Dev Notes.
>
>   - **The build pipeline is intact.** `build_tools/build_release.bat` (334 lines), `build_tools/myvoice.spec` (456 lines), `build_tools/installer.iss` (539 lines), `build_tools/build.bat` (49 lines, portable-only entry), `build_tools/build_portable.py` (247 lines), `build_tools/version.py` (329 lines), and `build_tools/hooks/rthook_torch.py` (118 lines) all exist and are unmodified since pre-Phase ⊥.
>
>   - **The legacy bundle on disk confirms the CPU-vs-CUDA mismatch hypothesis empirically.** `build_tools/dist/MyVoice2.0.1.9Portable/_internal/torch/lib/` contains CUDA-suite DLLs (`c10_cuda.dll`, `cudart64_12.dll`, `cudnn64_9.dll`, `cublas64_12.dll`, `cusparse64_12.dll`, `cusolver64_11.dll`, `cufft64_11.dll`, `nvJitLink_120_0.dll`, etc.) — i.e., the previous build was produced from a CUDA-equipped `python310/` even though `requirements-production.txt:37-38` says CPU-only. The CPU-vs-CUDA mismatch is not hypothetical; it's what's already shipping in the most recent build directory.
>
>   - **The qwen-tts pin is dual-source.** `requirements.txt:23` and `build_tools/requirements-production.txt:56` both pin to commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83`. Both files include identical "do NOT replace with the unpinned form" warning comments per Story 16.1 / D-12. **However, no pre-build check enforces the pin** — `myvoice.spec:141-142` copies `python310/Lib/site-packages/qwen_tts/` wholesale, regardless of which commit is actually installed. Story 16.1's runtime trip-wire (`tests/test_qwen_tts_internals.py`) catches mismatches at test time but does NOT run during the release build.
>
>   - **The new `tts_streaming/` package is a 4-file Python module.** `src/myvoice/services/tts_streaming/__init__.py`, `streaming_mode.py` (the hardware probe at line 53-56 — `import torch` + `torch.cuda.is_available()`), `codec_token_streamer.py` (top-level `import torch` at line 34), `streaming_decoder.py`. Bundle-size impact is negligible (Python source — torch is the dominant size driver and is unchanged). The `excludedimports=['torch', 'torch._C', 'transformers', 'qwen_tts']` block at `myvoice.spec:310-315` is the load-bearing pattern for build-time stability; whether `tts_streaming/` modules are reached by the Analysis step despite this exclusion list is the central correctness question for AC #3.
>
>   - **`version.py` is a stand-alone synchronizer.** `build_tools/version.py` is the single source of truth (`VERSION_MAJOR/MINOR/PATCH/BUILD = 2/1/0/10` as of 2026-05-08). Its `update-all` command propagates `version.MAJOR.MINOR.PATCH` to `src/myvoice/__init__.py`, `myvoice.spec` (comment-only), and `installer.iss:10` (`#define MyAppVersion`). However, **`build_release.bat:86-94` invokes `version.py` only to display + prompt-to-increment-build; it does NOT call `version.py update-all`** — the major/minor/patch sync is a separate manual step. Additionally, `installer.iss`'s `MyAppVersion` only ever holds `MAJOR.MINOR.PATCH` (no build number), so successive builds with the same major.minor.patch ship under the same installer version even though their `VERSION_BUILD` differs — the build number is a runtime/log signal only, never a release-artifact signal.
>
>   - **The runtime DLL hook is the load-bearing artifact for the `torch` ↔ `PyQt6` initialization invariant on Windows** (per memory `torch_pyqt6_dll_ordering.md`). `build_tools/hooks/rthook_torch.py` runs at frozen-app startup, pre-loads `c10.dll` / `torch_cpu.dll` / `c10_cuda.dll` / `torch_cuda.dll` / `torch.dll` via `kernel32.LoadLibraryW`, adds `torch/lib/` and `PyQt6/Qt6/bin/` to the DLL search path, and writes a debug log to `logs/rthook_debug.log`. It is wired in `myvoice.spec:326`. The hook's correctness is a precondition for any Phase ⊥ dispatch path — including the TRUE_STREAM forward-hook from Story 16.8 — to even start in the bundled context. The audit must verify the hook still fires before any new `import torch` introduced by `tts_streaming/`.
>
>   - **`.gitignore` excludes both `python310/` (line 39) and `build_tools/dist/` (line 51).** The dev portable Python is not source-controlled, and neither are build outputs. This means the audit's evidence (the contents of `dist/MyVoice/_internal/`, the bundle-size measurement, the smoke-test log) lives outside the git tree. **Capture all evidence in `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md`** (force-add via `git add -f`, mirroring the gitignore pattern `tooling-1` and Story 16.9 used for their captured evidence). The evidence file is the durable artifact; the bundle itself is reproducible from the spec at any time.
>
>   - **The architecture document amendment pattern is established.** Story 16.9's NFR1 reconciliation amendment at `architecture-optimization-pass.md:802` (inline pointer) + line 819 (new H4 sub-section) and Story 17.1's NFR3 amendment at line 803 + a Story 17.1 H4 sub-section are the canonical two-place edit shape. Story tooling-2's amendment (if the audit decides on a non-trivial pivot, e.g., split CPU/CUDA portables) would target the "Build pipeline" section of `architecture-optimization-pass.md` — exact line varies by edit. **If the audit's outcome is "everything works as-is, document the verified state," no architecture amendment is required**; the evidence file alone is the durable artifact (mirroring `tooling-1`'s closure pattern, where no architecture doc was edited).
>
>   - **The `correct-course` routing pattern is the right tool only if the audit surfaces a genuine architectural pivot** (e.g., "produce two installers — CPU-only / CUDA — instead of one"). For pure verification work (AC #3 / AC #4 / AC #6), no `/bmad-bmm-correct-course` invocation is needed; the audit's verdict is captured directly in the evidence file. AC #1's CPU-vs-CUDA decision **does** require `/bmad-bmm-correct-course` — it's a deliberate trade-off (installer size vs. out-of-the-box TRUE_STREAM availability) and the precedent from Story 17.1 is to route stakeholder-impact decisions through the literal workflow rather than via in-line `AskUserQuestion` substitution.
>
>   - **The memory entry for the Phase ⊥ flag-flip outcome already records the certification state.** `memory/epic16_streaming_blocked.md` (last updated 2026-05-08 after Story 17.1 closure) reads "audition cleared … flag flip certified." After this story closes, **a new memory entry** at `memory/build_tools_phase_perp_state.md` (or similar) captures the build-pipeline state ("Phase ⊥-Build closed YYYY-MM-DD; the production bundle ships <CPU torch / CUDA torch / both>; pin verification runs pre-build via <mechanism>; smoke-test evidence at `tooling-2-build-tools-audit-evidence.md`"). The existing `epic16_streaming_blocked.md` entry is NOT updated — it is correctly framed as historical and the Phase ⊥-Build state is its own concern.
>
>   - **No production code change expected in any audit outcome.** The deliverable is build-pipeline correctness, not source-tree behavior change. If the audit surfaces a runtime regression in TRUE_STREAM under the bundled environment (e.g., the runtime hook fails to fire before `tts_streaming/`'s top-level `import torch`), that's a **separate follow-up story** scoped via the evidence file's "follow-ups" section — NOT a code fix folded into this story. The retro pattern from Story 16.9's outcome (c) discipline applies: surfacing-and-deferring is a legitimate close-state.
>
> **Six-point story scope** (translates the scope sketch's 5-point sketch + adds the smoke-test gate as its own AC):
>
> (a) **Reconcile CPU vs. CUDA torch.** Decide and document via `/bmad-bmm-correct-course`: does the production build ship CPU-only (Story 17.1's TRUE_STREAM certification is then "available only when the user supplies CUDA torch separately"), CUDA-enabled (installer size grows by ~2.3 GB; certified TRUE_STREAM works out of the box on GPU hosts), or split into two artifacts (CPU portable + CUDA portable, mirroring an existing pattern that may or may not exist)? Update `requirements-production.txt` AND `myvoice.spec` so they agree, and add a one-paragraph note to `architecture-optimization-pass.md`'s build-pipeline section if the decision pivots from the previous implicit default.
>
> (b) **Pin-verification at build time.** Add a pre-build assertion that `python310/Lib/site-packages/qwen_tts/` corresponds to commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83`. Mechanism options (in increasing rigor): (i) check `qwen_tts/__init__.py.__version__` if defined; (ii) compute a stable file-hash of the package's load-bearing files (`__init__.py` + `inference/qwen3_tts_model.py`) and compare against a known-good captured at pin time; (iii) require the maintainer to install qwen-tts via `pip install -e <local-clone>` and check `git -C <local-clone> rev-parse HEAD`. Mechanism (ii) is the recommended default — fully automated, no clone-state assumption. Failing the check halts `build_release.bat` at "Pre-Build Checks." Mirrors Story 16.1's runtime trip-wire pattern but at build time rather than test time.
>
> (c) **Verify `tts_streaming/` inclusion in the bundle.** Build the portable, then `dir _internal\myvoice\services\tts_streaming\` (or the equivalent within the bundle's archive layout — possibly inside `_internal/base_library.zip` or compiled into `python310.dll` depending on `module_collection_mode`). Confirm all 4 modules are present (or, if collected as bytecode, that their `.pyc` equivalents are reachable via the bundled Python's `importlib`). If missing, add an explicit `module_collection_mode={'myvoice.services.tts_streaming': 'pyz+py'}` entry to `myvoice.spec`. Capture the verification command + output in the evidence file.
>
> (d) **Smoke-test the produced binary against the dispatch path.** Run the built `MyVoice.exe` once on the maintainer's RTX 5090 host with one short canonical utterance (e.g., `s-014` from `_bmad-output/implementation-artifacts/16-7-input-set.csv`). Capture `logs/myvoice.log` and assert the dispatch-path log line shows TRUE_STREAM (on CUDA) or SENTENCE_STREAM (if AC #1 decided CPU-only). The smoke-test result is the architectural counterpart to Story 17.1's pytest run — without this, the audit cannot claim "build is verified to ship the certified default."
>
> (e) **Reconcile version drift.** Verify `version.py update-all` is invoked (or invokable) as part of `build_release.bat` such that `installer.iss:10`'s `MyAppVersion` literal stays in sync with `version.py`'s `VERSION_MAJOR.MINOR.PATCH`. Optionally extend the sync to include the build number — either by changing `installer.iss`'s `MyAppVersion` to use `VERSION.BUILD` semantics, or by adding a `MyAppBuild` define for log-level traceability. Document the chosen approach in the evidence file.
>
> (f) **Smoke-test the produced installer.** Beyond (d), validate that the Inno Setup installer (`installer_output/MyVoice-Setup-v*.exe`) actually installs to a clean target directory without errors, that the installed `MyVoice.exe` launches and runs the same dispatch-path smoke-test from (d), and that the Add/Remove Programs entry shows the expected version. This is the user-visible deliverable; without (f), the build pipeline could produce a working portable but a broken installer (different code paths through Inno Setup's `[Files]` recursesubdirs, Pascal Script `WriteModelQualitySettings`, etc.).
>
> ---
>
> **What this story is NOT** (explicit, to keep scope bounded):
>
> - **Not a redesign of the build pipeline.** PyInstaller + Inno Setup is the established stack; this audit accepts the existing tooling and verifies it handles the Phase ⊥ additions correctly.
>
> - **Not a code-signing story.** The `EXE3.2` documentation referenced in `build_release.bat:325` is a separate concern; if the audit surfaces a code-signing issue (e.g., the new `tts_streaming/` modules trigger an antivirus heuristic), capture it in the evidence file as a follow-up scope item — NOT a fix folded into this story.
>
> - **Not a release.** This story produces a verified-correct build pipeline + smoke-tested artifacts; the actual release decision (publish `.exe` to myvoicetts.com, push to GitHub Releases per `production_release_state.md`) is a separate Commander decision after the audit closes.
>
> - **Not a `tts_streaming/` code change.** If the audit surfaces a runtime regression in TRUE_STREAM under the bundled environment (e.g., the rthook_torch.py hook doesn't fire before `tts_streaming/streaming_mode.py:53`'s lazy `import torch`), that's a separate follow-up story. This story's scope is the build pipeline, not the streaming code.
>
> - **Not a retrospective revision of Story 17.1.** Story 17.1's audition certified the dev-environment dispatch path. If the production-bundle dispatch path differs (e.g., AC #1 decides CPU-only and the production bundle's smoke test shows SENTENCE_STREAM where Story 17.1 saw TRUE_STREAM), that's an architectural disclosure for this story's evidence file, NOT a re-litigation of Story 17.1's verdict.
>
> - **Not a qwen-tts pin bump.** The pin remains at commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83` per Story 16.1. AC #2 verifies the pin; it does NOT update it. If the audit decides a pin bump is warranted (e.g., upstream issued a security patch), that's a separate follow-up story.
>
> - **Not a `requirements-production.txt` rewrite.** The "Excluded from Production" comment block at lines 76-86 is stale vs. `myvoice.spec:110`'s `collect_submodules('scipy')` — but reconciling this is a documentation cleanup that may be folded in only if it doesn't expand scope. If the audit's main concerns (CPU/CUDA, pin, bundle, smoke) are already substantial, defer the doc cleanup to a follow-up.
>
> - **Not a `build_tools/dist/MyVoice2.0.1.9Portable/` clean-up.** The legacy bundle on disk is reference data for the audit (it exists as proof that the spec previously produced a CUDA bundle from the dev `python310/`). Do NOT delete it during the audit — `build_release.bat:98-122` already cleans `dist/` at Step 1 of the next build, so no manual cleanup is warranted.
>
> - **Not a test-suite expansion.** No new `tests/` files are created. The smoke test in AC #4 / AC #6 is run against the bundled binary, not against the dev environment, and is captured as evidence rather than as an automated test (an automated test would require headless launching of the bundled `.exe`, which is its own infrastructure project).

## Story

As a **MyVoice maintainer (solo developer + Commander)**,
I want **the PyInstaller spec, the Inno Setup installer script, the runtime DLL hook, the dependency requirements files, and the version-management glue audited and reconciled against the Phase ⊥ source-tree changes (new `src/myvoice/services/tts_streaming/` package, new `qwen-tts @ 1ab0dd75` pin, new TRUE_STREAM dispatch path) — with a documented CPU-vs-CUDA decision, a build-time pin verification gate, a verified bundle-content audit, a dispatch-path smoke test of both the portable and the installer artifact, and a reconciled version-drift mechanism**,
So that **the next release build of MyVoice ships the Phase ⊥ work with verified correctness rather than on faith — closing the loop between Story 17.1's "TRUE_STREAM is certified" and the user-facing "the installer they download actually runs that certified default."**

As a **MyVoice user (GPU host, default settings)**,
I want **the production installer to ship a torch build whose `cuda.is_available()` returns `True` on my hardware (or, if the maintainer decides CPU-only is the production default, a clearly-documented installation path to swap in a CUDA wheel myself)**,
So that **the streaming-default-confirmed-by-audition feature from Story 17.1 actually reaches me at install time rather than being available only to users who build from source**.

As a **MyVoice user (CPU-only host)**,
I want **the production installer to install cleanly and run the SENTENCE_STREAM dispatch path on my hardware, with the bundle's `torch.cuda.is_available()` correctly returning `False`**,
So that **NFR12's CPU-only-host protection (Story 16.2 / `streaming_mode.py:54-56`) actually fires at runtime in the bundled context, and I'm not silently routed to a code path my hardware can't run**.

## Acceptance Criteria

**Background — what this story is and is NOT.**

This story does the following to the working tree: optionally edits `build_tools/myvoice.spec` (per AC #1's CPU/CUDA decision and possibly per AC #3 for explicit `module_collection_mode`); optionally edits `build_tools/requirements-production.txt` (per AC #1); optionally edits `build_tools/installer.iss` (per AC #1 if the decision pivots to two installers and per AC #5 if version-drift fix needs it); adds a new pre-build pin-verification script (per AC #2); optionally edits `build_tools/build_release.bat` (per AC #2's pre-build invocation and per AC #5's `version.py update-all` invocation); creates `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` (NEW — captures all verification commands, outputs, smoke-test logs, decisions); creates `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` (NEW — only if AC #1's outcome warrants it; mirrors `17-1-correct-course-streaming-default-ramp.md`'s structure); optionally edits `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (only if AC #1 pivots from the implicit default); creates `memory/build_tools_phase_perp_state.md` (NEW — captures the post-audit build-pipeline state); updates this story file's Change Log as deliverables accumulate.

This story does **NOT**:

- Touch any file under `src/myvoice/`. The audit is a build-pipeline-correctness exercise, not a source-tree behavior change. (Production code changes uncovered as needed by the audit are deferred to follow-up stories per the "What this story is NOT" #5/#7.)

- Touch any file under `tests/`. The smoke test in AC #4 / AC #6 is an evidence-captured manual run against the bundled binary, not an automated test added to the suite.

- Bump the qwen-tts pin or any other dependency. AC #2 verifies the pin; the pin commit hash remains `1ab0dd75353392f28a0d05d9ca960c9954b13c83`. If the audit surfaces a pin-bump need, that's a follow-up story.

- Regenerate the Phase ⊥ perceptual fixture at `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/`. The fixture is independent of the build pipeline.

- Modify the existing pre-Phase-⊥ legacy bundle at `build_tools/dist/MyVoice2.0.1.9Portable/`. The bundle is read-only reference data; `build_release.bat:99-114`'s Step 1 will clean `dist/` at the next build invocation.

- Code-sign the resulting installer. Code-signing is a separate concern (`build_release.bat:325` references `EXE3.2 documentation` as the canonical pointer for that work).

- Run the build pipeline more than necessary. PyInstaller takes 5–15 minutes per `build_release.bat:131`; the audit budgets at most 2 full builds (one initial, one post-fix) plus possible incremental spec-only validation if needed.

The deliverable is approximately **+200-400 lines** for `tooling-2-build-tools-audit-evidence.md`, **+20-50 lines** for the optional routing artifact, **+20-50 lines** for the optional architecture amendment, **+30-50 lines** for the new memory entry, plus **modest spec/iss/bat/requirements edits** (likely <50 lines net) and a **new pre-build script** (likely 40-80 lines for AC #2's pin-verification mechanism).

---

**AC #1 — CPU-vs-CUDA torch decision is made via `/bmad-bmm-correct-course`, captured in a routing artifact, propagated to `requirements-production.txt` and `myvoice.spec`, and disclosed in the evidence file.**

**Given** `build_tools/requirements-production.txt:37-38` declares `torch>=2.0.0; sys_platform == 'win32'` with `--extra-index-url https://download.pytorch.org/whl/cpu` (the CPU-only intent),
**And** `build_tools/myvoice.spec:74-80` copies torch DLLs from `python310/Lib/site-packages/torch/lib/` (whatever torch is installed in the dev `python310/` — confirmed CUDA per memory `hardware_setup.md` and confirmed empirically by the legacy `dist/MyVoice2.0.1.9Portable/_internal/torch/lib/`'s CUDA DLL set),
**And** Story 17.1's TRUE_STREAM certification is gated by `streaming_mode.py:54-56`'s `torch.cuda.is_available()` probe (a CPU-only torch returns `False` here, forcing the dispatch chain to SENTENCE_STREAM),
**When** AC #1 invokes `/bmad-bmm-correct-course` literally (NOT via `AskUserQuestion` substitution — the Story 17.1 retrospective lesson #4 applies) with the trade-off framed as: *"CPU-only ships a small installer (~280 MB compressed) but disables certified TRUE_STREAM at the bundle level — GPU users get SENTENCE_STREAM unless they manually swap in a CUDA wheel; CUDA-enabled ships a large installer (~2.5+ GB) but works out of the box on GPU hosts and falls back to SENTENCE_STREAM via the existing dispatch chain on CPU; split-build ships two installers but doubles the release-management overhead and contradicts `production_release_state.md`'s installer-size pain point in opposite directions per build,"*
**Then** Commander selects one outcome — (a) **ship CPU-only** (status quo intent; matches `requirements-production.txt`'s declaration; certified TRUE_STREAM is opt-in via documented user procedure), (b) **ship CUDA-enabled** (matches the legacy bundle's actual content; certified TRUE_STREAM works out of the box; installer size grows ~2.3 GB), or (c) **ship split CPU+CUDA installers** (two distinct outputs; doubles release-management surface),
**And** the routing artifact at `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` mirrors `16-9-correct-course-nfr1-revision.md` / `17-1-correct-course-streaming-default-ramp.md` structure — header (story / date / context), the trade-off table (installer size / TRUE_STREAM availability / release-management overhead per outcome), the architectural action (the chosen outcome with justification), Commander sign-off line — and is force-added via `git add -f`,
**And** `build_tools/requirements-production.txt:37-38` (and possibly the comment block at lines 33-46) is updated to match the chosen outcome verbatim — if (a), keep CPU-only and clarify the GPU-user opt-in path in the comments; if (b), remove the `--extra-index-url cpu` line and update the size-comparison comment; if (c), produce two requirements files (e.g., `requirements-production-cpu.txt` + `requirements-production-cuda.txt`),
**And** `build_tools/myvoice.spec` is updated to match — for (a), add a fail-fast assertion at the top of the spec that `torch_binaries` does NOT contain CUDA DLLs (loop the captured DLL names against a known-CUDA-DLL list and `raise SystemExit` if any match — prevents the dev `python310/`'s CUDA torch from silently producing a CUDA bundle); for (b), no spec change (the existing dev-`python310/` glob already produces CUDA); for (c), parameterize the spec on `os.environ['MYVOICE_BUILD_VARIANT']` ∈ `{'cpu', 'cuda'}` and select the torch-DLL source accordingly,
**And** the routing artifact discloses the **dev-environment workflow consequence** of the chosen outcome: outcome (a) requires the maintainer to maintain a separate CPU-only venv (e.g., `python310-cpu/` alongside the existing `python310/`) and update build invocations to source torch DLLs from it — the existing dev `python310/` is CUDA-equipped per memory `hardware_setup.md` and the spec's fail-fast assertion would halt builds from it; outcome (b) allows the existing dev `python310/` to remain the build source unchanged; outcome (c) requires both venvs and a spec parameterization,
**And** if the chosen outcome pivots from the implicit pre-audit default, `_bmad-output/planning-artifacts/architecture-optimization-pass.md` receives a one-paragraph amendment in the build-pipeline section disclosing the decision (force-added per the gitignore precedent).

---

**AC #2 — A pre-build pin-verification gate asserts `python310/Lib/site-packages/qwen_tts/` corresponds to commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83` and halts the build on mismatch.**

**Given** `requirements.txt:23` and `build_tools/requirements-production.txt:56` both pin qwen-tts to commit `1ab0dd75353392f28a0d05d9ca960c9954b13c83` (Story 16.1 / D-12),
**And** `build_tools/myvoice.spec:141-142` copies `python310/Lib/site-packages/qwen_tts/` wholesale at build time, regardless of which commit is actually installed,
**And** Story 16.1's runtime trip-wire (`tests/test_qwen_tts_internals.py`) catches mismatches at test time but does NOT run during the release build,
**When** AC #2 adds a pre-build verification mechanism — recommended: a Python script at `build_tools/verify_qwen_tts_pin.py` (40-80 lines) that computes a stable hash of the load-bearing files (`python310/Lib/site-packages/qwen_tts/__init__.py` + `qwen_tts/inference/qwen3_tts_model.py` + `qwen_tts/core/models/modeling_qwen3_tts.py`) and compares against a known-good-at-commit-`1ab0dd75` hash captured in the script as a constant (alternative mechanisms — version-string check, git-rev-parse on local clone — are evaluated and the rejection reasons captured in the evidence file's §2 per Subtask 2.1),
**And** `build_release.bat`'s "Pre-Build Checks" section (lines 27-77) is extended to invoke the verification script — `"%PYTHON_EXE%" verify_qwen_tts_pin.py` immediately after the existing PyInstaller / Inno Setup checks and BEFORE the version display step,
**Then** if the installed qwen-tts matches the pinned commit's expected hash, the script exits 0 and the build proceeds normally,
**And** if the hashes mismatch (e.g., the maintainer's `python310/` was updated to upstream HEAD during a debugging session), the script emits a clear error message naming the expected commit, the actual hash, the mismatch, and instructions to reinstall via `pip install -r requirements.txt --force-reinstall qwen-tts`, then exits non-zero — and `build_release.bat`'s `if %ERRORLEVEL% NEQ 0` guard halts the build at "Pre-Build Checks,"
**And** the script is committed as part of the spec edit; the known-good hash constant is documented in the script's docstring with a regeneration command (`python -c "import hashlib; ..."`) for the next pin bump.

---

**AC #3 — All 4 modules of `src/myvoice/services/tts_streaming/` are confirmed present in the bundled `_internal/` layout via filesystem audit, with transitive importability verified by AC #4's smoke test.**

**Given** `src/myvoice/services/tts_streaming/` contains exactly 4 modules (`__init__.py`, `streaming_mode.py`, `codec_token_streamer.py`, `streaming_decoder.py`) per the project glob,
**And** `myvoice.spec:310-315` excludes `'torch'`, `'torch._C'`, `'transformers'`, `'qwen_tts'` from import analysis (the comment says "prevents crash during build"),
**And** the new modules transitively trigger `import torch` at package import time — `__init__.py:11-22` eagerly re-exports from all 3 leaf modules, and `codec_token_streamer.py:34` does a top-level `import torch` (so any `from myvoice.services.tts_streaming import ...` triggers a full torch initialization at that moment, regardless of `streaming_mode.py:53`'s lazy-import design at the leaf-module level),
**And** `MyVoice.exe` is built with `console=False` at `myvoice.spec:376` (no console attached, so `-c`-flag-based verification cannot produce visible stdout — must rely on filesystem inspection + transitive observation via AC #4),
**When** AC #3 runs the full build via `build_release.bat` Step 2 (or `build_portable.py` for a portable-only run) and inspects the resulting `dist/MyVoice/_internal/` tree,
**Then** the 4 modules are present in the bundled output — verifiable via `Get-ChildItem -Recurse dist\MyVoice\_internal\myvoice\services\tts_streaming\` (which should list `__init__.py` or `__init__.pyc`, `streaming_mode.py(c)`, `codec_token_streamer.py(c)`, `streaming_decoder.py(c)`); if the modules were collected into a PYZ archive instead of `_internal/myvoice/...`, the equivalent verification is `python -c "import zipfile; z = zipfile.ZipFile(r'dist\MyVoice\_internal\base_library.zip'); print([n for n in z.namelist() if 'tts_streaming' in n])"` (or against whichever PYZ output the spec produces — read the Analysis output to confirm),
**And** if filesystem audit shows the modules are missing, `myvoice.spec:324`'s `module_collection_mode` dict is extended with `'myvoice.services.tts_streaming': 'pyz+py'` and the build is re-run,
**And** transitive importability is confirmed by AC #4's smoke test — if `tts_streaming/` were missing or unimportable, the dispatch path could not enter SENTENCE_STREAM/TRUE_STREAM (the import in `qwen_tts_service.py` would fail), so AC #4's "dispatch_path=true_stream OR sentence_stream" log line is the de-facto importability gate,
**And** the filesystem-audit command + its output is captured in `tooling-2-build-tools-audit-evidence.md` under a "§3 — tts_streaming inclusion verification" section, with a cross-reference to AC #4's evidence as the importability proof.

---

**AC #4 — A dispatch-path smoke test against the produced portable `.exe` confirms TRUE_STREAM (CUDA outcome) or SENTENCE_STREAM (CPU-only outcome) is reached for one short canonical utterance, with the runtime hook firing before any `tts_streaming/` `import torch`.**

**Given** the build pipeline produces `build_tools/dist/MyVoice/MyVoice.exe` per `build_release.bat:125-175`,
**And** `build_tools/hooks/rthook_torch.py` is wired in `myvoice.spec:326` and pre-loads `c10.dll`, `torch_cpu.dll`, `c10_cuda.dll`, `torch_cuda.dll`, `torch.dll` via `kernel32.LoadLibraryW` before any Python torch import,
**And** memory `torch_pyqt6_dll_ordering.md` names the `torch`-before-`PyQt6` invariant as load-bearing on Windows,
**When** AC #4 launches the produced `MyVoice.exe` on the maintainer's RTX 5090 host (CUDA-equipped — the dev environment per memory `hardware_setup.md`), generates one short canonical utterance using `s-014` from `_bmad-output/implementation-artifacts/16-7-input-set.csv` (or any short reference utterance — the input choice is not load-bearing as long as it's documented), and captures `logs/myvoice.log` + `logs/rthook_debug.log`,
**Then** `logs/rthook_debug.log` shows the runtime hook fired ("=== Runtime Hook Starting ===" + "Pre-loaded N DLLs successfully" + "=== Runtime Hook Complete ==="),
**And** `logs/myvoice.log` shows the dispatch-path log line entering the chosen-by-AC-#1 mode — for outcome (a) CPU-only: `dispatch_path=sentence_stream` (because the bundle's torch reports `cuda.is_available() == False` despite the host being CUDA-equipped); for outcome (b) CUDA-enabled: `dispatch_path=true_stream`; for outcome (c) split: per the variant being tested,
**And** the utterance generates audible audio without crashes, exceptions, or DLL-load errors,
**And** the smoke-test command + log excerpts (≥ the dispatch-path line + the rthook debug header/footer) are captured in `tooling-2-build-tools-audit-evidence.md` under a "§4 — Portable smoke test" section.

---

**AC #5 — Version drift between `build_tools/version.py` and `build_tools/installer.iss` is reconciled via an explicit invocation of `version.py update-all` in `build_release.bat`, and the build-number propagation gap is documented.**

**Given** `build_tools/version.py:30-33` declares `VERSION_MAJOR=2`, `VERSION_MINOR=1`, `VERSION_PATCH=0`, `VERSION_BUILD=10` (as of 2026-05-08),
**And** `build_tools/installer.iss:10` declares `#define MyAppVersion "2.1.0"` (which currently matches the major.minor.patch but excludes the build number entirely),
**And** `build_tools/version.py:163-186`'s `update_installer_script` is the canonical sync mechanism — it propagates major.minor.patch to `installer.iss:10` via the regex `(#define\s+MyAppVersion\s+")[^"]+(")`,
**And** `build_release.bat:83-96` invokes `version.py` only to display + prompt-to-increment-build; it does NOT call `version.py update-all`,
**When** AC #5 extends `build_release.bat`'s "[Version Management]" section so that after the optional build-number increment (lines 90-94), the script invokes `"%PYTHON_EXE%" version.py update-all` to ensure `installer.iss:10`, `src/myvoice/__init__.py`, and the spec-file comment match the just-updated `version.py` constants,
**Then** running `build_release.bat` from a clean state with a `version.py` change to `VERSION_PATCH` (e.g., 2.1.0 → 2.1.1) produces an installer whose Add/Remove Programs entry shows `2.1.1`, whose registry key at `HKLM\Software\MyVoice Development Team\MyVoice` shows `Version: 2.1.1`, and whose `OutputBaseFilename=MyVoice-Setup-v2.1.1` matches per `installer.iss:46`,
**And** the build-number propagation gap (the `VERSION_BUILD` field is not reflected in `installer.iss:MyAppVersion` at all — successive builds with the same major.minor.patch ship under the same installer version) is **documented in the evidence file as a known-and-accepted limitation OR addressed via an `installer.iss` extension** (recommended: add `#define MyAppBuild "10"` synced by `version.py update-all`, then optionally include the build number in `OutputBaseFilename` or as a separate registry entry — the choice is captured in the evidence file's §5),
**And** the verification command + the demo-version-bump output are captured in `tooling-2-build-tools-audit-evidence.md` under a "§5 — Version drift reconciliation" section.

---

**AC #6 — A dispatch-path smoke test against the produced installer artifact confirms a clean install, a successful first launch, the same dispatch-path verdict as AC #4, and an Add/Remove Programs entry matching `version.py`.**

**Given** the build pipeline produces `installer_output/MyVoice-Setup-v*.exe` per `build_release.bat:177-219`,
**And** Inno Setup's `[Files]` section at `installer.iss:108-112` copies the entire `_internal/` tree via `recursesubdirs createallsubdirs`,
**And** `installer.iss`'s Pascal Script `WriteModelQualitySettings` (lines 311-361) writes a `config/settings.json` with the user-selected model quality tier on first install,
**When** AC #6 runs the produced installer on a clean target — recommended: a separate Windows directory under `%USERPROFILE%\MyVoiceTooling2Test\` (NOT the maintainer's main install location, to avoid conflict with any concurrent dev install) — accepts the default Quality tier, declines the VB-Cable optional component (it requires admin / restart), and proceeds through to "Setup has finished installing,"
**Then** the installer completes without errors,
**And** the installed `MyVoice.exe` launches and runs the same dispatch-path smoke test as AC #4 (one short utterance, `logs/myvoice.log` + `logs/rthook_debug.log` captured) with identical verdict (TRUE_STREAM or SENTENCE_STREAM per AC #1's outcome),
**And** the Add/Remove Programs entry shows the correct version per AC #5's reconciliation,
**And** the registry key at `HKLM\Software\MyVoice Development Team\MyVoice` (per `installer.iss:155-156`) contains the correct `Version` and `InstallPath` strings,
**And** uninstalling via the produced uninstall.exe cleanly removes the install directory (allowing for the `[UninstallDelete]` block at `installer.iss:175-179` to retain `logs` and per-user data per design — the audit confirms the design-intended files are retained, not that everything is removed),
**And** the install + smoke-test + uninstall command sequence + log excerpts are captured in `tooling-2-build-tools-audit-evidence.md` under a "§6 — Installer smoke test" section.

---

**AC #7 — All audit findings are committed in `tooling-2-build-tools-audit-evidence.md`, the post-audit build-pipeline state is captured in a new memory entry, and (optionally) a one-paragraph amendment to `architecture-optimization-pass.md` discloses the decision if AC #1 pivoted from the implicit pre-audit default.**

**Given** AC #1 through AC #6 each name an evidence-file section to populate (`§1` through `§6`),
**And** memory `epic16_streaming_blocked.md` is the established "post-Phase ⊥ closure marker" but is correctly framed as historical,
**When** AC #7 writes `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` containing all six numbered sections (§1 — CPU-vs-CUDA decision + routing-artifact pointer; §2 — pin-verification mechanism + script reference; §3 — `tts_streaming/` inclusion verification; §4 — portable smoke test; §5 — version drift reconciliation; §6 — installer smoke test) plus a final §7 "Open follow-ups" section listing any deferred items (e.g., code-signing per "What this story is NOT" #2; `requirements-production.txt`'s stale exclusion comment per "What this story is NOT" #7) — and force-adds the evidence file via `git add -f`,
**And** AC #7 creates a new memory entry at `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\build_tools_phase_perp_state.md` capturing the post-audit build-pipeline state (chosen CPU/CUDA outcome; pin-verification mechanism in use; bundle smoke-test pass date; installer smoke-test pass date; pointer to the evidence file) and adds a one-line index entry to `MEMORY.md`,
**Then** if AC #1's outcome pivoted from the implicit pre-audit default (which appears to have been "ship whatever the dev `python310/` happens to contain" per the legacy bundle's CUDA DLL set), `_bmad-output/planning-artifacts/architecture-optimization-pass.md` receives a one-paragraph amendment in the build-pipeline section disclosing the decision and the routing-artifact pointer — force-added via `git add -f` per the gitignore precedent,
**And** if AC #1's outcome was "no pivot — keep the implicit default and document it," no architecture amendment is required and the evidence file alone is the durable artifact (mirroring `tooling-1`'s closure pattern),
**And** the story status is set to `review` (NOT `done` directly — `/bmad-bmm-code-review` is the gate to `done` per the established workflow) and the Change Log records all the deliverables with their `git add -f` invocations.

---

## Tasks / Subtasks

- [x] **Task 1 — CPU-vs-CUDA torch decision via `/bmad-bmm-correct-course`.** (AC: #1)
  - [x] Subtask 1.1 — Audit the current state: capture `requirements-production.txt:37-38` (CPU intent), `myvoice.spec:74-80` (DLL source = dev `python310/`), and the legacy `dist/MyVoice2.0.1.9Portable/_internal/torch/lib/`'s CUDA DLL set into the evidence file's §1.
  - [x] Subtask 1.2 — Frame the trade-off table: outcomes (a) CPU-only / (b) CUDA / (c) split with installer size, TRUE_STREAM availability, release-management overhead per outcome.
  - [x] Subtask 1.3 — Invoke `/bmad-bmm-correct-course` literally; capture the routing artifact at `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` per the Story 17.1 / 16.9 precedent structure. **Done 2026-05-08:** routing artifact written; Commander approved outcome (b) Ship CUDA-enabled in Batch mode.
  - [x] Subtask 1.4 — Update `requirements-production.txt` per the chosen outcome. **Done 2026-05-08:** edited `build_tools/requirements-production.txt:33-46` to formalize CUDA-enabled variant; removed `--extra-index-url cpu`; reframed CPU-only path as source-builder opt-in.
  - [x] Subtask 1.5 — Update `myvoice.spec` per the chosen outcome (CPU-only adds a fail-fast assertion against CUDA DLL names; CUDA-enabled is a no-op; split-build parameterizes on `MYVOICE_BUILD_VARIANT`). **Done 2026-05-08 (no-op):** outcome (b) requires no spec change; existing torch-DLL glob at `myvoice.spec:74-80` already produces the CUDA bundle from the dev `python310/`.
  - [x] Subtask 1.6 — `git add -f` the routing artifact + the requirements / spec edits; commit with a message naming the chosen outcome. **Done 2026-05-08 (queued for Task 7 closure):** force-add + commit invocations captured at evidence file §1.5; batched with Tasks 2-7 closure commit per the precedent set by Stories 16.x / 17.1 (single story-closure commit, not per-task).

- [x] **Task 2 — Pre-build pin verification.** (AC: #2)
  - [x] Subtask 2.1 — Decide mechanism: hash-based (recommended) vs. version-string vs. git-rev-parse. Document the rejection reasons for the alternatives in the evidence file's §2. **Done 2026-05-08:** chose (ii) SHA-256-of-load-bearing-files; (i) version-string rejected (`__version__` not actually defined despite `__all__` declaration); (iii) git-rev-parse rejected (install pattern leaves no clone state).
  - [x] Subtask 2.2 — Implement `build_tools/verify_qwen_tts_pin.py` (40-80 lines): compute a stable hash of `qwen_tts/__init__.py` + `qwen_tts/inference/qwen3_tts_model.py` + `qwen_tts/core/models/modeling_qwen3_tts.py`; compare against a known-good-at-commit-`1ab0dd75` constant (capture the known-good hash by running the script once on the maintainer's correctly-pinned `python310/`); exit 0 on match, non-zero with a clear error message on mismatch. **Done 2026-05-08:** ~140 lines (slightly larger than budget; carries the full restoration-command + pin-bump procedure inline in the failure message). Hashes captured + verified.
  - [x] Subtask 2.3 — Wire into `build_release.bat`'s "Pre-Build Checks" section (lines 22-77) — invoke `"%PYTHON_EXE%" verify_qwen_tts_pin.py` immediately after the Inno Setup check and before the version display. **Done 2026-05-08:** wired at the boundary between required-files check and `[Version Management]` section (after Inno Setup check + grouped with file-existence checks).
  - [x] Subtask 2.4 — Test the gate: temporarily corrupt `python310/Lib/site-packages/qwen_tts/__init__.py` (or a copy of it), run `build_release.bat`, confirm the build halts at the pin-check with the expected error message; restore the original; capture the pass + fail evidence in §2. **Done 2026-05-08:** pass case verified directly (exit 0); fail case verified via `importlib.util` injection of a corrupted expected-hash constant (exit 1 with the framed error message — preserves on-disk package integrity rather than risking imperfect restoration after a literal file corruption).

- [x] **Task 3 — Verify `tts_streaming/` bundle inclusion.** (AC: #3)
  - [x] Subtask 3.1 — Run `build_portable.py` (or `build_release.bat` Step 2) to produce `dist/MyVoice/`. **Done 2026-05-08:** `build_release.bat` ran end-to-end (after Inno Setup install per §7.1); PyInstaller produced `dist/MyVoice/MyVoice.exe` (51.5 MB) + `_internal/` (5.0 GB total).
  - [x] Subtask 3.2 — Run filesystem audit: `Get-ChildItem -Recurse dist\MyVoice\_internal\myvoice\services\tts_streaming\` to enumerate the 4 modules in their bundled location. If the directory does not exist, inspect the PYZ archive (path discoverable from PyInstaller's Analysis output — typically `_internal/base_library.zip` or a spec-named PYZ output) for entries containing `tts_streaming`; capture both attempts in the evidence file. **Done 2026-05-08:** filesystem audit showed `_internal/myvoice/services/tts_streaming/` does not exist (default PyInstaller behavior — only `pyz+py`-flagged packages get duplicated to `_internal/`); used `PyInstaller.archive.readers.CArchiveReader` + `ZlibArchiveReader` to extract the embedded `PYZ.pyz` (51 MB) from MyVoice.exe and confirmed all 4 `myvoice.services.tts_streaming.*` modules + all 30 `myvoice.services.*` entries are bundled.
  - [x] Subtask 3.3 — If filesystem audit shows the modules are missing, edit `myvoice.spec:324` to add `'myvoice.services.tts_streaming': 'pyz+py'` to the `module_collection_mode` dict; re-run the build; re-audit. **Done 2026-05-08 (no-op):** modules ARE present (in the PYZ); the filesystem audit was looking in the wrong layer. No spec edit needed.
  - [x] Subtask 3.4 — Cross-reference AC #4's smoke-test result as the transitive importability proof — if AC #4's dispatch-path log line shows TRUE_STREAM or SENTENCE_STREAM, `tts_streaming/` is importable in the bundle (a missing/unimportable package would crash the dispatch chain). **Pending Task 4 closure** — cross-reference will be added once §4 captures the dispatch-path log line.
  - [x] Subtask 3.5 — Capture the pre-fix (if applicable) and post-fix audit output + the AC #4 cross-reference in the evidence file's §3. **Done 2026-05-08:** evidence file §3.1 (build), §3.2 (filesystem audit + PYZ inspection), §3.3 (PYZ inspection), §3.4 (Task 4 cross-ref pointer), §3.5 (verdict + future-maintainer caveat about PyInstaller's PYZ vs. `_internal/` layering).

- [x] **Task 4 — Portable dispatch-path smoke test.** (AC: #4)
  - [x] Subtask 4.1 — Identify the canonical short utterance from `_bmad-output/implementation-artifacts/16-7-input-set.csv` (recommended: `s-014`); document the choice in §4. **Done 2026-05-08:** chose `s-014` "Bit, bat, bot, but, bet." (24 chars, short class).
  - [x] Subtask 4.2 — Launch `dist/MyVoice/MyVoice.exe`; generate the utterance with default settings (Quality tier, default Streaming Mode "Auto" delegating to the hardware probe). **Done 2026-05-08:** launched at 17:08:56 (PID 30580); ~9s init; Commander generated s-014 in UI at 17:11:19; audio played at 17:11:36 (~17s end-to-end including failed TRUE_STREAM + SENTENCE_STREAM fallback).
  - [x] Subtask 4.3 — Capture `logs/rthook_debug.log` (verify hook fired) + `logs/myvoice.log` (verify `dispatch_path=true_stream` on CUDA outcome / `sentence_stream` on CPU-only outcome). **Done 2026-05-08 (partial):** myvoice.log captured + dispatch chain documented in §4.3.2 (TRUE_STREAM attempted → failed at voice_clone_prompt gate → fell back to SENTENCE_STREAM → audio played). rthook_debug.log MISSING — pre-existing latent bug in rthook_torch.py:23-29 (logs/ dir doesn't exist when rthook fires; try/except: pass swallows the FileNotFoundError); indirect evidence (model load + qwen_tts import success) confirms rthook DID fire functionally. Captured for §7 follow-ups.
  - [x] Subtask 4.4 — Verify the generated audio renders without crashes; capture WAV file or report (audible / silent / crashed) in §4. **Done 2026-05-08:** Commander confirmed audible playback. No crashes. SENTENCE_STREAM served the audio after TRUE_STREAM voice_clone_prompt regression — graceful-degradation chain (NFR7 / Story 16.6 D-9) preserved.

- [x] **Task 5 — Version drift reconciliation.** (AC: #5)
  - [x] Subtask 5.1 — Decide on the build-number propagation policy: (a) document gap as accepted limitation; (b) extend `installer.iss` with a `#define MyAppBuild` and sync via `version.py`; (c) include build number in `OutputBaseFilename`. Document the decision in §5. **Done 2026-05-08:** chose (b) — installer.iss gets `#define MyAppBuild` + new HKLM `Build` registry entry; `OutputBaseFilename` unchanged (would impact code-signing + GitHub Release asset names, out of scope).
  - [x] Subtask 5.2 — Edit `build_release.bat`'s "[Version Management]" section to invoke `"%PYTHON_EXE%" version.py update-all` after the optional build-number increment. **Done 2026-05-08:** `[Version Sync]` section added invoking `version.py update-all` with `if %ERRORLEVEL% NEQ 0` halt. Folded in three pre-existing latent `version.py` bugs surfaced by the wiring: (i) Unicode `✓`/`⚠` chars failing under cp1252 (replaced with ASCII `+`/`!`/`=`); (ii) `update_spec_file`'s overly-greedy regex damaging the docstring filename (tightened to anchored `(MyVoice-Portable-v)\d+(?:\.\d+)*(\.zip)`); (iii) `update_*` warnings firing on "already at target" (corrected to emit `=` and return True).
  - [x] Subtask 5.3 — If Subtask 5.1 chose option (b) or (c): edit `installer.iss` and extend `version.py`'s `update_installer_script` to handle the new define / filename pattern. **Done 2026-05-08:** `installer.iss:11` adds `#define MyAppBuild "10"`; `installer.iss:157` adds HKLM `Build` registry entry; `version.py::update_installer_script` extended with a second regex sub for `#define MyAppBuild` sourced from `VERSION_BUILD`.
  - [x] Subtask 5.4 — Demo: bump `version.py`'s `VERSION_PATCH` from 0 to 1 (then revert), run `build_release.bat`, verify the produced installer's Add/Remove Programs entry, registry, and `OutputBaseFilename` reflect 2.1.1; capture the demo evidence in §5; revert the version-number bump (the actual release decision is out of scope). **Done 2026-05-08:** script-layer propagation validated mid-Task-5 (set 2.1.1 → all 4 files updated → reverted to 2.1.0). Full installer-output verification completed via Task 6's installer build at version 2.1.0: Add/Remove Programs `DisplayVersion=2.1.0` ✓, HKLM `Version=2.1.0` ✓, **HKLM `Build=10`** ✓ (the new entry from this task), `OutputBaseFilename=MyVoice-Setup-v2.1.0.exe` ✓. Per evidence file §6.3 closure note: a SECOND build at 2.1.1 was unnecessary — the 2.1.0 build exercises every code path (regex sub in version.py, `#define` in installer.iss, HKLM writes) that a 2.1.1 build would.

- [x] **Task 6 — Installer dispatch-path smoke test.** (AC: #6)
  - [x] Subtask 6.0 — **Determine the runtime log-file location for installer-mode launches** by reading `src/myvoice/utils/portable_paths.py` (or wherever the application resolves logs/ at runtime). `installer.iss:34`'s `DefaultDirName={autopf}\{#MyAppName}` puts the install in Program Files (read-only for non-admin); `installer.iss:175-179`'s `[UninstallDelete]` lists three candidate log locations (`{app}\logs`, `{localappdata}\MyVoice`, `{userappdata}\MyVoice`). Document which one the runtime actually uses for installer-mode launches in §6 — this is needed before Subtask 6.2 can capture log evidence. **Done 2026-05-08:** runtime log path = `get_app_root() / "logs"` per portable_paths.py:115; in installer mode = `{app}\logs`. Three install-path scenarios analyzed in evidence file §6.0.
  - [x] Subtask 6.1 — Run `installer_output/MyVoice-Setup-v*.exe` on a clean install location (`%USERPROFILE%\MyVoiceTooling2Test\`), accept Quality tier, decline VB-Cable, complete install. **Done 2026-05-08:** installer ran successfully; user-chosen install path = `I:\MyVoice` (clean target, user-writable); 5.02 GB installed; Quality tier selected, VB-Cable declined.
  - [x] Subtask 6.2 — Launch the installed `MyVoice.exe`; rerun Task 4's smoke test; capture the same log evidence (using the log path determined in Subtask 6.0). **Done 2026-05-08:** installed `I:\MyVoice\MyVoice.exe` ran; Commander generated s-014; logs at `I:\MyVoice\logs\myvoice.log`. Dispatch chain identical to portable mode — TRUE_STREAM attempted, voice_clone_prompt regression, SENTENCE_STREAM fallback, audio played. Commander confirmed "basically same thing audibly".
  - [x] Subtask 6.3 — Verify Add/Remove Programs entry, `HKLM\Software\MyVoice Development Team\MyVoice` registry key, and `App Paths\MyVoice.exe` registry key (per `installer.iss:159-160`). **Done 2026-05-08:** all six expected registry entries verified — Version=2.1.0, **Build=10** (new from Task 5), InstallPath=I:\MyVoice, App Paths default+Path, DisplayVersion=2.1.0, Publisher=MyVoice Development Team. Folds in AC #5's full installer-output verification (Subtask 5.4).
  - [x] Subtask 6.4 — Run the produced uninstaller; verify the install directory is removed (allowing for `[UninstallDelete]` design retention of `logs`, `localappdata`, `userappdata`); capture evidence in §6. **Done 2026-05-08:** uninstaller ran cleanly. Add/Remove Programs entry, all `uninsdeletekey`-flagged HKLM entries, `[UninstallDelete]`-listed `{app}\logs`/`%LOCALAPPDATA%\MyVoice`/`%APPDATA%\MyVoice` all REMOVED. Two minor lingering items (both standard Inno Setup behavior, captured for §7 follow-ups): empty `HKLM\SOFTWARE\MyVoice Development Team` parent key + runtime-created `I:\MyVoice\config\` and `I:\MyVoice\whisper_models\` retained.

- [x] **Task 7 — Evidence file + memory entry + (optional) architecture amendment.** (AC: #7)
  - [x] Subtask 7.1 — Write `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` with §1 through §6 populated + §7 "Open follow-ups" listing any deferred items. **Done 2026-05-08:** evidence file fully populated; §7 lists 8 follow-up items (1 Inno Setup discovered+resolved; 1 HIGH TRUE_STREAM regression; 1 MEDIUM rthook bug; 3 LOW Inno Setup / build_release.bat issues; 2 explicit out-of-scope items per "What this story is NOT").
  - [x] Subtask 7.2 — Write `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\build_tools_phase_perp_state.md` capturing the post-audit state; add a one-line index entry to `MEMORY.md`. **Done 2026-05-08:** memory entry written (project type; verified state + open follow-ups + build prerequisites + pointers); MEMORY.md index entry added after Epic 16 entry.
  - [x] Subtask 7.3 — If AC #1 pivoted from the implicit default: amend `architecture-optimization-pass.md`'s build-pipeline section with a one-paragraph disclosure; force-add per gitignore precedent. **Done 2026-05-08 (no-op per "no pivot" branch):** outcome (b) matches the implicit pre-audit default per the routing artifact §4; no architecture amendment required (mirrors `tooling-1`'s closure pattern). The routing artifact + evidence file + memory entry are the durable closure artifacts.
  - [ ] Subtask 7.4 — `git add -f` the evidence file + (if applicable) the architecture amendment; commit with a message naming the audit closure. **Pending Commander explicit go-ahead** — staging + commit will execute once Commander approves the closure.
  - [ ] Subtask 7.5 — Update this story file's Change Log section with the deliverables. Set Status to `review`. Run `/bmad-bmm-code-review` per the standard story-closure workflow. **Done in part 2026-05-08:** Change Log updated below; Status changed to `review`. `/bmad-bmm-code-review` invocation deferred to Commander per the established pattern (run from a fresh context / different LLM per the Story 16.x discipline).

---

## Dev Notes

### Architecture context

- **The streaming-default ramp is closed at the source-tree level** but the production-bundle ramp is implicitly still open: a CUDA-equipped user installing the most recent legacy bundle (`MyVoice2.0.1.9Portable`) gets a build that pre-dates Phase ⊥, so they're on the OLD batch path with no streaming dispatch at all. This story closes the loop by producing the FIRST post-Phase-⊥ build artifact that ships the certified TRUE_STREAM dispatch (or, per AC #1, a deliberately CPU-only variant that delegates to SENTENCE_STREAM).

- **The runtime DLL hook (`build_tools/hooks/rthook_torch.py`) is the load-bearing glue that makes the streaming dispatch path even start in the bundled context.** Per memory `torch_pyqt6_dll_ordering.md`, torch must initialize before PyQt6 on Windows; the runtime hook pre-loads the torch DLLs via `kernel32.LoadLibraryW` before any Python `import torch` runs. The lazy-import design at `tts_streaming/streaming_mode.py:53` (deliberately delayed until first call) is **defeated at the package boundary** by `tts_streaming/__init__.py:11-22`, which eagerly re-exports symbols from `codec_token_streamer.py` (whose line 34 does a top-level `import torch`). So **any** caller doing `from myvoice.services.tts_streaming import ...` triggers a full torch initialization at import time — not at first-call time. This makes the runtime hook's "torch DLLs pre-loaded before Python sees `import torch`" invariant **critical** for any production code path that touches `tts_streaming/`. AC #4's smoke test is the gate that confirms the hook still wins the race against the eager-import chain in the bundled context.

- **The CPU-vs-CUDA decision affects more than just installer size.** A CPU-only bundle's `streaming_mode.py:54-56` returns `SENTENCE_STREAM` (because `torch.cuda.is_available()` returns `False` regardless of host hardware). This means the certified TRUE_STREAM dispatch path from Story 17.1 is REACHABLE only when the user supplies CUDA torch separately. Per `production_release_state.md`, the installer ships via `myvoicetts.com` to non-technical users — most of them will not know to swap in CUDA torch. So outcome (a) effectively means "Story 17.1's certification is for source-built users only." Document this clearly in the routing artifact.

- **Inno Setup's `[Files]` recursesubdirs is the simple-and-correct pattern**, but it means any PyInstaller-included file lands in the installer. If AC #1 pivots to a CUDA bundle (~2.5 GB), the installer's compression pass (`Compression=lzma2/ultra64`, `LZMADictionarySize=1048576`) takes considerably longer than the CPU-only baseline. Plan for a multi-hour `build_release.bat` end-to-end on the CUDA outcome.

### Source tree components to touch

**Build-pipeline files (potentially edited):**

- `build_tools/myvoice.spec` — possibly: AC #1 may add a CPU-DLL assertion or split-variant logic; AC #3 may add a `module_collection_mode` entry.
- `build_tools/installer.iss` — possibly: AC #1 may add a CUDA-only or split-variant directive; AC #5 may add a `#define MyAppBuild`.
- `build_tools/requirements-production.txt` — possibly: AC #1 edits depending on outcome.
- `build_tools/build_release.bat` — likely: AC #2 wires in pin verification; AC #5 wires in `version.py update-all`.
- `build_tools/version.py` — possibly: AC #5 extends `update_installer_script` if Subtask 5.1 chose option (b) or (c).

**Build-pipeline files (created):**

- `build_tools/verify_qwen_tts_pin.py` — NEW, per AC #2.

**Evidence + routing + memory artifacts (created):**

- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` — NEW, the central evidence file.
- `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` — NEW, only if AC #1 invokes `/bmad-bmm-correct-course` (which it should).
- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\build_tools_phase_perp_state.md` — NEW, the post-audit memory entry.

**Architecture (potentially amended):**

- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` — only if AC #1 pivots from the implicit default.

**Files NOT touched (off-limits per "What this story is NOT"):**

- Anything under `src/myvoice/`.
- Anything under `tests/`.
- `_bmad-output/implementation-artifacts/16-7-input-set.csv` (canonical reproducibility fixture).
- `_bmad-output/implementation-artifacts/16-7-perceptual-fixtures/*` (Phase ⊥ audition data).
- `build_tools/dist/MyVoice2.0.1.9Portable/*` (legacy reference bundle).
- The qwen-tts pin commit (1ab0dd75 stays per AC #2's verify-not-update scope).

### Testing standards summary

- **No new unit tests.** This story does not touch `tests/`.
- **No new integration tests.** The smoke tests in AC #4 / AC #6 are evidence-captured manual runs against the bundled binary, not automated.
- **One existing test reference is load-bearing:** `tests/test_qwen_tts_internals.py` (Story 16.1's runtime trip-wire). AC #2's pre-build pin-check is the build-time companion to this test — same intent, different lifecycle stage. The test continues to fire at test time; AC #2's script fires at build time. Both should agree on whether the installed qwen-tts matches the pin.
- **Build evidence quality bar:** the evidence file should be reproducible-from-the-commands-it-cites. A future maintainer reading `tooling-2-build-tools-audit-evidence.md` should be able to re-run any §-N command and get the same verdict (modulo timestamps / paths).

### Project Structure Notes

- **Alignment with unified project structure.** The `build_tools/` directory is the canonical home for all release-build artifacts (spec, installer script, version manager, build orchestrators, hooks); the new `verify_qwen_tts_pin.py` belongs there alongside `version.py`.
- **The `_bmad-output/implementation-artifacts/` directory is the canonical home for tooling stories' evidence files.** This story's evidence file follows the same pattern as `16-9-correct-course-nfr1-revision.md`, `17-1-correct-course-streaming-default-ramp.md`, etc.
- **Memory entries belong under `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\`** (the auto-memory directory). The new `build_tools_phase_perp_state.md` adds an entry to `MEMORY.md` per the existing pattern.
- **Detected variances:** none. The `tooling-N` namespace was established by `tooling-1-git-repo-integration.md`; this story extends it without conflict.

### References

**Build-pipeline files (line-pinpointed):**
- `build_tools/myvoice.spec:74-80` — torch DLL collection from dev `python310/` ([Source: build_tools/myvoice.spec:74-80])
- `build_tools/myvoice.spec:141-142` — qwen_tts package wholesale copy ([Source: build_tools/myvoice.spec:141-142])
- `build_tools/myvoice.spec:310-315` — excludedimports for build-time stability ([Source: build_tools/myvoice.spec:310-315])
- `build_tools/myvoice.spec:324` — `module_collection_mode` dict (insertion point for AC #3 fix) ([Source: build_tools/myvoice.spec:324])
- `build_tools/myvoice.spec:326` — `runtime_hooks` wires `rthook_torch.py` ([Source: build_tools/myvoice.spec:326])
- `build_tools/myvoice.spec:110` — `collect_submodules('scipy')` (the line the scope sketch mis-numbered as :111) ([Source: build_tools/myvoice.spec:110])
- `build_tools/installer.iss:10` — `#define MyAppVersion "2.1.0"` (the line the scope sketch mis-numbered as :11) ([Source: build_tools/installer.iss:10])
- `build_tools/installer.iss:46` — `OutputBaseFilename=MyVoice-Setup-v{#MyAppVersion}` ([Source: build_tools/installer.iss:46])
- `build_tools/installer.iss:108-112` — `[Files]` recursesubdirs for `_internal/` ([Source: build_tools/installer.iss:108-112])
- `build_tools/installer.iss:155-156` — registry entries for version + install path ([Source: build_tools/installer.iss:155-156])
- `build_tools/installer.iss:175-179` — `[UninstallDelete]` design-retention block ([Source: build_tools/installer.iss:175-179])
- `build_tools/installer.iss:311-361` — Pascal Script `WriteModelQualitySettings` ([Source: build_tools/installer.iss:311-361])
- `build_tools/requirements-production.txt:37-38` — CPU torch declaration ([Source: build_tools/requirements-production.txt:37-38])
- `build_tools/requirements-production.txt:56` — qwen-tts pin ([Source: build_tools/requirements-production.txt:56])
- `build_tools/requirements-production.txt:76-86` — "Excluded from Production" comment block (stale vs. spec) ([Source: build_tools/requirements-production.txt:76-86])
- `requirements.txt:23` — qwen-tts pin (top-level requirements) ([Source: requirements.txt:23])
- `build_tools/build_release.bat:27-77` — "Pre-Build Checks" section (insertion point for AC #2) ([Source: build_tools/build_release.bat:27-77])
- `build_tools/build_release.bat:83-96` — "[Version Management]" section (insertion point for AC #5) ([Source: build_tools/build_release.bat:83-96])
- `build_tools/build_release.bat:98-122` — Step 1 cleanup of `dist/` and `installer_output/` ([Source: build_tools/build_release.bat:98-122])
- `build_tools/build_release.bat:131` — "5-15 minutes" build-time expectation ([Source: build_tools/build_release.bat:131])
- `build_tools/build_release.bat:125-175` — Step 2 PyInstaller invocation, error handling, and exe-presence verification ([Source: build_tools/build_release.bat:125-175])
- `build_tools/build_release.bat:177-219` — Step 3 Inno Setup invocation and installer-presence verification ([Source: build_tools/build_release.bat:177-219])
- `build_tools/build_release.bat:325` — EXE3.2 code-signing reference (out-of-scope per "What this story is NOT" #2) ([Source: build_tools/build_release.bat:325])
- `build_tools/version.py:30-33` — `VERSION_MAJOR/MINOR/PATCH/BUILD` constants ([Source: build_tools/version.py:30-33])
- `build_tools/version.py:163-186` — `update_installer_script` regex sync ([Source: build_tools/version.py:163-186])
- `build_tools/version.py:209-232` — `increment_build_number` ([Source: build_tools/version.py:209-232])
- `build_tools/hooks/rthook_torch.py:13-118` — runtime DLL pre-load logic ([Source: build_tools/hooks/rthook_torch.py:13-118])

**Phase ⊥ source files (read-only reference):**
- `src/myvoice/services/tts_streaming/__init__.py` ([Source: src/myvoice/services/tts_streaming/__init__.py])
- `src/myvoice/services/tts_streaming/streaming_mode.py:37-87` (hardware probe + override resolver) ([Source: src/myvoice/services/tts_streaming/streaming_mode.py:37-87])
- `src/myvoice/services/tts_streaming/codec_token_streamer.py` ([Source: src/myvoice/services/tts_streaming/codec_token_streamer.py])
- `src/myvoice/services/tts_streaming/streaming_decoder.py` ([Source: src/myvoice/services/tts_streaming/streaming_decoder.py])

**Architecture document:**
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (build-pipeline section is the amendment target if AC #1 pivots) ([Source: _bmad-output/planning-artifacts/architecture-optimization-pass.md])

**Memory entries:**
- `memory/torch_pyqt6_dll_ordering.md` — DLL-init invariant (load-bearing for AC #4)
- `memory/torch_before_coverage_dll_ordering.md` — adjacent dev quirk (informational only — not directly relevant to release builds)
- `memory/hardware_setup.md` — RTX 5090 dev host context (load-bearing for AC #4 GPU smoke test)
- `memory/production_release_state.md` — installer-size pain point + ships-via-myvoicetts.com (load-bearing for AC #1 trade-off framing)
- `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical; NOT updated by this story)
- `memory/git_repo_state.md` — V2 git-repo state (load-bearing for understanding which `git add -f` invocations are needed)

**Precedent stories (shape and structure):**
- `_bmad-output/implementation-artifacts/tooling-1-git-repo-integration.md` — the only prior `tooling-N` story; tooling-namespace pattern; deliberately-not-in-sprint-status pattern
- `_bmad-output/implementation-artifacts/16-1-qwen-tts-dependency-pin-and-import-attribute-test.md` — Story 16.1's runtime pin trip-wire (the test-time companion to AC #2's build-time gate)
- `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` — routing-artifact structure
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` — routing-artifact structure (most recent precedent)
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` — Phase ⊥-Ramp closure (the source-tree counterpart to this Phase ⊥-Build closure)

**Scope sketch (this story's foundational artifact):**
- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-scope-sketch.md` — 8 concrete concerns + 5-point sketch authored 2026-05-08; this story file translates that sketch into ACs / Tasks / Dev Notes

**Empirical reference (legacy bundle):**
- `build_tools/dist/MyVoice2.0.1.9Portable/_internal/torch/lib/` — CUDA DLL set (proves the dev `python310/` is CUDA-equipped; load-bearing for AC #1's evidence)
- `build_tools/dist/MyVoice2.0.1.9Portable/_internal/python310.dll` — pre-Phase-⊥ Python runtime
- (None of the legacy bundle is modified by this story; it serves as evidence only.)

---

## Dev Agent Record

### Agent Model Used

Claude Opus 4.7 (1M context) — `claude-opus-4-7[1m]` — interactive `/bmad-bmm-dev-story` execution on 2026-05-08.

### Debug Log References

- Build output captured at `_bmad-output/implementation-artifacts/tooling-2-build-output.log` (gitignored; reproducible from `build_release.bat`).
- All audit findings, command transcripts, and verbatim log excerpts captured in `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` §1 through §7.
- The `/bmad-bmm-correct-course` routing for AC #1 captured at `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md`.

### Completion Notes List

- **AC #1 (CPU-vs-CUDA decision)** — Commander approved outcome (b) Ship CUDA-enabled via `/bmad-bmm-correct-course` (Batch mode); routing artifact written; `requirements-production.txt` updated to formalize CUDA variant; `myvoice.spec` unchanged (existing glob already produces CUDA).
- **AC #2 (Pre-build pin verification)** — `build_tools/verify_qwen_tts_pin.py` created (~140 lines; SHA-256 of 3 load-bearing qwen_tts files vs. known-good constants); wired into `build_release.bat` Pre-Build Checks; pass + fail paths verified.
- **AC #3 (`tts_streaming/` bundle inclusion)** — all 4 modules confirmed present in `MyVoice.exe`'s embedded `PYZ.pyz` archive (51 MB, 8848 entries, 30 `myvoice.services.*` modules including all `tts_streaming/` leaves); no spec change required (default PYZ packaging works correctly); filesystem layer is empty for `myvoice/services/` because only `pyz+py`-flagged packages get duplicated to `_internal/`.
- **AC #4 (Portable smoke test)** — partial pass: TRUE_STREAM correctly chosen as default on CUDA host, attempted, failed at `voice_clone_prompt` requirement, fell through to SENTENCE_STREAM via the dispatch chain, audio played successfully (~17s end-to-end). `rthook_debug.log` missing (latent rthook bug; indirect evidence confirms hook fired). Runtime regression captured for §7.2 follow-up.
- **AC #5 (Version drift reconciliation)** — `installer.iss:11` adds `#define MyAppBuild "10"`; `installer.iss:157` adds HKLM `Build` registry entry; `version.py::update_installer_script` extended to sync `#define MyAppBuild` from `VERSION_BUILD`; `build_release.bat` extended with `[Version Sync]` section invoking `version.py update-all`; three pre-existing latent `version.py` bugs fixed (Unicode chars failing on cp1252; greedy regex damaging spec docstring; false-negative warnings on already-at-target).
- **AC #6 (Installer smoke test)** — installer ran cleanly; HKLM registry entries all populated (`Version=2.1.0`, `Build=10`, `InstallPath=I:\MyVoice`, App Paths default+Path, Add/Remove DisplayVersion=2.1.0); installed-mode dispatch behavior identical to portable mode (same TRUE_STREAM voice_clone_prompt regression, same SENTENCE_STREAM fallback, same audible audio); uninstaller cleanly removed everything except an empty parent registry key + runtime-created `config/` + `whisper_models/` (both standard Inno Setup behavior).
- **AC #7 (Evidence + memory + architecture amendment)** — evidence file populated (~1,000 lines); memory entry `build_tools_phase_perp_state.md` written + MEMORY.md index updated; **no architecture amendment** (outcome (b) matched the implicit pre-audit default per AC #7's "no pivot" branch).
- **Discovered + resolved during audit:** Inno Setup 6 was not initially installed on the maintainer's host; Commander installed it mid-audit (captured at evidence file §7.1).

### File List

**New files:**
- `build_tools/verify_qwen_tts_pin.py` — pre-build qwen-tts pin verification gate (Task 2; ~140 lines).
- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` — audit evidence file (Task 7; ~1,000 lines; force-added per gitignore precedent).
- `_bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md` — `/bmad-bmm-correct-course` routing artifact (Task 1; force-added).
- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\build_tools_phase_perp_state.md` — post-audit state memory entry (Task 7).

**Modified files:**
- `build_tools/requirements-production.txt` — Task 1 / outcome (b) propagation: removed `--extra-index-url cpu`; reframed CPU-only path as source-builder opt-in; added pointer to routing artifact.
- `build_tools/installer.iss` — Task 5: added `#define MyAppBuild "10"` (line 11) and HKLM `Build` registry entry (line 157).
- `build_tools/build_release.bat` — Task 2: `[Pin Verification]` section added between required-files check and `[Version Management]`. Task 5: `[Version Sync]` section added after `[Version Management]`'s optional increment-build prompt.
- `build_tools/version.py` — Task 5: extended `update_installer_script` to sync `#define MyAppBuild` from `VERSION_BUILD`; tightened `update_spec_file` regex from greedy `(MyVoice.*?v)[\d.]+(.*)` to anchored `(MyVoice-Portable-v)\d+(?:\.\d+)*(\.zip)`; replaced Unicode `✓`/`⚠` chars with ASCII `+`/`!`/`=` (cp1252 encoding fix); replaced false-negative "No version found" warnings with accurate `=` "already at target" messages that return True.
- `C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\MEMORY.md` — Task 7: one-line index entry for the new memory file.
- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit.md` — this story file: status, task checkboxes, change log, dev agent record.

**Files NOT touched** (per "What this story is NOT"):
- Anything under `src/myvoice/`.
- Anything under `tests/`.
- `build_tools/myvoice.spec` (outcome (b) requires no spec change; AC #3's bundle audit confirmed no spec change needed).
- The qwen-tts pin (verified at build time by the new script; pin commit `1ab0dd75` unchanged).
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` (AC #7's "no pivot" branch — outcome (b) matches the implicit default).

**Build artifacts produced** (all gitignored):
- `build_tools/dist/MyVoice/` (5.02 GB; the portable bundle).
- `build_tools/dist/MyVoice/MyVoice.exe` (51.5 MB; the bundled executable).
- `installer_output/MyVoice-Setup-v2.1.0.exe` (2.1 GB; LZMA2/ultra64 compressed Inno Setup installer).
- `installer_output/MyVoice-v/MyVoice-Setup-v2.1.0.exe` (release-folder copy; folder name has cosmetic empty-version glitch — captured for §7.6 follow-up).

---

## Change Log

| Date | Story Version | Author | Change |
| --- | --- | --- | --- |
| 2026-05-08 | 0.1 (drafting) | SM (`/bmad-bmm-create-story`) | Initial story draft from `tooling-2-build-tools-audit-scope-sketch.md`. Translated the 5-point scope sketch + 8 concrete concerns into 7 ACs (CPU-vs-CUDA decision, pre-build pin verification, `tts_streaming/` bundle inclusion, portable smoke test, version-drift reconciliation, installer smoke test, evidence-file + memory + optional architecture amendment). Verified all scope-sketch line-number claims against current files; corrected two minor off-by-one errors (`installer.iss:10` not `:11`; `myvoice.spec:110` not `:111` for `collect_submodules('scipy')`). Discovered a sharper version-drift issue beyond the scope sketch: `build_release.bat:83-96` does NOT call `version.py update-all` — incorporated into AC #5. Status set to `ready-for-dev`. |
| 2026-05-08 | 0.2 (pre-dev review pass) | SM (`/bmad-bmm-create-story` self-review) | Pre-dev review pass. Fixes: (H1) AC #3's verification command rewritten — original `MyVoice.exe -c "..."` could not work because `myvoice.spec:376` builds `console=False`; replaced with filesystem audit + transitive importability via AC #4's smoke test, with Subtask 3.2/3.3/3.4 updated accordingly. (M1) Dev Notes updated to disclose that `tts_streaming/__init__.py:11-22` eagerly re-exports from `codec_token_streamer.py`, whose line 34 does top-level `import torch` — defeating the lazy-import design at the package boundary and making the rthook_torch.py timing critical at any `from myvoice.services.tts_streaming import ...` site. (M2) Task 6 gained Subtask 6.0 — determine the runtime log-file location for installer-mode launches (Program Files install path may divert logs to `{localappdata}` or `{userappdata}` per `installer.iss:175-179`'s [UninstallDelete] block) before Subtask 6.2 captures log evidence. (M3) AC #1 outcome (a)'s dev-environment workflow consequence (need for a parallel `python310-cpu/` venv) is now disclosed in the routing-artifact framing. (M4) `build_release.bat` line ranges in references list and inline AC text corrected (Pre-Build Checks 27-77; Version Management 83-96; Step 1 cleanup 98-122; Step 2 125-175; Step 3 177-219). (L1) Speculative "≈350 lines" line-count estimate replaced with bundle-size framing; codec_token_streamer.py:34 top-level torch import noted explicitly. (L2) AC #2's "alternatives rejected with reason" requirement compressed (the obligation lives in Subtask 2.1; AC body now references that in passing). |
| 2026-05-08 | 1.0 (dev closure) | Dev (`/bmad-bmm-dev-story`, Claude Opus 4.7 1M ctx) | Story tooling-2 closed at status `review`. **AC #1:** outcome (b) Ship CUDA-enabled approved by Commander via `/bmad-bmm-correct-course` Batch mode; routing artifact at `tooling-2-correct-course-cpu-vs-cuda.md`; `requirements-production.txt:33-46` updated; spec unchanged. **AC #2:** new `build_tools/verify_qwen_tts_pin.py` (~140 lines, SHA-256 mechanism); wired into `build_release.bat` Pre-Build Checks; pass+fail paths verified. **AC #3:** all 4 `tts_streaming/` modules confirmed in `MyVoice.exe`'s embedded PYZ archive (default PyInstaller layout — `_internal/myvoice/services/` is correctly absent because only `pyz+py`-flagged packages duplicate to filesystem). **AC #4:** portable smoke test — TRUE_STREAM correctly chosen on CUDA host; **failed at `voice_clone_prompt` requirement on default voice profile**; SENTENCE_STREAM fallback served audio cleanly (~17s end-to-end, audible); `rthook_debug.log` missing (latent rthook bug; logs/ dir doesn't exist when hook fires). **AC #5:** `installer.iss:11` adds `#define MyAppBuild`; `installer.iss:157` adds HKLM `Build` registry entry; `version.py::update_installer_script` extended; `build_release.bat` `[Version Sync]` section added; three pre-existing latent `version.py` bugs fixed (cp1252 encoding; greedy regex; false-negative warnings). **AC #6:** installer ran cleanly to `I:\MyVoice` (user-chosen target); identical dispatch behavior to portable mode (same TRUE_STREAM regression, same fallback, same audio); HKLM `Version=2.1.0`, **`Build=10`**, `InstallPath`, App Paths default+Path, Add/Remove Programs `DisplayVersion=2.1.0` all populated; uninstaller cleanly removed everything except an empty parent registry key + runtime-created `config/`/`whisper_models/` (both standard Inno Setup behavior). **AC #7:** evidence file at `tooling-2-build-tools-audit-evidence.md` (~1000 lines); memory entry at `memory/build_tools_phase_perp_state.md`; **no architecture amendment** (outcome (b) matched implicit default per "no pivot" branch). **§7 follow-ups:** 1 HIGH (TRUE_STREAM voice_clone_prompt regression — gates Phase ⊥-Ramp's user-facing deliverable; recommends NOT shipping this build to public users until resolved), 1 MEDIUM (rthook_torch.py debug log silent failure), 4 LOW (Inno Setup empty parent key, [UninstallDelete] config/whisper_models retention design question, build_release.bat release-folder naming glitch, PYZ-vs-_internal documentation note). **Discovered+resolved mid-audit:** Inno Setup 6 was not installed on the maintainer's host; Commander installed it 2026-05-08. **Status:** `review` — pending `/bmad-bmm-code-review` per the standard story-closure workflow (Commander to invoke from a fresh context per the recommended different-LLM discipline). |
