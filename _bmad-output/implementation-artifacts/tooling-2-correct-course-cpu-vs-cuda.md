# Correct-Course Routing Artifact — Story tooling-2 CPU-vs-CUDA Torch Decision

> **Status:** Approved 2026-05-08 by Commander (sole stakeholder per `memory/production_release_state.md`).
> **Trigger:** Story `tooling-2-build-tools-audit` AC #1 — the production-bundle torch variant decision (CPU-only vs. CUDA-enabled vs. split installers) surfaced by the Phase ⊥-Build audit.
> **Routing surface:** `/bmad-bmm-correct-course` workflow invoked **literally** from inside `/bmad-bmm-dev-story` per Story tooling-2 Subtask 1.3. The Story 17.1 routing artifact (`17-1-correct-course-streaming-default-ramp.md`) is the structural precedent; the Epic 16 retrospective's §"What Could Have Gone Better" #4 named Story 16.9's `AskUserQuestion` substitution as the non-precedent. This routing honors the literal-invocation discipline.

## 1. Why this routing exists

The Phase ⊥-Build audit (Story tooling-2) discovered three sources of truth disagreeing about the production bundle's torch variant:

- **`build_tools/requirements-production.txt:33-46`** declares CPU-only intent (`--extra-index-url https://download.pytorch.org/whl/cpu`) and cites a ~2.3 GB size saving.
- **`build_tools/myvoice.spec:74-80`** globs `python310/Lib/site-packages/torch/lib/*.dll` — i.e., whatever the dev `python310/` contains, with no fail-fast assertion against the requirements declaration.
- **The legacy bundle on disk** (`build_tools/dist/MyVoice2.0.1.9Portable/_internal/torch/lib/`) contains the **full CUDA suite** (37 DLLs including `c10_cuda.dll`, `torch_cuda.dll`, `cublas64_12.dll`, `cudart64_12.dll`, `cudnn*64_9.dll`, `cufft64_11.dll`, `cusolver*.dll`, `cusparse64_12.dll`, `nvJitLink_120_0.dll`, `nvrtc*.dll`, etc.). The maintainer's current `python310/Lib/site-packages/torch/lib/` matches it. **Every prior production build has shipped a CUDA bundle**, despite the requirements declaration.

Per Story tooling-2 Subtask 1.3, the trade-off framing was captured at `tooling-2-build-tools-audit-evidence.md` §1.2 and the verdict was routed through `/bmad-bmm-correct-course` for stakeholder sign-off before any propagation to `requirements-production.txt`, `myvoice.spec`, or the architecture document. Story 16.9's deviation (substituting `AskUserQuestion` for the literal workflow) was named explicitly as the non-precedent in the Epic 16 retrospective and Story 17.1 retrospective lesson #4; this routing honors the literal-invocation rule.

## 2. Empirical evidence presented to stakeholder

### 2.1 Trade-off table (verbatim from `tooling-2-build-tools-audit-evidence.md` §1.2)

| Dimension | (a) Ship CPU-only | (b) Ship CUDA-enabled | (c) Split CPU + CUDA installers |
|---|---|---|---|
| Installer size (compressed) | ~280 MB | ~2.5+ GB | Two artifacts at (a) and (b) sizes |
| Story 17.1 TRUE_STREAM at install time | Reachable only if user manually swaps in CUDA torch wheel — non-technical myvoicetts.com audience will not do this | Works out of the box on GPU hosts; SENTENCE_STREAM fallback on CPU hosts via existing dispatch chain | Works out of the box per variant downloaded |
| Release-management overhead | Single artifact; matches `requirements-production.txt`'s declared intent | Single artifact; matches the legacy bundle's actual content; contradicts `requirements-production.txt`'s declared intent unless updated | Two artifacts; doubled checksum/upload/landing-page surface |
| Dev-environment workflow consequence | Maintainer must maintain separate CPU venv (e.g., `python310-cpu/`) and update build invocations to source torch DLLs from it; spec adds fail-fast assertion that halts CUDA-equipped dev env from building | Existing dev `python310/` remains the build source unchanged; `requirements-production.txt:37-38` updated to remove `--extra-index-url cpu` and corrected size-comparison comment | Both venvs required + spec parameterization on `MYVOICE_BUILD_VARIANT` |
| Bandwidth / download-page UX | Friendly to users on metered/slow connections | Heavy download for everyone, including CPU-only users who can't use TRUE_STREAM and download ~2.3 GB of unused CUDA DLLs | Best UX per host at the cost of a bigger landing page |
| NFR12 (CPU-only protection) | Fully satisfied — no CUDA DLLs anywhere | Satisfied at runtime by the probe — `streaming_mode.py:54-56` correctly routes CPU hosts to SENTENCE_STREAM | Satisfied per variant |
| Architecture-document amendment | Required (pivot from implicit default) | Not required (no pivot — formalizes the implicit default per `tooling-1`'s closure pattern) | Required (largest amendment — new build-variants section) |
| Failure modes if mis-shipped | A CUDA bundle slips through unchecked → installer balloons unnoticed; mitigated by the spec's CUDA-DLL-name fail-fast assertion | A CPU-only bundle slips through → certified TRUE_STREAM silently disabled for GPU users; **no build-time mitigation today** | Wrong-variant download by user → must rely on download-page labeling |

### 2.2 Empirical artifacts referenced

- Dev `python310/Lib/site-packages/torch/lib/` — 37 DLLs, full CUDA suite present (audited 2026-05-08; reproduced in evidence file §1.1).
- Legacy bundle `build_tools/dist/MyVoice2.0.1.9Portable/_internal/torch/lib/` — 36 DLLs, identical CUDA suite (proves the spec previously produced a CUDA bundle from the dev `python310/`).
- Memory `hardware_setup.md` — RTX 5090 Blackwell, `torch 2.10+cu128` confirms the dev env is CUDA-equipped.
- `streaming_mode.py:37-56` — `default_streaming_mode_for_hardware()` returns `TRUE_STREAM` when `torch.cuda.is_available()` is True, else `SENTENCE_STREAM` (NFR12 protection).
- Story 17.1 routing artifact (`17-1-correct-course-streaming-default-ramp.md`) — certified TRUE_STREAM as the streaming default on GPU hosts; this routing decides whether that certification reaches users at install time.

## 3. Decision presented and approved

**Outcome chosen: (b) Ship CUDA-enabled.**

The CUDA-enabled variant is approved as the production bundle's torch variant. Justification:

1. **Story 17.1's TRUE_STREAM certification reaches users at install time.** The user-facing purpose of the Phase ⊥-Build audit is to put the certified default into users' hands; outcome (a) silently disables that certification for non-technical users, which is the opposite of the intent.

2. **Matches the legacy bundle's actual content** (formalizes what was always shipped, rather than introducing a new CPU-only artifact that contradicts ~24 months of prior build outputs).

3. **No spec change required** beyond the `requirements-production.txt` reconciliation — the existing torch-DLL glob in `myvoice.spec:74-80` already produces a CUDA bundle from the dev `python310/`. This is the lowest-friction path forward and preserves the existing dev-environment workflow.

4. **NFR12 (CPU-only protection) is satisfied at runtime.** A CPU-only host running the CUDA-enabled bundle still gets routed to SENTENCE_STREAM via `streaming_mode.py:54-56`'s probe (which returns False because CUDA hardware is absent, not because torch is CPU-only). The dispatch chain TRUE_STREAM → SENTENCE_STREAM → BATCH (Story 16.6 D-9) handles the fallback gracefully.

5. **`production_release_state.md`'s installer-size pain point is acknowledged but accepted.** The ~2.5 GB installer is heavy for users on metered/slow connections, but the alternative (outcome (a)) silently disables certified TRUE_STREAM for the GPU users this product was built for. The Phase ⊥ work was specifically designed to let GPU users unlock TRUE_STREAM as a default; shipping CPU-only would invalidate that work at the install step. **If installer size becomes blocking later, outcome (c) (split installers) is the correct future pivot — not outcome (a).**

**Why not outcome (a) "ship CPU-only":** disables certified TRUE_STREAM for GPU users at install time; non-technical myvoicetts.com audience will not manually swap CUDA wheels; outcome contradicts the Phase ⊥ work's user-facing purpose. Listed as the contraindicated outcome.

**Why not outcome (c) "split installers":** doubles release-management overhead (two checksums, two uploads, two landing-page entries) for a use case that doesn't yet have a clear demand signal. The CPU-only audience is hypothetical; the GPU audience is documented (the Phase ⊥ certification was designed for them). If installer-size complaints accumulate, outcome (c) becomes the natural pivot — but that's a follow-up story, not an audit-time scope expansion.

## 4. Architectural action — formalize the implicit default

**Edit `build_tools/requirements-production.txt:33-46`** to reflect outcome (b):

- Remove the `--extra-index-url https://download.pytorch.org/whl/cpu` line (line 38).
- Drop the "CPU ONLY" / "save ~1GB of CUDA libraries" framing in the section header comment (line 34) and the "Alternative explicit CPU installation" comment block (lines 40-41).
- Update the "Size comparison" comment block (lines 43-46) to state the chosen variant: "Production bundle ships CUDA-enabled torch; ~2.5 GB. CPU-only variant available via `pip install torch --index-url https://download.pytorch.org/whl/cpu` for source-builders who prefer the smaller footprint." Keep the rationale visible for future readers but stop framing CPU as the default.
- Add a one-line pointer to this routing artifact: `# Production torch variant decided 2026-05-08 — see _bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md`.

**`build_tools/myvoice.spec`: NO CHANGE.** The existing torch-DLL glob at lines 74-80 already produces a CUDA bundle from the dev `python310/`. Outcome (b) formalizes this behavior rather than altering it.

**`_bmad-output/planning-artifacts/architecture-optimization-pass.md`: NO AMENDMENT REQUIRED.** Per Story tooling-2 AC #7's "no pivot" branch: outcome (b) matches the implicit pre-audit default (every prior production build has shipped CUDA), so the routing-artifact + evidence-file pair is the durable record. This mirrors `tooling-1-git-repo-integration.md`'s closure pattern (no architecture amendment).

**`build_tools/installer.iss`: NO CHANGE.** The Inno Setup script's `[Files]` recursesubdirs at lines 108-112 will pick up whatever PyInstaller produced; it is variant-agnostic.

**Compression budget acknowledgment.** Per the evidence-file Dev Notes "Architecture context" §3, LZMA2 compression on a ~2.5 GB CUDA bundle takes considerably longer than the CPU-only baseline. The next `build_release.bat` end-to-end run is expected to take **multiple hours** rather than the documented "5-15 minutes" per `build_release.bat:131`. This is captured as a follow-up scope item in the evidence file's §7 (no immediate action — the time investment is acceptable for a single-artifact monthly-or-less release cadence per `production_release_state.md`).

## 5. Implications acknowledged

- **Installer balloons from ~280 MB to ~2.5+ GB.** Acknowledged. The Phase ⊥ certification's user-reach value outweighs the bandwidth cost for the documented GPU-audience use case. If outcome (c) (split installers) becomes warranted later (e.g., CPU-user complaints from myvoicetts.com analytics), it's a separate follow-up story, not a pin bump on this routing.
- **No code change to `src/myvoice/`.** The CPU-host fallback to SENTENCE_STREAM is preserved by `streaming_mode.py:54-56`'s existing probe logic. NFR12 (CPU-only protection) remains satisfied at runtime.
- **No qwen-tts pin bump.** The pin remains at commit `1ab0dd75` per Story 16.1. AC #2 verifies the pin; this routing does not change it.
- **Build-time wall-clock budget grows.** Future `build_release.bat` runs will spend more time in the LZMA2 compression pass than the current `build_release.bat:131` "5-15 minutes" estimate suggests. This is documented in the evidence file's §7 follow-ups but not addressed by this story (the doc string itself is a `build_release.bat` comment, easy to update at the next opportunity; the compression algorithm is correct).
- **The CPU-only audience is a follow-up concern, not a missed deadline.** Source-builders and CPU-host users can install via `pip install torch --index-url https://download.pytorch.org/whl/cpu` per the updated `requirements-production.txt` comment block. If install-from-source becomes a friction point, outcome (c) is the documented pivot.
- **Outcome (b) is reachable as "no pivot" per the implicit pre-audit default.** Per Story tooling-2 AC #7's branching logic: outcome (b) matches the legacy bundle's actual content; therefore no architecture amendment is required and the evidence file + this routing artifact are the durable closure artifacts (mirroring `tooling-1`'s pattern). If a future audit pivots to (a) or (c), an architecture amendment will be required at that time.
- **No spec fail-fast assertion is added.** Outcome (a)'s mitigation pattern (a CUDA-DLL-name check that halts CUDA-equipped dev environments from building) does not apply to outcome (b) — which deliberately wants the CUDA bundle. If outcome (c) is chosen later, the spec's `MYVOICE_BUILD_VARIANT` parameterization will replace the implicit default.
- **Evidence file §1.4 propagation tasks (Subtasks 1.4–1.6) are scoped by this routing.** Specifically: edit `requirements-production.txt`, no spec edit, force-add the routing artifact, commit naming the chosen outcome.

## 6. Stakeholder sign-off

- **Stakeholder:** Commander (`wreckedmech@gmail.com`; sole stakeholder per `memory/production_release_state.md`).
- **Decision date:** 2026-05-08.
- **Decision channel:** `/bmad-bmm-correct-course` workflow invoked literally from inside `/bmad-bmm-dev-story` per Story tooling-2 Subtask 1.3. **Batch mode** (matches Story 17.1 precedent for single-decision routings). Honors the Epic 16 retrospective §"What Could Have Gone Better" #4 lesson and the Story 17.1 retrospective's lesson #4 — when an AC names a specific workflow, use that workflow rather than substituting `AskUserQuestion`.
- **Approved option:** **(b) Ship CUDA-enabled.** Single-artifact production bundle; CUDA torch wheel from dev `python310/`; `requirements-production.txt` updated to formalize the variant; no spec change; no architecture amendment (no pivot).
- **Conditions:** none. The decision was approved without modification.
- **Future pivot trigger documented:** if installer-size complaints accumulate from myvoicetts.com user feedback OR a metered-connection user research signal materializes, outcome (c) (split CPU + CUDA installers) is the natural follow-up — to be scoped as a separate `tooling-N` story at that time.

## 7. Cross-references

- Audit evidence file (the durable artifact): `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` (§1 captures the full audit; §1.3 will be updated with this routing artifact's pointer).
- Audit story file (parent): `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit.md` (Subtask 1.3 invoked this routing; Subtasks 1.4-1.6 propagate the outcome).
- Audit scope sketch (foundational input): `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-scope-sketch.md` §1 (CPU-vs-CUDA mismatch concern, authored 2026-05-08).
- Routing-artifact precedent (structural mirror): `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (Story 17.1 streaming-default-ramp routing; literal-invocation discipline source).
- Routing-artifact procedural precedent (NOT the channel precedent — Story 16.9 substituted `AskUserQuestion` and the Epic 16 retro named this as the non-precedent): `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md`.
- Story 17.1 closure (the certification this routing protects at install time): `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (Phase ⊥-Ramp closure; certified TRUE_STREAM as the GPU default).
- Memory entries:
  - `memory/torch_pyqt6_dll_ordering.md` — DLL-init invariant on Windows; load-bearing for AC #4's smoke test of the CUDA bundle.
  - `memory/hardware_setup.md` — RTX 5090 + cu128 dev host context; ship-target also covers RTX 30xx/40xx (the CUDA-bundle audience).
  - `memory/production_release_state.md` — installer-size pain point; ships via myvoicetts.com to non-technical users; documents the audience this routing serves.
  - `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical; NOT updated by this routing).
- Phase ⊥ source files (read-only reference):
  - `src/myvoice/services/tts_streaming/streaming_mode.py:37-87` — hardware probe; the runtime mechanism that routes outcome (b)'s CUDA-bundle GPU users to TRUE_STREAM and CPU users to SENTENCE_STREAM.
- Build-pipeline files affected by this routing:
  - `build_tools/requirements-production.txt` — edited per §4 above (CUDA-enabled formalization).
  - `build_tools/myvoice.spec` — unchanged.
  - `build_tools/installer.iss` — unchanged.
  - `_bmad-output/planning-artifacts/architecture-optimization-pass.md` — unchanged (no pivot from implicit default).
- Force-add commands (per gitignore precedent for `_bmad-output/`):
  - `git add -f _bmad-output/implementation-artifacts/tooling-2-correct-course-cpu-vs-cuda.md`
  - `git add -f _bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` (when §1.3 is updated to point to this routing)

---

**Workflow execution log:** `/bmad-bmm-correct-course` invoked 2026-05-08 from inside `/bmad-bmm-dev-story` for Story tooling-2 Subtask 1.3. Batch mode. Trade-off table read from evidence file §1.2. Single-question routing (no impact-analysis checklist iteration needed — the change is well-scoped). Outcome (b) approved without modification. Routing-artifact write completed; force-add deferred to Subtask 1.6 commit alongside `requirements-production.txt` edit.
