# Scope Sketch — Story 17.2: Lazy + Persistent voice_clone_prompt Precompute for CLONED Voices on TRUE_STREAM

> **Status:** scope sketch (input to `/bmad-bmm-create-story`); not yet a story file.
> **Authored:** 2026-05-08 by `/bmad-bmm-dev-story` follow-up turn after Story tooling-2 closure surfaced the runtime regression at evidence file §7.2.
> **Purpose:** Capture the regression's mechanics + the lazy-and-persistent fix shape Commander has chosen, so `/bmad-bmm-create-story` can convert this into a real story (`17-2-cloned-voice-truestream-prompt-precompute.md`) with full ACs / Tasks / Dev Notes. Mirrors the role `tooling-2-build-tools-audit-scope-sketch.md` played for Story tooling-2.
> **Phase tag:** Phase ⊥-Ramp completion (closes the gap between Story 17.1's "TRUE_STREAM is certified" and the user-facing "CLONED voice users actually receive TRUE_STREAM at install time"). Re-opens Epic 17 (currently `done`) with this second story; Epic 17 transitions to `done` again on closure of 17.2.

## Why this story exists

Story tooling-2's portable + installer smoke tests (2026-05-08) discovered that the bundled MyVoice.exe correctly probes CUDA → TRUE_STREAM as the default streaming mode, attempts the dispatch, and **fails on every UI-initiated generation** with:

```
QwenTTSService - ERROR - [QwenTTS] TRUE_STREAM talker error:
  TRUE_STREAM voice-clone path requires request.voice_clone_prompt
ValueError: TRUE_STREAM voice-clone path requires request.voice_clone_prompt
```

The graceful-degradation chain (Story 16.6 D-9 / NFR7) catches the failure and falls through to SENTENCE_STREAM, which serves the audio successfully. Users get audio. But they get it via SENTENCE_STREAM — **not the certified-by-Story-17.1 TRUE_STREAM path**. The Phase ⊥ work that Stories 16.x + 17.1 closed is functionally invisible to users on the default `Base (Clone)` voice profile (and any other CLONED voice).

**Root cause** (analyzed via `qwen_tts_service.py:2793-2798`):

```python
elif request.model_type == QwenModelType.BASE:
    if request.voice_clone_prompt is None:
        raise ValueError(
            "TRUE_STREAM voice-clone path requires "
            "request.voice_clone_prompt"
        )
    model.generate_voice_clone(
        text=request.text,
        language=request.language or "Auto",
        voice_clone_prompt=request.voice_clone_prompt,
        non_streaming_mode=False,
    )
```

TRUE_STREAM treats every BASE-model voice as if it had a pre-computed `voice_clone_prompt` embedding tensor (the EMBEDDING-voice pattern). But CLONED voices like `Sarira-F` use the BASE model with `ref_audio + ref_text` — they have a `.wav` file and (sometimes) a `.txt` transcript next to it; they do NOT ship with pre-computed embedding tensors. The non-TRUE_STREAM paths (SENTENCE_STREAM, BATCH) call `model.generate_voice_clone(text, ref_audio, ref_text, ...)` which lets the qwen-tts library compute the embedding internally on each call. TRUE_STREAM cannot afford that — the streaming dispatch needs to lock-step with `CodecTokenStreamer`'s bounded queue and `StreamingDecoderWorker`'s overlap-add (Story 16.3 / 16.4), so the embedding must be computed BEFORE the talker thread enters streaming mode.

**The dead infrastructure that pre-figured this fix.** `qwen_tts_service.py:631` declares `self._voice_clone_prompts: Dict[str, Any] = {}` — but this dict is **never written to** anywhere in the file. It was set up for exactly this use case (cache of voice_clone_prompts keyed by something — voice profile name? ref_audio path?) but the wiring was never completed. Story 17.2 fills in the missing wiring.

**Why "lazy + persistent" is the right design** (per Commander 2026-05-08):

- **Lazy** — first TRUE_STREAM request for a given CLONED voice triggers the precompute (Whisper transcription if missing + `create_voice_clone_prompt` to produce the embedding tensor); subsequent requests use the cached embedding. Voices the user never selects don't pay the cost. First-utterance latency for a fresh voice is ~1–3s additional (Whisper + embedding compute); acceptable trade-off vs. eager precompute at app startup which would slow startup by N × 1–3s for N voices.
- **Persistent** — the computed embedding (and the Whisper transcription) get saved next to the voice's `.wav` file (sidecar `.pt` and `.txt`, mirroring the existing `.txt` transcript auto-detect at `voice_profile.py:348-355`). One-time cost forever. App restarts don't recompute. The `.pt` file is small (~few MB depending on tier); the `.txt` is bytes.

## Pre-existing infrastructure already verified before drafting

- **`generate_voice_clone` flow** (`qwen_tts_service.py:1082-1121`) — the existing entry point for CLONED-voice generation; takes `(text, ref_audio, ref_text, ...)`, constructs a `QwenTTSRequest` with `model_type=BASE`, routes to `_dispatch_by_streaming_mode`. Does NOT currently set `voice_clone_prompt` on the request — that's the gap.

- **`create_voice_clone_prompt`** (`qwen_tts_service.py:1179-1228`) — the existing method that takes `ref_audio + ref_text` and returns a `voice_clone_prompt` (a tensor with `.ref_code` + `.ref_spk_embedding`). Already callable; already async-wrapped. Story 17.2 just needs to invoke it at the right moment + cache the result.

- **`create_voice_clone_prompt_for_tier`** (`qwen_tts_service.py:1230-1284`) — multi-tier variant for EMBEDDING voices that produces tier-specific embeddings (1.7B vs 0.6B). Story 17.2 likely needs this for CLONED voices too if the cached embedding is tier-locked (different model tiers have different embedding spaces). Open design question — see scope point (c).

- **`_voice_clone_prompts: Dict[str, Any] = {}`** (`qwen_tts_service.py:631`) — dead cache infrastructure already declared. Story 17.2 wires it up.

- **`voice_design_studio_dialog.py:1143-1162` and `:1526-1542`** — the existing precedent for `torch.save(voice_clone_prompt, str(embedding_path))`. Voice Design Studio already knows how to persist these tensors. Story 17.2 reuses the same mechanism but for CLONED voices instead of EMBEDDING voices.

- **`VoiceProfile.transcription` field + `.txt` auto-detect** (`voice_profile.py:219, 348-355`) — voices already support an optional `.txt` sidecar containing the transcription, auto-detected on profile load. Story 17.2's persistent transcription naturally extends this (uses the same `<voice>.txt` location).

- **`whisper_service.py` + `whisper_subprocess.py`** — Whisper integration already exists in the codebase; used for transcribing arbitrary audio. Story 17.2 invokes the existing service rather than introducing a new transcription dependency.

- **`TranscriptionStatus` enum** (`voice_profile.py:22-29`) — already has `NOT_STARTED`, `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`, `SKIPPED`. Story 17.2 uses these states to track the lazy-precompute progress for diagnostics + UI feedback.

- **The Phase ⊥ dispatch chain** (`qwen_tts_service.py::_dispatch_by_streaming_mode`, lines 3320-3399) — the three-mode fork that currently catches the voice_clone_prompt ValueError and falls through to SENTENCE_STREAM. Story 17.2 changes the failure mode from "always fall through" to "succeed on TRUE_STREAM after precompute completes". The fallback chain stays as the safety net for genuine TRUE_STREAM failures (CUDA OOM, unexpected library errors); only the voice_clone_prompt-missing case stops triggering it.

- **The certification context.** Story 17.1's audition (per `17-1-correct-course-streaming-default-ramp.md`) used a specific test fixture's request shape. The audition's verdict — *"PASS if and only if zero listeners flagged audible_seam for any TRUE_STREAM pair"* — was met under that fixture. Story 17.2 ensures the production UI's request shape achieves the same dispatch path. **Story 17.2 does NOT re-litigate Story 17.1's certification** — it just makes the certified path actually reach users.

## Concrete concerns surfaced by the audit

1. **`voice_clone_prompt` is a tensor, not a transcript.** The error message reads "voice-clone path requires `request.voice_clone_prompt`" but the conventional reading ("the user's text-to-clone prompt") is misleading. The actual expected value is the `Qwen3VoiceClonePrompt` dataclass (per `qwen_tts_service.py:180-184`'s wrapper) with `.ref_code` (a tensor) and `.ref_spk_embedding` (a tensor). The transcription is an **input** to producing this tensor (via `create_voice_clone_prompt(ref_audio, ref_text)`), not the value itself.

2. **The `_voice_clone_prompts: Dict[str, Any] = {}` cache is initialized but unused.** Grep confirms zero writes. This is dead infrastructure pre-figuring this exact use case; Story 17.2 wires it up.

3. **Cache key choice is undefined.** Most natural keys: voice profile name (e.g., `"Sarira-F"`) OR ref_audio path string OR a content hash of the .wav file. Voice profile name is simplest if profile names are unique within the app's voice library; ref_audio path string is unambiguous but less stable across reinstalls (paths differ portable vs. installed); content hash is fully content-addressed but pays a hash-cost on every key lookup. **Recommend: voice profile name**, with the persisted `.pt` file at `<voice_dir>/<voice_name>.pt`.

4. **Multi-tier consideration.** EMBEDDING voices have separate embeddings per tier (1.7B vs 0.6B per `voice_profile.py:130 VALID_TIERS`). Whether CLONED voices need the same — i.e., whether the embedding tensor is tier-specific — depends on whether the Base model's `create_voice_clone_prompt` produces tier-locked outputs. **Open question; needs source-tree investigation.** If yes: persist `<voice_name>.<tier>.pt`. If no: a single `<voice_name>.pt` works for both tiers.

5. **What happens if Whisper transcription fails.** Whisper might fail (corrupt .wav, unsupported codec, OOM, etc.). The lazy precompute needs an explicit error path: log + set `TranscriptionStatus.FAILED` + fall through to SENTENCE_STREAM via the dispatch chain (this preserves NFR7 — user still gets audio). On next attempt, retry Whisper (allows transient OOM / driver issues to recover) OR cache the FAILED state and skip retries until user manually clears (avoids retry storms on persistently-broken files). **Recommend: retry-with-backoff** (e.g., 3 attempts; cache FAILED on the third), with a UI "regenerate transcription" affordance for user-initiated retries.

6. **Concurrency.** The `_dispatch_by_streaming_mode` call is `async`. The lazy precompute will run inside it; if two concurrent generations request the same voice (e.g., user mashes the Generate button), both will trigger the precompute. **Recommend: per-voice asyncio.Lock keyed by voice_name** so the second request waits for the first's precompute to complete, then both use the cached result.

7. **Cache invalidation.** When does the `.pt` get re-computed? If the user replaces the `.wav` file (same name, different content), the cached `.pt` is stale. **Recommend: store ref_audio's mtime/size/hash inside the `.pt`** (or as a small adjacent `.pt.meta.json`); on load, compare against the current `.wav`'s state; recompute if mismatch. Cheaper than rehashing on every load if mtime+size suffices for change detection.

8. **Bundled-voices already-have-transcripts case.** Looking at `dist/MyVoice/_internal/voice_files/` (or wherever bundled voices ship), some voices may already have a `.txt` sidecar (per the auto-detect at `voice_profile.py:348-355`). For those, Whisper invocation is unnecessary — go straight to embedding compute. The lazy precompute should check for an existing transcript before invoking Whisper.

9. **First-run UX.** A 1–3s delay on first TRUE_STREAM use of a fresh voice is acceptable (Commander's framing) but should be **visible** to the user — a "Preparing voice for streaming..." indicator on the UI, otherwise the delay looks like a hang. The existing `ServiceStatusIndicator` (Epic 12) might already have an appropriate state, or a new sub-state is added.

10. **What this DOESN'T cover.** Story 17.1's audition did NOT explicitly test the precomputed-vs-on-the-fly embedding path — the audition fixture was constructed differently. Story 17.2 does NOT need to redo the audition (the embedding tensor is the same regardless of precompute timing; perceptual quality is preserved). But: if any users experience a TRUE_STREAM regression that didn't surface in Story 17.1's source-tree audition, Story 17.2's evidence file should capture it (e.g., if precomputed embeddings differ subtly from on-the-fly embeddings due to RNG seeding inside the Base model — unlikely but worth noting).

## Five-point scope sketch (for the SM workflow to expand into ACs)

(a) **Wire `_voice_clone_prompts: Dict[str, Any]` cache into `generate_voice_clone`.** When a CLONED-voice generation is dispatched and TRUE_STREAM is the resolved mode, check the cache before constructing the request. Cache miss → invoke the lazy-precompute flow (steps b + c). Cache hit → set `request.voice_clone_prompt` to the cached value before dispatch. Per-voice asyncio.Lock to prevent duplicate precompute on concurrent requests. (Concerns 2, 3, 6, 7.)

(b) **Lazy transcription via existing whisper_service.** When the cache miss happens AND the voice profile's `transcription` is None (or absent `.txt` sidecar), invoke `whisper_service.transcribe(ref_audio_path)` in an asyncio thread. On success: write the transcription to `<voice_name>.txt` next to the `.wav` (matching the existing auto-detect convention) AND update the in-memory `VoiceProfile.transcription`. On failure: log, mark `TranscriptionStatus.FAILED`, propagate up to (a) which falls through to SENTENCE_STREAM. Retry policy: 3 attempts before persistent FAILED state. (Concerns 5, 8.)

(c) **Persistent embedding via `create_voice_clone_prompt` + `torch.save`.** Once a transcription is available (either pre-existing or freshly Whisper-generated), call the existing `create_voice_clone_prompt(ref_audio, ref_text)` to compute the embedding tensor. Persist via `torch.save(prompt, "<voice_dir>/<voice_name>.pt")` (mirroring `voice_design_studio_dialog.py:1162`). Update the in-memory `_voice_clone_prompts[voice_name] = prompt`. On subsequent app launches, load via `torch.load("<voice_dir>/<voice_name>.pt")` (matching the EMBEDDING-voice load path that already exists at `qwen_tts_service.py:1562+`). Adjacent metadata file `<voice_name>.pt.meta.json` records ref_audio's mtime + size for cache-invalidation detection. (Concerns 1, 2, 4, 7.)

(d) **UI feedback for first-run lazy precompute.** Add a "Preparing voice for streaming..." state to the existing `ServiceStatusIndicator` (Epic 12) that fires when (a)/(b)/(c) are running on a cache miss. Visible for 1–3s on first generation of a fresh voice. Disappears once the cache is populated (subsequent generations are instant). If an existing indicator state is appropriate (e.g., "Loading model"), reuse it; otherwise add a new sub-state. (Concern 9.)

(e) **Smoke-test against the bundled environment.** Re-run the Story tooling-2 §4 portable smoke (and §6 installer smoke) on the produced bundle: launch MyVoice.exe → generate `s-014` on `Base (Clone)` voice → verify `myvoice.log` shows `TRUE_STREAM` succeeds end-to-end (no fallback to SENTENCE_STREAM) on the SECOND attempt (after first attempt populates the cache). Also: clean install + first-ever generation on a fresh `voice_files/` should trigger the Whisper transcription + embedding precompute path. Capture the smoke-test evidence in this story's evidence file. **Effectively: Story tooling-2's HIGH §7.2 follow-up gets resolved here, and the closure smoke test re-runs the build pipeline to confirm.**

## What this story is NOT

- **Not a re-litigation of Story 17.1's audition.** Story 17.1's certification stands; the embedding tensor produced by precompute is the same one the audition's fixture would have used. Perceptual equivalence is preserved.

- **Not a Voice Design Studio change.** EMBEDDING voices already have precomputed embeddings via Voice Design Studio; their flow is unchanged. This story is specifically for CLONED voices that traditionally compute on-the-fly.

- **Not a transcription quality story.** Whisper's transcription quality is whatever it is; this story uses the existing whisper_service without re-tuning. If a voice's transcription is wrong (Whisper misheard), the user can manually correct the `.txt` file and the next embedding compute will use the corrected value.

- **Not a multi-tier optimization.** Multi-tier embedding compute (per scope concern #4) is in scope ONLY if source investigation shows tier-locked embeddings; otherwise a single `.pt` per voice covers all tiers. The story does NOT optimize tier-switching latency further than what `_voice_clone_prompts` provides as a cache.

- **Not a build-pipeline change.** Story tooling-2 closed Phase ⊥-Build; the production bundle ships CUDA torch + verified pin + version sync. Story 17.2 produces source-tree code changes that get picked up by the next `build_release.bat` run; no spec / installer / build_release.bat edits.

- **Not a sprint-status entry for 17.2 specifically until Epic 17 is re-opened.** Epic 17 currently reads `done` in `sprint-status.yaml`; SM workflow needs to add `17-2-cloned-voice-truestream-prompt-precompute: ready-for-dev` AND flip `epic-17` back to `in-progress` until 17.2 closes. Optional: explicitly note in `sprint-status.yaml` that the epic was re-opened post-tooling-2 discovery.

- **Not a re-run of the production release.** After 17.2 closes, the next build pipeline run produces a new installer with the fix; that build's release decision is a separate Commander decision. The "circle back and rebuild" note from Commander 2026-05-08 captures this implicit follow-on; not folded into 17.2's scope.

- **Not a Whisper integration overhaul.** The existing `whisper_service.py` / `whisper_subprocess.py` is reused as-is. If Whisper's API has constraints that affect (b) (e.g., subprocess-based transcription has worse latency than expected), capture as a follow-up scope item — not a rewrite.

## References

**Source tree (read + likely-edit candidates):**

- `src/myvoice/services/qwen_tts_service.py:631` — dead `_voice_clone_prompts` cache (gets wired)
- `src/myvoice/services/qwen_tts_service.py:1082-1121` — `generate_voice_clone` (likely edit: precompute hook)
- `src/myvoice/services/qwen_tts_service.py:1179-1228` — `create_voice_clone_prompt` (read-only; called by precompute)
- `src/myvoice/services/qwen_tts_service.py:2793-2798` — TRUE_STREAM voice_clone_prompt requirement (read-only; understanding the contract)
- `src/myvoice/services/qwen_tts_service.py:3320-3399` — `_dispatch_by_streaming_mode` fallback chain (read-only; preserved as safety net)
- `src/myvoice/models/voice_profile.py:22-29, 219-226, 348-355` — `TranscriptionStatus` + `transcription` field + `.txt` auto-detect (likely edit: extend with embedding-cache state)
- `src/myvoice/services/whisper_service.py` — existing Whisper transcription entry point (read-only; called by precompute)
- `src/myvoice/services/whisper_subprocess.py` — Whisper subprocess wrapper (read-only)
- `src/myvoice/ui/dialogs/voice_design_studio/voice_design_studio_dialog.py:1143-1162, 1526-1542` — existing `torch.save(voice_clone_prompt, ...)` precedent (read-only; structural reference)

**Architecture references:**

- `_bmad-output/planning-artifacts/architecture-optimization-pass.md` — Story 17.1 H4 sub-section (NFR3 audition); NFR7 graceful degradation; D-9 hardware-aware default

**Memory:**

- `memory/build_tools_phase_perp_state.md` — names this story as the HIGH follow-up gating Phase ⊥-Ramp's user-facing deliverable
- `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical)
- `memory/hardware_setup.md` — RTX 5090 CUDA dev host (informs eager-vs-lazy recommendation)

**Precedent stories:**

- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` §7.2 — the source of this story's framing (HIGH severity follow-up captured 2026-05-08)
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` — Story 17.1 closure (the certification this story makes user-reachable)
- `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` — routing artifact (perceptual equivalence framing)
- `_bmad-output/implementation-artifacts/16-7-empirical-validation-gates-for-streaming-default.md` — empirical-validation harness (informs the smoke-test design in scope point (e))
- `_bmad-output/implementation-artifacts/16-3-codectokenstreamer-with-bounded-queue.md` — explains why TRUE_STREAM cannot afford on-the-fly embedding compute (the bounded-queue lock-step constraint)

**Empirical reference (regression evidence):**

- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` §4.3.2 — verbatim portable-smoke log lines showing the regression
- `_bmad-output/implementation-artifacts/tooling-2-build-tools-audit-evidence.md` §6.2 — verbatim installed-smoke log lines (identical regression in installer-mode bundle)

## Suggested story-file naming

`_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md`

## Suggested Phase tag

`Phase ⊥-Ramp completion` — Story 17.1 closed the audition; Story 17.2 closes the user-reach gap. Once 17.2 closes, Epic 17 is genuinely done in the user-facing sense (not just the certification sense).

## Suggested sprint-status edit (for SM workflow when creating the story)

```yaml
# Epic 17 — Streaming Default Ramp (Phase ⊥-Ramp — audition-gated follow-up to Epic 16)
# Re-opened 2026-05-08 per Story tooling-2 closure: discovered TRUE_STREAM
# voice_clone_prompt regression for CLONED voices in bundled UI flow; Story 17.2
# precomputes the embedding lazily on first use and persists to <voice>.pt.
epic-17: in-progress
17-1-streaming-default-ramp: done
17-2-cloned-voice-truestream-prompt-precompute: ready-for-dev
epic-17-retrospective: optional
```
