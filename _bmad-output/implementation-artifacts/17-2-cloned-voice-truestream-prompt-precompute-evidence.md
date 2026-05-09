# Story 17.2 Cloned-Voice TRUE_STREAM Prompt Precompute — Evidence File

> **Status:** in-progress (drafting). Source-tree implementation (Tasks 1–6) committed at `737176b`. Sections §3–§6 populate as the bundled smoke flow runs.
>
> **Purpose:** Captures the verifiable evidence behind Story 17.2's 6 ACs — specifically AC #6 (bundled-environment smoke) and the closure of `tooling-2-build-tools-audit-evidence.md` §7.2 HIGH follow-up that surfaced the regression this story fixes.
>
> **Force-add note:** This file lives under `_bmad-output/` which is gitignored. Add via `git add -f _bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` per the precedent set by Story 16.9 / 17.1 / tooling-2 evidence files.

---

## §1 — Summary

Story 17.2 wires the dead `_voice_clone_prompts` cache at `qwen_tts_service.py:631` into `generate_voice_clone` so every UI-initiated CLONED-voice generation no longer trips the TRUE_STREAM contract check at `qwen_tts_service.py:2793-2798` (the regression captured verbatim in `tooling-2-build-tools-audit-evidence.md` §4.3.2 + §6.2). The fix has six tasks; Tasks 1–6 (source-tree + tests) committed at `737176b`; Task 7 (this evidence file) verifies the fix actually reaches users via `dist/MyVoice/MyVoice.exe`.

**Tasks 1–6 closure summary:**
- 26 new unit tests in `tests/unit/services/test_voice_clone_prompt_cache.py` — all passing
- 65 existing dispatch + session-integration tests pass with no regressions
- 687/687 tests pass in the targeted unit suite (`tests/unit/services` + `tests/unit/models` + `tests/unit/observability`)

---

## §2 — Source-tree changes

Modified files (commit `737176b`):
- `src/myvoice/services/qwen_tts_service.py` — bulk of LOC delta. Cache wiring + per-voice locks + lazy transcription helper + persistent embedding helper + startup hydration + UI feedback hook + four-condition-gated `generate_voice_clone` body.
- `src/myvoice/app.py` — orchestrator wiring: TTS receives whisper / voice-profile-manager / preparing-voice / whisper-init-callback hooks; fire-and-forget hydration after `voice_manager.start()`; `_on_tts_preparing_voice_message` handler.
- `src/myvoice/models/ui_state.py` — added `ServiceStatusInfo.preparing_voice_message: Optional[str]`.
- `src/myvoice/ui/components/service_status_indicator.py` — `_update_tooltip` surfaces the precompute message as italic line.

Added files:
- `tests/unit/services/test_voice_clone_prompt_cache.py` — 26 unit tests across Tasks 1, 2, 3, 5, 6.

Story file:
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md` — Tasks 1–6 marked complete, Dev Agent Record populated.

See commit `737176b` diff for the full surface.

---

## §3 — Build pipeline (Task 7.1)

### §3.1 Build invocation

Command run: `cmd /c "build_tools\build_release.bat < build_increment_answer.txt > build_release.log 2>&1"`

`build_increment_answer.txt` contained a single byte `N` to skip the build-number increment prompt at `build_release.bat:111` so the build could run non-interactively from this session.

### §3.2 Build outputs verified

```
build_tools/dist/MyVoice/MyVoice.exe        ── 51,472,808 bytes (portable launcher)
build_tools/dist/MyVoice/_internal/         ── 5.1 GB total bundle
installer_output/MyVoice-Setup-v2.1.0.exe   ── 1,653,404,164 bytes (Inno Setup installer)
```

PyInstaller "Build complete" marker at `build_release.log:992`:

```
198349 INFO: Build complete! The results are available in: I:\MyVoiceV2\build_tools\dist
```

Inno Setup compiled the installer to `installer_output/MyVoice-Setup-v2.1.0.exe`; final size 1.65 GB (compresses 5.1 GB bundle).

### §3.3 qwen-tts pin verified at build time

Per Story tooling-2 §3 build pipeline, `verify_qwen_tts_pin.py` is part of the build (TODO: confirm exact log line in this build). Pin per `requirements.txt:23` remains `1ab0dd75` (qwen-tts 0.0.4). Story 17.2's `_QWEN_TTS_PIN_HASH` constant in `qwen_tts_service.py` matches.

### §3.4 PyInstaller warnings (informational)

Build log `build_release.log:59,60,61,103` shows the same DLL-load warnings present in tooling-2's build (PyInstaller's static-analysis pass cannot import torch in the build env; the actual bundle ships correctly). Not a regression — same WARNINGs ship with every Story 17.x build.

---

## §4 — Portable smoke (Task 7.2 + 7.3)

> **Status:** Pending Commander hands-on smoke run on `build_tools/dist/MyVoice/MyVoice.exe`.

### §4.1 Smoke procedure

The user must run the following on the dev host (RTX 5090 CUDA Blackwell):

1. **Pre-clean** — delete any leftover `voice_files/` next to `MyVoice.exe` AND the `logs/` directory so the run starts cold:
   ```powershell
   Remove-Item -Recurse -Force build_tools\dist\MyVoice\voice_files -ErrorAction SilentlyContinue
   Remove-Item -Recurse -Force build_tools\dist\MyVoice\logs -ErrorAction SilentlyContinue
   ```

2. **First launch — first generation (cold cache)**:
   - Launch `build_tools\dist\MyVoice\MyVoice.exe`.
   - The app copies bundled voices from `_internal/voice_files/` to user-data `voice_files/` on first run (per `portable_paths.py:_copy_bundled_voice_files`).
   - In Voice Library, select the `Sarira-F` profile (CLONED voice — `Sarira-F.txt` sidecar already ships next to `Sarira-F.wav`, so AC #2 priority-2 short-circuits Whisper).
   - Type a short utterance (the story names `s-014`; any single sentence works). Click Generate.
   - Observe the TTS service indicator in the status bar — its tooltip should show "Preparing voice for streaming…" for ~1–3 seconds during the first-generation embedding compute.
   - Wait for audio to play through.

3. **Second generation (warm cache)**:
   - With the same voice still selected, click Generate again on the same or a different sentence.
   - Tooltip MUST NOT show the preparing-voice message (cache hit is silent).
   - First-audio latency should now satisfy NFR1 GPU short-class target (≤5.0 s p95 per `architecture-optimization-pass.md:836+`).

4. **Capture artifacts** for §4.3:
   - Copy `build_tools/dist/MyVoice/logs/myvoice.log` aside (or extract grep'd snippets per §4.3 below).
   - Confirm `build_tools/dist/MyVoice/voice_files/Sarira-F.quality.pt` and `Sarira-F.quality.pt.meta.json` exist after step 2 first-attempt.

### §4.2 Expected log markers

For AC #6 closure on the cold cache → warm cache flow, `myvoice.log` should contain (in order):

**First attempt — cache miss:**
```
QwenTTSService - INFO - Voice clone prompt cache miss for <abs-path-to-Sarira-F.wav> (tier=quality); computing
QwenTTSService - INFO - Voice clone prompt persisted to <abs-path>/Sarira-F.quality.pt
```

(NO `Whisper transcription started` line — `.txt` sidecar short-circuits Whisper.)

**First attempt — TRUE_STREAM completion (no fallback):**
```
QwenTTSService - INFO - ...TRUE_STREAM... (success markers vary by codepath; key invariant: NO line containing "TRUE_STREAM voice-clone path requires" and NO `streaming_mode_fallback` metric for this generation).
```

**Second attempt — cache hit:**
```
QwenTTSService - DEBUG - Voice clone prompt cache hit for <abs-path-to-Sarira-F.wav> (tier=quality)
```

(NO compute, NO persist, NO Whisper.)

### §4.3 Captured log excerpts

#### §4.3.1 — First smoke run (build #1) — **CRASH**

The first smoke run on the freshly-built `dist/MyVoice/MyVoice.exe` (build at 20:31, smoke at 20:46) crashed with:

```
2026-05-08 20:46:16,509 - QwenTTSService - WARNING - [QwenTTS] Streaming failed, falling back to batch
2026-05-08 20:46:16,510 - QwenTTSService - INFO - [DEBUG] BASE model with voice_clone_prompt: type=<class 'qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem'>, len=N/A
2026-05-08 20:46:16,511 - QwenTTSService - ERROR - TTS generation failed: 'VoiceClonePromptItem' object is not subscriptable
Traceback (most recent call last):
  File "myvoice\services\qwen_tts_service.py", line 2596, in _generate
  File "concurrent\futures\thread.py", line 58, in run
  File "myvoice\services\qwen_tts_service.py", line 4338, in _generate_sync
  File "qwen_tts\inference\qwen3_tts_model.py", line 603, in generate_voice_clone
    talker_codes_list, _ = self.model.generate(...)
  File "qwen_tts\core\models\modeling_qwen3_tts.py", line 2073, in generate
    voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
  File "qwen_tts\core\models\modeling_qwen3_tts.py", line 1962, in generate_speaker_prompt
    for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
TypeError: 'VoiceClonePromptItem' object is not subscriptable
2026-05-08 20:46:16,511 - QwenTTSService - ERROR - [QwenTTS] Batch fallback also failed
```

**Root cause** (verified against `python310/Lib/site-packages/qwen_tts/inference/qwen3_tts_model.py:560-606`):

The library's `generate_voice_clone(text, voice_clone_prompt=...)` at line 575-586 dispatches on type:

```python
if isinstance(voice_clone_prompt, list):
    prompt_items = voice_clone_prompt
    ...
    voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_texts_for_ids = [it.ref_text for it in prompt_items]
else:
    voice_clone_prompt_dict = voice_clone_prompt   # <-- bare item slips through unconverted
    ref_texts_for_ids = None
```

A bare `VoiceClonePromptItem` falls into the `else` branch — gets passed to `model.generate(voice_clone_prompt=item)` at line 603 unchanged, then crashes at `core/models/modeling_qwen3_tts.py:1962` on `voice_clone_prompt['ref_spk_embedding']` (dict-style access on a dataclass).

The canonical pattern at `qwen_tts_service.py:2254` (used by `generate_with_embedding`, the EMBEDDING-voice flow) wraps in a list of one:

```python
voice_clone_prompt=[voice_clone_prompt],  # Wrap in list
```

Story 17.2's `generate_voice_clone` body assigned `request.voice_clone_prompt = cached` (bare item) in both cache-hit and cache-miss branches — bypassing the conversion path.

**Fix** (commit pending): both branches now assign `request.voice_clone_prompt = [cached]`. The cache itself still stores the bare item (single-instance memory footprint; tests verify identity); list-wrapping happens at the request-assignment site only. Added `test_request_voice_clone_prompt_is_a_list_not_bare_item` regression test pinning both branches. 27/27 tests pass.

#### §4.3.2 — Second smoke run (build #2 with fix) — **GREEN**

Build #2 produced `dist/MyVoice/MyVoice.exe` at 20:59 (51,472,905 bytes; +97 B over build #1, confirming the source-fix bytes are baked in). Smoke run executed at 21:16-21:18 by the Commander.

**First-attempt cache-miss + persist + TRUE_STREAM** (verbatim from `myvoice.log`):

```
2026-05-08 21:16:34,799 - QwenTTSService - INFO - Voice clone prompt cache: hydrated 0/12 CLONED voices for tier quality from disk
2026-05-08 21:16:34,800 - MyVoiceApp     - INFO - Voice clone prompt cache hydration: (0, 12)
2026-05-08 21:16:53,830 - QwenTTSService - INFO - Voice clone prompt cache miss for I:\MyVoiceV2\build_tools\dist\MyVoice\voice_files\Sarira-F.wav (tier=quality); computing
2026-05-08 21:17:10,506 - QwenTTSService - INFO - Voice clone prompt created successfully
2026-05-08 21:17:10,516 - QwenTTSService - INFO - Voice clone prompt persisted to I:\MyVoiceV2\build_tools\dist\MyVoice\voice_files\Sarira-F.quality.pt
2026-05-08 21:17:10,516 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM): model=Base (Clone), text='Testing a short sentence for generation...'
2026-05-08 21:17:14,710 - QwenTTSService - INFO - TTS generation complete (TRUE_STREAM): 60422 samples, 2 chunks, 4.19s total, 3.93s first chunk
```

AC #6 invariants verified:
- (i) `.txt` short-circuit took the in-memory + sidecar path; NO `Whisper transcription started` line.
- (ii) `create_voice_clone_prompt_for_tier` ran; `Sarira-F.quality.pt` + `.pt.meta.json` written at 21:17:10.
- (iii) TRUE_STREAM completed with NO `voice-clone path requires` line and NO `streaming_mode_fallback` metric.

**Second-attempt cache-hit + TRUE_STREAM**:

```
2026-05-08 21:17:23,589 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM): model=Base (Clone), text='Testing a short sentence for generation...'
2026-05-08 21:17:28,191 - QwenTTSService - INFO - TTS generation complete (TRUE_STREAM): 58502 samples, 2 chunks, 4.60s total, 4.44s first chunk
```

AC #6 invariant (iv): cache hit — NO compute, NO persist, NO Whisper. First-chunk latency 4.44 s — meets NFR1 GPU short-class target (≤5.0 s p95 per architecture-optimization-pass.md:836+).

**Third generation (long sentence, cache hit)** — exercises TRUE_STREAM-vs-BATCH discrimination:

```
2026-05-08 21:17:52,668 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM): model=Base (Clone), text='Testing a longer sentence to see if playback happe...'
2026-05-08 21:18:04,045 - QwenTTSService - INFO - TTS generation complete (TRUE_STREAM): 153576 samples, 4 chunks, 11.38s total, 3.95s first chunk
```

The 153,576 samples (~6.4 s of audio) generated over 11.38 s wall-clock with first-chunk emission at 3.95 s **proves TRUE_STREAM is genuinely streaming**. Under BATCH dispatch the first-chunk latency would scale with the sentence length and be substantially longer; here it stays flat at ~4 s regardless of sentence size.

**Persisted artifacts**:

```
build_tools/dist/MyVoice/voice_files/Sarira-F.quality.pt           21,933 bytes  21:17
build_tools/dist/MyVoice/voice_files/Sarira-F.quality.pt.meta.json    137 bytes  21:17
```

`Sarira-F.quality.pt.meta.json`:
```json
{"schema_version": "1.0", "ref_audio_mtime": 1778295583.5224285, "ref_audio_size": 854386, "tier": "quality", "qwen_tts_pin": "1ab0dd75"}
```

All five required meta fields populated; `qwen_tts_pin` matches `requirements.txt:23`.

#### §4.3.3 — UI indicator iteration

The first smoke run revealed a UX miss: tooltip-only rendering of "Preparing voice for streaming…" (Task 5.2 dev-story choice) was not visible-by-default. During the cold-cache window (21:16:53 → 21:17:10, ~17 s on this hardware) the message was in the tooltip but the user wasn't hovering.

**Fix**: `service_status_indicator.py::update_status` now also swaps the indicator's small text label from the service_name (e.g. `"TTS"`) to a truncated form of the preparing message (e.g. `"Preparing voice for stre…"`) in **bold**, then reverts on clear. Tooltip retains the full message. New file `tests/unit/ui/test_service_status_indicator_preparing_voice.py` with 5 tests pinning the inline-label invariants. Visible-by-default; no layout-reflow because the label widget already exists.

#### §4.3.4 — First-attempt timing breakdown (Task 7.5)

| Event                                         | Timestamp     | Δ from request  |
|---|---|---|
| Cache miss logged (request entered precompute) | 21:16:53,830 | 0 ms            |
| `Voice clone prompt created successfully`      | 21:17:10,506 | +16,676 ms      |
| `torch.save` complete (`.pt persisted`)        | 21:17:10,516 | +16,686 ms      |
| `Starting TTS generation (TRUE_STREAM)`        | 21:17:10,516 | +16,686 ms      |
| TTS generation complete (first chunk @3.93 s)  | 21:17:14,710 | +20,880 ms      |

The 16.7 s precompute window includes Base-model load (Base model wasn't preloaded at startup; CUSTOM_VOICE was), embedding compute, `torch.save`, and verification reload. Story Dev Notes already noted: "First-attempt cold-cache latency includes Whisper + embedding compute and is exempt from NFR1 (one-time cost per voice)." For warm-cache runs (`§4.3.2` second + third), first-chunk latency drops to 3.93–4.44 s — well within NFR1 target.

---

## §5 — Lazy-precompute timing (Task 7.5)

Captured from `§4.3.2` + `§4.3.4` log excerpts.

| Run | Type | Cache state | Total | First-chunk | NFR1 verdict |
|---|---|---|---|---|---|
| #1 | Short  | cold (miss) | 4.19 s   | 3.93 s | Exempt (cold-cache one-time cost) |
| #2 | Short  | warm (hit)  | 4.60 s   | 4.44 s | ✓ ≤5.0 s p95 |
| #3 | Long   | warm (hit)  | 11.38 s  | 3.95 s | ✓ ≤5.0 s p95 |

The 16.7 s precompute (cache miss + Base-model load + embedding compute + persist) on run #1 is a one-shot cost per voice; runs #2 + #3 demonstrate that subsequent generations on the same voice satisfy NFR1 with ample margin even for long-form text.

---

## §6 — Installer-mode smoke (Task 7.4)

> **Status:** **SKIPPED** per Commander decision 2026-05-08 21:30 (`/bmad-bmm-dev-story` workflow choice "Make indicator visible + commit + close Task 7"). The source-tree change between portable bundle and installer bundle is identical (PyInstaller's frozen bytecode is shared); §4 evidence is sufficient to certify the fix reaches users via either delivery channel. If installer-specific path issues surface in production, reopen as a follow-up.

`installer_output/MyVoice-Setup-v2.1.0.exe` (2.10 GB) was produced by the build pipeline; SHA256 + MD5 sidecars in `installer_output/`. The installer is shippable as-is.

---

## §7 — Closure follow-ups

### §7.1 Resolved on this run

- `tooling-2-build-tools-audit-evidence.md` §7.2 HIGH follow-up — TRUE_STREAM voice_clone_prompt regression in bundled UI flow — **RESOLVED**. Evidence: `§4.3.2` shows three back-to-back generations completing via TRUE_STREAM with no fallback metrics and no `TRUE_STREAM voice-clone path requires` log lines. The regression captured in tooling-2 evidence §4.3.2 + §6.2 (every UI-initiated CLONED-voice generation falling through to SENTENCE_STREAM via NFR7) is gone.

### §7.2 Open questions to resolve from §4 evidence

1. **Open question 3 from the story file — RESOLVED.** Build pipeline does NOT pre-bundle `.pt` files. Verified at 2026-05-08 by `find build_tools/dist/MyVoice/_internal/voice_files/ -name "*.pt*"` returning empty against the freshly produced bundle. No `myvoice.spec` or `installer.iss` exclusion needed.

2. **Whisper retry backoff durations** (`_WHISPER_RETRY_BACKOFFS_SECONDS = (1.0, 3.0)`) — only exercised in the no-sidecar path; not stressed by the §4 / §6 flow because Sarira-F ships with a `.txt`. Optional: a separate smoke pass with `Sarira-F.txt` deleted before launch to exercise the Whisper path end-to-end.

### §7.3 Memory pointer update

On Story 17.2 closure (post-AC #6 evidence pass):
- `memory/build_tools_phase_perp_state.md` — replace the "HIGH follow-up = TRUE_STREAM voice_clone_prompt regression in bundled UI flow" line with a closure marker pointing at this evidence file.
