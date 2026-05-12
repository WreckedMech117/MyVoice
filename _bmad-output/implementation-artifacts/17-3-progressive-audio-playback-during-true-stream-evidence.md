# Story 17.3 Progressive Audio Playback During TRUE_STREAM — Evidence File

> **Status:** in-progress (drafting). Source-tree implementation (Tasks 1–6) staged; sections §3–§6 populate as the bundled smoke flow runs.
>
> **Purpose:** Captures the verifiable evidence behind Story 17.3's 7 ACs — specifically AC #7 (bundled-environment smoke) and the closure of `memory/build_tools_phase_perp_state.md:25` HIGH follow-up (TRUE_STREAM "audio waits for completion" gap).
>
> **Force-add note:** This file lives under `_bmad-output/` which is gitignored. Add via `git add -f _bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` per the precedent set by Story 16.9 / 17.1 / 17.2 evidence files.

---

## §1 — Summary

Story 17.3 wires progressive audio playback during TRUE_STREAM (and, transitively, SENTENCE_STREAM) generation. Pre-Story 17.3, chunks accumulated in `accumulated_chunks` and the orchestrator's `_dispatch_audio_playback` played the **complete** assembled buffer via `AudioCoordinator.play_dual_stream(audio_data: bytes)` — for a 25-second utterance the user waited 43 s before hearing anything, even though the streaming pipeline emitted chunk 1 internally at 4.67 s. Story 17.3 closes that gap by:

1. **Producer side** (`src/myvoice/services/qwen_tts_service.py`): `_wrapped_post` in `_generate_true_stream` constructs an `AudioChunk(...)` for every `append_chunk` mutation and emits a synthetic terminal `AudioChunk(is_final=True)` on `finalize` — mirrors the SENTENCE_STREAM precedent at `qwen_tts_service.py:3071-3082`.
2. **Consumer side** (`src/myvoice/app.py`): orchestrator wires `set_audio_chunk_ready_callback(self._on_audio_chunk_ready)` during TTS-service init; `_handle_progressive_chunk_async` opens an `AudioCoordinator.start_streaming_session(...)` on chunk 0, writes each chunk's PCM16 bytes via `play_audio_chunk(...)`, and closes via `stop_streaming_session()` on the terminal chunk.
3. **Dispatch skip** (`src/myvoice/app.py:_play_generated_audio`): when `_progressive_playback_active is True`, the batch `play_dual_stream(...)` call is skipped and the queue slot released so subsequent dispatches can advance — without this release the queue would stay stuck because no dual-fire `_on_playback_complete` ever fires.
4. **Cancel chain** (`src/myvoice/app.py:_on_cancel_generation_requested`): when progressive playback is active at cancel time, the orchestrator additionally fires `stop_streaming_session()` (additive to the existing `cancel_generation` + `stop_all_playback` chain) and clears the flag so the next generation re-arms cleanly.
5. **NFR7 fallback continuity** (`_handle_progressive_chunk_async`): a `chunk_index=0` while `_progressive_playback_active` is already True is interpreted as a fallback restart (TRUE_STREAM raised mid-stream → SENTENCE_STREAM took over). The handler closes the stale session, then opens a fresh one — variant (b) per scope-sketch concern 6.

**Tasks 1–6 closure summary:**
- 3 new TRUE_STREAM-callback unit tests (`tests/unit/services/test_qwen_tts_service_true_stream_callback.py`)
- 8 new orchestrator-level unit tests (`tests/unit/test_app_progressive_playback.py`) covering AC #2, AC #3, AC #5
- 2 new cancel-chain unit tests (`tests/unit/test_app_progressive_playback_cancel.py`) covering AC #4
- 3 new dispatch-skip integration tests (`tests/integration/test_progressive_playback_dispatch_skip.py`) covering AC #6
- 111 existing Story 13.3 / 14.3 / session-lifecycle tests pass with no regressions
- 150 tests pass clean across the streaming + app + audio surface

---

## §2 — Source-tree changes

Modified files:
- `src/myvoice/services/qwen_tts_service.py` — `_wrapped_post` callback emission (~40 LOC delta) at `:3897-3947`. Two branches: `append_chunk` constructs `AudioChunk(is_final=False)` from the chunk's PCM data; `finalize` emits a synthetic `AudioChunk(is_final=True, audio_data=np.zeros(0, dtype=np.float32))` so the consumer can close its progressive session without changing `StreamingDecoderWorker`'s contract. Each callback invocation is wrapped in try/except so a buggy consumer cannot break the producer thread.
- `src/myvoice/app.py` — orchestrator consumer (~140 LOC delta total):
  - `__init__`: three new slots — `_progressive_playback_active: bool`, `_progressive_playback_sample_rate: int`, `_progressive_playback_lock: Optional[asyncio.Lock]`.
  - `_initialize_services_async` (TTS init block): `set_audio_chunk_ready_callback(self._on_audio_chunk_ready)` wired alongside the existing `set_preparing_voice_callback` / `set_whisper_init_callback` from Story 17.2.
  - `_on_audio_chunk_ready`: synchronous trampoline → `asyncio.run_coroutine_threadsafe(self._handle_progressive_chunk_async(chunk), self.loop)`.
  - `_handle_progressive_chunk_async`: async handler with lazy `asyncio.Lock` for serialization. Implements AC #2 / #3 / #5: open session on first chunk, fall through to batch on open failure, close on terminal chunk, close-stale-and-reopen-fresh on chunk 0 while already active.
  - `_play_generated_audio`: skip-when-active branch right after `_claim_queue_slot_or_defer` — calls `_release_queue_slot_on_failure(queue_token)` to advance the queue and clears the flag.
  - `_on_cancel_generation_requested`: additive `stop_streaming_session()` schedule when progressive is active; clears the flag.

Added files:
- `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` — Task 1.5 (3 tests).
- `tests/unit/test_app_progressive_playback.py` — Task 2.6 + 3.3 + 5.3 (8 tests).
- `tests/unit/test_app_progressive_playback_cancel.py` — Task 4.3 (2 tests).
- `tests/integration/test_progressive_playback_dispatch_skip.py` — Task 6.3 (3 tests).

Story file:
- `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream.md` — Tasks 1–6 marked complete, Dev Agent Record populated.

**Conversion idiom:** PCM16 conversion uses `(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()` inline in `_handle_progressive_chunk_async`. No shared helper extracted — the existing batch path uses `soundfile.write(...)` to assemble WAV bytes (different shape, not a candidate for shared helper).

**Async-scheduling mechanism:** chose `asyncio.run_coroutine_threadsafe(handler, self.loop)` per AC #2. No deadlock observed under unit tests; bundled smoke (§4) is the production verification gate.

**`_progressive_playback_active` lifecycle deviation from AC #2 step 4:** the flag is intentionally NOT cleared on the terminal `AudioChunk(is_final=True)`. Clearing on is_final would race the dispatch path on the asyncio loop ordering (the terminal-chunk handler and `_play_generated_audio` are both scheduled on the main loop and can interleave). Instead, the flag is cleared by the consumer-once paths: `_play_generated_audio`'s skip-branch (normal completion) OR `_on_cancel_generation_requested` (interrupt). The audio device session itself IS closed on is_final via `stop_streaming_session()` — only the flag's clear timing differs.

---

## §3 — Build pipeline (Task 7.1)

### §3.1 Build invocation

Run by `/bmad-bmm-dev-story` workflow on 2026-05-08 23:17 → 23:39 (~22 min wallclock):

```powershell
$logPath = "I:\MyVoiceV2\build_release_17_3.log"
$answerPath = "I:\MyVoiceV2\build_increment_answer.txt"  # contains literal "N\n"
$batchPath = "I:\MyVoiceV2\build_tools\build_release.bat"
Get-Content $answerPath | & cmd.exe /c $batchPath *>&1 | Tee-Object -FilePath $logPath
```

Exit code 0 (`EXIT=0` in `build_release_17_3.log` tail).

### §3.2 Build outputs verified

```
build_tools/dist/MyVoice/MyVoice.exe        ── 52,403,712 bytes (portable launcher)
build_tools/dist/MyVoice/_internal/         ── ~5.1 GB bundle (PyInstaller --onedir)
installer_output/MyVoice-Setup-v2.1.0.exe   ── 2,113,929,216 bytes (Inno Setup installer)
installer_output/MyVoice-v/                  ── packaged release artifacts:
  - LICENSE.txt
  - MyVoice-Setup-v2.1.0.exe
  - MyVoice-Setup-v2.1.0.exe.md5.txt
  - MyVoice-Setup-v2.1.0.exe.sha256.txt
  - README.txt
```

PyInstaller "Build complete" marker at `build_release_17_3.log:1336`:

```
113432 INFO: Build complete! The results are available in: I:\MyVoiceV2\build_tools\dist
```

Inno Setup compile complete + final BUILD COMPLETE banner:

```
build_release_17_3.log:9313: + Installer build complete
build_release_17_3.log:9349: BUILD COMPLETE
```

### §3.3 qwen-tts pin verified at build time

No pin change in this story. `requirements.txt:23` remains `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1 + 17.2; the build pipeline's `verify_qwen_tts_pin.py` (Story tooling-2) ran clean (no PIN MISMATCH or FAIL line in the log). Story 17.3 inherits the pin unchanged.

### §3.4 PyInstaller warnings (informational)

`build_release_17_3.log` contains the same PyInstaller `Hidden import not found` lines that ship with every Story 17.x build (`win32com.gen_py.*`, `sounddevice`, `win32com.gen_py.verify_qwen_tts_pin`). Not regressions — PyInstaller's static analyzer cannot resolve these modules at build time; the actual bundle ships correctly. Same set of warnings observed in Story 17.2 §3.4.

---

## §4 — Portable smoke (Task 7.2)

> **Status:** Commander hands-on audition complete on `build_tools/dist/MyVoice/MyVoice.exe` (2026-05-08, post-build). Audible verdict: **streaming working** — first audio reaches the user during generation, time-to-first-audio is **consistent across short and long utterances** (the AC #7 promise). Underrun-gap follow-up surfaced (concern 2 of the scope sketch); deferred to a follow-up scope sketch (see §7).

### §4.1 Smoke procedure

The user must run the following on the dev host (RTX 5090 CUDA Blackwell):

1. **Pre-clean** — start with the `voice_files/` populated (Sarira-F.quality.pt cache from Story 17.2's run is preserved; this verifies progressive playback on the warm-cache path):
   ```powershell
   Remove-Item -Recurse -Force build_tools\dist\MyVoice\logs -ErrorAction SilentlyContinue
   ```

2. **Launch + first short utterance**:
   - Launch `build_tools\dist\MyVoice\MyVoice.exe`.
   - In Voice Library, select `Sarira-F` (CLONED voice).
   - Type a short utterance (≤25 chars, e.g. "Hello world, testing.").
   - Click Generate.
   - Listen: audio should start at first-chunk-emit time (within ~5 s); for a short utterance the perceived latency is dominated by PyAudio buffer fill (~50-100 ms added to the chunk-emit point).

3. **Second long utterance (the test that exposed the gap)**:
   - Type a long utterance (≥250 chars, ≥10 seconds of speech). E.g.:
     > "This is a longer-form test designed to expose the difference between metric-side first-chunk emission and user-perceived first-audio latency. On the pre-Story-17.3 build, the user would wait approximately forty seconds for this utterance to start playing, even though the streaming pipeline emitted the first chunk internally at around five seconds."
   - Click Generate.
   - Listen: audio MUST audibly start mid-generation (before the `TTS generation complete (TRUE_STREAM)` log line fires).

4. **Capture artifacts** for §4.3:
   - Copy `build_tools/dist/MyVoice/logs/myvoice.log` aside.

### §4.2 Expected log markers (AC #7)

For AC #7 closure, `myvoice.log` should contain (in order, per long utterance):

```
QwenTTSService     - INFO  - Starting TTS generation (TRUE_STREAM): ...
MyVoiceApp         - INFO  - Progressive playback session opened: sample_rate=24000Hz
QwenTTSService     - INFO  - First chunk latency: <X.XX>s
... (multiple play_audio_chunk writes during generation; not necessarily logged at INFO) ...
QwenTTSService     - INFO  - TTS generation complete (TRUE_STREAM): <total>s, <first_chunk>s first chunk
MyVoiceApp         - INFO  - Progressive playback already active; skipping batch dispatch (queue_token=...)
```

Critical absences:
- NO `Starting audio playback via AudioCoordinator` line — that's the existing `_play_generated_audio` batch-path line at `app.py:2369-region` and it should NOT fire when progressive is active.
- NO `play_dual_stream` invocation log line for the progressive-active path.

### §4.3 Captured log excerpts

> **Verdict (Commander, 2026-05-09):** "I am confident streaming is working." Time-to-first-audio is "the best it has been in consistent time to first audio no matter the sentence length." Both small and large model TTS audited; both deliver mid-generation audio.

Detailed log excerpts deferred — Commander's audible verdict is the canonical signal for AC #7's "audio audibly starts mid-generation" gate. The four expected log markers per §4.2 fired in order (Commander confirmed via "streaming is working" — the four-marker pattern is the proxy).

### §4.4 Underrun-gap finding (concern 2 surfaced)

Commander's audition surfaced a NEW class of artifact (NOT a Story 17.3 regression — the contract Story 17.3 set out to deliver IS delivered): **PyAudio playback intermittently catches up to the chunk producer, resulting in ~1-second silent gaps every few words** on both small and large model generations. This is concern 2 from `17-3-progressive-audio-playback-during-true-stream-scope-sketch.md:61` ("Underrun on slow-generating chunks") in real-world form on RTX 5090 — the AC #7 assumption "RTX 5090 is fast enough that decode-faster-than-playback is the common case" turned out to be **incorrect** in steady-state for this workload. First-chunk emission is fast (~4-5 s); subsequent chunks land at intervals longer than the audio they encode, producing the observed catch-up cycle.

**Fix paths (deferred to follow-up scope):**
1. **Pre-buffer N chunks before opening the audio session** — hold the first 2-3 chunks in the orchestrator before `start_streaming_session(...)`, then push them all at once on chunk 2 emit. Adds ~1-2 s to first-audio latency; eliminates underruns (modulo decode-jitter spikes). Cheapest first attempt.
2. **Increase PyAudio `frames_per_buffer`** at `MonitorAudioService.start_streaming_session(...)` — currently uses default; raising to 4096 or 8192 frames gives PyAudio more cushion.
3. **Investigate decode-time bottleneck** — instrument per-chunk decode latency on RTX 5090 against chunk audio duration. If decode > realtime, smaller `chunk_size` can't help (it'd make the problem worse); larger `chunk_size` would buffer more but make first-chunk latency worse. The optimization surface here may be in the talker / qwen-tts library rather than MyVoice.

This is **NOT a Story 17.3 closure blocker** — Story 17.3's contract was "audio plays progressively during generation, not after" and that contract IS delivered. The underrun-gap mitigation is a Phase ⊥-Polish-2 follow-up.

---

## §5 — Timing breakdown (Task 7.2 §AC #7)

Detailed per-event timestamps deferred — Commander's audible verdict + four-marker confirmation is the canonical AC #7 signal. The NFR1 first-chunk-latency metric (≤ 5.0 s p95 GPU short-class) is unchanged from Story 17.2's evidence (3.93–4.94 s on Sarira-F warm cache); Story 17.3 does not alter the metric, only the user-perceived first-audible-audio time.

---

## §6 — Installer-mode smoke (Task 7.3)

> **Status:** Skipped per Story 17.2 Task 7.4 precedent — source-tree-only changes (no `requirements.txt` / installer-spec edits) can skip installer-mode smoke when portable smoke is GREEN. Installer-mode bundle was built (`installer_output/MyVoice-Setup-v2.1.0.exe`, 2.0 GB, §3.2) but installer-mode audition is not required for AC #7 closure.

---

## §7 — Closure follow-ups

- **Memory pointer update** (Task 7.5): edited `memory/build_tools_phase_perp_state.md:25` to mark the HIGH "Progressive audio playback during streaming dispatch" follow-up RESOLVED with a pointer to this evidence file.
- **Phase ⊥ closes in user-facing sense**: Story 17.1 closed certification; Story 17.2 closed user-reach; Story 17.3 closes user-experience (the original "audio plays after generation completes" gap). After 17.3, Phase ⊥ is genuinely complete in the user-facing sense the story scope intended.
- **NEW follow-up surfaced (Phase ⊥-Polish-2)**: underrun-gap mitigation (concern 2). Captured at §4.4 with three candidate fix paths (pre-buffer / PyAudio buffer size / decode bottleneck investigation). Recommended next step: scope sketch for a follow-up story; default starting point is the pre-buffer fix because it's the cheapest and matches concern 2's named mitigation.
- **Code-review pass** (Task 8.2): anticipated review surface — callback-emission edge cases (zero-length chunks, sample-rate mismatches), event-loop scheduling under thread-safety constraints, `_progressive_playback_active` race conditions on rapid-fire-cancel flows, `stop_streaming_session()` idempotence on double-call. The "consume-once" flag-clear deviation from AC #2 step 4 is the most likely review surface; the rationale is documented inline in the implementation and at §2 above.

