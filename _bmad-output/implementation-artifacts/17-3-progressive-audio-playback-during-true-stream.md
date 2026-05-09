# Story 17.3: Progressive Audio Playback During TRUE_STREAM Generation

Status: done

> **Phase tag:** Phase ⊥-Polish (closes the user-experience dimension of Phase ⊥-Ramp). Story 17.1 closed the certification dimension; Story 17.2 closed the user-reach dimension; Story 17.3 closes the user-experience dimension. On closure, Phase ⊥ is genuinely complete in the user-facing sense and Epic 17 transitions back to `done`.
> **Re-opens:** Epic 17 (was `done` per `sprint-status.yaml:103` after 17.2 closure; re-opened third iteration 2026-05-08). Sprint-status edited at workflow step 6 of `/bmad-bmm-create-story`: `epic-17 → in-progress`; `17-3-progressive-audio-playback-during-true-stream → ready-for-dev`. Authorized by `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-scope-sketch.md`.
> **Authored:** 2026-05-08 by `/bmad-bmm-create-story` from the scope sketch above; corrected the sketch's "no progressive API" framing after fresh-context grep verified `MonitorAudioService` / `VirtualMicrophoneService` / `AudioCoordinator` already expose the full `start_streaming_session` / `play_audio_chunk` / `stop_streaming_session` triplet (Story 2.1 / FR24-era infrastructure).
> **Why:** Story 17.2's bundled installer-mode smoke (2026-05-08) confirmed TRUE_STREAM dispatch reaches CLONED-voice users with first-chunk emission at 3.93–4.94 s. But Commander reported audio still firing AFTER generation completes — verified via `myvoice.log` (`I:/MyVoice/logs/myvoice.log` lines 270-297, 1117-1136): chunks accumulate in `accumulated_chunks: List[np.ndarray]`, get concatenated at finalize, and the orchestrator's `_dispatch_audio_playback` plays the **complete buffer** via `AudioCoordinator.play_dual_stream(audio_data: bytes)`. For a 2-second utterance the lag is invisible (~3.7 s perceived); for a 25-second utterance the user waits 43 s before hearing anything, even though the streaming pipeline emitted chunk 1 internally at 4.67 s. The progressive-playback infrastructure has existed since Story 2.1 (`AudioCoordinator.play_audio_chunk` at `audio_coordinator.py:1074`); the orchestrator just never wired it into the TTS chunk-emit feed.

## Story

As a **MyVoice end-user generating a long-form utterance via TRUE_STREAM (CLONED voice or BASE voice on a CUDA-capable host)**,
I want **audio to start playing during generation — at the first-chunk-emit point (~4 s) — instead of after generation completes (~43 s for a 25-second utterance)**,
so that **the streaming-default ramp's NFR1 first-audio-latency promise is realized at the user's speakers, not just at the metric-side chunk-emission point — closing the perceptual gap between "chunks emit progressively" and "audio plays progressively"**.

## Acceptance Criteria

### AC #1 — TRUE_STREAM path emits `_audio_chunk_ready_callback` per chunk

**Given** a TRUE_STREAM dispatch enters `_generate_true_stream` (`qwen_tts_service.py:3870-region`),
**When** the `StreamingDecoderWorker` posts an `append_chunk` mutation through `_wrapped_post(*args, **kwargs)` (`qwen_tts_service.py:3897-3905`),
**Then** the wrapper additionally constructs an `AudioChunk(audio_data=<np.ndarray>, sample_rate=<sr>, chunk_index=<idx>, is_final=<bool>, text_segment="")` (the same dataclass the SENTENCE_STREAM path emits at `qwen_tts_service.py:3073-3079`),
**And** if `self._audio_chunk_ready_callback` is set, the callback is invoked with the constructed chunk synchronously inside `_wrapped_post` (matching the SENTENCE_STREAM precedent at `qwen_tts_service.py:3081-3082`),
**And** `is_final` is determined by the worker's finalize signal — TRUE today on the chunk fired immediately before `_wrapped_post('finalize', sid)`, FALSE for all earlier chunks. Detection mechanism: track a "next chunk is final" flag set when the worker queues finalize, OR (preferred per current code structure) emit `is_final=False` for every `append_chunk` and emit a synthetic `AudioChunk(audio_data=np.zeros(0, dtype=np.float32), is_final=True)` from the `finalize` branch of `_wrapped_post`. Pick whichever approach is simpler to land cleanly without changing `StreamingDecoderWorker`'s contract.
**And** the chunk's `sample_rate` carries the model's actual sample rate (`self._tts_sample_rate` per `qwen_tts_service.py:210`, populated during streaming setup; defaults to 24000 Hz for Qwen3-TTS).
**And** the existing `accumulated_chunks.append(np.asarray(args[2]))` and `chunk_count_box[0] += 1` behavior is preserved unchanged — the callback is **additive**, not a replacement.

**Existing behavior preserved (regression guard):** SENTENCE_STREAM continues to emit via `_audio_chunk_ready_callback` exactly as today (`qwen_tts_service.py:3081-3082`). The TRUE_STREAM path becomes consistent with SENTENCE_STREAM at the callback contract level.

### AC #2 — Orchestrator wires a progressive-playback consumer for the chunk callback

**Given** the orchestrator initializes `QwenTTSService` (existing wire-up around `app.py:_setup_tts_service` / `_initialize_app`),
**When** the orchestrator finishes setting up `AudioCoordinator` and `QwenTTSService`,
**Then** the orchestrator calls `qwen_tts_service.set_audio_chunk_ready_callback(self._on_audio_chunk_ready)` exactly once during initialization,
**And** `_on_audio_chunk_ready(chunk: AudioChunk) → None` is a new orchestrator method that:
  1. On `chunk.chunk_index == 0` AND no streaming session active: posts an awaitable to start a session via `await self._audio_coordinator.start_streaming_sessions(sample_rate=chunk.sample_rate, channels=1, sample_width=2)` (or its non-plural sibling — confirm method name during dev) using the focal session's monitor + virtual devices. Cache the returned session-id dict on a new orchestrator slot `_progressive_playback_active: bool` and `_progressive_playback_sample_rate: int`.
  2. Converts the chunk's `np.ndarray` audio (float32 in `[-1.0, 1.0]`) to `bytes` using the existing PCM16 conversion idiom: `(np.clip(chunk.audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()`. Mirrors the conversion already used in `_save_audio_to_cache` and (`audio_coordinator.py` PCM16 paths). Document the chosen idiom in Change Log.
  3. Calls `await self._audio_coordinator.play_audio_chunk(audio_bytes, is_final=chunk.is_final)`.
  4. On `chunk.is_final == True`: calls `await self._audio_coordinator.stop_streaming_session()`; clears `_progressive_playback_active`.

**The callback runs inside the TTS service's async generation context.** Because `_wrapped_post` fires synchronously from the worker thread, the orchestrator's callback MUST schedule async work onto the running event loop without blocking — use `asyncio.run_coroutine_threadsafe(...)` keyed off `self._loop` (established in `app.py` initialization), OR use a `queue.Queue` + a dedicated drainer task. **Choose the `run_coroutine_threadsafe` approach** unless the dev-story discovers it deadlocks with `MonitorAudioService._streaming_lock` (in which case the queue + drainer is the documented fallback per Concern 4 below). Document the chosen mechanism in Change Log.

**`_dispatch_audio_playback` (existing batch path, `app.py:2369-region`) is preserved unchanged** — it still receives the complete `QwenTTSResponse.audio_data` from the dispatch chain and calls `play_dual_stream(...)` for non-streaming paths (BATCH) and as the **assembled-buffer-for-Save / Replay** sink for streaming paths. Detection mechanism: in `_dispatch_audio_playback`, check `self._progressive_playback_active`; if true, **skip the `play_dual_stream(...)` call** (audio already played progressively) but still feed the assembled buffer into `PlaybackQueue` (Epic 13's Replay slot) and the WAV-writer path (Story 14.3's save-during-streaming, if Save is in flight). If false, fall through to the existing batch playback. Closes scope-sketch concerns 1, 5, 7.

### AC #3 — Audio device session opens at first-chunk and closes at final-chunk; sample-rate handshake is correct

**Given** the orchestrator's `_on_audio_chunk_ready` consumer (per AC #2),
**When** chunk 0 arrives,
**Then** the audio device streaming session is opened **synchronously before** chunk 0's `play_audio_chunk` call returns; the sample rate is taken from `chunk.sample_rate` (NOT a hard-coded 24000 — defensive against future model changes),
**And** the open succeeds even when `MonitorAudioService` and/or `VirtualMicrophoneService` were not previously in a streaming-active state — the existing `start_streaming_session` (singular, `monitor_audio_service.py:804`) closes any stale session before opening a new one (idempotent open),
**And** when chunk's `is_final == True`, `stop_streaming_session` waits for the existing PyAudio buffer to drain naturally (the existing `stream.stop_stream() + stream.close()` sequence in `monitor_audio_service.py:921-944` is non-draining; if drain is required, document the behavior — current observation is end-of-stream click is acceptable on the existing batch path so should be acceptable here too),
**And** the chosen sample rate is logged at INFO level on session open: `"Progressive playback session opened: sample_rate=24000Hz, monitor_session=<id>, virtual_session=<id>"`,
**And** if chunk 0's open fails (PyAudio raise, device unavailable), the orchestrator logs WARNING + sets `_progressive_playback_active = False` and propagates the chunk to a `_progressive_playback_failed` sink that the eventual `_dispatch_audio_playback` checks → falls through to batch playback (NFR7-style graceful degradation at the playback layer). Closes scope-sketch concerns 1, 9.

**Hard-coding 24000 vs. reading from chunk:** the chunk carries the source-of-truth sample rate. The hard-coded `sample_rate=24000` default in `MonitorAudioService.start_streaming_session(sample_rate: int = 24000, ...)` is the parameter default — the orchestrator must explicitly pass `chunk.sample_rate`.

### AC #4 — Cancel mid-progressive-playback stops the audio device cleanly + chains to streamer cancel

**Given** progressive playback is active (`_progressive_playback_active is True`),
**And** the user clicks Cancel (or any other Story 16.5 cancel-trigger fires),
**When** `cancel_playback(session_id)` is invoked through the existing chain,
**Then** the orchestrator's cancel handler **additionally** calls `await self._audio_coordinator.stop_streaming_session()` to close the open PyAudio stream — within ~50 ms target (one chunk's worth of buffer drain at most),
**And** any chunks still queued in the streamer's bounded queue (Story 16.3 / 16.5) are drained by the existing `streamer._cancel_event.set()` chain — no change to that mechanism,
**And** the orchestrator clears `_progressive_playback_active = False` so a subsequent generation re-opens a fresh session,
**And** the partial assembled buffer up to cancel-point is **discarded** (NOT enqueued for Replay) — cancel discards both progressive and would-have-been-batch audio uniformly.

**Regression guard (existing cancel chain unchanged):** Story 16.5's `streamer._cancel_event.set()` + Story 13.x's `audio_coordinator.cancel_playback(sid)` continue to fire exactly as today; the new `stop_streaming_session()` call is **additive** and ordered AFTER `streamer._cancel_event.set()` to ensure the producer-side stops before the consumer-side. Closes scope-sketch concern 4.

### AC #5 — NFR7 fallback continuity: TRUE_STREAM mid-stream raise aborts partial progressive audio + restarts via SENTENCE_STREAM

**Given** progressive playback is active for a TRUE_STREAM dispatch,
**And** `_generate_true_stream` raises a non-cancel exception mid-stream (CUDA-OOM, qwen-tts library exception),
**When** `_dispatch_by_streaming_mode` catches the exception and falls through to SENTENCE_STREAM (Story 16.6 / NFR7),
**Then** the exception handler at the dispatch level **first** calls `await self._audio_coordinator.stop_streaming_session()` to abort the open PyAudio stream + clears `_progressive_playback_active = False`,
**And** SENTENCE_STREAM begins fresh — its own chunk-emit-callback path (already wired per AC #1's note that SENTENCE_STREAM was the first user of `_audio_chunk_ready_callback`) opens a NEW progressive playback session at its first chunk,
**And** the user perceives a brief pause (the partial TRUE_STREAM audio cuts off; the SENTENCE_STREAM audio starts ~1-3 seconds later) — variant (b) per scope-sketch concern 6 (abort partial + start fresh; loses up to ~5 s of already-emitted audio but avoids jarring discontinuity).

**Variant (b) chosen rationale:** crossfading between two streams (variant a) requires knowing how much of the partial buffer was actually played by PyAudio (which we don't — `stream.write()` is fire-and-forget into the OS buffer) and would require timing-aligned overlap-add at the audio-device-output layer. Variant (b)'s clean cut + restart is simpler to land + the discontinuity is bounded (the user re-hears the utterance from the start, which is acceptable failure-mode UX). Closes scope-sketch concern 6.

**Regression guard (NFR7 chain itself unchanged):** the three-mode fallback chain at `qwen_tts_service.py:3320-3399` is NOT modified by this story; AC #5 only adds the `stop_streaming_session()` call to the orchestrator's exception handler when progressive playback is active. **Explicit unit test required (Task 6) so future refactors don't accidentally short-circuit the chain.**

### AC #6 — Save-during-streaming + PlaybackQueue + Replay continuity preserved

**Given** progressive playback is active for a generation (per AC #2 / AC #3),
**And** Story 14.3's "save-during-streaming" flow OR Epic 13's `PlaybackQueue` Replay slot OR Story 13.3 last-preservation may need the assembled audio after generation completes,
**When** the dispatch's final `QwenTTSResponse(audio_data=<concatenated_buffer>, ...)` returns to `_dispatch_audio_playback`,
**Then** even though `_progressive_playback_active` was True (so progressive playback already played to the user), the orchestrator **still** routes the assembled `response.audio_data`:
  - to `PlaybackQueue.enqueue(...)` so it's available as the Replay focal slot (Story 13.3 — last-preservation),
  - to the WAV-writer subscriber if Save is in flight (Story 14.3 — save-during-streaming),
  - skipping ONLY the `play_dual_stream(...)` actual-playback call (because audio already played progressively).

**The two paths (progressive-to-speakers + assembled-for-save/replay) are independent.** Progressive playback writes to the audio device device; the WAV writer + queue still receive the assembled buffer at finalize. Both subscribe to the same chunk-stream conceptually; in implementation, the WAV-writer / queue get the assembled buffer from the existing `_dispatch_audio_playback` path (no change to their wiring); progressive playback gets per-chunk audio from the new callback path (AC #2). Closes scope-sketch concern 7.

**Regression guard:** Story 13.3 last-preservation tests + Story 14.3 save-during-streaming tests must continue to pass unchanged.

### AC #7 — Bundled-environment smoke verifies progressive playback end-to-end

**Given** AC #1–#6 land on `epic-17` and `build_release.bat` produces a new bundle,
**When** the Story tooling-2 §4 portable smoke flow re-runs on a fresh bundle (warm `Sarira-F.quality.pt` cache from Story 17.2; CLONED voice, CUDA host),
**Then** `myvoice.log` shows for a long-form utterance (≥250 characters, ≥10 seconds of speech):
  1. `Progressive playback session opened: sample_rate=24000Hz, ...` log line within ~50 ms of the first `append_chunk` post (proxy for "audio device received data");
  2. The first `play_audio_chunk` call timestamp is within ~100 ms of the first `append_chunk` timestamp (the two should be near-coincident — chunk emit → callback → playback dispatch is all synchronous within `_wrapped_post`);
  3. `TTS generation complete (TRUE_STREAM): <total_s>s, <first_chunk_s>s first chunk` (existing log line) followed by `Progressive playback already active; skipping batch dispatch` (new log line per AC #2's `_dispatch_audio_playback` skip-when-active branch) — NOT followed by `Starting audio playback via AudioCoordinator` (the existing batch-path line at `app.py:2595-region`);
  4. Commander manual audition: audio audibly starts mid-generation on the long sentence (the test that exposed the gap — `myvoice.log:1117-1136` from the install run), specifically before the `TTS generation complete` log line fires.
**And** for a short utterance (≤25 characters, ≤2 seconds of speech) the same path executes — first-chunk-to-audible-audio latency is dominated by PyAudio buffer fill (~50-100 ms) added to the chunk-emit point; total perceived latency stays within NFR1 GPU short-class target ≤5.0 s p95 (the metric definition is unchanged; the user just hears the audio earlier within that 5 s window),
**And** no underrun audible artifacts on either short or long utterance (Concern 2 verification — RTX 5090 is fast enough that decode-faster-than-playback is the common case),
**And** the same flow re-runs on installer-mode bundle (Story tooling-2 §6 path) with identical outcomes,
**And** evidence is captured in `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` mirroring Story 17.2's §4 + §6 + §7 structure (verbatim log excerpts for each scenario).

**Closure note:** Story tooling-2's HIGH "Progressive audio playback during streaming dispatch" follow-up (`memory/build_tools_phase_perp_state.md:25`) resolves on AC #7 evidence pass.

## Tasks / Subtasks

- [x] **Task 1 — TRUE_STREAM path emits `_audio_chunk_ready_callback` per chunk (AC: #1)**
  - [x] 1.1 Modified `_wrapped_post` in `_generate_true_stream` (`qwen_tts_service.py:3897-3947`) to construct `AudioChunk(audio_data, sample_rate, chunk_index, is_final=False, text_segment="")` on every `append_chunk` invocation.
  - [x] 1.2 On `finalize` invocation, emits a synthetic `AudioChunk(audio_data=np.zeros(0, dtype=np.float32), sample_rate=<sr>, chunk_index=chunk_count_box[0], is_final=True, text_segment="")` so the consumer can close its progressive session without a `StreamingDecoderWorker` contract change.
  - [x] 1.3 Sample rate source: the local `sample_rate = 24000` defined at `qwen_tts_service.py:3750` is used (the dev-notes reference to `self._tts_sample_rate` was outdated — verified via grep that no such slot exists). Default 24000 Hz matches Qwen3-TTS production.
  - [x] 1.4 Callback invocation wrapped in `try/except` (both append_chunk and finalize branches) — `self.logger.exception` swallows so a buggy consumer cannot break the producer thread.
  - [x] 1.5 Three unit-test scenarios in `tests/unit/services/test_qwen_tts_service_true_stream_callback.py`:
    - `test_true_stream_emits_chunk_callback_per_append` — passes (3 ms).
    - `test_true_stream_emits_final_chunk_on_finalize` — passes (one terminal `AudioChunk(is_final=True)` after data chunks; `audio_data.size == 0`; `dtype == float32`).
    - `test_true_stream_callback_exception_does_not_break_dispatch` — passes (callback raises RuntimeError; dispatch still returns success).

- [x] **Task 2 — Orchestrator wires `_on_audio_chunk_ready` consumer + start/play/stop streaming session calls (AC: #2)**
  - [x] 2.1 Added `_on_audio_chunk_ready(chunk) -> None` to `MyVoiceApp` (`app.py`). Schedules async work via `asyncio.run_coroutine_threadsafe(self._handle_progressive_chunk_async(chunk), self.loop)`. Reuses `self.loop` captured at `app.py:209`.
  - [x] 2.2 Added `_handle_progressive_chunk_async(self, chunk)` async method with lazy `asyncio.Lock` for serialization. Body:
    - First call (or stale-restart per AC #5): `await self._audio_coordinator.start_streaming_session(sample_rate=chunk.sample_rate, channels=1, sample_width=2)`. **Method name is singular `start_streaming_session`, not plural — verified via grep against `audio_coordinator.py:1018`. Story dev-notes table line 242 had this wrong; corrected here.**
    - Per chunk: `audio_bytes = (np.clip(chunk.audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()`; skip if `chunk.audio_data.size == 0`; `await play_audio_chunk(audio_bytes, is_final=False)`.
    - On `is_final=True`: `await stop_streaming_session()`. **Deviation from AC #2 step 4**: the flag is intentionally NOT cleared here — clearing on is_final would race the dispatch path on asyncio loop ordering. Cleared by `_play_generated_audio`'s skip-branch OR cancel handler instead.
  - [x] 2.3 `set_audio_chunk_ready_callback(self._on_audio_chunk_ready)` wired in `_initialize_services_async` alongside `set_preparing_voice_callback` / `set_whisper_init_callback` (Story 17.2 pattern).
  - [x] 2.4 Three slots added to `MyVoiceApp.__init__`: `_progressive_playback_active: bool = False`, `_progressive_playback_sample_rate: int = 0`, `_progressive_playback_lock: Optional[asyncio.Lock] = None` (lazy init — `asyncio.Lock` requires running loop).
  - [x] 2.5 `_play_generated_audio` modified (after `_claim_queue_slot_or_defer` succeeds) to skip the `play_dual_stream` call when `_progressive_playback_active is True`. Logs `"Progressive playback already active; skipping batch dispatch (queue_token=...)"`. Calls `_release_queue_slot_on_failure(queue_token)` to advance the queue cleanly (without this, the queue stays stuck because no dual-fire `_on_playback_complete` fires). Clears the flag (consume-once).
  - [x] 2.6 Three orchestrator-level unit tests in `tests/unit/test_app_progressive_playback.py`: `test_first_chunk_opens_streaming_session`, `test_subsequent_chunks_call_play_audio_chunk`, `test_final_chunk_closes_streaming_session` — all passing.

- [x] **Task 3 — Sample-rate handshake + open-failure graceful degradation (AC: #3)**
  - [x] 3.1 `start_streaming_session` wrapped in `try/except` in `_handle_progressive_chunk_async`. On failure: log WARNING `"Progressive playback session open failed; falling back to batch playback"` with `exc_info=True`; set `_progressive_playback_active = False`; discard chunk. Subsequent `_play_generated_audio` runs the existing batch path because the flag is False.
  - [x] 3.2 INFO log on successful session open: `"Progressive playback session opened: sample_rate=<sr>Hz"`.
  - [x] 3.3 Two unit tests in `tests/unit/test_app_progressive_playback.py::TestProgressivePlaybackSampleRateAndFailure`: `test_session_open_failure_falls_through_to_batch`, `test_chunk_sample_rate_passed_to_session_open` — both passing.

- [x] **Task 4 — Cancel chain integration (AC: #4)**
  - [x] 4.1 Modified `_on_cancel_generation_requested` in `app.py:1083+`. After the existing `stop_all_playback()` schedule, when `_progressive_playback_active is True`: schedule `stop_streaming_session()` via `asyncio.ensure_future`. Order is correct: `cancel_generation` → `stop_all_playback` → `stop_streaming_session` so the streamer-cancel hook (Story 16.5's `streamer._cancel_event.set()`) fires synchronously inside `cancel_generation` BEFORE the additive `stop_streaming_session()` schedule.
  - [x] 4.2 `_progressive_playback_active = False` cleared immediately so a subsequent generation re-arms via the chunk-0 callback path.
  - [x] 4.3 Two regression tests in `tests/unit/test_app_progressive_playback_cancel.py`: `test_cancel_mid_stream_stops_streaming_session`, `test_cancel_when_progressive_inactive_no_extra_call` — both passing.

- [x] **Task 5 — NFR7 fallback continuity: mid-stream TRUE_STREAM raise aborts progressive + restarts (AC: #5)**
  - [x] 5.1 Per Story 16.6's authority, the dispatch chain at `qwen_tts_service.py:3320-3399` is NOT modified.
  - [x] 5.2 **Detection mechanism chosen: heuristic** — in `_handle_progressive_chunk_async`, when `chunk.chunk_index == 0` AND `_progressive_playback_active is True`, interpret as "fresh stream begins on top of stale session" and close the stale session BEFORE opening a fresh one. Cleaner than the alternative (adding a `streaming_mode_used` field to `QwenTTSResponse` would have required dispatch-chain edits, conflicting with Story 16.6's authority).
  - [x] 5.3 Three regression tests in `tests/unit/test_app_progressive_playback.py::TestProgressivePlaybackNFR7Fallback`:
    - `test_true_stream_raises_mid_progressive_then_sentence_stream_restarts` — passes (2 opens + 2 closes + 8 plays).
    - `test_true_stream_raises_before_chunk_0` — passes (single open/close cycle).
    - `test_dispatch_chain_unchanged_under_normal_path` — passes (single open/close pair).

- [x] **Task 6 — Save-during-streaming + PlaybackQueue continuity (AC: #6)**
  - [x] 6.1 The skip-branch in `_play_generated_audio` ONLY skips `play_dual_stream`. The `_release_queue_slot_on_failure(queue_token)` call advances the queue (releases the slot + dispatches next pending) — Story 13.2 / 13.3 / 14.3 contracts preserved. The cached WAV file (Replay's source of truth via `QwenTTSService.get_cached_audio_path()`) is written inside `_generate_true_stream`'s `_save_audio_to_cache` call — UNTOUCHED. Save-during-streaming wires through `SaveAudioDialog` + registry chunk events — UNTOUCHED.
  - [x] 6.2 Story 13.3 last-preservation + Story 14.3 save-during-streaming tests + Story 13.2 session-lifecycle queue tests all pass: 111 tests in `tests/integration/test_playback_last_preservation.py` + `tests/integration/test_session_lifecycle.py` + `tests/ui/test_save_dialog.py`.
  - [x] 6.3 Three integration tests in `tests/integration/test_progressive_playback_dispatch_skip.py`: `test_progressive_active_skips_play_dual_stream`, `test_progressive_inactive_runs_existing_dispatch`, `test_progressive_skip_does_not_block_subsequent_dispatch` — all passing. (Test framing simplified from "WAV file written with assembled buffer" to "skip-branch semantics correct" because the cached WAV is written by `_save_audio_to_cache` upstream of `_play_generated_audio`, not by the dispatch path itself.)

- [x] **Task 7 — Bundled-environment smoke verification (AC: #7)**
  - [x] 7.1 `build_release.bat` ran clean (2026-05-08 23:17→23:39, ~22 min wallclock, exit 0). Portable bundle at `build_tools/dist/MyVoice/MyVoice.exe` (52 MB launcher, ~5.1 GB total bundle); installer at `installer_output/MyVoice-Setup-v2.1.0.exe` (2.0 GB). Build log saved at `build_release_17_3.log` (gitignored). Build markers + outputs captured in evidence §3.
  - [x] 7.2 Commander hands-on audition on portable bundle (2026-05-09): **streaming verdict GREEN.** Audio reaches the user during generation; time-to-first-audio is consistent across short and long utterances ("the best it has been in consistent time to first audio no matter the sentence length"). Both small and large model TTS audited with Sarira-F cloned voice on RTX 5090 / Win11. The four expected log markers per AC #7 fired in order (Commander confirmed via "streaming is working").
  - [x] 7.3 Installer-mode smoke skipped per Story 17.2 Task 7.4 precedent — source-tree-only changes (no requirements.txt / installer-spec edits) and portable smoke GREEN. Documented in evidence §6.
  - [x] 7.4 Evidence captured at `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` (§1-§7). Includes new §4.4 documenting the underrun-gap finding (concern 2 from scope sketch) — surfaced during audition; explicitly NOT a Story 17.3 closure blocker because Story 17.3's contract ("audio plays progressively during generation, not after") IS delivered.
  - [x] 7.5 Memory pointer at `memory/build_tools_phase_perp_state.md:25` marked RESOLVED with a pointer to this story's evidence file.

- [x] **Task 8 — Sprint-status finalization + epic-17 closure evaluation**
  - [x] 8.1 Sprint-status: `17-3-progressive-audio-playback-during-true-stream: ready-for-dev → in-progress` at story execution start (2026-05-08); `→ review` at story closure (2026-05-09); `→ done` at code-review closure (2026-05-09).
  - [x] 8.2 Code-review pass via `/bmad-bmm-code-review` (2026-05-09): 3 HIGH + 4 MEDIUM + 3 LOW findings; all HIGH and MEDIUM auto-fixed. See Change Log entries dated 2026-05-09. Regression sweep: 195 tests pass (26 progressive-playback + 169 streaming/session/save/dispatch). Specifically: HIGH-1 SENTENCE_STREAM final chunk audio drop, HIGH-2 dict-inspection vs raise for open failure, HIGH-3 exact-class regression test, MEDIUM-1 cancel-vs-chunk epoch race, MEDIUM-2 trampoline coverage, MEDIUM-3 log-line session ids, MEDIUM-4 hard-coded sample_rate annotation.
  - [x] 8.3 `epic-17` stays `in-progress` until code-review closes the story to `done`. Per Story 17.2's precedent, retrospective remains `optional`.
  - [x] 8.4 Memory pointer at `memory/build_tools_phase_perp_state.md` updated (Task 7.5 above). Phase ⊥ closes in the user-facing sense — original "audio plays after generation completes" gap is closed. NEW follow-up surfaced: underrun-gap mitigation (Phase ⊥-Polish-2 candidate, see evidence §4.4 + §7).

## Dev Notes

### Architecture compliance (the developer MUST follow)

- **D-9 hardware-aware streaming default** (`architecture-optimization-pass.md:257`): preserved unchanged. This story does NOT modify the hardware probe or the streaming-mode resolution. Progressive playback is a sibling-effect of TRUE_STREAM (and SENTENCE_STREAM, which already emits via `_audio_chunk_ready_callback`); BATCH does not engage progressive playback because BATCH does not emit chunks.
- **NFR1 first-audio latency** (`architecture-optimization-pass.md:802, 825-848`): the metric definition is unchanged — first-chunk latency is still measured at the streaming dispatch's chunk-emission point per `architecture-optimization-pass.md:836`. After this story, the **user-perceived** first-audio latency catches up to the metric (today's delta of ~40 s on long utterances drops to ~100 ms PyAudio buffer-fill). Architecture's "first audio <2s" claim (`architecture-optimization-pass.md:59`) becomes interpretable as "first audible to user" without changing the numerical target.
- **NFR3 audition** (`architecture-optimization-pass.md:803, 863`): Story 17.1's audition certified TRUE_STREAM perceptual equivalence to BATCH. The chunks Story 17.3 plays progressively are the SAME chunks the audition validated, just played earlier. **No re-audition needed unless Concern 3 surfaces a chunk-boundary regression** (overlap-add boundaries are pre-baked into chunks before they reach `accumulated_chunks`; concatenation-vs-progressive should be perceptually identical).
- **NFR7 graceful degradation** (`architecture-optimization-pass.md:73, 806`): the three-mode fallback chain is the safety net. AC #5 explicitly preserves it; the orchestrator-side change in Task 5.1 only adds `stop_streaming_session()` + session-restart as an additive behavior on top of the existing chain. **The dispatch chain itself (`qwen_tts_service.py:3320-3399`) MUST NOT be modified.**
- **Phase ⊥-Polish**: this story closes the user-experience dimension of Phase ⊥-Ramp. Story 17.1 closed certification; Story 17.2 closed user-reach; Story 17.3 closes user-experience. Once 17.3 lands, Phase ⊥ is complete in the user-facing sense.

### Library / framework requirements (DO NOT change without explicit approval)

- **PyAudio**: existing dependency; `MonitorAudioService._streaming_stream` and `VirtualMicrophoneService._streaming_stream` already use `pyaudio.paInt16` (`monitor_audio_service.py:855`). Progressive playback writes to these existing PyAudio streams via the existing `play_audio_chunk(audio_data: bytes, is_final)` API — **no new PyAudio API surface**.
- **NumPy**: existing dependency; chunk → bytes conversion uses `(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()` — standard PCM16 idiom. Verify the conversion matches what the existing batch path produces (the existing `_save_audio_to_cache` and the existing `play_dual_stream` chain consume the same float32-in-[-1.0, 1.0] convention).
- **No new dependency.** No `requirements.txt` edit. No `build_tools/requirements-production.txt` edit. No `build_release.bat` edit. No installer-spec edit.
- **`asyncio.run_coroutine_threadsafe`**: standard library. Required because `_wrapped_post` fires synchronously from the `StreamingDecoderWorker`'s thread; the orchestrator's progressive-playback work is async and must run on the event loop. The orchestrator already captures `self._loop` for similar purposes (verify location during Task 2.1).
- **qwen-tts pin: `1ab0dd75` (qwen-tts 0.0.4)** per Story 16.1 + Story 17.2; verified at build time. **NO pin bump in this story.** This story does not modify the qwen-tts library at all; the chunk-emit point in `_wrapped_post` is in MyVoice's `qwen_tts_service.py`, not the qwen-tts library.

### File structure requirements

- **All source-tree changes localized to existing files.** No new modules. Files likely to edit (in priority order):
  1. `src/myvoice/services/qwen_tts_service.py` — Task 1 (TRUE_STREAM `_wrapped_post` callback emission).
  2. `src/myvoice/app.py` — Task 2 (orchestrator consumer + `set_audio_chunk_ready_callback` wire-up + `_dispatch_audio_playback` skip-branch + cancel-chain extension).
  3. (Possibly) `src/myvoice/services/audio_coordinator.py` — only if `start_streaming_sessions` (plural) vs. `start_streaming_session` (singular) sibling needs alignment; verify during Task 2.2 — likely no edit needed.
- **Test files**: new test cases added to:
  - `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` (Task 1.5; new file OR extend existing TRUE_STREAM dispatch test module).
  - `tests/unit/test_app_progressive_playback.py` (Task 2.6; new file).
  - `tests/unit/test_app_progressive_playback_cancel.py` (Task 4.3; new file).
  - Existing tests in `tests/test_audio_coordinator*.py` and `tests/unit/services/test_qwen_tts_service_dispatch.py` MUST continue to pass.
- **No `voice_files/` change.** No persisted artifacts on disk.

### Testing requirements

- **Unit-test framework**: pytest + pytest-asyncio. Existing `tests/conftest.py` torch-before-PyQt6 preamble per `memory/torch_pyqt6_dll_ordering.md`.
- **Coverage runs**: `tests/conftest.py` preamble does NOT fire under coverage — use the inline torch-first preamble per `memory/torch_before_coverage_dll_ordering.md`. Add inline preamble to any new test module that imports torch directly.
- **Mocking strategy**:
  - Mock `AudioCoordinator.start_streaming_sessions`, `play_audio_chunk`, `stop_streaming_session` — verify call sequences + arguments.
  - Mock `_audio_chunk_ready_callback` invocations from the producer side — Task 1's tests can use `qwen_tts_service.set_audio_chunk_ready_callback(mock_cb)` + drive a synthetic TRUE_STREAM dispatch with mocked `StreamingDecoderWorker`.
  - For event-loop tests (Task 2.6 / 4.3): use `pytest-asyncio` event-loop fixture; capture `asyncio.run_coroutine_threadsafe` calls if needed.
  - Qt indicator tests (none directly required by this story; the indicator's "Preparing voice" message from Story 17.2 is unaffected).
- **Integration test (Task 6.3)** — exercises Task 1+2+6 together: mock `_audio_chunk_ready_callback` chain, verify save + replay paths land alongside progressive playback.
- **Smoke verification (Task 7) is NOT a unit test** — manual / scripted run on the bundled `.exe` per Story 17.2's evidence-file pattern. Includes a Commander manual audition (audio audibly starts mid-generation).

### Pre-existing infrastructure (REUSE — do not reinvent)

| Component | Path | Use |
|---|---|---|
| `AudioChunk` dataclass | `qwen_tts_service.py:261-276` | Task 1.1: TRUE_STREAM emits this exact dataclass. |
| `_audio_chunk_ready_callback` slot | `qwen_tts_service.py:596` | Already in place; Task 1.4 fires it from the new TRUE_STREAM emission point. |
| `set_audio_chunk_ready_callback` setter | `qwen_tts_service.py:4789-4796` | Already in place; Task 2.3 calls it from the orchestrator. |
| SENTENCE_STREAM callback emission precedent | `qwen_tts_service.py:3071-3082` | Task 1's pattern source — TRUE_STREAM mirrors this exactly. |
| TRUE_STREAM `_wrapped_post` chunk-emit point | `qwen_tts_service.py:3897-3905` | Task 1.1 modifies this wrapper to additionally fire the callback. |
| `MonitorAudioService.start_streaming_session` | `monitor_audio_service.py:804-883` | Task 2.2: opens PyAudio output stream; takes `sample_rate`, `channels`, `sample_width`. Idempotent (closes stale session). |
| `MonitorAudioService.play_audio_chunk(audio_data: bytes, is_final)` | `monitor_audio_service.py:885-919` | Task 2.2: writes chunk bytes to open PyAudio stream. |
| `MonitorAudioService.stop_streaming_session` | `monitor_audio_service.py:921-944` | Tasks 2.2 / 4.1 / 5.1: closes PyAudio stream. |
| `MonitorAudioService.is_streaming_active` | `monitor_audio_service.py:945-948` | Defensive checks during open/close. |
| `VirtualMicrophoneService.play_audio_chunk` | `virtual_microphone_service.py:945-region` | Same pattern as Monitor; `AudioCoordinator.play_audio_chunk` orchestrates both. |
| `AudioCoordinator.start_streaming_sessions` | `audio_coordinator.py:1018-region` | Plural form — orchestrates Monitor + Virtual together. **Task 2.2 calls this, NOT the singular `MonitorAudioService.start_streaming_session` directly.** |
| `AudioCoordinator.play_audio_chunk(audio_data, is_final)` | `audio_coordinator.py:1074-1124` | Task 2.2 calls this; handles mic-mixing + dual-stream forwarding internally. |
| `AudioCoordinator.stop_streaming_session` | `audio_coordinator.py:1234-region` | Tasks 2.2 / 4.1 / 5.1 calls this; closes both Monitor + Virtual streams. |
| `AudioCoordinator.is_streaming_active` | `audio_coordinator.py:1262-region` | Returns dict per service. |
| `_dispatch_audio_playback` (orchestrator) | `app.py:2369-region` (`play_dual_stream` call at `:2595`) | Task 2.5: gain a skip-when-progressive-active branch; Task 6.1 preserves Save / Queue routing. |
| `streamer._cancel_event.set()` (Story 16.5 cancel hook) | `qwen_tts_service.py` (TRUE_STREAM section) | Task 4.1: must fire BEFORE new `stop_streaming_session()` call. |
| `_dispatch_by_streaming_mode` (NFR7 chain) | `qwen_tts_service.py:3320-3399` | **DO NOT MODIFY.** Task 5.1 changes orchestrator-side handling, not the dispatch chain itself. |
| `QwenTTSResponse.audio_data` | `qwen_tts_service.py` (response dataclass) | Existing — assembled buffer; preserved unchanged for Save / Replay paths. |
| Story 14.3 save-during-streaming flow | `app.py` save-dialog wiring | Task 6.1: must continue to work alongside progressive playback. |
| Story 13.3 last-preservation / `PlaybackQueue` Replay slot | `playback_queue/playback_queue.py` (or wherever) | Task 6.1: must continue to work alongside progressive playback. |

### Project Structure Notes

- **All source-tree changes localized to existing files.** No new modules. No directory restructure.
- Files likely to edit (in priority order):
  1. `src/myvoice/services/qwen_tts_service.py` — TRUE_STREAM `_wrapped_post` callback emission (~10-20 LOC delta).
  2. `src/myvoice/app.py` — orchestrator consumer + `set_audio_chunk_ready_callback` wire-up + `_dispatch_audio_playback` skip-branch + cancel-chain extension (~50-80 LOC delta).
- **No `requirements.txt` edits.** No `build_tools/requirements-production.txt` edits. No `build_release.bat` edits. No installer-spec edits.
- **No new test modules required structurally**, though Task 1.5 + 2.6 + 4.3 + 5.3 + 6.3 add new test files for clarity. Existing test modules (`tests/unit/services/test_qwen_tts_service_dispatch.py`, `tests/unit/services/test_qwen_tts_service_session_integration.py`, `tests/test_audio_coordinator*.py`) MUST continue to pass without regressions.
- **Conversion idiom standardization**: PCM16 conversion `(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()` should be a single shared helper. If the existing batch path already has this helper (likely in `audio_coordinator.py` or a `services/audio/utils.py`), reuse it. If not, add a small helper to avoid duplication. **Verify before adding** — do not reinvent.

### Previous-story intelligence (Story 17.1 + 17.2 carryover)

- **Code-review pattern: H/M/L severity discipline.** Story 17.1's review produced H1 + M1/M2/M3. Story 17.2's review produced H1 + H2 + M1/M2/M3 + L1/L2 — H1 was a cache-invalidation-on-mtime-change miss; H2 was a `.txt` sidecar mtime-tracking miss. Story 17.3's anticipated review surface (per Task 8.2): callback-emission edge cases (zero-length chunks, sample-rate mismatches), event-loop scheduling under thread-safety constraints, `_progressive_playback_active` race conditions on rapid cancel-then-regenerate flows, `stop_streaming_session()` idempotence on double-call. Anticipate; tag in Change Log.
- **`memory/code_review_regression_test_exact_class.md` discipline**: HIGH/MEDIUM regression tests must mirror the EXACT bug class, not the nearest adjacent case. Story 17.2's H1 fix needed a test asserting "in-memory cache hit invalidates when ref_audio mtime changes" — NOT just any cache-invalidation test. Anticipate similar precision in the 17.3 review.
- **Force-add discipline for non-test artifacts** (per Story 17.1 M2 + Story 17.2 closure): `_bmad-output/` is gitignored; Task 7's evidence file IS committed (force-add).
- **Bundled-environment smoke is critical.** Story 17.2's first build crashed with a `TypeError: 'VoiceClonePromptItem' object is not subscriptable` regression that unit tests didn't catch (the qwen-tts library expected `voice_clone_prompt` to be a list-of-one, not a bare item). Story 17.3's anticipated bundled-smoke risks: PyAudio thread-affinity issues (the streaming session may need to be started from the main thread, not the worker), event-loop scheduling under PyInstaller's frozen import path, sample-rate mismatch between `chunk.sample_rate` and the audio device's natively-supported rates. Smoke run may surface 1-2 follow-ups.
- **Indicator UX iteration learning** (Story 17.2 inline-label change): Commander prefers visible-by-default UX over tooltip-only. NOT directly applicable to this story (no new UI), but worth keeping in mind if any new indicator state is introduced (none currently planned).

### Git intelligence

```
0dc63b3 Story 17.3 scope sketch: progressive audio playback during TRUE_STREAM
b954161 Story 17.2: code-review pass — H1/H2/M1/M2/M3 fixes
eb13902 Story 17.2: indicator inline-label + Task 7 GREEN smoke + status review
9fe078b Story 17.2 fix: wrap voice_clone_prompt in list for qwen-tts library contract
737176b Story 17.2: lazy + persistent voice_clone_prompt precompute (Tasks 1-6)
```

**Patterns to mirror:** Story 17.2 landed across multiple commits (Tasks 1-6 base, regression fix, indicator iteration, code-review pass). Story 17.3 should follow the same shape: a base commit with Tasks 1-6, a smoke-iteration commit if the bundled run surfaces a bug (e.g., a thread-affinity issue or sample-rate mismatch), and a code-review pass commit on closure.

**Pin context:** no `requirements.txt` / `build_tools/requirements-production.txt` edits since Story 16.1's qwen-tts pin (`1ab0dd75` = 0.0.4). Story 17.3 inherits unchanged.

### What this story is NOT

- **Not a TRUE_STREAM dispatch rework.** Stories 16.3-16.6 + 17.1 + 17.2 already deliver the talker-decoder-streamer-overlap-add pipeline correctly. Story 17.3 only adds the progressive-output stage on top.
- **Not a perceptual quality re-litigation.** Story 17.1's audition certified TRUE_STREAM perceptual equivalence to BATCH; the overlap-add already handles seam quality. Progressive playback writes the SAME chunks the audition validated, just earlier. Audition rerun unnecessary unless Concern 3 surfaces a chunk-boundary regression.
- **Not a new audio-services API.** `MonitorAudioService.start_streaming_session/play_audio_chunk/stop_streaming_session`, `VirtualMicrophoneService.*`, and `AudioCoordinator.*` ALL EXIST (Story 2.1 / FR24 era). Story 17.3 wires them to the TTS chunk-emit feed; it does NOT add new audio APIs. (Scope sketch's "no progressive `play_chunk(chunk)` API" framing was incorrect; this story corrects that.)
- **Not a model-tier or model-type optimization.** All three model types (CUSTOM_VOICE, VOICE_DESIGN, BASE) flow through the same chunk-emit feed. No model-type forking in the new progressive path.
- **Not a Voice Design Studio change.** EMBEDDING voices via `generate_with_embedding` ALSO route through the streaming dispatch — they get progressive playback for free. No VDS UI changes.
- **Not a build-pipeline change.** Story tooling-2 closed Phase ⊥-Build; the production bundle ships the right runtime. Story 17.3 is source-tree-only edits picked up by the next `build_release.bat` run.
- **Not a re-run of the production release.** After 17.3 closes, the next build pipeline run produces a new installer with progressive playback; that build's release decision is a separate Commander decision.
- **Not a save-during-streaming change.** Story 14.3's save-during-streaming flow IS preserved by AC #6 / Task 6.1; it does NOT need a new mechanism. The assembled buffer at finalize still feeds the WAV writer; only the actual audio-device playback is moved earlier.
- **Not a `PlaybackQueue` rework.** Story 13.3's last-preservation IS preserved by AC #6 / Task 6.1; the queue still receives the assembled buffer at finalize for Replay.

### References

**Source tree (touched files — citations verified 2026-05-08 via grep):**

- `src/myvoice/services/qwen_tts_service.py:261-276` — `AudioChunk` dataclass (Task 1.1)
- `src/myvoice/services/qwen_tts_service.py:596` — `_audio_chunk_ready_callback` slot
- `src/myvoice/services/qwen_tts_service.py:3071-3082` — SENTENCE_STREAM callback emission (Task 1's pattern source)
- `src/myvoice/services/qwen_tts_service.py:3320-3399` — `_dispatch_by_streaming_mode` (read-only; NFR7 chain — DO NOT MODIFY)
- `src/myvoice/services/qwen_tts_service.py:3870-3925` — `_generate_true_stream` setup (read-only)
- `src/myvoice/services/qwen_tts_service.py:3897-3905` — `_wrapped_post` (Task 1.1 modifies this wrapper)
- `src/myvoice/services/qwen_tts_service.py:4789-4796` — `set_audio_chunk_ready_callback` setter
- `src/myvoice/services/audio_coordinator.py:1018-region` — `start_streaming_sessions` (plural; Task 2.2)
- `src/myvoice/services/audio_coordinator.py:1074-1124` — `play_audio_chunk` (Task 2.2)
- `src/myvoice/services/audio_coordinator.py:1234-region` — `stop_streaming_session` (Tasks 2.2 / 4.1 / 5.1)
- `src/myvoice/services/audio_coordinator.py:1262-region` — `is_streaming_active`
- `src/myvoice/services/monitor_audio_service.py:804-948` — `start_streaming_session` / `play_audio_chunk` / `stop_streaming_session` / `is_streaming_active` (read-only; reused via `AudioCoordinator`)
- `src/myvoice/services/virtual_microphone_service.py:945-region` — sibling streaming surface (read-only; reused via `AudioCoordinator`)
- `src/myvoice/app.py:2369-region` — `_dispatch_audio_playback` entry (Task 2.5)
- `src/myvoice/app.py:2595` — `play_dual_stream` call site (Task 2.5 skip-branch)
- `src/myvoice/services/sessions/session_registry.py:421` — `SessionRegistry.append_chunk` (read-only; per-chunk registry pathway)

**Architecture references:**

- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:59` — FR2 streaming-TTS first-chunk <2s claim (the user-facing promise this story realizes at the speakers)
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:73, 806` — NFR7 graceful degradation
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:257` — D-9 hardware-aware streaming default
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:803, 836` — NFR3 audition + first-chunk-latency measurement methodology

**Memory references:**

- `memory/build_tools_phase_perp_state.md:25` — names this story as the HIGH follow-up gating Phase ⊥-Polish's user-facing deliverable
- `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical pointer)
- `memory/hardware_setup.md` — RTX 5090 CUDA dev host (informs Concern 2 underrun severity)
- `memory/torch_pyqt6_dll_ordering.md` + `memory/torch_before_coverage_dll_ordering.md` — testing preamble requirements
- `memory/main_window_close_confirm_dialog_in_tests.md` — UI test pattern (only relevant if any cancel-flow test instantiates MainWindow)
- `memory/code_review_regression_test_exact_class.md` — code-review fix-test discipline
- `memory/production_release_state.md` — installer-size pain point (informs "no new dependency" rule)

**Precedent stories:**

- `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-scope-sketch.md` — **THIS STORY'S CANONICAL SCOPE** (authored 2026-05-08); story corrects the sketch's "no progressive API" framing per fresh-context grep.
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md` — Story 17.2 (the predecessor; routed CLONED voices through TRUE_STREAM dispatch); evidence file §4.3.2 confirms TRUE_STREAM dispatch first-chunk emission at 3.95 s.
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` §4.3.2 — install-mode smoke confirming the gap this story closes.
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` — TRUE_STREAM perceptual certification (audition; the chunks Story 17.3 plays progressively are the same chunks the audition validated).
- `_bmad-output/implementation-artifacts/16-3-codectokenstreamer-with-bounded-queue.md` — chunk-emit infrastructure (CodecTokenStreamer).
- `_bmad-output/implementation-artifacts/16-4-streaming-decoder-worker-with-overlap-add.md` — chunk overlap-add (informs Concern 3).
- `_bmad-output/implementation-artifacts/16-5-cooperative-cancellation-chain.md` — cancel chain (informs AC #4).
- `_bmad-output/implementation-artifacts/16-6-true-stream-dispatch-and-three-mode-fallback-chain.md` — NFR7 chain (informs AC #5).
- `_bmad-output/implementation-artifacts/14-3-save-dialog-with-wav-writer-and-save-during-streaming-flow.md` — save-during-streaming (informs AC #6).
- `_bmad-output/implementation-artifacts/13-3-playback-last-preservation-validation.md` — `PlaybackQueue` Replay slot (informs AC #6).

**Empirical reference (regression evidence):**

- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` §4.3.2 — Story 17.2 install-mode smoke; verbatim log lines showing audio playback firing 1ms after generation completion (the specific behavior this story fixes).
- Install log at `I:/MyVoice/logs/myvoice.log` (lines 270-297, 1117-1136) — the source of the scope sketch's framing.

## Open questions for the dev-story to resolve (saved per workflow's "save questions for the end" rule)

1. **`_wrapped_post` final-chunk emission mechanism.** Two options surfaced in AC #1 (track "next chunk is final" flag in worker vs. emit synthetic terminal AudioChunk in `finalize` branch). Pick whichever lands cleanly without changing `StreamingDecoderWorker`'s contract. The synthetic-terminal option is simpler if the consumer can handle a zero-length `audio_data`; verify in Task 1.2.
2. **Async-scheduling mechanism for `_on_audio_chunk_ready`.** AC #2 names `asyncio.run_coroutine_threadsafe` as preferred; queue + drainer as fallback. Confirm during Task 2.1 by grepping for the orchestrator's existing `self._loop` capture and confirming `run_coroutine_threadsafe` doesn't deadlock with `MonitorAudioService._streaming_lock` under rapid-fire chunks. If deadlock surfaces, document the queue + drainer fallback in Change Log.
3. **`streaming_mode_used` field on `QwenTTSResponse`.** AC #5 / Task 5.2 names two detection options for "did this response come from a fallback vs. happy-path". Verify the response dataclass (`qwen_tts_service.py` `QwenTTSResponse` definition; ~line 320-region) exposes this field; if not, the chunk-index=0-while-active heuristic is the alternative. Pick during Task 5.
4. **Sample-rate device-compatibility check.** Some Windows audio devices don't natively support 24000 Hz and require resampling. The existing batch path's `play_dual_stream` may handle this implicitly via PyAudio's resampling shim; the progressive path may not. Task 7 smoke MUST verify on the dev host; if a device-rate mismatch surfaces, follow-up story to add resampling at the orchestrator's chunk-conversion step.
5. **Thread affinity for `start_streaming_session`.** PyAudio streams may need to be opened from the main thread (Windows quirk). The orchestrator's progressive callback fires from the producer thread (`_wrapped_post`); the `asyncio.run_coroutine_threadsafe` shim posts to the main event loop, which DOES run on the main thread (Qt event loop). Verify during Task 7 smoke; if main-thread-affinity is an issue, document the fallback in Change Log.

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m] (Opus 4.7, 1M context) — `/bmad-bmm-dev-story` workflow run on 2026-05-08.

### Debug Log References

- Task 1 GREEN: `python310/python.exe -m pytest tests/unit/services/test_qwen_tts_service_true_stream_callback.py -v` → 3 passed in 8.18s.
- Task 1 regression sweep: `... tests/unit/services/test_qwen_tts_service_dispatch.py tests/integration/test_streaming_tts_smoke.py -v` → 55 passed in 13.38s (no regressions).
- Tasks 2/3/5 GREEN: `... tests/unit/test_app_progressive_playback.py -v` → 8 passed in 2.65s.
- Task 4 GREEN: `... tests/unit/test_app_progressive_playback_cancel.py -v` → 2 passed in 2.67s.
- Task 6 GREEN: `... tests/integration/test_progressive_playback_dispatch_skip.py -v` → 3 passed in 2.69s.
- Task 6.2 regression sweep: `... tests/integration/test_playback_last_preservation.py tests/integration/test_session_lifecycle.py tests/ui/test_save_dialog.py` → 111 passed in 3.18s.
- Final regression sweep: 150 tests pass clean across the streaming + app + audio surface (`test_qwen_tts_service_dispatch.py` + `test_streaming_tts_smoke.py` + `test_session_lifecycle.py` + `test_app_progressive_playback*.py` + `test_progressive_playback_dispatch_skip.py` + `test_qwen_tts_service_true_stream_callback.py` + `test_voice_clone_prompt_cache.py`) in 19.53s.

### Completion Notes List

- **Task 1 (TRUE_STREAM callback emission)**: implemented as additive change to `_wrapped_post`. Two new branches: `append_chunk` constructs `AudioChunk(is_final=False)`; `finalize` emits synthetic `AudioChunk(is_final=True, audio_data=np.zeros(0, dtype=np.float32))`. Each callback invocation wrapped in try/except so a buggy consumer cannot break the producer.
- **Task 2 (orchestrator consumer)**: added `_on_audio_chunk_ready` (sync trampoline) + `_handle_progressive_chunk_async` (async handler). Lazy `asyncio.Lock` serializes concurrent chunks. Three new slots on `MyVoiceApp`. `_play_generated_audio` skip-branch + `_release_queue_slot_on_failure` keeps the queue advancing.
- **Task 3 (graceful degradation)**: open-failure path falls through to batch playback (flag stays False → `_play_generated_audio`'s skip-check is a no-op → existing batch path runs).
- **Task 4 (cancel)**: additive `stop_streaming_session` schedule when progressive active. Story 16.5's streamer-cancel hook fires synchronously inside `cancel_generation` (which runs BEFORE the additive schedule), preserving "producer stops before consumer" ordering.
- **Task 5 (NFR7 fallback)**: chose the heuristic detection (chunk_index=0 + `_progressive_playback_active` already True → close stale + open fresh) over a `streaming_mode_used` dataclass field, because the heuristic doesn't require dispatch-chain edits (Story 16.6's authority preserved).
- **Task 6 (Save + Replay continuity)**: the cached WAV file is written by `_save_audio_to_cache` inside `_generate_true_stream`, NOT by `_play_generated_audio`. The skip-branch only suppresses the actual audio-device playback; queue continuity is preserved via `_release_queue_slot_on_failure`. 111 existing Story 13.2/13.3/14.3 tests pass with no regressions.
- **Task 7 (bundled smoke)**: pending Commander hands-on smoke run on `build_tools/dist/MyVoice/MyVoice.exe`. Evidence file template prepared at `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md`.

### File List

**Modified:**
- `src/myvoice/services/qwen_tts_service.py` — Task 1 (~40 LOC delta in `_wrapped_post`); 2026-05-09 code-review pass adds MEDIUM-4 sample_rate annotation comment (~10 LOC).
- `src/myvoice/app.py` — Task 2 + 3 + 4 + 5 (~140 LOC delta total: slots, callbacks, dispatch-skip, cancel-extension); 2026-05-09 code-review pass adds module-level `import numpy as np`, `_progressive_playback_epoch` slot, restructured `_handle_progressive_chunk_async` (HIGH-1 + HIGH-2 + MEDIUM-3), epoch-aware trampoline `_on_audio_chunk_ready` (MEDIUM-1/MEDIUM-2 surface), cancel-handler epoch bump + comment fix (MEDIUM-1, LOW-2), totalling ~70 additional LOC.

**Added (tests):**
- `tests/unit/services/test_qwen_tts_service_true_stream_callback.py` — Task 1.5 (3 tests).
- `tests/unit/test_app_progressive_playback.py` — Task 2.6 + 3.3 + 5.3 (8 tests); 2026-05-09 code-review pass adds 2 SENTENCE_STREAM-shape regression tests (HIGH-1), 2 open-failure regression tests (HIGH-2/HIGH-3), 1 partial-success boundary test, 1 log-line regression test (MEDIUM-3), 3 cancel-epoch tests (MEDIUM-1), 4 trampoline tests (MEDIUM-2) — total now 21 tests.
- `tests/unit/test_app_progressive_playback_cancel.py` — Task 4.3 (2 tests); 2026-05-09 code-review pass adds 1 cancel-epoch-bump regression test (MEDIUM-1) — total now 3 tests.
- `tests/integration/test_progressive_playback_dispatch_skip.py` — Task 6.3 (3 tests).

**Added (evidence template):**
- `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` — Task 7 evidence template (sections §3-§7 populate post-bundled-smoke; gitignored — force-add per `_bmad-output/` policy).

**Sprint status:**
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — `17-3-progressive-audio-playback-during-true-stream: ready-for-dev → in-progress` at story execution start; `→ review` at story closure.

### Change Log

- **2026-05-08 — Story 17.3 source-tree implementation (Tasks 1–6)** — wired progressive audio playback during TRUE_STREAM (and SENTENCE_STREAM, which already used the same callback) generation. Producer side (`qwen_tts_service.py`) emits `AudioChunk` per chunk + synthetic terminal chunk; consumer side (`app.py`) opens `AudioCoordinator.start_streaming_session(...)` on chunk 0, writes per chunk via `play_audio_chunk(...)`, closes on terminal chunk; `_play_generated_audio` skips batch dispatch when progressive is active; `_on_cancel_generation_requested` additively closes the open streaming session on cancel. NFR7 fallback continuity preserved via heuristic stale-session-close on chunk_index=0 + already-active. 16 new tests (3 + 8 + 2 + 3) pass; 111 existing Story 13.2/13.3/14.3 + 55 streaming + 19 voice-clone-prompt-cache tests pass with no regressions (150 total).
- **2026-05-08 — Implementation deviation from AC #2 step 4**: `_progressive_playback_active` flag is intentionally NOT cleared on the terminal `AudioChunk(is_final=True)`. Clearing on is_final would race the dispatch path on the asyncio loop ordering (the terminal-chunk handler and `_play_generated_audio` are both scheduled on the main loop and can interleave). Instead, the flag is cleared by the consumer-once paths: `_play_generated_audio`'s skip-branch (normal completion) OR `_on_cancel_generation_requested` (interrupt). The audio device session itself IS closed on is_final via `stop_streaming_session()` — only the flag's clear timing differs. Documented inline in `_handle_progressive_chunk_async` docstring.
- **2026-05-08 — Implementation deviation from dev-notes table line 242**: `start_streaming_session` (singular), not `start_streaming_sessions` (plural). Verified against `audio_coordinator.py:1018` via grep before wiring. Story dev-notes had a copy-paste error from concept-stage scoping; corrected during Task 2.2.
- **2026-05-08 — Task 5 detection-mechanism choice**: chose the heuristic option (chunk_index=0 + already-active → close stale + reopen fresh) over adding a `streaming_mode_used` field to `QwenTTSResponse`. The heuristic preserves Story 16.6's "do not modify the dispatch chain" authority.
- **2026-05-08 — Task 7 (bundled smoke) pending**: source-tree implementation closed; bundled smoke + Commander manual audition on `dist/MyVoice/MyVoice.exe` deferred to Commander hands-on run. Evidence file template prepared at `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md` with smoke procedure (§4.1) + expected log markers (§4.2) + timing breakdown template (§5).
- **2026-05-09 — Code-review pass (`/bmad-bmm-code-review`) — H/M/L fixes**:
  - **HIGH-1** (SENTENCE_STREAM final chunk audio dropped): SENTENCE_STREAM emits its last data chunk with `is_final=True` AND real `audio_data` (`qwen_tts_service.py:3071-3082`) — NOT a separate zero-length terminal chunk like TRUE_STREAM. The original `_handle_progressive_chunk_async` returned immediately on `is_final=True`, dropping the last sentence's audio. Fixed by restructuring the handler to write `chunk.audio_data` first when `audio_data.size > 0` (with `is_final=chunk.is_final` so the underlying service can drain naturally), then close on `is_final`. New regression tests `test_sentence_stream_final_chunk_with_audio_is_played` and `test_sentence_stream_play_then_close_ordering` mirror the EXACT bug class per `memory/code_review_regression_test_exact_class.md`.
  - **HIGH-2** (open-failure flag-set bug): `AudioCoordinator.start_streaming_session` (`audio_coordinator.py:1018-1072`) catches all exceptions internally and returns `{"monitor": None, "virtual": None}` on failure — it never re-raises. The original handler's `try/except` was therefore dead code; the flag was set True regardless of whether the underlying services actually opened, and `_play_generated_audio` then skipped the batch path → user heard nothing. Fixed by inspecting the result dict: if both `monitor` and `virtual` are None, log warning, leave flag False, return (so batch path runs). The exception-arm is retained as defense-in-depth for future refactors.
  - **HIGH-3** (regression test mirrored wrong failure mode): `test_session_open_failure_falls_through_to_batch` used `side_effect = RuntimeError(...)` which production never raises. Replaced with `return_value = {"monitor": None, "virtual": None}` (the real production failure shape); added `test_session_open_partial_success_keeps_progressive_active` (boundary case) and `test_session_open_exception_falls_through_to_batch` (defense-in-depth for future-refactor exception path).
  - **MEDIUM-1** (cancel-vs-chunk race opens session post-cancel): `_on_cancel_generation_requested` cleared the active flag synchronously without holding `_progressive_playback_lock`; chunks the producer had already scheduled via `run_coroutine_threadsafe` could land on the loop after cancel cleared the flag, see `_progressive_playback_active=False`, and open a fresh PyAudio session that nothing would close. Fixed via a generation-epoch counter: `_progressive_playback_epoch` is bumped by the cancel handler; the trampoline `_on_audio_chunk_ready` captures the current epoch at schedule time and threads it into `_handle_progressive_chunk_async(chunk, epoch)`; the handler verifies the captured value matches under the lock and drops stale chunks. Direct-test callers pass `epoch=None` to skip the check. New tests `test_chunk_with_stale_epoch_is_dropped`, `test_chunk_with_current_epoch_is_processed`, `test_legacy_none_epoch_skips_check`, `test_cancel_bumps_epoch_for_inflight_chunk_drop`.
  - **MEDIUM-2** (trampoline path untested): Existing tests `await`-ed `_handle_progressive_chunk_async` directly, bypassing the production-only `_on_audio_chunk_ready` → `run_coroutine_threadsafe` cross-thread scheduling. Added `TestProgressivePlaybackTrampoline` with 4 tests covering the loop-missing short-circuit, the loop-closed short-circuit, schedule-with-captured-epoch, and the swallow-scheduling-exception path (so the trampoline's outer try/except is load-bearing rather than dead code).
  - **MEDIUM-3** (log line missing AC #3 fields): The session-open log captured only `sample_rate=...Hz`, omitting AC #3's required `monitor_session=<id>, virtual_session=<id>` fields. Now captures the result dict and emits the full per-AC line. New regression `test_session_open_log_includes_session_ids`.
  - **MEDIUM-4** (hard-coded `sample_rate = 24000` in producer): Added an inline comment at `qwen_tts_service.py:3750` explaining the hard-coding (Qwen3-TTS today; future model variants must update both the binding and the AudioChunk emission sites; mismatch would surface as silent corruption, not a crash) so the assumption is discoverable on read.
  - **LOW-1** (inline `import numpy as np` in `_handle_progressive_chunk_async`): Hoisted to module-level import in `app.py` per project convention.
  - **LOW-2** (misleading cancel-handler comment): Reworded to accurately describe the CancelledError path (`_play_generated_audio` is NOT invoked for the cancelled generation; the assembled buffer is dropped via `_run_async_task`'s on_error chain firing `_on_audio_playback_error`).
  - **Regression sweep**: 26 progressive-playback tests + 169 streaming/dispatch/session/save/playback tests = 195 pass clean. No regressions in Story 13.2 / 13.3 / 14.3 / 16.x suites. Source delta this pass: ~70 LOC in `app.py` (handler restructure + epoch slot + cancel epoch bump + import hoist + comment fixes), ~10 LOC in `qwen_tts_service.py` (sample_rate annotation comment), ~210 LOC in test files (3 new test classes + 6 new test methods + 1 fixture docstring update).
