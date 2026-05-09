# Scope Sketch — Story 17.3: Progressive Audio Playback During TRUE_STREAM Generation

> **Status:** scope sketch (input to `/bmad-bmm-create-story`); not yet a story file.
> **Authored:** 2026-05-08 by `/bmad-bmm-dev-story` follow-up turn after Story 17.2 closure smoke surfaced the user-perceived "audio waits for completion" behavior.
> **Purpose:** Capture the architectural gap between Phase ⊥'s streaming-dispatch infrastructure (Stories 16.3-16.6 + 17.2) and the user-facing "I hear audio progressively as it generates" promise of FR2 / NFR1. Mirrors the role `17-2-cloned-voice-truestream-prompt-precompute-scope-sketch.md` played for Story 17.2.
> **Phase tag:** Phase ⊥-Polish (post-ramp progressive-playback). Re-opens Epic 17 (currently `done`) with this third story; Epic 17 transitions to `done` again on closure of 17.3.

## Why this story exists

Story 17.2's bundled-environment smoke evidence (`17-2-cloned-voice-truestream-prompt-precompute-evidence.md` §4.3.2) confirmed TRUE_STREAM dispatch reaches CLONED-voice users with first-chunk emission at 3.93–4.94 s — well below NFR1's GPU short-class ≤5.0 s target. Commander's installer-mode smoke on `I:/MyVoice/MyVoice.exe` (2026-05-08 22:06–22:23) **also** showed every generation completing via TRUE_STREAM with the metric-side latency targets met.

But the Commander reported: *"audio is not firing during generation, it still seems to be waiting for it to finish."*

Inspection of `myvoice.log` from the install confirmed the perception:

```
22:06:48,069 - QwenTTSService - INFO - Starting TTS generation (TRUE_STREAM)
22:06:50,395 - QwenTTSService - INFO - TTS generation complete (TRUE_STREAM): 2.33s total, 2.32s first chunk
22:06:50,396 - MyVoiceApp     - INFO - Starting audio playback via AudioCoordinator   ← 1ms AFTER complete
22:06:51,834 - monitor_audio_service - INFO - Monitor audio playback completed
```

Audio playback fires **after** the TRUE_STREAM dispatch's `_generate_true_stream` returns its `QwenTTSResponse(audio_data=concatenated_buffer)`. For a 2-second utterance the lag is invisible (2.33 s gen + 1.4 s play = ~3.7 s perceived); for a 25-second utterance (line 1117–1136 in the install log: 596,769 samples, 13 chunks, 43.09 s total, 4.67 s first-chunk) the user waits a full 43 s before hearing anything, even though the streaming pipeline emitted chunk 1 internally at 4.67 s.

**Root cause (analyzed via grep):**

- `qwen_tts_service.py:4789` exposes `set_audio_chunk_ready_callback(callback)` — designed for progressive playback.
- `qwen_tts_service.py:3081-3082` (SENTENCE_STREAM) AND `qwen_tts_service.py:3897-3905` (TRUE_STREAM via `_wrapped_post('append_chunk', ...)`) emit per-chunk events.
- `app.py` orchestrator **never calls `set_audio_chunk_ready_callback`**. Grep returns zero wires across the entire `src/myvoice/` tree.
- `qwen_tts_service.py:3088-3091` (SENTENCE_STREAM) and `qwen_tts_service.py:3904 + downstream concat` (TRUE_STREAM) accumulate chunks in `accumulated_chunks: List[np.ndarray]`, concatenate at the end, return as `QwenTTSResponse.audio_data: np.ndarray`.
- `app.py:_dispatch_audio_playback` (line 2369-region) calls `AudioCoordinator.play_dual_stream(audio_data=bytes(...), ...)` with the **complete** buffer.
- `AudioCoordinator.play_dual_stream(audio_data: bytes, ...)` (`audio_coordinator.py:466`) takes a complete buffer; `MonitorAudioService.play_audio(audio_data: bytes, ...)` (`monitor_audio_service.py:261`) opens a PyAudio stream, writes the full buffer, closes — there is **no progressive `play_chunk(chunk)` API** to push chunks to an already-open PyAudio stream.

**This affects all three streaming paths.** TRUE_STREAM, SENTENCE_STREAM, and BATCH all funnel through the same end-of-generation handoff; none of them produce progressive audio playback to the user. Pre-Story 17.2, CLONED voices fell through to SENTENCE_STREAM via NFR7 — same batched-playback behavior. Story 17.2's correct routing didn't introduce this gap; it merely made it visible on long-form text where the difference between metric-time and perceived-time is dramatic.

## What Phase ⊥ promised vs. what it delivered

Architecture references:
- `architecture-optimization-pass.md:59` — "FR2 | Streaming TTS, first chunk <2s | Today met for long inputs only; this pass guarantees it for short inputs too via true streaming"
- `architecture-optimization-pass.md:823+` — NFR1 first-chunk targets (≤5.0 s p95 GPU short)
- `architecture-optimization-pass.md:836` — "first-chunk latency" measured at the streaming dispatch's chunk-emission point, NOT at the user's audio-output device

Stories 16.3 (`CodecTokenStreamer`), 16.4 (`StreamingDecoderWorker` overlap-add), 16.6 (TRUE_STREAM dispatch), 16.8 (real wire-up), 16.9 (NFR1 reconciliation), 17.1 (audition + streaming default ramp), and 17.2 (CLONED-voice routing) all delivered chunks-emitted-progressively into `accumulated_chunks`. None of them wired chunks-played-progressively to PyAudio. The audio playback path remained the V1 batch contract.

Phase ⊥-Build (Story tooling-2) verified the build pipeline. Phase ⊥-Ramp (Story 17.1 + 17.2) verified TRUE_STREAM reaches users at install time. **Phase ⊥-Polish (this story) closes the user-perceived progressive-playback gap.**

## Pre-existing infrastructure already verified before drafting

- **`qwen_tts_service.py:4789` `set_audio_chunk_ready_callback`** — TTS-side hook is already wired into both SENTENCE_STREAM and TRUE_STREAM emission points. Just needs a consumer in the orchestrator.
- **`SessionRegistry.append_chunk(session_id, audio: np.ndarray)`** (`session_registry.py:421`) — chunks already flow through here for TRUE_STREAM. Subscribers can listen via `session_state_changed` signals OR a new chunk-emitted signal added to the registry.
- **`AudioCoordinator.play_dual_stream(audio_data: bytes, ...)`** — current contract takes a complete buffer. Needs a sibling `play_dual_stream_progressive(open_handle)` API where the orchestrator pushes chunks to an open PyAudio stream as they arrive.
- **`MonitorAudioService` + `VirtualMicrophoneService`** — both wrap PyAudio streams that already support progressive `stream.write(chunk_bytes)` after `stream.start_stream()`. The progressive API exists at the PyAudio layer; just needs to be exposed up through the audio-services contract.
- **`PlaybackQueue` (Epic 13)** — handles queued static-buffer playback. Does NOT today support in-flight streaming sessions. Story 17.3 needs a queue-of-streams concept OR a "pass-through" mode where a streaming session bypasses the queue while it's actively producing chunks.
- **Sample-rate handshake** — `qwen_tts_service.py:210` notes `_tts_sample_rate` is "tracked during streaming". The first chunk carries `sample_rate=24000`; PyAudio stream needs to be opened with that rate before chunk 1 arrives. The handshake exists in concept; needs explicit wiring.
- **Cancel semantics** (Story 16.5) — `streamer._cancel_event.set()` + `audio_coordinator.cancel_playback(sid)` already in place; the progressive-playback path needs to also stop the open PyAudio stream cleanly mid-stream when cancel arrives.

## Concrete concerns surfaced by the analysis

1. **Sample-rate handshake — when is the PyAudio stream opened?** Today, `play_dual_stream(audio_data, ...)` opens the PyAudio stream right before writing. With progressive chunks, the stream must open BEFORE chunk 1 arrives so writes don't drop. Either (a) open eagerly at generation start (assumes sample rate known a priori, which it is — 24000 Hz), or (b) defer first write until chunk 1 arrives + open inside the chunk-handler (latency cost on chunk 1).

2. **Underrun on slow-generating chunks.** PyAudio stream is hungry for data. If decode takes longer than playback rate (rare on RTX 5090 but possible on slower GPUs), the stream underruns and the user hears clicks. Mitigation: pre-buffer N chunks before starting playback, OR add silence-padding when underrun detected, OR fall back to batched playback if first-chunk-to-second-chunk latency exceeds a threshold.

3. **Overlap-add boundary effects** (Story 16.4). The streaming decoder uses overlap-add windowing, so consecutive chunks share boundary samples. Concatenating chunks with simple `np.concatenate` (today's path) handles this implicitly because the overlap-add is done in the decoder before chunks land in `accumulated_chunks`. Progressive playback writes chunks individually to PyAudio; need to verify the overlap-add boundary samples are NOT duplicated across chunks (they shouldn't be, since the worker's chunk emission is post-overlap-add — but verify with audition).

4. **Cancel-mid-playback semantics.** User clicks Cancel while audio is mid-stream. Today: `cancel_playback(sid)` aborts the queued buffer. Progressive: must stop the open PyAudio stream, drain the chunk queue, AND cancel the streamer's talker thread (Story 16.5 already handles the talker-cancel). The stream-stop must be quick (≤50 ms) to feel responsive.

5. **PlaybackQueue interaction (Epic 13).** Today's PlaybackQueue queues static `(audio_data: bytes, voice: str)` tuples. Progressive sessions don't fit this shape — the audio is in-flight. Options: (a) treat the in-flight session as a "current" slot that bypasses the queue while streaming, then enqueue the final assembled buffer (for Replay), (b) extend the queue with a `StreamingSession` variant that holds a session_id + chunk-emitter handle.

6. **NFR7 fallback continuity.** When TRUE_STREAM raises mid-stream, the dispatch chain falls back to SENTENCE_STREAM. With progressive playback wired, what happens to the partially-streamed audio? Options: (a) play the partial audio + restart from beginning under SENTENCE_STREAM (jarring discontinuity), (b) abort the partial audio + only play the SENTENCE_STREAM result (loses the first ~5 s of audio that the user was about to hear). Story 17.3 should default to (b) — abort partial + start fresh — and document the tradeoff.

7. **Save-during-streaming (Story 14.3).** Saving a generation that's mid-stream needs the assembled-buffer-at-finalize path to keep working alongside progressive playback. The two paths (progressive-to-speakers vs. assembled-for-save) MUST be independent: progressive playback writes to the audio device; the WAV writer captures from the same chunk-emit feed and writes to disk. Both subscribe to the same chunk-stream.

8. **First-audio latency claim.** Today the metric is "first chunk emitted by the streaming dispatch" (~4 s on RTX 5090). After Story 17.3, the metric becomes "first audible audio at the user's speakers" — which adds PyAudio buffer fill time (typically ~50-100 ms). The architecture's "first audio <2s" promise (`architecture-optimization-pass.md:59`) was previously interpretable as "first chunk emitted"; post-17.3 it should be interpretable as "first audible to user". The numerical target should hold (4 s + ~100 ms buffer fill ≈ 4.1 s, well under 5.0 s).

9. **Progressive playback verification methodology.** AC #6 needs a way to *verify* audio plays progressively, not just that the metric reports first-chunk-latency. Options: (a) timestamp the first non-zero PCM frame at the audio device output (requires loopback recording — heavy), (b) timestamp the first `pyaudio.Stream.write()` call after chunk 1 arrives (proxy for "audio device received data") and compare to chunk-1-emit time, (c) Commander manual smoke verifying audio audibly starts mid-generation on a long sentence (the test that exposed the gap).

10. **Voice Design Studio chunk-replay path.** EMBEDDING voices via `generate_with_embedding` ALSO go through this dispatch chain — they should benefit from progressive playback identically. No special-casing needed; the chunk-emit feed is downstream of model-type forks.

## Five-point scope sketch (for the SM workflow to expand into ACs)

(a) **Wire `set_audio_chunk_ready_callback` consumer in the orchestrator.** App.py wires a chunk handler that pushes each `AudioChunk(audio_data, sample_rate, chunk_index, is_final, text_segment)` to a new `AudioCoordinator.play_chunk_progressive(coordination_id, chunk)` API. The orchestrator opens the coordination at generation start (`AudioCoordinator.begin_progressive_session(sample_rate=24000, session_id=sid, ...)`) and closes it on `is_final=True`. Closes concerns 1, 9.

(b) **Add `AudioCoordinator.begin_progressive_session(...) → progressive_handle` + `play_chunk_progressive(handle, chunk)` + `end_progressive_session(handle, final_buffer=None)` API.** Internally opens PyAudio streams on both Monitor + Virtual services BEFORE chunk 1 arrives (sample rate known a priori = 24000 Hz). `play_chunk_progressive` writes chunk bytes to both open streams; underrun-tolerant. `end_progressive_session` waits for stream drain + closes the streams + optionally stores the assembled `final_buffer` for Save / Replay. Closes concerns 1, 2, 7.

(c) **Update Cancel + NFR7 chains for progressive playback.** Cancel mid-stream: stop both PyAudio streams (`stream.stop_stream()` + `stream.close()`), abort the streamer talker thread (Story 16.5 hook already in place), discard accumulated chunks. NFR7 fallback mid-stream: abort the partial progressive playback (variant b per concern 6); SENTENCE_STREAM restart begins a fresh progressive session. Closes concerns 4, 6.

(d) **PlaybackQueue (Epic 13) interaction.** During an in-flight progressive session, the queue is in "pass-through" mode — the streaming session occupies the focal slot but the queue's `enqueue/dequeue` semantics still work for backlogged static-buffer plays after the streaming session finalizes. The assembled `final_buffer` is also enqueued at session finalize for Replay (Story 13.3 last-preservation). Closes concern 5.

(e) **Smoke-verification + bundled audition.** Re-run Story tooling-2's portable + installer smoke on a fresh bundle. Generate a 25-character utterance + a 250-character utterance on Sarira-F (CLONED voice, post-17.2 cache hit). Verify: (i) myvoice.log shows the first `MonitorAudioService.write()` call within ~100 ms of the first `append_chunk` post; (ii) Commander manual audition confirms audio starts audibly during generation, not after; (iii) no underrun audible artifacts on either short or long utterance; (iv) NFR1 first-chunk latency metric stays ≤5.0 s p95 (the metric definition is unchanged). Capture in `_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md`. Closes concern 9.

## What this story is NOT

- **Not a TRUE_STREAM dispatch rework.** Stories 16.3-16.6 + 17.1 + 17.2 already deliver the talker-decoder-streamer-overlap-add pipeline correctly. Story 17.3 only adds the progressive-output stage on top.

- **Not a perceptual quality re-litigation.** Story 17.1's audition certified TRUE_STREAM perceptual equivalence to BATCH; the overlap-add already handles seam quality. Progressive playback writes the SAME chunks the audition validated, just earlier. Audition rerun unnecessary unless concern 3 surfaces a boundary-sample regression.

- **Not a model-tier or model-type optimization.** All three model types (CUSTOM_VOICE, VOICE_DESIGN, BASE) flow through the same chunk-emit feed. No model-type forking in the new progressive path.

- **Not a Voice Design Studio change.** EMBEDDING voices use `generate_with_embedding` which ALSO routes through the streaming dispatch — they get progressive playback for free. No VDS UI changes.

- **Not a build-pipeline change.** Story tooling-2 closed Phase ⊥-Build; the production bundle ships the right runtime. Story 17.3 is source-tree-only edits picked up by the next `build_release.bat` run.

- **Not a re-run of the production release.** After 17.3 closes, the next build pipeline run produces a new installer with progressive playback; that build's release decision is a separate Commander decision.

## References

**Source tree (read + likely-edit candidates):**

- `src/myvoice/services/qwen_tts_service.py:3081-3082` — SENTENCE_STREAM chunk emission (read-only; demonstrates the existing pattern)
- `src/myvoice/services/qwen_tts_service.py:3897-3905` — TRUE_STREAM `_wrapped_post('append_chunk', ...)` (read-only; the chunk-emit point for the progressive feed)
- `src/myvoice/services/qwen_tts_service.py:4789-4796` — `set_audio_chunk_ready_callback` (read-only; the consumer hook to wire)
- `src/myvoice/services/audio_coordinator.py:466` — `play_dual_stream` (current batch API; sibling progressive API to add)
- `src/myvoice/services/monitor_audio_service.py:261` — Monitor audio service `play_audio` (today's batch playback site; will gain progressive `write_chunk` siblings)
- `src/myvoice/services/virtual_microphone_service.py:887-region` — Virtual mic service play surface (parallel to monitor)
- `src/myvoice/services/sessions/session_registry.py:421` — `SessionRegistry.append_chunk` (read-only; the registry-side chunk-emit handler)
- `src/myvoice/app.py:2369-region` — `_dispatch_audio_playback` (the orchestrator-side site that needs the chunk-handler wiring)
- `src/myvoice/services/playback_queue/playback_queue.py` (or wherever the PlaybackQueue lives) — Epic 13's queue (touched for concern 5 / scope (d))

**Architecture references:**

- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:59` — FR2 streaming-TTS first-chunk <2s claim
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:823+` — NFR1 first-chunk per-class targets
- `_bmad-output/planning-artifacts/architecture-optimization-pass.md:836` — "first-chunk latency" measurement methodology

**Memory:**

- `memory/build_tools_phase_perp_state.md` — Phase ⊥-Build closure marker (this story is the Phase ⊥-Polish follow-up)
- `memory/epic16_streaming_blocked.md` — Phase ⊥ closure marker (historical pointer; this story extends Phase ⊥)
- `memory/hardware_setup.md` — RTX 5090 dev host (informs concern 2 underrun severity)

**Precedent stories:**

- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md` — Story 17.2 (the predecessor; routed CLONED voices through TRUE_STREAM dispatch — Story 17.3 builds on this by routing those chunks through PyAudio progressively)
- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` §4.3.2 — confirms TRUE_STREAM dispatch first-chunk emission at 3.95 s; the gap is from there to user's audio device
- `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` — TRUE_STREAM perceptual certification (audition; the chunks Story 17.3 plays progressively are the same chunks the audition validated)
- `_bmad-output/implementation-artifacts/16-3-codectokenstreamer-with-bounded-queue.md` — chunk-emit infrastructure
- `_bmad-output/implementation-artifacts/16-4-streaming-decoder-worker-with-overlap-add.md` — chunk overlap-add (informs concern 3)
- `_bmad-output/implementation-artifacts/16-5-cooperative-cancellation-chain.md` — cancel chain (informs concern 4)
- `_bmad-output/implementation-artifacts/16-6-true-stream-dispatch-and-three-mode-fallback-chain.md` — NFR7 chain (informs concern 6)
- `_bmad-output/implementation-artifacts/14-3-save-dialog-with-wav-writer-and-save-during-streaming-flow.md` — save-during-streaming (informs concern 7)

**Empirical reference (regression evidence):**

- `_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute-evidence.md` §4.3.2 — Story 17.2 install-mode smoke; verbatim log lines showing audio playback firing 1ms after generation completion (the specific behavior 17.3 fixes)
- Install log at `I:/MyVoice/logs/myvoice.log` (lines 270-297, 1117-1136) — the source of this scope sketch's framing

## Suggested story-file naming

`_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream.md`

## Suggested Phase tag

`Phase ⊥-Polish` — Story 17.1 closed the audition (Phase ⊥-Ramp's certification dimension); Story 17.2 closed the user-reach (Phase ⊥-Ramp's user-facing dimension); Story 17.3 closes the user-experience (Phase ⊥-Polish's progressive-playback dimension). Once 17.3 lands, Phase ⊥ is genuinely complete in the user-facing sense.

## Suggested sprint-status edit (for SM workflow when creating the story)

```yaml
# Epic 17 — Streaming Default Ramp (Phase ⊥-Ramp + Phase ⊥-Polish)
# Re-opened 2026-05-08 (third iteration) per Story 17.2 closure smoke
# revealing user-perceived "audio plays after completion" gap on long-form
# text. Story 17.3 wires progressive chunk playback during TRUE_STREAM
# generation so audio is audible at first-chunk-emit time, not at
# generation finalize.
epic-17: in-progress
17-1-streaming-default-ramp: done
17-2-cloned-voice-truestream-prompt-precompute: done
17-3-progressive-audio-playback-during-true-stream: ready-for-dev
epic-17-retrospective: optional
```
