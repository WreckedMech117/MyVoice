---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: complete
completedAt: '2026-04-27'
inputDocuments:
  - _bmad-output/optimization-pass/01-streaming-tts-research.md
  - _bmad-output/optimization-pass/02-audio-buffer-lifecycle.md
  - _bmad-output/optimization-pass/03-feature-pack-buffer-lifecycle.md
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/ux-design-specification.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/planning-artifacts/architecture-voicedesign-embeddings.md
  - docs/QWEN3_TTS_INTEGRATION.md
  - docs/dual_service_audio_architecture_design.md
workflowType: 'architecture'
scope: 'optimization-pass'
projectContext: brownfield
parentArchitecture: _bmad-output/planning-artifacts/architecture.md
relatedScopes:
  - true_streaming_tts
  - audio_buffer_lifecycle
  - playback_queue
  - save_generation
  - clear_comms_button
  - status_indicator_truth
project_name: 'MyVoice V2 (Optimization Pass)'
user_name: 'Commander'
date: '2026-04-27'
---

# Architecture Decision Document — MyVoice V2 Optimization Pass

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

**Scope:** Optimization-pass enhancements to the sealed V2 architecture. Covers true-streaming TTS, the `GenerationSession` abstraction, playback queueing, and the four user-facing features (Save, Clear Comms, Consecutive Playback, Indicator State Truth). Does **not** revisit V2 baseline decisions (model selection, dual-service audio, PyQt6, voice library, etc.) — those remain governed by `architecture.md` (sealed 2026-01-31).

**Relationship to parent architecture:** This document refines and extends, never contradicts, the V2 architecture. Where V2 decisions are silent on a question (e.g., per-utterance state ownership, GPU stream concurrency), this document fills the gap. Where they spoke, this document inherits.

## Project Context Analysis

### Scope of this Optimization Pass

This pass addresses four cohesive concerns surfaced after the V2 baseline shipped:

1. **True streaming TTS** — replace today's sentence-level pseudo-streaming with token-level streaming via the `qwen_tts` library's HuggingFace `.generate()` hook and chunked codec decode (per `01-streaming-tts-research.md`).
2. **`GenerationSession` foundation** — introduce a per-utterance state object that unifies today's split-brain coordination between `QwenTTSService.GenerationState` and `AudioCoordinator.active_playback_tasks` (per `02-audio-buffer-lifecycle.md`).
3. **Four user-facing consumers** — Save current generation, Clear Comms button, consecutive playback (no overlap), and indicator state truth (per `03-feature-pack-buffer-lifecycle.md`).
4. **Cross-cutting policy decisions** — threading model, GPU stream concurrency, library-pin discipline, lifecycle/memory policy.

The pass is brownfield. It extends, never contradicts, the sealed V2 architecture (`architecture.md`, completed 2026-01-31).

### Inherited Requirements (from V2 PRD)

**Functional touchpoints this pass must continue to satisfy:**

| FR | Requirement | How this pass relates |
|---|---|---|
| FR2 | Streaming TTS, first chunk <2s | Today met for long inputs only; this pass guarantees it for short inputs too via true streaming |
| FR3 | Batch fallback if streaming unavailable | Becomes one of three lifecycle paths (`BATCH`, `SENTENCE_STREAM`, `TRUE_STREAM`) on the session |
| FR4 | User can cancel in-progress generation | Session lifecycle adds `CANCELLED` transition that must propagate cleanly through streamer + decoder |
| FR28–FR32 | Playback Last (cache file replay) | **Reconcile, don't replace.** `%TEMP%/myvoice_last.wav` is a serialized projection of the prior session's `complete_audio`. The new abstraction makes today's implicit cache explicit. Lifecycle policy must preserve "prior session is replayable" semantics across supersession (decision deferred to Step 4) |
| FR42 | Display current TTS service status | Directly addressed by feature #4 (Indicator State Truth) |

**Non-Functional Requirements that drive decisions in this pass:**

| NFR | Requirement | Architectural implication |
|---|---|---|
| NFR1 | First audio <2s | True streaming has a higher floor than estimated (~1.5–1.8s on GPU; potentially >2s on CPU-only). Comfortably meets NFR1 on GPU; **conditional on hardware** for CPU. Step 7 validation gate |
| NFR3 | No audio stuttering | Overlap-add seam quality is the *only* new stutter risk introduced by this pass — drives mandatory A/B perceptual testing before flipping the streaming default |
| NFR4 | UI responsiveness <200ms | Drives threading model: session mutations cannot block the Qt UI thread |
| NFR6 | No crashes | Reaching past the `qwen_tts` public API introduces a brittleness vector — drives version-pin + import-attribute test policy |
| NFR7 | Graceful degradation | Drives the three-mode session-population design (`BATCH`, `SENTENCE_STREAM`, `TRUE_STREAM`) — failure in one mode falls back to the previous |
| NFR11 | <4GB RAM with model | Session buffer policy (single "saveable" + immediate `chunks.clear()` after concat) is a memory discipline, not just a feature |
| NFR12 | CPU-only support | Streaming-default-flag must be hardware-aware: enabled on GPU, conservative on CPU |

### Local FR-equivalents (from doc 03)

The optimization pass introduces user-facing behavior that hasn't yet been formalized as PRD FRs but functions as such:

- **OFR-A** Save current generation to user-chosen file path (WAV)
- **OFR-B** Clear Comms button + Settings UI for source selection (file or "use last generation")
- **OFR-C** Consecutive playback — generations play in submission order, never overlap (bug fix)
- **OFR-D** Status indicator reflects single source-of-truth session state (bug fix)

These should be back-propagated into the PRD as a follow-up administrative task (see Out-of-Scope). Treated as scope-bound requirements for this architecture document.

### Scale & Complexity

| Indicator | Assessment |
|---|---|
| New / refactored components | ~6 (`GenerationSession`, `SessionRegistry`, `PlaybackQueue`, `CodecTokenStreamer`, `StreamingDecoder`, refactored `QwenTTSService` mediation surface) |
| New external dependencies | Likely zero (existing numpy/scipy/wave/soundfile already in-tree; verify in Step 4) |
| Cross-service touchpoints | TTS service → Session, Session → Audio coordinator, Session → UI signals, Settings → Preloaded sessions |
| Concurrency surface | Generation thread + audio playback callback + Qt UI thread + (new) HF streamer callback context + (potentially new) dedicated CUDA stream |
| Risk profile | **Medium-high.** The streaming POC carries audio-quality risk (seams) and library-fragility risk (private API). The session refactor is mechanically straightforward but requires disciplined migration to avoid leaving two state systems coexisting |

**Primary domain:** Real-time interactive desktop audio pipeline (Python/PyQt6/PyTorch).
**Complexity level:** Medium-high — the abstraction work itself is moderate; the streaming POC carries the bulk of the risk; the four user features are thin layers if the foundation is right.

### Technical Constraints & Dependencies

**Inherited from V2 arch (non-negotiable for this pass):**
- PyQt6 signal-based inter-component communication
- snake_case naming for code, `{property}_changed` / `{event}_past_tense` for signals
- Dual-service audio architecture (`MonitorAudioService` + `VirtualMicrophoneService` + `AudioCoordinator`)
- Lazy single-model loading discipline
- Existing signals (`audio_chunk_ready`, `generation_started`, `generation_complete`, `playback_complete`) — must remain wire-compatible or be migrated explicitly with deprecation note

**New constraints introduced by this pass:**
- `qwen_tts` library version must be pinned with a CI test that imports the exact internal symbols (`model.model.generate`, `speech_tokenizer.decode`) and fails on rename
- True-streaming code path must be feature-flagged so a streaming regression cannot brick the app — fallback to `SENTENCE_STREAM` then `BATCH` must be automatic
- Audio buffer lifetime policy (when buffers free) must be deterministic enough that QA can verify in tests, not just in interactive use

### Cross-Cutting Concerns

These concerns span multiple components and must be settled in Step 4 (Decisions) before per-feature tech-specs are written:

1. **Threading & ownership model** — who owns session mutation, how state changes reach the Qt UI thread, how generation/playback/UI threads interact. Pure application architecture concern; affects every component below.
2. **GPU stream concurrency** — whether talker `.generate()` and `speech_tokenizer.decode()` share a CUDA stream, and what serialization that implies for streaming latency. Affects only the streaming POC, but is a blocker for it.
3. **Session lifecycle & memory policy** — saveable-session disambiguation rule, PRELOADED-clone exclusion from supersession, immediate `chunks.clear()` after concat, `clone_for_replay()` semantics, **Playback Last preservation across supersession**. Affects Save, Clear Comms, Queue, and Indicator features.
4. **Signal contracts** — what new signals are added, what existing signals are deprecated, how `session_state_changed` composes with the existing `generation_state_changed` and `audio_chunk_ready`.
5. **Library-pin & private-API discipline** — how we guard against upstream `qwen_tts` refactors silently breaking us. Pure operational concern but worth one explicit decision.
6. **Telemetry** — what we measure (per-chunk decode latency, queue depth, session-state-duration histogram) and where it lands (existing `_avg_first_chunk_latency` infrastructure or new).
7. **Backward compatibility & migration** — how we ship the foundation (`02`) without forcing an immediate rewrite of every UI subscriber. The foundation should be net-zero behavior change at first; the four features then sit on top.
8. **Queue gating granularity (invariant)** — playback queue must serialize at *session* level, not at the audio-task level. `AudioManager.active_playback_tasks` is per-device by design; each dequeued session still fans out to monitor + virtual microphone in parallel. Capture as architectural invariant, not just queue policy.
9. **NFR1 hardware-conditional satisfaction** — true-streaming first-audio latency is comfortably under NFR1's 2s on GPU but may exceed it on CPU-only hardware (PRD NFR12). Streaming-default-flag must be hardware-aware: enabled on GPU, conservative on CPU. Step 7 validation gate.

### What is explicitly *out of scope* for this pass

- Model quantization (int8/fp8) — separate optimization vector, deferred
- vLLM-Omni online serving — wait-and-see; revisit if/when upstream lands
- Multi-history save (saving generations older than the most recent) — deferred per `03 §2.2`
- Soundboard / multi-button Clear Comms — deferred per `03 §3.2`
- **PRD back-propagation of OFR-A through OFR-D** — administrative follow-up, **not blocking this architecture pass**. Owner: PM/Commander. Tracked here so it doesn't go cold.

**Note:** The visible queue-depth indicator/badge, originally deferred in `03 §5.4`, has been **promoted into scope** for this pass on UX-trust grounds (party-mode discussion, 2026-04-27). The signal contract is specified in Step 5; the widget treatment is delegated to the per-feature tech-spec / UX review.

## Starter Template Evaluation

**Not applicable — brownfield project.** The existing MyVoice V2 codebase is the foundation. No greenfield starter selection is needed; this section instead documents the baseline being inherited and the gaps the optimization pass will fill.

### Inherited Codebase Baseline

**Project type:** Python desktop application (Windows-first), already shipped at V2 baseline.

**Existing services (in `src/myvoice/services/`):**

| Service | Role | Touched by this pass? |
|---|---|---|
| `qwen_tts_service.py` | TTS generation, sentence-pseudo-streaming, model lifecycle | Yes — adds `TRUE_STREAM` mode, mutates `GenerationSession` instead of singleton state |
| `audio_coordinator.py` | Orchestrates dual playback (monitor + virtual mic), dispatches chunks | Yes — gains `PlaybackQueue` integration, session-level dispatch gating |
| `audio_service.py` | `AudioManager` with `active_playback_tasks` | Yes — task collection becomes per-session-fanout, not free-form |
| `monitor_audio_service.py` | Monitor speaker output | No — invariant |
| `virtual_microphone_service.py` | Virtual mic output | No — invariant |
| `voice_profile_service.py` | Voice library, profiles | No — invariant |
| `model_loading_manager.py`, `model_registry.py` | Lazy model loading (V2 pattern) | No — invariant |
| `whisper_service.py`, `transcription_*` | Speech-to-text features | No — out of scope |
| `quick_speak_service.py` | Quick Speak presets | No — invariant; but Clear Comms shares some UX vocabulary |

**Existing UI components (in `src/myvoice/ui/components/`):**

| Component | Role | Touched by this pass? |
|---|---|---|
| `service_status_indicator.py` | Status bar at bottom of main window | Yes — rewired to a single `session_state_changed` source (OFR-D) |
| `settings_dialog.py` | Tabbed settings | Yes — gains "Clear Comms" section (OFR-B) |
| `voice_selector.py`, `voice_library_widget.py` | Voice management | No — invariant |
| `emotion_button_group.py` | Emotion preset row | No — invariant |
| `quick_speak_dialog.py`, `quick_speak_menu.py` | Quick Speak UI | No — invariant |

**Inherited tech stack (from `requirements.txt`):**

| Layer | Technology | Version policy |
|---|---|---|
| GUI | PyQt6 ≥ 6.6.0 | Inherited from V2; no change |
| Async glue | qasync ≥ 0.27.0 | PyQt6 + asyncio bridge — relevant to threading model decision in Step 4 |
| TTS | `qwen-tts` from `git+https://github.com/QwenLM/Qwen3-TTS.git` | **Currently unpinned (no commit / tag).** Architectural recommendation in Step 4: pin to a specific commit |
| ML runtime | PyTorch ≥ 2.0 | CPU and CUDA variants both supported; relevant to GPU-stream concurrency decision |
| Audio I/O (high-level) | `soundfile` ≥ 0.12.1 | Used by qwen-tts internally; available for our WAV save path (likely choice for OFR-A) |
| Audio I/O (device) | PyAudioWPatch ≥ 0.2.12.6 (Windows) | Already in use; no change |
| Numerics | numpy ≥ 1.24 | Already in use; no change |

### What the Baseline Already Provides

- **PyQt6 signal infrastructure** — every cross-component event already flows through Qt signals. The `session_state_changed` emitter slots into this pattern naturally.
- **Asyncio integration via qasync** — long-running generation already runs without blocking the UI thread. The threading model decision (Step 4) builds on this rather than replacing it.
- **`GenerationState` enum on the TTS service** — the *state vocabulary* is already established (`IDLE`, `LOADING_MODEL`, `GENERATING`, `STREAMING`, `COMPLETE`, `CANCELLED`, `ERROR`). The session refactor inherits and enriches this; it is not invented from scratch.
- **`audio_chunk_ready` signal** — already plumbed through to the audio coordinator. The streaming POC in `01` becomes "emit smaller chunks more often through the same channel," not "build a new channel."
- **`_avg_first_chunk_latency` telemetry** (`qwen_tts_service.py:447`) — baseline measurement infrastructure exists. Step 7 validation just adds counters, not a new pipeline.
- **`%TEMP%/myvoice_last.wav` cache (FR31/FR32)** — Playback Last already implements per-utterance buffer persistence as a side-effect file write. This pass reframes it as a serialized projection of `session.complete_audio` rather than a separate mechanism.

### What the Baseline is Missing

These are the gaps this pass fills. They become the components designed in Step 6:

1. **Per-utterance state object (`GenerationSession`)** — today's state is a service singleton. No place to attach lifecycle-aware behaviors.
2. **Session ownership / registry** — no neutral home for "the current session" reference; UI subscribes to disjoint sources (root cause of OFR-D).
3. **Playback queue** — `audio_service.py:130` `active_playback_tasks: Dict` is per-device by design, with no cross-session serialization (root cause of OFR-C).
4. **Token-level streamer** — no `BaseStreamer` subclass; today's streaming is sentence-batched.
5. **Chunked codec decoder** — no overlap-add seam handling; today's decoder runs once per sentence on full token sequence.
6. **Save-to-file pipeline** — no service writes generated audio to user-chosen paths (OFR-A is greenfield).
7. **PRELOADED-source audio loader** — no service decodes user-uploaded files into a session-compatible numpy array (OFR-B prerequisite).
8. **Hardware-aware streaming default** — no mechanism to enable streaming on GPU and disable on CPU at runtime.

### Initialization Command

**Not applicable** — no project bootstrap; the codebase already exists. The architectural equivalent is captured as a recommended **dependency-pin update** (Step 4 decision):

```diff
- qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git
+ qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@<pinned-commit-hash>
```

Plus a CI-time import-and-attribute test that fails on upstream rename of `model.model.generate` or `speech_tokenizer.decode`.

### Web Search Note

The Step 3 protocol normally calls for web-search version verification of starter templates. That doesn't apply here, but two version questions *do* deserve verification before Step 4 commits to specific decisions:

- **Current `qwen_tts` upstream HEAD vs. last-known-good commit** — verify before pinning.
- **vLLM-Omni online-serving status** — out of scope per Step 2, but worth a periodic check; if it lands, the streaming POC could be obviated.

These are tracked as research-tickets, not decisions in this document.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical decisions (block implementation):** D-1, D-2, D-3, D-4, D-5, D-8, D-9, D-12, D-13, D-14, D-20

**Important decisions (shape architecture):** D-6, D-7, D-10, D-11, D-15, D-16, D-17, D-19

**Per-feature decisions (limit blast radius to one feature):** D-18

### Cluster A — Session Ownership & Threading

**D-1 Session ownership home.** A new `SessionRegistry` singleton owns the current-session reference, the playback queue, and the saveable slot. Neither `QwenTTSService` nor `AudioCoordinator` is the owner; both mutate sessions through the registry's API. **Rationale:** sessions cross both subsystems; coupling either as owner is wrong; UI rewires to one emitter.

**D-2 Threading discipline.** `SessionRegistry` is owned by the Qt main thread. State mutations from generation/playback threads are posted via `QMetaObject.invokeMethod` (or Qt signals). All `session_state_changed` emissions originate on the Qt thread — UI subscribers can read state synchronously. **Rationale:** state mutations are bursty and fast; matches existing PyQt6 idiom; satisfies NFR4 because mutation work is bounded to state transitions, not waveform I/O.

### Cluster B — Lifecycle & Memory Policy

**D-3 Saveable disambiguation.** "Current saveable session" = the session whose `finalize()` fires most recently in wall-clock time, regardless of subsequent `PLAYING` transitions on other sessions. **Rationale:** matches user mental model — "the thing I just heard finishing is what's saveable."

**D-4 Playback Last preservation.** The prior `GENERATED` session lingers in a "saveable-but-not-active" slot until the *next* `GENERATED` session reaches `DONE`. Only then is the prior discarded. The `%TEMP%/myvoice_last.wav` cache becomes a *write-through projection* of this slot — written on `finalize()`, read only as a crash-recovery path. **Rationale:** preserves FR28–FR32 semantics natively in the new abstraction; collapses two parallel mechanisms into one. **Note:** revises `02 §3.5(2)` — supersession of saveable triggers on next `DONE`, not next `PLAYING`.

**D-5 PRELOADED-clone exclusion.** Only sessions with `source == GENERATED` participate in saveable supersession. PRELOADED clones (Clear Comms playbacks) never invalidate the saveable slot. **Rationale:** prevents Clear Comms from silently destroying the user's last real generation.

**D-6 `clone_for_replay()` semantics.** Clone shares the original's `complete_audio` numpy array by reference; immutability is enforced by convention (sessions never mutate `complete_audio` after `finalize()`). New `session_id`, new state-machine starting at `READY_TO_PLAY`. **Rationale:** zero-copy; sessions don't mutate post-finalize, so no race; refcount machinery avoided.

**D-7 Memory hygiene.** `chunks.clear()` is called immediately after `np.concatenate(chunks)` populates `complete_audio` in `finalize()`. Eliminates the transient 2× memory peak.

### Cluster C — Streaming TTS

**D-8 GPU stream concurrency.** Initial implementation uses the default CUDA stream; decoder runs in the streamer's `put()` callback context, serialized with talker generation. A dedicated `torch.cuda.Stream` for the decoder is tracked as a measured optimization, to be applied only if profiling shows decode is the bottleneck. **Rationale:** simplest correct path; talker yields the GPU naturally between iterations; over-engineering is the wrong start.

**D-9 Hardware-aware streaming default.** At startup, probe `torch.cuda.is_available()`. If true, default `streaming_mode = TRUE_STREAM`. If false, default `streaming_mode = SENTENCE_STREAM`. User can override in settings. **Rationale:** satisfies NFR1 on GPU; protects CPU-only users from latency regression (NFR12).

**D-10 Backpressure.** `CodecTokenStreamer` uses a bounded `queue.Queue(maxsize=N)` between token producer (HF `.generate()`) and the decode worker. Default `N = 4 × chunk_size_in_tokens`. When full, the streamer's `put()` blocks; HF's `.generate()` yields naturally. **Rationale:** bounded memory; correct backpressure; HF tolerates streamer slowness.

**D-11 Cancellation.** Cooperative via `threading.Event`. The streamer's `put()` and the decoder loop both check the event; when set, they early-return / drain. HF's `.generate()` keeps running for a few iterations until streamer becomes no-op, then completes; results discarded. **Rationale:** no exceptions raised through HF internals; CUDA state stays clean; small wasted compute is acceptable.

**D-12 `qwen-tts` pin policy.** Pin the dependency to the currently-installed commit hash in `requirements.txt`. Add `tests/test_qwen_tts_internals.py` that imports the exact private symbols (`model.model.generate`, `speech_tokenizer.decode`) and asserts they exist; failing this test in CI blocks the build. **Rationale:** brittle integration is acceptable if loud — silent breakage is not. Tagged-release pin preferred if upstream offers one (verify before merge).

### Cluster D — Signal Contracts & Feature Decisions

**D-13 New SessionRegistry signals.**

| Signal | Payload | Purpose |
|---|---|---|
| `session_state_changed` | `(session_id: str, new_state: SessionState)` | Fires on every state transition |
| `current_session_changed` | `(session_id: Optional[str])` | Fires when the indicator's focal session changes (queue advances, etc.) |
| `playback_queue_depth_changed` | `(depth: int)` | Drives the indicator badge |
| `saveable_session_changed` | `(session_id: Optional[str])` | Drives Save button enablement |

**D-14 Existing signals.** Kept wire-compatible during this pass. `generation_state_changed`, `generation_started`, `generation_complete`, `audio_chunk_ready`, `playback_complete` continue to fire identically. `generation_state_changed` is marked deprecated in code comment with a pointer to `session_state_changed`. Hard removal deferred to a future pass after subscribers migrate.

**D-15 `is_audible` substate.** `GenerationSession` exposes `is_audible: bool`, set true once the first sample is written to the output device. Indicator and queue-depth widget read this directly. **Rationale:** keeps presentation logic out of the indicator's state-mapping table.

**D-16 Save format (OFR-A).** WAV via `soundfile.write()`. Float32 → int16: `(audio * 32767).clip(-32768, 32767).astype(np.int16)`. Sample rate inherited from `session.sample_rate` (model-native, typically 24 kHz). Mono. Default filename: `myvoice-<voice>-<yyyy-mm-dd-hhmmss>.wav` (sanitized).

**D-17 Audio loader for Clear Comms (OFR-B).** `soundfile.read()`, WAV-only in v1. MP3/M4A support deferred. **Rationale:** the "use last generation" path is the primary case; uploaded files are secondary; format expansion adds dependency surface for limited initial value.

**D-18 Clear Comms default.** Interrupt by default — pressing Clear Comms calls `playback_queue.cancel_current()` before enqueueing the PRELOADED clone. Settings exposes a `clear_comms_queue: bool` toggle for users who prefer queue behavior. **Rationale:** "Clear Comms" is mentally a call-out interrupt; queue would defeat the purpose.

**D-19 Telemetry.** All four metrics extend the existing `_avg_first_chunk_latency` infrastructure:
- per-chunk decode latency (histogram)
- playback queue depth (gauge / time-series)
- session-state-duration per state (per-transition timer)
- streaming mode in use (counter, BATCH/SENTENCE_STREAM/TRUE_STREAM)

**D-20 Migration plan (phased).**

| Phase | Deliverable | Reverts cleanly? |
|---|---|---|
| 1 — Foundation | `GenerationSession`, `SessionRegistry`, `session_state_changed` signal. Existing UI unchanged | Yes |
| 2 — Indicator (OFR-D) | Rewire `service_status_indicator.py` to single signal | Yes |
| 3 — Queue (OFR-C) | `PlaybackQueue` interposes between session DONE and `AudioCoordinator` dispatch | Yes |
| 4 — Save (OFR-A) | Save button, WAV writer, save dialog | Yes |
| 5 — Clear Comms (OFR-B) | Settings UI, PRELOADED-source loader, main-window button | Yes |
| ⊥ — Streaming (parallel) | `TRUE_STREAM` mode behind hardware-aware feature flag | Yes |

Phases 1–5 must land in order. Phase ⊥ (streaming) is independent and can land before, between, or after any phase ≥ 1.

### Cross-Component Dependencies

- **OFR-D depends on Phase 1** (signals must exist).
- **OFR-A depends on D-3, D-4, D-15** (saveable slot, Playback Last preservation, button enablement).
- **OFR-B depends on OFR-A's WAV writer** (snapshot-as-Clear-Comms reuses the file path) and on **D-5 + D-18** (lifecycle + interrupt default).
- **OFR-C depends on D-2** (Qt-thread dispatch) and **D-13** (queue-depth signal).
- **TRUE_STREAM depends on D-8, D-9, D-10, D-11, D-12** (none of the others).
- **Existing FR28–FR32 (Playback Last) depends on D-4** — must not regress in Phase 1 or 3.

### Decisions explicitly *not* made here (delegated to per-feature tech-spec)

- Save button placement in main window — UX review territory
- Indicator widget visual treatment — UX review territory
- Queue-depth badge visual treatment — UX review territory
- Settings dialog layout for Clear Comms — UX review territory
- Default chunk size and overlap parameters for streaming — measured experimentally before tech-spec lock
- WAV sidecar metadata (text, voice, model) — deferred until requested

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Critical conflict points identified in this pass:** 9 patterns (P-1 through P-9). V2 baseline conventions (snake_case naming, `{property}_changed` signals, PyQt6 idiom, structured logging) are inherited unchanged from `architecture.md` and not restated here.

### P-1 — Session API: state-bound method validity

`GenerationSession` exposes methods that are only valid in specific states. Calling a method outside its valid states **must** raise `InvalidSessionStateError(current_state, method, expected_states)`. No silent no-ops.

| Method | Valid states |
|---|---|
| `append_chunk(audio)` | `GENERATING` |
| `finalize()` | `GENERATING` (transitions to `READY_TO_PLAY`) |
| `mark_playing()` | `READY_TO_PLAY` |
| `mark_audible()` | `PLAYING` (sets `is_audible = True`) |
| `mark_done()` | `PLAYING` |
| `cancel()` | any except `DONE`, `DISCARDED` |
| `discard()` | `DONE`, `CANCELLED`, `ERROR` (transitions to `DISCARDED`; frees buffers) |
| `clone_for_replay()` | requires `complete_audio is not None` |

**Anti-pattern:**

```python
if session.state == SessionState.GENERATING:
    session.finalize()
# silently skipped if state is wrong — bug hides
```

**Pattern:**

```python
session.finalize()  # raises if state is wrong; caller surfaces or handles
```

### P-2 — State transitions go through a single helper

All state transitions on `GenerationSession` happen via `_transition_to(new_state)`. This helper:

1. Validates the transition is permitted (uses a static `_VALID_TRANSITIONS` map).
2. Sets `self.state = new_state`.
3. Emits the `session_state_changed` signal *via the registry* (never directly from the session).
4. Records timestamp in `self._state_durations` for telemetry (D-19).

No code outside the session class assigns to `self.state` directly. No code emits `session_state_changed` directly.

**Rationale:** without this discipline, two agents writing different code paths can independently set `state = X` and forget to emit, producing UI desync bugs that are extremely hard to reproduce.

### P-3 — Signal emission originates from the Qt main thread

Per D-2: the `SessionRegistry` lives on the Qt main thread; all signals must emit from that thread.

**From the Qt main thread** (UI handlers, slot callbacks): emit directly.

**From any worker thread** (generation thread, decoder thread, audio callback): post the mutation via `QMetaObject.invokeMethod(registry, 'method_name', Qt.ConnectionType.QueuedConnection, ...)` — *never* call registry mutation methods directly from a worker.

**Anti-pattern:**

```python
# Inside the codec decoder worker thread
self.registry.mark_audible(session_id)  # WRONG — crosses thread without queueing
```

**Pattern:**

```python
QMetaObject.invokeMethod(
    self.registry,
    'mark_audible',
    Qt.ConnectionType.QueuedConnection,
    Q_ARG(str, session_id),
)
```

A small helper `registry.post_mutation(method_name, *args)` wraps this so call sites stay readable.

### P-4 — Signals: naming, payload, and connection type

**Inherited from V2:**
- Event signals: `snake_case`, past tense (`generation_started`, `audio_chunk_ready`).
- State change signals: `{property}_changed` (`current_voice_changed`, `volume_changed`).

**New for this pass:**
- All four registry signals (D-13) **carry `Optional[str]` session IDs**, never the session object itself. Subscribers look up sessions via `registry.get(session_id)`. Avoids stale-reference bugs across signal boundaries.
- Signal connections from worker-thread emitters to UI slots **must** use `Qt.ConnectionType.QueuedConnection`. Auto-connection inference is forbidden in this pass.

**Anti-pattern:**

```python
self.session_state_changed.emit(session)  # passes object, not id
```

### P-5 — Streamer contract

`CodecTokenStreamer(transformers.generation.streamers.BaseStreamer)` has exactly three responsibilities:

1. **`put(value)`** — receive a token (or token batch) from HF `.generate()`. Must:
   - Check `self._cancel_event.is_set()`; if so, return early (no-op).
   - Append to internal token buffer.
   - When buffer reaches `chunk_size + lookahead`, push the chunk onto the bounded queue (D-10) and slide the buffer.
   - Block on full queue (backpressure).

2. **`end()`** — final flush; pushes any remaining tokens with an `end_of_stream` marker.

3. **`reset()`** — clear state; called between sessions; must not be called mid-generation.

**Forbidden:** the streamer never calls into the session, the registry, or the audio coordinator. It only feeds tokens to the queue. Composition with the decoder worker is the registry's job.

### P-6 — Decoder worker contract

A single decoder worker thread per active session. It:

1. Pulls token chunks from the streamer's bounded queue.
2. Decodes with overlap-add (chunk + lookahead, trim middle).
3. Posts each PCM segment via `registry.post_mutation('append_chunk', session_id, pcm)` (P-3).
4. On `end_of_stream` marker: posts `registry.post_mutation('finalize', session_id)`.
5. On cancel event: drains queue, posts `registry.post_mutation('cancel', session_id)`, exits.

**Forbidden:** the decoder worker does not write to disk, does not emit signals, and does not touch audio devices. Strict separation of concerns.

### P-7 — Cancellation propagation

Cancellation flows in one direction: **user → registry → streamer's `_cancel_event` → talker `.generate()` returns → decoder drains → session transitions to CANCELLED → DISCARDED**.

Invariants:
- `session.cancel()` sets the event; it does *not* immediately transition state.
- The decoder worker's drain-on-cancel posts the actual `CANCELLED` transition (so we never have a "cancelled but still emitting chunks" window).
- After `CANCELLED`, the next transition is always `DISCARDED` — buffers are freed in the same Qt-main-thread tick.
- Cancelled sessions never become saveable. The Save button reads `saveable_session_id`; cancellation triggers `saveable_session_changed.emit(<unchanged>)` only if the cancelled session was *not* the saveable one.

### P-8 — PlaybackQueue invariants

Per D-2 and the Step-2 invariant on queue gating granularity:

- Queue serializes at **session level**. Each dequeued session still fans out to monitor + virtual mic in parallel via the existing `AudioCoordinator` plumbing — that is *intra-session parallelism*, not inter-session.
- The queue holds `Deque[str]` of session IDs (not session objects — P-4 invariant).
- Dispatch is gated on `AudioCoordinator._playback_complete_callback`. On each completion, the queue dequeues the next ID and calls `coordinator.dispatch(session)`.
- `playback_queue_depth_changed.emit(len(queue))` fires on every enqueue and dequeue.
- **Streaming exception** (per `02 §3.6`): a session in `GENERATING + is_streaming=True` may dispatch to the audio coordinator *while still GENERATING* if and only if the queue is empty. Subsequent chunks stream into the active playback task via the existing `play_audio_chunk` path. This is one queue slot, not zero.

### P-9 — Telemetry log format

All optimization-pass metrics use a single structured-log format consumable by the existing logger:

```python
log.info("metric", extra={
    "metric_name": "first_chunk_latency_ms" | "decode_chunk_latency_ms" | "queue_depth" | "session_state_duration_ms" | "streaming_mode",
    "value": <number_or_string>,
    "session_id": <optional_str>,
    "tags": {"model_type": ..., "hardware": "gpu"|"cpu"},
})
```

A thin `metrics.record(name, value, **tags)` helper wraps this so call sites don't repeat the boilerplate. The existing `_avg_first_chunk_latency` calculation migrates to read from this stream rather than maintaining its own counter.

### Enforcement Guidelines

**All AI agents implementing this pass MUST:**

1. Use `_transition_to()` for every state change on `GenerationSession` (P-2).
2. Use `registry.post_mutation()` (or equivalent `QMetaObject.invokeMethod` form) when calling registry mutation methods from any worker thread (P-3).
3. Pass session IDs (not session objects) through every signal (P-4).
4. Add the import-attribute test (`tests/test_qwen_tts_internals.py`) with any change that reaches into private `qwen_tts` symbols (D-12).
5. Use the bounded-queue + cancel-event pattern in any new streamer subclass (P-5, P-7).
6. Emit telemetry through `metrics.record()`, never direct `log.info` for metric values (P-9).

**All AI agents implementing this pass MUST NOT:**

1. Assign to `session.state` outside the `_transition_to()` helper.
2. Call registry mutation methods directly from worker threads.
3. Pass `GenerationSession` objects through signals.
4. Add new direct-dispatch paths from generation to audio (must go through queue per P-8).
5. Reach into private `qwen_tts` symbols without updating the import-attribute test.

### Anti-pattern catalog

| Anti-pattern | Why it's wrong | Correct pattern |
|---|---|---|
| `session.state = SessionState.PLAYING` (direct assignment) | Skips signal emission; UI desyncs | `session._transition_to(SessionState.PLAYING)` |
| `self.session_state_changed.emit(session)` | Stale references possible across thread boundaries | Emit `(session_id, state)` tuple |
| Worker thread calling `registry.mark_done(session_id)` directly | Cross-thread Qt mutation | `registry.post_mutation('mark_done', session_id)` |
| `time.time()` in metric path | No structure for downstream analysis | `metrics.record('decode_chunk_latency_ms', value, ...)` |
| `audio_coordinator.play_audio(session.complete_audio)` from generation thread | Skips queue, causes overlap | Enqueue via `registry.enqueue_for_playback(session_id)` |
| Adding a new HF private-symbol import without updating the internals test | Silent breakage on upstream rename | Update `tests/test_qwen_tts_internals.py` in same change |

### Pattern Examples

**Correct session lifecycle (illustrative):**

```python
# Generation thread
session = registry.create_session(text=..., voice=..., source=SessionSource.GENERATED)
# session enters PENDING, then GENERATING when first chunk arrives

# Decoder worker emits chunks via the registry
for pcm_chunk in stream:
    registry.post_mutation('append_chunk', session.id, pcm_chunk)

# End of stream
registry.post_mutation('finalize', session.id)

# Registry, on Qt main thread, transitions to READY_TO_PLAY,
# then enqueues the session for playback if appropriate.
```

**Correct signal subscription:**

```python
class StatusIndicatorWidget(QWidget):
    def __init__(self, registry: SessionRegistry, ...):
        ...
        registry.session_state_changed.connect(self._on_state_changed)
        registry.playback_queue_depth_changed.connect(self._on_queue_depth_changed)

    @pyqtSlot(str, SessionState)
    def _on_state_changed(self, session_id: str, new_state: SessionState):
        session = self.registry.get(session_id)
        if session is None or not session.is_focal_for_indicator():
            return
        self._render(state=new_state, is_audible=session.is_audible)
```

## Project Structure & Boundaries

### Approach

Brownfield. The complete V2 project structure is documented in `architecture.md` (sealed) and need not be repeated. This section covers only:

1. **New modules and files** added by this pass.
2. **Existing files modified** by this pass and the nature of the modification.
3. **Module boundaries** — what the new modules may and may not import.
4. **Test additions** mapped to each new module.
5. **Migration map** showing which phase from D-20 introduces each file.

### New & Modified File Map

```
src/myvoice/
├── services/
│   ├── sessions/                                 ← NEW (Phase 1 — Foundation)
│   │   ├── __init__.py
│   │   ├── generation_session.py                 ← GenerationSession, SessionState,
│   │   │                                            SessionSource, InvalidSessionStateError,
│   │   │                                            _VALID_TRANSITIONS
│   │   ├── session_registry.py                   ← SessionRegistry (Qt main-thread owner),
│   │   │                                            post_mutation helper, the four signals (D-13)
│   │   └── playback_queue.py                     ← PlaybackQueue (Phase 3),
│   │                                                queue_depth_changed wiring
│   ├── tts_streaming/                            ← NEW (Phase ⊥ — Streaming track)
│   │   ├── __init__.py
│   │   ├── streaming_mode.py                     ← StreamingMode enum, hardware probe
│   │   │                                            (torch.cuda.is_available), feature-flag default
│   │   ├── codec_token_streamer.py               ← CodecTokenStreamer(BaseStreamer)  (P-5)
│   │   └── streaming_decoder.py                  ← Decoder worker thread, overlap-add  (P-6)
│   ├── qwen_tts_service.py                       ← MODIFIED — adds TRUE_STREAM dispatch path,
│   │                                                routes state mutations through registry,
│   │                                                keeps batch + sentence_stream paths intact
│   ├── audio_coordinator.py                      ← MODIFIED — accepts sessions from
│   │                                                PlaybackQueue, gates dispatch on
│   │                                                _playback_complete_callback (P-8)
│   ├── audio_service.py                          ← MODIFIED — active_playback_tasks
│   │                                                explicitly per-session-fanout
│   │                                                (semantics-only; no API change)
│   └── ...                                       (unchanged: voice_profile, model_loading,
│                                                   transcription_*, whisper_*, etc.)
│
├── observability/                                ← NEW (Phase 1)
│   ├── __init__.py
│   └── metrics.py                                ← metrics.record() helper, structured-log
│                                                    schema (P-9)
│
├── ui/
│   ├── components/
│   │   ├── service_status_indicator.py           ← MODIFIED (Phase 2 — OFR-D) —
│   │   │                                            subscribes to session_state_changed +
│   │   │                                            playback_queue_depth_changed; reads
│   │   │                                            is_audible from focal session
│   │   ├── save_button.py                        ← NEW (Phase 4 — OFR-A) — enabled per
│   │   │                                            saveable_session_changed
│   │   ├── clear_comms_button.py                 ← NEW (Phase 5 — OFR-B) — main-window
│   │   │                                            button; triggers interrupt+enqueue
│   │   └── ...
│   └── dialogs/
│       ├── save_dialog.py                        ← NEW (Phase 4 — OFR-A) — OS file dialog
│       │                                            wrapper, "Finalizing..." toast logic
│       └── settings/
│           └── clear_comms_settings_panel.py     ← NEW (Phase 5 — OFR-B) — radio
│                                                    (file vs last-generation), test-playback,
│                                                    interrupt-vs-queue toggle
│
└── ...

tests/
├── unit/
│   └── services/
│       ├── sessions/                             ← NEW (Phase 1)
│       │   ├── __init__.py
│       │   ├── test_generation_session.py        ← state transitions (P-1, P-2),
│       │   │                                        InvalidSessionStateError raises,
│       │   │                                        chunks.clear() after concat (D-7)
│       │   ├── test_session_registry.py          ← lifecycle policy (D-3, D-4, D-5),
│       │   │                                        post_mutation thread-safety,
│       │   │                                        signal payloads (P-4)
│       │   └── test_playback_queue.py            ← FIFO ordering, depth signal,
│       │                                            session-level gating (P-8)
│       └── tts_streaming/                        ← NEW (Phase ⊥)
│           ├── __init__.py
│           ├── test_codec_token_streamer.py      ← bounded queue (D-10), backpressure,
│           │                                        cancel-event propagation (D-11),
│           │                                        end_of_stream marker (P-5)
│           └── test_streaming_decoder.py         ← overlap-add seam correctness (mocked
│                                                    decoder), drain-on-cancel (P-7)
├── integration/
│   ├── test_session_lifecycle.py                 ← NEW (Phase 1) — end-to-end through
│   │                                                generation → registry → queue → playback
│   │                                                with mocked TTS + audio
│   ├── test_streaming_tts_smoke.py               ← NEW (Phase ⊥) — happy-path with
│   │                                                mocked HF .generate(); verifies first-chunk
│   │                                                latency telemetry path
│   └── test_playback_last_preservation.py        ← NEW (Phase 3) — D-4 invariant; the
│                                                    saveable-but-not-active slot
├── ui/
│   ├── test_status_indicators.py                 ← MODIFIED (Phase 2) — re-expressed
│   │                                                against session_state_changed source
│   └── test_playback_last.py                     ← MODIFIED (Phase 3) — verify FR28-FR32
│                                                    semantics survive D-4 refactor
└── test_qwen_tts_internals.py                    ← NEW (Phase ⊥) — D-12 import-attribute
                                                     test; CI-blocking on rename

requirements.txt                                  ← MODIFIED (Phase ⊥) — pin qwen-tts
                                                     to specific commit hash
```

### Module Boundaries (import rules)

```
sessions/
├─ generation_session.py     may import: stdlib, numpy, dataclass utilities only
│                            may NOT import: PyQt, services, audio, ui
├─ session_registry.py       may import: PyQt6, sessions.generation_session, observability.metrics
│                            may NOT import: services.qwen_tts_service, services.audio_coordinator
│                            (those services depend on the registry, not the other way around)
└─ playback_queue.py         may import: PyQt6, sessions.generation_session
                             may NOT import: audio_coordinator
                             (queue exposes a callback; audio_coordinator wires it up)

tts_streaming/
├─ streaming_mode.py         may import: torch (for cuda.is_available probe), stdlib
├─ codec_token_streamer.py   may import: transformers (BaseStreamer), torch, queue, threading
│                            may NOT import: sessions, services
└─ streaming_decoder.py      may import: tts_streaming.codec_token_streamer, qwen_tts internals,
                                         numpy, threading, observability.metrics
                             may NOT import: sessions (posts via callback supplied at init)

observability/
└─ metrics.py                may import: stdlib logging only.
                             EVERYTHING may import metrics; metrics imports nothing.

services/
└─ qwen_tts_service.py       may import: sessions.session_registry (passes session_id, never object),
                                         tts_streaming.* (when in TRUE_STREAM mode),
                                         observability.metrics
└─ audio_coordinator.py      may import: sessions.session_registry (read-only access via get()),
                                         sessions.playback_queue (subscribes to it),
                                         observability.metrics

ui/
└─ All ui files              may import: sessions.session_registry (read-only),
                                         sessions.generation_session (for type hints + enum values)
                             may NOT import: tts_streaming, qwen_tts_service directly
                             (UI talks to services through the existing service-locator pattern
                              and to sessions through the registry)
```

**The single forbidden import direction:** `sessions/*` does not import any `services/*`. The registry exposes a callback API; services register callbacks at startup. This keeps the session model unit-testable without a full service stack.

### Requirements → Structure Mapping

| Requirement | Implementation home |
|---|---|
| **OFR-A** Save current generation | `ui/components/save_button.py`, `ui/dialogs/save_dialog.py`, WAV write inline (uses `soundfile`) |
| **OFR-B** Clear Comms button + settings | `ui/components/clear_comms_button.py`, `ui/dialogs/settings/clear_comms_settings_panel.py`; PRELOADED-source loader inline in the panel |
| **OFR-C** Consecutive playback | `services/sessions/playback_queue.py` + `audio_coordinator.py` modifications |
| **OFR-D** Indicator state truth | `ui/components/service_status_indicator.py` modifications |
| **TRUE_STREAM** capability | `services/tts_streaming/*` + `qwen_tts_service.py` modifications |
| **FR28-FR32** Playback Last (preserved) | `services/sessions/session_registry.py` (saveable slot) + write-through cache helper |
| **D-12** `qwen_tts` pin discipline | `requirements.txt` + `tests/test_qwen_tts_internals.py` |

### Cross-Cutting Concerns Mapping

- **Threading discipline (D-2, P-3):** centralized in `session_registry.py`'s `post_mutation` helper. Every worker-thread-to-Qt-thread crossing flows through this single chokepoint.
- **State machine (P-1, P-2):** centralized in `generation_session.py`'s `_transition_to`. Every state mutation in the codebase goes through this single chokepoint.
- **Metrics (P-9):** centralized in `observability/metrics.py`. Every metric-bearing call site goes through `metrics.record()`.
- **Cancellation (P-7):** flows through the `threading.Event` owned by `CodecTokenStreamer`, propagated by the registry on `session.cancel()`.

### Integration Boundaries

#### Internal communication (within this pass)

```
Generation thread             Decoder worker thread          Qt main thread
─────────────────             ───────────────────            ──────────────
qwen_tts_service                streaming_decoder              session_registry
       │                              │                              │
       │                              │  post_mutation('append_chunk')
       │                              │ ─────────────────────────────►
       │                              │                              │
       │  post_mutation('finalize')                                   │
       │ ─────────────────────────────────────────────────────────►   │
       │                                                              │
                                                                     emit
                                                              session_state_changed
                                                                      │
                                                                      ▼
                                                                UI subscribers
```

#### External integration (existing V2 plumbing)

- **AudioCoordinator → MonitorAudioService + VirtualMicrophoneService:** unchanged from V2. Sessions are dispatched via existing fan-out at the `dispatch(session)` boundary; intra-session parallelism is preserved (P-8 invariant).
- **`qwen_tts` library:** reached via two private symbols (`model.model.generate`, `speech_tokenizer.decode`) for the streaming path; the public API (`generate_custom_voice`, etc.) still drives batch and sentence_stream paths unchanged.
- **`requirements.txt` / dependency graph:** zero new packages — all new files use libraries already in tree (`numpy`, `soundfile`, `torch`, `transformers`, `PyQt6`, `qasync`, stdlib).

### File Organization Patterns

- **New service namespaces** (`services/sessions/`, `services/tts_streaming/`, `observability/`) follow the existing flat-with-subdirectories convention used elsewhere in the repo (`services/integrations/`, `services/core/` are precedent).
- **Tests mirror source layout** under `tests/unit/services/` for unit tests; integration tests live at `tests/integration/`. This matches the pre-existing dual-convention in the test tree (V2 architecture chose this; we don't change it).
- **No module-level circular dependencies** allowed. The import-rule table above is the enforcement.

### Development Workflow Integration

- **Existing dev server / launch flow:** unchanged. `python -m myvoice` (or whatever entry point V2 documents) continues to work; new services initialize in the existing service-locator at startup.
- **Build process:** unchanged for Phases 1–5. Phase ⊥ adds the `qwen_tts` private-API import test to CI; existing build steps absorb it as a unit-test.
- **Migration order matches D-20.** Each phase is a self-contained PR-able unit. Phase 1 ships with no user-visible change; subsequent phases are individually flagged in commit messages for revert clarity.

### Out of structural scope

- New top-level directories outside `src/myvoice/`. None proposed.
- Repackaging of existing V2 modules (`whisper_*`, `transcription_*`, `voice_profile_*`, etc.). Untouched.
- Changes to `requirements-production.txt` beyond mirroring the `qwen-tts` pin from `requirements.txt`.

## Architecture Validation Results

### Coherence Validation ✅

**Decision compatibility:** All decisions composed without contradiction, with one composition that warrants explicit explanation:

- **D-3 (saveable = most recent finalize) composed with D-4 (lingering slot until next DONE).** These do not conflict; they describe two different things. **D-3 governs the *reference* (which session ID `saveable_session_id` points at).** **D-4 governs the *buffer* (when the previous session's `complete_audio` is freed).** Concretely:
  - When session A finalizes → `saveable_session_id = A`. Cache file written through.
  - When session B finalizes → `saveable_session_id = B`. A's buffer is *not yet freed* — it lingers as the "previous saveable" until B reaches DONE.
  - When B reaches DONE → A is discarded; A's buffer freed. Cache file already reflects B (overwritten on B's finalize).
  - Save button always operates on `saveable_session_id`. The lingering A buffer is grace-period memory, not user-visible content.

- **D-9 (hardware-aware streaming default) composed with NFR1 (first audio <2s).** On CPU the default is `SENTENCE_STREAM`, which today already meets NFR1 for non-trivially-short inputs and is the V2 baseline. NFR1 satisfaction on CPU is therefore inherited, not promised by streaming. Validation gate moved to *post-implementation measurement* per Step 2 callout.

**Pattern consistency:** P-1 through P-9 align with D-1 through D-20 without override. The single chokepoint patterns (P-2 `_transition_to`, P-3 `post_mutation`, P-9 `metrics.record`) match the single-owner principle established in D-1 and D-2. No pattern requires a decision that wasn't made.

**Structure alignment:** Module boundaries support the chokepoint patterns. The forbidden import direction (`sessions/*` → `services/*`) enforces the dependency inversion that lets the registry be unit-testable without a service stack.

### Requirements Coverage Validation ✅

**Inherited FR coverage:**

| FR | Covered by |
|---|---|
| FR2 First audio <2s (streaming) | D-8/D-9/D-10 (streaming POC) for short inputs; existing `SENTENCE_STREAM` for long inputs and CPU fallback |
| FR3 Batch fallback | D-9 hardware-aware default + P-7 cancellation enables clean rollback through three modes |
| FR4 User can cancel | P-7 cancellation chain (registry → streamer event → decoder drain → CANCELLED transition) |
| FR28 Playback Last replays cached | D-4 saveable slot (in-memory) + write-through cache projection |
| FR29 Available in main and Voice Design | Read from registry; no scope change to dialog this pass |
| FR30 Device change handled | Inherited V2 plumbing — unchanged |
| FR31 Save most-recent to cache | D-4 write-through projection on `finalize()` |
| FR32 Playback Last from cache (no regen) | Reads from saveable slot first; cache file is restart-recovery path |
| FR42 Display TTS service status | OFR-D rewire (Phase 2) |

**Inherited NFR coverage:**

| NFR | Covered by |
|---|---|
| NFR1 First audio <2s | GPU: meets via TRUE_STREAM (~1.5–1.8s estimated). CPU: meets via inherited SENTENCE_STREAM (per FR2 row). **Empirical measurement gate at Phase ⊥ POC.** *(Story 16.9 reconciled 2026-05-08 — empirical contradiction; per-class targets adopted. See follow-up note below.)* |
| NFR3 No audio stuttering | D-8 chunk + overlap-add with seam-quality A/B testing before flipping streaming default; sentence-stream/batch unchanged from V2 *(Story 17.1 audition cleared 2026-05-08 — see follow-up note below.)* *(Story 18.3 bf16 audition DEFERRED 2026-05-10 pending Story 18.4 producer-bottleneck close — measured no speedup over Story 18.2 fp32+TF32 baseline; revisit post-18.4. See follow-up note below.)* *(Story 18.4 joint audition FULL PASS 2026-05-11 — certified bf16 + compile + pin-bump from QwenLM/Qwen3-TTS@1ab0dd75 to dffdeeq/Qwen3-TTS-streaming@3fdb4682; OFR-E producer-bottleneck closed (ratio 3.23× → 0.670×); zero `audible_seam` flags across all 30 trials × 2 conditions = 60 defect observations from 3 listeners; one slight bf16_compile preference (L1 on s-015), 29 equivalent. See follow-up note below.)* |
| NFR4 UI <200ms | D-2 Qt-thread ownership of registry; mutation work bounded to state transitions, not waveform I/O |
| NFR6 No crashes | D-12 import-attribute test + P-1 state-bound method validity (no silent no-ops) + P-7 clean cancellation (no half-state CUDA) |
| NFR7 Graceful degradation | D-9 + the three-mode dispatch in `qwen_tts_service.py` (BATCH ← SENTENCE_STREAM ← TRUE_STREAM fallback chain) |
| NFR11 <4GB RAM with model | D-7 chunks.clear after concat; D-4 single saveable + lingering = bounded extra footprint (~5MB for 30s utterance); inherited single-model lazy load |
| NFR12 CPU-only support | D-9 hardware-aware streaming default (CPU stays on SENTENCE_STREAM) |

**Local FR-equivalent coverage:**

| OFR | Covered by |
|---|---|
| OFR-A Save current generation | D-3, D-4, D-15, D-16 + new `save_button.py`/`save_dialog.py` (Phase 4) |
| OFR-B Clear Comms | D-5, D-17, D-18 + new `clear_comms_button.py`/settings panel (Phase 5) |
| OFR-C Consecutive playback | D-2, P-8 + new `playback_queue.py` (Phase 3) |
| OFR-D Indicator state truth | D-13, D-15, P-4 + modified `service_status_indicator.py` (Phase 2) |

#### Story 16.9 Follow-up Note (NFR1 Reconciliation, 2026-05-08)

Empirical measurement on the maintainer's RTX 5090 + qwen-tts 0.0.4 host (Story 16.7 §3.2 + Story 16.9 Tasks 2 / 3.2 / 6) demonstrated that the original NFR1 projection above (`~1.5–1.8s estimated`) was empirically contradicted across all input classes:

| Path | n | first-chunk p95¹ | NFR1 (<2s) |
|------|---|-----------------|------------|
| GPU TRUE_STREAM (post-Story-16.8 wire-up) | 50 | 6.37s | FAIL (3.19× over) |
| GPU SENTENCE_STREAM short (steady-state)¹ | 17 (gen) / 16 (clearance) | 4.18s | FAIL (2.09× over; 9/16 clear) |
| GPU SENTENCE_STREAM medium | 17 | 8.74s | FAIL (4.37× over; 0/17 clear) |
| GPU SENTENCE_STREAM long | 16 | 25.23s | FAIL (12.62× over) |
| GPU SENTENCE_STREAM small-tier (0.6B) short | 17 | 7.94s | FAIL (small tier ~2× *slower* than 3B `quality` on Blackwell) |
| CPU SENTENCE_STREAM short stratified | 4 | 5.40s | FAIL |
| CPU SENTENCE_STREAM medium stratified | 4 | 7.85s | FAIL |
| CPU SENTENCE_STREAM long stratified | 2 | 22.37s | FAIL |

> **¹** The "GPU SENTENCE_STREAM short (steady-state)" row mixes column sources (clarification added by code-review pass — Story 16.9 Change Log #4 / M1): cited 4.18s is `generate_seconds` p95 at n=17 (steady-state per-utterance dispatch cost; user-facing first-chunk latency once model is warm); "9/16 clear" is computed from `first_chunk_latency_seconds` at n=16 (drop s-001 warmup whose 4.79s includes a 3.65s cold model_load contribution paid once per session). Strictly disambiguated: `first_chunk_latency_seconds` p95 (n=17, includes s-001) = 4.93s; `first_chunk_latency_seconds` p95 (n=16, drop s-001) = 4.26s; `generate_seconds` p95 (n=17) = 4.18s. All three FAIL the original 2s ceiling and pass the new ≤5.0s short-class target. For medium/long classes the columns are equivalent because cold model load contributes only to s-001 (class-short).

Phase-decomposition profiling (Story 16.9 AC #1) showed `_generate_sync` (the `model.generate_custom_voice` invocation site) accounts for **≥97% of first-chunk wallclock** on both GPU and CPU; `_split_text_for_streaming`, registry `post_mutation`, and chunk-delivery overhead are individually ≤0.1% of total. The 3B `quality` model on RTX 5090 + qwen-tts 0.0.4 has a **~1.2s per-utterance floor** for any input; the length-latency slope is ~+0.10 sec/char (Pearson r = +0.915, n=49). The 0.6B `small` tier is empirically slower than the 3B `quality` tier on Blackwell (Story 16.9 Task 3.2 reversal), so a model-tier fallback is **ruled out**. Aggressive splitter changes cannot clear the original 2s target without harming voice quality (audition deferred to a future "streaming default ramp" story per Story 16.7 §6.1).

Story 16.9 reconciled NFR1 with empirical reality via outcome (c) "contract revision" (no production code change). The revised wording is:

> **NFR1 (revised 2026-05-08, Story 16.9): First-audio latency under streaming dispatch.**
>
> | Class | First-chunk char range | GPU `quality` p95 target | Empirical (Story 16.9 Task 2) |
> |---|---|---|---|
> | Short | ≤30 chars | ≤5.0s | 4.18s ✓ |
> | Medium | 30–100 chars | ≤10.0s | 8.74s ✓ |
> | Long | >100 chars | informational only (no formal target); UI provides progress indicator | 25.23s |
>
> **CPU SENTENCE_STREAM** is exempted from the streaming-NFR1 contract; CPU users fall back to the V2 baseline. Hardware-aware default (D-9 / NFR12) ensures CPU users do not encounter TRUE_STREAM. The 0.6B `small` tier is empirically slower on Blackwell + qwen-tts 0.0.4; tier-switching is not a viable NFR1 lever.
>
> Rationale: the original "~1.5–1.8s estimated" projection was authored before the 3B model's per-token cost on this hardware was empirically known. The 1.2s structural floor in `_generate_sync` is upstream-bound (qwen-tts 0.0.4); any future material improvement is pin-bump-conditional and tracked separately.

The streaming-default flag flip (the user-facing release of TRUE_STREAM as the GPU default — a one-line edit at `streaming_mode.py:54-56` or a settings UI initializer) was conjunction-blocked on Story 16.8 (TRUE_STREAM viable — closed 2026-05-08) AND Story 16.9 (NFR1 reconciled — closed 2026-05-08). With both stories closed, the flag flip's **remaining prerequisite is the multi-listener perceptual A/B audition** (Story 16.7 AC #2's deferred protocol), tracked in a future "streaming default ramp" story.

NFR7 (graceful degradation — TRUE_STREAM → SENTENCE_STREAM → BATCH fallback chain) is preserved unchanged; D-9 / NFR12 (hardware-aware streaming default; CPU stays on SENTENCE_STREAM) is preserved unchanged. No qwen-tts pin bump (the pin remains at commit `1ab0dd75` = qwen-tts 0.0.4 per Story 16.1).

Source artifacts:
- `_bmad-output/implementation-artifacts/16-9-nfr1-reconciliation-report.md` (load-bearing report)
- `_bmad-output/implementation-artifacts/16-9-correct-course-nfr1-revision.md` (stakeholder routing artifact)
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-phase-profile.csv` (50 rows)
- `_bmad-output/implementation-artifacts/16-9-gpu-sentence_stream-small-tier-comparison.csv` (17 rows)
- `_bmad-output/implementation-artifacts/16-9-cpu-sentence_stream-stratified.csv` (10 rows)

#### Story 17.1 Follow-up Note (Streaming Default Ramp Audition, 2026-05-08)

Story 17.1 (single-story Epic 17 — Streaming Default Ramp) executed the deferred Story 16.7 AC #2 multi-listener perceptual A/B audition against the post-Story-16.8 regenerated fixture (`_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/`). Three listeners (L1 = Commander; L2 + L3 = co-located in-person walkthrough listeners) labeled all 10 utterances of the perceptual-difficult subset (`s-014/15/16/17`, `m-011/12/13/14`, `l-013/14`) per the controlled defect vocabulary in `LISTENING-INSTRUCTIONS.md`. Total: 30 trials × A and B renditions per trial.

Per-system defect-flag count (verbatim from `17-1-perceptual-ab-results.csv` joined against `16-8-perceptual-fixtures/_perlistener_truthtable.json`):

| System | Trials | none | audible_seam | clipping | phase_artifact | tonal_distortion | other |
|---|---|---|---|---|---|---|---|
| TRUE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |
| SENTENCE_STREAM | 30 | 30 | **0** | 0 | 0 | 0 | 0 |

Per-listener subtotals: L1 = 0/10 audible_seam (TRUE_STREAM); L2 = 0/10; L3 = 0/10. Per-utterance subtotals: 0/3 audible_seam on every utterance for every system. No `(listener_id, utterance_id)` rows missing; no schema-validation or truth-table-join errors.

**Verdict per the LISTENING-INSTRUCTIONS.md gate verbatim** (PASS iff zero listeners flagged `audible_seam` on any TRUE_STREAM pair): **PASS — outcome (a) per Story 17.1 AC #3.**

**Architectural decision: streaming-default flag flip certified.** The existing `streaming_mode.py:54-56` hardware probe's TRUE_STREAM-on-CUDA default is the audited release default — no code change required (Epic 16 wired this in at Story 16.8 and the dispatch path has been live on this branch since 2026-05-08). NFR7 graceful-degradation chain (TRUE_STREAM → SENTENCE_STREAM → BATCH) is preserved unchanged. D-9 / NFR12 hardware-aware default (CPU stays on SENTENCE_STREAM) is preserved unchanged. No qwen-tts pin bump (pin remains at commit `1ab0dd75`). No NFR1 revisit (Story 16.9's per-class targets stand).

**Methodology limitations (escalated by Story 17.1 code-review pass per H1 — read these as bounding the verdict's confidence, not as soft caveats).** L2 and L3 sessions were conducted as in-person walkthroughs with Commander as scribe (listener-id arg passed to `17-1-l1-audition-helper.py`; helper's per-utterance forced playback of trial-A and trial-B blind ensured per-pair attention before labeling; helper's controlled-vocabulary input gates enforced label discipline). Three structural limitations apply:

1. **Single-room listening environment.** Listeners were co-located on Commander's playback hardware rather than auditioning on independent setups. The story's `>` header at line 39 had specified "own playback hardware (headphones if available; not a single shared listening environment) — this is the LISTENING-INSTRUCTIONS.md protocol verbatim and reflects realistic Discord-call usage." The walkthrough format substitutes for the prose-named protocol; this substitution was Commander-approved at runtime via Story 17.1 Change Log #7 but the substitution narrows the auditing population's hardware diversity to one. Discord-call users on lower-bitrate / smaller-driver setups are not represented.
2. **Single-scribe prompt-framing risk.** Commander asked the same yes/no controlled-vocab questions ("did you hear an audible click? clipping? phase artifact? tonal distortion?") to L2 and L3 in succession. This introduces a single source of question-framing for both listeners that the helper's input-validation gates do not address. A listener uncertain about a defect may default to `none` rather than commit to a label in front of the maintainer; the verdict's data is consistent with this — TRUE_STREAM AND SENTENCE_STREAM both register 30/30 `none` on every controlled-vocabulary category, which means the audition either (i) genuinely surfaces no defects worth labeling, or (ii) has lower discriminative power than its N=3 framing suggests. Outcome (a) certification stands because the gate is `audible_seam`-specific (zero-flag is the literal pass condition) — but a stronger inferential claim of "TRUE_STREAM ≡ SENTENCE_STREAM perceptually" is not supported by this audition.
3. **L1 anonymization not preserved (and not preservable in solo-dev framing).** The committed audit identifies L1 as Commander explicitly — necessarily, since Commander is the sole stakeholder per `memory/production_release_state.md`. Anonymization was preserved for L2/L3 in the committed CSV; private listener-ID-to-human mapping is held informally by Commander.

The verdict (PASS, outcome (a)) is the gate's literal reading of the data and is the right closure for Phase ⊥-Ramp. These limitations bound how the audition's data should be cited in future work — a follow-up audition with independent listeners on diverse playback hardware would supply stronger evidence if a future qwen-tts pin bump or chunk-size retune story needs to re-validate NFR3. Captured in the routing artifact's §6 "Stakeholder sign-off" methodology disclosure for full transparency.

**Informational signal.** L1 noted on m-012 that trial A (TRUE_STREAM in L1's randomization) was perceived as quieter than trial B (SENTENCE_STREAM) — a volume-amplitude observation, not a defect (zero `audible_seam` flagged on either trial; L1's `a_or_b_preferred = B` for that one utterance). L2 and L3 did not flag the same observation independently. Captured here for future hardware-aware default-tuning consideration; not actionable in this story.

The streaming-default flag flip's three architectural prerequisites — (1) TRUE_STREAM viable (Story 16.8, closed 2026-05-08), (2) NFR1 reconciled (Story 16.9, closed 2026-05-08), (3) perceptual gate cleared (this story, 2026-05-08) — are all resolved. **Phase ⊥-Ramp closes. The V2 optimization pass closes.**

Source artifacts (✓ = git-tracked; ○ = working file under gitignored `_bmad-output/`, retained on Commander's filesystem only):

- ✓ `_bmad-output/implementation-artifacts/17-1-perceptual-ab-results.csv` (30 rows; force-added) — the **only fully reproducible artifact**; the rows are the audited verdict's evidentiary surface.
- ✓ `_bmad-output/implementation-artifacts/17-1-correct-course-streaming-default-ramp.md` (stakeholder routing artifact, AC #4; force-added).
- ✓ `_bmad-output/planning-artifacts/sprint-change-proposal-2026-05-08.md` (correct-course workflow native output; force-added).
- ✓ `_bmad-output/implementation-artifacts/17-1-l1-audition-helper.py` (audition driver; **force-added by Story 17.1 code-review pass per M2** — preserves blinding by playing WAVs without printing filenames; see helper docstring lines 17–22 for the blinding constraint LISTENING-INSTRUCTIONS.md cannot satisfy on its own).
- ○ `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/` (10 paired WAVs from Story 16.8 regeneration; truth-table `_perlistener_truthtable.json` with L1/L2/L3 randomizations). **Not in git** (binary fixture under gitignored `_bmad-output/`); the verdict's truth-table join was performed against this file, but a fresh clone cannot reproduce the audition without it. Story 17.1 forbids fixture regeneration (would invalidate prior audit data); the canonical fixture state is held only on Commander's filesystem. **Reproducibility implication (M1):** future maintainers re-validating this verdict need either the original fixture or a fresh full audit cycle (regenerate fixture → re-audit at N≥3 → recompute verdict).
- ○ `_bmad-output/implementation-artifacts/16-8-perceptual-fixtures/LISTENING-INSTRUCTIONS.md` (canonical protocol; not in git for the same reason; byte-identical to the 16-7 directory's copy).
- ✓ `_bmad-output/implementation-artifacts/17-1-streaming-default-ramp.md` (story document; Change Log #8 contains the verbatim verdict-computation tables; Change Log #10 documents the code-review pass).

#### Story 18.3 Follow-up Note (bf16 Precision Audition — DEFERRED, 2026-05-10)

Story 18.3 (Epic 18 / Phase ⊥-Polish-2 third story — bf16 Precision on Talker + Decoder) executed Tasks 1 (dtype audit) + 7 (NFR1 first-chunk-latency measurement) on the RTX 5090 dev host (Blackwell GeForce, capability 12.0). Tasks 8 (≥3-listener perceptual A/B audition) and 9-as-PASS-amendment were **deferred** per Open Question #3 routing — the empirical NFR1 measurement did not deliver the 30–50% speedup anticipated at `epics-optimization-pass.md:1381`, and Commander selected option (b) "defer to future investigation post-Story-18.4" per OQ #3's three-option framing.

**Audit findings — bf16 IS engaged end-to-end (Task 1 confirmed via env-var-gated `_instrument_dtype_audit` in `model_registry.py`):**

- `model.model.talker.dtype = torch.bfloat16` ✓
- `model.model.speech_tokenizer.model` (the inner `Qwen3TTSTokenizerV2Model` — qwen-tts wrapper's `nn.Module` for codec decoding) — every sampled parameter is `torch.bfloat16` ✓ (surprising — Story 18.3 Dev Notes' central audit hypothesis was that vocoders typically stay in fp32 for numerical stability; qwen-tts 0.0.4 actually loads the codec in bf16 too)
- Talker forward-hook capture: every tensor kwarg + output tensor is `torch.bfloat16` ✓ (no autocast/upcast erasing the bf16 compute gain)

**NFR1 measurement — bf16 vs fp32 (cold-start; N=10 fresh-process launches per branch; one Sarira-F long-form Sarira-F utterance per launch):**

| Statistic | bf16 (auto) | fp32 (override) | delta (ms) | delta (%) |
|---|---|---|---|---|
| median first-chunk-latency | 5029 ms | 4846 ms | -183 | **-3.77%** (bf16 slightly slower) |
| p90 | 5657 ms | 5409 ms | -247 | -4.57% |
| p95 | 5705 ms | 5634 ms | -70 | -1.25% |

Producer steady-state ratio (Story 18.1 §4.4 methodology — mean inter-chunk-emit interval ÷ mean chunk audio duration):

| Branch | mean interval | mean duration | ratio | vs Story 18.1 baseline (3.23×) |
|---|---|---|---|---|
| bf16 | 3213 ms | 1981 ms | **1.62** | -50% (good) |
| fp32+TF32 | 2782 ms | 1981 ms | **1.40** | -57% (better) |

**Diagnosis:** the [30%, 50%] anticipated gate at `:1381` was an estimate based on bf16's tensor-core advantage at training-style batch shapes. On the V2 inference workload — autoregressive single-token talker forwards with kernel-launch overhead + KV-cache management dominating — the matmul-throughput advantage does not materialize. Story 18.2's TF32 + cuDNN benchmark engagement (3.23 → 1.40 producer ratio) had already collected the bulk of the producer-bottleneck win on this architecture; bf16's residual headroom over fp32-with-TF32-engaged is small or slightly negative. Per-launch variance is large (±1500 ms in some pairs); even if a 5–10% real effect existed, N=10 lacks the statistical power to distinguish it from noise.

**Architectural decision: ship Story 18.3 source-tree work + setting + audit infrastructure; defer the bf16-as-default decision to post-Story-18.4 retrospective.** Specifically:

- ✓ `AppSettings.tts_precision` field with `auto`/`bf16`/`fp32` validation lands as designed (NFR7 fp32 fallback path is ready for any user who hits a future perceptual issue).
- ✓ `resolve_tts_precision(override)` resolver in `tts_streaming.torch_runtime` lands as designed.
- ✓ `ModelRegistry.__init__` precedence rule + telemetry (`tts_precision_resolved` metric) + INFO log breadcrumb (`precision_source='...'`) land as designed.
- ✓ `tts_precision="auto"` continues to resolve to `bfloat16` on Ampere+ — the engaged-but-no-measured-speedup path. This is the **conservative ship-as-engaged choice**; Commander can flip to fp32-default in a future story if the post-18.4 retrospective shows bf16 still doesn't pay.
- **Deferred:** Task 8 ≥3-listener perceptual A/B audition. The audition is a load-bearing perceptual gate for a default that doesn't pay for itself on perf — recruiting listeners at this point would burn the listener-recruitment budget on a decision the data can't yet justify. Re-running the same `03_*.bat` + `04_*.bat` harness post-Story-18.4 (with `torch.compile`'s CUDA graphs / kernel-launch-overhead collapse) will give a cleaner answer; if bf16 starts helping there, the audition fires at that point.

**NFR3 status (re-clearance not in this story).** The Story 17.1 audition's verdict (PASS — zero `audible_seam` flags across 30 trials on the post-Story-16.8 regenerated fixture) remains the canonical NFR3 clearance for the streaming-default ramp. Story 18.3 does NOT re-clear NFR3; the audition is deferred. The bf16 path is engaged in production today (V2 default; reaches users via the new resolver path post-Story-18.3), so any user-reported perceptual defect on bf16 surfaces immediately and Commander has the `tts_precision="fp32"` setting as the NFR7 escape hatch — that is the perceptual safety net while the audition is deferred.

**Methodology composition (per Story 18.3 AC #10).** The fp32 branch in this A/B is **fp32-with-TF32-engaged**, not strict-fp32. Story 18.2's TF32 + cuDNN benchmark autotune engages at startup on every Ampere+ host regardless of precision; the bf16-vs-fp32 comparison here is therefore bf16 vs (fp32 + TF32 + cuDNN benchmark). Strict-fp32 (TF32 disabled) is out of scope per AC #10 — Story 18.2 closed it as null on the producer-bottleneck workload and Story 18.3 does not re-litigate.

**Methodology limitations (mirrored from Story 17.1's Phase ⊥-Ramp pattern).** Three structural limitations apply to Story 18.3's NFR1 measurement and inform the deferred audition's reproducibility:

1. **Single-host RTX 5090 measurement.** Captured only on Commander's RTX 5090 Blackwell dev host (capability 12.0 / GeForce variant). Earlier Ampere/Ada hosts (RTX 30xx/40xx) may show a different bf16-vs-fp32-with-TF32 profile — Blackwell's TF32 tensor cores are unusually capable. The architecture amendment's data should not be cited as "bf16 doesn't help on Ampere+" generally; only as "bf16 doesn't help on Blackwell GeForce 12.0 in our V2 inference workload."
2. **Cold-start variance.** Per-launch first-chunk-latency varies by ±1500 ms across N=10 fresh-process pairs (model load + first-token kernel JIT + cuDNN benchmark autotune cache warmup all live in the cold-start window). N=10 lacks the statistical power to distinguish a 5–10% real effect from this noise floor. A future re-measurement with N=30 or with a warmup-discount methodology would tighten the confidence interval.
3. **Audition not run.** Per OQ #3 option (b), the listener audition (Task 8) was deferred. The architecture amendment lands without an NFR3 re-clearance for the bf16 path specifically — the Story 17.1 NFR3 clearance covers the streaming-mode A/B (TRUE_STREAM vs SENTENCE_STREAM) but not the precision-tier A/B (bf16 vs fp32). Future maintainers re-validating bf16 for a future qwen-tts pin bump or for a Story 18.4 re-measurement need to run the audition fresh.

Source artifacts (✓ = git-tracked; ○ = working file under gitignored `_bmad-output/`, retained on Commander's filesystem only):

- ✓ `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder.md` (story document with Tasks 1–11 closure state and Change Log entries).
- ✓ `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md` (evidence file with full §"Pre-implementation audit" + §"End-to-end dtype audit" + §"Streaming pipeline dtype audit" + §"NFR1 first-chunk-latency measurement" captures + §"Side observation — finalization race surfaced by bf16 engagement (FIXED in-story)").
- ✓ `_bmad-output/implementation-artifacts/18-3-set-precision.py` (settings.json mutation helper for the NFR1 measurement bats).
- ✓ `_bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py` (N=10 aggregator + steady-state ratio analysis + auto-OQ-#3 detection).
- ✓ `_bmad-output/implementation-artifacts/18-3-l1-audition-helper.py` (audition helper — adapted from 17-1; **kept on disk for the deferred audition**; not exercised in this story's closure).
- ✓ `02_Story_18.3_DType_Audit.bat` + `03_Story_18.3_NFR1_BF16.bat` + `04_Story_18.3_NFR1_FP32.bat` (Commander-routed harness at repo root).
- ✓ `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16.csv` (consolidated N=10 cold-start first_chunk_latency_ms; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32.csv` (consolidated N=10 cold-start first_chunk_latency_ms; force-added).
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run<NN>.csv` (10 per-run CSVs; force-add at Commander's discretion).
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32-run<NN>.csv` (10 per-run CSVs).
- ○ `_bmad-output/implementation-artifacts/18-3-perceptual-fixtures/` — **not produced** (audition deferred). The directory will be created if/when the audition is re-attempted post-Story-18.4.
- ○ `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv` — **not produced** (audition deferred).

#### Story 18.4 Follow-up Note (Joint bf16 + Compile + Pin-Bump Audition — FULL PASS, 2026-05-11)

Story 18.4 (Epic 18 / Phase ⊥-Polish-2 fourth and final story — torch.compile Decoder + Persistent Compile Cache) executed all dev-agent-autonomous tasks + Task 8 (NFR1 3-way A/B/C measurement) + Task 9.2 (fixture regen) + Task 9.4 (full L1 + L2 + L3 audition). The architecture's load-bearing OFR-E producer-bottleneck-close target is **ACHIEVED**; the perceptual NFR3 gate is **FULL PASS** — zero `audible_seam` flags across all 30 trials × 2 conditions from 3 listeners.

**D-22 verification — Branch B fires (architecture's high-risk gate, named at line 1319 of `architecture-streaming-acceleration-and-lightning-tier.md`).**

Per architecture D-22 Branch A vs Branch B framing, the empirical question was: does `qwen-tts@1ab0dd75353392f28a0d05d9ca960c9954b13c83` ship `Qwen3TTSModel.enable_streaming_optimizations`? Dev-agent verification at Task 1.1 (`grep -rn "enable_streaming_optimizations" python310/Lib/site-packages/qwen_tts/`) returned **zero matches** — the API does not exist at the existing pin. **Branch B fires.** The pin-bump landed at `requirements.txt:23` + `build_tools/requirements-production.txt:61` from `QwenLM/Qwen3-TTS@1ab0dd75` (upstream main lineage) to `dffdeeq/Qwen3-TTS-streaming@3fdb468233d73fa537202b94a1cc7c4e7a6160b8` (community fork, introducing commit "compile and fast codebook" 2026-02-03). The fork is a drop-in replacement (same `qwen-tts 0.0.4` package name/version; +50/-6 lines additive diff; no removed symbols MyVoice depends on).

**Pin-bump rationale (D-22 Branch B EXECUTED):**

- Same package name (`qwen-tts`) + same version (0.0.4) — no `pyproject.toml` resolution surprises.
- Additive diff verified at Task 1.2 via `gh api repos/dffdeeq/Qwen3-TTS-streaming/commits/3fdb4682` — 3 files changed, +50/-6 lines, all additive. No removed symbols.
- `Qwen3TTSModel.enable_streaming_optimizations` signature: `(decode_window_frames=80, use_compile=True, use_cuda_graphs=True, compile_mode="reduce-overhead", use_fast_codebook=False, compile_codebook_predictor=True, compile_talker=True)`. MyVoice invokes 4 of 7 kwargs per the production wire-up (`decode_window_frames=30, use_compile=True, compile_mode="reduce-overhead", compile_talker=False`) — the **`compile_talker=False`** override is required for Story 16.8's TRUE_STREAM forward-hook compatibility (the talker forward hook captures per-step `codec_ids` from `Qwen3TTSTalkerOutputWithPast.hidden_states[1]`; `torch.compile`-wrapping the talker breaks the capture path, producing the `ValueError: finalize() called with no chunks` regression observed in the first bundled-smoke iteration 2026-05-10).
- Story 17.2 cached `voice_clone_prompt` `.pt` files are invalidated by the pin-bump per `_QWEN_TTS_PIN_HASH = "3fdb4682"` constant bump at `qwen_tts_service.py:1150` (verified at Task 11.5 via `test_pin_mismatch_invalidates_pt` continuing to pass against the new constant).
- Quarterly upstream check (analogous to architecture D-33's `chatterbox-streaming` discipline): check whether `QwenLM/Qwen3-TTS` upstream picks up the patch via PR from the fork author. If yes, swap the pin back to upstream + drop the community-pin discipline. Tracked in `memory/build_tools_phase_perp_state.md`.

**Fixture regeneration methodology (Task 9.2 — automated 2026-05-11):**

The original story plan routed fixture regen via the production GUI as Commander-routed manual work. Replaced by `_bmad-output/implementation-artifacts/18-4-regen-fixture.py` which drives `Qwen3TTSModel.generate_voice_clone` directly with the precomputed Sarira-F voice_clone_prompt cache (Story 17.2 cache at pin `3fdb4682`). All 20 WAVs generated in 2 min flat: Branch A (fp32+eager) in 60 s; Branch B (bf16+compile-engaged) in 40 s. The fork's internal compile targets engaged cleanly: Tokenizer (decoder forward), Decoder forward, CodePredictor. Cold-compile cost (~14 s on s-014) was amortized across the 9 subsequent B generations (~2-3 s each). Per-utterance B vs A speedup ranged from 1.83× (m-013) to 2.50× (s-015 warm) — additional empirical evidence the compile path delivers real per-call speedup on the perceptual-difficult utterance subset.

Truth-table at `_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/_perlistener_truthtable.json` was built via `18-4-generate-truthtable.py` with deterministic per-listener randomization (seed `"Story 18.4 joint audition:<listener_id>"`); L1 / L3 = 5/5 fp32-as-trial-A counts (balanced), L2 = 4/6 fp32-as-trial-A counts (mild bf16-as-A bias acceptable per the architectural-blinding contract).

**NFR1 measurement (Task 8) — OFR-E producer-bottleneck close ACHIEVED.**

Three branches × N=10 fresh-process launches per branch; one Sarira-F long-form utterance per launch (Story 17.3 §4.1 step 3 canonical paragraph; ≥250 chars / ~22 s of speech):

| metric | A (bf16+compile) | B (bf16+eager) | C (fp32+eager) |
|---|---|---|---|
| median first_chunk_latency_ms | 5929.4 | 5517.8 | 5455.3 |
| p90 | 6072.9 | 6388.9 | 5601.9 |
| p95 | 6057.3 | 6191.5 | 5593.5 |
| **producer-bottleneck steady-state ratio** | **0.670×** ✓ | 1.663× | 1.430× |

Pairwise first-chunk-latency deltas (positive = treatment faster than baseline):
- A vs B (compile gain over bf16-eager): median Δ = -411.5 ms (-7.46%) — compile slower on first-chunk
- A vs C (compounded gain over fp32-eager): median Δ = -474.1 ms (-8.69%) — compile+bf16 slower on first-chunk
- B vs C (bf16-only re-validation): median Δ = -62.6 ms (-1.15%) — bf16-only ~tied with fp32; reconfirms Story 18.3's empirical-null finding.

**Producer-bottleneck ratio history:** Story 18.1 baseline 3.23× → Story 18.2 fp32+TF32 1.40× → Story 18.3 bf16+TF32 1.62× (net null) → **Story 18.4 bf16+compile 0.670×** (producer emits chunks at ~1.49× real-time; Story 17.3 §4.4 underrun gaps are *structurally impossible* under `tts_compile="auto"` on Ampere+).

**Split-verdict mechanism (architecturally important).** First-chunk latency reflects talker speed (still eager under `compile_talker=False`); steady-state throughput reflects compiled codebook-predictor + decoder speed (~21× per-call faster per the standalone smoke at `_bmad-output/implementation-artifacts/18-4-qwen-compile-smoke.py`). Net user-perceived experience under `tts_compile="auto"`: audio starts ~400 ms later than fp32-eager baseline; once it starts, plays through without underrun gaps. The OQ #1 routing trigger (sub-20% A-vs-B speedup gate at line 1402 of the story's anticipated 1.5–3× warm-cache decode speedup) fired in the aggregator output, but Commander **overrode** the trigger 2026-05-11 — the OQ #1 framing measured first-chunk latency as a proxy for "did compile work?"; the actual OFR-E acceptance criterion (producer-bottleneck ratio) is met.

**NFR3 joint audition (Task 9) — FULL PASS.**

All 3 listeners auditioned (L1 = Commander; L2 + L3 = co-located in-person walkthrough listeners per the Story 17.1 protocol). 30 trials × 2 conditions = 60 defect observations.

Per-actual-mode defect-flag counts (N=30 trials per actual mode after un-blinding via truth-table):

| defect | fp32_eager | bf16_compile |
|---|---|---|
| `none` | 30 | 30 |
| `audible_seam` (← verdict gate) | **0** | **0** |
| `clipping` | 0 | 0 |
| `phase_artifact` | 0 | 0 |
| `tonal_distortion` | 0 | 0 |
| `other_describe_in_notes` | 0 | 0 |

Per-listener actual-mode preference (un-blinded via truth-table):
- L1: bf16_compile = 1 (s-015, with note *"Seemed like better quality/volume"*); fp32_eager = 0; equivalent = 9
- L2: bf16_compile = 0; fp32_eager = 0; equivalent = 10
- L3: bf16_compile = 0; fp32_eager = 0; equivalent = 10

**Full audition verdict: PASS.** Zero `audible_seam` flags on bf16_compile trials (and zero on fp32_eager — clean across all 60 observations); zero defects of any kind on either system across all 3 listeners; only non-equivalent preference (1/30 trials) favors bf16_compile. The bf16+compile+pin-bump composite is perceptually indistinguishable from fp32+eager across the 3-listener panel.

**Architectural decision: ship Story 18.4 source-tree work + compile-as-default-on-Ampere+ pending full audition closure.**

- ✓ `requirements.txt:23` + `build_tools/requirements-production.txt:61` bumped to `dffdeeq/Qwen3-TTS-streaming@3fdb4682`; `_QWEN_TTS_PIN_HASH = "3fdb4682"` (Story 17.2 cache invalidation discipline triggers cleanly on first run after the bump).
- ✓ `src/myvoice/services/tts_streaming/compile_cache.py` (7-dim cache key per D-24; per-key `%LOCALAPPDATA%/MyVoice/torch_compile_cache/` storage; H1+H2 cache-invalidation discipline mirrored from Story 17.2).
- ✓ `src/myvoice/services/tts_streaming/torch_runtime.py::engage_compile_optimizations` (8-branch reason enum; `compile_talker=False`; D-25 invariant; P-12 probe; NFR7 graceful-degradation fallback chain).
- ✓ `ModelRegistry._load_model_sync` engages compile after `from_pretrained`; INFO log line extended with `compile_engaged='deferred'` at `__init__` + `compile_engaged='True/False'` at post-load.
- ✓ `AppSettings.tts_compile` field with `auto`/`on`/`off` validation. Default **flipped to `"off"`** by Fix #4 (2026-05-10 bundled-smoke triton-on-Windows blocker; per OQ #4 routing). The compile source-tree machinery is LIVE but bypassed at runtime by the "off" default; advanced users opt-in via hand-edit of `settings.json`.
- ✓ `QwenTTSService.warmup_compile_async` (fire-and-forget; "Preparing TTS engine…" indicator; persistent-cache hit/miss/failure branches). Code-review pass added `tts_compile="off"` gate at H1 (prevents the warmup priming generation from playing audible "Hello world." to users on first launch under the "off" default).
- ✓ `audio_coordinator.stop_streaming_session(wait_for_drain=True)` extended for producer-FASTER-than-real-time regime — `max(last_chunk_remaining, total_queued_audio_s)` handles both Story 18.3 M6's producer-slower case AND Story 18.4's compile-engaged producer-faster case.
- ✓ **Full audition complete (Task 9.3/9.4):** L1 = Commander + L2 + L3 = co-located in-person walkthrough listeners per the Story 17.1 protocol. 30 trials × 2 conditions = 60 defect observations; zero `audible_seam` flags on bf16_compile trials (and zero on fp32_eager); zero defects of any kind on either system across all 3 listeners.
- **Deferred to Story 18.5:** the production-bundle packaging path. The dev-env triton-on-Windows blocker (`Cannot find a working triton installation`) is **resolved** in the dev environment (Python 3.10.11 headers + libs + CUDA Toolkit 12.8 + `pip install --no-deps triton-windows`; verified via the standalone smoke and the real-model smoke at `_bmad-output/implementation-artifacts/18-4-triton-smoke.py` and `18-4-qwen-compile-smoke.py`). The bundle-side problem is a packaging-only issue (need to ship CUDA Toolkit redistributables + Python headers + triton-windows in the PyInstaller bundle), not a fundamental compatibility problem. Story 18.5 scope.

**NFR3 status (FULLY RE-CLEARED).** The Story 17.1 audition's verdict (PASS — zero `audible_seam` flags across 30 trials on the post-Story-16.8 regenerated fixture) remains the canonical NFR3 clearance for the streaming-default ramp. Story 18.4's full audition (3 listeners × 10 utterances × A/B = 60 defect observations) **re-clears NFR3** for the bf16+compile+new-pin composite. The bf16+compile path is engaged in production *only* when the user explicitly sets `tts_compile != "off"` (the bundled-smoke default is "off" pending Story 18.5); production users running the bundle today get the pre-Story-18.4 Story 18.3 bf16-eager baseline. Once Story 18.5 ships the bundle infrastructure, flipping the default back to `tts_compile="auto"` is **architecturally pre-cleared** by this audition.

**Methodology composition.** The fp32 branch in this 3-way A/B/C is **fp32-with-TF32-engaged**, not strict-fp32 (Story 18.2's TF32 + cuDNN benchmark autotune engages at startup on every Ampere+ host regardless of precision). The bf16+compile vs fp32+eager comparison is therefore (bf16 + compile + TF32) vs (fp32 + TF32). Strict-fp32 was closed-as-null by Story 18.2 on the producer-bottleneck workload and Story 18.4 does not re-litigate.

**Methodology limitations (mirrored from Story 17.1 / 18.3 follow-up notes).** Three structural limitations apply:

1. **Single-host RTX 5090 measurement.** Captured only on Commander's RTX 5090 Blackwell dev host (capability 12.0 / GeForce variant). Earlier Ampere/Ada hosts (RTX 30xx/40xx) may show a different compile-engaged-vs-eager profile — Blackwell's tensor cores are unusually capable. The architecture amendment's data should not be cited as "compile delivers 0.670× ratio on Ampere+" generally; only as "compile delivers 0.670× ratio on Blackwell GeForce 12.0 in our V2 inference workload with the canonical Story 17.3 §4.1 step 3 long-form Sarira-F utterance."
2. **Cold-compile run discarded.** Branch A's run #1 (6154.6 ms) was discarded from the median calculation per the aggregator's cold-compile discipline at Task 8.4. The architecture's anticipated 10-30 s cold-compile budget is respected — 6.1 s observed is well within budget on Blackwell — but the cold cost is real for first-process-launch UX and the persistent compile cache (D-23) amortizes it across launches.
3. **Audition reproducibility.** Per Story 17.1's M1 reproducibility section, the audition's external reproducibility depends on the gitignored fixture remaining on Commander's filesystem; force-add of the 20 WAVs to `_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/` is the canonical cross-session retention path. The truth-table seed `"Story 18.4 joint audition:<listener_id>"` is reproducible; if the truth-table is lost, regenerate via `18-4-generate-truthtable.py`. Future maintainers re-validating bf16+compile for a future qwen-tts pin bump can re-run the audition fresh — or amend in place if the new pin is a minor patch over `3fdb4682`.

Source artifacts (✓ = git-tracked; ○ = working file under gitignored `_bmad-output/`, retained on Commander's filesystem only):

- ✓ `_bmad-output/implementation-artifacts/18-4-torch-compile-decoder-persistent-cache.md` (story document with full Tasks 1–13 closure state and Change Log entries; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-torch-compile-decoder-persistent-cache-evidence.md` (evidence file with full §"D-22 verification" + §"Pin-bump rationale" + §"P-12 probe selection" + §"Bundled smoke" (4-fix iteration) + §"NFR1 first-chunk-latency measurement (3-way A/B/C)" + §"NFR3 joint audition verdict (L1 partial PASS)" + §"Triton-on-Windows dev-env smoke" + §"Real-model compile smoke"; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-triton-smoke.py` + `18-4-qwen-compile-smoke.py` (dev-env smoke probes; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-set-precision-and-compile.py` (settings.json mutator for the 3 NFR1 branches; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-aggregate-nfr1.py` (3-way A/B/C aggregator + OFR-E gate check + OQ #1 routing surface; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-regen-fixture.py` (one-shot fixture-regen driver via `model.generate_voice_clone`; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-generate-truthtable.py` (deterministic truth-table builder; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-compute-verdict.py` (verdict computation cross-referencing truth-table; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-l1-audition-helper.py` (audition helper — adapted from 17-1 / 18-3; force-added).
- ✓ `05_Story_18.4_NFR1_BF16_COMPILE.bat` + `06_Story_18.4_NFR1_BF16_EAGER.bat` + `07_Story_18.4_NFR1_FP32_EAGER.bat` (3-way NFR1 harness; CRLF; paren-trap free; force-added).
- ✓ `08_Story_18.4_NFR3_Audition.bat` (Commander-facing audition launcher; mirrors `01_Run_MyVoice_With_CSV_Capture.bat` structure; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-rtx5090-bf16-compile.csv` + `18-4-rtx5090-bf16-eager.csv` + `18-4-rtx5090-fp32-eager.csv` (consolidated N=10 cold-start `first_chunk_latency_ms`; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-rtx5090-bf16-compile-run<NN>.csv` + `-bf16-eager-run<NN>.csv` + `-fp32-eager-run<NN>.csv` (30 per-run CSVs; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-perceptual-fixtures/` (20 paired WAVs + `_perlistener_truthtable.json` + `LISTENING-INSTRUCTIONS.md`; force-added).
- ✓ `_bmad-output/implementation-artifacts/18-4-bf16-compile-pinbump-audition.csv` (30 rows = L1 + L2 + L3 × 10 utterances; force-added).

### Implementation Readiness Validation — with surfaced gaps

Below are issues found during the audit that warrant explicit resolution before tech-spec.

**Critical (block tech-spec without resolution):**

None. All blocking decisions made in Step 4.

**Important (would cause AI-agent inconsistency if left unresolved):**

1. **Focal session semantics for the indicator (P-3 references `session.is_focal_for_indicator()` without definition).**
   *Resolution:* Define formally — *focal* is, in priority order: (a) the session currently in `PLAYING`; (b) if none, the session most recently in `GENERATING` or `READY_TO_PLAY`; (c) otherwise, the most recent terminal session (DONE/CANCELLED/ERROR) within a 5-second decay window; (d) otherwise, none. Implement as a property on `SessionRegistry.focal_session_id`, recomputed on every state change. `current_session_changed` fires on (a) → (b) → (c) → none transitions.

2. **Save during streaming with mid-playback cancellation.**
   *Resolution:* The Save button is enabled only when `saveable_session_id is not None`. If user clicks Save while the saveable session is still streaming (not yet finalized), the dialog opens, file path captured; on user confirm, the WAV write happens after `finalize()` fires (UI shows "Finalizing…" indicator). If the user cancels the *generation* before finalize, the save attempt is aborted with a transient error toast; the save dialog closes. Codified in the `save_dialog.py` flow.

3. **Cancel-during-playback with TRUE_STREAM (session is generating AND audible).**
   *Resolution:* Per P-7, `session.cancel()` sets the event. Two follow-on actions are required and not yet specified — adding them now:
   - **(i) Stop the active playback task.** The registry, on session cancel, calls `audio_coordinator.cancel_playback(session_id)` which stops the monitor + virtual-mic dual-stream tasks gracefully (existing AudioCoordinator API).
   - **(ii) Discard buffered chunks.** The decoder worker drain-on-cancel drops all queued chunks rather than dispatching them.
   The session transitions PLAYING → CANCELLED → DISCARDED in one tick; audio fades cleanly (no abrupt cut from mid-buffer playout) using the existing AudioCoordinator stop semantics.

4. **`current_session_changed` firing conditions are vague in D-13.**
   *Resolution:* Tightened to fire on `focal_session_id` changes per the resolution above. Contract: emits the new focal session ID (or `None`); subscribers re-read state synchronously.

5. **Session creation ownership.**
   *Resolution:* `QwenTTSService.generate()` is the *creator* — it calls `registry.create_session(text, voice, model_type, source=GENERATED)` and receives the session ID. UI does not create sessions; it reads them via the registry. This is consistent with the import-rule table (services may import registry; UI imports registry read-only).

**Nice-to-have (deferable):**

1. **Save button visual hint during streaming.** A small "(streaming…)" suffix in the button tooltip when the saveable session has `is_streaming=True` would prevent user surprise at the "Finalizing…" toast. Defer to UX review.

2. **Telemetry sink choice.** `metrics.record()` writes to the existing logger today. Whether to pipe metrics to a structured-log file separate from app logs is a future operations concern, not architecture.

3. **vLLM-Omni periodic check.** Quarterly: revisit upstream status. If online serving lands, the streaming POC could simplify dramatically.

### Gap analysis summary

| Severity | Count | Resolution status |
|---|---|---|
| Critical | 0 | — |
| Important | 5 | Resolved inline above; integrated into the document |
| Nice-to-have | 3 | Documented; not blocking |

### Architecture Completeness Checklist

**Requirements analysis**
- [x] Project context thoroughly analyzed (Step 2)
- [x] Scale and complexity assessed (medium-high)
- [x] Technical constraints identified (V2 inheritance + 3 new constraints)
- [x] Cross-cutting concerns mapped (9 items)

**Architectural decisions**
- [x] Critical decisions documented (D-1 through D-20)
- [x] Versions verified (qwen-tts pin to be captured at implementation; all other deps inherited from V2)
- [x] Integration patterns defined (P-1 through P-9)
- [x] Performance considerations addressed (NFR1 hardware-conditional, NFR3 A/B gate, NFR11 memory hygiene)

**Implementation patterns**
- [x] Naming conventions inherited from V2 (no override)
- [x] State machine pattern (P-1, P-2)
- [x] Threading discipline (P-3)
- [x] Signal contracts (P-4)
- [x] Streamer/decoder contracts (P-5, P-6)
- [x] Cancellation pattern (P-7)
- [x] Queue invariants (P-8)
- [x] Telemetry format (P-9)

**Project structure**
- [x] New & modified file map complete
- [x] Module boundaries with explicit import rules
- [x] Requirements → structure mapping complete
- [x] Test additions mapped to phases
- [x] Migration order matches D-20

**Validation resolutions integrated**
- [x] Focal session semantics defined
- [x] Save-during-streaming flow specified
- [x] Cancel-during-playback two-step action specified
- [x] `current_session_changed` firing conditions tightened
- [x] Session creation ownership specified

### Architecture Readiness Assessment

**Overall status:** READY FOR TECH-SPEC.

**Confidence level:** High for Phases 1–5 (foundation through features). Medium for Phase ⊥ (streaming) — the only meaningful uncertainty is empirical: GPU stream concurrency overhead and overlap-add seam quality must be measured during the POC before flipping the default. Architecture defines the *fallback path* (D-9, NFR7) so even an unfavorable POC outcome doesn't strand the work.

**Key strengths:**

- **Single-chokepoint discipline.** Three patterns (`_transition_to`, `post_mutation`, `metrics.record`) make state changes, threading, and telemetry uniformly enforceable.
- **Phased migration** (D-20) — each phase ships independently, reverts cleanly, and has a defined revert path.
- **Backward-compatible signal layer.** No existing UI subscriber breaks during foundation; deprecation is gradual.
- **Streaming is a parallel track**, not a hard dependency. Phases 1–5 land regardless of POC outcome.
- **Lifecycle policy fully specifies the ambiguous cases** Mary surfaced in the briefs (saveable slot, PRELOADED clones, supersession ordering, Playback Last preservation).

**Areas for future enhancement:**

- Quantization (int8/fp8) of talker weights — composes with streaming for the lowest-end hardware.
- Multi-history save (save older than most recent) — natural extension once the save-pipeline lands.
- Soundboard / multi-button Clear Comms — extension of the PRELOADED-source mechanism.
- Custom virtual mic driver (PRD aspirational) — orthogonal to this pass.

### Implementation Handoff

**AI-agent guidelines:**

- Start from this document. Read alongside the parent `architecture.md` (sealed V2) and the three optimization-pass briefs in `_bmad-output/optimization-pass/`.
- Implementation order is **D-20 phased migration**, not "all at once."
- Every new state mutation flows through `_transition_to` (P-2). Every cross-thread call flows through `post_mutation` (P-3). No exceptions.
- Touching private `qwen_tts` symbols requires updating `tests/test_qwen_tts_internals.py` in the same change (D-12).
- Per-feature tech-specs are owned by **Bob (Scrum Master)** for story preparation and **Barry (Quick Flow)** or **Amelia (Dev)** for tech-spec authoring.

**First implementation priority:**

Phase 1 — Foundation. Specifically: create `services/sessions/generation_session.py` with `GenerationSession`, `SessionState`, `SessionSource`, `InvalidSessionStateError`, and `_VALID_TRANSITIONS`; covered by `tests/unit/services/sessions/test_generation_session.py`. Net-zero behavior change to the running app. Ship it, then proceed to `session_registry.py`.
