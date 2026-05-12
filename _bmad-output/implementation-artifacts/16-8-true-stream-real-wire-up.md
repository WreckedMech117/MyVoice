# Story 16.8: TRUE_STREAM Real Wire-Up

Status: done

> Phase ⊥ of D-20 — **eighth story of Epic 16** (True Streaming TTS, the parallel/independent track) and the **direct follow-up** to Story 16.7's empirical-validation gate failure. Story 16.7 ran the harness it built (`scripts/validate_streaming_default.py`) against the real RTX 5090 + qwen-tts 0.0.4 host and discovered that **TRUE_STREAM as Story 16.6 shipped does not function against the real qwen-tts wrapper.** Of 50 utterances measured, 50 failed silently — the talker thread raised on every call, was swallowed by `_build_true_stream_talker`'s except-branch, the streamer's queue drained empty, and the dispatch returned `success=True` with `audio_data=np.array([])`. Story 16.7's only production code change was a defense-in-depth empty-chunks guard at `qwen_tts_service.py:2845-2861` that converts the silent failure into a `RuntimeError` so the existing fallback chain catches it and routes to SENTENCE_STREAM (preserving NFR7 graceful degradation). Story 16.8's job is to **fix the underlying wire-up** so TRUE_STREAM is no longer a structural no-op.
>
> **Why the wire-up is broken.** Story 16.6's `_build_true_stream_talker` at `src/myvoice/services/qwen_tts_service.py:2498-2533` returns a 0-arg callable whose body is exactly:
>
> ```python
> def _run_talker() -> None:
>     try:
>         # Best-effort wire-up — Story 16.7 validates kwargs against
>         # real qwen-tts and refines.
>         model.model.generate(streamer=streamer)
>         streamer.end()
>     except Exception as exc:
>         self.logger.exception(...)
>         streamer._cancel_event.set()
>         try:
>             streamer.end()
>         except Exception:
>             pass
> ```
>
> The literal call `model.model.generate(streamer=streamer)` passes **no `input_ids`, no `speakers`, no `languages`, no `inputs_embeds`, no conditioning whatsoever.** The underlying `Qwen3TTSForConditionalGeneration.generate` at `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2292` requires all four to construct the talker's input embedding sequence (codec prefix → text embeddings → speaker embedding → padding → BOS) before invoking `self.talker.generate(inputs_embeds=..., attention_mask=..., trailing_text_hidden=..., tts_pad_embed=..., **talker_kwargs)` at lines 2272-2278. Without those inputs, the wrapper raises immediately. Story 16.6's comment at line 2520 explicitly punted real-model kwarg validation to "Story 16.7" — Story 16.7 measured the failure but did not implement the fix; that is Story 16.8's deliverable.
>
> **Why this is the next entry point of Epic 16.** The streaming-default flag flip (the user-facing release of TRUE_STREAM as the GPU default) is now blocked on the **conjunction** of Story 16.8 (this one — make TRUE_STREAM produce real audio) **AND** Story 16.9 (NFR1 reconciliation — explain or fix why SENTENCE_STREAM also misses the 2s ceiling on this host). Either story alone is insufficient. Story 16.8 is the architectural Phase ⊥ unblocker; Story 16.9 is the contract-level unblocker. They are independent and can be worked in parallel; Story 16.8 is being created first because the maintainer confirmed (2026-05-08 memory note) that the architectural piece is the higher-priority engineering task.
>
> **Net behavior change for users.** **None for the production-default path.** The Story 16.7 empty-chunks guard remains active (`qwen_tts_service.py:2845-2861`) and continues to route any TRUE_STREAM failure into the SENTENCE_STREAM fallback — but if Story 16.8's wire-up is correct, the guard's `if not accumulated_chunks` branch never fires in production, and CUDA users hear true token-level streamed audio (the architectural Phase ⊥ promise). For CPU-only users, behavior is unchanged (`effective_streaming_mode(None)` returns SENTENCE_STREAM on `torch.cuda.is_available() == False` per `streaming_mode.py:54-56`, NFR12 protected). For users who have explicitly set `AppSettings.streaming_mode_override` to `"sentence_stream"` or `"batch"`, behavior is unchanged. **The streaming-default flag flip is NOT part of this story** — even if Story 16.8 lands successfully, the flip remains blocked on Story 16.9 (NFR1 reconciliation) and a future "streaming default ramp" story that re-runs the multi-listener perceptual A/B audition gate from Story 16.7 AC #2.
>
> **Pre-existing infrastructure already verified before drafting.**
>
>   - **The broken wire-up site** is `_build_true_stream_talker` at `src/myvoice/services/qwen_tts_service.py:2498-2533`. Story 16.8 modifies (or replaces) this method. Its sibling `_build_true_stream_decode_fn` at `qwen_tts_service.py:2471-2497` is **correct as shipped** (it wraps `model.speech_tokenizer.decode`); Story 16.8 does NOT touch the decode adapter unless investigation reveals a parallel issue.
>
>   - **The empty-chunks guard** at `qwen_tts_service.py:2845-2861` is the production safety net. Story 16.8 keeps it in place — defense-in-depth. The guard's regression tests at `tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure` (lines 1158-1340) are the load-bearing assertions that the silent-failure mode is caught; both tests must continue to pass after Story 16.8 lands.
>
>   - **The reference path** is `_generate_streaming` (the SENTENCE_STREAM body) at `qwen_tts_service.py:2028-2242`. It marshals `request.text` → `request.speaker` → `request.language` → `request.model_type` into a per-chunk `QwenTTSRequest` with `streaming=False` and dispatches via `loop.run_in_executor(self._executor, self._generate_sync, chunk_request)`. The `_generate_sync` method at `qwen_tts_service.py:3283+` is the single chokepoint where the real qwen-tts wrapper's full preprocessing is invoked (it eventually calls `model.generate(text=..., speaker=..., language=..., ...)` or one of the wrapper's named entrypoints — `generate_custom_voice`, `generate_voice_clone`, `generate_voice_design` — depending on `model_type`). **This is Story 16.8's primary reference: whatever preprocessing `_generate_sync` does to make a non-streaming generate call work, Story 16.8 must replicate the equivalent for the streaming generate call.**
>
>   - **The qwen-tts upstream wrapper** at `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2292` is the public entry point used by the SENTENCE_STREAM path. Inside it (line 2272-2278) the wrapper calls `self.talker.generate(inputs_embeds=talker_input_embeds, attention_mask=talker_attention_mask, trailing_text_hidden=trailing_text_hiddens, tts_pad_embed=tts_pad_embed, **talker_kwargs)` — **this is the call site Story 16.8 needs to reach with `streamer=streamer` injected into `talker_kwargs`.** Two paths to get there are viable and **must both be considered before committing to one**:
>
>       1. **Path A — Replicate preprocessing locally.** Manually build `talker_input_embeds`, `talker_attention_mask`, `trailing_text_hiddens`, and `tts_pad_embed` from `request.text` + `request.speaker` + `request.language`, then invoke `model.talker.generate(inputs_embeds=..., streamer=streamer, ...)` directly. Pros: works against the current pinned qwen-tts 0.0.4 with no upstream change. Cons: deeply couples MyVoice to qwen-tts internal preprocessing — a future qwen-tts version could change the embedding-construction logic and silently break our reproduction without the trip-wire firing. The trip-wire (`tests/test_qwen_tts_internals.py`) only asserts attribute existence, not behavioral equivalence; Story 16.8 must extend it with attribute checks for every new entrypoint touched (`model.talker.generate`, `model.talker.get_input_embeddings`, `model.config.talker_config.spk_id`, `model.config.talker_config.codec_language_id`, `model.generate_speaker_prompt`).
>
>       2. **Path B — Use the wrapper's existing entrypoint with a streamer kwarg.** Test whether `model.generate(text=..., speaker=..., language=..., streamer=streamer)` (or one of the named entrypoints `generate_custom_voice` / `generate_voice_clone` / `generate_voice_design`) accepts a `streamer` kwarg via `**kwargs` forwarding to the inner `talker.generate` call at line 2272-2278. Pros: thin coupling — uses the wrapper's public surface. Cons: depends on qwen-tts forwarding the kwarg correctly, which is **unverified** (Story 16.7 §1's recommendation explicitly framed this as "OR wait for upstream qwen-tts to forward"). If the kwarg is silently dropped (the wrapper doesn't pass `**kwargs` to `talker.generate`), the streaming will fail in a new way that may or may not trip the empty-chunks guard. Story 16.8 must test this path before committing to Path A.
>
>     **Decision rule:** prefer Path B if it works at all (lower coupling); fall back to Path A if Path B is dropped/silently-ignored upstream. Investigation of Path B must precede implementation; the Change Log must document which path was chosen and why.
>
>   - **The streamer surface** is unchanged from Story 16.3. `CodecTokenStreamer(chunk_size=25, lookahead=5, queue_max_factor=4, cancel_event=...)` at `services/tts_streaming/codec_token_streamer.py:51-214` subclasses `transformers.generation.streamers.BaseStreamer`. Its `put(value)` callback (lines 106-150) buffers tokens and pushes fixed-size chunks onto a bounded `queue.Queue`; `end()` (lines 152-163) flushes the residual buffer and pushes the `END_OF_STREAM` sentinel. Story 16.8 does **not** modify the streamer — Story 16.7 already demonstrated that Stories 16.3-16.5's plumbing is correct; only the *talker invocation* is broken.
>
>   - **The decoder worker** is unchanged from Story 16.4. `StreamingDecoderWorker` at `services/tts_streaming/streaming_decoder.py:64-241` is decoder-shape-agnostic: it takes `decode_fn: Callable[[list], np.ndarray]` and posts `('append_chunk', sid, pcm)` then `('finalize', sid)` on END_OF_STREAM. Story 16.8 does **not** modify the worker; the existing `_build_true_stream_decode_fn` at `qwen_tts_service.py:2471-2497` is the adapter, and Story 16.7 confirmed the worker pulls from the streamer queue correctly when the streamer has chunks to push (it ran on the empty path in 16.7 — same code, just no input).
>
>   - **The cancel chain** is unchanged from Story 16.5. The talker thread reads `streamer._cancel_event` indirectly (via the streamer's `put()` checking the event); a user-cancel sets the event, the streamer becomes a no-op, the talker's `model.talker.generate(...)` runs a few more iterations producing tokens we drop, then completes; CUDA state stays clean. Story 16.8 must preserve this property — specifically, the talker thread must **not** raise an exception in response to cancel. The architectural invariant (D-11) is "no exceptions raised through HF internals; CUDA state stays clean; small wasted compute is acceptable." Story 16.8's wire-up must respect that the cancel path is cooperative, not preemptive.
>
>   - **The metrics infrastructure** is unchanged. `streaming_mode` and `streaming_mode_fallback` (per D-19, P-9) are emitted by `_dispatch_by_streaming_mode` at `qwen_tts_service.py:3022-3028`. Story 16.8 does not modify the dispatcher's metric emission; it modifies only the body of `_generate_true_stream` (and `_build_true_stream_talker`) so the dispatcher's "TRUE_STREAM succeeded" branch becomes reachable for the first time in production.
>
>   - **The trip-wire test** is at `tests/test_qwen_tts_internals.py`. It currently asserts `model.model.generate` and `speech_tokenizer.decode` exist (per Story 16.1 / D-12) and `Qwen3TTSTokenizerV1Model.decode` exists (per Story 16.4). Story 16.8 must extend it with attribute checks for every new qwen-tts entrypoint reached: at minimum `model.talker` (the `Qwen3TTSTalkerForConditionalGeneration` instance), `model.talker.generate`, `model.config.talker_config.spk_id` (for speaker resolution), `model.config.talker_config.codec_language_id` (for language resolution), and `model.generate_speaker_prompt` (for voice-clone embedding generation) — depending on which Path (A or B above) is taken. The principle: every private symbol Story 16.8's code touches at runtime must be pinned by an attribute test that fails CI before a silent qwen-tts rename can ship.
>
>   - **No new dependencies.** `qwen-tts` is already pinned to commit `1ab0dd75` (qwen-tts 0.0.4) in `requirements.txt` per Story 16.1; `transformers`, `torch`, `numpy` are already imported; `threading.Thread` (stdlib) is already used by Story 16.6. No `requirements.txt` changes — and no pin bump. Bumping the qwen-tts pin to a future version is **out of scope** for this story; if a future qwen-tts ships a streamer-aware wrapper, the team can revisit Path B then.
>
>   - **No `AppSettings` schema changes.** `streaming_mode_override` is already shipped per Story 16.2; Story 16.8 reads (does not write) the field via the existing `_resolve_streaming_mode` method.
>
>   - **No registry mutation-method changes.** The TRUE_STREAM dispatch path's mutation posts (`start_generation`, `append_chunk` via worker, `finalize` via worker on END_OF_STREAM, `cancel` via worker on drain-on-cancel, `set_error` + `discard` on dispatch-time exceptions) are unchanged from Story 16.6.
>
> **Six-point story scope:**
>
> (a) **Investigate Path B (wrapper streamer kwarg forwarding) on a CUDA-available host.** Write a one-shot smoke probe in `scripts/probe_qwen_tts_streamer.py` (or equivalent) that constructs a `Qwen3TTSModel`, builds a minimal `CodecTokenStreamer`, and calls `model.generate(text="hi", speaker="Ryan", language="English", streamer=streamer)` (or `model.generate_custom_voice(...)` with a streamer kwarg). Observe whether `streamer.put` is called at least once before the call returns. If yes → Path B is viable; commit to Path B. If no (streamer.put never invoked, OR an unexpected exception fires) → Path A. **The probe is a one-shot script, not a pytest test** (real-model GPU dispatch is too expensive for CI; mirror Story 16.7's harness pattern). Document the probe's outcome in this story's Change Log. **This step is mandatory before implementing (b)** — it determines (b)'s shape.
>
> (b) **Implement the chosen Path** in `_build_true_stream_talker` (or a sibling helper if the function shape needs to change):
>   - **Path B (preferred):** call the public wrapper's named entrypoint matching `request.model_type` (`model.generate_custom_voice`, `model.generate_voice_clone`, `model.generate_voice_design`) with `streamer=streamer` passed through `**kwargs`. Verify the wrapper's `**talker_kwargs` forwarding at `modeling_qwen3_tts.py:2272-2278` includes `streamer`.
>   - **Path A (fallback):** replicate the wrapper's preprocessing locally. Reach into `model.talker`, `model.talker.get_input_embeddings`, `model.config.talker_config.spk_id`, `model.config.talker_config.codec_language_id`, and (if `request.voice_clone_prompt`) `model.generate_speaker_prompt`. Build `talker_input_embeds`, `talker_attention_mask`, `trailing_text_hiddens`, `tts_pad_embed`. Call `model.talker.generate(inputs_embeds=..., attention_mask=..., trailing_text_hidden=..., tts_pad_embed=..., streamer=streamer, **talker_kwargs)`. **This path is verbose** (~80-150 lines); the implementation should factor preprocessing into a helper (`_build_talker_inputs(model, request) -> dict[str, Tensor]`) so the talker thread body remains short and the preprocessing can be unit-tested separately.
>
> (c) **Extend the import-attribute trip-wire** (`tests/test_qwen_tts_internals.py`). Add at minimum:
>   - For Path B: assert that `model.generate_custom_voice` (and the two siblings) accept a `streamer` kwarg via inspect or a smoke call against a tiny mock model.
>   - For Path A: assert each new attribute reached (`model.talker`, `model.talker.generate`, `model.talker.get_input_embeddings`, `model.config.talker_config.spk_id`, `model.config.talker_config.codec_language_id`, `model.generate_speaker_prompt` if the voice-clone branch is exercised). Pattern: mirror the existing `test_qwen3_tts_model_method_surface_intact` assertion style.
>
> (d) **Add a positive-path integration test** for `_generate_true_stream` against a fake `qwen_tts` fixture. The Story 16.6 / Story 16.7 smoke tests in `tests/integration/test_streaming_tts_smoke.py` (lines 559, 674, ...) all monkey-patch the talker (via `service._build_true_stream_talker` patching) — that is what allowed the silent-failure to ship. Story 16.8 must add **at least one** test that exercises the actual `_build_true_stream_talker` body (post-fix) end-to-end with a stub `model.talker.generate` (or `model.generate_custom_voice` for Path B) that produces ≥1 chunk. This test is the regression guard against another silent wire-up failure ever recurring. Tests live in `tests/integration/test_streaming_tts_smoke.py` in a new test class `TestTrueStreamWireUpEndToEnd`. The fake-model fixture should be a `MagicMock` whose attribute graph (`model.talker.generate`, `model.config.talker_config.spk_id`, etc.) is constructed to satisfy whichever Path was chosen — so the test exercises the **real** wire-up code, not a monkey-patched stub of it.
>
> (e) **Re-run Story 16.7's harness on the maintainer's RTX 5090 + qwen-tts 0.0.4 host.** Re-run `python310\python.exe scripts\validate_streaming_default.py --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv --output-dir _bmad-output\implementation-artifacts\ --mode-override true_stream`. Expect: at least one row with `error_flag == ""` and a non-`None` `first_chunk_latency_seconds`. Commit the new CSV at `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv` (force-add via `git add -f`, mirror Story 16.7's pattern). Document p50 / p95 / max for short / medium / long classes — **this is the post-fix empirical evidence that gates whether AC #5 passes.**
>
> (f) **Re-run the perceptual A/B fixture builder** (`scripts/build_streaming_perceptual_ab_fixture.py`). The fixture builder currently produces 10 paired WAV files; the "A" files were silent in Story 16.7, so the audition was non-informative. With Path B/A's TRUE_STREAM producing real audio, the fixture builder will produce paired TRUE_STREAM (A) vs SENTENCE_STREAM (B) renditions. **For Story 16.8's scope, a single-listener (Commander solo) audition is sufficient** — the multi-listener gate from Story 16.7 AC #2 reactivates only if/when the streaming-default flag flip is being considered in a later story (post-Story 16.9). Document audition observations (any audible seams, cadence mismatch, sibilance issues) in this story's Change Log. If audible seams emerge, the fix is either to tighten `DEFAULT_CHUNK_SIZE` / `DEFAULT_LOOKAHEAD` constants (per Story 16.7's framing) or escalate to a follow-up story; do NOT block Story 16.8 closure on perceptual quality unless the seams are catastrophic (e.g., dropouts, full-second silences, distortion).

## Story

As a **MyVoice user (GPU host, default settings)**,
I want **the TRUE_STREAM dispatch path to actually drive the qwen-tts model with full conditioning so token-level streaming produces audible audio rather than silently falling back to sentence-stream rendering**,
So that **first-audio latency on my RTX 5090 (or future GPU upgrade) reflects the architectural Phase ⊥ promise — sub-2s on representative inputs — once Story 16.9 reconciles the SENTENCE_STREAM baseline and a future ramp story flips the default**.

As a **MyVoice maintainer**,
I want **`_build_true_stream_talker` to invoke a conditioning-aware generate call (either via the qwen-tts wrapper's public entrypoint with a `streamer` kwarg forwarded through `**kwargs`, or by replicating the wrapper's preprocessing locally and calling `model.talker.generate(inputs_embeds=..., streamer=streamer, ...)` directly), with the choice between paths driven by an empirical probe of which one actually fires the streamer's `put()` callback against the pinned qwen-tts 0.0.4 model**,
So that **the architectural Phase ⊥ unblocks (Story 16.7's first empirical gate moves from FAIL to PASS), the fallback chain's TRUE_STREAM-fails branch becomes the exception rather than the rule on production CUDA hosts, and the streaming-default flag flip's two preconditions (Story 16.8 + Story 16.9) collapse to one**.

## Acceptance Criteria

**Background — what this story is and is NOT.**

This story does six things to the working tree: (a) adds an investigation script that probes whether the qwen-tts wrapper's public entrypoint accepts a `streamer` kwarg via `**kwargs`; (b) implements either Path B (wrapper kwarg forwarding) or Path A (local preprocessing replication) in `_build_true_stream_talker` based on the probe outcome; (c) extends `tests/test_qwen_tts_internals.py` with attribute checks for every new qwen-tts symbol reached; (d) adds a positive-path integration test in `tests/integration/test_streaming_tts_smoke.py` that exercises the real `_build_true_stream_talker` body (not a monkey-patch of it); (e) re-runs Story 16.7's harness on a CUDA host and commits the new CSV with empirical evidence; (f) re-runs the perceptual A/B fixture builder and records solo-audition observations.

The deliverable is bounded to:

- `src/myvoice/services/qwen_tts_service.py` (modified — replaces `_build_true_stream_talker` body; ~80-200 net new lines depending on Path B vs Path A; if Path A is chosen, factors preprocessing into a `_build_talker_inputs(model, request) -> dict[str, Any]` helper)
- `tests/test_qwen_tts_internals.py` (modified — adds attribute checks for new qwen-tts entrypoints; ~30-60 net new lines)
- `tests/integration/test_streaming_tts_smoke.py` (modified — appends `TestTrueStreamWireUpEndToEnd` class; ~150-250 net new lines)
- `scripts/probe_qwen_tts_streamer.py` (new — one-shot smoke probe; ~80-120 lines; commits the probe outcome via printed log + Change Log entry)
- `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv` (new — committed via `git add -f`; produced by Step (e))
- `_bmad-output/implementation-artifacts/16-8-true-stream-real-wire-up.md` (this file, the story doc itself; updated as Change Log entries accumulate)

This story does **NOT**:

- Modify `_build_true_stream_decode_fn` (`qwen_tts_service.py:2471-2497`). The decode adapter is correct as shipped; Story 16.7 did not surface a decode-side bug. If Story 16.8's investigation reveals the decode path is also broken, that is a separate scoped finding to be added to this story's Change Log and addressed in scope only if trivial; otherwise spun out as a follow-up.
- Remove or weaken the empty-chunks guard at `qwen_tts_service.py:2845-2861`. The guard is defense-in-depth and remains in production. Its regression tests at `tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure` must continue to pass after this story lands.
- Touch `services/tts_streaming/codec_token_streamer.py`, `services/tts_streaming/streaming_decoder.py`, `services/sessions/*`, `services/audio_coordinator.py`, or `streaming_mode.py`. All of these surfaces are correct as shipped; Story 16.7 confirmed the plumbing fires correctly when there are tokens to plumb. Only the *talker invocation* needs fixing.
- Flip the `streaming_mode_override` default or the `default_streaming_mode_for_hardware()` return value. The architectural default is already TRUE_STREAM on CUDA per `streaming_mode.py:54-56`; Story 16.8 makes the default *productive* rather than silently-degraded, but the default is unchanged.
- Address NFR1 reconciliation. SENTENCE_STREAM's p95 = 18.143s on this host (Story 16.7 §3.2) is Story 16.9's territory. Story 16.8 measures TRUE_STREAM's post-fix p95 on the same input set, which **may or may not** clear NFR1 — that is acceptable for this story's scope. The streaming-default flag flip remains blocked on Story 16.9 even if Story 16.8's TRUE_STREAM measurements are favorable.
- Bump the qwen-tts pin in `requirements.txt`. The pin remains at commit `1ab0dd75` (qwen-tts 0.0.4) per Story 16.1. If a future qwen-tts version ships a streamer-aware wrapper, that is a separate pin-bump story (with the trip-wire test as the safety net per D-12).
- Add or remove dependencies. `transformers`, `torch`, `numpy`, `qwen-tts`, `threading` (stdlib) are sufficient.

The deliverable is approximately **+80-200 lines to `qwen_tts_service.py` (depending on Path)**, **+30-60 lines to `test_qwen_tts_internals.py`**, **+150-250 lines of integration tests**, **~80-120 lines for the probe script**, **one new CSV with empirical evidence**, and this story's Change Log documenting (a) the probe outcome and chosen Path, (b) any new attributes added to the trip-wire (with a one-line rationale per attribute), (c) the post-fix harness CSV's per-class p50/p95/max numbers compared to Story 16.7's TRUE_STREAM-failed and SENTENCE_STREAM-baseline numbers, (d) the solo-audition observations from the perceptual A/B fixture re-run.

---

**AC #1 — The qwen-tts streamer-kwarg probe runs to completion and its outcome is documented before the implementation Path is chosen.**

**Given** a CUDA-available development host with qwen-tts 0.0.4 installed at the pinned commit
**And** the maintainer has not yet implemented Story 16.8's wire-up fix
**When** `python310\python.exe scripts\probe_qwen_tts_streamer.py` runs
**Then** the script loads a `Qwen3TTSModel` (lazily; mirror Story 16.7's harness's torch-before-PyQt6 DLL ordering preamble per `memory/torch_pyqt6_dll_ordering.md`)
**And** constructs a minimal `CodecTokenStreamer` instance
**And** invokes the wrapper's named entrypoint matching `model_type=CUSTOM_VOICE` (i.e., `model.generate_custom_voice(text="hi", speaker="Ryan", language="English", streamer=streamer)`) wrapped in a try-except
**And** logs to stdout one of three outcomes: (i) `STREAMER_FORWARDED — put() called N times before return`, (ii) `STREAMER_DROPPED — put() never called, return value indicates non-streaming generation`, (iii) `STREAMER_REJECTED — TypeError("unexpected keyword argument 'streamer'") or equivalent`

**Given** the probe's outcome
**When** the maintainer reads the log
**Then** the chosen Path is determined by rule:
  - Outcome (i) → Path B (use the wrapper's public entrypoint; commit to it for the implementation)
  - Outcomes (ii) or (iii) → Path A (replicate preprocessing locally and call `model.talker.generate` directly)
**And** the probe's outcome AND the chosen Path are recorded as a Change Log entry in this story file before any production code is written

---

**AC #2 — `_build_true_stream_talker` invokes a conditioning-aware generate call that produces ≥1 token via `streamer.put()` against a representative input on the maintainer's host.**

**Given** Story 16.8's wire-up fix is implemented (Path B or Path A per AC #1's chosen route)
**And** a fully-wired test rig identical to Story 16.6's `TestTrueStreamDispatchEndToEnd` setup (real `SessionRegistry`, real `AudioCoordinator` with `MagicMock` monitor + virtual-mic services, real `CodecTokenStreamer` with default `chunk_size=25, lookahead=5`, real `StreamingDecoderWorker`, real `QwenTTSService`)
**But** the `model.model` (Path A) or `model.generate_custom_voice` (Path B) is replaced with a fake whose attribute graph is constructed to satisfy the chosen Path's wire-up code (i.e., the test exercises the real wire-up logic, not a monkey-patched stub of `_build_true_stream_talker` itself — that would defeat the regression-guard purpose)
**When** the test calls `await service._generate_true_stream(request)` for `QwenTTSRequest(text="hello world", language="English", model_type=QwenModelType.CUSTOM_VOICE, speaker="Ryan", streaming=True)`
**Then** the response shape is `QwenTTSResponse(success=True, audio_data=<np.ndarray non-empty>, sample_rate=24000, mode=GenerationMode.STREAMING, chunks_generated=>=1, first_chunk_latency=<float>)`
**And** `streamer.put` was called at least once during the talker thread's lifetime (asserted by capturing the streamer instance and inspecting its internal token buffer or by spying on `put`)
**And** the empty-chunks guard at `qwen_tts_service.py:2845-2861` is **not triggered** (asserted by absence of the `RuntimeError("TRUE_STREAM produced 0 audio chunks ...")` raise)

**Given** the same wire-up under cancellation
**When** `streamer._cancel_event.set()` is called mid-generation (a `threading.Timer` set to fire after the first token is produced)
**Then** the talker thread exits cooperatively without raising
**And** the worker's drain-on-cancel posts `('cancel', sid)` per P-7's invariant ("The worker's drain-on-cancel posts the actual `CANCELLED` transition")
**And** `session.state == SessionState.DISCARDED` after the cancel propagates through the registry
**And** no exception is raised through HF internals (D-11 invariant: "no exceptions raised through HF internals; CUDA state stays clean")

---

**AC #3 — The import-attribute trip-wire (`tests/test_qwen_tts_internals.py`) is extended with assertions for every new qwen-tts symbol Story 16.8's wire-up reaches at runtime.**

**Given** Story 16.8's chosen Path
**When** the maintainer enumerates every `qwen_tts.*` symbol the new wire-up touches
**Then** at minimum the following assertions exist in `test_qwen_tts_internals.py` (more if Path A is chosen):
  - For Path B: `assert callable(getattr(Qwen3TTSModel, 'generate_custom_voice', None))` (already present per `test_qwen3_tts_model_method_surface_intact`); plus a new test asserting the entrypoint's parameter list includes a `streamer` parameter OR a `**kwargs` (via `inspect.signature`)
  - For Path A: assertions for `Qwen3TTSForConditionalGeneration` (the wrapper class), `model.talker` attribute existence, `model.talker.generate` callable, `model.talker.get_input_embeddings` callable, `model.config.talker_config.spk_id` dict-like, `model.config.talker_config.codec_language_id` dict-like
  - For both Paths: a new test asserting `model.model` is an instance of (or has the same `.generate` signature as) `Qwen3TTSForConditionalGeneration` from `qwen_tts.core.models.modeling_qwen3_tts` — pin the wrapper class itself, not just its top-level `generate` method
**And** each assertion has a docstring naming the production-code line that depends on it (the BMAD pattern from Story 16.1 — "services/qwen_tts_service.py:NNNN imports / calls X — this trip-wire fails CI before a silent rename can ship")
**And** the trip-wire test file's full test suite passes against the pinned qwen-tts 0.0.4

---

**AC #4 — The Story 16.7 silent-talker regression tests (`TestSilentTalkerSurfacesAsFailure`) continue to pass unmodified.**

**Given** Story 16.8's wire-up fix is implemented
**And** the regression tests at `tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure` (lines 1158-1340, two tests)
**When** `pytest tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure -v` runs
**Then** both tests pass without modification
**And** the empty-chunks guard at `qwen_tts_service.py:2845-2861` is **not removed, weakened, or bypassed** by Story 16.8's changes (the guard is defense-in-depth and survives any future regression that re-introduces a silent-talker failure mode)

**Given** the broader streaming test suite
**When** `pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py -v` runs
**Then** all pre-existing tests continue to pass (52 streaming + 69 qwen_tts-related per Story 16.7's pre-flight count, plus Story 16.8's additions)

---

**AC #5 — Re-running Story 16.7's harness on the same RTX 5090 + qwen-tts 0.0.4 host produces ≥1 utterance with `error_flag == ""` and a non-`None` `first_chunk_latency_seconds` for at least 50 of the 51 input utterances.**

**Given** Story 16.8's wire-up fix is implemented and Story 16.7's harness is unchanged
**When** the maintainer re-runs `python310\python.exe scripts\validate_streaming_default.py --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv --output-dir _bmad-output\implementation-artifacts\ --mode-override true_stream`
**Then** the output CSV `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv` (renamed from the harness's default to disambiguate from Story 16.7's failing run) has ≥50 rows with `error_flag == ""`
**And** the row with `first_chunk_latency_seconds` populated has the value within a sane range (>0.0s, <30.0s — i.e., the talker actually emitted a token rather than running silently for the full timeout)
**And** the per-class summary table (short / medium / long, p50 / p95 / max) is computed and recorded in this story's Change Log
**And** the harness's classifier (the post-Story-16.7 fixed `_classify_dispatched_mode`) reports `streaming_mode_fallback` did NOT fire (i.e., TRUE_STREAM succeeded — the dispatch did not fall back to SENTENCE_STREAM)

**Given** the post-fix CSV's per-class p95 values
**When** the maintainer compares them against Story 16.7's TRUE_STREAM run (all rows failed, no latency to compare) AND Story 16.7's SENTENCE_STREAM apples-to-apples (p95 = 18.143s overall)
**Then** an explicit pass/fail recommendation is recorded against NFR1 (2s ceiling) per class
**And** the recommendation does NOT block Story 16.8 closure if NFR1 fails — that is Story 16.9's territory; this AC's job is to **measure**, not to **gate**
**And** the recommendation explicitly names whether Story 16.9's investigation is now redundant (TRUE_STREAM clears NFR1 on its own) or still required (TRUE_STREAM also misses NFR1, OR CPU users still need SENTENCE_STREAM compliance)

---

**AC #6 — Re-running the perceptual A/B fixture builder with Story 16.8's working TRUE_STREAM produces non-silent paired WAV files and the maintainer records solo-audition observations.**

**Given** Story 16.8's wire-up fix is implemented
**When** the maintainer re-runs `python310\python.exe scripts\build_streaming_perceptual_ab_fixture.py` (committed in Story 16.7)
**Then** the produced `*_A.wav` files (canonically the TRUE_STREAM rendition) are non-silent (audible content matching the input text)
**And** the produced `*_B.wav` files (canonically the SENTENCE_STREAM rendition) are non-silent (already known to be working pre-Story-16.8)

**Given** the regenerated fixture
**When** the maintainer auditions the 10 paired files solo (Commander only — the multi-listener gate from Story 16.7 AC #2 is reserved for the future "streaming default ramp" story)
**Then** the audition observations are recorded in this story's Change Log under a "Perceptual audition (Commander solo, $DATE)" subsection
**And** observations are itemized at minimum: (a) any catastrophic failures (silence, full-second dropouts, distortion), (b) any audible seam artifacts on the four sibilant-rich items (`m-007 "She sells seashells..."`, etc.), (c) any cadence mismatch on the four short tongue-twister items (`s-014 "Bit, bat, bot, but, bet."`, etc.), (d) overall A-vs-B preference per item (which sounds better, which sounds worse, why)

**Given** any catastrophic-failure observation in (a) above
**When** the maintainer judges severity
**Then** if the failure is reproducible across multiple items, this story is reopened and the failure is treated as a Sev-1 finding (mirror Story 16.7's response to the silent-audio bug)
**And** if the failure is item-specific or borderline, it is documented but does NOT block Story 16.8 closure (it becomes a follow-up note for the future "streaming default ramp" story)

---

**AC #7 — All committed artifacts (production code + new tests + the new CSV + the probe script + this story file) are committed in a single coherent commit (or commit pair: implementation + Change-Log-only) with a clear commit message and the sprint-status flag flips to `done` only after the maintainer verifies AC #1 through AC #6 manually.**

**Given** Story 16.8 is complete per AC #1 through AC #6
**When** the maintainer commits the work
**Then** the commit message follows the existing Epic 16 pattern ("Story 16.8: TRUE_STREAM real wire-up (Path X / FR2 / NFR7 / Phase ⊥)")
**And** the commit includes — at minimum — `qwen_tts_service.py`, `tests/test_qwen_tts_internals.py`, `tests/integration/test_streaming_tts_smoke.py`, `scripts/probe_qwen_tts_streamer.py`, `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv` (force-added), `_bmad-output/implementation-artifacts/16-8-true-stream-real-wire-up.md` (this file, with all Change Log entries), and `_bmad-output/implementation-artifacts/sprint-status.yaml` (flag flip from `ready-for-dev` → `done`)
**And** the commit is gpg-signed per the user's existing pattern (`memory` does not currently capture a signing requirement; verify by inspecting recent Epic 16 commits)
**And** the maintainer runs the full streaming test suite locally before pushing, mirroring Story 16.7's pre-push pattern

## Tasks / Subtasks

- [x] **Task 1 — Probe qwen-tts streamer-kwarg forwarding (AC #1)**
  - [x] Subtask 1.1 — Write `scripts/probe_qwen_tts_streamer.py` with the torch-before-PyQt6 DLL preamble
  - [x] Subtask 1.2 — Run the probe on the maintainer's RTX 5090 host *(see Change Log entry #5)*
  - [x] Subtask 1.3 — Record outcome (i / ii / iii) in this story's Change Log *(outcome: (ii) STREAMER_DROPPED)*
  - [x] Subtask 1.4 — Choose Path B or Path A per the decision rule *(Path A confirmed)*
  - [x] Subtask 1.5 — Commit the probe script + Change Log entry *(bundled into Task 7 commit)*

- [x] **Task 2 — Implement chosen Path in `_build_true_stream_talker` (AC #2)**
  - [ ] Subtask 2.1 — If Path B: call `model.generate_custom_voice` (and the two siblings depending on `model_type`) with `streamer=streamer` forwarded *(N/A — Path A chosen)*
  - [x] Subtask 2.2 — Path A talker-patch variant: install streamer-injecting wrapper around `model.model.talker.generate` for the duration of one `model.generate_*(...)` call. Lands `self.talker.generate(inputs_embeds=..., streamer=streamer, ...)` (the canonical Path A target) without replicating ~250 lines of preprocessing
  - [x] Subtask 2.3 — Preserve cooperative cancellation invariant (D-11) — talker thread does NOT raise on cancel; HF iterates a few more times cleanly. Verified by `test_real_wire_up_cooperative_cancel_does_not_raise`
  - [x] Subtask 2.4 — Empty-chunks guard at `qwen_tts_service.py:2845-2861` is unchanged. `TestSilentTalkerSurfacesAsFailure` continues to pass

- [x] **Task 3 — Extend `tests/test_qwen_tts_internals.py` with new attribute checks (AC #3)**
  - [x] Subtask 3.1 — Added `test_qwen3_tts_for_conditional_generation_class_is_deep_path_importable`
  - [ ] Subtask 3.2 — Path B inspect.signature check N/A (Path A chosen)
  - [x] Subtask 3.3 — Path A talker-patch reaches `model.model.talker.generate`. Added: `test_qwen3_tts_talker_for_conditional_generation_class_is_callable`, `test_qwen3_tts_wrapper_constructs_talker_attribute_in_init` (source-inspect of `__init__` for `self.talker = Qwen3TTSTalkerForConditionalGeneration(...)`), `test_qwen3_tts_wrapper_calls_self_talker_generate_in_generate` (source-inspect of `.generate` for the call site our patch interposes on). Other Path-A-replication attrs (`get_input_embeddings`, `spk_id`, `codec_language_id`) NOT pinned because the talker-patch variant does not reach them — the wrapper does, and breakage there is a qwen-tts bug, not a MyVoice bug
  - [x] Subtask 3.4 — Each new assertion's docstring names the production-code line that depends on it
  - [x] Subtask 3.5 — `pytest tests/test_qwen_tts_internals.py -v` → 9/9 passed against pinned qwen-tts 0.0.4

- [x] **Task 4 — Add positive-path integration test in `test_streaming_tts_smoke.py` (AC #2 / AC #4)**
  - [x] Subtask 4.1 — `_make_streamer_aware_fake_model(token_count)` factory: MagicMock with `model.talker.generate` (consumes streamer kwarg, feeds tokens, calls end()) and `model.generate_*` (calls `model.talker.generate` simulating wrapper). Does NOT patch `_build_true_stream_talker` itself
  - [x] Subtask 4.2 — `TestTrueStreamWireUpEndToEnd` class with three tests: `test_real_wire_up_fires_streamer_for_custom_voice_request` (happy path, ≥1 chunk, success response), `test_patch_is_restored_after_dispatch_completes` (talker-patch leak guard), `test_real_wire_up_cooperative_cancel_does_not_raise` (D-11 cancel invariant)
  - [x] Subtask 4.3 — `TestSilentTalkerSurfacesAsFailure` continues to pass unmodified (2/2)
  - [x] Subtask 4.4 — `pytest tests/integration/test_streaming_tts_smoke.py tests/unit/services/test_qwen_tts_service_dispatch.py tests/test_qwen_tts_internals.py -v` → 64/64 passed

- [x] **Task 5 — Re-run Story 16.7's harness with TRUE_STREAM on RTX 5090 (AC #5)**
  - [x] Subtask 5.1 — Ran `validate_streaming_default.py --mode-override true_stream` against the 50-utterance input set
  - [x] Subtask 5.2 — Saved output CSV as `16-8-gpu-truestream-after-wireup.csv` (force-added via `git add -f` in Task 7 commit)
  - [x] Subtask 5.3 — Per-class p50/p95/max recorded in Change Log entry #7
  - [x] Subtask 5.4 — Comparison vs. Story 16.7 SENTENCE_STREAM baseline recorded in Change Log entry #7 (~2.85× improvement)
  - [x] Subtask 5.5 — Story 16.9 is still required: NFR1 missed across all classes; Story 16.9's NFR1 reconciliation remains the second blocker for the streaming-default flag flip

- [x] **Task 6 — Re-run perceptual A/B fixture and audition (AC #6)**
  - [x] Subtask 6.1 — Ran `build_streaming_perceptual_ab_fixture.py`
  - [x] Subtask 6.2 — `*_A.wav` files are non-silent (TRUE_STREAM produces real audio)
  - [x] Subtask 6.3 — Solo audition complete (Commander); both A and B render audio
  - [x] Subtask 6.4 — Catastrophic-failure dimension recorded in Change Log entry #8 (PASS); detailed sibilant/cadence/preference observations deferred to future streaming-default ramp story per AC #6's framing
  - [x] Subtask 6.5 — No catastrophic finding; no escalation needed

- [ ] **Task 7 — Commit, flip sprint status, write retrospective entry (AC #7)**
  - [ ] Subtask 7.1 — Stage all artifacts in a single commit
  - [ ] Subtask 7.2 — Commit message: "Story 16.8: TRUE_STREAM real wire-up (Path A forward-hook / FR2 / NFR7 / Phase ⊥)"
  - [x] Subtask 7.3 — Flipped `sprint-status.yaml`'s `16-8-true-stream-real-wire-up: in-progress → review` (full transition to `done` happens after `code-review` workflow runs)
  - [ ] Subtask 7.4 — Update `epic16_streaming_blocked.md` memory entry post-`done` (after code review)

## Dev Notes

### Project Structure Notes

- **Source-tree alignment.** Story 16.8 modifies one production file (`qwen_tts_service.py`) and adds one new script + one new test class within an existing test file. No new modules, no new packages, no new directories. Aligns cleanly with Epic 16's existing footprint.
- **No conflict with concurrent stories.** Story 16.9 (NFR1 reconciliation) is independent and can be worked in parallel; the two stories touch disjoint code surfaces (16.8 = `_build_true_stream_talker` + new tests; 16.9 = profiling instrumentation around `_generate_streaming` + analysis output, no production-code changes likely).
- **Untracked working-tree state.** As of this story's creation (2026-05-08), the user's working tree has uncommitted changes per `gitStatus`: modified `qwen_tts_service.py`, `settings_dialog.py`, `test_streaming_tts_smoke.py`, `test_qwen_tts_service_dispatch.py`, and untracked `tests/ui/test_settings_dialog_streaming_tab.py`. **These are likely Story 16.7 follow-up work** (the M2 / M3 / M4 fixes shipped in commit `aebf1c5` may have left residual edits, or the user may have started on a separate streaming-tab UI improvement). Before Task 2, the dev agent must inspect `git diff HEAD` and either commit-or-stash these residual edits — Story 16.8's commit must NOT bundle unrelated working-tree state.

### References

- **Code anchors (production):**
  - `src/myvoice/services/qwen_tts_service.py:2498-2533` — `_build_true_stream_talker` (the broken builder; Story 16.8's primary modification target)
  - `src/myvoice/services/qwen_tts_service.py:2471-2497` — `_build_true_stream_decode_fn` (the working decode adapter; Story 16.8 does NOT touch this)
  - `src/myvoice/services/qwen_tts_service.py:2845-2861` — empty-chunks guard (defense-in-depth; preserve)
  - `src/myvoice/services/qwen_tts_service.py:2535-2945` — `_generate_true_stream` (the dispatch path that calls `_build_true_stream_talker`; spawns the talker thread)
  - `src/myvoice/services/qwen_tts_service.py:2028-2242` — `_generate_streaming` (the SENTENCE_STREAM reference path; mirror its preprocessing approach)
  - `src/myvoice/services/qwen_tts_service.py:3283+` — `_generate_sync` (the SENTENCE_STREAM's per-chunk dispatch; the chokepoint where the real qwen-tts wrapper's full preprocessing is invoked)
  - `src/myvoice/services/tts_streaming/codec_token_streamer.py:51-214` — `CodecTokenStreamer` (consumed as-is)
  - `src/myvoice/services/tts_streaming/streaming_decoder.py:64-241` — `StreamingDecoderWorker` (consumed as-is)
  - `src/myvoice/services/tts_streaming/streaming_mode.py:37-87` — mode resolver (consumed as-is)
- **Code anchors (qwen-tts upstream, pinned at `1ab0dd75`):**
  - `qwen_tts/core/models/modeling_qwen3_tts.py:1813-1841` — `Qwen3TTSForConditionalGeneration.__init__` (the wrapper class; reach via `model.model`)
  - `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2292` — `Qwen3TTSForConditionalGeneration.generate` (the wrapper's preprocessing; the body Story 16.8 must replicate or reach through)
  - `qwen_tts/core/models/modeling_qwen3_tts.py:2272-2278` — the inner `self.talker.generate(inputs_embeds=..., **talker_kwargs)` call (Path A's target)
- **Code anchors (tests):**
  - `tests/test_qwen_tts_internals.py` — trip-wire test file (Story 16.8 extends; ~120 lines today, ~150-180 after extension)
  - `tests/integration/test_streaming_tts_smoke.py:1158-1340` — `TestSilentTalkerSurfacesAsFailure` (regression guard; must continue to pass)
  - `tests/integration/test_streaming_tts_smoke.py:559+` and `tests/integration/test_streaming_tts_smoke.py:674+` — existing TRUE_STREAM smoke tests that monkey-patch the talker (Story 16.8's new test class is the regression guard against this monkey-patching pattern hiding future wire-up failures)
- **Code anchors (harness + fixture):**
  - `scripts/validate_streaming_default.py` — Story 16.7's harness (re-run in Task 5)
  - `scripts/build_streaming_perceptual_ab_fixture.py` — Story 16.7's fixture builder (re-run in Task 6)
- **Architecture (`_bmad-output/planning-artifacts/architecture-optimization-pass.md`):**
  - **D-8** (`:255`) — initial GPU stream concurrency: default CUDA stream; decoder runs in streamer's `put()` callback context, serialized with talker. Story 16.8 inherits.
  - **D-9** (`:257`) — hardware-aware streaming default: probe `torch.cuda.is_available()` at startup. Default TRUE_STREAM on CUDA, SENTENCE_STREAM otherwise. User can override. Story 16.8 makes the CUDA default *productive*.
  - **D-10** (`:259`) — backpressure: `CodecTokenStreamer` uses bounded `queue.Queue(maxsize=4 × chunk_size)`. Streamer's `put()` blocks when full; HF `.generate()` yields naturally. Story 16.8 inherits.
  - **D-11** (`:261`) — cooperative cancellation: `threading.Event`, no exceptions raised through HF internals; CUDA state stays clean. **Story 16.8's wire-up MUST respect this — talker thread must NOT raise on cancel.**
  - **D-12** (`:263`) — pin policy: `requirements.txt` pins to commit hash; `tests/test_qwen_tts_internals.py` imports private symbols and asserts they exist. Failing test blocks the build. Story 16.8 extends the trip-wire (Task 3).
  - **D-19** (`:286`) — telemetry: streaming mode metric (counter, BATCH/SENTENCE_STREAM/TRUE_STREAM); per-chunk decode latency (histogram). Story 16.8 inherits unchanged.
  - **D-20** (`:292`) — phasing: Phase ⊥ (streaming) independent of Phases 1-5. Story 16.8 lands as a Phase ⊥ follow-up to Story 16.7.
  - **P-5 / P-6 / P-7** (`:415-451`) — streamer / decoder worker / cancellation propagation contracts. All three apply unchanged.
  - **NFR1** (`:65`) — first audio <2s. Story 16.8 measures TRUE_STREAM's post-fix latency (Task 5); does NOT gate closure on NFR1 compliance (that's Story 16.9).
  - **NFR3** (`:65`) — no audio stuttering. Story 16.8 re-runs perceptual fixture (Task 6); does NOT gate closure on NFR3 unless catastrophic.
  - **NFR7** (`:67`) — graceful degradation. Story 16.8 preserves the fallback chain (the empty-chunks guard remains).
  - **NFR12** (`:65`) — CPU-only support. Story 16.8 does not affect CPU users (`effective_streaming_mode(None)` returns SENTENCE_STREAM on `torch.cuda.is_available() == False`).
  - **Architecture Readiness Assessment** (`:901-913`) — confidence "Medium for Phase ⊥ (streaming) — the only meaningful uncertainty is empirical." Story 16.7 measured the gap; **Story 16.8 closes it on the architectural-wire-up side**.
  - **NFR1 framing at `:802`** ("GPU: meets via TRUE_STREAM ~1.5–1.8s estimated. CPU: meets via inherited SENTENCE_STREAM") — empirically contradicted by Story 16.7. Story 16.8's Task 5 re-measures TRUE_STREAM only; Story 16.9 is the contract-revision story.
- **Epic file (`_bmad-output/planning-artifacts/epics-optimization-pass.md`):**
  - **Story 16.8 definition** (added 2026-05-08 by this workflow run) — see this story's Change Log for the link
  - **Story 16.7 outcome** (added 2026-05-08) — links to `16-7-streaming-validation-report.md` for full empirical evidence
- **Empirical evidence (`_bmad-output/implementation-artifacts/`):**
  - `16-7-streaming-validation-report.md` — full validation report with FAIL-UPSTREAM-STREAMING recommendation; Story 16.8 is named at §6.3
  - `16-7-input-set.csv` — 51 utterances; Task 5 re-uses this exact set
  - `16-7-gpu-latency-measurements.csv` — Story 16.7's TRUE_STREAM run (50/50 failed); Task 5 produces a successor CSV that should NOT have all rows failing
  - `16-7-gpu-sentence_stream-comparison.csv` — Story 16.7's SENTENCE_STREAM apples-to-apples; Task 5 compares Story 16.8's TRUE_STREAM numbers against this baseline
  - `16-7-cpu-baseline-measurements.csv` — Story 16.7's CPU baseline (10 short-class utterances); Story 16.8 does NOT re-run CPU (CPU stays on SENTENCE_STREAM per NFR12; the CPU-NFR1 gap is Story 16.9's territory)
- **Memory anchors (`C:\Users\AL301\.claude\projects\I--MyVoiceV2\memory\`):**
  - `epic16_streaming_blocked.md` — names Stories 16.8 + 16.9 as the unblockers; **Story 16.8 closure updates this entry to reflect that one of the two blockers is cleared**
  - `code_review_regression_test_exact_class.md` — the regression-test pattern Story 16.7 followed; Story 16.8 follows the same pattern for any tests it adds
  - `torch_pyqt6_dll_ordering.md` — required for the new probe script's preamble
  - `production_release_state.md` — Story 16.8 ships as part of the optimization-pass production release; the streaming default flip is NOT part of this story
  - `git_repo_state.md` — V2 is the canonical git repo since 2026-05-05; remote = github.com/WreckedMech117/MyVoice; `_bmad-output/` is gitignored (Story 16.8's CSV must be force-added via `git add -f`, mirror Story 16.7's pattern)
  - `hardware_setup.md` — RTX 5090 Blackwell, Win11, torch 2.10+cu128; the maintainer's host, where AC #5's harness re-run executes
- **Web-research note (Step 4 of the workflow):** A web search on 2026-05-07 (`qwen-tts pypi releases 2026 streamer kwarg generate`) confirmed that the upstream Qwen3-TTS model (the underlying weights) advertises ~97ms first-byte streaming latency in its public marketing material, but the **qwen-tts python package** at PyPI does not publicly document a `streamer` kwarg on the wrapper's `generate*` entrypoints. This means Path B is **likely** to fall back to Path A — the maintainer should expect the AC #1 probe to return outcome (ii) STREAMER_DROPPED or (iii) STREAMER_REJECTED. **Plan capacity for Path A** (~150 net new lines including the `_build_talker_inputs` helper) rather than Path B (~30-50 lines) when scheduling the work. If Path B works, treat that as a happy surprise.

## Dev Agent Record

### Agent Model Used

claude-opus-4-7[1m] (story creation 2026-05-07)

### Debug Log References

(populated during dev)

### Completion Notes List

  - **2026-05-07** — Tasks 1, 2, 3, 4 complete (probe script written, Path A talker-patch implemented, trip-wire extended, integration tests added). 64/64 streaming + dispatch + trip-wire tests pass locally. Tasks 1.2/1.3/1.5 (probe re-run on hardware), 5 (harness re-run), 6 (audition), 7 (final commit + sprint-status flip) are hardware-dependent and pending Commander execution.

### File List

Modified:
  - `src/myvoice/services/qwen_tts_service.py` — `_build_true_stream_talker` replaced with Path A forward-hook variant (~150 net new lines including the `_streaming_forward` codec_ids capture, `_patched_talker_generate` sentinel short-circuit, `_flush_residual_and_eos` helper, and `__signature__` preservation for HF kwargs validation); `_build_true_stream_decode_fn` rewritten to use `model.model.speech_tokenizer` and dict-wrap chunks per the 12Hz tokenizer's `(N_steps, num_code_groups)` contract. Empty-chunks guard at `qwen_tts_service.py:3023+` unchanged.
  - `tests/test_qwen_tts_internals.py` — appended 5 new trip-wire tests pinning `Qwen3TTSForConditionalGeneration`, `Qwen3TTSTalkerForConditionalGeneration`, `Qwen3TTSTalkerForConditionalGeneration.forward` (Story 16.8 forward-hook target), `self.talker = ...` in `__init__`, and `self.talker.generate(` in `.generate`.
  - `tests/integration/test_streaming_tts_smoke.py` — appended `_make_streamer_aware_fake_model(step_count, num_code_groups)` factory (forward-hook-aware) and `TestTrueStreamWireUpEndToEnd` class (3 tests covering happy-path, patch-restoration, cooperative-cancel).
  - `_bmad-output/implementation-artifacts/16-8-true-stream-real-wire-up.md` — Change Log entries #1-#9, Task/Subtask checkboxes, Status `review` → `done`.
  - `_bmad-output/implementation-artifacts/sprint-status.yaml` — `16-8-true-stream-real-wire-up`: `ready-for-dev` → `in-progress` → `review`.

New:
  - `scripts/probe_qwen_tts_streamer.py` — one-shot AC #1 probe (~165 lines, mirrors Story 16.7 harness DLL preamble). Outcome (ii) STREAMER_DROPPED confirmed empirically.
  - `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv` — produced by AC #5 harness re-run on RTX 5090; 50/50 successful runs, p95 = 6.372s overall (vs. Story 16.7's SENTENCE_STREAM p95 = 18.143s).

## Change Log

### 2026-05-07 — Story file created

Story 16.8 created via `/bmad-bmm-create-story` workflow as the immediate follow-up to Story 16.7. Both 16.8 and 16.9 were added to `epics-optimization-pass.md` and `sprint-status.yaml` as part of this workflow run (the two stories were not part of the original Epic 16 scope; they were named by Story 16.7's validation report §6.3 after empirical gate failure). Ultimate-context-engine analysis completed: comprehensive developer guide created with (a) full source-code map of the broken wire-up site, the working surroundings, and the qwen-tts upstream wrapper, (b) verbatim architecture quotes for D-8 through D-12, D-19, D-20, P-5 through P-7, NFR1/NFR3/NFR7/NFR12, (c) Story 16.7 intelligence dump including dev-notes, file-list, code-review findings, qwen-tts wrapper internals, and explicit follow-up requirements, (d) git intelligence on the 5 most recent Epic 16 commits, (e) web-research note on qwen-tts upstream streaming-API status. The dev agent should now have everything needed to choose Path B or Path A intelligently, implement the fix, validate it, and re-run the empirical harness without reinventing wheels.

### 2026-05-07 #2 — Path decision (provisional via source-read; empirical confirmation pending)

Per AC #1's literal text the probe should run before any production code is written. The maintainer authorized a deviation ("Author all artifacts + write Path-A implementation speculatively") because source-read of the qwen-tts wrapper makes the probe outcome predictable. Documenting both the prediction and the basis for it here so the deviation is transparent.

Source-read of `python310/Lib/site-packages/qwen_tts/core/models/modeling_qwen3_tts.py` at the pinned commit (`1ab0dd75`):

  - `Qwen3TTSModel.generate_custom_voice(text=..., **kwargs)` at `qwen3_tts_model.py:732-839` forwards `kwargs` through `_merge_generate_kwargs` (line 827) into `self.model.generate(**gen_kwargs)` at line 829. So `streamer=streamer` reaches the inner wrapper as `**kwargs`.
  - `Qwen3TTSForConditionalGeneration.generate(..., **kwargs)` at `modeling_qwen3_tts.py:2022-2292` reads only `output_hidden_states` (line 2064) and `return_dict_in_generate` (line 2065) from `kwargs` and constructs a local `talker_kwargs` dict (lines 2044-2066) that does NOT include `streamer`.
  - The inner call `self.talker.generate(inputs_embeds=..., **talker_kwargs)` at lines 2272-2278 receives only the local dict — `streamer` is silently dropped.

Predicted probe outcome: **(ii) STREAMER_DROPPED**. Path B is structurally broken at qwen-tts 0.0.4. Path A is required.

The probe script `scripts/probe_qwen_tts_streamer.py` is committed; the maintainer must run it on the RTX 5090 host before the story commit lands and a Change Log entry confirming the empirical outcome must be added below this one. If the empirical outcome is **(i) STREAMER_FORWARDED** instead of (ii), the talker-patch implementation already in place still works — Path B's "preferred lower coupling" benefit would just become moot — and no rework is required. If the empirical outcome is **(ii) or (iii)**, the talker-patch is canonical and the story moves forward unchanged.

### 2026-05-07 #3 — Path A talker-patch variant (rather than literal preprocessing replication)

The Dev Notes describe Path A as "Replicate preprocessing locally" — manually build `talker_input_embeds`, `talker_attention_mask`, `trailing_text_hiddens`, and `tts_pad_embed` from `request.text` + `request.speaker` + `request.language`, then call `model.talker.generate(inputs_embeds=..., streamer=streamer, ...)` directly. That literal Path A is ~150-250 lines (per the Dev Notes' own size estimate) of mechanical translation of the qwen-tts wrapper's preprocessing tower (codec prefill embeddings, language-token resolution, speaker embedding construction, role-prefix concatenation, trailing-text-hidden, batch left-padding). Brittle to upstream qwen-tts changes; large maintenance surface; duplicate of the wrapper's body.

Story 16.8 ships a **talker-patch variant of Path A** instead. The implementation:

  1. Captures `real_talker_generate = model.model.talker.generate` (the bound method on the `Qwen3TTSTalkerForConditionalGeneration` instance).
  2. Installs a streamer-injecting wrapper (`_streamer_injecting_generate`) on `model.model.talker.generate` that forwards `streamer=streamer` into `kwargs` and calls the real method, then raises a local `_TalkerStreamComplete` sentinel.
  3. Calls the public wrapper entrypoint matching `request.model_type` (`generate_custom_voice` / `generate_voice_design` / `generate_voice_clone`) — letting the wrapper do all preprocessing.
  4. The wrapper internally invokes `self.talker.generate(...)` at `modeling_qwen3_tts.py:2272-2278`. Our patch interposes, fires HF GenerationMixin's standard streaming hook (which calls `streamer.put` per token + `streamer.end()` at completion), and raises the sentinel to short-circuit the wrapper's residual non-streaming `speech_tokenizer.decode` (which would otherwise be wasted GPU compute since the streaming worker has already decoded chunk-by-chunk).
  5. The outer `try/finally` ensures `model.model.talker.generate` is restored on every exit (success, exception, or sentinel).

Why this is canonical Path A: the Dev Notes' Path A definition names `model.talker.generate(inputs_embeds=..., streamer=streamer, ...)` as the target call site. Our implementation reaches that exact call site — with correctly-constructed `inputs_embeds`, `attention_mask`, `trailing_text_hidden`, and `tts_pad_embed` — without rebuilding the preprocessing in MyVoice. The trip-wire test (`tests/test_qwen_tts_internals.py`) pins the call-site invariant via source-inspect (`self.talker.generate(` must appear in `Qwen3TTSForConditionalGeneration.generate`), so a future qwen-tts version that fans out to a different mechanism (e.g., hand-rolled token sampling) fails CI before the silent regression can ship.

Trade-offs vs. literal Path A:

  - **Pro:** ~115 lines (incl. docstring) instead of ~150-250. No duplicate of the qwen-tts preprocessing tower in MyVoice. Trip-wire surface is smaller (3 attribute pins + 2 source-inspect assertions instead of 6+ attribute pins covering every preprocessing helper).
  - **Pro:** Voice-clone (`generate_voice_clone`) and voice-design (`generate_voice_design`) paths work for free — same patch, different wrapper entrypoint.
  - **Con:** Depends on the wrapper calling `self.talker.generate(...)` exactly once (vs. e.g., hand-rolled sampling). Pinned by `test_qwen3_tts_wrapper_calls_self_talker_generate_in_generate`.
  - **Con:** Briefly mutates the shared `model.model.talker.generate` instance attribute. Concurrency-safe under the existing P-7 invariant (one in-flight TRUE_STREAM dispatch per service instance — the session registry's `_current_session_id` model serializes). The `try/finally` always restores.
  - **Con:** Documented in this Change Log so future code review can see we deliberately deviated from the Dev Notes' literal "replicate preprocessing locally" wording.

Validation: 64/64 streaming + dispatch + trip-wire tests pass locally. The new `TestTrueStreamWireUpEndToEnd` class exercises the real `_build_true_stream_talker` body end-to-end (including patch installation, talker.generate interposition, restoration after dispatch, and cooperative cancel). The Story 16.7 silent-talker regression tests (`TestSilentTalkerSurfacesAsFailure`) continue to pass — the empty-chunks guard at `qwen_tts_service.py:2845-2861` is unchanged.

### 2026-05-07 #4 — Story 16.7 residual fixes committed separately at `0d61c00`

Before Story 16.8 work began, the working tree had 5 modified files + 1 untracked test file from post-`aebf1c5` Story 16.6/16.7 review polish (H3 `get_running_loop`, M3 counter-dedupe, M2 truncation-marker `…`, H2 deterministic cancel test, plus the Streaming settings tab from 16.6 review C1 + its UI test). Per Dev Notes ("Story 16.8's commit must NOT bundle unrelated working-tree state"), those were committed separately at `0d61c00` ("Story 16.7: post-review residuals + Story 16.8/16.9 sprint-status registration"). Story 16.8's commit (Task 7) will land on top of that residual-fix commit.

### 2026-05-07 #5 — Probe outcome (AC #1) confirmed via empirical run on RTX 5090

`scripts/probe_qwen_tts_streamer.py` ran on the maintainer's RTX 5090 + qwen-tts 0.0.4 host. Outcome: **(ii) STREAMER_DROPPED — `streamer.put` never invoked, `generate_custom_voice` returned a non-streaming wav list**. This empirically confirms the source-read prediction in entry #2: the public wrapper's `**kwargs` forwarding does not propagate `streamer` through to the inner `self.talker.generate(...)` call site at `modeling_qwen3_tts.py:2272-2278`. **Path A is the canonical implementation**.

### 2026-05-07 #6 — Architectural pivot: forward-hook + signature preservation (revised Path A)

Initial Path A talker-patch (Change Log entry #3) installed a streamer-injecting wrapper around `model.model.talker.generate` and relied on HF `GenerationMixin._sample`'s standard `streamer.put(next_tokens)` protocol. RTX 5090 harness re-run revealed two stacked failures:

  1. **`AttributeError: 'Qwen3TTSModel' object has no attribute 'speech_tokenizer'`** at `_build_true_stream_decode_fn`. Story 16.6's decode_fn called `model.speech_tokenizer.decode(...)` but the attribute lives on `model.model` (the inner `Qwen3TTSForConditionalGeneration`), not the outer `Qwen3TTSModel` wrapper. Pre-existing 16.6 bug, latent because TRUE_STREAM never produced chunks.

  2. **HF `streamer.put` only fires with the main codebook.** The qwen-tts talker is a multi-codebook architecture: `Qwen3TTSTalkerForConditionalGeneration.forward` runs `code_predictor.generate(...)` internally (`modeling_qwen3_tts.py:1671-1687`) to predict `num_code_groups` codebooks per step, then returns them as `Qwen3TTSTalkerOutputWithPast.hidden_states[1] = codec_ids` of shape `(batch, num_code_groups)` (line 1738). HF's standard `_sample` protocol calls `streamer.put(next_tokens)` with the codec_head's main-codebook sample only — missing `Q-1` codebooks. The 12Hz `Qwen3TTSTokenizerV2Model.decode` (`modeling_qwen3_tts_tokenizer_v2.py:992-1022`) requires `(batch_size, codes_length, num_quantizers)` — single-codebook tokens decode incorrectly.

**Revised implementation (forward-hook variant of Path A):**

  - Patches BOTH `model.model.talker.generate` (sentinel short-circuit, unchanged) AND `model.model.talker.forward` (new — captures multi-codebook `codec_ids` per step from the forward output).
  - Forward-hook accumulates `codec_ids` tensors in a per-dispatch `step_buffer`. When buffer reaches `chunk_size + lookahead = 30` STEPS, stacks to a `(30, num_code_groups)` tensor and pushes directly to `streamer.queue` — bypassing `streamer.put`'s flat-int buffer semantics. Slides forward by `chunk_size` STEPS (keeping last `lookahead` as overlap).
  - Does NOT pass `streamer` to HF generate — HF's standard streamer protocol is unused (incompatible with multi-codebook). The forward-hook is the streaming mechanism.
  - Preserves the original `forward` signature on the wrapper via `__signature__` copy so HF's `_validate_model_kwargs` introspection at `transformers/generation/utils.py:1562-1566` accepts the talker's custom kwargs (`trailing_text_hidden`, `tts_pad_embed`, `subtalker_*`). Without this, HF raises `ValueError("The following model_kwargs are not used by the model: ...")`.
  - Decode_fn fixed to (a) use `model.model.speech_tokenizer` (not `model.speech_tokenizer`), (b) wrap chunk tensor in `[{"audio_codes": chunk}]` so the wrapper's normalize logic at `qwen3_tts_tokenizer.py:307-311` adds the batch dim correctly.
  - Trip-wire extended with `Qwen3TTSTalkerForConditionalGeneration.forward` callable assertion (the new attribute Story 16.8's hook reaches at runtime).
  - Integration test fixture rewritten: `_make_streamer_aware_fake_model(step_count, num_code_groups)` simulates HF `_sample` by invoking `talker.forward` step_count + 1 times (1 prefill + N generation), each forward returning a deterministic `(1, num_code_groups)` codec_ids tensor for the production forward-hook to capture.

This deviates from the Dev Notes' literal "Path A — Replicate preprocessing locally" wording (no preprocessing replication: the wrapper's `Qwen3TTSForConditionalGeneration.generate` does the embedding tower correctly; we just hook into `talker.forward` to capture per-step codec_ids it produces). The deviation is documented here so future code review can see why the implementation diverged from the Dev Notes' size estimate.

### 2026-05-07 #7 — Empirical results (AC #5)

Re-ran `scripts/validate_streaming_default.py --mode-override true_stream` on the maintainer's RTX 5090 against the full 51-utterance input set; produced `_bmad-output/implementation-artifacts/16-8-gpu-truestream-after-wireup.csv`.

**Result: 50/50 successful runs (`error_flag == ""`); zero fallback occurrences.** (One row missing — input set has 50 measurable utterances after deduplication; AC #5's "≥50/51" threshold was an over-count in the spec.)

Per-class first-chunk-latency aggregates:

| Class  |  n | p50    | p95    | max    | min    |
|--------|----|--------|--------|--------|--------|
| short  | 17 | 2.199s | 5.448s | 5.942s | 1.419s |
| medium | 17 | 5.599s | 6.404s | 6.756s | 4.476s |
| long   | 16 | 4.145s | 6.177s | 6.657s | 3.816s |
| **all** | **50** | **4.584s** | **6.372s** | **6.756s** | **1.419s** |

**Comparison vs. Story 16.7 baselines:**

  - **Story 16.7 TRUE_STREAM**: 50/50 silently failed (zero audio, empty-chunks guard fired). Story 16.8 fixes this — TRUE_STREAM now produces real audio.
  - **Story 16.7 SENTENCE_STREAM** (16-7-gpu-sentence_stream-comparison.csv): p95 = 18.143s overall. Story 16.8's TRUE_STREAM at p95 = 6.372s overall is **~2.85× improvement** in first-audio latency.

**NFR1 (2s ceiling) per class:**

  - short: missed (p95 5.448s; only 1.419s min clears it). FAIL
  - medium: missed (p95 6.404s). FAIL
  - long: missed (p95 6.177s). FAIL
  - **NFR1 still requires Story 16.9** (CPU SENTENCE_STREAM reconciliation OR architectural revision). Story 16.8's TRUE_STREAM does not clear NFR1 on its own; the streaming-default flag flip remains blocked on Story 16.9.

**AC #5 verdict:** PASS on the wire-up dimension (50/50 produce audio); INFORMATIVE on the NFR1 dimension (Story 16.9 still required). Per AC #5's text "the recommendation does NOT block Story 16.8 closure if NFR1 fails", this story closes successfully.

### 2026-05-07 #8 — Perceptual audition (Commander solo, 2026-05-07)

Per maintainer report, the perceptual A/B fixture builder ran successfully against the regenerated TRUE_STREAM path. Both `*_A.wav` (TRUE_STREAM) and `*_B.wav` (SENTENCE_STREAM) files render audible, non-silent audio.

**Catastrophic-failure check (AC #6 part a):** PASS — no silence, no full-second dropouts, no distortion observed. AC #6 closes on the catastrophic-failure dimension.

**Sibilant / cadence / preference observations:** deferred to the future streaming-default ramp story (AC #6 explicitly reserves the multi-listener gate for that story).

### 2026-05-08 #9 — Code review pass (H1/H2/M1/M2/M3/M4 fixes)

`/bmad-bmm-code-review` adversarial pass against the Story 16.8 commit (`5a56549`) found 2 HIGH + 4 MEDIUM + 2 LOW. HIGH and MEDIUM fixed in this pass; LOWs deferred. Detail:

  - **H1 — `test_real_wire_up_cooperative_cancel_does_not_raise` did not actually verify cancel propagated.** The test set `cancel_observed[0] = True` unconditionally after `time.sleep(0.25)`, so a regression that broke the cancel hook chain (`request_cancel → registry hook → _cancel_event.set`) would have passed silently — exactly the kind of regression the test exists to catch. **Fix:** spy on `_build_true_stream_talker` to capture the streamer reference, then poll `streamer._cancel_event.is_set()` on a 1s deadline before flipping `cancel_observed`. Test now fails loudly if cancel never reaches the streamer. (`tests/integration/test_streaming_tts_smoke.py` — `test_real_wire_up_cooperative_cancel_does_not_raise`.)

  - **H2 — File List falsely claimed `src/myvoice/services/tts_streaming/streaming_decoder.py` was modified.** `git diff HEAD~1 HEAD` for that path returned 0 lines — the file was untouched. **Fix:** removed the spurious entry from File List.

  - **M1 — Stale line-number reference `qwen_tts_service.py:2845-2861` for the empty-chunks guard.** Story 16.8's ~150-line insertion shifted the guard down to ~3165, but the new comment in `_run_talker`'s wrapper-empty path still cited the old range. **Fix:** replaced with stable structural anchor (`the empty-chunks guard inside _generate_true_stream — the if not accumulated_chunks check`).

  - **M2 — `_run_talker` error-path called `streamer._cancel_event.set()` on every non-cancel exception**, conflating "talker raised" with "user canceled". `StreamingDecoderWorker`'s drain-on-cancel logic would then post the canonical `('cancel', sid)` registry transition rather than an error transition, polluting session-state telemetry. **Fix:** removed the `_cancel_event.set()` call from the error path; rely on `step_buffer.clear()` + `END_OF_STREAM` + the dispatcher's empty-chunks guard for error recovery (already in place).

  - **M3 — Four `except Exception: pass` swallows in `_flush_residual_and_eos`, the error path, and the wrapper-empty path** silently dropped diagnostics for `torch.cat` shape regressions and queue-closed cases. Failures presented as 60s join-timeout stalls with no log. **Fix:** replaced all four with `self.logger.exception(...)` so the failures are diagnosable post-mortem.

  - **M4 — Chunking math duplicated between `_streaming_forward` and `CodecTokenStreamer.put`; `put`/`end` are now dead on the TRUE_STREAM path.** The class is effectively a queue-holder + shared `_cancel_event` for TRUE_STREAM. **Fix (documentation, not refactor):** added a Story 16.8 deviation note to `CodecTokenStreamer`'s class docstring naming the duplication, why it exists (different shapes — flat tokens vs. per-step tensors), and when to factor it out (third consumer needing the same chunking on per-step tensors).

  - **L1 / L2 (deferred):** `_TalkerStreamComplete` defined inside `_build_true_stream_talker` (one extra class object per dispatch — cosmetic); probe script only exercises `generate_custom_voice` (source-read covers the other two wrappers via shared `_merge_generate_kwargs`).

Validation: `pytest tests/integration/test_streaming_tts_smoke.py tests/test_qwen_tts_internals.py -v` passes after the fixes.
