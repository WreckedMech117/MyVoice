# Story 20.2 — Warm-Path Compile Priming: evidence

Phase ⊥-Polish-3 · Epic 20 (First-Audio Latency) · captured 2026-08-31 on the
RTX 5090 host (`torch 2.10.0+cu128`, capability 12.0, 31.8 GiB).

---

## 0. Headline

| claim | verdict |
|---|---|
| Priming on the warm cache collapses segment 1b from the ~3.96 s class to the ~100 ms class | **CONFIRMED** — long: 3,593 → 86 ms; short: 4,442 → 98 ms (medians, n=5 each) |
| First-generation TTFA lands in the same band as steady-state TTFA | **CONFIRMED** — long 1,526 ms vs Story 20.1 pooled 1,785 ms; short 1,849 ms vs pooled 1,651 ms |
| Priming reaches no audio consumer, with a consumer wired from process start | **CONFIRMED** on real hardware (`consumer chunks seen during priming = 0`, 12/12 launches) and in unit tests |
| Priming reaches no *other* user-facing channel either — speakers/virtual mic, the Replay Last cache, the session registry | **CONFIRMED after the review pass (§7).** The first implementation of this story gated only the chunk callback and **leaked three other channels**; all four are now gated and each has a mutation-verified test. |
| Reversible | **YES** — `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1` |
| The change is reachable from the shipped GUI | **NO — see §6.** `warmup_compile_async` is fired *before* the model preload in `app.py`, so it short-circuits at `no_model_loaded` on every launch. This is a **pre-existing Story 18.4 defect**, not one this story introduces, but it means the measured win is not yet delivered to users. |

---

## 1. What changed

### 1.1 AC #2 / #3 — the audio-suppression mechanism (Task 1)

**Chosen mechanism: a per-request flag, consulted at every channel that can
reach a user.**

* `QwenTTSRequest.suppress_audio_output: bool = False`
  (`qwen_tts_service.py`, dataclass at :290).
* `QwenTTSService._is_suppressed(request)` — the single predicate.
* `QwenTTSService._audio_chunk_sink(request)` — returns the wired
  `_audio_chunk_ready_callback` for an ordinary request and `None` for a
  suppressed one. It is the **only** place in the module that reads the
  callback attribute.

**Four channels, not one.** The first implementation of this story gated only
the chunk callback; the review pass (§7) found three more channels that reach a
user without ever consulting it. All four are now gated on `_is_suppressed`:

| # | channel | what it reaches | gated at |
|---|---|---|---|
| 1 | `_audio_chunk_ready_callback` | progressive playback | 3 emit sites, via `_audio_chunk_sink` |
| 2 | `audio_coordinator.play_dual_stream` | **the monitor device + the virtual microphone** | TRUE_STREAM playback kick |
| 3 | `_save_audio_to_cache` | `myvoice_current.wav` → Replay Last, and transitively `_generation_complete_callback` | all 3 dispatch paths |
| 4 | `SessionRegistry.create_session` | `_saveable` / `_focal` → the Save button, Stop / Clear Comms focal-cancel | all 3 dispatch paths, plus `registry_post` nulled in TRUE_STREAM |

Channels 2-4 are gated in **all three** dispatch modes, not just TRUE_STREAM,
because the flag rides the TRUE_STREAM → SENTENCE_STREAM → BATCH fallback chain
intact — so any mode can end up running a suppressed generation.

**Why this one, and not the alternatives the story offered (Task 1.1).**

| candidate | rejected because |
|---|---|
| detach-and-restore `_audio_chunk_ready_callback` around the priming call | Service-wide for the duration of priming. Priming now runs on **every** launch and takes ~4.4 s; a user generation dispatched inside that window would have its audio silenced too. That is AC #3's named failure mode — "a latency fix turned into a no-audio bug". |
| a service-level `self._suppressing` boolean | Same defect, same reason. Scoped to a *window of time*, not to a generation. |
| **per-request flag + single resolver** | The suppression decision travels with the object it describes. Two concurrent generations cannot confuse it. It holds regardless of when `set_audio_chunk_ready_callback` ran. |

**What makes it hard to get wrong from a future caller's perspective.** A
per-request flag is only as strong as the discipline that every channel asks the
predicate. That discipline is **mechanical**, not documentary — and the review
pass showed why the mechanism has to be broader than one attribute name:

* `test_every_user_facing_channel_call_site_is_suppression_gated` strips
  comments from `qwen_tts_service.py` and requires every call site of
  `play_dual_stream`, `_save_audio_to_cache`, and `create_session` to have
  `suppressed` in its guard window. It also fails if a channel's marker stops
  matching anything, so a rename cannot silently retire the protection.
* `test_no_emit_site_reads_the_raw_callback` keeps the narrow guard on channel 1
  (the one with many call sites): `self._audio_chunk_ready_callback` may appear
  only in its declaration, its setter, and the single `return` inside
  `_audio_chunk_sink`.
* `_run_compile_priming` re-checks `_is_suppressed` on its own request and
  **raises** rather than dispatching if the flag has gone missing (review F6).
  `_is_suppressed` reads the field through `getattr(..., False)`, which fails
  *open* on purpose — an unrecognised request is treated as user-facing, because
  silencing a real user is the worse failure — so the trip-wire is what converts
  a field rename from "priming quietly becomes audible again" into a loud,
  non-fatal `priming_failed`.

**Task 1.3 — the docstring.** `_run_compile_priming`'s claim that *"No audio
output reaches the user (the priming runs before the
`set_audio_chunk_ready_callback` wires consumers up — and even if a consumer is
wired, the generation is short enough that any audible artifact is bounded)"* is
**deleted**. It is replaced by a description of the positive mechanism, and by
an explicit note that the pre-20.2 text asserted a guarantee the code did not
provide. The `tts_compile="off"` gate's comment (which records the launch where
the ordering assumption lost and audible "Hello world." reached a user's
speakers) is updated to mark reason (a) *retired by this story* and reason (b)
— a meaningless `meta.json` sidecar plus wasted startup time — as the one that
still keeps the gate. The gate itself is unchanged (AC #5).

### 1.2 AC #1 / #5 / #6 — the warm path (Task 2)

`warmup_compile_async`'s `is_warm(key) is True` branch no longer returns early.
It now:

1. checks the AC #6 gate `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1` — if set,
   the exact pre-20.2 behavior is restored (silent, `reason="cache_hit"`, no
   priming, indicator never fires);
2. otherwise emits the `"Preparing TTS engine…"` indicator, runs
   `_run_compile_priming()`, and records `reason="primed_warm"` with
   `value=1.0`;
3. **does not call `compile_cache.mark_warm(key)`** — the key is already warm
   and the cold path's mark-on-success contract is untouched;
4. on failure records `reason="priming_failed"` + `error=<ExcType>`, logs a
   WARNING, clears the indicator, and returns normally. A failed warm prime is
   never fatal: the pre-story behavior (lazy inductor reload on the first user
   generation) is still a fully working fallback, so the cost is latency, not
   function.

The telemetry `reason` vocabulary is now nine values; `"primed_warm"` is
separable from `"primed_cold"` and `"cache_hit"` in the metric stream, and
`"cache_hit"` stays meaningful as the gated-off signal.

**All five existing fast-exits are untouched and still fire** (AC #5) — the new
branch lives strictly *after* them:

| gate | still fires | test |
|---|---|---|
| `MYVOICE_DISABLE_COMPILE_WARMUP=1` | yes | `test_warmup_disabled_by_env_var_short_circuits` |
| `tts_compile="off"` | yes | `test_warmup_tts_compile_off_skips_priming_and_gate_log` |
| non-Ampere / no CUDA | yes | `test_warmup_pre_ampere_skips_priming` |
| no model registry | yes | `test_warmup_no_model_registry_skips` |
| no model loaded | yes | `test_warmup_no_model_loaded_defers_to_first_generation` |

CPU-only and pre-Ampere hosts are behaviorally identical to before: they exit
at the hardware probe, above everything this story changed.

**Known trap honoured.** `compute_key(..., decode_window_frames=30)` is
untouched (Dev Notes / Story 20.1 §5.4). Changing it would invalidate every
existing warm cache directory for no benefit; that is Follow-up B's job.

---

## 2. Method (Task 3)

`tools/ttfa_spike_harness.py` — the Story 20.1 harness, which drives the
**production** `_generate_true_stream` path with the six shipped `ttfa_*`
boundary metrics and a real `StreamingChunkBuffer` consumer. One new flag:

```
--prime    run one suppressed priming generation at startup, before the
           measured runs (stands in for warmup_compile_async's warm branch)
```

The priming request is the production shape — BASE model, the same
`voice_clone_prompt`, `text="Hello world."` (matching
`QwenTTSService._COMPILE_PRIMING_TEXT`), and **`suppress_audio_output=True`**.
The consumer (`ConsumerSim`) is wired via `set_audio_chunk_ready_callback`
immediately after `service.start()`, i.e. **before** priming — deliberately
violating the precondition the pre-20.2 docstring relied on — and it counts
every chunk it is ever handed (`total_chunks_seen`).

Capture: **one fresh process per sample**, `--runs 1 --warmup 0`, so run 1 *is*
the user's first generation after launch. `--compile auto`, warm on-disk cache
(`%LOCALAPPDATA%/MyVoice/torch_compile_cache/391c2f2be3340b07`, 38 MB, present
since 2026-05-20). n = 5 launches per cell, four cells.

Drivers: `20-2-capture-long.sh`, `20-2-capture-short.sh`. Aggregator:
`20-2-aggregate.py`. Raw rows: `20-2-{before,after}-{long,short}-r0{1..5}.csv`.

**Deviation from Task 3.1 as written.** The task names
`MYVOICE_PROGRESSIVE_PLAYBACK_CSV` as the capture channel. That env var engages
`progressive_playback_csv_capture`, which subscribes to the *same* metric
stream and writes the *same* six `ttfa_*` boundary rows — but it is wired from
`MyVoiceApp`, i.e. it captures the GUI, which per §6 never reaches the code
this story changed. The harness's own collector reads the identical metric
stream — `_TTFA_BOUNDARIES` in `ttfa_spike_harness.py` covers the five
producer-side `ttfa_*` boundaries verbatim (the sixth,
`ttfa_first_playback_write_ms`, is consumer-side and emitted by
`AudioCoordinator`, which the harness stands in for with a real
`StreamingChunkBuffer`) — and pre-reduces it to segments, which is also what
every Story 20.1 number was captured with. Segments 1a/1b/2/3, the only ones
this story moves, come entirely from the producer-side five. Same instrumentation,
same boundaries, headless driver — chosen so the measurement isolates the
mechanism rather than the unreachable call site.

**Discarded sample.** The very first trial launch of the session recorded
segment 1b = 17,378 ms — a genuine *cold FX-graph compile*, not a reload,
because this working tree's build had not yet compiled under the current key.
It is excluded and not counted in any n; every reported row is from the warm
regime the story targets. The second trial (1b = 4,600 ms) already matched the
Story 20.1 §2.6 cold-run figure, confirming the cache was warm from that point
on.

---

## 3. Results (AC #4)

### 3.1 Long utterance (Story 17.3 §4.1 paragraph, `chunk_size=25`)

Per-launch, first generation after process start:

| launch | 1a model load | **1b first-forward** | 2 talker | 3 decode | TTFA(post) |
|---|---:|---:|---:|---:|---:|
| before 1 | 4,532 | **3,926** | 1,694 | 183 | 10,334 |
| before 2 | 4,770 | **3,987** | 1,662 | 191 | 10,610 |
| before 3 | 4,466 | **3,586** | 1,529 | 165 | 9,745 |
| before 4 | 4,426 | **3,555** | 1,568 | 144 | 9,694 |
| before 5 | 4,362 | **3,593** | 1,566 | 157 | 9,678 |
| **before median** | **4,466** | **3,593** | **1,568** | **165** | **9,745** |
| after 1 | 0.5 | **86** | 1,378 | 58 | 1,522 |
| after 2 | 0.5 | **85** | 1,338 | 58 | 1,481 |
| after 3 | 0.5 | **80** | 1,379 | 67 | 1,526 |
| after 4 | 1.0 | **90** | 1,592 | 79 | 1,762 |
| after 5 | 0.5 | **87** | 1,527 | 87 | 1,701 |
| **after median** | **0.5** | **86** | **1,379** | **67** | **1,526** |

### 3.2 Short utterance (Clear Comms interjection class)

| launch | 1a model load | **1b first-forward** | 2 talker | 3 decode | TTFA(post) |
|---|---:|---:|---:|---:|---:|
| **before median (n=5)** | **4,795** | **4,442** | **1,878** | **159** | **11,275** |
| **after median (n=5)** | **0.5** | **98** | **1,639** | **86** | **1,849** |

### 3.3 Reading the numbers honestly

**Segment 1b is the term this story owns, and it moves exactly as predicted.**

| cell | 1b before | 1b after | delta |
|---|---:|---:|---:|
| long | 3,593 ms | 86 ms | **−3,507 ms** |
| short | 4,442 ms | 98 ms | **−4,344 ms** |

Story 20.1 §2.5/§2.6 measured the cold first-forward at 3,961 / 4,008 ms and
the warm steady state at 92–99 ms. Both cells reproduce that split and land in
the "~100 ms class" the AC names.

**Segment 1a is NOT this story's win, and must not be claimed as one.** The
harness loads the model lazily on first dispatch, so in the `before` cells the
~4.5 s model load lands inside the measured generation; with `--prime` the
priming absorbs it and the measured run sees ~0.5 ms. **The shipped GUI already
preloads the model at startup** (`app.py:607-618`), so 1a is ~1 ms in
production either way. The production-equivalent comparison therefore subtracts
1a from both sides:

| cell | first-generation TTFA before (TTFA − 1a) | after | Story 20.1 pooled steady state |
|---|---:|---:|---:|
| long | **5,316 ms** | **1,526 ms** | 1,785 ms |
| short | **6,480 ms** | **1,849 ms** | 1,651 ms |

**AC #4's second clause is met.** First-generation TTFA after this change is
1,526 ms (long) and 1,849 ms (short) against Story 20.1's pooled steady-state
1,785 ms / 1,651 ms. Both sit inside the documented 11–37 % session-to-session
spread — long is 15 % *below* pooled steady state, short is 12 % above. **No
delta smaller than that spread is claimed**: the correct statement is *"first
generation is now indistinguishable from steady state"*, not *"first generation
is faster/slower than steady state by X"*.

The claim that *is* far outside the spread, and is the story's result:
**first-generation TTFA drops 3.4×–3.5× (long 5,316 → 1,526 ms; short 6,480 →
1,849 ms).**

### 3.4 Startup cost (AC #4, third clause)

Priming wall-clock, as reported by the harness (`startup priming: ok in …`):

| cell | n | median | min | max |
|---|---:|---:|---:|---:|
| long | 5 | 8,882 ms | 8,817 | 10,167 |
| short | 5 | 9,724 ms | 9,354 | 10,249 |

Those figures **include the lazy model load** (the harness has no preload). The
model-load term measured in the same session is 4,362–4,976 ms (segment 1a of
the `before` cells). The **marginal startup cost of priming, on a host whose
model is already loaded, is therefore ≈ 4.4–4.9 s** — which is, as expected,
the same ~4 s bill, simply moved off the user's first utterance and onto
startup.

**Does priming block the UI thread? No.** Three reasons, all in code:

1. `warmup_compile_async` is dispatched fire-and-forget through
   `MyVoiceApp._run_async_task` → `asyncio.ensure_future` (`app.py:987`). It is
   a coroutine on the qasync loop, never a blocking call.
2. Inside `_generate_true_stream`, the talker runs on its own
   `threading.Thread`, and the coroutine waits for it with
   `await asyncio.sleep(poll_s)` in a polling loop — so the qasync/Qt main
   thread yields continuously for the whole ~4.4 s.
3. Decoding runs on the `StreamingDecoderWorker` thread.

The one real interaction cost is **not** a UI block: `_generate_true_stream`
holds `self._request_semaphore` for the duration of the generation, so a user
generation dispatched while priming is in flight is **serialized behind it**
rather than running concurrently. It completes correctly and its audio reaches
the user — that is exactly what
`test_user_generation_during_priming_still_reaches_consumer` asserts — but it
can wait up to the remaining priming time. This is pre-existing behavior for
any two concurrent generations, and it is bounded by the ~4.4 s figure above.
Worth knowing; not a defect introduced here.

### 3.5 AC #2 confirmed on real hardware

Every one of the 10 `--prime` launches printed:

```
startup priming: ok in ~9,000ms; consumer chunks seen during priming = 0
```

with the consumer wired **before** priming started and counting every chunk it
was ever handed, across a real TRUE_STREAM generation that genuinely produced
audio (each priming run completed successfully and would otherwise have emitted
1–2 chunks plus a terminal chunk). The ordering race is closed positively.

---

## 4. Reversibility (AC #6)

```
MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1
```

Set it and `warmup_compile_async` restores the pre-20.2 warm-cache path
**exactly**: no priming generation, no indicator, one telemetry record with
`reason="cache_hit"` and `value=0.0`, immediate return. Pinned by
`test_warm_priming_env_gate_restores_pre_story_cache_hit`.

Deliberately a **separate** switch from `MYVOICE_DISABLE_COMPILE_WARMUP`: that
one disables warmup entirely, including the cold-path priming Story 18.4
shipped. This story's gate reverts only what this story added.

---

## 5. Regression sweep (AC #5, Task 4)

`python310\python.exe -m pytest -q`, portable interpreter per
`memory/test_interpreter_portable_python310.md`:

| surface | result |
|---|---|
| `tests/unit/services tests/unit/observability tests/unit/models` | **896 passed** |
| `tests/integration tests/test_qwen_tts_internals.py` | 174 passed, **4 failed (all pre-existing)** |
| the two above combined in one invocation | 1,057 passed, **9 failed** — the same 4 plus 3 `clear_comms` rows that fail only in the combined ordering. Verified identical on the stashed baseline (9 failed there too), so it is a pre-existing cross-suite state-leak, not this story. |
| `tests/services tests/settings tests/utils` | 288 passed, **7 failed (all pre-existing)** |
| `tests/ui` | 711 passed, **7 failed (all pre-existing)** |
| `tests/unit` (whole tree) | 1,503 passed, 30 failed + 4 errors — **all outside `tests/unit/services`**, all pre-existing |
| new: `tests/unit/services/test_compile_priming_audio_suppression.py` | **12 passed** |
| updated: `tests/unit/services/test_qwen_tts_service_compile_warmup.py` | **12 passed** (was 7) |

**Every failure was verified pre-existing** by stashing
`src/myvoice/services/qwen_tts_service.py` +
`tests/unit/services/test_qwen_tts_service_compile_warmup.py` and re-running
the failing files: identical failure sets, identical counts. They are UI /
voice-profile / session-manager drift and a Windows temp-file lock in
`voice_design_studio/test_audio_player_widget.py` — none touch the streaming
dispatch chain. The touched surface (`tests/unit/services`, `tests/integration`
streaming rows) is **clean**.

> Note on the "514 tests" figure in AC #5: Story 20.1's 514 was the sum of two
> hand-picked pytest invocations whose exact path lists were not recorded. The
> sweep above is a superset (2,600+ selected tests across the whole tree) and is
> reproducible from the commands as written.

**Mutation check.** Every fix in this story is reverted in turn and the two
story test files re-run (harness: `scratchpad/mutate.py`, reproduced in §7.3).
**8 of 8 mutations are caught.** No fix in this story rests on an assertion that
does not exist.

---

## 6. ⚠ Load-bearing finding: the win is not yet reachable from the GUI

**`warmup_compile_async` never gets past its `no_model_loaded` fast-exit in the
shipped application.** This predates Story 20.2 — it is a Story 18.4 call-site
defect — but it means the measured 3.4× first-generation improvement is **not
delivered to users** by this story alone.

The mechanism, in `app.py::_initialize_services`:

```
:593   self._run_async_task(                                   # -> asyncio.ensure_future
:594       self._tts_service.warmup_compile_async(), ...)
:607+  preferred_model = self._voice_manager.get_active_profile_model_type()
:613   success, error = await self._tts_service.preload_model(preferred_model)
```

`_run_async_task` calls `asyncio.ensure_future` (`app.py:987`), so the warmup
coroutine is *scheduled*, not run. It first executes at the enclosing
coroutine's next suspension point — which is the `await preload_model(...)` on
line 613. At that instant no model is loaded (`QwenTTSService.start()` is
explicitly lazy: *"models will load on first use"*, `qwen_tts_service.py:745`),
and every statement in `warmup_compile_async` between entry and the
`get_loaded_model()` check is synchronous. So the warmup runs straight through
to:

```
reason="no_model_loaded"   → return
```

deterministically, on every launch. Both the cold path (18.4) and the new warm
path (20.2) are unreachable.

### Why the obvious fix is wrong

Moving the warmup call below the preload block is a two-line change — and it
would make things **worse** on a large fraction of launches:

* `_run_compile_priming` dispatches a **CUSTOM_VOICE** generation.
* `ModelRegistry.ensure_model_loaded` keeps **exactly one** model resident:
  *"If a different model is currently loaded, it will be unloaded first"*
  (`model_registry.py:329`, `:364-365`).
* A user whose active voice is CLONED preloads **BASE** (`app.py:607-618` →
  `voice_profile_service.py:1383-1398`).

So on a CLONED-voice launch the sequence would become: preload BASE (~4.5 s) →
priming unloads BASE and loads CUSTOM_VOICE (~4.5 s) → primes CUSTOM_VOICE's
graph → user generates → unload CUSTOM_VOICE, reload BASE (~4.5 s) **and pay a
fresh engage for BASE**. Strictly worse than today.

### The correct fix, and why it is not in this story

Priming must use the **model type that is actually loaded**. For CUSTOM_VOICE
that is today's path; for BASE it needs a `voice_clone_prompt` (Story 17.2's
hydrated cache can supply one, but that is a new dispatch shape with new
failure modes). That is a design decision beyond this story's stated scope
("Scoped deliberately narrow… the story's weight sits on AC #2/#3"), and the
story's Dev Notes forbid dispatch-chain changes. It is written up as
**Follow-up A′** below rather than smuggled in here.

**What this story does deliver, unconditionally:** the audio-suppression hazard
is closed for *both* priming paths (that was the story's stated weight, and it
is a live safety fix regardless of reachability), the warm-path logic and its
telemetry are in place and tested, and the value of fixing the call site is now
**measured** rather than estimated — §3 is the business case for Follow-up A′.

### Follow-up A′ (recommended next, HIGH)

1. Move the `warmup_compile_async` firing below the model-preload block in
   `app.py::_initialize_services`.
2. Make `_run_compile_priming` prime the **currently loaded** model type:
   CUSTOM_VOICE via today's path; BASE via a hydrated `voice_clone_prompt`
   (skip with a new telemetry reason, e.g. `no_priming_prompt`, when none is
   available — never thrash the registry).
3. Re-measure with the same four cells; §3 is the baseline.

---

## 7. Review pass — the suppression wall was one channel wide

An adversarial review of commit `2f19b64` found that this story's headline
claim — *"audio chunks from the priming generation reach no consumer, enforced
by an explicit mechanism"* — was **false as shipped**. The mechanism was real,
but it guarded a single channel. Every finding was reproduced against the
source before being fixed; none is disputed.

### 7.1 Findings and fixes

**F1 (HIGH) — TRUE_STREAM bypassed the wall entirely.**
`_generate_true_stream` awaits `audio_coordinator.play_dual_stream(audio_data=...,
session_id=sid)` with **no** request consultation. That call *is* the monitor
device plus the virtual microphone. Priming always resolves to TRUE_STREAM — it
only runs on Ampere+ CUDA, which is exactly when `_resolve_streaming_mode()`
returns TRUE_STREAM. Verified at the call site; the review's probe on this
suite's own rig showed chunk-callback calls 0, `play_dual_stream` calls 1. It is
dormant in the shipped GUI **only** because `app.py:492` happens to construct
the service without an `audio_coordinator` — safety by coincidence, the exact
class of hazard this story exists to remove.
*Fix:* the playback block is gated on `not suppressed`;
`coordinator.play_dual_stream.assert_not_called()` is asserted in both AC #2
rows via `ChannelSpies`.

**F4 (MEDIUM-HIGH) — priming promoted itself to saveable and focal.**
Priming registered a `source=GENERATED` session, so the registry made it
`_saveable` and `_focal`. Unlike F1 this one is **live**: `app.py:492` *does*
pass `session_registry`. User-visible result: the Save button lit at startup
pointing at "Hello world."; a prime finalising after a real generation demotes
the user's audio to `_previous_saveable`; `_play_generated_audio` can snapshot
the prime as focal, breaking the Stop / Clear Comms focal-cancel paths.
*Fix:* suppressed generations create no session (`sid` stays `None`) in all
three dispatch paths, and `registry_post` is nulled in TRUE_STREAM so the
worker's `append_chunk` / `finalize` posts cannot reach the registry either.
Asserted via `focal_session_id`, `saveable_session_id`, and a `create_session`
spy.

**F3 (MEDIUM-HIGH) — priming overwrote the shared audio cache.**
`_save_audio_to_cache` ran unconditionally at `:4802`, and
`get_cached_audio_path()` is what `_on_replay_last_requested` plays. Once
Story 20.3 activates priming, every launch would leave `myvoice_current.wav`
containing "Hello world." and Replay Last would play it audibly.
*Fix:* the cache write is skipped for suppressed requests in **all three**
dispatch paths (`:3350`, `:3675`, `:4802`). Because
`_generation_complete_callback` is guarded on a truthy `audio_file`, this closes
that channel too.

**F2 (HIGH) — priming's `finally` clobbered a concurrent generation's cancel
bookkeeping.** `_current_session_id` and `_current_generation_task` are claimed
at `:4429-4440`, **before** `async with self._request_semaphore` at `:4522`. So
a user request registers its bookkeeping, parks behind the in-flight prime, and
the prime's `finally` nulls it out: the user presses Speak during startup, then
Stop, and the audio does not stop. The review correctly noted that this suite's
own concurrency test creates exactly that interleaving and asserted nothing
about it.
*Fix, in two parts:* (a) a suppressed generation claims neither singleton, and
does not reset `_cancel_requested` either — resetting it would swallow a Stop
the user had just pressed; (b) every `finally` in all three paths now releases
**only what it claimed** (`is`-identity on the task, `==` on the sid), which
fixes the race for any two concurrent generations, not only priming.
`test_user_generation_during_priming_still_reaches_consumer` now asserts that
the parked user generation's sid and task survive the prime's finally, and
`test_priming_does_not_clear_a_pending_user_cancel` covers the cancel flag.

**F5 (MEDIUM) — the source-invariant test gave false confidence.** It grepped a
single string and so was structurally blind to F1, F3 and F4, while its name
promised "no emit site reads the raw callback" and the AC promised "no chunk
reaches the user". The rig also stubbed `play_dual_stream` and never counted it,
so the suite exercised the leak and reported green.
*Fix:* the suite is rebuilt around a `ChannelSpies` recorder covering all four
channels plus `_generation_complete_callback` and the registry's focal/saveable
slots. `test_every_user_facing_channel_call_site_is_suppression_gated` is a
source invariant over *channels* (and fails if a channel's marker stops matching
anything, so a rename cannot silently retire the protection).
`test_unsuppressed_request_uses_every_channel` is the control that proves the
rig can reach each channel, so "untouched" means something.
`test_docstring_no_longer_claims_ordering_safety` is dropped — a prose
assertion with no behavioural value should not be counted toward coverage.

**F6 (LOW) — the predicate fails open.** `_is_suppressed` reads the field
through `getattr(..., False)`, so a rename would silently un-suppress priming.
*Fix:* `_run_compile_priming` re-checks the predicate on the exact object about
to be dispatched and raises (routing to the non-fatal `priming_failed`
telemetry) rather than dispatching. The dataclass default stays `False` —
**silencing a real user is the worse failure**, so the flag is not inverted.

**F7 (LOW) — the warm-path `finally` erased someone else's indicator.** It
cleared the single-slot preparing-voice indicator unconditionally, wiping a
concurrent Story 17.2 "Preparing voice for streaming…" message; both producers
run at startup.
*Fix:* `_emit_preparing_voice` records the last message and
`_clear_preparing_voice_if_mine` clears only if the slot still shows our own.
Applied to both the warm and the cold path.

**Also closed — the SENTENCE_STREAM `chunk_request` copy** at `:3599-3613`
dropped the flag. Harmless today (the copy only reaches `_generate_sync`, which
never emits), but a request-copy site that silently drops the flag becomes a
leak the moment anyone routes it through a dispatcher. Flag copied, with a
source-invariant test on the copy site.

The review also reported two clean areas, and both re-check out: the flag cannot
leak onto a user request (7 construction sites, only priming sets it `True`, no
`replace` / `asdict` / shared object), and the TRUE_STREAM → SENTENCE_STREAM →
BATCH fallback chain preserves it correctly.

### 7.2 Why the fix is structured this way

The temptation with F1 / F3 / F4 is three one-line guards. What was actually
wrong is that "suppressed" had been implemented as *"the chunk callback returns
None"* rather than as a property of the generation. So the fix introduces one
predicate, `_is_suppressed(request)`, whose docstring **enumerates the channels**
it is responsible for, and every gate reads that predicate. The enumeration is
load-bearing: it is what a future author reads when they add channel five, and
it is what `test_every_user_facing_channel_call_site_is_suppression_gated`
mechanically enforces.

The F2 fix is deliberately broader than priming. Ownership-scoped release in
`finally` is correct for *any* two concurrent generations — priming is merely
the case that made it observable. Fixing only the priming case would have left
the same bug reachable by a second user dispatch.

### 7.3 Mutation results — 8 of 8 caught

Each fix reverted in turn, the two story test files re-run
(`scratchpad/mutate.py`):

```
F1 play_dual_stream ungated                     CAUGHT   1 failed, 23 passed
F3 cache write ungated (TRUE_STREAM)            CAUGHT   3 failed, 21 passed
F4 registry session ungated (TRUE_STREAM)       CAUGHT   4 failed, 20 passed
F2 finally clears unconditionally (TRUE_STREAM) CAUGHT   1 failed, 23 passed
F2b prime resets _cancel_requested              CAUGHT   1 failed, 23 passed
sink returns the callback regardless            CAUGHT   4 failed, 20 passed
F7 indicator cleared unconditionally            CAUGHT   1 failed, 23 passed
F6 trip-wire removed                            CAUGHT   1 failed, 23 passed
```

The pre-review suite scored 4 of 9 on a single mutation, and had **no** coverage
at all for F1, F2, F3, F4, F6 or F7.

### 7.4 Re-verification after the fixes

* **Story tests:** 12 + 12 = **24 passed**.
* **Regression sweep, re-run in full:** `tests/unit/services` +
  `tests/unit/observability` + `tests/unit/models` = **902 passed** (was 896;
  +6 new rows). `tests/integration` + pin trip-wire = 174 passed, 4 failed.
  `tests/services tests/settings tests/utils tests/ui` = 999 passed, 14 failed.
  **The failing counts are identical to the pre-fix baseline**, and every one was
  previously verified pre-existing by stashing the source change.
* **Hardware, post-fix** (`20-2-after-long-postfix-r0{1,2}.csv`): priming still
  succeeds, still reports `consumer chunks seen during priming = 0`, and
  segment 1b stays in the warm class (90 ms / 130 ms) with TTFA 1,755 / 2,133 ms
  — unchanged in class from the §3 tables. The headline measurements stand.

**Scope note on the hardware runs.** `tools/ttfa_spike_harness.py` constructs
the service with `audio_coordinator=None` and `session_registry=None`, so the
hardware runs exercise channel 1 only. Channels 2-4 are covered by the unit rig,
which wires a **real** `AudioCoordinator` and a **real** `SessionRegistry`
before priming starts. That split is deliberate — the hardware run measures
latency, the unit rig proves the wall — but it is why "12/12 launches reported
zero chunks" is not, by itself, evidence for F1, F3 or F4.

### 7.5 Standing risk for Story 20.3

Every one of these is a **latent** hazard today for the reason §6 describes:
`warmup_compile_async` short-circuits at `no_model_loaded` on every GUI launch,
so priming never runs in the shipped app. Story 20.3 activates it. At that
point F4 would have been live immediately (the registry *is* wired), F3 would
have been live on the first Replay Last, and F1 would have been one
`audio_coordinator` wiring change away from audible. They are fixed ahead of
that, as intended.

---

## 8. File list

**Source**

* `src/myvoice/services/qwen_tts_service.py`
  * `QwenTTSRequest.suppress_audio_output` field + docs
  * `QwenTTSService._is_suppressed(request)` (new) — the single predicate, whose
    docstring enumerates the four user-facing channels it governs
  * `QwenTTSService._audio_chunk_sink(request)` (new) — channel 1
  * three chunk-emit sites rewired through the sink
  * **review F1** — `audio_coordinator.play_dual_stream` gated (channel 2)
  * **review F3** — `_save_audio_to_cache` gated in all three dispatch paths
    (channel 3), which also closes `_generation_complete_callback`
  * **review F4** — `SessionRegistry.create_session` gated in all three paths
    and `registry_post` nulled in TRUE_STREAM (channel 4)
  * **review F2** — suppressed generations claim neither
    `_current_session_id` nor `_current_generation_task` and do not reset
    `_cancel_requested`; every `finally` now releases only what it claimed
  * **review F6** — `_run_compile_priming` raises rather than dispatching an
    unprovably-suppressed request
  * **review F7** — `_emit_preparing_voice` records the last message;
    `_clear_preparing_voice_if_mine` (new) replaces the unconditional clear on
    both the warm and cold paths; `_PREPARING_TTS_ENGINE_MESSAGE` constant
  * SENTENCE_STREAM `chunk_request` copy now carries the flag
  * `warmup_compile_async` — warm-path priming, `primed_warm` telemetry,
    `MYVOICE_DISABLE_WARM_COMPILE_PRIMING` gate, docstring rewrite
  * `_COMPILE_PRIMING_TEXT` unchanged; new
    `_WARM_COMPILE_PRIMING_DISABLE_ENV` constant
  * `_run_compile_priming` — builds its own suppressed request and dispatches
    through `_dispatch_by_streaming_mode` (instead of `generate_custom_voice`,
    so `suppress_audio_output` stays off the public signature); docstring's
    ordering claim deleted
  * `tts_compile="off"` gate comment amended (gate itself unchanged)

**Tests**

* `tests/unit/services/test_compile_priming_audio_suppression.py` (NEW, 12
  tests — rewritten in the review pass to spy on all four channels, with a
  control row proving the rig can reach each one)
* `tests/unit/services/test_qwen_tts_service_compile_warmup.py` (MODIFIED —
  the cache-hit row became three rows: warm prime, failed warm prime, env gate;
  plus the no-model-registry fast-exit row and two F7 indicator rows; 7 → 12)

**Measurement (tools + artifacts, not shipped)**

* `tools/ttfa_spike_harness.py` — `--prime` flag, `PRIMING_TEXT`,
  `ConsumerSim.total_chunks_seen`
* `_bmad-output/implementation-artifacts/20-2-{before,after}-{long,short}-r0{1..5}.csv`
* `_bmad-output/implementation-artifacts/20-2-after-long-postfix-r0{1,2}.csv`
  (post-review-fix confirmation, §7.4)
* `_bmad-output/implementation-artifacts/20-2-capture-{long,short}.sh`,
  `20-2-aggregate.py`

**Untouched:** `requirements.txt`, `build_tools/*`, the bundled `python310/`
tree, `torch_runtime.py`, the Story 16.8 forward hook, `DEFAULT_CHUNK_SIZE`,
the adaptive cushion, and `compute_key`'s `decode_window_frames=30`.

---
