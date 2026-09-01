# Story 20.2 — Warm-Path Compile Priming: evidence

Phase ⊥-Polish-3 · Epic 20 (First-Audio Latency) · captured 2026-08-31 on the
RTX 5090 host (`torch 2.10.0+cu128`, capability 12.0, 31.8 GiB).

---

## 0. Headline

| claim | verdict |
|---|---|
| Priming on the warm cache collapses segment 1b from the ~3.96 s class to the ~100 ms class | **CONFIRMED** — long: 3,593 → 86 ms; short: 4,442 → 98 ms (medians, n=5 each) |
| First-generation TTFA lands in the same band as steady-state TTFA | **CONFIRMED** — long 1,526 ms vs Story 20.1 pooled 1,785 ms; short 1,849 ms vs pooled 1,651 ms |
| Priming reaches no audio consumer, with a consumer wired from process start | **CONFIRMED** on real hardware (`consumer chunks seen during priming = 0`, 10/10 launches) and in 9 unit tests |
| Reversible | **YES** — `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1` |
| The change is reachable from the shipped GUI | **NO — see §6.** `warmup_compile_async` is fired *before* the model preload in `app.py`, so it short-circuits at `no_model_loaded` on every launch. This is a **pre-existing Story 18.4 defect**, not one this story introduces, but it means the measured win is not yet delivered to users. |

---

## 1. What changed

### 1.1 AC #2 / #3 — the audio-suppression mechanism (Task 1)

**Chosen mechanism: a per-request flag, read through a single sanctioned
resolver.**

* `QwenTTSRequest.suppress_audio_output: bool = False`
  (`qwen_tts_service.py`, dataclass at :290).
* `QwenTTSService._audio_chunk_sink(request)` — returns the wired
  `_audio_chunk_ready_callback` for an ordinary request and `None` for a
  suppressed one. It is the **only** place in the module that reads the
  callback attribute.
* All three chunk-emit sites now resolve through it:
  SENTENCE_STREAM (`_generate_streaming`), and both TRUE_STREAM sites in
  `_wrapped_post` (the per-`append_chunk` data chunk and the synthetic
  terminal chunk on `finalize`).

**Why this one, and not the alternatives the story offered (Task 1.1).**

| candidate | rejected because |
|---|---|
| detach-and-restore `_audio_chunk_ready_callback` around the priming call | Service-wide for the duration of priming. Priming now runs on **every** launch and takes ~4.4 s; a user generation dispatched inside that window would have its audio silenced too. That is AC #3's named failure mode — "a latency fix turned into a no-audio bug". |
| a service-level `self._suppressing` boolean | Same defect, same reason. Scoped to a *window of time*, not to a generation. |
| **per-request flag + single resolver** | The suppression decision travels with the object it describes. Two concurrent generations cannot confuse it. It holds regardless of when `set_audio_chunk_ready_callback` ran. |

**What makes it hard to get wrong from a future caller's perspective.** A
per-request flag is only as strong as the discipline that every emit site asks
the resolver. That discipline is now **mechanical**, not documentary:
`TestSuppressionMechanismIsHardToBypass::test_no_emit_site_reads_the_raw_callback`
scans `qwen_tts_service.py` and fails if `self._audio_chunk_ready_callback`
appears anywhere except its declaration, its setter, and the single `return`
inside `_audio_chunk_sink`. A future emit site that reaches for the attribute
directly breaks the build.

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
| new: `tests/unit/services/test_compile_priming_audio_suppression.py` | **9 passed** |
| updated: `tests/unit/services/test_qwen_tts_service_compile_warmup.py` | **10 passed** (was 7) |

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

**Mutation check on the new tests.** With `_audio_chunk_sink` reduced to
`return self._audio_chunk_ready_callback` (i.e. the suppression removed), 4 of
the 9 new tests fail — including both AC #2 rows and the AC #3 row. The suite
is not vacuous.

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

## 7. File list

**Source**

* `src/myvoice/services/qwen_tts_service.py`
  * `QwenTTSRequest.suppress_audio_output` field + docs
  * `QwenTTSService._audio_chunk_sink(request)` (new)
  * three chunk-emit sites rewired through it
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

* `tests/unit/services/test_compile_priming_audio_suppression.py` (NEW, 9 tests)
* `tests/unit/services/test_qwen_tts_service_compile_warmup.py` (MODIFIED —
  the cache-hit row became three rows: warm prime, failed warm prime, env gate)

**Measurement (tools + artifacts, not shipped)**

* `tools/ttfa_spike_harness.py` — `--prime` flag, `PRIMING_TEXT`,
  `ConsumerSim.total_chunks_seen`
* `_bmad-output/implementation-artifacts/20-2-{before,after}-{long,short}-r0{1..5}.csv`
* `_bmad-output/implementation-artifacts/20-2-capture-{long,short}.sh`,
  `20-2-aggregate.py`

**Untouched:** `requirements.txt`, `build_tools/*`, the bundled `python310/`
tree, `torch_runtime.py`, the Story 16.8 forward hook, `DEFAULT_CHUNK_SIZE`,
the adaptive cushion, and `compute_key`'s `decode_window_frames=30`.
