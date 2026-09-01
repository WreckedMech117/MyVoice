# Story 20.2: Warm-Path Compile Priming (Phase ⊥-Polish-3)

Status: review

<!-- Phase tag: Phase ⊥-Polish-3. Second story of Epic 20 (First-Audio Latency). -->
<!-- Source: Story 20.1 evidence §2.5 + §6.4 Follow-up A (the highest value-per-hour item the spike identified). -->
<!-- Risk: MEDIUM. Small code change, but it runs a real TRUE_STREAM generation on a path where the existing audio-suppression guarantee is an ordering assumption, not a guarantee — and that assumption has already failed once in production. -->

## Story

As **a MyVoice user who has launched the app before**,
I want **the first utterance after launch to be as fast as every later one**,
so that **the interjection I actually reach for — the first one — is not the slowest thing the app does**.

## Context

Story 20.1 established that Epic 18's headline TTFA number was a
first-generation-of-process artifact, and located the cause precisely
(`20-1-ttfa-spike-faster-qwen3-tts-evidence.md` §2.5):

| regime | segment 1b (first forward) | segment 2 (talker) | TTFA |
|---|---:|---:|---:|
| compile, run 0 (cold) | **3,961 ms** | 1,910 ms | 10,485 ms |
| compile, runs 1-3 (warm) | 98-99 ms | ~1,570 ms | ~1,770 ms |

`warmup_compile_async` (`qwen_tts_service.py:1918`) runs a priming generation
**only when `compile_cache.is_warm(key)` is False**. On the warm-cache path —
which is every launch after the first-ever — it logs a `cache_hit` breadcrumb
and returns, deliberately, per its own docstring: *"the inductor cache reloads
from disk lazily on first user-facing generation."*

That lazy reload is the ~3.96 s bill, and it is handed to the user's first
utterance on every single launch. Priming on the warm path too moves it into
startup, where nobody is waiting on it.

**This is the largest single term in the number Epic 18 measured, and the
cheapest to remove.**

## The hazard this story must close

`_run_compile_priming` (`qwen_tts_service.py:2153`) dispatches a **real
TRUE_STREAM generation** through `generate_custom_voice`. Its docstring claims
audio safety on two grounds, and **both are assumptions rather than guarantees**:

> *"No audio output reaches the user (the priming runs before the
> `set_audio_chunk_ready_callback` wires consumers up — and even if a consumer
> is wired, the generation is short enough that any audible artifact is
> bounded)."*

The first is a startup-ordering race. The second is not a safety property at
all — "bounded" audible audio is still audible audio.

**We know the race can be lost.** The `tts_compile="off"` gate at
`qwen_tts_service.py:1990-2005` exists because it *was* lost: its comment
records that without the gate, a first launch on Ampere+ CUDA produced
audible *"Hello world."* spurious audio on the user's speakers.

Priming on the warm path makes this strictly more dangerous — it runs on every
launch instead of one, so a rare race becomes a recurring one.

## Acceptance Criteria

### AC #1 — Warm-path priming

**Given** an Ampere+ CUDA host with `tts_compile="auto"`, a loaded model, and
`compile_cache.is_warm(key)` returning **True**
**When** `warmup_compile_async` runs at startup
**Then** it runs a priming generation instead of returning early
**And** it emits a distinct telemetry reason — `"primed_warm"` — so warm-path
priming is separable from the existing `"primed_cold"` and `"cache_hit"` in the
metric stream
**And** it does **not** call `compile_cache.mark_warm(key)` (already warm; the
cold path's contract is unchanged)
**And** on priming failure it lands the existing `"priming_failed"` telemetry
and the app continues normally — a failed warm prime must never be fatal, since
the pre-story behavior (lazy reload on first generation) remains a working
fallback

### AC #2 — Audio suppression is positive, not ordering-dependent (load-bearing)

**Given** the priming generation dispatches through the production TRUE_STREAM path
**When** priming runs — on either the cold or the warm path
**Then** audio chunks from the priming generation reach **no** consumer, and this
is enforced by an **explicit mechanism**, not by startup ordering or by the
utterance being short
**And** the mechanism holds even if `set_audio_chunk_ready_callback` has already
wired a consumer when priming starts
**And** the existing cold-path priming is brought under the same mechanism — this
story fixes the latent race for both paths, not just the one it adds
**And** a regression test proves it: wire a recording consumer **before** priming,
run priming, assert the consumer received **zero** chunks

> Implementation latitude: detach-and-restore around the priming call, an
> explicit suppression flag the emit sites honour, or a priming-specific request
> flag that routes chunks nowhere. Pick the one that is hardest to get wrong from
> a future caller's perspective and say why in the Dev Notes. Do **not** rely on
> "it runs before the callback is wired."

### AC #3 — A user generation during priming is never blocked or corrupted

**Given** priming takes ~4 s and now runs on every launch
**When** the user triggers a real generation while priming is still in flight
**Then** the user's generation completes correctly and its audio reaches the user
**And** the user's audio is never suppressed by AC #2's mechanism (the suppression
must be scoped to the priming generation, not to a window of time)
**And** the interaction is covered by a test that starts priming, dispatches a
user generation before it completes, and asserts the user's chunks arrive

> This is the sharpest risk in the story. A suppression flag scoped by time or by
> a service-wide boolean will silence the user's first utterance — turning a
> latency fix into a "no audio" bug.

### AC #4 — Measured, with the instrumentation Story 20.1 shipped

**Given** the six `ttfa_*` boundary metrics and the env-var CSV capture are now
shipped surface
**When** the developer measures first-generation-after-launch TTFA on the RTX 5090,
warm cache, `tts_compile="auto"`, ≥5 launches each, before and after
**Then** segment 1b (first-forward) on the user's first generation drops from the
~3.96 s class to the ~100 ms class
**And** first-generation TTFA lands in the same band as steady-state TTFA
(Story 20.1 pooled: **1,785 ms** long / **1,651 ms** short, with the documented
11-37 % session-to-session spread — do not claim a delta smaller than that spread)
**And** the **startup cost** is reported explicitly: how much later the app becomes
ready-to-generate, and whether priming blocks the UI thread at any point
**And** results land at
`_bmad-output/implementation-artifacts/20-2-warm-path-compile-priming-evidence.md`

### AC #5 — Existing gates preserved

**Given** the D-9 / NFR12 hardware-aware discipline
**When** the new path is added
**Then** every existing fast-exit still fires unchanged: `MYVOICE_DISABLE_COMPILE_WARMUP=1`,
non-Ampere / CPU hosts, `tts_compile="off"`, no model registry, no model loaded
**And** the CPU-only and pre-Ampere paths are behaviorally identical to before
**And** the full regression surface passes with zero regressions (Story 20.1 closed
at **514** tests)

### AC #6 — Reversible

**Given** this changes startup behavior on every launch
**When** it ships
**Then** warm-path priming is gated by a setting or env var that restores the
pre-story behavior exactly, and the gate is documented in the evidence file

## Tasks / Subtasks

- [x] **Task 1 — Close the audio-suppression race** (AC: #2, #3)
  - [x] 1.1 Choose the suppression mechanism; record the choice and its rationale in Dev Notes. Prefer one scoped to the priming *generation*, not to a time window or a service-wide flag (AC #3's failure mode).
  - [x] 1.2 Apply it to `_run_compile_priming` so it covers the existing cold path as well as the new warm path.
  - [x] 1.3 Delete or correct the docstring's ordering claim — it must no longer assert a guarantee the code does not provide.
  - [x] 1.4 Regression test: consumer wired *before* priming receives zero chunks.
  - [x] 1.5 Regression test: user generation dispatched mid-priming receives its chunks.

- [x] **Task 2 — Warm-path priming** (AC: #1, #5, #6)
  - [x] 2.1 Replace the `is_warm` early return with a priming call; keep `mark_warm` on the cold path only.
  - [x] 2.2 Add the `"primed_warm"` telemetry reason; keep `"cache_hit"` meaningful (e.g. when the gate in 2.3 disables warm priming).
  - [x] 2.3 Add the AC #6 gate restoring pre-story behavior.
  - [x] 2.4 Verify all five existing fast-exits still fire.

- [x] **Task 3 — Measure** (AC: #4)
  - [x] 3.1 Capture ≥5 launches before and after via `MYVOICE_PROGRESSIVE_PLAYBACK_CSV`, warm cache, `tts_compile="auto"`.
  - [x] 3.2 Report segment 1b and total TTFA on the first user generation, against Story 20.1's pooled baseline and its 11-37 % spread.
  - [x] 3.3 Report the startup-time cost and whether the UI thread ever blocks.
  - [x] 3.4 Write the evidence file.

- [x] **Task 4 — Regression sweep** (AC: #5)
  - [x] 4.1 Run the Story 20.1 surface (514 tests) plus the new tests; zero regressions.

## Dev Notes

### What this story is NOT

- **Not a chunk-size change.** That is Follow-up B, and Story 20.1 found it is
  coupled to Follow-up C (`chunk_size=10` worsens the sub-16 GiB cushion-to-talker
  ratio from 2.5× to 4.0×). B and C ship together, in a later story. Do not touch
  `DEFAULT_CHUNK_SIZE` here.
- **Not an adaptive-cushion change.** Follow-up C, later, with B.
- **Not a port.** PORT-b is Follow-up E and its marginal value is to be re-measured
  *after* this story lands, against the new baseline — not against today's.
- **Not a dispatch-chain change.** The TRUE_STREAM path, the Story 16.8 forward
  hook, and `torch_runtime.py` are all out of scope.

### Why this one first

Story 20.1 ranked the follow-ups by measured value per engineering hour. A is
first because it is the largest single term (~3.96 s), it is independent (B and C
are coupled to each other; A is not), it is reversible, and it does not touch the
audited dispatch chain. It also directly retires the number Epic 18 measured,
which makes every later comparison in Epic 20 cleaner.

### Known trap

`compile_cache.compute_key(...)` in `warmup_compile_async` hard-codes
`decode_window_frames=30` (`qwen_tts_service.py:2078`), matching
`engage_compile_optimizations`' own hard-coded default. Story 20.1 §5.4 found this
means the cache key never varies with the streamer's real geometry. **Leave it
alone in this story** — it is correct today and threading the real geometry is
Follow-up B's job. Just do not "fix" it here; changing it would invalidate every
existing warm cache directory for no benefit.

## References

- `_bmad-output/implementation-artifacts/20-1-ttfa-spike-faster-qwen3-tts-evidence.md` §2.5 (the cold/warm split), §6.4 Follow-up A (this story's scope), §2.1 (the shipped `ttfa_*` metrics this story measures with)
- `src/myvoice/services/qwen_tts_service.py:1918` (`warmup_compile_async`), `:2153` (`_run_compile_priming`), `:1990` (the `tts_compile="off"` gate whose comment records the audible-audio incident)

## Dev Agent Record

### Agent Model Used

claude-opus-5[1m] (Claude Code, subagent)

### Completion Notes List

**AC #2 mechanism (Task 1.1 rationale).** Chose a **per-request flag**
(`QwenTTSRequest.suppress_audio_output`) read through a single predicate
(`QwenTTSService._is_suppressed(request)`). Rejected detach-and-restore of
`_audio_chunk_ready_callback` and a service-level suppression boolean: both are
scoped to a *window of time*, so both would silence a user generation dispatched
while the ~4.4 s priming is in flight — AC #3's named failure mode. The
per-request flag travels with the generation it describes, so concurrency is a
non-issue and startup ordering is irrelevant.

**⚠ The first implementation of this got the scope of "consumer" wrong, and an
adversarial review caught it (evidence §7).** Suppression was implemented as
"the chunk callback resolves to None" rather than as a property of the
generation, so it guarded ONE channel while three others reached the user
untouched: `audio_coordinator.play_dual_stream` (the monitor device + virtual
mic — TRUE_STREAM calls it directly and priming always resolves to TRUE_STREAM),
`_save_audio_to_cache` (writes `myvoice_current.wav`, which Replay Last plays),
and `SessionRegistry.create_session` (makes the prime `_saveable` / `_focal`,
lighting the Save button at startup and breaking focal-cancel). The review also
found that priming's `finally` clobbered a concurrent user generation's
`_current_session_id` / `_current_generation_task` — those are claimed *before*
the request semaphore, so a user request could park behind the prime and lose
its cancel bookkeeping, leaving Stop unable to stop audio (AC #3's failure mode
arriving through a different door). All four channels are now gated on
`_is_suppressed`, whose docstring enumerates them; the `finally` blocks in all
three dispatch paths release only what they claimed; and the trip-wire in
`_run_compile_priming` raises rather than dispatching a request it cannot prove
is suppressed. **8 of 8 mutations are now caught** (harness:
`20-2-mutation-harness.py`); the pre-review suite scored 4 of 9 on one mutation
and had zero coverage for any of the leaks.

**AC #2 covers the cold path too.** `_run_compile_priming` is the single entry
point for both paths, and it is what carries the flag — so Story 18.4's
cold-path priming is now under the same guarantee.
`test_cold_path_warmup_reaches_no_consumer` drives the full
`warmup_compile_async` cold branch end-to-end with a consumer wired first.

**AC #1 / #6.** Warm path primes, emits `reason="primed_warm"`, never calls
`mark_warm`, is non-fatal on failure. `MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1`
restores the exact pre-story early return (`reason="cache_hit"`).

**AC #4 measured, four cells, n=5 launches each, RTX 5090, warm cache,
`tts_compile="auto"`.** Segment 1b: long 3,593 → 86 ms; short 4,442 → 98 ms.
Production-equivalent first-generation TTFA (model-load term removed, since
`app.py` preloads): long 5,316 → 1,526 ms; short 6,480 → 1,849 ms — both inside
Story 20.1's pooled steady-state band (1,785 / 1,651 ms) given the documented
11-37 % spread. Startup cost ≈ 4.4-4.9 s marginal; the Qt main thread is never
blocked (the priming coroutine yields with `await asyncio.sleep` while the
talker runs on its own thread). All 10 primed launches reported
`consumer chunks seen during priming = 0` with the consumer wired before
priming started.

**⚠ Blocking finding for the story's user-visible value — see evidence §6.**
`warmup_compile_async` is fired at `app.py:593`, *above* the model preload at
`app.py:613`, via `asyncio.ensure_future`. It therefore first runs at that
preload's `await`, when no model is loaded, and short-circuits at
`reason="no_model_loaded"` — deterministically, every launch. Both the Story
18.4 cold path and this story's warm path are currently unreachable from the
GUI. Moving the call below the preload is NOT a safe two-line fix: priming
dispatches CUSTOM_VOICE, the registry keeps exactly one model resident, and a
CLONED-voice user preloads BASE — so the naive move would thrash the registry
and be strictly worse. The correct fix (prime the model type that is actually
loaded) is a dispatch-shape decision the story's Dev Notes place out of scope.
Written up as **Follow-up A′** in the evidence file, with §3 as its business
case. This story's audio-safety fix (AC #2/#3) lands unconditionally and is
valuable regardless.

**Regression sweep (re-run after the review fixes).** Touched surface clean:
`tests/unit/services` + `tests/unit/observability` + `tests/unit/models` =
**902 passed**; `tests/integration` + pin trip-wire = 174 passed / 4 failed;
`tests/services tests/settings tests/utils tests/ui` = 999 passed / 14 failed.
Every failure verified **pre-existing** by stashing the source change and
re-running — identical failure sets and counts before and after. **24 tests
directly cover this story** (12 in the rewritten suppression suite + 12 in the
updated warmup file), and the mutation harness catches 8 of 8 reverted fixes.

**Hardware re-verified post-fix** (`20-2-after-long-postfix-r0{1,2}.csv`):
priming still succeeds, still reports `consumer chunks seen during priming = 0`,
segment 1b still in the warm class (90 / 130 ms). The §3 measurements stand.
Note the harness runs with `audio_coordinator=None` and `session_registry=None`,
so they cover channel 1 only — channels 2-4 are proven by the unit rig, which
wires a real `AudioCoordinator` and a real `SessionRegistry` before priming.

### File List

**Source (shipped)**
- `src/myvoice/services/qwen_tts_service.py` — MODIFIED
  (`QwenTTSRequest.suppress_audio_output`; new `_audio_chunk_sink`; three
  chunk-emit sites rewired; `warmup_compile_async` warm-path priming +
  `primed_warm` telemetry + `MYVOICE_DISABLE_WARM_COMPILE_PRIMING` gate;
  `_run_compile_priming` builds its own suppressed request and its ordering
  claim is deleted; `tts_compile="off"` gate comment amended, gate unchanged)

**Tests**
- `tests/unit/services/test_compile_priming_audio_suppression.py` — NEW
  (12 tests; rewritten in the review pass around a `ChannelSpies` recorder that
  watches all four user-facing channels, plus a control row proving the rig can
  reach each one, a channel-level source invariant, and the F6 trip-wire row)
- `tests/unit/services/test_qwen_tts_service_compile_warmup.py` — MODIFIED
  (cache-hit row split into warm-prime / failed-warm-prime / env-gate rows;
  added the no-model-registry fast-exit row and two F7 indicator rows;
  7 → 12 tests)

**Measurement (tools + artifacts, not shipped)**
- `tools/ttfa_spike_harness.py` — MODIFIED (`--prime` flag, `PRIMING_TEXT`,
  `ConsumerSim.total_chunks_seen`)
- `_bmad-output/implementation-artifacts/20-2-warm-path-compile-priming-evidence.md` — NEW
- `_bmad-output/implementation-artifacts/20-2-{before,after}-{long,short}-r0{1..5}.csv` — NEW (20 files)
- `_bmad-output/implementation-artifacts/20-2-after-long-postfix-r0{1,2}.csv` — NEW
- `_bmad-output/implementation-artifacts/20-2-capture-long.sh`,
  `20-2-capture-short.sh`, `20-2-aggregate.py`,
  `20-2-mutation-harness.py` — NEW

**Untouched:** `requirements.txt`, `build_tools/*`, bundled `python310/`,
`torch_runtime.py`, the Story 16.8 forward hook, `DEFAULT_CHUNK_SIZE`, the
adaptive cushion, `compute_key`'s `decode_window_frames=30`, and `app.py`.

## Change Log

- 2026-08-31 — Drafted by Winston from Story 20.1's Follow-up A. Scoped deliberately narrow: the priming change itself is small, so the story's weight sits on AC #2/#3 — closing an audio-suppression race that the existing code documents as an assumption and that has already failed once in production.
- 2026-08-31 — Implemented by the dev agent. AC #1/#2/#3/#5/#6 met; AC #4 measured across four cells on the RTX 5090. One blocking finding recorded: the `app.py:593` call-site ordering makes `warmup_compile_async` short-circuit at `no_model_loaded` on every GUI launch, so the measured win is not yet delivered to users; the naive fix is unsafe (model-registry thrash) and is written up as Follow-up A' rather than taken in this story.
- 2026-08-31 — Review response. An adversarial review of commit `2f19b64` found the "all emit sites are sinked" claim false: the suppression wall covered the chunk callback only, while `play_dual_stream` (the user's speakers), `_save_audio_to_cache` (Replay Last) and `create_session` (Save button / focal-cancel) reached the user unguarded, and priming's `finally` could strip a concurrent user generation's cancel bookkeeping. All seven findings verified against source and fixed; suppression is now a property of the generation enforced at four enumerated channels in all three dispatch modes. Test suite rebuilt around channel spies with a control row; mutation harness added and all 8 reverted fixes are caught. Regression sweep and hardware verification re-run with identical results.
