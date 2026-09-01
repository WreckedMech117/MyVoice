# Story 20.4: Chunk-Size Retune + Adaptive-Cushion Fix (Phase ⊥-Polish-3)

Status: in-progress — AC #1/#2/#3/#4/#6/#7 complete and verified. AC #5 round 1 FAILED (blocking); root cause found to be a splice-alignment bug deleting 15-19 ms of speech at EVERY chunk boundary on both geometries, plus a codec-state mismatch — both fixed in `streaming_decoder.py`, seam discontinuity 12.3x -> 1.3x the non-seam baseline. Round-2 audition is the only outstanding task (evidence §8b)

<!-- Phase tag: Phase ⊥-Polish-3. Fourth story of Epic 20 (First-Audio Latency). -->
<!-- Source: Story 20.1 evidence §5 (Follow-up B) + §2.6 (Follow-up C), which are COUPLED. -->
<!-- Risk: MEDIUM-HIGH. Changes the streamer geometry every generation flows through, and the consumer-side release policy on the ship-target hardware tier. Carries an NFR3 perceptual gate that the previous three Epic 20 stories did not. -->

## Story

As **a MyVoice user on any supported GPU**,
I want **audio to start sooner on short utterances and not to sit behind a ten-second cushion on a mid-range card**,
so that **Clear Comms is usable as an interjection tool rather than a delayed broadcast**.

## Context

Stories 20.2 + 20.3 delivered Follow-up A: first-generation TTFA **5,051 → 1,353 ms**
on the RTX 5090, confirmed through the shipped GUI. Follow-ups **B** and **C** are the
next two items on Story 20.1 §6.4's ranked list — and Story 20.1 found they **must ship
together**.

**B — chunk sizing.** `DEFAULT_CHUNK_SIZE = 25` / `DEFAULT_LOOKAHEAD = 5`
(`codec_token_streamer.py:46-47`) were inherited verbatim from a research example and
never tuned. At 12 Hz that means **30 frames = 2.5 s of audio must be generated before
any PCM is emitted**. Story 20.1 §5.2 measured the curve:

| `chunk_size` | window | audio/chunk | seg 2 talker | seg 4 cushion | **perceived TTFA** | producer ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 417 ms | 577 | **316** | 1,093 | 0.881 |
| **10** | 15 | 833 ms | 828 | 0.0 | **1,015** ← min | 0.776 |
| 15 | 20 | 1,250 ms | 989 | 0.5 | 1,157 | 0.671 |
| 25 (today) | 30 | 2,083 ms | 1,496 | 0.3 | 1,662 | 0.659 |

**The optimum is 10, not the smallest value** — at 5 each chunk carries 417 ms, below
the 500 ms static watermark, so the consumer holds two chunks and hands back 316 ms.
`chunk_size >= 6` keeps the watermark a no-op.

`chunk_size = 10` also drops the first-emit threshold from 30 frames (2.5 s) to 15
(1.25 s), which is what actually fixes the **short-utterance degeneration**: Story 20.1
measured TRUE_STREAM falling back to batch on 11 of 20 short runs at cs25 versus
**0 of 5** at cs10, with short TTFA 1,651 → 921 ms.

**C — the adaptive cushion.** On sub-16 GiB hosts `StreamingChunkBuffer` switches from
the static 500 ms watermark to an adaptive pre-buffer. Story 20.1 §2.6 derived that the
cushion overtakes the talker whenever `P < 0.87`; the RTX 3060 is documented at
`P ≈ 0.5`. Simulation against the shipped buffer refined this: **`MAX_PRE_DELAY_SECONDS`
is the binding escape for every `P <~ 0.78`**, and because the cap is only evaluated
inside `push`, the effective wait at `P = 0.5` is **~12.5 s, not 10 s**.

**Why they are coupled.** At `chunk_size = 10` the sub-16 GiB cushion-to-talker ratio
**worsens from 2.5× to 4.0×**, because the talker segment shrinks while the 10 s cap
does not. Shipping B alone would speed up ≥16 GiB hosts and leave the RTX 30xx tier
pinned at the cap — measurably worse in relative terms than before.

## Acceptance Criteria

### AC #1 — Retune the chunk geometry, and thread it where it is actually read

**Given** `DEFAULT_CHUNK_SIZE = 25` / `DEFAULT_LOOKAHEAD = 5`
**When** the default is changed to `chunk_size = 10` (lookahead unchanged at 5)
**Then** the committed constants reflect the measured optimum
**And** the **D-25 trap Story 20.1 §5.4 identified is closed in the same change**:
`engage_compile_optimizations` declares `streamer_chunk_size: int = 25,
streamer_lookahead: int = 5` as **hard-coded defaults** (`torch_runtime.py:519-520`)
and the sole production call site (`model_registry.py:591`) passes **neither**, so
`decode_window_frames` resolves to 30 regardless of the streamer's real geometry.
Retuning the constant without threading the real values through would tell the compile
path 30 while the streamer emits 15 — silently violating the very invariant the D-25
assertion exists to protect
**And** the real geometry is read from the streamer module rather than duplicated as a
second literal, so the two cannot drift again
**And** a test asserts the compile path receives the streamer's actual window, and fails
if the constants and the compile geometry diverge

> Story 20.1 §5.4 notes this is harmless *today* only because the fork skips its manual
> `capture_cuda_graph` under `compile_mode="reduce-overhead"` and our decode path calls
> `speech_tokenizer.decode(...)` directly. Do not rely on that; it is why the invariant
> currently reads as decorative.

### AC #2 — Fix the cushion so a slow host is not pinned at the cap

**Given** `_adaptive_ready_to_dispatch` (`streaming_chunk_buffer.py:260+`) evaluates, in
order: `is_final` → `_chunks_held >= max_hold_chunks` → `elapsed >= max_pre_delay_seconds`
→ `P >= 1.0` → `audio_buffered_seconds >= τ_min`
**And** at `P ≈ 0.5` the τ_min comparison never binds because the clamp puts it at the
10 s cap, so release happens via the elapsed/held escapes at ~12.5 s
**When** the cushion policy is revised
**Then** a sub-16 GiB host starts audio **materially sooner than the cap**, and the
chosen policy is justified against the measured/derived numbers rather than asserted
**And** the policy is stated explicitly in the evidence file as a **product trade**:
Clear Comms is an interjection feature (`memory/clear_comms_purpose_framing.md`), so
**starting sooner with a possible gap is preferred over starting late with none** —
if the implementation concludes otherwise, it must say why and surface it rather than
silently choosing gaplessness
**And** the `≥16 GiB` static-watermark path is **behaviourally unchanged** — this AC
touches only the adaptive branch
**And** the T_a estimator overshoot Story 20.1 found is addressed or explicitly deferred
with a reason: `CHARS_TO_AUDIO_SECONDS = 0.08` produced an estimate of 27.92 s against a
measured 19.32 s (**~44 % high**) on the canonical fixture, and that estimate feeds τ_min
directly

> Do **not** simply raise or remove `MAX_PRE_DELAY_SECONDS`. It is a safety bound against
> unbounded waits (cold compile, CPU-only). Changing the release policy is in scope;
> removing the guardrail is not.

### AC #3 — The coupling is verified, not assumed

**Given** Story 20.1 derived that `chunk_size = 10` worsens the sub-16 GiB
cushion-to-talker ratio from 2.5× to 4.0×
**When** both changes are in place
**Then** the combined effect on the adaptive path is re-derived at the new geometry —
the AC #2 fix must hold **at `chunk_size = 10`**, not merely at 25
**And** if the two changes interact in a way the derivation did not predict, that is
reported rather than averaged away
**And** neither change is committed without the other

### AC #4 — OFR-E producer gate survives

**Given** the architecture's producer emit/drain target of `< 1.0×` sustained, met at
0.659× today and measured at 0.776× at `chunk_size = 10` (Story 20.1 §5.2)
**When** the retune lands on the current post-20.3 code
**Then** the ratio is **re-measured**, not carried over — the 0.776× figure predates
Stories 20.2/20.3 and the compile-priming change
**And** it remains `< 1.0×` sustained. If it does not, stop and report rather than
shipping a regression to the gate Epic 18 was built to close

### AC #5 — NFR3 perceptual gate (this story needs one; the last three did not)

**Given** this changes the streamer's chunk boundaries, and the decoder's overlap-add
trims a lookahead-sized tail per chunk (`streaming_decoder.py`), so chunk-stitching
behaviour changes for **every** generation
**When** the retune is complete
**Then** a perceptual A/B is run before the story closes — Commander solo is sufficient,
mirroring the Story 18.1/18.2 discipline rather than Story 17.1's multi-listener protocol
**And** it covers short **and** long utterances on a CLONED voice, since the short class
changes dispatch path entirely (residual-flush → threshold)
**And** any audible chunk-boundary artefact — clicks, discontinuities, altered prosody at
stitch points — is a **blocking** finding, not a note

### AC #6 — Measured on the reachable tier; derived for the other

**Given** the RTX 3060 remains unreachable (Story 20.3 AC #2b Phase 3, still deferred)
**When** the change is measured
**Then** the RTX 5090 static-watermark path is measured directly through the shipped GUI
using `10_Story_20.3_AC4_GUI_Capture.bat`, ≥5 launches, short **and** long utterances
**And** results are compared against the post-20.3 baseline established 2026-09-01:
**1b 192 ms / TOTAL 1,353 ms** long-form — *not* against the pre-20.3 numbers
**And** the sub-16 GiB effect is **derived** from the shipped buffer's own logic and
labelled derived, never observed
**And** the deferred 3060 confirmation is restated with what it would now check
**And** results land at
`_bmad-output/implementation-artifacts/20-4-chunk-size-and-adaptive-cushion-evidence.md`

> **CSV analysis note, learned the hard way in Story 20.3 §4.1a:** these captures contain
> three sessions each — the priming generation, its `no-registry` post, and the user's
> generation. Group by `session_id` and filter to the one carrying
> `ttfa_first_playback_write_ms`. A naive first-match join splices priming's segments
> onto the user's and produces nonsense.

### AC #7 — No regressions

**Given** the suites Epic 20 has accumulated
**When** the change lands
**Then** they pass with zero new failures, and the tree's known pre-existing failures are
unchanged in count and identity
**And** the compile cache gains one new key (the window changes 30 → 15), so exactly one
cold compile is expected on first launch after this ships — note it, and confirm
Story 20.3's priming then warms the **new** key

## Tasks / Subtasks

- [x] **Task 1 — Chunk geometry** (AC: #1)
  - [x] 1.1 Change the streamer constants to `chunk_size = 10`, `lookahead = 5`.
  - [x] 1.2 Thread the real geometry into `engage_compile_optimizations` from the call site, reading the streamer module rather than adding a second literal. **Found and closed a THIRD site the spike did not name** — `qwen_tts_service.warmup_compile_async` carried its own `decode_window_frames=30` in the compile-cache key; left in place it would have had Story 20.3's priming warm a key the engage path never reads.
  - [x] 1.3 Test that the compile path receives the streamer's actual window and fails on divergence. New file `test_decode_window_geometry_coherence.py` (6 rows, runtime + static arms) plus a resolution row in `test_torch_runtime.py`.

- [x] **Task 2 — Cushion policy** (AC: #2)
  - [x] 2.1 Two-regime feasibility policy: buy the gapless cushion when it fits inside `cushion_budget_seconds = 2.0`, otherwise fall back to the static watermark. `MAX_PRE_DELAY_SECONDS` untouched as a guardrail. Justified against the simulation and the Story 20.3 TTFA baseline in evidence §2.2.
  - [x] 2.2 **Addressed, not deferred.** `CHARS_TO_AUDIO_SECONDS = 0.08` (+44.5 %) replaced with an affine `0.5 + 0.055·chars` (+2.0 %), calibrated on both measured fixtures and independently validated against a fresh 19,527 ms measurement (evidence §4.2).
  - [x] 2.3 Proved by two tests: byte-identical static-path output across three budget values, and an exact watermark release point with `last_release_reason is None`.
  - [x] 2.4 Nine new rows across two classes, written at the `chunk_size = 10` chunk geometry.

- [x] **Task 3 — Coupling** (AC: #3)
  - [x] 3.1 `20-4-adaptive-cushion-sim.py` re-derives both policies at both geometries against the shipped buffer, and **cross-checks its pre-20.4 reproduction against all 10 numbers Story 20.1 §2.7 published (10/10 match)**. Ratio at cs10 goes 4.00× → 0.67×. Three unpredicted interactions reported in evidence §3.3.

- [x] **Task 4 — Measure** (AC: #4, #6)
  - [x] 4.1 Re-measured on current code: **0.619× at cs10** vs 0.585× at cs25, contemporaneous, same host, same session. Gate `< 1.0×` holds with 38 % margin.
  - [x] 4.2 GUI capture — **operator run 2026-09-01, PASSED**: long TTFA median **976 ms**, short **1,065 ms**, producer ratio 0.602 / 0.403, short class on the threshold path at 3 chunks. Against the 1,353 ms Story 20.3 baseline. AC #6 closed.
  - [x] 4.3 Sub-16 GiB effect derived and labelled derived; the deferred 3060 check restated with five specific things it would now falsify (evidence §7).
  - [x] 4.4 Evidence file written.

- [~] **Task 5 — Audition** (AC: #5) — **round 1 RUN and FAILED** (blocking): `m-020` clean at cs25, `tonal_distortion` at cs10; `l-020`/`l-021` defective on **both** arms; cs10 never preferred. Root-caused (see Task 7), fixed, and a round-2 fixture built. **Round-2 listening session is the only outstanding operator task.**

- [x] **Task 6 — Regression sweep** (AC: #7) — zero new failures on every surface; pre-existing set unchanged in count and identity. Exactly one new compile-cache key directory observed, as predicted. Re-run after the Task 7 fix: unchanged.

- [x] **Task 7 — Seam root-cause + fix** (AC: #5; added 2026-09-01 after the round-1 failure)
  - [x] 7.1 Determined the mechanism by measurement rather than assumption — captured the decoder's posted arrays AND its pre-trim `pcm_full`, which consecutive chunks share by `lookahead` frames, making the alignment answerable by cross-correlation.
  - [x] 7.2 **It is not the consumer crossfade.** That crossfade is 2.67 ms; the defect is a 15.4 ms deletion (5.8× wider) plus a mismatch spanning 377 ms (141× wider). No consumer-crossfade sweep was run — it would have been aimed at the wrong mechanism at the wrong layer.
  - [x] 7.3 Found and fixed a **splice-alignment bug**: `decode(N) = 1920N − 555`, and the trim treated the fixed 555-sample edge loss as proportional, deleting `555·cs/(cs+la)` samples of real speech at every seam — 370 at cs10, 463 at cs25 — since Story 16.4.
  - [x] 7.4 Found a **codec-state mismatch** (~35 % NRMSE between the two decodes of the shared frames) and swept the fix at the correct layer — a decoder-side overlap-add over the tail the module used to discard. Chose 1024 samples on evidence; cost stated (5.3 % of the stream inside a blend at cs10).
  - [x] 7.5 Verified end-to-end on the shipped path: seam step **12.3× → 1.3×** the non-seam baseline; the blocking utterance's spectral seam resolved to baseline.
  - [x] 7.6 Round-2 fixture built with round 1's `cs25` files reused verbatim as a calibration anchor; round 1's evidence preserved intact.

## Dev Notes

### Operator dependency

Tasks 4.2 and 5 need Commander at the keyboard. Get everything else to a verified state
first, then hand over a single consolidated run — do not ask for GUI launches piecemeal.

### What this story is NOT

- **Not PORT-b.** Follow-up E is re-scoped after this lands, and its case is materially
  weaker now: it was ranked against a 5,051 ms baseline that is already 1,353 ms.
- **Not the qasync call-site audit.** Story 20.3 fixed only the startup site and flagged
  the hazard as general. That is its own story.
- **Not a change to `MAX_PRE_DELAY_SECONDS` as a guardrail**, nor to the static-watermark
  constants on `≥16 GiB` hosts.
- **Not a re-litigation of Story 20.1's curve.** The four-point sweep stands; this story
  commits the optimum it identified and verifies it survives on current code.

## References

- `_bmad-output/implementation-artifacts/20-1-ttfa-spike-faster-qwen3-tts-evidence.md` §5 (the curve, the optimum, the D-25 trap at §5.4), §2.4 (short-utterance degeneration), §2.6 (cushion break-even), §6.4 (Follow-ups B and C, and the coupling)
- `_bmad-output/implementation-artifacts/20-3-prime-the-resident-model-evidence.md` §4.1 (the post-20.3 baseline this is measured against), §4.1a (the session_id analysis trap)
- `src/myvoice/services/tts_streaming/codec_token_streamer.py:46-47` — the constants
- `src/myvoice/services/tts_streaming/torch_runtime.py:519-520` — the hard-coded geometry defaults; `model_registry.py:591` — the call site that passes neither
- `src/myvoice/services/streaming_chunk_buffer.py:192-203` (τ_min), `:260+` (release order)
- `src/myvoice/services/audio_coordinator.py:61-90` — watermark and adaptive thresholds

## Dev Agent Record

### Agent Model Used

claude-opus-5[1m]

### Completion Notes List

1. **The D-25 trap had three sites, not two.** Story 20.1 §5.4 named
   `torch_runtime`'s hard-coded defaults and `model_registry`'s silent call
   site. Implementing the retune surfaced a third: `warmup_compile_async`
   (`qwen_tts_service.py:2217`) carried its own `decode_window_frames=30` in
   the compile-cache key. That one is not decorative — the window is one of
   the seven key dimensions, so leaving it would have made Story 20.3's
   startup priming warm a key the engage path never reads, silently
   reverting the ~4 s first-generation win with no error and no log line.
   Closed in the same change; `test_warmup_cache_key_moves_with_a_streamer_retune`
   is the regression row for exactly that class.

2. **The cushion fix is a feasibility test, not a lower cap.** The story
   forbids simply moving `MAX_PRE_DELAY_SECONDS`, and this does not: the
   guardrail is unchanged at 10 s and still tested. What changed is that
   `τ_gapless` is now computed *unclamped* and compared against a
   `cushion_budget_seconds = 2.0` feasibility budget. The insight that makes
   it obvious: at `P = 0.5` the old policy paid a 12.5 s wait **and gapped
   anyway**, because the clamped cushion was nowhere near the ~19 s
   gaplessness actually required. That is a strictly dominated outcome.

3. **The product trade is stated, and so is where it costs.** Evidence §2.4
   records that total silence is conserved (a cushion second not spent up
   front reappears as a gap second later), that MyVoice's answer is
   "starting sooner", and that there is a real band — roughly
   `0.70 ≤ P ≤ 0.90` — where the old policy *did* buy gaplessness with its
   wait and the new one trades it away. That band is not the ship-target
   operating point (`P ≈ 0.5`), and the budget is one named constant if the
   deferred 3060 run says otherwise.

4. **The estimator was fixed rather than deferred, and then validated by
   accident.** The affine `0.5 + 0.055·chars` was calibrated on Story 20.1's
   two measured fixtures. The AC #4 harness run then independently reported
   `total_audio_ms` median 19,527 ms for the long fixture — the new
   estimator predicts 19,695 ms (+0.9 %), the old one 27,920 ms (+43.0 %).

5. **Three integration assertions went red, and one of them was the point.**
   `test_streaming_tts_smoke.py` hard-coded `4`, `30` and `20`. The `20` case
   is the instructive one: it did not merely fail, it *changed meaning* —
   with the window at 15, a 20-step fixture exercises the threshold path
   while asserting residual-flush behaviour. All three now derive from the
   streamer constants, with a self-check row pinning the helper against both
   shipped geometries.

6. **Everything about the sub-16 GiB tier is DERIVED.** Nothing in this story
   was observed on sub-16 GiB hardware. The simulation's credibility rests on
   reproducing all 10 of Story 20.1 §2.7's published pre-20.4 numbers before
   reporting any post-fix ones; it exits non-zero if that cross-check fails.

7. **Operator-gated work is consolidated into evidence §8** — one GUI capture
   run and one listening session, ~35 minutes total, with both launchers
   preflighting themselves.

--- added 2026-09-01, after the round-1 audition failure ---

8. **The round-1 failure had a cause neither candidate hypothesis named.**
   The two hypotheses on the table were the consumer crossfade being too
   narrow, or the decoder's independent chunk decodes. The actual dominant
   defect was a third thing nobody had looked for: `decode(N frames)` returns
   `1920N − 555` samples — 1920 per frame plus a FIXED 555-sample edge loss —
   and the trim treated that fixed loss as proportional. Every chunk boundary
   was therefore **deleting 15-19 ms of real speech**, and had been since
   Story 16.4. That is why round 1 flagged long-form seams on the *reference*
   arm too: `cs25` was never a clean baseline.

9. **The two defects are not independent, and that changed the design.**
   Correcting the alignment ALONE makes the click measurably **worse** at
   `chunk_size = 25` (8.46× → 18.04× the non-seam baseline), because it butts
   two genuinely different renditions of the same instant together where
   before they were at least offset. Fixing the splice without the overlap-add
   would have shipped a regression while looking like a fix.

10. **The fix uses material the decoder already computes and threw away.**
    `pcm_full` extends 9,045 samples (377 ms) past the splice point, covering
    exactly what the next chunk re-decodes at its head. The "overlap-add" was
    an overlap-*discard*. Retaining it makes the boundary a true cross-fade
    between two renditions of the same instant, so unlike widening the
    consumer crossfade it costs **no audio and no duration**.

11. **The constants are verified, not trusted.** `1920`/`555` were measured
    and are re-checked on every chunk; a failure drops the session to the
    pre-20.4 trim and emits `decode_geometry_unverified`. Deriving a splice
    from a stale constant would mis-cut every chunk — worse than the bug being
    replaced — so the fallback is deliberate.

12. **A documentation defect, harmless but worth recording:** 1920
    samples/frame is **12.5 Hz**, not the 12 Hz this codebase's prose has said
    since Story 16.3. No code used 12, so nothing behavioural changes; every
    "seconds of audio per chunk" figure in comments and evidence is ~4 %
    optimistic.

13. **This reframes Mary's research Finding 1.** Codec state caching across
    chunks was filed as a throughput optimisation. §11.4 measures the cost of
    not doing it — ~35 % NRMSE between the two decodes at every boundary — so
    it is an **audio-quality** item, and the only fix that removes the cause
    rather than masking it. Out of scope here; recommended for re-filing.

14. **The `chunk_size = 15` retreat was prepared for and deliberately not
    taken.** It does not fix the defect; it halves how often you meet it, at
    the cost of the story's speed win, while shipping a now-understood
    one-line-fixable bug. It remains the fallback if round 2 fails.

### File List

**Source**
- `src/myvoice/services/tts_streaming/streaming_decoder.py` (seam fix)
- `src/myvoice/services/tts_streaming/codec_token_streamer.py`
- `src/myvoice/services/tts_streaming/torch_runtime.py`
- `src/myvoice/services/model_registry.py`
- `src/myvoice/services/qwen_tts_service.py`
- `src/myvoice/services/streaming_chunk_buffer.py`
- `src/myvoice/services/audio_coordinator.py`

**Tests**
- `tests/unit/services/test_decode_window_geometry_coherence.py` (new)
- `tests/unit/services/test_streaming_chunk_buffer.py`
- `tests/unit/services/test_audio_coordinator.py`
- `tests/unit/services/tts_streaming/test_torch_runtime.py`
- `tests/unit/services/tts_streaming/test_codec_token_streamer.py`
- `tests/integration/test_streaming_tts_smoke.py`
- `tests/unit/services/tts_streaming/test_streaming_decoder.py` (+8 rows)

**Tooling**
- `tools/ttfa_spike_harness.py`
- `11_Story_20.4_AC6_GUI_Capture.bat` (new)
- `12_Story_20.4_AC5_Audition.bat` (new)

**Artifacts** (`_bmad-output/implementation-artifacts/`)
- `20-4-chunk-size-and-adaptive-cushion-evidence.md` (new)
- `20-4-adaptive-cushion-sim.py` + `20-4-adaptive-cushion-sim.txt` (new)
- `20-4-aggregate-gui.py` (new)
- `20-4-regen-audition-fixture.py` (new)
- `20-4-l1-audition-helper.py` (new)
- `20-4-perceptual-fixtures/` — 14 WAVs + `_perlistener_truthtable.json` (new)
- `20-4-gui-utterances.txt` (new)
- `20-4-ratio-long-cs10.csv`, `20-4-ratio-long-cs25.csv`,
  `20-4-ratio-short-cs10.csv`, `20-4-ratio-short-cs25.csv` (new)
- `20-1-adaptive-cushion-sim.py` — superseded-header note only
- `20-4-seam-capture.py`, `20-4-seam-capture-full.py`, `20-4-seam-analysis.py`,
  `20-4-seam-fix-sweep.py` (new) — the root-cause investigation
- `20-4-seam-raw/`, `20-4-seam-rawfull/`, `20-4-seam-rawfull-prefix/` (new) — its captures
- `20-4-regen-audition-fixture-r2.py`, `20-4-perceptual-fixtures-r2/` (new) — round 2
- `20-4-chunk-retune-audition.csv` — round-1 result, preserved

## Change Log

- 2026-09-01 — Drafted by Winston from Story 20.1 Follow-ups B and C, shipped as one story because Story 20.1 found them coupled: `chunk_size = 10` worsens the sub-16 GiB cushion-to-talker ratio from 2.5× to 4.0×, so B alone would speed up large-VRAM hosts and leave the RTX 30xx tier pinned at the cap.
- 2026-09-01 — Implemented. AC #1/#2/#3/#4/#7 complete and verified; AC #5 fixture generated and AC #6 tooling in place, both awaiting the operator run consolidated in evidence §8. Producer ratio re-measured at 0.619× (gate `< 1.0×`). Long-form TTFA 1,491 → 829 ms and short-form 1,409 → 784 ms on current code, with the short class off the residual-flush path 6/6. Sub-16 GiB cushion at `P = 0.5` derived at 10.00 s → 1.67 s, cushion/talker 4.00× → 0.67×.
- 2026-09-01 — AC #6 measured and PASSED (long 976 ms / short 1,065 ms). AC #5 round 1 FAILED; investigation found the dominant cause was a splice-alignment bug deleting 370-463 samples of speech at every chunk boundary — present at the shipped chunk_size=25, not introduced by the retune — with a codec-state mismatch underneath it. Both fixed in `streaming_decoder.py` (exact splice + decoder-side overlap-add over the tail that was previously discarded); seam discontinuity 12.3x -> 1.3x the non-seam baseline, verified end-to-end on the shipped path. The consumer crossfade was ruled out by measurement and not swept. Round-2 fixture built; re-audition outstanding.
