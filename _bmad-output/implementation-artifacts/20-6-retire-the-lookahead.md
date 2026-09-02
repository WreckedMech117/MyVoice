# Story 20.6: Retire the Lookahead and the Post-Decode Trim (Phase ⊥-Polish-3)

Status: done — 2026-09-02. AC #1/#2/#4/#5 met; **AC #3 MET** after the kill-switch A/B (§12) overturned §11's baseline-artefact finding. Long TTFA −361.3 ms (−21.0 %), short −251.9 ms (−14.6 %) against a same-sitting pre-20.5 control.

<!-- Phase tag: Phase ⊥-Polish-3. Sixth story of Epic 20. Follow-up F1 from Story 20.5. -->
<!-- Risk: MEDIUM. Small change, but the machinery being retired is the ONLY seam handling the stateless fallback path has — see AC #2, which is the story's load-bearing constraint. -->

## Story

As **a MyVoice user**,
I want **the streamer to stop generating and discarding five frames it no longer needs**,
so that **first audio arrives sooner and the decoder stops doing twice the work per chunk**.

## Context

Story 20.5 carries codec state across chunk boundaries. Consecutive chunks are now
**continuous by construction** — head NRMSE 0.0078, correlation 1.0000, zero lag jitter,
single-chunk decode bit-exact. The 5-frame lookahead exists to solve a problem that no
longer exists.

Story 16.4 introduced it as *future-lookahead overlap-add*: decode `chunk_size + lookahead`
frames, post only `chunk_size` worth, and let the next chunk's leading lookahead tokens
"re-establish the same audio with better priming." **The carried state is now that
priming**, and it is exact rather than approximate.

Three costs are being paid for it:

1. **TTFA.** The streamer's first emit waits for `chunk_size + lookahead = 30` frames.
   At 25 it waits for 25 — five fewer talker steps. From the 2026-09-01 GUI capture the
   talker leg is ~1,147 ms for 30 frames, so this is worth roughly **190 ms off a
   1,353 ms TTFA**, subject to measurement.
2. **Producer throughput.** `codec_state_cache.py:116-126` — the two-pass decode
   (decode `chunk_size` on live state → snapshot → decode `lookahead` on the snapshot →
   restore) exists **solely** to serve the lookahead. Without it the decode is single-pass.
   Story 20.5 measured the tax at +7–10 ms/chunk; Phase 1's bench, which had no lookahead,
   measured carrying state as 21–30 % *faster*.
3. **Complexity.** A trim, a snapshot/restore, and a blend all exist to manage an overlap
   that no longer needs managing.

## AC #2 is the story. Read it before AC #1.

### AC #1 — Retire the lookahead and the trim

**Given** carried state makes consecutive chunks continuous
**When** the lookahead is retired
**Then** the streamer emits exactly `chunk_size` frames per chunk with no overlap, the
decoder posts the full decode with no trim, and the two-pass snapshot/restore collapses
to a single pass
**And** the geometry threading built in Story 20.4 carries `decode_window_frames`
30 → 25 to all three sites automatically — verify it does, including
`warmup_compile_async`'s cache key, exactly as the `chunk_size` revert verified it
**And** one cold compile is expected on first launch (new cache key); confirm Story 20.3's
priming then warms the new key rather than the old one

> **The Story 20.4 seam blend is a *dependent*, not a second variable.** It cross-fades
> the retained lookahead tail into the next chunk's head. With no lookahead there is no
> tail, so removing it is not a separate decision — it is removed *by construction*. Say
> this plainly in the evidence rather than presenting it as a bundled second change. Story
> 20.5 measured it as near-inert in the state-cached regime anyway (inputs differing by
> NRMSE 0.0037–0.0046).

### AC #2 — This must not touch the stateless path (load-bearing)

**Given** `MYVOICE_CODEC_STATE_CACHE` is an operator kill switch, and
`_build_true_stream_decode_fn` falls back to a cold-state adapter with
`carries_codec_state = False`
**And** on that path chunks are **still** independent renderings, so the lookahead, the
trim and the blend are the *only* seam handling it has
**When** the lookahead is retired
**Then** retirement is **conditional on carried state being active**, by the same
producer-declares/consumer-acts pattern Story 20.5 used for the crossfade — not a global
constant change
**And** with the kill switch set, or on any fallback to the stateless adapter, the
pre-20.6 geometry (lookahead 5, trim, blend) is restored **exactly**
**And** a test proves the stateless path still gets the full pre-20.6 treatment, and that
flipping the kill switch at runtime does not leave a half-retired geometry
**And** the same staleness trap Story 20.5 closed for the crossfade is closed here: if the
declaration is set on one path and not reset, a later generation on another path inherits
it. Pin it with a source-derived invariant, not a single-generation test

> Retiring the lookahead globally would silently strip the fallback path of every seam
> defence it has and reintroduce the exact artefact this epic spent six audition rounds
> eliminating — on the path a user lands on precisely when something has already gone
> wrong. That is the failure mode this AC exists to prevent.

### AC #3 — Measured

**Given** the post-20.5 baseline
**When** the change lands
**Then** TTFA is measured through the shipped GUI (`10_Story_20.3_AC4_GUI_Capture.bat`),
≥5 launches, short **and** long, against **1,353 ms**
**And** the producer emit/drain ratio is re-measured and the OFR-E `< 1.0×` gate confirmed
**And** the per-chunk decode time is compared against Story 20.5's +7–10 ms tax, to
confirm the single-pass decode recovers it
**And** the `session_id` grouping trap from Story 20.3 §4.1a is respected when reading the
CSVs — priming emits its own boundaries first

### AC #4 — NFR3 audition

**Given** this changes what the decoder emits at every boundary
**When** the change is complete
**Then** an audition runs, Commander solo, reference = **what ships today** (state caching
+ gated crossfade + lookahead), candidate = this change
**And** both arms are decoded from **one talker run** per pair where possible, as Story
20.5 did — nothing upstream of the streamer's chunking changes, so verify whether token
reuse is still available; if the chunking change makes the talker runs differ, say so and
account for take-to-take variance per Story 20.4 §17
**And** a falsifiable prediction is recorded **before** the round, including one that
could embarrass the diagnosis
**And** a zero-seam control trial is included, as Story 20.5's fixtures did

### AC #5 — No regressions

**Then** the exact bars from Story 20.5 still hold — single-chunk streaming ==
`forward` bit-for-bit, fp64 chunked == whole to ~1e-06 — and the accumulated suites pass
with the pre-existing failure set unchanged in count and identity
**And** the Story 16.5 cooperative-cancel chain is unaffected
**And** `DEFAULT_CHUNK_SIZE` stays 25; this story does not touch geometry beyond the
lookahead

## Tasks / Subtasks

- [x] **Task 1 — Conditional retirement** (AC: #1, #2) — producer declares, consumer acts; stateless path keeps the pre-20.6 geometry exactly; source-derived invariant against the staleness trap.
- [x] **Task 2 — Single-pass decode** (AC: #1) — collapse the snapshot/restore; confirm the exact bars still hold.
- [x] **Task 3 — Geometry propagation** (AC: #1) — verify all three sites follow 30 → 25; confirm priming warms the new key.
- [x] **Task 4 — Bench measurement** (AC: #3, partial) — per-chunk decode time and producer ratio, headless.
- [x] **Task 5 — Audition fixture + prediction** (AC: #4).
- [x] **Task 6 — Regression sweep** (AC: #5).
- [x] **Task 7 — Operator hand-off** — GUI capture + audition as ONE consolidated request, not piecemeal.

**Status: awaiting operator.** AC #1, #2, #5 are closed. AC #3's headless half
and AC #4's fixture + prediction are closed; their remaining halves need a
human — see the evidence file §7.

## Dev Notes

### What this story is NOT

- **Not the chunk-size reopen (F2).** That is the next story and it is the reason this one
  goes first: the two-pass tax scales with chunk count, which is exactly what F2 varies,
  and shipping them together would recreate the two-variable confound that cost Story 20.4
  its round-2 audition. `DEFAULT_CHUNK_SIZE` stays 25 here.
- **Not a change to codec state caching.** Story 20.5's wrapper is the foundation this
  stands on; it is not being modified.
- **Not the qasync call-site audit (F7)** or the RTX 3060 confirmation (F6).

### Lessons this story inherits and must not relearn

- **One variable per audition round.** Story 20.4 round 2 conflated geometry with
  stitching and cost a listening session.
- **Offline metrics do not gate NFR3.** Two independent detectors failed against the exact
  files the listener judged. Mechanism metrics are fine for the bench; only the ear gates
  AC #4.
- **Record the prediction before the round.** It is what made 20.4 round 3 and 20.5 round 2
  readable rather than arguable — including when it was falsified.
- **Group the capture CSVs by `session_id`.** Priming emits its own boundaries before the
  user's generation; a first-match join produces nonsense.

## References

- `_bmad-output/implementation-artifacts/20-5-codec-state-caching.md` — follow-up F1, and the AC #5 the chunk-size reopen is deferred to
- `_bmad-output/implementation-artifacts/20-5-phase2-evidence.md` — the +7–10 ms tax, the two smoothing layers, the Phase 4 unanimous pass
- `src/myvoice/services/tts_streaming/codec_state_cache.py:116-126` (the two-pass decode and why it exists), `:177-188` (the kill switch), `:712` (`carries_codec_state`)
- `src/myvoice/services/tts_streaming/codec_token_streamer.py` (the constants), `streaming_decoder.py` (trim + seam blend)
- `_bmad-output/implementation-artifacts/20-3-prime-the-resident-model-evidence.md` §4.1 (the 1,353 ms baseline), §4.1a (the `session_id` trap)

## Dev Agent Record

Full evidence: `_bmad-output/implementation-artifacts/20-6-retire-the-lookahead-evidence.md`.

### The shape (AC #2, which drove everything)

`DEFAULT_LOOKAHEAD` stays **5** and `DEFAULT_CHUNK_SIZE` stays **25**. The
retirement is resolved per stream from the decode_fn's own
`carries_codec_state` declaration — the producer-declares / consumer-acts shape
Story 20.5 used for the crossfade, now with two consumers of one declaration:

```
decode_fn.carries_codec_state
  ├─ self._progressive_stream_continuous     # the consumer crossfade (20.5)
  └─ streamer.apply_codec_state_geometry()   # the lookahead          (20.6)
```

`codec_token_streamer.effective_lookahead()` is the rule;
`CodecTokenStreamer.apply_codec_state_geometry()` is the single reversible
entry point that applies it (it sets `lookahead` and `_chunk_with_lookahead`
together, from the constructed value, and refuses to run mid-generation).

**The trim and the seam blend are removed by construction.** `streaming_decoder.py`
has no behavioural change: its `is_full_window` predicate already requires
`lookahead > 0`, so with the lookahead retired the trim arm is never entered
(the worker posts the whole decode) and `_pending_overlap` is never populated
(there is no retained tail to blend). Said plainly here as the story asked,
rather than presented as a bundled second change.

**The two-pass decode collapses without modifying `codec_state_cache.py`.**
The dispatch builds `StatefulCodecDecoder` with `lookahead = 0`, so
`window_frames == commit_frames`, so its own commit rule takes the
`commit = n_frames` branch and never snapshots. Counted, not inferred: 3
snapshots over 3 chunks at the old geometry, **0** at the retired one.

### AC #1's second clause — the answer is "no, and that is the finding"

The Story 20.4 threading carries a change *to the constants* automatically. It
cannot carry this one, because AC #2 forbids changing them. So the derivation
itself became conditional in one place — `torch_runtime.resolve_streamer_geometry()`
— and all three sites now read it. Verified both ways: `decode_window_frames`
is **25** normally and **30** under the kill switch, at all three sites, with
the warm-path priming key equal to the engage-path key in both regimes; and a
static invariant now forbids any source file from summing the raw constants.

### Measured (AC #3, headless half) — `20-6-lookahead-bench.json`

RTX 5090, bf16, Story 20.5's own captured tokens (reused, not redrawn):

| | l-020 | l-021 | m-020 |
|---|---|---|---|
| per-chunk, lookahead 5 | 23.39 ms | 23.29 ms | 17.15 ms |
| per-chunk, retired | 11.82 ms | 11.85 ms | 11.45 ms |
| **saving** | **+11.57** | **+11.44** | **+5.70** |

Per-chunk decode time roughly **halves**, over-recovering Story 20.5's
+7–10 ms tax because the second pass also decoded 5 extra frames. Posted
sample counts identical to a whole-sequence decode in every case. First-emit
threshold 30 → 25 frames.

### AC #4 — fixture generated, prediction recorded

`20-6-perceptual-fixtures/` (16 trials), driven by
`14_Story_20.6_AC4_Audition.bat`. Reference = what ships today; candidate =
this change; one talker run per pair (**token reuse verified as still
available** — the chunking is downstream of generation, so the fixture recovers
the flat frame sequence and re-slices it per arm). Zero-seam control
`ctl-020` is byte-identical, confirmed on both takes. Offline: identical length
16/16, worst level delta 0.005 dB, median waveform difference −70 dB. P1–P5
recorded in the evidence §6 and in the helper's docstring, including P4 — the
one that would embarrass the diagnosis and force a revert.

### Regressions (AC #5)

Full suite run twice on this machine — `main` state vs this branch — with the
story's `src/`/`tests/`/`tools/` changes stashed for the first: **2,852 passed
/ 49 failed / 4 errors** before, **2,883 passed / 49 failed / 4 errors** after.
The pre-existing failure set is unchanged in count *and identity* (sorted diff
of the two FAILED lists is empty both ways); none of the 49 touch streaming.
The exact bars hold at both geometries, the stateless path's rows are
untouched, Story 16.5's cancel chain is unaffected, and `DEFAULT_CHUNK_SIZE`
is still 25.

### Awaiting operator

Evidence §7, one sitting: `13_Story_20.6_AC3_GUI_Capture.bat` then
`14_Story_20.6_AC4_Audition.bat`.

## Change Log

- 2026-09-01 — Drafted by Winston as Story 20.5 follow-up F1, sequenced ahead of the chunk-size reopen because the two-pass tax scales with chunk count and would otherwise confound it. Scope is dominated by AC #2: the machinery being retired is the only seam handling the stateless fallback path has, so retirement must be conditional on carried state rather than a global constant change.
- 2026-09-01 — Implemented. Retirement is conditional on `carries_codec_state`, resolved by `effective_lookahead()` and applied by `apply_codec_state_geometry()`; the constants are unchanged. The trim, the seam blend and the two-pass decode all fall away by construction. AC #1's "carries 30 → 25 automatically" turned out to be false and the derivation had to be made conditional at one point — recorded rather than quietly reinterpreted. AC #3's headless half and AC #4's fixture + prediction are complete; both remaining halves are one consolidated operator request.
