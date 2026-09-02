# Story 20.8: Re-baseline and Reopen the Chunk-Size Question (Phase ⊥-Polish-3)

Status: ready-for-dev

<!-- Phase tag: Phase ⊥-Polish-3. Eighth story of Epic 20. Follow-up F2 from Story 20.5. -->
<!-- Story class: PHASE-GATED. Phase 1 is a headless re-baseline with a hard go/no-go. Phase 2 does not start until Commander approves it. -->
<!-- Risk: MEDIUM. The code change is a constant. The risk is spending audition rounds on a prize that may have moved. -->

## Story

As **Commander deciding whether to spend more listening time on chunk size**,
I want **the latency curve re-measured on the code we actually ship now**,
so that **the decision rests on current numbers rather than on a curve measured before three stories changed the thing underneath it**.

## Context — every number justifying F2 is stale

Story 20.4 closed the chunk-size question at `cs25` after four auditions, and
Story 20.5 §17's amendment recorded that its **stated reopening condition has been
met**: the seam harm that killed `cs10` is now removed at the cause, not masked.

But the *latency* case for reopening rests on numbers that no longer describe the
system. Story 20.1 §5.2's sweep was measured:

- **before** codec state caching (Story 20.5),
- **before** the lookahead retirement (Story 20.6), and
- with `decode_window_frames` **pinned at 30 regardless of `chunk_size`** — Story
  20.1 §5.4 found the compile-cache key never varied with the geometry, which is
  precisely the trap Story 20.4's threading closed.

And Story 20.6 §12 established that the session those numbers came from is not
comparable to today at all: the same code measured today yields **46.66 ms/frame**
on the pre-20.5 geometry against that session's 38.25.

**Three separate reasons the old curve cannot be used. Re-measure before spending
an audition round.**

### What has changed in the candidate's favour

With the lookahead retired, `chunk_size = N` means first emit at **N frames**, not
`N + 5`. At `cs25` that is 25 frames; at `cs10` it would be **10**. Against a
measured ~45 ms/frame that is a far larger lever than the old curve implied — the
old sweep's `cs10` still waited 15 frames.

### What has changed against it

Every sweep point is now a **distinct compile-cache key**, because Story 20.4
threaded the real geometry through and Story 20.6 made the derivation conditional.
Story 20.1's sweep was free precisely because the key was pinned; this one is not.
Budget a cold compile per point (~22.5 s, Story 18.4).

## Phase gate

**Phase 1 is a headless re-baseline. Phase 2 does not begin until Commander
approves the Phase 1 result.** No audition time is spent until the numbers say
there is something worth auditioning.

## Acceptance Criteria

### AC #1 — Phase 1: re-measure the curve on current code (the gate)

**Given** `tools/ttfa_spike_harness.py` already drives `_generate_true_stream`
headlessly with a `--chunk-size` override
**When** the sweep is re-run on current `main`
**Then** it reports, for each `chunk_size` in at least {10, 15, 25} — adding
smaller points only if the watermark analysis below says they are viable:
  - first-emit threshold in frames, and the resulting segment-2 talker time
  - end-to-end TTFA, long **and** short class
  - producer emit/drain ratio against the OFR-E `< 1.0×` gate
  - chunk count and per-chunk decode time
**And** every point is measured in **one sitting on one machine**, because Story
20.6 §12 showed cross-session comparison is the thing that produced a false
conclusion once already
**And** `cs25` is measured as the control in the same sitting — not carried over
from Story 20.6's capture
**And** the cold-compile cost per point is stated, and the measurements are taken
warm

**Derive, do not assume, the watermark floor.** The consumer holds a static
500 ms watermark on ≥16 GiB hosts. At 12.5 Hz a chunk carries `N / 12.5` seconds,
so points below ~6–7 frames per chunk re-introduce the cushion penalty Story 20.1
§5.2 measured at `cs5` (316 ms handed back). Compute the exact floor and report
which points sit above it.

**Go/no-go, stated before the work:**
- **GO** if a viable point beats `cs25`'s same-sitting TTFA by a margin larger
  than the within-arm spread (Story 20.6 measured ~83 ms), **and** its producer
  ratio stays under 1.0×
- **NO-GO** if no point clears both — the geometry question is then closed for
  good on current evidence, and Story 20.4 §17 gets a second amendment saying so
- **In between** — report and let Commander decide; do not self-authorise Phase 2

### AC #2 — Phase 1: state what the audition would have to answer

**Given** Story 20.4's four rounds established `cs10` as perceptually worse, and
Story 20.5 removed the cause of that harm
**When** Phase 1 reports
**Then** it states plainly what is now *unknown* rather than implying the
perceptual question is settled: seam count still rises with smaller chunks, and
"the cause of the old harm is removed" is a mechanism argument, not an audition
result
**And** it estimates the audition cost — how many rounds, how many trials, and
whether Story 20.5's one-talker-run-per-pair trick is available here (it is
**not**: chunk size perturbs the talker, so the arms are necessarily different
takes, and Story 20.4 §17's take-to-take variance warning is live again)

### AC #3 — Phase 2 (gated): implement and audition

**Given** a GO verdict and Commander's approval
**When** the chosen geometry is committed
**Then** it goes through the geometry threading rather than as a bare constant
edit, and all three sites follow — Story 20.6 verified this works in both
directions
**And** an NFR3 audition runs with a falsifiable prediction recorded first,
reference = what ships today
**And** the take-to-take variance problem is addressed explicitly, per AC #2

### AC #4 — No regressions

**Then** the exact bars hold — single-chunk streaming == `forward` bit-for-bit,
fp64 chunked == whole to ~1e-06 — and the accumulated suites pass with the
pre-existing failure set unchanged in count and identity

## Tasks / Subtasks

- [ ] **Task 1 — Watermark floor** (AC: #1) — derive the minimum viable `chunk_size`; report it before sweeping.
- [ ] **Task 2 — Sweep** (AC: #1) — one sitting, one machine, `cs25` control included, warm measurements, cold-compile cost stated.
- [ ] **Task 3 — Audition cost estimate** (AC: #2).
- [ ] **Task 4 — GATE.** Report to Commander. Stop.
- [ ] **Task 5 — Phase 2** (AC: #3), gated.
- [ ] **Task 6 — Regression** (AC: #4).

## Dev Notes

### Do not re-derive these

- Current `cs25` on the shipping path: long TTFA **1,362.4 ms**, segment 2
  **1,125.2 ms**, **45.01 ms/frame**, producer ratio ~0.58 (Story 20.6 §12).
- Per-chunk decode at `cs25`: **11.8 ms** post-retirement (Story 20.6).
- The codec is **12.5 Hz**; `decode(N) == 1920·N` with state caching.
- Story 20.1's curve is superseded and must not be quoted as current.

### What this story is NOT

- Not a change to codec state caching or the lookahead retirement.
- Not F6 (RTX 3060) or F7 (the qasync audit).
- **Not an audition.** Phase 1 spends no listening time. That is the entire point
  of the gate.

## References

- `_bmad-output/implementation-artifacts/20-1-ttfa-spike-faster-qwen3-tts-evidence.md` §5.2 (the superseded curve), §5.4 (why it was free and no longer is)
- `_bmad-output/implementation-artifacts/20-4-chunk-size-and-adaptive-cushion-evidence.md` §17 + its 2026-09-01 amendment (the closure and its reopening condition)
- `_bmad-output/implementation-artifacts/20-6-retire-the-lookahead-evidence.md` §12 (the same-sitting control, and why cross-session numbers misled)
- `tools/ttfa_spike_harness.py`, `_bmad-output/implementation-artifacts/20-6-compare-arms.py`

## Dev Agent Record

_(to be filled by dev agent)_

## Change Log

- 2026-09-02 — Drafted by Winston at Commander's direction to re-baseline before reopening. Phase-gated because every number justifying F2 predates three stories that changed the system underneath it, and because Story 20.6 §12 showed cross-session comparison has already produced one false conclusion in this epic.
