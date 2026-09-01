# Story 20.4: Chunk-Size Retune + Adaptive-Cushion Fix (Phase ⊥-Polish-3)

Status: ready-for-dev

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

- [ ] **Task 1 — Chunk geometry** (AC: #1)
  - [ ] 1.1 Change the streamer constants to `chunk_size = 10`, `lookahead = 5`.
  - [ ] 1.2 Thread the real geometry into `engage_compile_optimizations` from the call site, reading the streamer module rather than adding a second literal.
  - [ ] 1.3 Test that the compile path receives the streamer's actual window and fails on divergence.

- [ ] **Task 2 — Cushion policy** (AC: #2)
  - [ ] 2.1 Revise the release policy so a low-`P` host starts materially sooner than the cap; justify against the numbers.
  - [ ] 2.2 Address or explicitly defer the `CHARS_TO_AUDIO_SECONDS` overshoot.
  - [ ] 2.3 Prove the `≥16 GiB` static path is untouched.
  - [ ] 2.4 Tests for both branches, including the low-`P` release point.

- [ ] **Task 3 — Coupling** (AC: #3)
  - [ ] 3.1 Re-derive the adaptive-path behaviour at `chunk_size = 10` with the AC #2 fix in place.

- [ ] **Task 4 — Measure** (AC: #4, #6)
  - [ ] 4.1 Re-measure the producer emit/drain ratio on current code.
  - [ ] 4.2 GUI capture, ≥5 launches, short + long, against the 1b 192 ms / TOTAL 1,353 ms baseline. Group by `session_id`.
  - [ ] 4.3 Derive the sub-16 GiB effect; restate the deferred 3060 check.
  - [ ] 4.4 Write the evidence file.

- [ ] **Task 5 — Audition** (AC: #5) — Commander solo, short + long, CLONED voice.

- [ ] **Task 6 — Regression sweep** (AC: #7)

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

_(to be filled by dev agent)_

### Completion Notes List

_(to be filled by dev agent)_

### File List

_(to be filled by dev agent)_

## Change Log

- 2026-09-01 — Drafted by Winston from Story 20.1 Follow-ups B and C, shipped as one story because Story 20.1 found them coupled: `chunk_size = 10` worsens the sub-16 GiB cushion-to-talker ratio from 2.5× to 4.0×, so B alone would speed up large-VRAM hosts and leave the RTX 30xx tier pinned at the cap.
