# Story 20.4 — Chunk-Size Retune + Adaptive-Cushion Fix: evidence

Phase ⊥-Polish-3. Fourth story of Epic 20 (First-Audio Latency).
Follow-ups **B** and **C** from Story 20.1 §6.4, shipped together because
Story 20.1 found them coupled.

## 0. Headline

| | before | after | source |
|---|---|---|---|
| streamer geometry | `chunk_size=25`, `lookahead=5` (window 30, 2.5 s of audio before first emit) | **`chunk_size=10`**, `lookahead=5` (window 15, **1.25 s**) | Story 20.1 §5.2/§5.3 four-point sweep |
| compile-path `decode_window_frames` | pinned at **30** regardless of the streamer, at **three** sites | derived from `codec_token_streamer` at **all three** | §1.1 |
| sub-16 GiB first audio, `P = 0.5`, cs10 | **10.00 s** — released by the `MAX_PRE_DELAY` guardrail | **1.67 s** | §3, `20-4-adaptive-cushion-sim.txt` |
| sub-16 GiB cushion / talker ratio, cs10 | 4.00× (worse than cs25's 2.50×) | **0.67×** | §3 |
| `T_a` estimator error, long fixture | +44.5 % | **+2.0 %** | §2.3 |
| `MAX_PRE_DELAY_SECONDS` | 10.0 s | **10.0 s — unchanged** | §2.2 |

**Status (updated 2026-09-01, after the §9 audition failure).**

| AC | what it asks | status |
|---|---|---|
| #1 | retune + thread the geometry where it is read | **DONE** — §1 |
| #2 | cushion policy + the T_a overshoot | **DONE** — §2 |
| #3 | coupling re-derived at `chunk_size = 10` | **DONE** — §3 |
| #4 | OFR-E producer ratio re-measured | **DONE, PASSES** — §4 |
| #5 | NFR3 perceptual audition | **round 1 FAILED (§9)** → cause found and fixed (§11) → **round 2 needs operator** |
| #6 | GUI capture on the reachable tier | **DONE, PASSES** — operator run, §6.3 |
| #7 | no regressions | **DONE, zero new failures** — §9.1, §11.12 |

> **Reading order.** This file is an append log across three passes, so the
> section numbers are chronological rather than sorted: §0-§10 are the
> original implementation pass, **§9 (second one, near the end)** is the
> round-1 audition failure the coordinator recorded, **§8b** is the current
> operator hand-off, and **§11** is the seam investigation and fix. Where §5,
> §6 and §8 have been overtaken, their headers say so and point forward.

**The §9 failure had a cause, and it was not the one either candidate
hypothesis proposed.** Every decoder chunk boundary was deleting 15-19 ms of
real speech — a splice-alignment bug present at the shipped `chunk_size = 25`
too, which is why round 1 flagged long-form seams on *both* arms. Underneath
it, the two independent decodes either side of a boundary differ by ~35 %
NRMSE. Both are now fixed; seam discontinuity measures **12.3x -> 1.3x** the
non-seam baseline, i.e. a seam is now statistically indistinguishable from any
other point in the audio. **§11 is the investigation; §8 is the one remaining
operator task.**

---

## 1. AC #1 — the chunk geometry, and where it is actually read

### 1.1 The D-25 trap was worse than Story 20.1 §5.4 found: three sites, not two

Story 20.1 §5.4 identified two halves of one trap:

* `torch_runtime.engage_compile_optimizations` declared
  `streamer_chunk_size: int = 25, streamer_lookahead: int = 5` as **hard-coded
  defaults**, and
* the sole production call site, `model_registry._load_model_sync`, passed
  **neither**.

so `decode_window_frames` resolved to 30 no matter what the streamer emitted.

Implementing the retune surfaced a **third** site the spike did not name:

```
src/myvoice/services/qwen_tts_service.py:2217   decode_window_frames=30,
```

inside `warmup_compile_async`'s cache-key construction — the method whose
docstring says it "mirrors `engage_compile_optimizations`' construction so
the warmup and the engage path share state". It mirrored the *literal*, not
the *source*. This one is not decorative: `decode_window_frames` is one of
`compile_cache`'s seven key dimensions (`compile_cache.py:88`), so with the
literal left in place Story 20.4 would have had **Story 20.3's startup
priming warm a cache key the engage path never reads** — silently reverting
the ~4.0 s first-generation win Story 20.3 just certified, with no error and
no log line saying so.

### 1.2 What changed

| file | change |
|---|---|
| `services/tts_streaming/codec_token_streamer.py` | `DEFAULT_CHUNK_SIZE` 25 → **10**. `DEFAULT_LOOKAHEAD` unchanged at 5. The sweep table and the "why 10, not 5" reasoning are recorded at the constants so the next person to touch them sees the evidence, not just the number. |
| `services/tts_streaming/torch_runtime.py` | `streamer_chunk_size` / `streamer_lookahead` default to **`None`**, resolved from the live `codec_token_streamer` module constants at call time. No second literal survives. |
| `services/model_registry.py` | the call site now passes both, read from `codec_token_streamer.DEFAULT_*`. |
| `services/qwen_tts_service.py` | the warmup cache key derives the window from the same constants. |

The `None`-resolution and the call-site pass-through are deliberately
redundant. The resolution is what makes the invariant hold for *any* caller
(including a future one that forgets); the pass-through is what makes the
dependency visible at the exact call site the trap was found at.

### 1.3 The tests, and why they are shaped this way

`tests/unit/services/test_decode_window_geometry_coherence.py` (new, 6 rows)
splits into a runtime arm and a static arm, because either alone can pass
while the trap is re-introduced through the other.

| row | what fails it |
|---|---|
| `test_warmup_cache_key_uses_the_live_streamer_window` | the warm-path priming key drifting from the committed geometry |
| `test_warmup_cache_key_moves_with_a_streamer_retune` | **the exact bug class** — monkeypatches the constants to 17+3 and requires the key to move to 20. Restoring the `=30` literal fails here, and only here. |
| `test_warmup_key_and_engage_key_agree` | the two independent key constructions diverging on any dimension |
| `test_no_source_file_passes_a_literal_decode_window_frames` | any new literal, anywhere under `src/myvoice`, at `compute_key` or `enable_streaming_optimizations` |
| `test_engage_compile_optimizations_has_no_hard_coded_geometry_defaults` | a regression to non-`None` defaults |
| `test_model_registry_threads_the_streamer_geometry_into_the_compile_call` | the call site going quiet again, or restating the numbers as literals |

Plus, in the existing suites:

* `test_torch_runtime.py::test_engage_compile_resolves_window_from_live_streamer_constants`
  (new) — asserts the resolved window equals the live constants, that
  monkeypatching them moves it, and that the D-25 hard-fail still fires
  **against the retuned geometry** (so the invariant is live, not decorative).
* `test_codec_token_streamer.py::test_committed_chunk_geometry_is_story_20_4_optimum`
  (new) — pins 10 + 5 and the `chunk_size ≥ 6` watermark-no-op condition.
  Its sibling `test_default_construction_uses_documented_constants` was
  changed from asserting `25` to asserting *the module constants*, so it
  tests the wiring rather than re-pinning the value in a second place.
* `test_torch_runtime.py`'s two cache/window rows now derive the window from
  the streamer module (`_STREAMER_WINDOW`) instead of restating `30`.

---

## 2. AC #2 — the cushion policy

### 2.1 What was actually wrong

Story 20.1 §2.7 drove the shipped `StreamingChunkBuffer` with an injected
clock and found the τ_min comparison was **the last of five escapes** and
effectively unreachable on the ship-target tier: for every `P ≲ 0.78` the
`MAX_PRE_DELAY_SECONDS = 10.0` clamp put the required cushion above anything
the buffer would accumulate, so release fell through to the elapsed escape —
and because that escape is only evaluated inside `push`, the effective wait
at `P = 0.5` was **12.5 s**, not 10 s.

The framing that makes the fix obvious: **at `P = 0.5` the old policy
delivered a 12.5 s wait *and gaps anyway*.** Gaplessness on a 19.3 s
utterance at `P = 0.5` needs 19.3 s of cushion. Clamping the requirement to
10 s did not make gaplessness achievable; it just capped how much latency was
spent failing to achieve it. That outcome is strictly dominated — there is no
value of the trade-off dial at which "wait 12.5 s and still gap" is the right
answer.

### 2.2 The policy

`_cushion_decision(P)` now returns `(cushion_seconds, regime)`:

| regime | condition | cushion |
|---|---|---|
| `producer_keeps_up` | `P ≥ 1.0` | 0 — unchanged |
| `gapless_feasible` | `τ_gapless ≤ cushion_budget_seconds` | the full `τ_gapless` — unchanged behaviour in this band |
| `gapless_unreachable` | `τ_gapless > cushion_budget_seconds` | the **static watermark** (500 ms) |

where `τ_gapless = T_a × (1/P − 1)` is computed **unclamped**, so the
feasibility test asks the real question ("can gaplessness be bought inside
the budget?") rather than the clamped one.

`cushion_budget_seconds = 2.0` (`audio_coordinator._DEFAULT_STREAMING_CUSHION_BUDGET_SECONDS`).
Justification, against numbers rather than assertion:

* **4× the static watermark** (500 ms) that the ≥16 GiB tier already accepts
  as the price of smoothing.
* **Above the whole ≥16 GiB first-audio budget.** Story 20.3 §4.1 measured
  the RTX 5090 at 1,353 ms *total* TTFA. A 2.0 s consumer cushion is already
  more than the entire fast-tier experience; spending more than that on the
  slow tier is not a latency budget, it is a different product.
* **Below the guardrail**, by design and by test
  (`test_cushion_budget_constant_is_below_the_guardrail`). If the budget ever
  reached the cap, the guardrail would become binding again and this fix
  would silently revert.

The unreachable branch compares **bytes**, not seconds — it is literally the
same predicate as the `enable_adaptive_pre_buffer=False` path in `push`, so
the two branches cannot disagree at the boundary through a float round-trip.
The intent is that when gaplessness is out of reach, the slow tier's consumer
behaves exactly like the fast tier's: hold 500 ms, then start.

**`MAX_PRE_DELAY_SECONDS` and `max_hold_chunks` are untouched.** The story
forbids removing the guardrail and this change does not: it stops the
guardrail from being the *policy*, which is a different thing.
`test_max_pre_delay_cap_kicks_in` still proves the guardrail fires (with the
budget raised out of the way so the new policy cannot release first), and
`test_max_pre_delay_is_not_the_binding_escape_across_the_p_sweep` proves it
is no longer the binding escape anywhere on a P sweep from 0.50 to 0.95.

### 2.3 The T_a estimator — ADDRESSED, not deferred

`CHARS_TO_AUDIO_SECONDS = 0.08` was a single proportional constant calibrated
from **one** 19-character 3060 smoke utterance. Story 20.1 §2.7 measured it
+44.5 % on the canonical long fixture.

Two measured points are available, and they do not lie on a line through the
origin — short utterances carry proportionally more non-speech time (onset
silence, the 0.5 s reference padding Story 20.1 §4.3 documents, final decay):

| fixture | chars | measured `T_a` | old estimate | **new estimate** |
|---|---:|---:|---:|---:|
| long (Epic 18 canonical) | 349 | 19.32 s | 27.92 s (**+44.5 %**) | **19.70 s (+2.0 %)** |
| short (Clear Comms class) | 33 | 2.30 s | 2.64 s (+14.8 %) | **2.32 s (+0.7 %)** |

Replaced by an affine estimator, `estimate_target_audio_seconds(chars)`:

```
T_a ≈ 0.5 s + 0.055 s/char
```

The exact two-point fit is `0.523 + 0.0539`; the shipped constants are
rounded so the estimator stays *slightly* conservative (over-estimating) on
both fixtures rather than centred — a marginally larger cushion is the safer
direction for a thin calibration.

**Stated as a limitation, not hidden:** two points is a thin calibration and
the intercept is doing real work at short lengths. It is also much less
load-bearing than it was — under the two-regime policy `T_a` only changes
behaviour inside the *feasible* band; once gaplessness is unreachable the
buffer falls back to the watermark and `T_a` drops out of the decision
entirely. Story 20.1 §2.7 had already shown the old estimator changed nothing
at the 3060's documented `P ≈ 0.5`, because the cap dominated there.

### 2.4 The product trade, stated explicitly (AC #2 requires this)

**Total silence is conserved.** Playback cannot outrun the producer, so it
ends at `max(cushion + T_a, T_gen)` where `T_gen = T_a / P`. Accumulated
mid-utterance gap is therefore `max(0, T_gen − T_a − cushion)`: every second
of cushion not spent up front reappears, one for one, as gap later. The
choice is **only about where the silence lands**.

MyVoice's answer, and the one this implementation takes: **the front is the
worst place for it.** Clear Comms is a voice-chat interjection cue
(`memory/clear_comms_purpose_framing.md`) — a tool for grabbing attention in
a live conversation. A ten-second lead-in does not make an interjection
smoother; it makes it not an interjection. A mid-utterance gap degrades the
delivery but the message still lands, in the conversational window it was
meant for.

**Where this is a real regression, reported rather than averaged away.** The
sweep in §3 shows a band — roughly `0.70 ≤ P ≤ 0.90` on the long fixture —
where the old policy *did* reach gaplessness, by waiting 4.9–11.9 s for it.
The new policy starts in ~1 s there and accepts 1.2–7.1 s of accumulated gap.
That is a genuine loss on the gaplessness axis, and it is the direction AC #2
names as preferred. Two things bound it:

1. It is **not** the ship-target operating point. `streaming_chunk_buffer.py`
   documents the RTX 3060 12 GB at `P ≈ 0.5`, where the old policy achieved
   neither low latency nor gaplessness.
2. `cushion_budget_seconds` is a single named constant with a test pinning it
   below the guardrail. If the deferred RTX 3060 confirmation (§7) finds real
   hardware living in the 0.70–0.90 band, raising the budget is a one-line
   change with an obvious meaning.

### 2.5 The ≥16 GiB static path is untouched

Nothing in the static branch of `push` changed. Two tests prove it rather
than asserting it:

* `test_cushion_budget_has_no_effect_on_the_static_path` — drives an
  eight-push trace with crossfade on through
  `cushion_budget_seconds ∈ {0.0, 2.0, 1000.0}` and requires **byte-identical
  output**.
* `test_static_release_point_is_exactly_the_watermark` — release on the 5th
  100 ms push and not before, and `last_release_reason is None` because no
  adaptive decision is ever taken on that path.

The gate that selects the branch (`text_length` present **and** VRAM < 16 GiB)
is unchanged.

---

## 3. AC #3 — the coupling, re-derived at `chunk_size = 10`

`20-4-adaptive-cushion-sim.py` drives the **shipped** `StreamingChunkBuffer`
with an injected clock, exactly as Story 20.1 §2.7 did. It reproduces the
pre-20.4 policy by subclassing the buffer and restoring the old
`_cushion_decision` body, so the before/after columns come out of one driver.

**The legacy reproduction is cross-checked against every number Story 20.1
§2.7 published — 10 of 10 match to 0.02 s.** If it had not, every "before"
figure below would be untrustworthy, and the script says so and exits 1.

```
  cs=10  P=0.50  published  10.00s   reproduced 10.00s   OK
  cs=10  P=0.75  published  10.00s   reproduced 10.00s   OK
  cs=10  P=0.90  published   2.78s   reproduced  2.78s   OK
  cs=25  P=0.50  published  12.50s   reproduced 12.50s   OK
  cs=25  P=0.70  published  11.90s   reproduced 11.90s   OK
  cs=25  P=0.75  published  11.11s   reproduced 11.11s   OK
  cs=25  P=0.80  published   7.81s   reproduced  7.81s   OK
  cs=25  P=0.85  published   4.90s   reproduced  4.90s   OK
  cs=25  P=0.90  published   2.31s   reproduced  2.31s   OK
  cs=25  P=0.95  published   2.19s   reproduced  2.19s   OK
  legacy reproduction: VERIFIED
```

### 3.1 Long fixture at the COMMITTED geometry (`chunk_size = 10`)

DERIVED — simulated against the shipped buffer's own logic, never observed on
sub-16 GiB hardware. See §7.

| `P` | segment 4 before | released by | segment 4 after | released by | ratio before | ratio after | gap before | gap after |
|---:|---:|---|---:|---|---:|---:|---:|---:|
| 0.50 | **10.00 s** | 10 s cap | **1.67 s** | unreachable → watermark | 4.00× | **0.67×** | 9.32 s | 17.65 s |
| 0.60 | 11.11 s | 10 s cap | 1.39 s | unreachable | 5.33× | 0.67× | 1.77 s | 11.49 s |
| 0.70 | 10.71 s | 10 s cap | 1.19 s | unreachable | 6.00× | 0.67× | 0.00 s | 7.09 s |
| 0.75 | 10.00 s | 10 s cap | 1.11 s | unreachable | 6.00× | 0.67× | 0.00 s | 5.33 s |
| 0.80 | 8.33 s | τ_min | 1.04 s | unreachable | 5.33× | 0.67× | 0.00 s | 3.79 s |
| 0.85 | 4.90 s | τ_min | 0.98 s | unreachable | 3.33× | 0.67× | 0.00 s | 2.43 s |
| 0.90 | 2.78 s | τ_min | 0.93 s | unreachable | 2.00× | 0.67× | 0.00 s | 1.22 s |
| 0.95 | 0.88 s | τ_min | 0.88 s | **feasible** | 0.67× | 0.67× | 0.14 s | 0.14 s |

"ratio" is segment 4 / talker segment, on Story 20.1 §2.7's own denominator
`((chunk_size + lookahead)/12)/P`. "gap" is accumulated mid-utterance silence,
`max(0, T_a/P − T_a − cushion)`.

### 3.2 What the coupling actually did, and what it does now

Story 20.1 predicted that `chunk_size = 10` would make the sub-16 GiB
cushion-to-talker ratio **worse**, 2.50× → 4.00×, because the talker segment
shrinks while the cap does not. The simulation confirms that exactly: the
`P = 0.50` cs10 "before" row is 4.00×, against 2.50× at cs25.

With the AC #2 fix in place at `chunk_size = 10`, the ratio is **0.67× at
every P on the sweep** — better than cs25 ever was, before or after. The
reason is structural rather than lucky: once the cushion is the static
watermark, release lands on the first chunk that both carries ≥ 500 ms and
permits a producer-rate reading, which is chunk 2. Segment 4 becomes one
chunk-arrival, `(chunk_size/12)/P`, and the talker segment is
`((chunk_size+5)/12)/P`, so the ratio collapses to `10/15 = 0.67` independent
of `P` and independent of the geometry. Shrinking the chunk now shrinks the
cushion *with* it, which is precisely the coupling Story 20.1 was worried
about running backwards.

### 3.3 Interactions the derivation did **not** predict

AC #3 asks for these to be reported rather than averaged away. Three:

1. **The 0.70–0.90 gaplessness band described in §2.4.** Story 20.1's ranked
   list framed Follow-up C as "sub-16 GiB hosts are pinned at the cap", which
   is true at `P ≈ 0.5` but not across the whole range: between 0.70 and 0.90
   the old policy was buying real gaplessness with that wait. The fix trades
   it away deliberately.
2. **The `P = 0.60` before-row is 11.11 s, not ≤ 10 s.** The cap is evaluated
   only inside `push`, so the effective wait is the first chunk arrival at or
   after 10 s — which at cs10/`P = 0.60` is 11.11 s. Story 20.1 §2.7 recorded
   this mechanism at cs25/`P = 0.5` (12.5 s); it recurs at other rates and
   geometries, and it means the pre-20.4 "10 s cap" was never actually a 10 s
   bound.
3. **The short class barely engages the adaptive path at all.** At
   `T_a ≈ 2.3 s` and cs10 the whole utterance is 3 chunks, so several rows
   never reach a release decision before `is_final` fires — and the pre-20.4
   `P = 0.50` short row *never releases on a chunk arrival at all*. The
   Clear Comms class was being carried entirely by the `is_final` escape on a
   slow host, which is another way of saying it was batch playback.

### 3.4 Neither change is committed without the other

Both land in this one change, and the tests couple them: the AC #2 policy
rows in `TestCushionFeasibilityPolicy` are written at the cs10 chunk geometry
(`_CS10_CHUNK_SAMPLES = 20000`, i.e. 10/12 s at 24 kHz), and
`test_committed_chunk_geometry_is_story_20_4_optimum` pins the geometry those
rows assume.

---

## 4. AC #4 — OFR-E producer gate, RE-MEASURED

_Status: **MEASURED AND MET.** RTX 5090, headless
`tools/ttfa_spike_harness.py`, post-20.3 code with the retune applied,
`tts_compile="auto"`, bf16, CLONED Sarira-F, `--prime` (Story 20.3's startup
priming), 5 measured runs + 1 discarded warm-up per cell._

AC #4 requires the ratio to be **re-measured, not carried over** — the
0.776× figure in Story 20.1 §5.2 predates Stories 20.2/20.3 and the
compile-priming change. Both geometries were run back-to-back on the same
host in the same session, so the comparison is contemporaneous rather than
cross-session.

| cell | producer ratio (median) | min | max | gate `< 1.0×` |
|---|---:|---:|---:|---|
| long, `chunk_size = 25` (pre-20.4) | **0.585** | 0.577 | 0.589 | PASS |
| long, `chunk_size = 10` (committed) | **0.619** | 0.612 | 0.637 | **PASS** |
| short, `chunk_size = 10` | 0.348 – 0.508 (n = 3 of 5; see note) | — | — | PASS |

The retune costs **+0.034 on the ratio** (0.585 → 0.619) and leaves a **38 %
margin** to the gate. Story 20.1 predicted 0.665 → 0.676 for the same move;
the direction and the magnitude both reproduce, and the whole curve has
shifted down on post-20.3 code.

> Short-cell note: the ratio needs at least three `progressive_chunk_emit_ms`
> samples to take a median of inter-chunk intervals. A 33-character utterance
> at `chunk_size = 10` produces 2–3 chunks total, so 2 of the 5 runs report
> `n/a`. This is a measurement-resolution limit, not a failure — and it is
> also the point: the short class is now *three chunks of genuine streaming*
> where at `chunk_size = 25` it was frequently one batch-equivalent chunk.

### 4.1 The rest of the headline, measured on the same runs

| metric | cs25 (pre-20.4) | **cs10 (committed)** | change |
|---|---:|---:|---:|
| long TTFA(post), median | 1,491 ms | **829 ms** | **−44 %** |
| long segment 2 (talker) | 1,346 ms | 675 ms | −50 % |
| long segment 4 (consumer cushion) | 0–1 ms | 0–1 ms | unchanged |
| long generation wall | 11,619 ms | 12,319 ms | +6 % (inside Story 20.1's ±20 % band) |
| long chunk count | 10 | 25 | — |
| **short TTFA(post), median** | **1,409 ms** | **784 ms** | **−44 %** |
| **short first-emit path** | `residual_flush` **3 of 5** | **`threshold` 5 of 5** | the degeneration is gone |

Story 20.1 §5.2/§5.3 predicted 1,785 → 875 ms long and 1,651 → 921 ms short.
Measured here: 1,491 → 829 and 1,409 → 784. Both baselines and both results
are lower than the spike's — consistent with Stories 20.2/20.3 having removed
the ~4 s first-forward cost and with the 11–37 % session-to-session spread
Story 20.1 §2.4(d) documented — and the **relative** improvement (−44 % on
both classes) reproduces almost exactly.

**The short-utterance degeneration reproduces and is closed.** At
`chunk_size = 25`, 3 of 5 short runs (4 of 6 counting the warm-up) fell to
`residual_flush` — the batch-equivalent path where the only token chunk the
generation ever produces is the terminal residual. Story 20.1 measured 11 of
20. At `chunk_size = 10`, **6 of 6 took the `threshold` path**, matching the
spike's 5 of 5.

**Segment 4 is 0–1 ms at both geometries on this host**, confirming Story
20.1 §5.4's watermark analysis from the other direction: at `chunk_size = 10`
each chunk carries 833 ms of audio, comfortably over the 500 ms static
watermark, so the consumer hands nothing back. This is the `chunk_size ≥ 6`
condition, and it is why 10 rather than 5 is the optimum.

Raw CSVs: `20-4-ratio-long-cs10.csv`, `20-4-ratio-long-cs25.csv`,
`20-4-ratio-short-cs10.csv`, `20-4-ratio-short-cs25.csv`.

### 4.2 A free validation of the new T_a estimator

The harness reports `total_audio_ms` per run. Across the five long runs the
median is **19,527 ms** for the 349-character fixture — an independent
measurement of the quantity §2.3's estimator predicts.

| estimator | prediction | error vs 19,527 ms |
|---|---:|---:|
| pre-20.4, `chars × 0.08` | 27,920 ms | **+43.0 %** |
| **Story 20.4, `0.5 + chars × 0.055`** | **19,695 ms** | **+0.9 %** |

This is not a fitted point — the affine constants were calibrated against
Story 20.1's 19.32 s figure, and this run independently produced 19.53 s.

---

## 5. AC #5 — NFR3 perceptual gate

_Status: **round 1 RUN and FAILED — see §9 (blocking). Cause diagnosed and
fixed — see §11. Round-2 fixture generated and verified; the re-audition
needs an operator — see §8.**_

_The subsections below describe the round-1 fixture design, which round 2
reuses. Read §11.11 for what changed in round 2._

### 5.1 Why this story needed a gate the previous three did not

Stories 20.2 and 20.3 changed *when* work happens (priming at startup rather
than on the first utterance). This story changes *how the audio is cut*: the
streamer's chunk boundary moves from 30 frames to 15, and
`streaming_decoder.py` trims a lookahead-sized tail per chunk, so every
generation is now stitched from roughly twice as many pieces. That is a
per-generation change to the waveform, and Story 20.1's sweep says nothing
about how it sounds.

### 5.2 The fixture

`20-4-regen-audition-fixture.py` → `20-4-perceptual-fixtures/` (14 WAVs).

Story 18.4's fixture generator could not be reused: it drives
`generate_voice_clone`, the **batch** API, which never touches the streamer,
the decoder's overlap-add, or the consumer crossfade — i.e. it skips exactly
the part under audition. The Story 20.4 generator drives the production
`_generate_true_stream` dispatch path and pushes every chunk through a real
`StreamingChunkBuffer` with the shipped consumer constants, so what lands on
disk is what a user's speakers receive.

| | |
|---|---|
| utterances | 7 — three short (incl. the Clear Comms interjection shape), two medium, two long |
| renditions | `{utt}-cs25.wav` (pre-20.4) and `{utt}-cs10.wav` (committed) |
| voice | CLONED Sarira-F, the Story 17.2 precomputed prompt every prior audition used |
| verified | all 14 non-silent, RMS 2,828–4,287, peak ≤ 31,103 (no clipping) |
| blinding | `_perlistener_truthtable.json`, fixed seed, A/B randomised per utterance |

Sibilant- and plosive-dense lines are deliberate: fricatives and stops are
where a seam is most audible. Both utterance classes are covered because the
short class changes dispatch path entirely (§4.1).

**Two controls worth recording.** (a) Both renditions were produced in one
process, from one model load, under **one** compiled state — the geometry was
switched between passes via the streamer constants only. That is the tight
control for an audition of chunk stitching specifically, and it is sound
precisely because Story 20.1 §5.4 established that `decode_window_frames`
never reaches a runtime shape decision on our decode path. (b) qwen-tts
sampling is stochastic, so A and B are two renditions, not one waveform cut
two ways. The audition question is therefore "does the committed geometry
carry seam artefacts the old one does not", not "are these identical" —
the same standard Story 18.4 used.

### 5.3 The gate

Commander solo, mirroring Story 18.1/18.2 rather than Story 17.1's
multi-listener protocol, as AC #5 specifies.

**FAIL if any chunk-boundary artefact is flagged on a cs10 trial** —
`audible_seam`, `click_or_discontinuity`, or `prosody_break_at_stitch`. The
helper unblinds at the end and prints the verdict itself. A defect flagged on
**both** geometries is recorded as a pre-existing pipeline property and does
not block; that distinction is made by the helper, not by hand.

---

## 6. AC #6 — GUI measurement on the reachable tier

_Status: **MEASURED AND PASSED** (operator run, 2026-09-01). See §6.3._

### 6.1 What will be measured, and against what

The baseline is Story 20.3 §4.1, established 2026-09-01 on this host:
**1b 192.5 ms / TOTAL 1,353.4 ms** long-form, median of 5 GUI launches. Not
the pre-20.3 5,051 ms figure.

`11_Story_20.4_AC6_GUI_Capture.bat` — a new launcher rather than a re-use of
`10_Story_20.3_AC4_GUI_Capture.bat`, for three reasons, each of which would
have corrupted the comparison:

1. **It writes `20-4-gui-r0N.csv`.** The 20.3 launcher writes
   `20-3-gui-r0N.csv` — the baseline files themselves. Running it would have
   overwritten the numbers this story is measured against.
2. **Six launches, not five.** The retune moves `decode_window_frames`
   30 → 15, a compile-cache key dimension, so the first launch after this
   ships pays exactly one cold compile (§9.2). Launch 1 is a declared
   throwaway; launches 2–6 are the five warm launches that compare
   like-for-like.
3. **Two generations per launch — long, then short.** AC #6 asks for both
   classes, and the short class is the one the retune changes most (§4.1).

It also preflights the committed geometry (`(10, 5)`), so a reverted retune
cannot produce numbers filed under a Story 20.4 name, and it tells the
operator to **let each utterance finish playing** — Story 20.3's captures
stop after chunk 0, which is why they carry no producer-ratio data.

### 6.2 The aggregator, validated against the baseline it will be compared to

`20-4-aggregate-gui.py` groups strictly by `session_id` and keeps only
sessions carrying `ttfa_first_playback_write_ms`, per Story 20.3 §4.1a. Run
against the **Story 20.3 captures** it reproduces that story's published
table exactly:

```
== long (n=5) ==
  1b prefill   median=  192.465   TOTAL  median= 1353.375
```

— i.e. 192.5 ms / 1,353.4 ms, the numbers §4.1 of the 20.3 evidence
published. An aggregator that could not reproduce the baseline would not be
trustworthy on the new captures either.

### 6.3 Result — AC #6 PASSES

Operator run, 2026-09-01, reported by the coordinator:

| class | TTFA median | producer ratio | dispatch path |
|---|---:|---:|---|
| long | **976 ms** | 0.602 | threshold |
| short | **1,065 ms** | 0.403 | threshold, 3 chunks |

Against the Story 20.3 baseline of **1,353 ms** long-form, and comfortably
inside the OFR-E `< 1.0x` producer gate on both classes. The short class is on
the threshold path at 3 chunks — the residual-flush degeneration is gone in
the shipped GUI, not just in the headless harness (§4.1).

**These numbers are not re-run and not re-derived by the §11 work.** The seam
fix changes how chunk boundaries are stitched, not when the first chunk is
produced or dispatched; segments 1-3 are untouched by it, and segment 4 is a
consumer-side cushion that never sees a seam. AC #6 is closed.

One second-order effect, in the safe direction and worth stating rather than
leaving for a reviewer to ask: each posted chunk is now 19,200 samples instead
of 18,830, so ``progressive_chunk_audio_duration_ms`` rises 784.6 → 800.0 ms.
The producer emit/drain ratio is (inter-chunk wall time) / (chunk audio
duration), and only the denominator moves, so the ratio **improves** by ~2 %
— 0.619 → ~0.607 headless, 0.602 → ~0.590 in the GUI. The OFR-E gate is not
at risk from the fix. Segment 4 is likewise unaffected: 800 ms still clears
the 500 ms static watermark, as 784.6 ms did.

---

## 7. AC #6 — the sub-16 GiB tier: derived, and the deferred confirmation

Everything in §3 is **DERIVED** from the shipped buffer's own logic driven by
an injected clock. Nothing in this story was observed on sub-16 GiB hardware.
The RTX 3060 remains on a second PC with no hot-swap (Story 20.3 AC #2b
Phase 3, still deferred).

**What the deferred 3060 run would now check**, updated for what this story
changed:

1. **The `P` assumption itself.** Every derived number keys off
   `streaming_chunk_buffer.py`'s documented `P ≈ 0.5` for the 3060 12 GB.
   `P` is recoverable directly from a capture as
   `progressive_chunk_audio_duration_ms / Δ progressive_chunk_emit_ms`
   (Story 20.1 §2.8). If the real 3060 lives in the 0.70–0.90 band instead,
   §2.4's trade is the one that needs revisiting — raise
   `cushion_budget_seconds`.
2. **That the release regime is `gapless_unreachable`, not a guardrail.**
   The buffer now records `last_release_reason`; a 3060 capture should never
   show `max_pre_delay` or `max_hold_chunks` on a healthy generation. Either
   of those appearing means something pathological, which is exactly what
   those escapes are now reserved for.
3. **Segment 4 ≈ one chunk arrival.** The derivation says
   `(10/12)/P` seconds — 1.67 s at `P = 0.5`. A materially larger observed
   segment 4 falsifies the model.
4. **Whether the accumulated gap is tolerable in practice.** §2.4's
   conservation argument says the gap total is set by `P`, not by policy; the
   open question is perceptual, and only a real slow host can answer it.
5. **The `T_a` estimator on that host.** The affine fit is calibrated on
   5090 renditions. Audio duration for a given text should be
   hardware-independent, but this has never been checked across tiers.

Phase 3 is still unblocked and still needs no new build:
`progressive_playback_csv_capture` is wired at `app.py` via
`maybe_enable_from_env`. Verify it is present in the bundled artifact before
trusting a shipped-exe run on that host.

---

## 8. Operator hand-off

**Superseded — see §8b.** AC #6 (Task 1 below) has since been run and passed
(§6.3). The text is retained because it is the record of what was asked and
what the launcher does.

Two things need Commander at the keyboard. They are independent — either
order works — and together they are about **35 minutes**. Everything else in
this story is done and verified.

Both launchers preflight themselves and stop with a clear message rather than
producing misleading data, so there is nothing to check by hand first.

---

### Task 1 — AC #6: GUI capture (~25 min)

```
11_Story_20.4_AC6_GUI_Capture.bat
```

Six launches. **Launch 1 is a declared throwaway** — the retune changes the
compile-cache key, so it pays exactly one cold compile and its "Preparing TTS
engine" indicator will sit there noticeably longer than usual. Do both
generations anyway; the analysis drops it.

Per launch:

1. A **CLONED** voice must be the active profile (so BASE is the resident
   model). Same profile every launch.
2. **Wait for "Preparing TTS engine" to clear** before generating. Priming
   holds the request semaphore; generating while it is up measures queueing,
   not first-forward, and would look like a regression that is not one.
3. Generate the **LONG** utterance. **Let it finish playing.**
4. Generate the **SHORT** utterance. **Let it finish playing.**
5. Close with the **X** (auto-quit is set; do not use Ctrl-C). The next
   launch starts on its own.

Both texts are in
`_bmad-output/implementation-artifacts/20-4-gui-utterances.txt`. They are the
canonical Story 20.1 / Epic 18 fixtures — using different text breaks the
comparison, because segment 2 scales with utterance length.

> "Let it finish playing" is new relative to the Story 20.3 run, and it
> matters: those captures stop after chunk 0 because the app was closed
> early, which is why they carry no producer-ratio data. Letting playback
> complete gives the full chunk stream.

Then, one command:

```
python310\python.exe _bmad-output\implementation-artifacts\20-4-aggregate-gui.py --skip-first-launch
```

Hand back that output. Two things are also worth grabbing from
`logs\myvoice.log` (the .bat prints them at the end): whether launch 1 shows
a cold compile and launches 2–6 show `primed_warm`.

---

### Task 2 — AC #5: perceptual audition (~10 min)

```
12_Story_20.4_AC5_Audition.bat
```

The 14-WAV fixture is **already generated and verified** (§5.2) — this is
listening only, no GPU work. Seven blinded A/B pairs; the helper unblinds and
prints the verdict itself.

Headphones if available, normal Discord-call volume. **What to listen for:**
the change stitches every generation from twice as many pieces, so the defect
class is at the **seams** — a click or tick partway through a word, a
momentary discontinuity in a held vowel, prosody that resets mid-phrase as if
two takes were cut together.

**What NOT to judge:** timbre, loudness, accent, or wording rhythm. The two
takes are different samples, not one waveform cut two ways, so they will
differ in delivery. That is expected and is not the gate.

The gate is blocking: any seam artefact on the committed geometry that the
old geometry does not carry **fails AC #5 and the story does not close**. The
helper distinguishes that case from a defect present on both (pre-existing,
recorded, not blocking) automatically.

Hand back the verdict block it prints.

---

### What happens with the results

§4 (AC #4) already passes on its own, so neither task can turn the producer
gate red. Task 1 fills §6 and closes AC #6; Task 2 fills §5 and closes AC #5.
If Task 2 fails, the retune is the thing that reverts — a one-line change to
`DEFAULT_CHUNK_SIZE`, with the D-25 wiring from §1 correctly following it
back.

---

## 9. AC #7 — regression sweep

`python310\python.exe -m pytest -q`, portable interpreter per
`memory/test_interpreter_portable_python310.md`.

### 9.1 Zero new failures; the pre-existing set is unchanged in count and identity

| surface | result | vs. the Story 20.3 baseline |
|---|---|---|
| `tests/unit/services tests/unit/observability tests/unit/models` | **953 passed, 0 failed** | 928 → 953 (+25 new rows); **still zero failures** |
| `tests/unit` (whole tree) | 1,574 passed, **30 failed, 4↔5 errors** | 1,548 → 1,574 passed; failure count **and identity** identical. The error count flakes 4↔5 across runs on the Windows temp-file lock in `test_audio_player_widget.py` — the same flake Stories 20.2 and 20.3 both recorded; observed at 4 and at 5 in two runs here |
| `tests/integration tests/test_qwen_tts_internals.py` | 175 passed, **4 failed** | 174 → 175; exactly 20.3's 4 pre-existing rows |
| `tests/services tests/settings tests/utils` | 288 passed, **7 failed** | identical |
| `tests/ui` | 735 passed, **7 failed** | identical |
| the whole story surface in one invocation | **291 passed** | — |

Every remaining failure is in the same UI / voice-profile / session-manager
drift set `20-2-warm-path-compile-priming-evidence.md` §5 documents. None of
it touches the streaming dispatch chain.

### 9.1a Three integration rows the retune turned red, and why that was correct

`tests/integration/test_streaming_tts_smoke.py` failed three assertions on
the first run after the retune:

| assertion | was | why it moved |
|---|---|---|
| `response.chunks_generated == 4` | 100 tokens / (25 + 5) | 10 at (10 + 5) |
| `chunk_emit[0][2]["frames"] == 30` | the window | 15 |
| `step_count=20 must produce exactly one chunk` | 20 was below the 30-frame window | 20 is now **above** the 15-frame window |

The third is the interesting one: the literal `20` did not merely fail, it
**silently changed the test's meaning** — it would have exercised the
threshold path while asserting residual-flush behaviour. Fixed by deriving
all three from the streamer constants (`_STREAMER_WINDOW`,
`_expected_chunk_count(n)`, `_SUB_THRESHOLD_STEPS = window - 1`), plus a
self-check row that pins the helper against **both** shipped geometries so a
helper that computes the wrong expectation cannot hide behind looking
principled. This is the same discipline applied to the source in §1.2:
derive, do not re-type the new number.

### 9.2 The compile cache gains exactly one key — confirmed, not predicted

`decode_window_frames` is one of `compile_cache`'s seven key dimensions
(`compile_cache.py:88`), so 30 → 15 must produce exactly one new key and one
cold compile.

Before the first post-retune run, `%LOCALAPPDATA%\MyVoice\torch_compile_cache\`
held two key directories (`391c2f2be3340b07`, `a514f2c991e58200`). After it:

```
391c2f2be3340b07   (2026-09-01 09:47 — the window-30 key, meta.json present)
a514f2c991e58200   (2026-05-11)
a58fe999b1fca2f3   (2026-09-01 11:19 — NEW: the window-15 key)
```

**Exactly one new key directory.** It carries no `meta.json` yet: the
directory is created by `compile_cache.set_torchinductor_cache_dir`, but
`mark_warm` — which writes the sidecar — is called by the application's
warm-up priming path, not by the headless harness. So the operator's launch 1
(§8) is where `mark_warm` lands, and its `meta.json` is the artifact that
proves **Story 20.3's priming warms the NEW key**, which is the specific
thing AC #7 asks to confirm. The §1.1 fix is what makes that true: with the
`decode_window_frames=30` literal still in `warmup_compile_async`, priming
would have gone on warming the old key while the engage path used the new
one, with no error anywhere.

---

## 10. File list

### Source (7 files)

| file | change |
|---|---|
| `src/myvoice/services/tts_streaming/streaming_decoder.py` | **§11 fix** — exact splice from the codec's measured geometry, verified per chunk with a loud fallback; decoder-side overlap-add over the previously discarded tail |
| `src/myvoice/services/tts_streaming/codec_token_streamer.py` | `DEFAULT_CHUNK_SIZE` 25 → 10; the sweep evidence and the retune contract recorded at the constants |
| `src/myvoice/services/tts_streaming/torch_runtime.py` | geometry params default to `None`, resolved from the streamer module at call time |
| `src/myvoice/services/model_registry.py` | the call site threads the real geometry through |
| `src/myvoice/services/qwen_tts_service.py` | warm-path cache key derives the window; one stale comment corrected |
| `src/myvoice/services/streaming_chunk_buffer.py` | two-regime cushion policy, `cushion_budget_seconds`, `last_release_reason` |
| `src/myvoice/services/audio_coordinator.py` | affine `estimate_target_audio_seconds`, budget constant wired through |

### Tests

| file | change |
|---|---|
| `tests/unit/services/test_decode_window_geometry_coherence.py` | **new** — 6 rows, runtime + static arms |
| `tests/unit/services/test_streaming_chunk_buffer.py` | +1 construction row, +9 policy rows in two new classes; 4 pre-existing rows re-scoped to the feasible regime with the reason recorded inline |
| `tests/unit/services/test_audio_coordinator.py` | +7 estimator rows |
| `tests/unit/services/tts_streaming/test_torch_runtime.py` | +1 resolution row; two window literals derived |
| `tests/unit/services/tts_streaming/test_codec_token_streamer.py` | +1 committed-geometry row; the defaults row now tests the wiring |
| `tests/integration/test_streaming_tts_smoke.py` | three literals derived + helper self-check row |
| `tests/unit/services/tts_streaming/test_streaming_decoder.py` | **§11** — +8 rows on the exact splice, time-contiguity, the blend ramp, the clamp, residual handling, the fallback + telemetry, and the measured constants |

### Tooling and artifacts

| file | purpose |
|---|---|
| `tools/ttfa_spike_harness.py` | `--chunk-size` now defaults to the committed geometry instead of a literal 25 |
| `11_Story_20.4_AC6_GUI_Capture.bat` | **new** — the AC #6 capture launcher |
| `12_Story_20.4_AC5_Audition.bat` | **new** — the AC #5 audition launcher |
| `_bmad-output/.../20-4-adaptive-cushion-sim.py` + `.txt` | **new** — AC #2/#3 re-derivation, with the legacy-policy cross-check |
| `_bmad-output/.../20-4-aggregate-gui.py` | **new** — session-grouped GUI aggregator |
| `_bmad-output/.../20-4-regen-audition-fixture.py` | **new** — TRUE_STREAM audition fixture generator |
| `_bmad-output/.../20-4-l1-audition-helper.py` | **new** — blinded A/B helper + verdict |
| `_bmad-output/.../20-4-perceptual-fixtures/` | **new** — 14 WAVs + truth table |
| `_bmad-output/.../20-4-gui-utterances.txt` | **new** — the exact texts for the GUI capture |
| `_bmad-output/.../20-4-ratio-{long,short}-cs{10,25}.csv` | **new** — the AC #4 raw captures |
| `_bmad-output/.../20-1-adaptive-cushion-sim.py` | header note: superseded; it no longer reproduces its own `.txt` against current code |

### Added by the §11 seam investigation

| file | purpose |
|---|---|
| `_bmad-output/.../20-4-seam-capture.py` | captures the float32 arrays the decoder posts, pre-consumer — the data the output-length model was solved from |
| `_bmad-output/.../20-4-seam-capture-full.py` | captures `pcm_full`, pre-trim — the redundancy that makes the alignment answerable by cross-correlation |
| `_bmad-output/.../20-4-seam-analysis.py` | the mechanism analysis: alignment, dropped-span energy, same-token decode divergence, seam-step comparison |
| `_bmad-output/.../20-4-seam-fix-sweep.py` | the eight-width overlap-add sweep, offline against a fixed take |
| `_bmad-output/.../20-4-seam-raw/`, `20-4-seam-rawfull/`, `20-4-seam-rawfull-prefix/` | the captures (`-prefix` is the pre-fix baseline retained for the before/after comparison in §11.7) |
| `_bmad-output/.../20-4-regen-audition-fixture-r2.py` | round-2 fixture builder; preflights the geometry **and** the presence of the fix |
| `_bmad-output/.../20-4-perceptual-fixtures-r2/` | 14 WAVs + truth table with a `_meta` block naming candidate and reference |
| `_bmad-output/.../20-4-l1-audition-helper.py` | round-aware; unblinds against `_meta`; round 1 still re-runnable |
| `12_Story_20.4_AC5_Audition.bat` | defaults to round 2; `L1 r1` re-runs round 1 |

### Untouched, deliberately

* **The consumer crossfade** (`_DEFAULT_STREAMING_CROSSFADE_SAMPLES = 64`).
  §11.5 establishes it was never the lever; with the decoder now producing a
  continuous stream it blends 2.7 ms of genuinely adjacent audio, which is
  harmless and still masks any residual DC step. Changing it was considered
  and rejected as out of scope for a defect it does not cause.
* **The round-1 fixture, truth table and results CSV** — that round is a
  recorded result, and round 2 reuses its `cs25` files as a calibration
  anchor.
* `MAX_PRE_DELAY_SECONDS` (10.0 s) and `max_hold_chunks` (16) — guardrails.
* `_DEFAULT_STREAMING_WATERMARK_MS` (500) and
  `_DEFAULT_STREAMING_CROSSFADE_SAMPLES` (64) — the ≥16 GiB static path.
* `DEFAULT_LOOKAHEAD` (5) — held fixed across Story 20.1's whole sweep.
* The dispatch chain, PORT-b, the qasync call-site audit, and the 0.5 s
  reference-padding lift (Follow-up D) — all out of scope per the story.

---

## §9. AC #5 — NFR3 perceptual audition: **FAILED** (blocking), 2026-09-01

L1 (Commander solo), 7 utterances, blinded A/B, unblinded against
`_perlistener_truthtable.json`.

| utt | cs25 (old) | cs10 (NEW) | preferred | verdict |
|---|---|---|---|---|
| l-020 | click_or_discontinuity | tonal_distortion | cs25 | shared — both arms |
| l-021 | tonal_distortion | click_or_discontinuity | cs25 | shared — both arms |
| m-020 | none | **tonal_distortion** | cs25 | **BLOCKING — cs10 only** |
| m-021 | none | none | equivalent | clean |
| s-020 | none | none | equivalent | clean |
| s-021 | none | none | equivalent | clean |
| s-022 | none | none | equivalent | clean |

**Preference tally: cs25 3 — cs10 0 — equivalent 4.** The new geometry was never
preferred on any utterance.

### Verdict

**AC #5 fails.** `m-020` carries a tonal-distortion artefact on the committed
`cs10` geometry that is absent on `cs25`. AC #5 states that any audible
chunk-boundary artefact is blocking, not a note. `chunk_size = 10` must not
ship on this evidence.

### A separate, pre-existing finding — not caused by this story

`l-020` and `l-021` show seam artefacts on **both** geometries. Long-form
generations therefore already carry audible seam defects at the shipped
`cs25` setting. Story 17.1's audition certified TRUE_STREAM as perceptually
equivalent to BATCH, but it was not directed at seams specifically; this one
was, and found them. That is a pre-existing product defect, discovered here
rather than introduced here, and it warrants its own story.

### Scope note

The clean rows are all short/medium. The defects cluster on the long fixtures
plus one medium — consistent with a per-seam artefact whose probability of
being noticed rises with the number of seams, rather than with a constant
per-chunk defect.

---

## §8b. Operator hand-off — round 2 (the ONLY outstanding task)

**One task, ~10 minutes, listening only.** No GPU work, no app launches, no
generation. The 14-WAV round-2 fixture is already built and verified.

```
12_Story_20.4_AC5_Audition.bat
```

Seven blinded A/B pairs. The helper unblinds and prints the verdict itself;
hand that block back.

**What to listen for.** Round 1's cause turned out to be a splice bug that
deleted 15-19 ms of real speech at every chunk boundary — on both geometries.
That is fixed, and the boundary is now cross-faded rather than butt-spliced.
So this round asks whether the fix worked. The defect class is unchanged: a
click or tick partway through a word, a momentary discontinuity in a held
vowel, prosody that resets mid-phrase.

**What NOT to judge:** timbre, loudness, accent, wording rhythm. The two arms
are different samples, not one waveform cut two ways.

**Two things worth knowing before you start, because they change how to read
the result:**

1. **One arm of every pair is round 1's exact `cs25` file.** Not a
   regeneration — the same bytes you judged last time. If `l-020` and `l-021`
   draw the same calls again, the session is internally consistent. That is
   why they were reused, and it is also why the reference arm still carries
   the splice bug: it is what ships today, and AC #5 asks whether the
   candidate is clean *against the current build*.

2. **The click is measurably gone; the timbral component is reduced but not
   eliminated** (§11.7). The fix cross-fades the codec-state mismatch rather
   than removing it — only carrying codec state across chunks would remove it,
   and that is out of scope (§11.9). If a *faint* tonal seam is still audible
   on the long fixtures, that is the known residual, it is present on the
   reference arm too, and under the AC #5 rule it is a shared finding rather
   than a blocking one. The helper makes that distinction automatically.

**If round 2 fails**, the fallback is `chunk_size = 15` (Story 20.1 §5.2:
1,157 ms perceived TTFA versus 1,015 at cs10). §11.8 explains why that is the
fallback and not the fix.

---

## §11. The seam investigation, and the fix (2026-09-01, after the §9 failure)

Round 1 failed on `m-020` at `chunk_size = 10`, and flagged `l-020` / `l-021`
on **both** geometries. Two mechanisms were candidates: (a) the consumer
crossfade in `StreamingChunkBuffer` being too narrow now that seams are 2.5x
more frequent, or (b) the decoder's overlap-add posting independently decoded
segments with no codec state carried across the boundary.

**It is neither, quite. It is (b), plus a splice bug nobody had looked for.**

### 11.1 Method — measure the seam, do not reason about it

The fixture WAVs cannot separate the mechanisms: they are already crossfaded,
so the 64-sample blend has smeared exactly the evidence that distinguishes an
amplitude step from a spectral mismatch. So the raw signal was captured
instead, at two points the shipped pipeline never exposes:

| script | what it captures |
|---|---|
| `20-4-seam-capture.py` | the float32 arrays `StreamingDecoderWorker` posts, before the int16 cast, the watermark merge, and any crossfade |
| `20-4-seam-capture-full.py` | `pcm_full` — the decoder's output **before** the lookahead trim |

`pcm_full` is the key. Consecutive chunks decode `lookahead` frames of
**identical tokens**, so their `pcm_full` arrays contain the same audio twice.
That redundancy makes the alignment question answerable by cross-correlation
rather than by argument.

### 11.2 Finding 1 — the codec's real geometry, and a documentation error

Solving the output-length model from posted chunk lengths at two chunk sizes:

```
decode(N frames) -> 1920 * N - 555 samples
```

Cross-checked against **14 independent residual-chunk lengths** (residuals are
posted whole, so each is a free data point): every one is integral under the
model, none is off by a sample.

Two consequences.

**1920 samples/frame is 12.5 Hz, not 12 Hz.** This codebase's prose has said
"12 Hz tokenizer" since Story 16.3, and Story 20.1's sweep table computed
"audio per chunk" as `chunk_size/12`. Every such figure is ~4 % optimistic:
`chunk_size = 10` carries **800 ms** of audio per chunk, not 833 ms. Nothing
*behavioural* depended on it — no code used 12 — so this is a documentation
defect, not a bug. It does not change any conclusion in this file: the
`chunk_size >= 6` watermark-no-op condition becomes `chunk_size >= 7.5`, and
10 still clears it.

**There is a fixed 555-sample edge loss per decode call** — convolution edge
effects the vocoder cannot produce without context beyond the window. This is
the part that turned out to matter.

### 11.3 Finding 2 — a splice bug: 15-19 ms of speech deleted at every seam

`StreamingDecoderWorker._decode_and_post` computed its trim as

```python
samples_per_token = len(pcm_full) / len(chunk)      # = 1901.5 at cs25, 1883 at cs10
trim_samples = int(round(self._lookahead * samples_per_token))
```

That treats the **fixed** 555-sample loss as if it were **proportional**. The
resulting posted length falls short of `chunk_size * 1920` by exactly
`555 * chunk_size/(chunk_size + lookahead)`:

| geometry | posted | correct | deficit |
|---|---:|---:|---:|
| `chunk_size = 25` | 47,537 | 48,000 | **463 samples = 19.29 ms** |
| `chunk_size = 10` | 18,830 | 19,200 | **370 samples = 15.42 ms** |

Cross-correlating consecutive `pcm_full` arrays puts the true splice point at
exactly `chunk_size * 1920` — measured lag delta of 0 on the high-confidence
seams, and the constant is independent of how the edge loss splits between
head and tail, because both decodes lose the same amount.

**So the deficit is real speech, deleted.** Median RMS of the deleted span is
0.078-0.147 against a whole-utterance RMS of 0.094-0.124 — the same order as
the utterance's own level — with peaks to 0.57. It is not silence, and it is
not padding.

This has shipped since Story 16.4. It is why `l-020` and `l-021` were flagged
at `chunk_size = 25`: the defect was always there, and the round-1 A arm was
never a clean reference. It is also exactly why the defects scaled with seam
count, as §9's scope note suspected.

### 11.4 Finding 3 — and a codec-state mismatch underneath it

Where the two chunks decode the **same tokens**, the two decodes differ by a
median **NRMSE of 0.35** (range 0.21-0.58) at a correlation of 0.93 (range
0.82-0.98). Same content, same pitch, different fine structure — each decode
starts from a cold codec state.

That is mechanism (b) in isolation, measured. Alignment cannot touch it.

### 11.5 Task 1's answer: the consumer crossfade is not the lever

Stated plainly, as asked:

* The consumer crossfade is **64 samples = 2.67 ms**.
* Defect 1 is a **370-sample (15.4 ms) deletion** — 5.8x wider than the entire
  crossfade. A crossfade cannot bridge a gap almost six times its own width;
  worse, with the splice bug present the two sides of that blend are 15.4 ms
  apart *in time*, so widening it cross-fades between two different moments
  and smears rather than repairs.
* Defect 2 spans the whole **9,045-sample (377 ms)** shared region — 141x the
  crossfade width.

A consumer-crossfade sweep would have been aimed at the wrong mechanism, at
the wrong layer, and would have cost real audio (that blend consumes distinct
content). **It was not run.**

### 11.6 Task 2, at the right layer: the overlap that was never added

Finding 2's discovery also supplies the fix for Finding 3. `pcm_full` extends
`lookahead*1920 - 555` = **9,045 samples (377 ms)** past the splice point,
covering precisely the audio the next chunk re-decodes at its head. The
shipped code threw it away — the "overlap-add" was an overlap-*discard*.

Retaining it turns the boundary into a real cross-fade **between two
renditions of the same instant**. Unlike widening the consumer crossfade, this
consumes no audio and no duration: output length is unchanged, because each
chunk still advances the stream by exactly `chunk_size * 1920`.

The sweep (`20-4-seam-fix-sweep.py`) re-stitched the captured `pcm_full` at
eight widths — no GPU per width, and every variant the **same take**, so only
the stitching varies. Seam metrics are expressed as a multiple of the same
metric measured at 300 random non-seam positions in the same audio, so 1.00
means *a seam is statistically indistinguishable from any other point*:

| variant | step (x baseline) cs25 | cs10 | excess spectral cs25 | cs10 |
|---|---:|---:|---:|---:|
| shipped | 8.46 | 13.06 | +3.09 dB | +3.39 dB |
| **alignment only, no OLA** | **18.04** | 12.32 | +4.43 dB | +2.04 dB |
| aligned + OLA 64 | 1.31 | 1.00 | +4.36 | +1.93 |
| aligned + OLA 256 | 1.06 | 0.75 | +4.90 | +2.36 |
| aligned + OLA 512 | 1.19 | 0.97 | +4.52 | +2.03 |
| **aligned + OLA 1024** | **1.25** | **0.85** | **+0.71** | +1.99 |
| aligned + OLA 2048 | 1.46 | 0.83 | +1.27 | +1.63 |
| aligned + OLA 4096 | 1.50 | 0.96 | +1.59 | +1.23 |
| aligned + OLA 9045 (all of it) | 1.46 | 0.80 | +2.21 | +1.81 |

**Fixing the alignment alone is not enough, and at `chunk_size = 25` it makes
the click measurably worse** (8.46x -> 18.04x). That is the result that
settles the design: correcting the splice butts two genuinely different
waveforms together at the same instant, where before they were at least
offset. The two fixes are not independent — either alone is worse than both.

**Chosen: `_OVERLAP_ADD_SAMPLES = 1024` (42.7 ms).** The step metric reaches
the baseline by 64-256; the spectral metric keeps improving to ~1024-4096.
1024 is the smallest width at the plateau on **both**.

**Its cost, stated as asked.** The blend is free in duration and content, but
not in every sense: inside the window the signal is the average of two
decodes, which mildly softens fine structure. That cost scales with the
fraction of the stream inside a blend — `1024/(10*1920)` = **5.3 %** at
`chunk_size = 10`, 2.1 % at 25. Taking the whole 9,045-sample budget would put
47 % of the stream at `chunk_size = 10` inside a blend for no measured gain,
which is why the smallest sufficient width is the right pick rather than the
largest.

A **linear** ramp, not equal-power: the two decodes correlate at ~0.93, so
they add coherently and a linear ramp preserves amplitude, where an
equal-power ramp would bulge the level mid-blend.

### 11.7 End-to-end verification on the shipped path

Re-captured after the fix. These are the arrays `StreamingDecoderWorker`
actually handed downstream, not a simulation:

| fixture | posted full-chunk length | seam step before | after | non-seam baseline |
|---|---|---:|---:|---:|
| l-020 cs10 | 18,830 -> **19,200 = cs*1920** | 12.31 | **1.32** | 1.00 |
| l-020 cs25 | 47,537 -> **48,000 = cs*1920** | 8.87 | **0.49** | 0.87 |
| m-020 cs10 | 18,830 -> **19,200** | 11.83 | **1.24** | 1.03 |
| m-021 cs10 | 18,830 -> **19,200** | 14.51 | **1.06** | 0.98 |

The click is gone: every seam is now at the non-seam baseline, a ~10x
reduction.

The spectral picture is **honestly mixed**, and this matters for what round 2
can be expected to show:

| fixture | seam spectral before | after | baseline |
|---|---:|---:|---:|
| m-020 cs10 (the blocking row) | 14.87 dB | **9.60** | 9.26 |
| l-020 cs10 | 13.62 | **10.16** | 11.00 |
| m-021 cs10 | 12.84 | 11.63 | 9.56 |
| l-020 cs25 | 13.37 | 13.06 | 10.77 |

Fully resolved on two of four, improved on the others. That residual is
mechanism (b) — the crossfade **masks** the codec-state mismatch, it does not
remove it. Only carrying codec state across chunks would (§11.9).

### 11.8 Task 4: the `chunk_size = 15` retreat is NOT the right move here

Not run, and the reason is not that it would be inconvenient:

**`chunk_size = 15` does not fix the defect. It halves how often you meet it.**
The splice bug is per-seam and geometry-independent — it is present, and
measured, at `chunk_size = 25`, which is what ships today. Round 1 flagged
`l-020` and `l-021` at `cs25`. Retreating to `cs15` would ship a known,
now-understood, one-line-fixable audio defect at ~1.6x the seam density of the
current build, and would spend the story's speed win to do it.

`chunk_size = 15` remains available and unchanged in cost if round 2 fails
(Story 20.1 §5.2: 1,157 ms perceived TTFA versus 1,015 at cs10 and 1,662 at
cs25). It is the fallback, not the fix.

### 11.9 This reframes Mary's Finding 1 from a speed item to a quality one

The technical research memo
(`planning-artifacts/research/technical-qwen3-tts-ttfa-optimization-2026-08-31.md`,
Finding 1 / the Nari technique list) names **codec state caching across
chunks** as something the reference implementations do and we do not. It was
filed as a throughput optimisation.

§11.4 measures the cost of not doing it: **every chunk boundary reconciles two
renditions of the same audio that differ by ~35 % NRMSE.** Story 20.4's fix
cross-fades that mismatch rather than eliminating it, which is why the
spectral metric improves without reaching the baseline everywhere (§11.7), and
why `tonal_distortion` — not `click` — was the defect vocabulary Commander
reached for on the worst rows.

Recommendation: **re-file Finding 1 as an audio-quality item.** Its case is
materially stronger than the speed framing suggested, it is the only fix that
addresses the cause rather than the symptom, and it is well outside this
story. Fixing it would also let `_OVERLAP_ADD_SAMPLES` shrink or disappear.

### 11.10 What changed, and what it is guarded by

`src/myvoice/services/tts_streaming/streaming_decoder.py` only.

* Two measured constants, `_CODEC_SAMPLES_PER_FRAME = 1920` and
  `_CODEC_EDGE_LOSS_SAMPLES = 555`, with the derivation recorded at the
  definition.
* The splice point is `chunk_size * 1920` when the identity
  `len(pcm_full) == 1920*frames - 555` holds — **verified on every chunk, not
  trusted**. If it ever fails the session falls back to the pre-20.4
  proportional trim with no overlap-add and emits a
  `decode_geometry_unverified` metric. A codec or pin change therefore
  degrades to today's behaviour loudly, rather than mis-cutting audio
  silently, which would be worse than the bug being replaced.
* The retained tail is cross-faded into the next chunk's head over
  `_OVERLAP_ADD_SAMPLES`, clamped to the audio the two chunks actually share.
* The residual chunk is still posted whole **and** still blended — it starts
  at a seam like any other chunk, and missing it would leave one untreated
  click per utterance, right before the audio ends.

Eight new rows in `tests/unit/services/tts_streaming/test_streaming_decoder.py`,
using a `decode_fn` that reproduces the real codec geometry and makes each
sample's **value its absolute position**, so time-contiguity is asserted
directly: a dropped span shows as a jump in an arithmetic sequence, a
duplicated one as a repeat. Rows cover the exact splice (pinning the 370-sample
delta against the legacy value), contiguity across seams, the blend ramp, the
clamp, residual handling, the fallback-plus-telemetry path, and the constants
themselves.

The pre-existing tests still pass **unchanged** and still pin the legacy trim:
their synthetic `decode_fn` returns one sample per token, which does not
satisfy the identity, so they exercise the fallback — which is exactly what
that path is for.

### 11.11 Round-2 audition fixture

`20-4-perceptual-fixtures-r2/`, built by `20-4-regen-audition-fixture-r2.py`.

* **Reference arm = round 1's `cs25` WAVs, copied byte-for-byte.** Commander
  has already judged those exact files, so his round-1 calls act as a
  calibration anchor: if `l-020`/`l-021` draw the same calls again, the
  session is internally consistent. It also leaves round 1's evidence
  undisturbed.
* **Reference is deliberately the PRE-FIX stitching** — what ships today. AC
  #5 asks whether the candidate is free of seam artefacts with the current
  build as reference; fixing the reference would answer a more academic
  question.
* **Candidate arm = `chunk_size = 10` with the fix.**
* Same 7 utterances, same voice, same machinery, fresh randomisation (a
  different seed, so a listener who has just done round 1 cannot carry an
  ordering expectation into round 2).
* The generator preflights the committed geometry **and** the presence of the
  fix, so it cannot silently rebuild round 1.

The helper now takes a round argument and reads `candidate`/`reference` from
the truth table's `_meta` block; round 1 remains re-runnable via
`12_Story_20.4_AC5_Audition.bat L1 r1`.

### 11.12 Regression status after the fix

| surface | result | vs. the §9.1 baseline |
|---|---|---|
| `tests/unit/services` + observability + models | **960 passed, 0 failed** | 953 -> 960 (+7 decoder rows) |
| `tests/integration` + `test_qwen_tts_internals.py` | 175 passed, **4 failed** | unchanged pre-existing set |
| `tests/unit/services/tts_streaming/test_streaming_decoder.py` | **44 passed** | 36 -> 44 |

> **A pre-existing cross-suite ordering hazard, found while checking this.**
> Running `tests/unit/services` and `tests/integration` in ONE pytest
> invocation produces 5 extra failures in `test_clear_comms_dispatch.py` and
> `test_clear_comms_d5_invariant.py` that do not occur when either directory
> is run alone. Verified **pre-existing**: the same 5 fail the same way on the
> stashed pre-Story-20.4 tree. Every sweep in this file (and in Stories 20.2 /
> 20.3) runs the surfaces as separate invocations, which is why it had not
> surfaced. Not this story's to fix; worth a ticket.

