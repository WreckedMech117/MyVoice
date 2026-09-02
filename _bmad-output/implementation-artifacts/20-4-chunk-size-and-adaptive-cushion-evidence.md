# Story 20.4 — Chunk-Size Retune + Adaptive-Cushion Fix: evidence

Phase ⊥-Polish-3. Fourth story of Epic 20 (First-Audio Latency).
Follow-ups **B** and **C** from Story 20.1 §6.4, shipped together because
Story 20.1 found them coupled.

## 0. Headline

> **Read §15 first.** This story does not deliver what it was drafted to
> deliver. Follow-up B — the `chunk_size` retune this story exists for — is
> **closed unsuccessful**: it was measured, shipped as far as the NFR3
> perceptual gate, failed it twice, and was reverted. What ships instead is
> a fix for an audio defect that has been in the product since Story 16.4,
> found only because the retune forced someone to look at the seams.

### What actually ships

| | before | after | source |
|---|---|---|---|
| **streamer geometry** | `chunk_size=25`, `lookahead=5` | **unchanged — `25` / `5`** (retune attempted, reverted) | §15.1 |
| **chunk-boundary stitching** | 15–19 ms of real speech **deleted at every seam**, since Story 16.4 | exact splice + decoder-side overlap-add | §11, §14 |
| compile-path `decode_window_frames` | pinned at **30** regardless of the streamer, at **three** sites | derived from `codec_token_streamer` at **all three** | §1.1 |
| sub-16 GiB first audio, `P = 0.5` (derived) | **12.00 s** — released by the `MAX_PRE_DELAY` guardrail | **4.00 s** | §3, §15.4 |
| sub-16 GiB cushion / talker ratio (derived) | 2.50× | **0.83×** | §3, §15.4 |
| `T_a` estimator error, long fixture | +44.5 % | **+2.0 %** | §2.3 |
| `MAX_PRE_DELAY_SECONDS` | 10.0 s | **10.0 s — unchanged** | §2.2 |
| **shipped first-audio TTFA** | 1,353 ms (Story 20.3) | **1,353 ms — unchanged** | §15.5 |

### AC verdicts

| AC | what it asks | verdict |
|---|---|---|
| #1 | retune + thread the geometry where it is read | **PARTIALLY MET** — the threading landed and is what made the revert a one-line edit; the retune it existed to enable is reverted. §15.2 |
| #2 | cushion policy + the `T_a` overshoot | **MET**, restated at `cs25` — §2, §15.4 |
| #3 | coupling re-derived | **MET**, re-derived at the shipped `cs25` — §3, §15.4 |
| #4 | OFR-E producer ratio re-measured | **MET** — gate passes at both geometries; the headline figure belongs to `cs10`, which is not shipped. §15.5 |
| #5 | NFR3 perceptual audition | **PASS on round 3, for the seam fix only.** Rounds 1 and 2 failed and are why the geometry reverted. §9, §12, §14. **Round 4 (`cs15`) is open** — §16 |
| #6 | GUI capture on the reachable tier | **MET** — measured; the 976 ms figure is the `cs10` result, not the shipped one. §15.5 |
| #7 | no regressions | **MET** — zero new failures across every surface, re-run after the revert. §15.6 |

> **Reading order.** This file is an append log, so section numbers are
> chronological rather than sorted, and several early sections describe a
> configuration that was later reverted. **§15 is the reconciliation and the
> definitive statement of what ships.** In order of events: §0–§10 the
> implementation pass; **§9 (second one)** round-1 audition FAILED; **§11**
> the seam investigation and fix; **§12** round-2 audition FAILED worse;
> **§13** the click mechanism plus a metric that failed its own validation;
> **§14** round-3 isolating audition PASSED; **§15** the outcome map
> executed. Overtaken sections carry a banner pointing forward.

**One task outstanding: the round-4 audition (§16).** Commander asked for the
`chunk_size = 15` question to be settled rather than left open. The fixture is
built and a prediction is recorded; §16.7 is the hand-off. Everything else in
this story is complete — three auditions and one GUI capture already run, and
the shipped configuration (`cs25` + the seam fix) is unaffected by round 4's
outcome. Every decoder chunk boundary was deleting 15-19 ms of
real speech — a splice-alignment bug present at the shipped `chunk_size = 25`
too, which is why round 1 flagged long-form seams on *both* arms. Underneath
it, the two independent decodes either side of a boundary differ by ~35 %
NRMSE. Both are now fixed; seam discontinuity measures **12.3x -> 1.3x** the
non-seam baseline, i.e. a seam is now statistically indistinguishable from any
other point in the audio. **§11 is the investigation; §8 is the one remaining
operator task.**

---

## 1. AC #1 — the chunk geometry, and where it is actually read

> **PARTIALLY SUPERSEDED (§15.2).** The threading described here shipped
> and proved itself on the revert. The `chunk_size = 10` retune it was
> built to enable did NOT ship — see §15.1.


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

> **SUPERSEDED (§15.4).** Re-derived at the shipped `chunk_size = 25` and
> at the codec's measured 12.5 Hz. The tables below describe the reverted
> geometry and are kept as the record of the coupling analysis.


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

> **RELABELLED (§15.5).** The headline figures here are the `chunk_size =
> 10` result, which is NOT shipped. The gate passes at both geometries;
> the shipped ratio is 0.585× and the shipped TTFA is Story 20.3's
> 1,353 ms.


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

> **RELABELLED (§15.5).** The 976 ms / 1,065 ms figures are the
> `chunk_size = 10` result, which is NOT shipped. Not re-measured, and no
> further capture requested: the seam fix does not change first-audio
> timing, so the shipped figure remains Story 20.3's 1,353 ms.


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
| `src/myvoice/services/tts_streaming/streaming_decoder.py` | **§11 fix, SHIPS** — exact splice from the codec's measured geometry, verified per chunk with a loud fallback; decoder-side overlap-add over the previously discarded tail |
| `src/myvoice/services/tts_streaming/codec_token_streamer.py` | `DEFAULT_CHUNK_SIZE` **unchanged at 25** — retuned to 10, then reverted (§15.1). The sweep evidence, the perceptual reason for the revert, and the `cs15` open question are recorded at the constants |
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
| `12_Story_20.4_AC5_Audition.bat` | defaults to round 3; `L1 r1` / `L1 r2` re-run the earlier rounds |

### Added by the §13 mechanism analysis and the §14 round-3 audition

| file | purpose |
|---|---|
| `_bmad-output/.../20-4-click-mechanism.py` | the LPC click detector validated against the listener (and failing), plus the cold-start and timing-jitter analyses |
| `_bmad-output/.../20-4-regen-audition-fixture-r3.py` | round-3 isolating fixture builder; preflights that the committed geometry is untouched |
| `_bmad-output/.../20-4-perceptual-fixtures-r3/` | 14 WAVs + truth table; both arms `cs25`, stitching the only variable |
| `_bmad-output/.../20-4-chunk-retune-audition-r2.csv`, `-r3.csv` | rounds 2 and 3 results, preserved alongside round 1's |

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

## §8b. Operator hand-off — round 2 (COMPLETE; round 2 ran and FAILED, see §12)

> **Nothing is outstanding for the operator.** Rounds 2 and 3 both ran.
> The story is complete — see §15.

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


---

## §12. AC #5 round 2 — **FAILED, and worse than round 1**, 2026-09-01

Candidate = `cs10` + the §11 seam fix. Reference = round 1's exact `cs25` WAVs
(pre-fix, byte-for-byte).

| utt | cs25 ref | cs10+fix candidate | preferred | verdict |
|---|---|---|---|---|
| l-020 | click | click | cs25 | shared |
| l-021 | none | **click** | cs25 | **BLOCKING** |
| m-020 | none | **click** | cs25 | **BLOCKING** |
| m-021 | none | none | equivalent | clean |
| s-020 | none | none | equivalent | clean |
| s-021 | none | none | equivalent | clean |
| s-022 | none | **click** | cs25 | **BLOCKING** |

**Preference: cs25 4 — cs10+fix 0 — equivalent 3.** Blocking rows went 1 → 3.

### The listener data is trustworthy

The reference arm was the identical round-1 files, so it doubles as a
consistency check: **6 of 7 match** across rounds. Only `l-021` differs
(`tonal_distortion` → `none`). The result is not listener noise.

### Two things got worse, and one changed character

- `s-022` — a **short** fixture, clean on both arms in round 1 — is now
  blocking. The seam work introduced a defect where none existed.
- The candidate's defect class changed from `tonal_distortion` (round 1) to
  `click_or_discontinuity` (round 2). The fix traded a spectral artefact for an
  amplitude one, and made it *more* frequent.

### The offline metric and the ear disagree

§11 measured seam step falling to 0.85× the non-seam baseline at `cs10` —
"statistically indistinguishable from any other point". The listener hears
clicks on 4 of 7. **The seam-step metric does not capture what is audible**, so
it cannot be used to clear this gate on its own.

### The experiment conflated two variables — a design error in the round-2 setup

The candidate arm changed **both** `chunk_size` (25 → 10) **and** the stitching
(shipped trim → aligned splice + 1024-sample overlap-add), against a reference
that had neither. The result therefore cannot attribute the clicks to either
change. That is a flaw in how the round was specified, not in how it was run.

**This matters beyond the retune**: the seam fix is geometry-independent and
alters the *shipping* `cs25` path too. On this evidence we cannot say whether it
improves or harms what users hear today.

### Next experiment — isolate

Audition **`cs25` + seam fix** against **`cs25` shipped**. One variable.

- candidate clean or better ⇒ the fix is good and `chunk_size = 10` is the
  problem ⇒ keep the fix, retreat the geometry
- candidate shows clicks ⇒ the fix itself is harmful at any geometry ⇒ revert
  it, and the 19.3 ms deletion needs a different remedy

Neither `DEFAULT_CHUNK_SIZE = 10` nor the seam fix should reach a release until
that separation is made.

---

## §13. Round-3 prep — the mechanism, and a metric that failed its own test

> **§13.5's prediction was FALSIFIED by §14, in the good direction.** The
> mechanism below is real but is not the dominant term at `cs25`. See
> §15.1 for the reconciliation.


Two questions were put after the §12 failure: **is there a plausible mechanism
by which the 1024-sample overlap-add itself produces an audible click**, and
**can an isolating experiment attribute it**. Both are answered here from
analysis. No fix was changed in this pass, and no width was touched.

### 13.1 First, the uncomfortable one: my metrics do not track the ear

§11 cleared the fix on a seam-step metric that fell to 0.85× the non-seam
baseline. The listener then heard clicks on 4 of 7. Before proposing any
mechanism, I built a second, better-motivated detector and checked whether it
could have predicted the audition — an **LPC prediction-error spike**, fitted
on clean audio *before* each seam and applied forward, which is the standard
de-clicking arrangement and is far closer to what an ear flags than a
single-sample amplitude step.

It was run on the **exact WAVs the listener judged**, at exactly known seam
positions, across all three conditions (`cs25` pre-fix, `cs10` pre-fix,
`cs10`+fix) — 21 files, 8 flagged by the listener and 13 clean.

```
  FLAGGED by listener (n=8):  6628.1, 1021.3, 541.2, 220.1, 105.2, 95.5, 60.0, 22.0
  CLEAN per listener  (n=13): 5081.4, 327.1, 246.2, 65.8, 63.5, 55.9, 40.6, ...

  lowest FLAGGED = 22.0   highest CLEAN = 5081.4
  separable by one threshold: NO
```

**It fails, and it fails badly** — the single highest score in the whole set
(`s-022` at `cs10` pre-fix, 5081) is a file the listener called *clean*, while
files scoring 22–105 were flagged.

The reason is almost certainly **auditory masking**, which neither metric
models: `s-022` is "Bit, bat, bot, but, bet", and a broadband spike that lands
inside a plosive burst is inaudible, while a much smaller one in a sustained
vowel is not.

**Conclusion, and it is a constraint on everything below: no offline seam
metric available here can clear this gate.** Two independent metrics, on the
same 21 files, both fail to reproduce the listener ranking. Analysis is
therefore used in this section for **mechanism only** — to explain and to
predict *direction* — and every number below is offered as explanation, never
as evidence of audibility. The coordinator's instruction not to tune against
the metric again is adopted as a standing rule for this story.

> A methodological caveat on this subsection, recorded rather than buried:
> the detector flagged 4 of 21 files (`!GEOM`) where the residual length did
> not satisfy the codec identity, meaning my seam-position model for those
> files was wrong and some scores were measured away from a real seam. That
> weakens the detector further — it does not rescue it.

### 13.2 The answer: yes, and the mechanism is specific

**The overlap-add blends *into* the worst-decoded audio in the stream, over
exactly the window where it is worst.**

Measuring the RMS difference between the two decodes of the *same* audio, as a
function of position into the next chunk's decode, normalised by local RMS:

| fixture | 0–128 | 128–256 | 256–512 | 512–1024 | 1024–2048 | 2048–4096 | 4096–8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| l-020 cs10 | **0.824** | 0.668 | 0.581 | 0.495 | 0.355 | 0.367 | 0.279 |
| l-020 cs25 | **0.711** | 0.525 | 0.489 | 0.393 | 0.306 | 0.247 | 0.236 |
| m-020 cs25 | **0.539** | 0.490 | 0.484 | 0.401 | 0.368 | 0.356 | 0.248 |
| m-021 cs10 | **1.163** | 1.402 | 1.244 | 1.172 | 0.639 | 0.601 | 0.171 |

The next chunk's decode is **worst at its very head** — error of 0.5 to 1.4×
the local signal RMS in the first 128 samples — and improves monotonically
with position. That is the same phenomenon as the 555-sample edge loss: the
vocoder has no left context, so its first output samples are the least
supported.

**The 1024-sample blend covers the first four bands — the worst four.** The
ramp gives that copy monotonically increasing weight, reaching 100 % exactly
where it is still poor, and the stream then continues on it.

And the previous chunk's audio, which is *well* supported there, remains
available for **9,045 samples** past the splice — far past where the next
chunk settles (~2048–4096). The fix discards good audio in order to fade into
bad audio, and it does so at every seam.

### 13.3 A correction to §11.4 that made this worse than I reported

§11 justified the blend on a measured correlation of **0.93** between the two
decodes. That figure was computed over a 12,000-sample window — dominated by
the settled region, not the region the blend actually mixes.

Re-measured **on the blend region only** (the first 1024 samples), at
per-seam resolution:

| fixture | seams | best lag (median / min / max) | correlation (median / min) |
|---|---:|---|---|
| l-020 cs10 | 21 | 2 / −34 / +22 | 0.875 / **0.131** |
| l-020 cs25 | 8 | 2 / 0 / +24 | 0.691 / **0.136** |
| m-020 cs10 | 4 | −2 / −8 / +12 | 0.614 / 0.322 |
| m-021 cs10 | 4 | 5 / −35 / +21 | 0.550 / **0.111** |

**Median correlation in the blend region is 0.55–0.88, not 0.93, and the worst
seams fall to 0.11.** There is also real timing jitter — best lag wanders up to
±35 samples (±1.5 ms), which at speech formant frequencies is a substantial
phase error.

So the premise §11 relied on — "both sides of the blend are the same moment in
time", justifying a linear ramp as amplitude-preserving — **is not true in the
region that matters**. Cross-fading two signals correlated at 0.11–0.55 with
±1.5 ms of relative jitter is a recipe for comb filtering and for rendering a
transient twice at partial amplitude. On a plosive-dense short fixture with
few seams but sharp attacks, that is exactly a click — which is what `s-022`
became.

### 13.4 Why the defect class changed, which the mechanism has to explain

| | pre-fix | with the fix |
|---|---|---|
| what happens at a seam | one butt splice: a localised discontinuity, plus 15–19 ms of speech deleted | 42.7 ms of the stream replaced by a time-varying mix of a good copy and a poor, partly-decorrelated, slightly time-shifted copy |
| defect character | localised, spectral — `tonal_distortion` | distributed, broadband, transient-like — `click_or_discontinuity` |
| scaling | per seam | per seam, but each event is 42.7 ms wide instead of instantaneous |

This is the shape of the round-1 → round-2 change: the fix **traded a
localised defect for a distributed one and made each event much longer**,
which is consistent with more rows flagged and with `s-022` — clean on both
arms in round 1 — becoming blocking.

### 13.5 What this predicts for round 3, stated before the audition

On this mechanism, the harm is **geometry-independent**: it is a property of
the blend, and `cs25` has the same blend at every seam, merely 2.5× fewer of
them. So the prediction is that **the round-3 candidate will also carry
clicks, fewer of them than at `cs10`** — which by the pre-agreed outcome map
means the fix is harmful at any geometry and should be reverted.

It is recorded here **before** the audition so the result is a genuine test of
it rather than something rationalised afterwards. If the candidate comes back
clean, this mechanism is wrong and §13.2/§13.3 need re-examining.

### 13.6 If it is reverted, the 19.3 ms deletion still needs a remedy

Reverting restores a real, measured defect: 15–19 ms of speech deleted at
every seam (§11.3), present in the shipping build. The mechanism above also
points at what a better remedy looks like, and it is **not** a different
crossfade width:

* **Prefer the previous chunk's audio wherever both exist.** It is decoded
  with full left context and stays good for 9,045 samples past the splice;
  the next chunk's head is the worst audio in the stream. Splicing at
  `chunk_size*1920` with *no* blend already fixes the deletion — §11.6 showed
  that alone worsens the step metric, but the step metric has now failed its
  own validation, so that result carries no weight and the option is open
  again.
* **If a blend is used at all, place it after the next chunk has settled**
  (~2048–4096 samples in), not at the splice — using the previous chunk's
  good audio to cover the cold-start region entirely. This is a different
  design, not a width, and it is what a width sweep would never have found.
* **Carrying codec state across chunks (§11.9)** removes the cause of both
  the edge loss and the decode divergence, and would make the blend
  unnecessary. Still out of scope, and its case is now stronger again.

None of these is implemented in this pass.

### 13.7 Round-3 fixture

`20-4-perceptual-fixtures-r3/`, built by `20-4-regen-audition-fixture-r3.py`.

| | |
|---|---|
| reference | `cs25`, shipped pre-fix stitching — **round 1's exact files**, third audition, still the calibration anchor |
| candidate | `cs25`, **with** the seam fix |
| isolates | the stitching only; both arms are `chunk_size = 25` |
| randomisation | fresh seed (20040903) |

`DEFAULT_CHUNK_SIZE` and the seam fix are **unchanged**. The candidate's
geometry is set in-process via the harness rebinder — the same mechanism
Story 20.1 used for its sweep — so there is no source-tree edit to revert, and
the generator's preflight asserts the committed constants are still `(10, 5)`
before it will run.

Rounds 1 and 2 are untouched. The helper takes `r1` / `r2` / `r3` and defaults
to `r3`; `12_Story_20.4_AC5_Audition.bat` follows.

**Pre-agreed outcome map** (from the coordinator, recorded so the audition
decides the story rather than being interpreted after the fact):

* **candidate clean or better** → the fix is good and `chunk_size = 10` is the
  problem → keep the fix, retreat the geometry (`cs15` or `cs25`).
* **candidate shows clicks** → the fix is harmful at any geometry → revert it,
  and the deletion needs one of the §13.6 remedies.


---

## §14. AC #5 round 3 — **PASS.** The fix is good; the geometry was the problem. 2026-09-01

One variable: both arms `cs25`, differing only in stitching.

| utt | cs25 shipped | cs25 + fix | preferred | verdict |
|---|---|---|---|---|
| l-020 | click_or_discontinuity | **none** | cs25+fix | **fix is CLEANER** |
| l-021 | click_or_discontinuity | **none** | cs25+fix | **fix is CLEANER** |
| m-020 | none | none | equivalent | clean |
| m-021 | none | none | equivalent | clean |
| s-020 | none | none | equivalent | clean |
| s-021 | none | none | equivalent | clean |
| s-022 | none | none | equivalent | clean |

**Preference: cs25+fix 2 — cs25 shipped 0 — equivalent 5. Zero blocking rows.
The fix was never worse on any utterance, and removed the defect on both long
fixtures.**

Consistency against the identical round-1 `cs25` files: 6 of 7 exact matches;
`l-021` differs only in *category* (`tonal_distortion` → `click_or_discontinuity`),
with "defect present" agreeing in both rounds.

### The recorded prediction was falsified, and that is the point

§13.5 predicted, before the audition, that the harm was geometry-independent and
the round-3 candidate would also show clicks ⇒ revert. **It did not.** At `cs25`
the fix is a net improvement.

The mechanism in §13.2/§13.3 is not wrong, but it is **not the dominant term at
`cs25`**. Both effects scale with seam count: at `cs25` removing the 19.3 ms
deletion dominates the blend's harm; at `cs10`, with 2.5× the seams, the blend
harm overtakes it. That reconciles all three rounds without discarding any of
them.

### Outcome, per the map set before round 3

*"candidate clean or better ⇒ the fix is good and `chunk_size = 10` is the
problem ⇒ keep the fix, retreat the geometry."*

- **KEEP** the seam fix. It repairs a real audio defect that has shipped since
  Story 16.4 — audible clicks on long-form streamed generations — and it is now
  the only change in this story with direct perceptual evidence *for* it.
- **REVERT** `DEFAULT_CHUNK_SIZE` to 25. Follow-up B does not ship.
- **KEEP** the adaptive-cushion work (C) and the D-25 geometry threading. C is
  independent of geometry; the coupling was only that B made C worse. The
  threading is what makes any future retune safe.

### What this costs and what it buys

Costs the `1,353 → 976 ms` increment; Follow-up B is closed unsuccessful.
Buys a fix to a shipped audio defect, on evidence, plus a chunk-size knob that
is now safe to turn.

`chunk_size = 15` remains untested perceptually: 1.5× the seams of `cs25`
against `cs10`'s 2.5×, and Story 20.1 measured 1,157 ms perceived TTFA there.
Genuinely uncertain — it would need its own audition round.

---

## §15. The outcome map, executed — what this story actually delivers

Round 3's pass triggered the outcome branch agreed before it ran: **keep the
fix, revert the geometry.** This section is the reconciliation, and it is the
authoritative statement of what ships. Where an earlier section describes
`chunk_size = 10`, it is a record of an attempt, not a description of the
product.

### 15.1 Follow-up B — CLOSED UNSUCCESSFUL

`DEFAULT_CHUNK_SIZE` is back to **25**. Follow-up B does not ship.

**Why, in one line:** the latency sweep said 10 and the ear said 25, and AC #5
makes the ear the gate.

The full reason, because "it failed an audition" understates it:

* Seam artefacts scale with **seam count**, and `chunk_size = 10` has **2.5×**
  the seams of 25. Commander flagged defects on 1 of 7 fixtures at `cs10`
  pre-fix (§9) and 3 of 7 at `cs10` with the seam fix (§12), and never once
  preferred `cs10` on any utterance in either round.
* Round 3 (§14) then isolated the variables and showed the **seam fix is good
  on its own at `cs25`** — cleaner on both long fixtures, never worse. So the
  two things this story built are separable, and the geometry is the half that
  fails.

**What the reconciliation looks like** — and it accounts for all three rounds
without discarding any:

| | alignment gain (removing the 15–19 ms deletion) | blend harm (§13.2/§13.3) | net |
|---|---|---|---|
| `cs25` | per seam | per seam | **gain wins** — round 3 PASS |
| `cs10` | per seam, 2.5× as many | per seam, 2.5× as many, and each event is 42.7 ms wide | **harm wins** — round 2 FAIL |

Both terms scale with seam count; they simply cross over somewhere between the
two geometries.

> **§13.5's prediction was falsified, in the good direction.** It predicted the
> round-3 candidate would also show clicks because the blend harm is
> geometry-independent. It was wrong: the harm is geometry-independent *per
> seam*, but so is the gain, and at `cs25` the gain dominates. Recording that
> prediction **before** the audition is what made the result readable rather
> than something to argue about afterwards — it turned a disappointing outcome
> into a clean discriminating test. Worth repeating.

**Open question, explicitly not pursued.** `chunk_size = 15` is perceptually
untested — 1.5× the seams of 25 against `cs10`'s 2.5×, and Story 20.1 measured
1,157 ms TTFA there versus 1,015 at `cs10` and 1,662 at `cs25`. Now that both
the harm and the gain are known to scale with seam count, the balance at
intermediate geometries is **genuinely uncertain** and cannot be predicted from
either the latency sweep or the seam analysis alone. It needs its own story
with its own NFR3 audition. Recorded as an open question, not a recommendation.

### 15.2 AC #1 — PARTIALLY MET, and stated as such

The AC asked for two things. One landed and is valuable; the other is reverted.

| half | verdict |
|---|---|
| retune `DEFAULT_CHUNK_SIZE` to 10 | **REVERTED** (§15.1) |
| thread the real geometry to where it is read, so the two cannot drift | **DONE, and it proved itself** (§1) |

The threading is not consolation. It was the AC's own stated purpose — *"so
the two cannot drift again"* — and the revert is the test it was built for:

* `DEFAULT_CHUNK_SIZE = 10 → 25` was a **one-line edit**. Everything followed:
  `engage_compile_optimizations` resolved the window back to 30 from the live
  module constants, `model_registry` passed the reverted values through, and
  `warmup_compile_async`'s cache key moved with it.
* **All 6 rows of `test_decode_window_geometry_coherence.py` passed unchanged
  across the revert**, as did the 3 derived rows in
  `test_streaming_tts_smoke.py`. Tests written against derived geometry
  survived a geometry change in the opposite direction from the one they were
  written for, which is the strongest evidence available that the drift is
  actually closed.
* Verified after the revert: no literal `10` or `15` survives at any
  geometry-bearing site, and
  `test_no_source_file_passes_a_literal_decode_window_frames` enforces it
  across all of `src/myvoice`.

Had the threading not been done, this revert would have been a hunt through
three files for stale literals — including the `warmup_compile_async` cache-key
site (§1.1) that was found only because AC #1 forced the search.

### 15.3 What ships from the seam work — and what it costs

`streaming_decoder.py` keeps the exact splice and the decoder-side
overlap-add. This is the substantive deliverable, and it is **not** what the
story was written to produce:

* It repairs **15–19 ms of real speech deleted at every chunk boundary**
  (§11.3), a defect present since Story 16.4 in every TRUE_STREAM generation
  every user has ever heard.
* Round 3 is **direct perceptual evidence for it at the shipped geometry**:
  cleaner on both long fixtures, never worse on any, preferred 2–0 (§14).
* It is guarded: the codec output-length identity is verified per chunk, with
  a loud fallback to the pre-20.4 trim and a `decode_geometry_unverified`
  metric if a codec or pin change ever breaks it.

**The known residual, stated rather than buried.** §13.2/§13.3 established
that the blend fades into the next chunk's cold-start region — measured decode
error 0.5–1.4× local RMS in its first 128 samples — and that in the blend
region the two copies correlate at 0.55 median (min 0.11), not the 0.93 the
design was originally justified on. At `cs25` the alignment gain dominates
that harm and the net is positive, which round 3 confirms perceptually. **It
is still there.** The better-shaped remedies are recorded in §13.6 and remain
unimplemented; the strongest of them is codec state caching (§11.9), which
would remove the cause rather than mask it.

### 15.4 AC #2 and AC #3 — MET, restated and re-derived at `cs25`

The cushion work is geometry-independent. Story 20.1's coupling was only that
Follow-up B made the cushion *worse*; with B reverted, C stands on its own.

`20-4-adaptive-cushion-sim.py` was re-run at the shipped geometry (and at the
codec's **measured 12.5 Hz**, not the 12 Hz the earlier tables assumed):

| `P` | segment 4 before | released by | segment 4 after | released by | ratio before | ratio after |
|---:|---:|---|---:|---|---:|---:|
| 0.50 | **12.00 s** | 10 s cap | **4.00 s** | unreachable → watermark | 2.50× | **0.83×** |
| 0.60 | 10.00 s | 10 s cap | 3.33 s | unreachable | 2.50× | 0.83× |
| 0.70 | 11.43 s | 10 s cap | 2.86 s | unreachable | 3.33× | 0.83× |
| 0.75 | 10.67 s | 10 s cap | 2.67 s | unreachable | 3.33× | 0.83× |
| 0.80 | 7.50 s | τ_min | 2.50 s | unreachable | 2.50× | 0.83× |
| 0.85 | 4.71 s | τ_min | 2.35 s | unreachable | 1.67× | 0.83× |
| 0.90 | 2.22 s | τ_min | 2.22 s | unreachable | 0.83× | 0.83× |
| 0.95 | 2.11 s | τ_min | 2.11 s | **feasible** | 0.83× | 0.83× |

The guardrail is never the binding escape anywhere on the sweep, and the
cushion-to-talker ratio is a flat 0.83× — which is `chunk_size / (chunk_size +
lookahead)` = 25/30, because release lands on one chunk arrival.

The legacy-policy reproduction still cross-checks against **all 10** of Story
20.1 §2.7's published numbers (run at that story's 12.0 Hz convention so the
comparison is like-for-like; the `P = 0.50` row reads 12.00 s in the table
above and 12.50 s in Story 20.1 purely because of that 4 % frame-rate
correction, not because anything drifted).

**One consequence of `cs25` worth stating.** A posted chunk carries **2.0 s**
of audio, which already equals the entire 2.0 s cushion budget. So on the
shipped geometry the feasible and unreachable regimes release at the *same*
point — chunk 2, the first push where a producer rate is measurable at all —
and the policy's whole effect there is to stop the guardrail binding. The
regime distinction still matters for any future smaller geometry, and both
behaviours are still tested; it simply is not a mid-stream decision at 25.
`test_at_the_shipped_geometry_the_feasible_branch_is_granularity_bound` pins
this so §2's regime table is not misread.

`T_a` estimator, `cushion_budget_seconds`, `MAX_PRE_DELAY_SECONDS` and the
static ≥16 GiB path are all unchanged from §2. All cushion tests were restated
at `cs25` — a cushion test describing a configuration nobody runs is worse
than no test.

### 15.5 AC #4 and AC #6 — the numbers are real, and they belong to `cs10`

**Not re-measured, and no further GUI capture requested.** The seam fix
changes how chunks are stitched, not when the first chunk is produced or
dispatched, so the shipped first-audio figure is Story 20.3's and is already
measured.

| figure | value | applies to |
|---|---:|---|
| **shipped first-audio TTFA** | **1,353 ms** | `cs25` — Story 20.3 §4.1, unchanged by this story |
| GUI long TTFA | 976 ms | **`cs10` — not shipped** |
| GUI short TTFA | 1,065 ms | **`cs10` — not shipped** |
| headless long TTFA | 829 ms | **`cs10` — not shipped** |
| headless long TTFA | 1,491 ms | `cs25` — contemporaneous, and the geometry that ships |

**AC #4's gate passes either way**, which is the part that survives the
revert: producer emit/drain measured **0.585× at `cs25`** and 0.619× at `cs10`
in the same session, both far under the `< 1.0×` OFR-E target. The shipped
configuration is the 0.585× one. The seam fix moves it very slightly *down*
(posted chunks grow 47,537 → 48,000 samples, so the denominator rises ~1 %) —
in the safe direction.

Everything §4 and §6 measured stands as measured. What changed is only which
row describes the product.

### 15.6 AC #7 — regression re-run after the revert

Every surface, re-run on the reverted tree, as separate invocations (§11.12
records why they must be separate):

| surface | result | vs. baseline |
|---|---|---|
| `tests/unit/services` + observability + models | **961 passed, 0 failed** | 953 → 961 |
| `tests/unit` (whole tree) | 1,582 passed, **30 failed, 4 errors** | pre-existing set, unchanged in count and identity |
| `tests/integration` + `test_qwen_tts_internals.py` | 175 passed, **4 failed** | unchanged |
| `tests/services` + settings + utils | 288 passed, **7 failed** | unchanged |
| `tests/ui` | 735 passed, **7 failed** | unchanged |

**Zero new failures.** Note that the revert required **no test rewrites for
correctness** — only restatements for *relevance* (the cushion rows, which
would otherwise describe an unshipped geometry). Every derived-geometry test
passed in both directions untouched.

### 15.7 Compile cache — one key, and it is the one that was already warm

The retune added one cache key (window 30 → 15, §9.2). The revert returns the
resolved window to **30**, which is the key that was already warm before this
story. The `a58fe999b1fca2f3` directory created for window 15 is now orphaned
— harmless, a few MB, and it will simply never be read. No cold compile is
expected on the first launch after the revert, because the window-30 key
(`391c2f2be3340b07`) still has its `meta.json`.

### 15.8 What a reader should take from this story

* **Follow-up B is closed unsuccessful.** The latency case for it was real and
  reproduced twice; the perceptual case against it was decisive. Do not re-open
  it on latency evidence alone.
* **A shipped audio defect was found and fixed** — 15–19 ms of speech deleted
  at every chunk boundary since Story 16.4 — with perceptual evidence for the
  fix at the shipped geometry.
* **The D-25 threading proved its worth in the direction nobody planned for**,
  by making the revert a one-line edit.
* **Two offline seam metrics failed to predict audibility** on the same 21
  judged files (§13.1). For this defect class the ear is the instrument;
  analysis explains mechanism.
* **`chunk_size = 15` is an open question**, and the codec runs at **12.5 Hz**,
  not the 12 Hz assumed since Story 16.3 (§11.2).

---

## §16. Round 4 — does `chunk_size = 15` survive the ear? (fixture + PREDICTION)

Commander asked for the §15.1 open question to be settled rather than left
open: *"15% is a fair time gain if we can achieve it."* This section is
written **before** the audition, per the practice §15's note 16 recommends.

### 16.1 The design — one variable, and a new baseline

| arm | geometry | stitching |
|---|---|---|
| **reference** | `chunk_size = 25` | **with** the seam fix |
| **candidate** | `chunk_size = 15` | **with** the seam fix |

Round 2's flaw was moving geometry *and* stitching against a reference that
had neither. Here both arms carry the fix, so anything heard is attributable
to the geometry alone.

**The reference is REGENERATED, not reused.** Rounds 1–3 anchored on round 1's
`cs25` files, but those are *pre-fix*. Reusing them would compare `cs15+fix`
against `cs25-pre-fix` — two variables again, and the exact mistake round 2
made. Both arms were generated in one process, from one model load, under one
compiled state.

`DEFAULT_CHUNK_SIZE` is untouched at 25; both geometries are set in-process
via the harness rebinder, and the generator's preflight refuses to run if the
committed constants have drifted.

Same 7 utterances as rounds 1–3, including **s-022** — clean at `cs25`,
blocking at `cs10`, plosive-dense, and the most sensitive row in the set.

**Loudness normalisation, declared.** All 14 files were normalised to equal
active-speech RMS after generation. The raw takes differed by **8 dB on
s-022, with the reference arm quieter** — sampling noise, not a property of
the geometry, but left alone it would attenuate the reference's own seams and
bias the audition *toward* the reference sounding cleaner, i.e. toward
confirming the prediction below. Applied uniformly to all 14 by one rule;
no clipping (worst peak 0.768); lengths and therefore seam positions
unchanged. This is a deviation from rounds 1–3 and is recorded as one.

### 16.2 Measured seam counts — the actual basis of the prediction

| utterance | reference `cs25` | candidate `cs15` | ratio |
|---|---:|---:|---:|
| s-020 | 1 | 2 | 2.0× |
| s-021 | 1 | 2 | 2.0× |
| s-022 | 1 | 2 | 2.0× |
| m-020 | 1 | 2 | 2.0× |
| m-021 | 1 | 3 | 3.0× |
| l-020 | 10 | 15 | 1.5× |
| l-021 | 7 | 12 | 1.7× |
| **total** | **22** | **38** | **1.73×** |

### 16.3 The model

Two things move against the candidate, and they compound:

**(a) Seam count.** 38 versus 22 — 1.73× more chances for an audible event.

**(b) Per-seam hazard, which is itself geometry-dependent.** The decode
window is `chunk_size + lookahead`: 30 at `cs25`, **20** at `cs15`, 15 at
`cs10`. A smaller window means less context, so the cold-start region the
blend fades into is worse. This is measured, not assumed (§13.2) — head
error at `cs10` is 0.824 against `cs25`'s 0.711, worse at every band.
Interpolating linearly in `1/window`, `cs15` lands **exactly halfway**
(0.767).

Estimating the per-seam hazard `p` from the two completed rounds, using
`P(file flagged) = 1 − (1−p)^N`:

* **`cs10 + fix`** (round 2): flagged l-020(24 seams), l-021(16), m-020(5),
  s-022(2); clean s-020(2), s-021(3), m-021(4). The small-N files flagged 2
  of 5 at a mean N ≈ 3.2 → **p ≈ 0.15**, and both large-N files flagging is
  consistent with it.
* **`cs25 + fix`** (round 3): 0 flagged of 7, over 22 seams → point estimate
  **p ≈ 0**, but honestly a 95 % upper bound of only ~0.13. **22 clean seams
  does not by itself prove `cs25`'s hazard is much below `cs10`'s** — the
  geometry dependence comes from the measured error profiles in (b), not from
  the audition counts.

Taking `p(cs15) ≈ 0.085` (halfway in error terms between a working
`p(cs25) ≈ 0.02` and `p(cs10) ≈ 0.15`) and the candidate's real seam counts:

| utterance | N | P(flag) |
|---|---:|---:|
| s-020 / s-021 / s-022 / m-020 | 2 | 0.16 each |
| m-021 | 3 | 0.23 |
| **l-021** | 12 | **0.66** |
| **l-020** | 15 | **0.74** |

**Expected flagged ≈ 2.3 of 7. P(clean sweep) ≈ 3 %** on the central estimate,
rising to ~30 % if the relationship is convex rather than linear (i.e. the
harm only bites past some density) and `p(cs15)` is nearer 0.03.

### 16.4 THE PREDICTION

**`chunk_size = 15` FAILS.** Most likely **2 flagged rows — l-020 and
l-021**, the two long fixtures, at 15 and 12 seams.

Confidence: **~75 % that at least one row flags.** Not higher, because the
`cs25` hazard is poorly constrained by 22 clean seams and the interpolation
could be convex.

**Where the crossover sits, since the number was asked for:** on this model
the fix's net benefit turns negative **just below `cs25` — around window 25,
i.e. `chunk_size ≈ 20`.** Even a hazard as low as `p = 0.02` yields ~0.7
expected flags across this fixture set; a confident pass needs `p ≲ 0.005`.
**`cs15` is already past the crossover, not straddling it.**

**The sharpest sub-prediction — s-022.** It flagged at `cs10` with only **2**
seams, far above the average per-seam hazard, which is what pointed at
plosive-dense content being especially vulnerable to transient doubling
(§13.3, H2). At `cs15` it again has 2 seams. **If content dominates, s-022
flags. If s-022 is clean while the long fixtures flag, the effect is purely
count-driven and H2's contribution to my model is wrong.**

### 16.5 What would falsify the model

| result | what it means |
|---|---|
| **0 of 7 flagged** | The model over-weights seam density. The crossover is sharper than linear and sits between 15 and 10. `cs15` is shippable — and *then* we measure its TTFA. |
| **Only l-020 / l-021 flag, all shorts clean** | Count-driven and content-independent. H2's transient-doubling term is not dominant; the model's (b) term needs re-deriving. |
| **s-022 flags but the long fixtures do not** | Content-driven and count-independent. The density term is wrong; seam *placement* matters more than seam *number*. |
| **≥4 flagged, several of them short** | Harm is worse than linear interpolation. The crossover is above `cs15`, and `cs25` is close to the edge — which would also put the shipped configuration under a question it does not currently have. |

### 16.6 Why this is worth a round despite predicting failure

Because the prize clears Commander's stated bar and the odds are not
negligible. Story 20.1 measured 1,157 ms at `cs15` against 1,662 at `cs25`
(−30 % on that story's scale). Interpolating Story 20.4's contemporaneous
re-measurement (`cs25` 1,491 ms, `cs10` 829 ms headless) puts `cs15` at
roughly **1,050–1,150 ms, a 23–30 % improvement** — comfortably past "15 % is
a fair time gain".

At a ~25 % chance of passing, one 10-minute audition to settle a 25 % latency
question permanently is a good trade. It is also the only way to settle it:
§13.1 established that no offline metric available here predicts audibility,
so the ear is the instrument.

### 16.7 Sequencing

**Audition only.** No GUI capture is prepared or requested. If `cs15` passes
the ear, its TTFA gets measured then; if it fails, a capture would have been
wasted. One ~10 minute listening task:

```
12_Story_20.4_AC5_Audition.bat
```

Defaults to round 4. Rounds 1–3 remain re-runnable (`L1 r1` / `r2` / `r3`)
and untouched.


---

## §17. AC #5 round 4 — cs15 vs cs25, both with the fix: **AMBIGUOUS, resolving to NO** 2026-09-01

| utt | cs25+fix (ref) | cs15+fix (cand) | preferred | verdict |
|---|---|---|---|---|
| l-020 | tonal_distortion | **none** | cs15 | cs15 cleaner |
| l-021 | none | **tonal_distortion** | cs25 | **BLOCKING — cs15 only** |
| m-020 | none | none | equivalent | clean |
| m-021 | none | none | equivalent | clean |
| s-020 | none | none | equivalent | clean |
| s-021 | none | none | equivalent | clean |
| s-022 | none | none | equivalent | clean |

**Preference: cs15 1 — cs25 1 — equivalent 5. A dead tie.** One defect on each
arm, both on long fixtures, and they swapped sides.

### The finding that settles it: the long-form residual is take-dependent

`cs25 + fix` is the *same configuration* in rounds 3 and 4, on regenerated audio:

| | l-020 | l-021 |
|---|---|---|
| round 3, cs25+fix | none | none |
| round 4, cs25+fix | **tonal_distortion** | none |

Identical configuration, different takes, different verdict. **The long-form
`tonal_distortion` is a property of the take, not of the geometry.** Round 4's
1–1 split is therefore within that noise, and this fixture set cannot separate
cs15 from cs25 at all.

That is consistent with it being the §11 codec-state residual — the ~35 % NRMSE
between the two decodes at every boundary — which is present at every geometry
and which the seam fix masks rather than removes.

### Both halves of §16.4's prediction fail in an informative way

- "At least one candidate row flags" — **true** (l-021). But it predicted 2.3
  expected flags on the candidate and got 1, with a matching flag on the
  reference it did not predict at all.
- The sharp sub-prediction was: *if s-022 is clean while the long fixtures flag,
  the hazard-geometry half of the model is wrong.* **s-022 is clean.** So the
  per-seam-hazard-scales-with-window half is **not supported**; the crossover
  estimate at `chunk_size ≈ 20` rests on it and should not be quoted as
  measured.

### Verdict: cs15 does not ship

Under AC #5 as written, one blocking row is a fail. Beyond the letter, the
substantive reasons are stronger:

1. **No evidence of benefit.** The preference is a dead tie. cs15 buys latency
   and demonstrably nothing perceptual.
2. **No evidence of separation.** Take-to-take variance on long content exceeds
   the difference between the two geometries on this fixture set. Separating
   them would need many samples per condition — a much larger listening burden
   than the value at stake.
3. **The binding constraint is not geometry.** Long-form quality is limited by
   the codec-state residual. Optimising chunk size around it is working on the
   wrong variable.

### The chunk-size question is now closed, with its boundary known

> **Superseded in part — see the AMENDMENT at the end of this file (2026-09-01).**
> The closure below stands as recorded; its stated reopening condition has since been
> met by Story 20.5.

- `cs10` — regresses, clearly (round 2, 3 blocking, never preferred)
- `cs15` — indistinguishable from `cs25`, with one blocking row (round 4)
- `cs25` — ships

Four auditions, 28 judgements. Anyone re-reading Story 20.1's latency sweep and
seeing `chunk_size = 10` marked "optimum" must read this section: **the sweep
optimised perceived latency and never asked the ear.**

### What would reopen it

Codec state caching across chunks (Mary's Finding 1, re-filed as an audio-quality
item in §11.9). It removes the residual rather than masking it. With long-form
takes reliably clean, the geometry question becomes separable and worth asking
again — with real headroom, since the harm that killed cs10 is the same residual.

### AMENDMENT 2026-09-01 — the reopening condition above has been MET

*Added by Story 20.5. Nothing above this line is changed: the four rounds, the 28
judgements, and the `cs10` / `cs15` / `cs25` verdicts all stand exactly as recorded. They
were correct for the code as it was then. What has changed is the code.*

Story 20.5 built the thing the previous paragraph names. Codec state caching across chunks
is implemented, verified and auditioned:

- the residual is **removed at the cause, not masked** — head NRMSE at the seams
  0.406 → **0.0078**, correlation **1.0000**, lag jitter **0 samples on every seam**,
  error-by-position **flat** rather than head-weighted, single-chunk decode bit-exact
  against `Decoder.forward`;
- the deterministic 555-sample edge loss reaches **zero** — `decode(N) == 1920·N` for every
  chunk after the first;
- two blinded NFR3 rounds followed. Round 1 (state caching vs what shipped) preferred the
  candidate **5–1** wherever the seam was exposed, and surfaced one coupling: with the cold
  start gone, the 64-sample consumer crossfade became the loudest thing left. Round 2, after
  scoping that crossfade to discontinuous producers, was a **unanimous pass** —
  candidate cleaner **10 of 14**, blocking **0**, preference **10 – 0 – 4**.

**So the precondition is satisfied.** "Long-form takes reliably clean" is now measured, not
hoped for: in round 2 the seam-carrying trials flagged a click on the crossfade arm and
**none** on the shipped arm.

Three things a reader of §17 should carry forward, because they change how the reopened
question must be asked:

1. **The take-to-take variance finding above is not repealed — it is sidestepped.** Story
   20.5's rounds decode *both arms of every pair from one talker run*, so within a pair the
   wording, prosody and duration are identical to the sample. Story 20.4 could not do that,
   because changing `chunk_size` changes what the streamer emits and therefore what the
   talker samples. **A chunk-size story still cannot**, so §17's warning about needing many
   samples per condition remains live for that question specifically.
2. **Reason 3 above — "the binding constraint is not geometry"** — was right, and is now
   spent. The codec-state residual was the binding constraint; it is gone. Geometry is next
   in line, which is exactly why the question is worth reopening.
3. **Do not reopen it yet.** Story 20.5's implementation carries a decode-time tax
   (+7–10 ms per chunk) that exists *only* to serve the 5-frame lookahead, and it scales with
   chunk count — the very variable a retune moves. Retiring the lookahead first removes that
   confound and independently gains 5 talker steps of TTFA. See Story 20.5's follow-up list,
   F1 before F2.

Evidence: `20-5-codec-state-caching.md`, `20-5-codec-state-caching-evidence.md` (Phase 1
mechanism), `20-5-phase2-evidence.md` (implementation, both audition rounds, the crossfade
finding). Raw judgements: `20-5-state-cache-audition.csv`,
`20-5-state-cache-audition-r2.csv`.
