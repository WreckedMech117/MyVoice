# Story 20.8 — Phase 1 evidence: the chunk-size re-baseline

**Status: Phase 1 COMPLETE. Tasks 1-4 done. STOPPED AT THE GATE (Task 4).**
Phase 2 (AC #3) and Task 6 are not started and are not self-authorised.
**Zero operator listening time was spent.** That was the point of the gate.

<!-- Filled by the dev agent, 2026-09-02. -->

## §0. Answer first

**Verdict: GO, and not marginally — but the GO is on LATENCY ONLY, and the
perceptual question is exactly as open as Story 20.4 left it.**

On current code, in one sitting, against a `cs25` control measured twice in
that same sitting:

| point | long TTFA | vs control | short TTFA | vs control | producer ratio | viable? |
|---:|---:|---:|---:|---:|---:|:--|
| **cs25** (control, pooled A+B) | **1,173.7 ms** | — | **1,179.0 ms** | — | 0.54 | ships today |
| cs15 | 740.9 ms | **−432.8 ms** | 737.9 ms | **−441.1 ms** | 0.547 | yes |
| cs10 | 528.4 ms | **−645.3 ms** | 516.4 ms | **−662.6 ms** | 0.558 | yes |
| **cs7** (the floor) | **391.6 ms** | **−782.1 ms** | **393.0 ms** | **−786.0 ms** | 0.567 | yes |

The GO bar was "beats the control by more than the within-arm spread (~83 ms)
with producer ratio < 1.0". The smallest margin on the board is **5.2× the
bar**, and the largest is **9.4×**. Every point's producer ratio is ~0.55,
roughly where `cs25` already sits. **No point fails either clause.**

Three things that make this a bigger result than the old curve implied, and
one that makes it smaller:

* **The lookahead retirement is what changed the lever.** `chunk_size = N` now
  means first emit at `N` frames, not `N + 5`. Segment 2 is essentially
  linear in the threshold — 1,036 / 606 / 391 / 261 ms at 25 / 15 / 10 / 7 —
  and per-frame talker cost is 41.4 / 40.4 / 39.1 / 37.3 ms, i.e. mildly
  *cheaper* at small N, so the win is slightly super-linear.
* **The watermark floor moved up, not down.** It is **7**, not Story 20.1's 6
  (§1.3). `cs7` is exactly the floor and is viable; `cs5` and `cs6` are not.
* **The short class stops degenerating.** 2 of 20 `cs25` control runs first-emit
  from `residual_flush` — TRUE_STREAM silently becoming batch. At 15, 10 and 7
  it is `threshold` in 10/10, every point (§3.3). That is a robustness win
  independent of the latency number.
* **Against: throughput costs ~5 %.** Long-form generation wall goes 10,666 →
  11,236 ms from `cs25` to `cs7`, monotone. Small, real, and paid on every
  generation.

**What has NOT moved, and must not be read into the above:** nothing here is
perceptual evidence. Seam count still rises as chunk size falls (Story 20.4
counted 22 → 38 seams from `cs25` to `cs15`; `cs7` is ~3.5× `cs25`), and "the
cause of the old harm is removed" is a mechanism argument. Story 20.4's four
rounds and 28 judgements were real. §5 states what an audition would have to
answer and what it would cost — and it is the expensive half of this story,
by roughly an order of magnitude.

---

## §1. Task 1 — the watermark floor, DERIVED

AC #1: *"Derive, do not assume, the watermark floor."*

### 1.1 The three source facts the derivation stands on

| fact | value | source |
|---|---|---|
| static watermark on ≥16 GiB hosts | **500 ms** | `audio_coordinator.py:62` (`_DEFAULT_STREAMING_WATERMARK_MS`), passed at `:1317`; the adaptive path is gated on `< 16 GiB` at `:131` |
| release rule | `buffered_bytes >= watermark_bytes` **or** `is_final` | `streaming_chunk_buffer.py:338-341` |
| codec geometry | `1920` samples/frame; **first** decode of a session pays a `555`-sample edge loss, every later one returns exactly `1920·N` | `streaming_decoder.py:201-202`; `codec_state_cache.py:130-135` |

At 24 kHz, `500 ms` = **12,000 samples**. `1920` samples/frame = **12.5 Hz**, so a
chunk carries `N × 80 ms` — the story's own stated basis.

### 1.2 The solve

The watermark is a no-op — i.e. the first chunk releases on its own and hands
back **nothing** — iff the first chunk alone clears the mark. On the shipping
state-carrying path the first chunk is the only one that pays the edge loss:

```
N·1920 − 555 ≥ 12000
        N     ≥ 6.5391
        N     ≥ 7          <- the floor
```

| `chunk_size` | 1st chunk | steady chunk | chunks held by the watermark |
|---:|---:|---:|---:|
| 4 | 296.9 ms | 320 ms | 2 |
| 5 | 376.9 ms | 400 ms | 2 |
| 6 | 456.9 ms | 480 ms | 2 |
| **7** | **536.9 ms** | **560 ms** | **1  ← FLOOR** |
| 10 | 776.9 ms | 800 ms | 1 |
| 15 | 1,176.9 ms | 1,200 ms | 1 |
| 25 | 1,976.9 ms | 2,000 ms | 1 |

### 1.3 The floor MOVED, and Story 20.1's stated floor is wrong for current code

Story 20.1 §5.4 concluded **"`chunk_size ≥ 6` keeps the watermark a no-op"**.
That was derived from its own measured per-chunk audio of ~83.3 ms/frame
(`2,083 ms` at `cs25`) on the pre-20.5 stateless geometry, where `500/83.3 = 6.0`.

On current code the measured per-chunk audio is **exactly 2,000.0 ms at `cs25`**
(§3, `median_chunk_audio_ms`, every point, every run) — i.e. exactly `80 ms`/frame,
exactly `1920·N`. Story 20.5's state caching is what made the identity exact.
So the floor is **7, not 6**, and `cs6` is now a cushion-penalty point rather
than a marginal one.

This is a small number with a large consequence for this story: it is precisely
the class of stale-constant carry-over the whole re-baseline exists to catch,
and it was found in the *derivation*, before any measurement.

### 1.4 Which sweep points are viable, stated before the sweep

`{25, 15, 10}` — required by AC #1 — are all comfortably above the floor.
**`cs7` was added** because it is the floor exactly, and a curve that stops at
10 cannot say whether the geometry lever is still paying out below it.
**`cs5` was NOT added**: it is below the floor, would hand back a cushion
penalty (Story 20.1 measured 316 ms at `cs5`), and its TTFA would not be a
geometry result. AC #1 permits smaller points "only if the watermark analysis
says they are viable"; 5 is not.

---

## §2. Task 2 — method

### 2.1 One sitting, one machine, both controls in it

Everything below was produced by a **single invocation** of
`20-8-sweep.py`, which runs every cell sequentially in one process tree on one
machine. Story 20.6 §12 is why that is not optional: the same code measured in
two sessions produced 46.66 vs 38.25 ms/frame, and a cross-session comparison
already produced one false conclusion in this epic.

**`cs25` is measured twice — first and last.** Story 20.6's numbers are NOT
carried over (AC #1 forbids it, and §3.4 shows how far off they would have
been). The second control pass exists because "one sitting" bounds
between-session drift but says nothing about drift *within* the sitting; the
A-vs-B difference measures it directly, and it feeds the go/no-go bar.

Host: RTX 5090, 31.8 GiB, cc 12.0, torch 2.10.0+cu128, `precision=auto`,
`compile=auto` (the shipping regime), voice `Sarira-F.quality.pt`,
`decode_window_frames` resolving to `(N, 0)` — the lookahead is retired on
every point, so `chunk_size = N` means first emit at **N** frames.

### 2.2 Warm measurements, and the cold-compile cost priced separately

AC #1: *"the cold-compile cost per point is stated, and the measurements are
taken warm."*

Every cell runs `--prime --warmup 1`: one suppressed priming generation, then
one discarded warm-up run, then the measured runs. Cells are ordered **long
before short within a point**, so the long cell's priming generation is the one
that meets a cold compile key and the short cell's is warm on the same key. The
difference between the two priming times is that point's cold-compile cost, and
it is paid outside every measured run. §3.5 reports what that measured, per
point, including which key each point actually lands on.

### 2.3 What was reused, and the one thing that was not

`tools/ttfa_spike_harness.py` drives the production `_generate_true_stream` with
the `--chunk-size` override, unchanged except for one **additive** column
(`median_decode_chunk_ms` / `n_decode_samples`): `decode_chunk_latency_ms` was
already being collected by `RunCollector` and then dropped on the floor, and
AC #1 asks for per-chunk decode time per point. No production file was touched.

The override was verified to still reach all three geometry sites before the
sitting: with `DEFAULT_CHUNK_SIZE` rebound to 10, `resolve_streamer_geometry()`
returns `(10, 0)`, so `decode_window_frames = sum(...) = 10` and each point
really is a distinct compile-cache key. Story 20.1 §5.5's "the sweep is free"
no longer holds, exactly as the story predicted.

**`20-6-compare-arms.py` was not re-pointed at these files, and the reason
matters.** It (and `20-4-aggregate-gui.py`, which it imports) parses *GUI
metric-stream captures*: one row per metric, keyed by `session_id`, with the
startup-priming generation, its registry-suppressed post, and the operator's
generations interleaved in one file. Its two hard-won behaviours — group by
`session_id`, and exclude a generation whose segment 1a exceeds 200 ms as
semaphore-contaminated — defend against hazards that **cannot occur in a
headless capture**: the harness writes one already-segmented row per run, swaps
a fresh collector between strictly sequential generations, and has no operator
to click Generate early. Feeding a different schema to that parser would have
reused its name, not its judgement.

What **was** reused is its judgement, in `20-8-aggregate-sweep.py`:
the same 200 ms dispatch bar applied to `seg1a_dispatch_overhead_ms` with
excluded rows named rather than dropped; per-frame talker cost taken from the
**long class only**, because a short utterance can first-emit from
`residual_flush` where the frame count is the whole utterance and varies per
take; medians with min/max carried.

### 2.4 One caveat on the absolute numbers

The harness reproduces everything up to and including `StreamingChunkBuffer.push`
but not the PyAudio device-open on chunk 0 (~50-100 ms, Story 17.3). Every
absolute TTFA here is therefore that much optimistic — **uniformly across all
points**, so the deltas the gate turns on are unaffected. This is also why the
control had to be re-measured rather than compared to Story 20.6's GUI figure.

---

## §3. Task 2 — the curve

Full aggregator output: `20-8-aggregate-output.txt`. Raw rows:
`20-8-sweep-*.csv`. `n = 10` warm runs per cell, one discarded warm-up, one
suppressed priming generation before each.

### 3.1 Long class

| point | n | first-emit frames | seg 2 talker | ms/frame | TTFA(post) | **TTFA(release)** | cushion | ratio | chunks | decode/chunk | audio/chunk | gen wall |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **cs25 [control A]** | 10 | 25 | 1,035.9 | 41.43 | 1,166.9 | **1,167.5** | 0.3 | 0.539 | 10 | 36.6 | 2,000.0 | 10,666 |
| **cs25 [control B]** | 10 | 25 | 1,049.3 | 41.97 | 1,180.7 | **1,181.5** | 0.0 | 0.547 | 10 | 32.2 | 2,000.0 | 10,924 |
| cs15 | 10 | 15 | 605.6 | 40.37 | 740.8 | **740.9** | 0.0 | 0.547 | 17 | 34.0 | 1,200.0 | 10,864 |
| cs10 | 10 | 10 | 390.7 | 39.07 | 528.1 | **528.4** | 0.0 | 0.558 | 25 | 33.6 | 800.0 | 11,058 |
| cs7 | 10 | 7 | 261.2 | 37.31 | 391.1 | **391.6** | 0.0 | 0.567 | 35 | 32.0 | 560.0 | 11,236 |

All times ms. `TTFA(release)` = `TTFA(post)` + the consumer cushion; it is the
headline because it is the only figure that moves when the watermark floor is
crossed, which is the misattribution §1 exists to prevent.

**`audio/chunk` is exactly `N × 80 ms` at every point** — 2,000.0 / 1,200.0 /
800.0 / 560.0. That is `1920·N` samples at 24 kHz, measured, on all four
points. It is the empirical confirmation of §1's floor arithmetic, and the
direct refutation of the 83.3 ms/frame the old curve was built on.

### 3.2 Short class

| point | n | seg 2 talker | TTFA(post) | **TTFA(release)** | cushion | chunks | first-emit path | gen wall |
|---|--:|--:|--:|--:|--:|--:|---|--:|
| **cs25 [control A]** | 10 | 1,036.5 | 1,175.5 | **1,176.5** | 1.0 | 2 | threshold 10/10 | 1,324 |
| **cs25 [control B]** | 10 | 1,050.5 | 1,186.4 | **1,186.9** | 0.5 | 1–2 | **threshold 8/10, `residual_flush` 2/10** | 1,338 |
| cs15 | 10 | 603.5 | 737.6 | **737.9** | 0.5 | 2–3 | threshold 10/10 | 1,346 |
| cs10 | 10 | 390.4 | 515.4 | **516.4** | 0.8 | 3–4 | threshold 10/10 | 1,456 |
| cs7 | 10 | 259.4 | 392.5 | **393.0** | 0.0 | 4–5 | threshold 10/10 | 1,376 |

Per-frame talker cost is deliberately **not** computed for the short class:
`residual_flush` runs never reach the nominal threshold, so the frame count is
the utterance and varies per take. That is `20-6-compare-arms.py` §2's rule,
applied unchanged.

### 3.3 The short class's degeneration reproduces — and it is a second reason to move

2 of the 20 `cs25` short control runs first-emitted from `residual_flush`
(TTFA 933 ms and 1,098 ms, 1 chunk): the take was shorter than one 25-frame
window, so TRUE_STREAM silently became batch for that generation. **Every point
at 15 and below took the `threshold` path in 10/10 runs.** This is Story 20.1
§5.3's B1 finding reproducing exactly at the current geometry, and it is an
argument about *reliability of the streaming behaviour*, not about latency.

It is also what inflates the short-class control spread to 273.5 ms (§4): the
control is bimodal, not noisy.

### 3.4 The cushion is a no-op at every measured point — as §1 predicted

`seg4_consumer_cushion_ms` is 0.0–1.0 ms on every point including `cs7`, and
`consumer_chunks_held` is 1. The watermark is releasing on the first chunk
everywhere, so **none of these TTFAs contain a cushion penalty** and every
delta in §3.1 is a geometry result. That was the specific misreading AC #1
asked to be ruled out, and it is ruled out by measurement, not assumption.

### 3.4a Per-chunk decode time is FLAT in chunk size — and it is not 11.8 ms

`decode_chunk_latency_ms` (pure `decode_fn` wall time,
`streaming_decoder.py:445-471`) measures **32–37 ms per chunk at every point**,
`cs25` through `cs7`, with no trend. Decode is latency-bound, not
throughput-bound: halving the chunk does not halve the decode.

The story's Dev Notes carry **11.8 ms** for `cs25` from Story 20.6, which this
contradicts by ~3×. The two are not the same quantity and both are right:
Story 20.6 §5's figure comes from `20-6-lookahead-bench.py`, a **decoder-only,
CUDA-synchronised bench replaying captured tokens with no talker running**.
The figure here is decode on the live streaming path, where the talker is
generating on the same GPU concurrently. For this story the in-situ number is
the relevant one, because it is what decides whether more chunks cost
throughput.

They do, a little: total decoder-thread work goes from 10 × 36.6 ≈ 366 ms at
`cs25` to 35 × 32.0 ≈ 1,120 ms at `cs7`. It stays concurrent with the talker —
the producer ratio only moves 0.539 → 0.567 — and it shows up end-to-end as the
~5 % generation-wall cost in §3.1 (10,666 → 11,236 ms). Neither figure
threatens the OFR-E `< 1.0×` gate, which has ~1.8× of headroom at every point.

**Do not carry the 11.8 ms forward as an in-situ per-chunk decode cost.**

### 3.5 Cold-compile cost per point — the story's premise is CONFIRMED, with one free coincidence

Each point resolves `decode_window_frames = sum(resolve_streamer_geometry())
= N + 0 = N`, so each is its own `compile_cache` key. Verified directly
(`20-8-cachekey-probe.py`, which primes once so `engage_compile_optimizations`
reaches `set_torchinductor_cache_dir`, then reads the env var back):

| `decode_window_frames` | key dir | state before the sitting |
|---:|---|---|
| 30 | `391c2f2be3340b07` | pre-existing — the pre-20.6 geometry (`cs25` + lookahead 5) |
| **25** | `205b94cf67d78d0a` | pre-existing — **what ships today** |
| **15** | `a58fe999b1fca2f3` | **pre-existing** — warmed 2026-09-01 12:24, when Story 20.4 briefly shipped `cs10` + lookahead 5, which is also window 15 |
| **10** | `ef8efd9237ad2a9f` | **created by this sitting** |
| **7** | `5afc7362ef27323b` | **created by this sitting** |

Priming time, which is where the cost lands (it is paid before any measured
run, so all measurements are warm):

| cell | priming | key state | cold-compile cost |
|---|--:|---|--:|
| cs25 long/short (A and B) | 8,921 / 8,903 / 8,714 / 8,540 ms | warm | — (baseline ≈ **8.9 s**) |
| cs15 long | 9,382 ms | warm (collision above) | **+0.5 s — free, by coincidence** |
| cs10 long | **28,108 ms** | cold, new key | **+19.2 s** |
| cs7 long | **27,113 ms** | cold, new key | **+18.2 s** |
| every short cell | 8.5–9.0 s | warm (long cell paid it) | — |

**So Story 20.1 §5.5's "the sweep is free" is dead, exactly as this story
predicted** — but `cs15` was free anyway, because its key collides with the one
Story 20.4 warmed. That is luck, not a property, and it would not repeat for a
`cs12` or `cs8`. Budget **~18–20 s of cold compile per genuinely new point**
(the story's ~22.5 s figure from Story 18.4 is a slight over-estimate here, and
a safe one).

For a *user*, this is a one-time cost on first launch after the retune, paid by
`warmup_compile_async` before the first generation — the same shape of cost
Story 20.6's retirement already imposed once.

---

## §4. The go/no-go

Stated in the story before the work:

> **GO** if a viable point beats `cs25`'s same-sitting TTFA by a margin larger
> than the within-arm spread (~83 ms), **and** its producer ratio stays under
> 1.0×.

### 4.1 The control, and the bar

`cs25` was measured **twice in the sitting**, first and last, never carried over
from Story 20.6:

| | long | short |
|---|--:|--:|
| pass A median TTFA(release) | 1,167.5 ms [1,140.3–1,206.0] | 1,176.5 ms [1,140.0–1,195.0] |
| pass B median TTFA(release) | 1,181.5 ms [1,153.1–1,202.0] | 1,186.9 ms [933.4–1,206.9] |
| **within-sitting drift (B − A)** | **+14.0 ms** | **+10.4 ms** |
| pooled within-arm spread (max−min, n=20) | **65.7 ms** | 273.5 ms (bimodal, §3.3) |

The long class's own spread (65.7 ms) is **tighter** than Story 20.6's 83 ms, and
the sitting drifted by only 14 ms end to end. The bar used below is
`max(this sitting's spread, 83 ms)` — the conservative choice.

### 4.2 The arithmetic

| point | long delta | bar | ratio | verdict |
|---|--:|--:|--:|---|
| cs15 | **−432.8 ms** | 83.0 | 0.547 | **GO** (5.2× the bar) |
| cs10 | **−645.3 ms** | 83.0 | 0.558 | **GO** (7.8×) |
| cs7 | **−782.1 ms** | 83.0 | 0.567 | **GO** (9.4×) |

| point | short delta | bar | ratio | verdict |
|---|--:|--:|--:|---|
| cs15 | **−441.1 ms** | 273.5 | 0.316 | **GO** |
| cs10 | **−662.6 ms** | 273.5 | 0.538 | **GO** |
| cs7 | **−786.0 ms** | 273.5 | 0.551 | **GO** |

Both clauses hold at every viable point, on both utterance classes, with the
conservative bar. **The go/no-go returns GO.**

### 4.3 What the GO does and does not authorise

It authorises **asking Commander whether to open Phase 2**. It is not itself
permission to open it, and Phase 2 has not been started. Specifically, this
result does **not** say which geometry should ship: `cs7` wins on latency by
the largest margin *and* carries the most seams and the largest throughput
cost, and there is no perceptual evidence at any of the three points. Choosing
between them is what the audition in §5 is for.

### 4.4 Anomalies — named, not hidden

1. **`cs7` short, first attempt: hard process death** (`0xC0000005`, access
   violation) during the priming generation, immediately after torch's
   "CUDAGraph … we have observed 9 distinct sizes" warning. **Retried
   immediately in the same sitting: clean, 10/10 runs, no warning of interest.**
   It cannot be attributed to `cs7`: the `cs7` *long* cell had already completed
   cleanly on the same key minutes earlier, and the retry reproduced nothing.
   Recorded because a native crash near a compiled path deserves to be on the
   record if it ever recurs — `20-8-sweep-P-short-cs7-FAILED-access-violation.log`.
2. **`cs25` short control pass B, first attempt: `KeyError` inside
   `torch/_dynamo/guards.py:4448`** (`get_guard_fail_reason` →
   `orig_code_map[code]`) — a torch-internal failure while *logging a
   recompilation reason*, not a MyVoice code path. Retried immediately: clean —
   `20-8-sweep-B-short-cs25-FAILED-dynamo-keyerror.log`.
3. **The retry invocation overwrote the pass-A short control CSV.** The
   launcher derives a cell's tag from `--control`, so re-running one point with
   `--no-second-control` re-labelled it "A" and clobbered the file. The pass-A
   rows were recovered at 1 ms resolution from the sitting's own console log
   (`20-8-sweep-console.log`) into `20-8-sweep-A-short-cs25.csv`, which is
   therefore **reconstructed, not raw** — it carries the printed fields (TTFA,
   segments 1–4, path, chunks, wall) and blanks for the fields the console line
   does not print. Nothing in §3.2 or §4.1 for the short class depends on a
   blanked field. A refuse-to-overwrite guard was added to the launcher.
4. Both retries ran within ~10 minutes of the sitting, on the same machine, with
   no launches in between. They are inside the sitting in every sense that
   Story 20.6 §12's finding cares about.

---

## §5. Task 3 — AC #2: what is still UNKNOWN, and what an audition would cost

### 5.1 The perceptual question is NOT settled, and the mechanism argument is not an audition result

Story 20.4 spent **four rounds and 28 judgements** establishing that `cs10`
regresses and `cs15` is indistinguishable-with-one-blocking-row. Those rounds
were real and they are not repealed. Story 20.5's amendment to §17 says the
*reopening condition* has been met — the codec-state residual that the seam fix
masked is now removed at the cause. That is a statement about a **mechanism**.

What is still unknown, stated plainly:

1. **Seam count still rises as chunk size falls, and nothing removed that.**
   Story 20.4 §16.2 counted 22 seams at `cs25` against 38 at `cs15` on the same
   7 fixtures — 1.73×. At `cs10` and `cs7` it is worse again. State caching made
   each seam far cleaner; it did not make them fewer. "Cleaner seams" and "more
   seams" are opposing terms and their product has never been auditioned at any
   geometry below 25 on post-20.5 code.
2. **The per-seam hazard model that predicted where the crossover sits is
   unsupported.** Story 20.4 §17 recorded that its sharp sub-prediction failed:
   `s-022` was clean while the long fixtures flagged, which falsifies the
   "per-seam hazard scales with window size" half. **The `chunk_size ≈ 20`
   crossover estimate must not be quoted as measured.** So there is no surviving
   model that says where the good geometry is; only measurement can say.
3. **Story 20.5's own rounds certify the seam at `cs25` only.** Round 2 was a
   unanimous pass — but every trial in it was `cs25` vs `cs25`. It is direct
   evidence that the *decode* is clean at the committed geometry and **no
   evidence at all** about `cs10` or `cs7`.
4. **A latency win is not a quality argument.** Story 20.4 §17's closing line is
   still the right warning: *"the sweep optimised perceived latency and never
   asked the ear."* §3's curve is the same kind of artefact. It is a reason to
   ask the question, not an answer to it.

### 5.2 Story 20.5's one-talker-run-per-pair trick is NOT available as-is

Story 20.5 §7.2 could render both arms of every pair from **one talker run**,
because nothing upstream of the decoder was touched: the tokens were literally
reused. That collapsed attribution to a single pair per utterance.

Story 20.4 §17's amendment states the consequence for this story explicitly:
*"A chunk-size story still cannot [do that], so §17's warning about needing many
samples per condition remains live for that question specifically."* If the two
arms are two production generations, they are different takes — and §17 measured
what that costs: the **same** configuration (`cs25 + fix`) flagged
`tonal_distortion` on `l-020` in round 4 and not in round 3. **The long-form
defect is a property of the take, not of the geometry.** Round 4's 1-1 split was
inside that noise, and the fixture set could not separate `cs15` from `cs25` at
all.

### 5.3 The cost, sized honestly

Treating each trial as a paired preference judgement and the round as a
two-sided sign test on the discordant (non-`equivalent`) judgements, at
α = 0.05 and 80 % power:

| true preference rate on discordant trials | discordant judgements needed | trials at d = 0.29 (Story 20.4 r4's observed discordance) | listening time @ ~1.4 min/trial |
|---|---:|---:|---:|
| 0.85 (large effect) | 14 | **47** | ~65 min |
| 0.80 | 20 | **68** | ~95 min |
| 0.75 | 29 | **102** | ~2.4 h |
| 0.70 (modest) | 47 | **164** | ~3.8 h |

Calibration: Story 20.5's 14-trial round and Story 20.6's 16-trial round were
each **~20 min** of Commander's time, hence ~1.4 min/trial. Story 20.4's four
rounds totalled **28 judgements** — and were still unresolvable.

So the honest estimate for a take-different audition that could actually settle
the question: **3-5 rounds of 20-30 trials each** (listener fatigue caps a
sitting well below 68 trials), i.e. **1.5-3.5 hours of Commander's listening
time**, plus ~30-60 min of machine time regenerating 100-140 long-form fixture
takes, plus a real risk of landing on another "ambiguous, resolving to no" —
which is exactly where round 4 landed with a much smaller ask.

`d = 0.29` is itself estimated from 7 trials. It is the weakest number in the
table and it drives the whole thing; the range above should be read as an order
of magnitude, not a plan.

### 5.4 A cheaper design that may be available — and the one check that decides it

This changes how Phase 2 should be scoped, so it is stated as a lead, not a
result.

The trick was ruled out because *chunk size perturbs the talker*. But on
inspection of the mechanism, chunk size does not enter the talker's forward
pass at all: `_build_true_stream_talker` captures `codec_ids` via a forward hook
and pushes them to the streamer's queue; the streamer only decides **when**
tokens are pulled off, and its bounded queue applies **backpressure**, which
changes timing, not values. The reason Story 20.4's arms were different takes is
that they were two separate *generations* with different RNG draws — not that
the geometry causally perturbs the token stream.

If that holds, the trick is recoverable by **offline re-chunking**: capture one
talker run's token stream (Story 20.5's `20-5-regen-audition-fixture.py` already
captures exactly this, at the decode boundary), concatenate it — with the
lookahead retired the emitted chunks are non-overlapping, so a plain `cat` is
exact — and re-split it at `cs25` and at the candidate `N`, rendering each
through a real `StreamingDecoderWorker` + real `StreamingChunkBuffer`. Both arms
then come from one take, identical to the sample, and the audition collapses to
Story 20.5's shape: **~14-16 trials, one round, ~20-25 min.**

**The check that decides it, and it is cheap:** render a captured token stream
offline at `cs25` and compare it **bit-for-bit** against production's live
`cs25` render of that same captured stream. If they match, the offline re-chunk
*is* the production decode path and the design is sound. If they do not, the
large audition in §5.3 is the real cost and the lead is dead. This is the same
exact-equality bar Story 20.5 used for its own decode claims, and it is one
short script — it should be Phase 2's first task, before any fixture is
generated.

One limitation survives either way: the offline design tests the **decode-side**
geometry faithfully, which is the seam question. It cannot test any effect chunk
size might have through the *live* producer/consumer interaction — the honest
statement is that it removes the take lottery, not that it makes the arms
identical in every respect a listener could reach.

### 5.5 What Phase 1 changes about how Phase 2 should be scoped

Five things, in the order they should be decided:

1. **Run the §5.4 bit-exactness check FIRST.** It costs one short script and it
   decides whether the audition is ~14 trials or ~50–170. Nothing else in
   Phase 2 should be scheduled before its answer is known.
2. **Do not scope Phase 2 as "commit `cs10`".** The old curve's "optimum is 10"
   is superseded twice over: the floor is 7 (not 6), the lookahead retirement
   changed the shape, and the largest latency win on the board is at **`cs7`**.
   The geometry to audition is an open choice between 15, 10 and 7 — and
   auditioning three candidates against one reference is 3× the trials, so
   narrowing the field on non-perceptual grounds (throughput, seam count, the
   `cs7` crash in §4.4.1) before listening is worth doing explicitly.
3. **`cs15` deserves a cheaper hearing than the others.** Story 20.4 round 4
   already auditioned `cs15` against `cs25` — and found a dead tie inside
   take-noise, on *pre-20.5* code. Its latency win here is −433 ms, more than
   half the total available. If Phase 2 needs a low-risk landing point, `cs15`
   is the one with the most prior listening already spent on it.
4. **Budget the cold compile as a user-visible first-launch cost**, ~18–20 s
   once, and confirm `warmup_compile_async` pays it rather than the user's
   first generation. Story 20.6 already imposed this shape of cost once, so the
   mechanism is proven; it is a release note, not a risk.
5. **AC #3's threading requirement is already verified end-to-end by this
   sweep.** Every point ran with `resolve_streamer_geometry()` returning
   `(N, 0)` and produced its own compile key, three sites following one
   constant (§3.5). Phase 2's geometry change is a one-line edit to
   `DEFAULT_CHUNK_SIZE`, as Story 20.6 found in the other direction.

---

## §6. Regression status

Not run. AC #4 belongs to Phase 2 (Task 6) and Phase 2 is gated. No production
source file was modified by Phase 1 — `git diff` is one additive column in
`tools/ttfa_spike_harness.py`, a `tools/` spike file that nothing under
`src/myvoice/` imports.

---

## §7. Artifacts

| file | what |
|---|---|
| `20-8-sweep.py` | the one-sitting launcher |
| `20-8-aggregate-sweep.py` | floor derivation + curve + go/no-go arithmetic |
| `20-8-sweep-{A,P,B}-{long,short}-cs{N}.csv` | raw per-run rows |
| `20-8-sweep-{A,P,B}-{long,short}-cs{N}.log` | per-cell stdout/stderr, incl. priming times |
| `20-8-sweep-manifest.json` | per-cell rc, wall time, new compile-cache keys |
| `20-8-sweep-summary.json` | the aggregated curve |
| `20-8-sweep-console.log` | the sitting's console (also the source of the reconstructed pass-A short control) |
| `20-8-aggregate-output.txt` | the aggregator run, verbatim |
| `20-8-cachekey-probe.py` | which compile-cache key each geometry lands on |
| `20-8-sweep-*-FAILED-*.log` | the two failed first attempts (§4.4) |
| `20-8-sweep-manifest-retry*.json` | the two retry invocations |
