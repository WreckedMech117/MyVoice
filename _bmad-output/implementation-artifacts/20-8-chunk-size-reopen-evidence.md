# Story 20.8 — evidence: re-baselining and reopening the chunk-size question

**Status: Phase 1 COMPLETE (§0–§6). Viability check COMPLETE (§7). Phase 2
IMPLEMENTED and VERIFIED (§8–§9) — `chunk_size` is committed at 10, the
threading is verified at all three sites in both directions, the exact bars
hold, and the suite is unchanged.**

**Outstanding: two things, both needing Commander at the keyboard, both in one
hand-off at §9.6** — the NFR3 audition (the blocking gate, prediction recorded
in §8.4 *before* the fixture was generated) and the two-arm GUI TTFA capture.

Phase 1 spent **zero** operator listening time; that was the point of its gate.

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

## §6. Regression status — Phase 1

**Phase 1 modified no production source at all.** Its only tree change was one
additive column in `tools/ttfa_spike_harness.py`, a `tools/` spike file nothing
under `src/myvoice/` imports.

AC #4 belonged to Phase 2 and has since been run — **see §9.3 (the exact bars)
and §9.4 (the suite comparison)**.

---

## §7. The one-talker-per-pair VIABILITY CHECK (run at Commander's direction, 2026-09-02)

Run before any candidate selection or fixture building. **Nothing was built, no
geometry was chosen, no production source was touched.**

### 8.1 Four claims, not two

§5.4 proposed one check. Commander correctly split it: verifying that an offline
render matches a live one says nothing about whether the *tokens* being
re-chunked are the tokens a real run at that geometry would produce. Two more
were added — one control on each side, because a claim of identity is
unreadable without knowing whether the thing is identical to itself:

| | claim | why it is here |
|---|---|---|
| **(a)** | offline re-chunk render == live render, bit-for-bit, through the REAL streamer / worker / buffer | the render is faithful |
| **(a-ctl)** | the same offline render, run twice, is bit-identical | if the decoder is not reproducible against itself, bit-exactness against a live run is unattainable and the bar must be restated, not failed |
| **(b)** | the talker's token stream does not depend on `chunk_size` | without it the candidate arm is a fiction |
| **(c)** | the pipeline reproduces its own token stream at a fixed seed at all | without it a (b) mismatch means "not reproducible", not "chunk size perturbs the talker" |

And a **fourth**, which the in-process design cannot reach and which I am
naming because nobody had: **(d) cross-compile-key invariance** — a shipped
`cs7` build loads with `decode_window_frames = 7`, one of `compile_cache`'s
seven key dimensions, so it reads a *different inductor cache directory* and
may run different compiled kernels. Chunk size can therefore reach the talker
by a route that has nothing to do with the forward pass.

Method: `20-8-onetalker-viability.py` (in-process, a/a-ctl/b/c) and
`20-8-crossprocess-tokens.py` (fresh process per capture, d). Both drive
production `_generate_true_stream`; the offline renderer constructs
`CodecTokenStreamer` → `_build_true_stream_decode_fn` → `apply_codec_state_geometry`
→ `StreamingDecoderWorker` in the same order and with the same arguments as
`qwen_tts_service.py:5281-5331, :5437-5444`, and pushes posted PCM through a
real `StreamingChunkBuffer` with **`crossfade_samples = 0`** — the value
`app.py:3242-3247` passes when the producer declares continuity, not the
harness's 64.

Two seeds (1234, 99991), long utterance, one model load, one compiled state.

### 8.2 (c) SEEDED DETERMINISM — **PASS**

Two live `cs25` runs at the same seed produce **bit-identical** token streams:
`[244, 16]` at seed 1234, `[252, 16]` at seed 99991. The pipeline is
seed-reproducible, so (b) is answerable.

### 8.3 (b) TOKEN INVARIANCE — **PASS, exactly, 4/4**

| seed | comparison | result |
|---|---|---|
| 1234 | live cs25 vs live cs7 | **identical**, `[244, 16]` |
| 1234 | live cs25 vs live cs10 | **identical**, `[244, 16]` |
| 99991 | live cs25 vs live cs7 | **identical**, `[252, 16]` |
| 99991 | live cs25 vs live cs10 | **identical**, `[252, 16]` |

Not "no differences found" — bit-identical tensors, same shape, same values.
Within one compiled state, **chunk size does not reach the talker**. The code
reading in §5.4 was right, and is now measured rather than argued.

A self-check also passed: re-splitting the captured flat stream at its own
chunk size reproduces the captured chunk boundaries exactly, so the
concatenate-and-re-split step introduces nothing.

### 8.4 (a) RENDER FIDELITY — **PASS, at the strongest bar the pipeline supports**

**Bit-exact in 9 of 12 comparisons. The 3 misses are decoder self-noise, and
the (a-ctl) control proves it rather than assuming it.**

Seed 1234 — every comparison bit-exact:

| comparison | result |
|---|---|
| (a-ctl) offline cs25 vs offline cs25 | bit-exact |
| (a-ctl) offline cs7 vs offline cs7 | bit-exact |
| (a-ctl) offline cs10 vs offline cs10 | bit-exact |
| (a) offline cs25 vs **live cs25** — same geometry | **bit-exact** (PCM and buffer bytes) |
| (a) offline cs7 vs **live cs7** — **cross geometry** | **bit-exact** (PCM and buffer bytes) |
| (a) offline cs10 vs **live cs10** — **cross geometry** | **bit-exact** (PCM and buffer bytes) |

Seed 99991 — three cells are not bit-exact, **and so is their own control**:

| comparison | bit-exact | differing samples | max abs diff | NRMSE |
|---|---|---:|---:|---:|
| **(a-ctl) offline cs25 vs itself** | **no** | 529 / 483,285 (0.11 %) | 4.88e-4 | 3.3e-5 |
| (a-ctl) offline cs7 vs itself | yes | — | — | — |
| **(a-ctl) offline cs10 vs itself** | **no** | 787 (0.16 %) | 6.10e-4 | 5.3e-5 |
| (a) offline cs25 vs live cs25 | no | 544 (0.11 %) | 4.88e-4 | 3.4e-5 |
| (a) offline cs7 vs live cs7 | **yes** | — | — | — |
| (a) offline cs10 vs live cs10 | no | 572 (0.12 %) | 4.88e-4 | 3.8e-5 |

**Every cell where offline≠live is a cell where the decoder differs from
itself, by the same magnitude, at the same location.** Where the decoder is
reproducible (cs7 at seed 99991, and all of seed 1234), offline == live
**bit-for-bit** — including in the cross-geometry form, which is the form the
audition needs.

**Where the non-determinism lives, mechanically.** Every miss starts at sample
479,829–479,836 of 483,285 — the final ~3,450 samples. At seed 99991 the
utterance is 252 frames; `cs25` leaves a terminal residual of **2 frames**,
`cs10` leaves **2**, and `cs7` divides exactly (36 × 7 = 252) and leaves
**none**. At seed 1234 (244 frames) the residuals are 19 / 4 / 6 frames and
everything is bit-exact. Six configurations, perfect separation:

> **The decoder is bit-reproducible except on a 2-frame terminal residual
> chunk.** A 2-frame decode is the smallest window the codec is ever handed,
> and it is the only shape that misbehaves in this data.

Magnitude: `max_abs_diff = 4.88e-4` in float32 is exactly **16 int16 units out
of 32,767** — about −66 dBFS, on 0.1 % of samples, in the last 144 ms. It is
present **between two live runs of the same code**, so it is not a cost of the
offline design.

**Labelled honestly, per the instruction not to round a near-miss up:** (a) is
not "bit-exact in all cases". It is *"bit-exact wherever the pipeline is
bit-exact against itself, and otherwise identical to within the decoder's own
self-noise, which is confined to a 2-frame terminal residual chunk at
−66 dBFS."* That is the strongest form of (a) this pipeline can support, and
the fixture generator can make it moot by not letting either arm end on a
2-frame residual.

### 8.5 (d) CROSS-COMPILE-KEY INVARIANCE — **FAILS**, and this is the one finding that changes the design

Fresh process per capture, same seed (1234), same prompt, same text:

| capture | compile-cache dir | frames | chunks |
|---|---|---:|---:|
| cs25 | `205b94cf67d78d0a` | 244 | 10 |
| cs25 (repeat, fresh process) | `205b94cf67d78d0a` | 244 | 10 |
| cs7 | `5afc7362ef27323b` | **256** | 37 |
| cs7 (repeat, fresh process) | `5afc7362ef27323b` | **256** | 37 |

| comparison | result |
|---|---|
| **control** — cs25 vs cs25, two fresh processes, same key | **IDENTICAL** `[244, 16]` |
| **control** — cs7 vs cs7, two fresh processes, same key | **IDENTICAL** `[256, 16]` |
| cs25 vs cs7, different key | **DIFFER** — `[244,16]` vs `[256,16]`, identical for the first **5 frames**, then diverge |

The two controls are what make this attributable. Process boundaries, seeding
and RNG are ruled out: the same key reproduces exactly across processes, twice.
The draw is a **deterministic function of (seed, compiled state)**, and the
compiled state depends on chunk size *only* through `decode_window_frames` →
the inductor cache directory. Different compiled kernels, different
floating-point rounding, divergent sampling from frame 6 onward.

**What it does and does not mean.** It is **not** a discovery that chunk size
perturbs the talker — §7.3 proves it does not, within a compiled state. It is
the same class of thing as a torch upgrade or a different GPU: the model draws
a different, equally valid sample. But it does mean the naive fixture recipe
("capture at cs25, re-chunk at cs7") produces a candidate arm whose *content*
no `cs7` build would have drawn for that seed — which is precisely the fiction
Commander named.

**And it is closed by a design choice, not by more measurement:** capture the
talker run in a process running **at the candidate geometry**, then re-chunk
that one stream for both arms. By §7.3 the tokens are chunk-size-invariant
within that process, and by §7.4 the render is faithful, so:

* the **candidate** arm is bit-for-bit what a shipped `cs7` build produces —
  the arm that has to be real;
* the **reference** arm is `cs25` geometry over that same stream, which is not
  what a `cs25` build would draw for that seed but *is* exactly what a `cs25`
  build would render if handed that stream (§7.3 again). It is the correct
  control: it holds content constant, which is the entire point.

### 8.6 Verdict

**The one-talker-run-per-pair trick IS valid for chunk size**, subject to one
design constraint (§7.5: generate in the candidate's process) and one named
tolerance (§7.4: a 2-frame terminal residual is reproducible only to −66 dBFS,
avoidable by fixture choice).

Consequences, using Commander's own framing:

* The audition collapses to Story 20.5's shape — **one round, ~14–16 trials,
  ~20–25 minutes**, instead of ~50–170 trials and 1.5–3.5 hours.
* Attribution is restored: within a pair, wording, prosody, pauses and duration
  are identical **to the sample**, so anything heard is caused by the geometry.
  Story 20.4 §17's take-to-take variance problem is not repealed — it is
  sidestepped, the same way Story 20.5 sidestepped it.
* Because it is cheap, **testing more than one candidate is affordable**:
  cs15 / cs10 / cs7 against one reference is ~3× the trials of a single pair
  set, i.e. roughly one longer sitting rather than five.

### 8.7 Residual risks, stated

1. **`n = 1` seed on the (d) length effect.** The `cs7`-keyed draw was 256
   frames against `cs25`-keyed 244 for the same seed — 4.9 % longer. With one
   seed there is no way to tell a coincidence from a systematic bias of the
   `cs7`-keyed build toward longer output. It does not affect audition
   validity under §7.5's design (both arms come from one draw), but it is a
   claim nobody should make in either direction yet. Cheap test: 5–10 seeds ×
   {cs25 key, cs7 key}, compare the length distributions.
2. **One host, one model.** Everything above is RTX 5090 / cc 12.0 /
   torch 2.10.0+cu128 / `Qwen3-TTS-12Hz-1.7B-Base` / bf16. (d) is a statement
   about compiled-kernel numerics and is exactly the kind of thing that varies
   by host — F6 (RTX 3060) would need its own check before any claim is made
   for that tier.
3. **The 2-frame residual non-determinism is unexplained**, only located. It is
   benign at −66 dBFS and it predates this story (it is present between two
   live runs), but it is now on the record and it has a falsifiable shape: if
   a future capture shows non-determinism on a residual larger than 2 frames,
   §7.4's mechanism is wrong.
4. **A bug was found and fixed in the check itself, not in production.** The
   first run of the offline renderer deadlocked at `cs7`: it filled the
   streamer queue before starting the worker, and the queue is bounded at
   `queue_max_factor × chunk_size` = 28, against 35 chunks. Story 20.5's
   fixture renderer has the same shape and never hit it because `cs25` gives
   maxsize 100 against 10 chunks. **Any Phase 2 fixture generator derived from
   `20-5-regen-audition-fixture.py` will deadlock at small chunk sizes unless
   it starts the worker first** — which is production's order anyway. Aborted
   run: `20-8-onetalker-viability-ABORTED-queue-deadlock.log`.

---

## §8. Phase 2 — AC #3b: narrowing to ONE candidate, on non-perceptual grounds

Three points are viable. Auditioning all three is affordable *only because* §7
made the round cheap, and "affordable" is not a reason to spend Commander's
ears. AC #3b asks for one primary candidate justified without reference to how
anything sounds.

### 8.1 The numbers that decide it

| point | seams (long) | TTFA(release) | Δ vs control | **marginal cost of the step** | decoder work | headroom above the floor |
|---|--:|--:|--:|--:|--:|--:|
| cs25 | 9 | 1,167.5 ms | — | — | 366 ms | 18.5 frames |
| cs15 | 16 | 740.9 ms | −432.8 | **60.9 ms saved per added seam** | 577 ms | 8.5 frames |
| **cs10** | **24** | **528.4 ms** | **−645.3** | **26.6 ms per added seam** | **840 ms** | **3.5 frames** |
| cs7 | 34 | 391.6 ms | −782.1 | **13.0 ms per added seam** | 1,135 ms | **0.46 frames** |

Four arguments, none of them about listening:

**1. The marginal rate collapses, and cs10→cs7 is where it falls off.**
Each step buys less per seam it adds: 60.9 → 26.6 → **13.0** ms. The last step
buys its 137 ms at **less than a quarter** the rate of the first. Put the other
way: **cs10 already captures 82.5 % of the total available win** (645 of 782 ms)
for 24 of the 34 seams.

**2. cs7 has no headroom above the watermark floor — literally none.**
The floor is 7 (§1). `cs7` clears it by **36.9 ms**, which is **0.46 of a
frame** — there is no integer geometry between `cs7` and failure. `cs10` clears
it by 276.9 ms, three-and-a-half frames.

That matters *specifically in this story* because **the floor already moved
once, during this story**: Story 20.1 §5.4 recorded 6, and §1 derived 7 on
current code, because Story 20.5 made `decode(N) == 1920·N` exact. The floor is
a function of three constants that have all moved within this epic — the
watermark (500 ms), the edge loss (555 samples), and samples/frame (1920).
Quantified: `cs10` survives the watermark rising to 776 ms, or the edge loss
growing by 6,645 samples (12×). `cs7` survives the watermark rising to 536 ms,
or the edge loss growing by 885 samples (1.6×). A qwen-tts pin bump that
changes the codec's edge geometry — Story 20.4 §11.2 found the documented value
was wrong once already — takes `cs7` below the floor and leaves `cs10` fine.

**3. Decoder-thread load, and the tier we have never measured.**
Per-chunk decode is flat in chunk size (§3.4a), so total decoder work scales
with chunk count: `cs7` is **35 % more than `cs10`** and 3.1× `cs25`. On this
host that is invisible — the producer ratio moves 0.539 → 0.567. But F6
(RTX 3060, sub-16 GiB) is **unmeasured**, its producer ratio is documented at
≈ 0.5, and it is the tier where the OFR-E gate has the least room. Taking the
step with the worst latency-per-seam rate *and* the largest throughput cost, on
the strength of measurements from the fastest host we ship to, is the trade
this epic keeps getting wrong.

**4. Two smaller things, recorded for completeness rather than weight.**
The probability that a generation ends on the one chunk shape where the decoder
is not bit-reproducible against itself (§7.4) is `1/N` — 14.3 % at `cs7`, 10 %
at `cs10`. And `cs7` is the only geometry that produced a native access
violation during Phase 1 (§4.4.1) — once, unreproduced, and explicitly not
attributed to `cs7`, but it is on the record and it is not on `cs10`'s record.

### 8.2 The counter-case, stated rather than skipped

`cs7`'s extra 137 ms is **26 % of `cs10`'s remaining TTFA**, and MyVoice's
streaming exists for Clear Comms, a voice-chat *interjection* feature
(`memory/clear_comms_purpose_framing.md`) where the front of the utterance is
exactly where latency hurts most. 137 ms is not noise, and if the seam lever
turns out to be inert the argument above is only about robustness.

It does not carry, for one reason: **the irreducible floor is ~140 ms**
(segment 1 ≈ 80 ms + segment 3 ≈ 60 ms), so `cs7` is buying the *last* 137 ms
of a 645 ms win at the worst available exchange rate, and paying for it in the
one currency — headroom — this story has already had to revalue once.

**Primary candidate: `chunk_size = 10`.**

### 8.3 What would make a SECOND candidate worth auditioning — decided now, not later

Stated before the round so the decision is made deliberately rather than by
drift. Read off the round's own scorecard:

| round-1 outcome | what it means | action |
|---|---|---|
| **Clean pass with margin** — zero blocking rows, reference preferred on ≤ 1 of 14 A/B trials, `equivalent` modal | the seam-count lever is demonstrably not binding at 24 seams | **`cs7` is worth ONE more round** (+16 trials, ~20 min, fixture machinery already built). It is the only condition under which the extra 137 ms is cheap to test |
| **Pass, but not clean** — zero blocking rows, reference preferred on ≥ 2 trials | the lever is live at 24 seams, so 34 is worse, not better | **ship `cs10`, test nothing further.** Do not spend a round confirming a worse point |
| **FAIL** — any blocking chunk-boundary defect on a candidate trial its reference does not carry | `cs10` does not ship | **fall back to `cs15`** (16 seams, still −433 ms, and the only sub-25 point with prior audition data — Story 20.4 round 4). One round on `cs15`, then stop regardless of outcome |
| **Control trial flags** — a preference or defect on the byte-identical control | the round's noise floor is above its signal | **discard the round**, do not re-scope. Re-run with a rested listener |

Note the asymmetry, and it is deliberate: a *good* result buys one more round;
a mediocre one buys none. The failure mode this table exists to prevent is
auditioning `cs7` because `cs10` was ambiguous — which is how Story 20.4 spent
four rounds.

### 8.4 The falsifiable prediction — RECORDED BEFORE THE ROUND

Reference = `chunk_size 25`, what ships today. Candidate = `chunk_size 10`.
16 trials: 7 utterances × 2 takes = 14 A/B, plus 2 byte-identical controls.

- **P1 (MAGNITUDE).** `equivalent` is modal, **≥ 10 of 16**. The mechanism:
  Story 20.5 removed the seam residual *at the cause* (head NRMSE 0.406 →
  0.0078, lag jitter 0 on every seam, edge loss 0) and Story 20.6's retirement
  removed the trim and the blend by construction, so each seam is now a
  state-continuous join rather than a splice between two independent
  renderings. The count rises 2.5×; the per-seam defect is the thing that was
  removed. **Falsified if ≤ 6 are equivalent.**
- **P2 (NO NEW HARM — BLOCKING, and it is the gate).** No chunk-boundary defect
  (`audible_seam`, `click_or_discontinuity`, `prosody_break_at_stitch`) on a
  candidate trial that its paired reference does not also carry. Both files in
  a pair are one take rendered two ways, so such a defect is caused by the
  geometry and nothing else. **Falsified by one.**
- **P3 (LOCATION).** Anything audible is on the LONG fixtures, which carry the
  most seams. **Falsified if a difference is heard on a short fixture but not on
  a long one** — that would say seam count is not the mechanism, which is the
  same sub-prediction Story 20.4 §17 falsified once already and the reason its
  `chunk_size ≈ 20` crossover estimate must not be quoted.
- **P4 (THE EMBARRASSING ONE).** Story 20.4's four rounds found `cs10`
  perceptually worse — blocking in rounds 2 and 3. The entire reopening rests
  on "Story 20.5 removed the cause." **If `cs10` flags a blocking seam defect
  again here, that mechanism argument is wrong**: state caching did not remove
  what actually made `cs10` worse, Story 20.5 §2's offline numbers (50–60× less
  seam error, zero lag jitter) do not describe what the ear responds to, and the
  geometry question is **closed for good — not retuned to `cs15`, closed.** It
  is stated first-class because it is the outcome that costs most, and because
  §0's GO verdict is more exposed to it than to anything else in this file.
- **P5 (THE OTHER DIRECTION, also embarrassing).** If the **candidate** is
  preferred on ≥ 4 trials, P1 is falsified in the exciting direction — and that
  is a finding needing an explanation, not a celebration. Nothing in this epic
  predicts that smaller chunks *improve* audio; if they do, the most likely
  reading is that the reference's longer decode windows accumulate codec drift
  that state caching does not fully remove, which would reopen Story 20.5's
  conclusion rather than confirm it.
- **P6.** **Latency is NOT under test here.** These are rendered files; nothing
  about TTFA is auditionable. The GUI capture (§9) is the only evidence for
  the shipped TTFA claim.

**Verdict gate (blocking, per AC #3):** FAIL if any chunk-boundary artefact is
flagged on a candidate trial that its paired reference does not also carry. A
defect on **both** files of a pair is upstream of the geometry — here
demonstrably, since they are the same take — and is recorded, not blocking.

---

## §9. Phase 2 — implementation, verification, and the operator hand-off

### 9.1 What changed

**One production constant. Nothing else in `src/`.**

| file | change |
|---|---|
| `src/myvoice/services/tts_streaming/codec_token_streamer.py` | `DEFAULT_CHUNK_SIZE` **25 → 10**, and the "THE COMMITTED GEOMETRY" record above it rewritten to carry the whole chain: why 20.4 reverted 10, what 20.5 and 20.6 changed underneath it, the 20.8 re-baseline table, why 10 and not 7, and the audition gate that is **not yet closed** |

`DEFAULT_LOOKAHEAD` is **untouched at 5** — it is the stateless fallback's
lookahead, and Story 20.6's whole design is that retirement is conditional and
never a constant edit.

Tests updated to follow the committed value (no behavioural change):

| file | rows |
|---|---|
| `tests/unit/services/tts_streaming/test_codec_token_streamer.py` | the geometry pin 25 → 10; and the watermark assertion strengthened from the **nominal** `N/12.5 >= 0.5` to the **exact** `N*1920 − 555 >= 12000` derived in §1. The nominal form agrees for every integer N but only by luck — it omits the edge loss |
| `tests/unit/services/test_lookahead_retirement.py` | three pins: the committed constant, `sum(resolve_streamer_geometry()) == 10`, and the kill-switch window `30 → 15` |
| `tests/unit/services/test_decode_window_geometry_coherence.py` | the priming-key row now asserts 10 (and `!= 25`, `!= 30`); the kill-switch row asserts 15 and was **renamed** off `..._reverts_to_30_...`, because a row asserting 15 under a name promising 30 is the exact drift this file exists to catch |
| `tests/integration/test_streaming_tts_smoke.py` | stale docstring only. `_expected_chunk_count` already reads the live constants, which is why the retune needed no edit here |

### 9.2 The threading — verified at all three sites, in both directions

`20-8-threading-verify.py` → `20-8-threading-verification.txt`. **All checks
pass.** Story 20.4 §1.1 found three D-25 drift sites, not the two Story 20.1
§5.4 predicted, so all three are exercised — and the kill switch is flipped,
because a constant edit that only moves the value forward proves nothing a
hard-coded literal equal to the new value would not also pass.

| site | check | result |
|---|---|---|
| **1** `resolve_streamer_geometry()` | state-carrying | `(10, 0)` → `decode_window_frames = 10` |
| **1** | **kill switch (the other direction)** | `(10, 5)` → **15** |
| **1** | `DEFAULT_LOOKAHEAD` untouched | 5 — it is the stateless path's lookahead |
| — | `CodecTokenStreamer()` follows the constant | `chunk_size = 10`; `apply_codec_state_geometry(True)` → lookahead 0, chunk size unchanged |
| **2** `engage_compile_optimizations` (via `model_registry`) | which inductor directory the compile actually reads | `TORCHINDUCTOR_CACHE_DIR = …\torch_compile_cache\ef8efd9237ad2a9f` = **the cs10 key** |
| **3** `warmup_compile_async` | which key priming WARMS | `reason='primed_cold'`, `meta.json` written into **`ef8efd9237ad2a9f`** |
| **2 ∧ 3** | **they agree** | priming warms exactly the directory engage reads |
| — | the OLD key is not disturbed | `205b94cf67d78d0a` (cs25) `meta.json` unchanged |

That last pair is the check the coordinator asked for specifically: the retune
is a new cache key, so first launch pays one cold compile, and what matters is
that **Story 20.3's priming warms the NEW key and not the old one**. It does —
`mark_warm` wrote into the cs10 directory, `is_warm` was `False` beforehand,
and the cs25 directory was left alone.

**Two false failures on the way there, recorded because they are the same class
of mistake this file exists to catch.** The first version of the verifier
computed the cache keys immediately after `service.start()` and reported three
failures. Both were artefacts of *ordering*, not of the threading:

* `service.start()` does not materialise the model — `model.model.name_or_path`
  is unset, so the key was computed from `model_id='unknown'` / `fp32` and
  named a directory nothing ever reads;
* `engage_compile_optimizations` only reaches `set_torchinductor_cache_dir` on
  the **first dispatch**, so site 2 is not observable until a generation has
  run.

A third followed: with those fixed, `warmup_compile_async` returned
`reason='no_priming_prompt'`. It resolves its priming text through the *active
profile's* cached `voice_clone_prompt` (Story 20.3 AC #2, in-memory lookup) and
a headless service has no profile manager. The verifier now supplies exactly
that one lookup and nothing else — the key computation, the `is_warm` check and
`mark_warm` all run for real.

Worth stating plainly: **each of these looked like a threading failure and was
not.** Reading the telemetry `reason` rather than inferring from the filesystem
is what separated them.

### 9.3 The exact bars (AC #4, first clause)

`20-8-exact-bars.txt` — all four pass at the retuned geometry:

| test | bar |
|---|---|
| `test_single_chunk_streaming_is_bit_for_bit_identical_to_forward` | single-chunk streaming == `decoder.forward`, **bit-for-bit** |
| `test_stitched_stream_reconstructs_the_whole_sequence_decode` | fp64 chunked == whole-sequence to ~1e-06 |
| `test_retired_lookahead_stitched_stream_reconstructs_the_whole_decode` | the same, at the retired geometry |
| `test_retired_lookahead_matches_the_lookahead_geometrys_output` | the two geometries post the same audio to 1e-06 |

### 9.4 Regressions (AC #4, second clause)

**Result: no streaming test moved. Nothing disappeared from the failure set.
The only difference between the two runs is one UI file that is
non-deterministically non-terminating on this machine — on both trees.**

#### 9.4.1 A plain `pytest tests/` does not complete on current `main`

This has to be said first, because it shaped the method. Two full-suite runs on
the **pre-change** tree stalled indefinitely — CPU pinned at a constant value,
no progress for 20+ minutes — at different points each time. `pytest-timeout`
is not installed in the portable interpreter, so the guard had to be external:
`20-8-suite-with-hang-guard.sh` runs each test directory in its own process
under `timeout`, and a directory that does not return has its **partial output
discarded** and is re-run file by file. Discarding matters: pytest is killed
before printing its summary line, so those tests would otherwise be counted
once from the partial output and again from the fallback.

The identical script produced both runs. That is what makes them comparable at
all.

#### 9.4.2 The counts

| | passed | failed | errors | hung files |
|---|---:|---:|---:|---:|
| **BEFORE** (pre-change tree) | 2,883 | 30 | 4 | **1** |
| **AFTER** (retune applied) | 2,921 | 49 | 5 | 0 |

#### 9.4.3 The identity diff — and the whole of the difference

* **In BEFORE but not AFTER: nothing.** Not one failure was removed.
* **In AFTER but not BEFORE: 19 — and all 19 are in a single file**,
  `tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py`.
  That is the file that **HUNG in the BEFORE run** and therefore contributed
  nothing to it: not its failures, not its ~38 passes, not one of its errors.
  The +38 passed / +19 failed / +1 error is that file, arriving.
* **No streaming test failed in either run.** The one grep hit on "chunk" is
  `test_session_lifecycle.py::TestAudioChunkPayloadStability::test_audio_chunk_field_set_unchanged`,
  which fails **identically in both** and is an `AudioChunk` dataclass-shape
  row, not a geometry row.
* The 49 failures sit in twelve files, all in the clusters
  `memory` and Story 20.6 §8 already record as pre-existing —
  `voice_design_studio` (30), `session_manager` (4), `voice_library_widget` (4),
  `settings_dialog`/clear-comms (2), `optimized_voice` (2), `emotion_tts` (2),
  `system_tray`, `voice_profile_manager`, `origin_gating`, `session_lifecycle`,
  `emotion_variants` (1 each). **None touches streaming.**

#### 9.4.4 The corroboration that makes AFTER the *better* run, not the worse one

**Story 20.6 §8 recorded the pre-existing failure set on the full suite as
exactly 49.** The AFTER run reproduces **49**. So AFTER captured the complete
pre-existing set; BEFORE was 19 short because a file hung, and the comparison
should be read that way round.

#### 9.4.5 What could not be made exactly comparable, stated rather than glossed

The clean thing would have been a BEFORE run in which that file also ran. It
was attempted and it is not achievable on this machine:

* run in isolation on the **pre-change** tree (source change stashed) — **hung**
  (no result in 400 s);
* run in isolation on the **post-change** tree — **hung** (no result in 400 s);
* run as part of its whole directory in the AFTER pass — **completed**, 538 s.

So the file is **non-deterministically non-terminating, and identically so on
both trees**. Its behaviour depends on how it is invoked, not on
`DEFAULT_CHUNK_SIZE`. A whole-directory re-run on the stashed tree was started
to close the last gap and was itself cut off by a harness time limit; the
partial log is not quoted.

**The honest statement of AC #4's second clause, therefore:** the failure set is
unchanged in identity — nothing was added except the contents of one file that
the BEFORE run could not execute, and nothing was removed — and the count
difference is fully attributed to that file. It is *not* a bare "count and
identity identical", and it is not claimed as one.

**This is pre-existing and belongs to somebody else.** The hangs are all
dialog/UI tests and the cluster grew with the settings-dialog and close-to-tray
work on `main`; Story 20.8 touched none of it. Worth raising separately: a
suite that cannot complete unattended is a problem for every story after this
one, and `pytest-timeout` in the portable interpreter would turn these hangs
into failures that can at least be counted.

### 9.5 The audition fixture — generated, checked, and AC #3a's four fixes verified in it

`20-8-regen-audition-fixture.py` → `20-8-perceptual-fixtures/` (32 WAVs +
truth table), `20-8-audition-manifest.json`, `20-8-fixture-generation.log`.

| | |
|---|---|
| A/B trials | **14** (7 utterances × 2 takes) |
| byte-identical control trials | **2** |
| seams, reference → candidate | **41 → 116** (**2.83×**) |
| lengths equal on every pair | **yes, 14/14** |
| worst within-pair level delta | **0.061 dB** (flag threshold 0.2; **not** normalised, because both arms share a take, so any level difference would be *caused by the change* and is a finding) |
| presentation balance | reference is trial A on **8**, candidate on **8** |

**The four AC #3a requirements, each observable in the output:**

1. **Captured at the candidate geometry.** Preflight printed
   `committed geometry = (10, 0) [capture runs here]` and refuses to run
   otherwise. The candidate arm is therefore bit-for-bit what a shipped `cs10`
   build emits (§7 (a)+(b)); the reference arm is `cs25` geometry over the same
   content.
2. **Worker before queue.** The candidate arm renders up to 25 chunks against a
   bounded queue of `4 × 10 = 40`, and the long fixtures at `cs10` reach 25 —
   the old fill-first order would still have survived here, but not at `cs7`,
   and the generator does not depend on that luck.
3. **No 1- or 2-frame terminal residual.** The rule fired **8 times** and the
   redraws are logged. Three are worth naming: `l-020` take 1 drew a 2-frame
   candidate residual **twice in a row** (232 and 242 frames), and `s-021` take
   1 drew a 1-frame residual **three times** (31 frames each — the same text at
   nearby seeds lands on the same length). Without the rule, four of the
   fourteen A/B trials would have carried the one chunk shape where the decoder
   is not bit-reproducible against itself. The §7 (a) caveat is now moot rather
   than argued away.
4. **Byte-identical controls.** Both assert-checked at generation time and
   measured at **−312.9 dB and −311.6 dB** waveform delta with **0.000 dB**
   level delta — i.e. exactly identical, as intended. The generator would have
   aborted rather than shipped a control that was not.

The A/B pairs measure ~**−40 dB** apart. That is expected and is the point: the
two arms genuinely differ at the seams, and there are 2.83× as many of them on
the candidate. Whether that is audible is the question the round exists to
answer, and §8.4's prediction says it should not be.

### 9.6 CONSOLIDATED OPERATOR HAND-OFF

**Everything Commander needs to do, in one place, in this order. About
65 minutes, mostly waiting.**

Both parts preflight themselves and stop with a clear message rather than
producing misleading data, so there is nothing to check by hand first.

---

#### Part 1 — the NFR3 audition (~25 min, Commander solo)

```
18_Story_20.8_AC3_Audition.bat
```

16 blinded A/B trials — 7 utterances × 2 takes, plus **2 byte-identical
control trials**. Replayable; the helper never says which arm is playing, and
it prints the unblinded verdict plus the prediction scorecard when the last row
is entered. Results append to `20-8-chunksize-audition.csv`; re-running skips
rows already recorded.

**What is being asked.** Only the chunk size differs: one arm cuts the stream
into 25-frame pieces (2.0 s), the other into 10-frame pieces (0.8 s), so the
second has ~2.5× as many joins. Both files in every pair are **the same
generation re-cut two ways** — identical words, timing and delivery, to the
sample — so anything heard is the cutting and nothing else.

**`equivalent` is the predicted answer on most trials.** That is not a
cop-out here; it is P1, and it is falsifiable (§8.4).

**Two trials (`ctl-020`) are byte-identical on purpose.** They are not a trick;
they set the round's noise floor. If a preference or a defect is reported on
one of them, §8.3 says discard the round and re-run rested — do not re-scope it.

**Blocking outcome:** any chunk-boundary defect flagged on a candidate trial
that its paired reference does not also carry. A defect on **both** files of a
pair is upstream of the geometry — demonstrably, since they are the same take —
and is recorded, not blocking.

**What happens next is already decided** (§8.3), and the helper prints it:
clean pass → `cs7` earns one more round; pass-but-not-clean → ship `cs10` and
test nothing further; fail → fall back to `cs15` for one round and then stop
regardless.

---

#### Part 2 — the GUI TTFA capture, BOTH ARMS, one sitting (~40 min)

```
16_Story_20.8_AC3_GUI_Capture.bat     arm B — chunk_size 10 (committed)
17_Story_20.8_CS25_Baseline.bat       arm A — chunk_size 25 (the control)
```

**Run 16_ first, then 17_, back to back, same machine, same cloned-voice
profile, same afternoon.** Five launches each; launch 1 of each arm is a
declared throwaway that pays that arm's one cold compile.

**Why both arms, and why this is not optional.** Phase 1's −645 ms is a
*headless* number; the shipped claim is a GUI-path claim. And Story 20.6 §12
established that quoting a GUI figure measured in another session is how this
epic produced a false conclusion once already — the same code read 46.66 vs
38.25 ms/frame two months apart. A `cs10` capture on its own could only be
compared to that, so it would be worth nothing.

**Arm A will be noticeably slower to first audio — by roughly 600 ms. That is
the measurement, not a fault.**

Arm A forces the geometry back to 25 **in-process**, through
`20-8-run-myvoice-at-cs25.py`, which rebinds the streamer's default and then
runs `main.py`. Nothing is edited; there is no source change to revert. Both
launchers write a manifest recording the geometry the process actually
resolved, and the scorer refuses to run if a manifest disagrees with the
declared arm.

**Per launch, the one thing that matters:** wait for *"Preparing TTS engine"*
to disappear before generating. Priming holds the request semaphore; generating
while the indicator is up measures queueing, not first-forward, and puts
840–1,383 ms of somebody else's work inside the number. Each launcher prints a
CHECK line afterwards saying whether that launch survived.

Then:

```
python310\python.exe _bmad-output\implementation-artifacts\20-8-compare-gui-arms.py
```

which prints, per class, segment by segment, arm B minus arm A — plus each
arm's per-frame talker cost and the cross-check that says whether the saving
really is the 15 fewer frames the geometry predicts, or something else.

---

#### What comes back to the session

The audition CSV, the ten capture CSVs, and the scorer's output. Those close
AC #3 and AC #3b. Nothing else is outstanding.

---

## §10. Artifacts

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
| `20-8-onetalker-viability.py` / `.json` / `.log` | §7's in-process check: claims (a), (a-ctl), (b), (c) |
| `20-8-onetalker-viability-ABORTED-queue-deadlock.log` | the first run, which found the bounded-queue deadlock (§7.7.4) |
| `20-8-crossprocess-tokens.py`, `xp-cs*.pt` | §7's fresh-process captures: claim (d) |
| `20-8-threading-verify.py` / `20-8-threading-verification.txt` | §9.2 — the three D-25 sites, both directions, and which key priming warms |
| `20-8-exact-bars.txt` | §9.3 — the four exact bars at the retuned geometry |
| `20-8-suite-with-hang-guard.sh` | §9.4 — the guarded suite runner (the only shape that terminates on current `main`) |
| `20-8-regression-{BEFORE,AFTER}.{log,summary,hung}` | §9.4 — the two runs and the identity diff |
| `20-8-regen-audition-fixture.py`, `20-8-perceptual-fixtures/`, `20-8-audition-manifest.json`, `20-8-fixture-generation.log` | §9.5 — the fixture and its checks |
| `20-8-l1-audition-helper.py` | the blinded audition helper |
| `20-8-run-myvoice-at-cs25.py`, `20-8-compare-gui-arms.py` | §9.6 — the reference-arm launcher and the two-arm scorer |
| `16_` / `17_` / `18_Story_20.8_*.bat` (repo root) | the three operator launchers |

---

## §10. Phase 2 operator results — AUDITION CLEAN, GUI CONFIRMS. 2026-09-14

### AC #3 audition — 16/16 `equivalent`, zero defects on either arm

Every trial, including both byte-identical controls, returned `equivalent` with
`none` on both arms. Zero blocking, zero shared, reference preferred **0** times.
Retuning `chunk_size` 25 → 10 is perceptually inert on this fixture — the fixture
built to the §7 (d) design, so the candidate arm is bit-for-bit what the shipped
build produces.

Against the pre-committed rule (§8.3): **clean pass** — 0 blocking, reference
preferred ≤ 1 of 14. Per that rule, `cs7` earns one round. See §10.3.

### AC #3 GUI — same sitting, `cs25` control forced in-process by `17_`

| long class | cs25 (arm A) | cs10 (arm B) | delta |
|---|---:|---:|---:|
| 1b prefill | 149.0 ms | 122.7 ms | −26.2 |
| **2 talker → first emit** | **1,298.4 ms** | **471.9 ms** | **−826.5 ms (−63.7 %)** |
| 3 decode | 135.6 ms | 133.7 ms | −1.9 |
| 4 cushion | 20.2 ms | 21.9 ms | +1.7 |
| **TOTAL** | **1,626.8 ms** | **767.5 ms** | **−859.3 ms (−52.8 %)** |
| producer ratio | 0.65 | 0.675 | still ≪ 1.0× |
| chunks | 10 | 24 | — |

| short class | cs25 | cs10 | delta |
|---|---:|---:|---:|
| 2 talker | 1,453.8 ms | 509.0 ms | −944.8 |
| **TOTAL** | **1,828.3 ms** | **840.7 ms** | **−987.5 ms (−54.0 %)** |

### The P3 falsifier did not fire — the mechanism holds

Arm A per-frame talker cost: **51.94 ms**. Fifteen fewer frames predicts
−779.1 ms on segment 2; observed −826.5 ms; **residual −47.4 ms**. Near zero and
marginally better than predicted. First emit is gated on the frame threshold, by
the route the story claimed.

### Session drift, again, and why the same-sitting control mattered

This sitting's `cs25` reads 1,626.8 ms long against Story 20.6 §12's 1,362.4 ms
twelve days earlier — a 19 % drift for the *same* geometry on the *same* machine.
Had the delta been scored against 20.6's control it would have read −595 ms
instead of −859 ms. The number that is real is the within-sitting delta, and it
is the one reported.

### §10.3 — On the round `cs7` has earned

The §8.3 rule was set before any trials to stop this becoming Story 20.4. It says
a clean pass earns `cs7` **one** round. It does not say that round must be spent.

For spending it: the headless curve puts `cs7` at −137 ms below `cs10`, and a
one-talker audition costs ~25 minutes.

Against: `cs7` sits **0.46 frames** above the watermark floor — it survives the
watermark rising to 536 ms or the edge loss growing 1.6×, where `cs10` survives
776 ms and 12×. The floor is a function of three constants that have all moved
within this epic, once inside this story. `cs7` buys the last slice of the win at
the worst marginal rate, in the one currency this story already had to revalue.

**`cs10` ships on this evidence regardless.** Whether to spend `cs7`'s earned
round is Commander's call, made with the fragility stated.
