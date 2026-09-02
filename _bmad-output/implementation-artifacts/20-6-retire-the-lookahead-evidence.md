# Story 20.6 — Retire the Lookahead and the Post-Decode Trim: evidence

Phase ⊥-Polish-3. Follow-up F1 from Story 20.5. Branch `story/20-6-retire-lookahead`.

Two acceptance criteria need a human at the keyboard — AC #3's GUI capture and
AC #4's audition. Everything else is closed here. §7 is the consolidated
operator hand-off; nothing else in this file asks for Commander's time.

---

## 1. What changed, and why AC #2 shaped it

AC #2 is the constraint the whole design hangs off, so it is stated first.

`_build_true_stream_decode_fn` reaches the stateless cold-state adapter three
ways: the `MYVOICE_CODEC_STATE_CACHE` operator kill switch, `probe_decoder`
refusing a decoder graph it has not verified, and the numerical self-test
failing on the loaded weights. On that path every chunk is still an independent
rendering of an overlapping token span, so **the lookahead, the post-decode
trim and the Story 20.4 seam blend are the only seam handling it has** — and
the last two are both gated on `lookahead > 0`. Editing `DEFAULT_LOOKAHEAD` to
0 would have removed all three at once, from the path a user reaches precisely
when something else has already failed, and no test of the state-cached path
would have noticed, because the state-cached path *wants* that behaviour.

So the retirement is conditional, by the producer-declares / consumer-acts
shape Story 20.5 used for the consumer crossfade. One declaration, two
consumers:

```
decode_fn.carries_codec_state          # the declaration (unchanged, 20.5)
  ├─ self._progressive_stream_continuous     # consumer 1: the crossfade (20.5)
  └─ streamer.apply_codec_state_geometry()   # consumer 2: the lookahead (20.6)
```

`DEFAULT_LOOKAHEAD` is still **5**. `DEFAULT_CHUNK_SIZE` is still **25**.

### The rule, in one place

`codec_token_streamer.effective_lookahead(carries_codec_state, lookahead=None)`
— a pure function returning `0` under carried state and the configured
lookahead otherwise. Everything that needs the answer calls it.

`CodecTokenStreamer.apply_codec_state_geometry(carries_codec_state)` is the
consumer half. It sets `lookahead` and `_chunk_with_lookahead` together, from
the lookahead the streamer was **constructed** with — so it is reversible and
idempotent, and no sequence of kill-switch flips can accumulate a geometry that
disagrees with the flag in force. It refuses to run mid-generation (buffered
tokens or queued chunks), the same caller contract `reset()` carries.

### Files touched

| File | Change |
|---|---|
| `services/tts_streaming/codec_token_streamer.py` | `RETIRED_LOOKAHEAD`, `effective_lookahead()`, `apply_codec_state_geometry()`, `_configured_lookahead`; docs |
| `services/tts_streaming/torch_runtime.py` | `resolve_streamer_geometry()` — the single derivation point for D-25's `decode_window_frames`; `engage_compile_optimizations` resolves its `None` defaults through it |
| `services/tts_streaming/__init__.py` | export `resolve_streamer_geometry` (append-only) |
| `services/model_registry.py` | compile call site derives via `resolve_streamer_geometry()` |
| `services/qwen_tts_service.py` | builds the stateful decoder in the retired geometry; dispatch calls `apply_codec_state_geometry`; `warmup_compile_async` key derives via `resolve_streamer_geometry()` |
| `services/tts_streaming/streaming_decoder.py` | docs (§2) + one construction-time guard against a decode_fn/streamer window mismatch (§10.6) — no change to the decode path |
| `services/tts_streaming/codec_state_cache.py` | **not modified** — see §3 |

---

## 2. The trim and the blend are removed *by construction*, not by a second edit

`streaming_decoder.py` has no behavioural change. Its predicate already reads

```python
is_full_window = (n_frames >= self._chunk_size + self._lookahead
                  and self._lookahead > 0)
```

With the lookahead retired that is False for every chunk, and three things
follow without a branch:

* the splice/trim arm is never entered, so `pcm_segment = pcm_full` — **the
  worker posts the whole decode, untrimmed.** Correct: with no overlap there is
  nothing to trim;
* `next_overlap` is never assigned, so `_pending_overlap` stays `None` and
  `_apply_overlap_add` returns its input — **the Story 20.4 seam blend is
  gone.** It cross-fades the retained lookahead tail into the next chunk's
  head; with no lookahead there is no tail. This is a dependent, not a bundled
  second change, and it is not a decision this story took;
* `_codec_state_frames += n_frames`, which still mirrors the decoder's own
  commit rule (§3).

The length identity is untouched — first decode of a session still pays the
555-sample edge loss, every later one returns exactly `1920*N` — so
`geometry_ok` and the `decode_geometry_unverified` trip-wire keep working at
the new window. That is pinned by
`test_retired_lookahead_keeps_the_geometry_trip_wire_live`.

Story 20.5 had already measured the blend as inert under carried state (the two
sides were bit-identical), so nothing audible is traded away here either.

---

## 3. The two-pass decode collapses without touching `codec_state_cache.py`

Story 20.5's wrapper is the foundation this stands on and is not modified. The
collapse is a consequence of the *argument* the dispatch builds it with.

`StatefulCodecDecoder.__call__`:

```python
commit = (self._commit_frames
          if n_frames >= self._window_frames and self._window_frames > self._commit_frames
          else n_frames)
```

Built with `lookahead = 0` the wrapper gets `window_frames == commit_frames`,
the guard `window_frames > commit_frames` is False, and `commit = n_frames` on
every chunk — so the snapshot / decode-the-lookahead-on-the-snapshot / restore
second pass is never entered. `build_stateful_decode_fn` already accepted
`window_frames == commit_frames` (only `<` is rejected).

Counted rather than inferred, on a real (tiny) `Qwen3TTSTokenizerV2Decoder` in
CI — `test_retired_lookahead_decode_is_single_pass`:

| geometry (test uses commit 10) | `CodecStreamState.snapshot` calls over 3 chunks |
|---|---|
| `window = commit + lookahead` (pre-20.6) | 3 |
| `window == commit` (retired) | **0** |

And the worker's advance still agrees with the decoder's commit at the new
geometry — `test_retired_commit_predicate_still_matches_the_decode_fn`.

---

## 4. AC #1's second clause: does the Story 20.4 threading carry 30 → 25?

**Not by itself, and that is the finding the clause exists to produce.**

Story 20.4 made three sites derive `decode_window_frames` from
`codec_token_streamer.DEFAULT_CHUNK_SIZE + DEFAULT_LOOKAHEAD` instead of
restating literals. That threading carries a change *to the constants*
automatically — which is what the `chunk_size` revert exercised, and it worked.
It cannot carry this change, because AC #2 forbids changing the constants: the
sum stays 30 whatever the decode path does.

So the derivation itself had to become conditional, in one place —
`torch_runtime.resolve_streamer_geometry()` — and the three sites now read it:

| site | before | after |
|---|---|---|
| `torch_runtime.engage_compile_optimizations` (`None` resolution) | `DEFAULT_CHUNK_SIZE`, `DEFAULT_LOOKAHEAD` | `resolve_streamer_geometry()` |
| `model_registry._load_model_sync` (the call site Story 20.1 §5.4 named) | passes both constants | passes both values from `resolve_streamer_geometry()` |
| `qwen_tts_service.warmup_compile_async` (the priming cache key) | sums both constants | `sum(resolve_streamer_geometry())` |

Verified, both arms:

* **runtime** — `decode_window_frames` is **25** with the kill switch unset and
  **30** with it set, at all three sites; the warm-path priming key equals the
  engage-path key in both regimes
  (`test_decode_window_geometry_coherence.py`, four rows, two of them new);
* **static** — no source file sums `DEFAULT_CHUNK_SIZE + DEFAULT_LOOKAHEAD` any
  more (`test_no_source_file_re_derives_the_compile_window_from_raw_constants`),
  and the `model_registry` call site is pinned to the single derivation point
  rather than to literals or to raw constants.

**Cold compile.** The key moves, so one cold compile is expected on first
launch. It is the *correct* invalidation: the decode call shape genuinely
changed. And priming warms the new key rather than the old one, because
`warmup_compile_async` and `engage_compile_optimizations` both resolve through
the same function — pinned by `test_warmup_key_and_engage_key_agree`, which now
runs against the conditional value.

**A note on what `decode_window_frames` actually does.** In the shipping
configuration `compile_mode="reduce-overhead"`, and the fork's
`Qwen3TTSTokenizerV2Model.enable_streaming_optimizations`
(`modeling_qwen3_tts_tokenizer_v2.py:1244-1250`) *skips* the manual
`capture_cuda_graph(window_size=decode_window_frames)` in that mode. So the
value's only live effect today is as a compile-cache key dimension: it decides
which inductor directory priming warms, not what gets compiled. That is why
resolving it from the kill switch at model-load time is safe even though the
other two fallback gates are only knowable at dispatch time — a mismatch costs
a cache miss, not wrong audio. This is recorded in the function's docstring
rather than only here.

---

## 5. Measured (AC #3, the headless half)

`20-6-lookahead-bench.py` → `20-6-lookahead-bench.json`. RTX 5090, bf16,
`tts_compile="auto"`, decoding the **same** token sequences Story 20.5 Phase 1
captured (reused, not redrawn, so no talker draw enters the number). 5 timed
reps per arm after a discarded warm-up, CUDA-synchronised per chunk.

| utterance | frames | ref per-chunk | cand per-chunk | saving | ref total | cand total |
|---|---|---|---|---|---|---|
| l-020 | 231 | 23.39 ms | 11.82 ms | **+11.57 ms** | 223.1 ms | 118.7 ms |
| l-021 | 176 | 23.29 ms | 11.85 ms | **+11.44 ms** | 153.1 ms | 95.3 ms |
| m-020 |  44 | 17.15 ms | 11.45 ms | **+5.70 ms** |  34.8 ms |  23.1 ms |

**The single-pass decode more than recovers Story 20.5's tax.** 20.5 measured
the two-pass cost at +7–10 ms/chunk; the median recovery here is **+11.4 ms**,
and per-chunk decode time roughly *halves* (23.4 → 11.8 ms on the long
fixtures). It over-recovers because the second pass was not only an extra
launch: it also decoded 5 extra frames, so the saving is the snapshot/restore
*plus* 20 % less decode work per chunk.

m-020's smaller saving is the residual-flush effect: at 44 frames it is one
full window plus a short residual, and the residual never paid the two-pass
cost in either arm.

**First-emit threshold.** 30 frames → 25. Five fewer talker steps before any
audio can be decoded. That is the mechanism behind the TTFA claim; converting
it to milliseconds is the GUI capture's job (§7), because the consumer cushion
is only real there.

**Not measured here:** the producer emit/drain ratio and the OFR-E `< 1.0×`
gate. Both are consumer-side and come out of the same CSV capture as the GUI
TTFA run — see §7, where they are one request rather than two.

**Output equality on the real model.** Posted sample counts are identical in
all three utterances and equal to a whole-sequence decode (`1920*N - 555`).
Waveform NRMSE between arms is exactly **0.0** where the residual flush lands
on the same frame, and **1.1e-03** (≈ −59 dB) on l-021, where it does not —
7 chunks in the reference against 8 in the candidate, so the last kernel
launches have different shapes and bf16 rounds differently. That is a float
floor, not a difference in what the decoder was asked to produce.

---

## 6. AC #4 — fixture generated, prediction recorded, waiting only on the ear

`20-6-regen-audition-fixture.py` → `20-6-perceptual-fixtures/` (16 trials).
`20-6-l1-audition-helper.py` drives the blinded round.

* **reference** = what ships today: state caching + gated (0-sample) consumer
  crossfade + lookahead 5;
* **candidate** = the same, lookahead retired.

**One variable.** Both arms carry state caching, chunk_size 25 and the same
gated crossfade. The blend is a dependent, not a second arm (§2).

**Token reuse is still available — verified, not assumed.** AC #4 asked whether
one talker run per pair survives a chunking change. It does: the chunking is
downstream of generation, so the fixture captures one run, recovers the flat
frame sequence (chunk *k* starts at frame `chunk_size * k` whatever the
window), and re-slices it per arm. Both files in a pair therefore have
identical wording, prosody and duration, and there is **no take-to-take
variance inside a pair** — Story 20.4 §17's variance caveat does not apply. The
second take samples the content lottery, not arm variance.

**Blinding is balanced, not coin-flipped.** Story 20.5's generator flipped a
coin per trial; on 16 trials that produced a 12/4 split — the reference in
position A three times as often as the candidate. Position bias in A/B
listening is real, and on a round whose predicted answer is "equivalent" it
would be the largest nuisance variable left. The assignment is now **8/8**,
shuffled from a fixed seed, so it stays reproducible from the generator and
uninferable from listening order. (The two control trials landed on opposite
orientations, which is a bonus rather than a design.)

**Zero-seam control.** `ctl-020` ("Say that again.", 15–16 frames) produces one
chunk in *both* geometries, so the two files are byte-identical — asserted by
the generator, confirmed on both takes. Any preference or defect reported there
is a property of the listening and calibrates the rest of the round.

### Measured before the round (mechanism only — offline metrics do not gate NFR3)

Identical length on **16/16** pairs. Worst within-pair level delta **0.005 dB**.
Waveform difference, candidate against reference:

| trial | ref/cand chunks | max Δ (int16) | NRMSE | dB |
|---|---|---|---|---|
| ctl-020-t1 / t2 | 1 / 1 | 0 | 0 | — (byte-identical) |
| s-020-t1 | 2 / 2 | 0 | 0 | — (byte-identical) |
| s-022-t1 / t2 | 2 / 2, 3 / 3 | 4 | 4.0e-05 | −88 |
| s-021-t1 | 2 / 2 | 8 | 7.4e-05 | −83 |
| m-021-t1 | 2 / 2 | 61 | 2.6e-04 | −72 |
| s-020-t2 | 2 / 2 | 46 | 2.5e-04 | −72 |
| m-020-t1 / t2 | 2 / 2 | 60 / 44 | ~4.0e-04 | −68 |
| m-021-t2 | 2 / 2 | 113 | 4.3e-04 | −67 |
| l-021-t1 / t2 | 7 / 7 | 123 / 176 | ~5.8e-04 | −65 |
| l-020-t1 / t2 | 10 / 11, 10 / 10 | 116 / 242 | ~6.0e-04 | −64 |
| **s-021-t2** | **1 / 2** | **272** | **7.2e-03** | **−42.8** |

Median −70 dB. The outlier is `s-021-t2`, and it is the informative one: at 29
frames the *reference* never reached its 30-frame threshold and decoded the
whole utterance as one residual, while the candidate reached its 25-frame
threshold and split it 25 + 4. That is the largest structural difference the
change can produce, and it is a residual-flush effect, not an interior seam.

### The prediction (recorded before the round; also in the helper's docstring)

* **P1 (magnitude).** `equivalent` is modal and near-total, ≥ 12 of 16.
  Falsified if ≤ 8 — a far larger effect than −70 dB can produce, meaning
  something else moved.
* **P2 (no new harm — blocking).** No chunk-boundary defect on any candidate
  trial that its paired reference does not also carry. Falsified by one.
* **P3 (location).** Any audible difference should be at the **end** of an
  utterance, where the residual flush splits differently; `s-021-t2` is the
  measured candidate. Falsified if differences are reported at interior seams.
* **P4 (the embarrassing one).** If the candidate is heard as **worse at the
  interior seams**, the diagnosis is wrong: the Story 20.4 blend was doing real
  perceptual work under carried state after all, Story 20.5's "the two sides
  are bit-identical" measurement does not describe what ships, and the
  retirement must be **reverted** rather than tuned — which would also re-open
  Story 20.5's Phase 4 conclusion.
* **P5.** Latency is not auditionable in rendered files. The GUI capture is the
  only evidence for TTFA.

The helper scores P1, P2 and P4 automatically and prints the control
calibration; P3 and P5 are read by hand from the notes.

---

## 7. OPERATOR HAND-OFF — one sitting, two things

Both need a human. Do them in this order; the whole sitting is ~35–45 minutes,
of which ~20 is listening.

### 7a. GUI capture — AC #3 (~25 min, mostly waiting)

```
13_Story_20.6_AC3_GUI_Capture.bat
```

New launcher, modelled on `11_Story_20.4_AC6_GUI_Capture.bat`. It writes
`20-6-gui-r0N.csv` (Story 20.3's and 20.4's captures are the baselines and are
left untouched), and it preflights three things before letting you start: that
`tts_compile` is `auto`, that the constants are still 25 + 5, and — the one
that matters — that `resolve_streamer_geometry()` returns `(25, 0)`, i.e. the
retirement is actually live in that environment and `MYVOICE_CODEC_STATE_CACHE`
is not set to a disabling value. Without that third check the whole run could
silently re-measure the pre-20.6 build.

`11_` will *refuse* to run — its preflight demands the reverted 10 + 5 geometry
— and would overwrite Story 20.4's CSVs if it did. Do not use it.

**Six launches. Launch 1 is a throwaway** — it pays the one expected cold
compile for the new `decode_window_frames=25` cache key (§4), so its
"Preparing TTS engine" takes noticeably longer. Do both generations in it
anyway; the aggregator drops it.

Per launch: cloned voice active, wait for "Preparing TTS engine" to clear,
generate the **long** utterance and **let it finish playing**, then the
**short** one and let that finish, then close with the X. Texts are in
`20-4-gui-utterances.txt` — the same ones Story 20.4 used, so the comparison
holds. Letting playback finish is what produces the producer-ratio data;
Story 20.3's captures lack it because the app was closed after chunk 0.

Then aggregate:

```
python310\python.exe _bmad-output\implementation-artifacts\20-4-aggregate-gui.py --glob "20-6-gui-r*.csv" --skip-first-launch
```

Three numbers come out of this and nothing else can produce them:

1. **TTFA against 1,353 ms** (the Story 20.3 §4.1 baseline), short and long.
   Expected direction: down by roughly 190 ms, subject to measurement.
2. **Producer emit/drain ratio**, and the **OFR-E `< 1.0×` gate** confirmed.
3. Whether the short class still lands on the `residual_flush` first-emit path
   — the threshold moved from 30 frames to 25, so a short utterance between 25
   and 29 frames now reaches the `threshold` path it used to miss.

**Read the CSVs grouped by `session_id`** (Story 20.3 §4.1a). The priming
generation emits its own boundaries *first*; a first-match join produces
nonsense. `20-4-aggregate-gui.py` already groups correctly — the trap is only
live if the numbers are read by hand.

### 7b. NFR3 audition — AC #4 (~20 min, Commander solo)

```
14_Story_20.6_AC4_Audition.bat
```

16 blinded trials, A/B, replayable. The fixture is already generated; the
helper never says which arm is playing and prints the unblinded verdict plus
the prediction scorecard when the last row is entered. Results append to
`20-6-lookahead-audition.csv`; re-running skips rows already recorded.

Two of the sixteen (`ctl-020`, both takes) are **byte-identical on purpose**.
They are not a trick — they set the round's noise floor.

**The expected answer is `equivalent` on most trials**, which is why P1 is
stated as a magnitude prediction rather than a hope. The round is asking
whether a −70 dB offline measurement is right about what the ear does; twice in
this epic an offline metric was not.

**Blocking outcome:** any chunk-boundary defect flagged on a candidate trial
that its paired reference does not also carry. Both files in a pair are the
same take, so such a defect is caused by the decode. If it lands at an
**interior** seam, that is P4 — revert the retirement, do not tune it.

---

## 8. Regression status (AC #5)

* **The exact bars still hold.** Single-chunk streaming == `decoder.forward`
  bit-for-bit; fp64 chunked == whole-sequence to 1e-06 — both re-asserted at
  the retired geometry as well as the pre-20.6 one
  (`test_retired_lookahead_stitched_stream_reconstructs_the_whole_decode`).
* **The two geometries post the same audio** — asserted to 1e-06 in float64 on
  the real (tiny) decoder in CI
  (`test_retired_lookahead_matches_the_lookahead_geometrys_output`), which is
  what makes §6's audition a test of a stated prediction rather than of an
  unknown.
* **The stateless path is untouched.**
  `test_stateless_path_is_untouched_by_the_state_cached_geometry` and the whole
  pre-20.5 block still run at lookahead 5.
* **Story 16.5's cooperative-cancel chain is unaffected** — no change to
  `_cancel_event`, the drain, or the three `reset()` points. The cancel and
  reset rows in `test_streaming_decoder.py` pass unchanged.
* **`DEFAULT_CHUNK_SIZE` stays 25**, pinned by
  `test_the_module_constant_is_not_globally_retired` in addition to the
  existing Story 20.4 row.

### Suite results

Full suite, `python310\python.exe -m pytest tests/ -q -p no:randomly -rf`, run
twice on this machine — once with the story's `src/`, `tests/` and `tools/`
changes stashed (the `main` state) and once with them applied:

| | passed | failed | errors |
|---|---|---|---|
| before (stashed to `main`) | 2,852 | 49 | 4 |
| after (this branch) | 2,883 | 49 | 4 |

**The pre-existing failure set is unchanged in count and in identity** — a
sorted diff of the two `FAILED` lists is empty in both directions. All 49 are
in `voice_design_studio`, `voice_library_widget`, `settings_dialog`,
`session_manager`, `emotion_tts`, `optimized_voice`, `origin_gating` and
`system_tray`; none touch streaming, and none moved.

The +31 passed are this story's new rows. (31, not 32: the "after" run
collected `test_lookahead_retirement.py` at 16 rows, before the last row was
added. That file is verified separately at **17/17**, and the focused streaming
suites — `tts_streaming/` plus the three service-level files — run **263
passed**, up from 231 on `main`.)

---

## 9. Tests added

| File | Rows |
|---|---|
| `tests/unit/services/test_lookahead_retirement.py` *(new)* | 17 — the rule, the streamer's reversibility and mid-generation guard, the three source invariants, the two decode-fn build paths, the conditional compile geometry, and one end-to-end row proving the fallback keeps lookahead + trim + blend **together** |
| `tests/unit/services/tts_streaming/test_streaming_decoder.py` | +6 — the retired geometry posts the whole decode, retains no tail, stays time-contiguous, totals a whole-sequence decode, keeps the commit predicate in step, keeps the trip-wire live |
| `tests/unit/services/tts_streaming/test_codec_state_cache.py` | +6 — the retired geometry is accepted, is single-pass (snapshot count 0), reconstructs the whole decode to 1e-06, matches the lookahead geometry's output; and the worker refuses a window mismatch, with the desync it refuses demonstrated directly |
| `tests/unit/services/test_decode_window_geometry_coherence.py` | +3, 2 updated — the priming key follows 30 → 25, reverts to 30 under the kill switch, and no source file sums the raw constants |
| `tests/unit/services/tts_streaming/test_torch_runtime.py` | 1 updated — a third arm proving the compile window is conditional |

### The three source-derived invariants (AC #2's last clause)

A runtime test proves only that the generation *it* ran was right. Story 20.5's
staleness trap — a declaration set on one path and inherited by the next —
needs an invariant over the source:

1. **every `CodecTokenStreamer` construction site declares its geometry in the
   same function.** A path that constructs one and never declares would emit at
   whatever the module constants say, regardless of which decode path it built.
2. **nothing outside `codec_token_streamer.py` assigns `.lookahead` or
   `._chunk_with_lookahead`.** One reversible door, so a half-retired geometry
   is unreachable: a direct `streamer.lookahead = 0` sets one field and not the
   other, and the worker — which snapshots `streamer.lookahead` — would then
   believe there was nothing to trim while 30-frame chunks arrived.
3. **`DEFAULT_LOOKAHEAD` is still 5, and `DEFAULT_CHUNK_SIZE` still 25.** The
   global-change failure mode, named directly.

---

## 10. Disagreements and things worth saying plainly

1. **AC #1's "carries 30 → 25 automatically" is not what happened, and could
   not have been.** The Story 20.4 threading propagates a change to the
   constants; AC #2 forbids changing them. Three one-line edits were needed to
   move the *derivation* behind a conditional. The verification the clause
   asked for is what surfaced this — recorded in §4 rather than quietly
   satisfied by a different reading.

2. **The TTFA estimate of "roughly 190 ms" may be optimistic.** It assumes the
   talker leg scales linearly at ~38 ms/frame. Five frames is a real saving,
   but the short-utterance class may see a *different* shape of win: the
   first-emit threshold moving from 30 to 25 frames means utterances of 25–29
   frames now reach the `threshold` dispatch path instead of waiting for
   `residual_flush`, which is a step change rather than a 190 ms shave. The GUI
   capture should be read for that, not only for the mean.

3. **The producer-throughput win is larger than the story predicted.** Story
   20.5 costed the two-pass tax at +7–10 ms/chunk; the measured recovery is
   +11.4 ms median, because the second pass also decoded 5 extra frames. Per-
   chunk decode time roughly halves. This is the number that matters most for
   the chunk-size reopen (F2), where the tax scaled with chunk count.

4. **Retiring the lookahead is not free of *all* risk, and the risk is at the
   end of an utterance, not the middle.** The residual flush splits differently
   — `s-021-t2` in the fixture is 1 chunk in the reference and 2 in the
   candidate. Every interior seam is unaffected. P3 exists to catch this being
   backwards.

5. **The kill switch now changes the compile-cache key.** Flipping
   `MYVOICE_CODEC_STATE_CACHE` between launches costs a cold compile in each
   direction. That is correct — the two regimes decode different window shapes
   — but it is a new operational cost worth knowing before an audition that
   flips the switch to generate both arms. (This round's fixture does not; both
   arms come from one build and one model load.)

6. **The change introduces exactly one silent-wrong-audio path, and it is now
   guarded.** A state-carrying decode_fn is built for one window width and
   commits at the splice implied by it. Pair it with a streamer emitting a
   *different* width and nothing raises: a decoder built for 25 handed a
   30-frame chunk fails its own `window_frames > commit_frames` guard, commits
   all 30, and the next chunk resumes five frames in its own future — 400 ms of
   speech skipped at every seam, with the length identity still satisfied so
   the `decode_geometry_unverified` trip-wire stays quiet. That is the Story
   20.4 defect class exactly, and no per-chunk check can see it.

   The dispatch cannot produce it (both sides come from one
   `carries_codec_state` read, pinned by the source invariants), but any other
   caller could. `StreamingDecoderWorker.__init__` now compares the decode_fn's
   window against the streamer's and raises. Soft on the attribute — a wrapper
   that does not expose its window is simply not checked — and hard on the
   answer, because a mismatch is provable rather than heuristic.
   `test_the_mismatch_the_worker_refuses_really_does_desync` demonstrates the
   desync directly, so the guard is justified in executable form.

   **Consequence worth knowing:** Story 20.5's fixture generators
   (`20-5-regen-audition-fixture*.py`) pair a lookahead-5 streamer with
   `_build_true_stream_decode_fn`, which now returns a retired decoder. They
   will raise rather than silently render desynced audio. They are working
   scripts for a closed story; if either is ever re-run, add
   `streamer.apply_codec_state_geometry(...)` the way `20-6-regen-audition-fixture.py`
   does.

7. **`decode_window_frames` is currently inert at runtime** (§4). It is worth
   recording that the D-25 invariant is, today, protecting a cache key rather
   than a CUDA-graph shape. If the fork ever stops skipping
   `capture_cuda_graph` under `reduce-overhead`, the resolution point built here
   becomes load-bearing for correctness, not just for cache hits.

---

## §11. Operator results, 2026-09-01 — AC #4 PASSES cleanly; AC #3's TTFA claim is NOT substantiated

### AC #4 — audition: unanimous, zero defects

16 blinded trials, including two byte-identical controls. **Every trial returned
`equivalent`, with `none` recorded on both arms.** Zero blocking, zero shared,
zero preference either way.

That is the predicted outcome and a clean pass: retiring the lookahead changes
nothing audible. Worth noting the reference arm also scored clean on the long
fixtures, where earlier rounds flagged it — consistent with Story 20.5 having
removed the underlying cause.

### AC #3 — the ~190 ms TTFA win did not appear

Retirement was live in every run (`myvoice.log`: *"TRUE_STREAM geometry:
chunk_size=25 lookahead=0 (carries_codec_state=True)"*), so this is not a
configuration miss.

Two of ten user generations carried an anomalous **840 ms / 1,383 ms** segment-1a
prefill against ~2 ms everywhere else — almost certainly generating before
priming released the request semaphore. Excluded as operator-procedure outliers,
which is stated rather than silently dropped.

Clean long-utterance results against the Story 20.3 baseline:

| | baseline (20.3) | now (20.6) |
|---|---:|---:|
| segment 1b (first forward) | 192.5 ms | **89.7 ms** |
| segment 2 (talker to first emit) | 1,147.5 ms | 1,125.2 ms |
| segment 3 (first decode) | 89.2 ms | 128.9 ms |
| **TOTAL** | **1,353.4 ms** | **1,364.4 ms** |

**Segment 2 is the one that should have moved and did not.** First emit now
requires 25 frames rather than 30, so it should have fallen by roughly a sixth —
~190 ms. Instead it is flat, which means **per-frame talker time rose from
~38.2 ms to ~45.2 ms**, offsetting the five frames saved.

### Why this cannot be attributed yet — a measurement gap

**There is no post-20.5, pre-20.6 GUI baseline.** Story 20.5's verification was
headless; the last GUI capture was Story 20.3's, taken *before* codec state
caching. So the 1,353 → 1,364 ms comparison spans **two** stories, and 20.5's and
20.6's TTFA effects cannot be separated from each other with the data in hand.

The headless decode measurement is unaffected and stands: per-chunk decode
**23.4 → 11.8 ms**, roughly halved, over-recovering Story 20.5's +7–10 ms tax.
That is a real, independently measured throughput win.

### The cheap experiment that would settle it

`MYVOICE_CODEC_STATE_CACHE=0` restores stateless decoding **and** the lookahead —
i.e. exactly the pre-20.5 geometry — on today's code and today's machine. A
kill-switch A/B in one sitting isolates 20.5+20.6 combined against the pre-both
baseline without the cross-session, cross-driver confounds that make the
20.3-vs-now comparison weak.

### Verdict

AC #4 met. AC #1, #2, #5 met (§4–§6). **AC #3 is partially met**: the producer-side
half is measured and positive; the TTFA half is **not substantiated and must not be
claimed**. The story's own Context section predicted ~190 ms and the measurement
does not support it.

This does not argue for reverting. The change removes real complexity, halves
per-chunk decode, and is perceptually inert on 16 blinded trials. It argues for
retiring the TTFA claim rather than the code, and for treating the per-frame
talker regression as its own question.
