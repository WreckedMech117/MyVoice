# Story 20.1: TTFA Spike — `faster-qwen3-tts` Adopt / Port / Reject (Phase ⊥-Polish-3)

Status: done
Baseline commit: 3e3e740 (branch `spike/20-1-ttfa`; API-server work committed first so AC #7's clean-tree check is meaningful)

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase tag: Phase ⊥-Polish-3 (D-20). Successor to Phase ⊥-Polish-2-Ship (Story 18.5). First story of Epic 20 (First-Audio Latency). -->
<!-- Story class: SPIKE. Ships evidence + a routed recommendation. Ships NO production behavior change. Structural precedent: Story 18.1 (instrumentation-only closure). -->
<!-- Risk: Low to the product (no dispatch-chain edits); HIGH to the schedule if allowed to sprawl. Timebox + kill-gates are load-bearing ACs, not decoration. -->
<!-- Source: `_bmad-output/planning-artifacts/research/technical-qwen3-tts-ttfa-optimization-2026-08-31.md` (Mary, 2026-08-31). -->

## Story

As **Commander, deciding whether to reopen the audited TRUE_STREAM dispatch chain**,
I want **a measured, decomposed account of where MyVoice's ~5.9 s first-chunk latency actually goes, and a benchmarked verdict on whether `faster-qwen3-tts`'s CUDA-Graph + StaticCache technique closes it on our 1.7B model**,
so that **the Epic 20 architecture pass is scoped against evidence rather than against a third party's 0.6B benchmark table on different hardware**.

## Context

Epic 18 closed the **producer-bottleneck** question: steady-state emit/drain ratio 3.23× → **0.670×**, underrun gaps structurally impossible (`18-4-torch-compile-decoder-persistent-cache-evidence.md` §"Producer-bottleneck steady-state ratio").

Epic 18 did **not** close the **time-to-first-audio** question, and its own evidence file says why:

> *"first chunk has to come out of the talker's autoregressive loop, and `compile_talker=False` (Fix #1 for Story 16.8's TRUE_STREAM forward-hook compatibility) keeps the talker eager. So first-chunk latency reflects talker speed (unchanged)."*

Measured Story 18.4 medians on the RTX 5090 dev host:

| Branch | Config | median `first_chunk_latency_ms` |
|---|---|---|
| A (shipping, `tts_compile="auto"`) | bf16 + compile | **5,929.4** |
| B | bf16 + eager | 5,517.8 |
| C | fp32 + eager | 5,455.3 |

**Compile is −7.46 % on TTFA** — it made first audio slightly *slower*. Story 18.4's Task 8.6 routing condition fired on this and was overridden by Commander because the load-bearing OFR-E gate (producer ratio) passed. The TTFA question was deferred, not answered. **This story is that deferred question.**

Mary's 2026-08-31 research identified `andimarafioti/faster-qwen3-tts` (MIT, 1.3k stars) as a single-stream optimization of the same Qwen3-TTS-12Hz family, reporting **4.1× RTF / 3.0× TTFA on an RTX 4090** via `torch.cuda.CUDAGraph` + Transformers `StaticCache`, and reporting that **`torch.compile` gave zero speedup because dynamic KV-cache shapes defeat the compiler** — an independent reproduction of our own Branch-A null result, including the `We have observed 9 distinct sizes` warning we logged.

## Acceptance Criteria

### AC #1 — Dependency-coexistence gate (run FIRST; can kill the story cheaply)

**Given** the bundled tree pins `transformers 4.57.3`, `torch 2.10.0+cu128`, and `qwen-tts` as a **git fork** (`dffdeeq/Qwen3-TTS-streaming@3fdb4682`, pin-hash asserted at `qwen_tts_service.py:1163` and trip-wired by Story 16.1's import-attribute test)
**When** the developer attempts to install `faster-qwen3-tts` — which depends on `qwen-tts-hf`, a **Transformers 5** compatible build — into an **isolated throwaway venv** (never the bundled `python310` tree)
**Then** the outcome is recorded as exactly one of three verdicts in the evidence file §1:
  - **COEXIST** — both import in one interpreter without version conflict
  - **COLLIDE-SEPARABLE** — they conflict, but `faster-qwen3-tts` runs standalone in its own venv (sufficient for benchmarking; adoption would require a dependency migration)
  - **COLLIDE-FATAL** — cannot be made to run on Windows + Python 3.10.11 + CUDA 12.8 at all
**And** on **COLLIDE-FATAL**, AC #3 and AC #4 are marked NOT APPLICABLE, the story proceeds to AC #2 + AC #5 + AC #6 only, and the recommendation defaults to **PORT-a / PORT-b / REJECT** (never ADOPT) — note both PORT paths remain fully evaluable on a COLLIDE-FATAL verdict, since `StaticCache` ships in our pinned `transformers 4.57.3`
**And** the developer does **not** modify `requirements.txt`, `build_tools/requirements-production.txt`, `build_tools/myvoice.spec`, or the bundled `python310` tree under any verdict

> **Expected going in:** Transformers 4.57.3 vs. 5.x is a major-version collision. COEXIST would be a surprise. Design the spike so a COLLIDE verdict is *cheap information*, not a dead end.

### AC #2 — Decompose our own 5.9 s (no third-party dependency required)

**Given** the existing Story 18.1 CSV-capture infrastructure (`MYVOICE_PROGRESSIVE_PLAYBACK_CSV`, `01_Run_MyVoice_With_CSV_Capture.bat`) is the established measurement surface
**When** the developer instruments the TRUE_STREAM path to attribute the interval from *generation start* to *first audible sample* across these four segments, on the canonical Sarira-F long-form CLONED utterance (RTX 5090, warm `voice_clone_prompt` cache):
  1. **Prefill / prompt-encode** — request accepted → talker's first decode step
  2. **Talker time-to-30-frames** — first decode step → 30th frame emitted (`chunk_size=25` + `lookahead=5`, the streamer's first-emit threshold)
  3. **First decode** — 30-frame chunk handed to `speech_tokenizer.decode` → PCM returned
  4. **Consumer watermark** — PCM posted → first PyAudio `write()` (includes the static 500 ms `_DEFAULT_STREAMING_WATERMARK_MS` at `audio_coordinator.py:61`)
**Then** the four segments are reported with median + p95 over ≥10 runs and **sum to within ±10 % of the measured `first_chunk_latency_ms` + watermark**, with any unattributed residual named explicitly
**And** the evidence file §2 states, in one sentence, **what fraction of user-perceived TTFA is talker-bound** — the number that determines whether Finding 1 (the expensive one) is worth its cost

#### AC #2b — Measure the product's worst case, not just the dev host's best case (Winston, 2026-08-31)

**Given** the decomposition above, run only on the canonical long-form utterance on the RTX 5090, samples the **best case on both axes** — while the product's TTFA pain lives at the other corner:
  - **Utterance class.** Clear Comms is a voice-chat **interjection** feature (`memory/clear_comms_purpose_framing.md`). TTFA is most user-perceptible on **short** utterances, which the long-form fixture does not represent.
  - **Hardware class.** `audio_coordinator.py:64-90` — the RTX 5090 stays on the **static 500 ms watermark** fast path. Hosts under the 16 GiB VRAM threshold (the RTX 30xx ship target per `memory/hardware_setup.md`) instead take the **adaptive pre-buffer** path, bounded by `MAX_PRE_DELAY_SECONDS = 10.0`. On that hardware the dominant TTFA term may be **our own consumer-side cushion rather than the talker at all**.
**When** the developer captures AC #2's four-segment decomposition
**Then** it is captured across a **2 × 2 matrix** — {short, long} × {RTX 5090 static-watermark, sub-16 GiB adaptive} — using the same instrumentation, with reduced run counts (≥5) permitted on the three off-diagonal cells
**And** the talker-bound-fraction sentence is stated **per cell**, not once globally
**And** if the sub-16 GiB cells show the adaptive pre-buffer dominating segment 4, the evidence file says so **prominently in §6**, because that finding redirects the entire epic: retuning the adaptive cushion is a different — and far cheaper — fix class than CUDA-graphing the talker
**And** the sub-16 GiB axis is executed in **three phases** per the ruling below, so that a card being unavailable delays a *confirmation* rather than blocking Gate B
**And** an unmeasured axis is never reported as a measured one — Phase 2's threshold is labelled as **derived**, not observed, wherever it appears

> **Ruling (Winston, 2026-08-31): the 3060 is unavailable near-term (no hot-swap; requires a transfer to the second PC). Do not block Gate B on it. Split the axis.**
>
> **Phase 1 — now, mandatory (Gate B).** Both RTX 5090 cells: {short, long} × static-watermark. Full four-segment decomposition, ≥10 runs each. This alone answers "is the talker dominant on a fast host, and does utterance length change the answer?"
>
> **Phase 2 — now, no hardware required. Solve for the break-even instead of guessing at it.** The adaptive cushion is not a black box: `streaming_chunk_buffer.py:192-203` implements `τ_min = T_a × (1/P − 1)`, clamped to `[0, max_pre_delay_seconds]`, returning `0.0` whenever `P ≥ 1.0`, and `P` is *observed at runtime* (`_worst_observed_producer_rate`, `:167-190`). So with `T_a` measured in Phase 1, the developer can compute — analytically, on the 5090 — **the producer rate `P` at which `τ_min` overtakes the talker segment**. Report that threshold per utterance class in evidence §2. This converts an unavailable measurement into a **single number the 3060 later either clears or does not**, which is a far stronger artifact than an unqualified "untested."
>
> **Phase 3 — deferred, and cheaper than it looks.** Confirming the 3060 does **not** require the spike's new instrumentation or a fresh build. Story 18.1's CSV capture already ships in the source tree and is env-var gated (`maybe_enable_from_env`, wired at `app.py:241-245`; metrics `progressive_chunk_emit_ms` + `progressive_chunk_audio_duration_ms` yield `P` directly as their ratio). Whenever the transfer is convenient: set `MYVOICE_PROGRESSIVE_PLAYBACK_CSV` on the 3060 host, run a handful of short and long generations, copy the CSV back, and compare the observed `P` against Phase 2's threshold. **Verify the capture module is present in the bundled artifact before relying on the shipped exe** — if it is not, fall back to a source-tree run on that host.
>
> **What this buys:** if Phase 2's threshold sits comfortably above any plausible 3060 producer rate, the adaptive-cushion hypothesis is effectively confirmed without the card, and the epic redirects on that basis. If it sits below, the talker stays the primary target and Phase 3 becomes a routine sanity check rather than a decision gate. Either way Gate B closes on schedule.

> This AC is the analytical spine of the spike. If segment 2 is not dominant — in *any* cell — the CUDA-Graph work is not the priority for that cell's users, and the story must say so.

### AC #3 — 1.7B benchmark parity (gated on AC #1 ≠ COLLIDE-FATAL)

**Given** every published `faster-qwen3-tts` benchmark table is **0.6B**, while MyVoice ships **Qwen3-TTS-12Hz-1.7B** (`service_enums.py:82-84`)
**When** the developer runs `faster-qwen3-tts` on the **1.7B** model on the RTX 5090 dev host, single-stream, at their default settings
**Then** RTF and TTFA are recorded for 1.7B, alongside a 0.6B run on the same host as a **published-number sanity check** (does their 4090 0.6B figure of RTF 5.56 / TTFA 152 ms reproduce in shape on a 5090?)
**And** the 1.7B TTFA is compared directly against our Branch-A median of **5,929.4 ms**, expressed as a ratio
**And** if their 1.7B numbers cannot be reproduced within 2× of the published 0.6B-scaled expectation, that discrepancy is investigated and reported rather than averaged away

### AC #4 — Feature-parity probe on our three shipped modes (gated on AC #1 ≠ COLLIDE-FATAL)

**Given** MyVoice ships CustomVoice, VoiceDesign, and Base/voice-clone (`service_enums.py:82-84`), and CLONED voices depend on Story 17.2's precomputed `<voice>.pt` `voice_clone_prompt` cache
**When** the developer probes `faster-qwen3-tts` against each of the three modes
**Then** the evidence file §4 records, per mode: **works / works-with-changes / unsupported**
**And** for the voice-clone path specifically, records whether Story 17.2's existing `<voice>.pt` artifacts are **consumable as-is**, **regenerable**, or **incompatible** — this is the single largest hidden migration cost and must not be left as "probably fine"
**And** records whether their **0.5 s reference-audio silence padding** (Finding 4) is separable from the rest of their stack — it is a quality fix we may want regardless of the adopt/reject verdict

### AC #5 — Chunk-size sensitivity sweep in our own code (no third-party dependency)

**Given** `DEFAULT_CHUNK_SIZE = 25` / `DEFAULT_LOOKAHEAD = 5` (`codec_token_streamer.py:46-47`) were inherited verbatim from a research example (`01-streaming-tts-research.md:184`) and have never been empirically tuned, and that at 12 Hz this means **2.5 s of audio must be generated before any PCM is emitted**
**When** the developer sweeps `chunk_size` over at least {5, 10, 15, 25} with `lookahead=5` held fixed, measuring `first_chunk_latency_ms` and the steady-state producer emit/drain ratio at each point
**Then** the TTFA-vs-ratio trade-off curve is recorded in evidence file §5
**And** the sweep runs under **`tts_compile="auto"`** — the shipping regime — per Winston's D-25 ruling below; if the developer instead runs it under `"off"` for budget reasons, that choice and its reason are stated explicitly in evidence §5, since a chunk-size win measured under eager may not survive compile engagement
**And** the sweep does **not** commit a new default — it produces the curve that a follow-up story will use

> **D-25 ruling (Winston, 2026-08-31): the assertion is not a blocker, and the earlier draft overstated it.**
> `torch_runtime.py:637` only asserts when a caller passes an explicit `decode_window_frames`. The sole production call site — `model_registry.py:591` — passes `model`, `app_settings`, `qwen_tts_pin_hash`, and `reload_cycle_idx`, and **never** passes `decode_window_frames`. It therefore takes the `None` → *inherit-from-streamer* branch, and the assertion never executes. A `chunk_size` sweep needs no D-25 waiver and no `"off"` fallback.
>
> **The real cost is wall-clock, not correctness.** `compile_cache.py:88,155` includes `decode_window_frames` in the cache key, so every sweep point is a distinct key that pays a **cold compile — measured at ~22.5 s in Story 18.4**. Four points × N runs under `"auto"` is a materially larger time bill than under `"off"`. Budget it against Gate B (AC #8) up front; if it does not fit, drop sweep points rather than dropping the `"auto"` regime — the curve's *shape* under the shipping configuration is worth more than extra points under a configuration we do not ship.

> **Rationale for including this in a spike about a third-party library:** it needs no third-party dependency, it is reversible, and it is the cheapest lever in Mary's memo. It also cross-checks AC #2 — if halving `chunk_size` roughly halves segment 2, the decomposition is sound.

### AC #6 — Routed recommendation with an explicit four-way verdict

**Given** ACs #1–#5 have produced data
**When** the developer writes evidence file §6
**Then** it states one of exactly four verdicts, with the measured numbers that justify it:
  - **ADOPT** — migrate to `faster-qwen3-tts` as the streaming engine. Highest cost: dependency migration + Story 16.8 dispatch rework + the field-migration cost in collision surface #3 below.
  - **PORT-a — vendor** — lift their `talker_graph.py` / `predictor_graph.py` into our tree under MIT attribution. Lower build cost; the standing cost is that we then own forked code while upstream keeps improving it (336 commits and active).
  - **PORT-b — build** — implement `StaticCache` + `torch.cuda.CUDAGraph` against `transformers 4.57.3` ourselves. Higher build cost, no vendored fork, composes with the existing dispatch chain. *(`StaticCache` already ships in our pinned transformers, so both PORT paths sidestep AC #1's collision entirely — evaluate them even on a COLLIDE-FATAL verdict.)*
  - **REJECT** — the measured 1.7B gain does not justify reopening the audited dispatch chain; bank the cheap wins (Findings 2, 3, 4) and stop
**And** PORT-a and PORT-b are costed **separately** — they have different build costs, different maintenance profiles, and different failure modes, and collapsing them into one "PORT" hides the actual trade-off
**And** the verdict names the **Story 16.8 forward-hook collision** explicitly: the hook on `model.model.talker.forward` that captures multi-codebook `codec_ids` is why `compile_talker=False` today, and any ADOPT/PORT path must say how it survives or replaces that hook — **including what replaces `_probe_compile_engaged`'s signal** (see collision surface #1)
**And** the recommendation is routed to **Winston (architect)** for the Epic 20 architecture pass, with a one-paragraph scope sketch of what that pass would have to decide
**And** if the verdict is REJECT, the evidence file still lands Findings 2/3/4 as independently shippable follow-up stories — a REJECT verdict must not discard the cheap wins

### AC #7 — Spike hygiene (non-negotiable)

**Given** this is a spike, not a feature
**When** the story closes
**Then** **zero** production behavior changes ship: no edits to `qwen_tts_service.py`'s dispatch chain, `torch_runtime.py`'s compile gating, `codec_token_streamer.py`'s committed defaults, or `audio_coordinator.py`'s watermark
**And** any instrumentation added for AC #2 follows the Story 18.1 precedent — additive `metrics.record` calls with verified ≤ 100 µs/call overhead (`18-1` Task 1.0 measured 1.35–2.40 µs on this host), gated so they cannot alter the timings they measure
**And** all `faster-qwen3-tts` experimentation lives in a throwaway venv + `tools/` scratch scripts, never in `src/myvoice/`
**And** the existing streaming regression suite passes with **zero regressions** (the Story 18.1 sweep surface: 32 progressive-playback + the broader ~166-test streaming/app/audio/observability sweep)
**And** `requirements.txt`, `build_tools/*`, and the bundled `python310` tree are untouched

### AC #8 — Staged timebox (hard stops; set by Commander 2026-08-31)

**Given** a spike's failure mode is not producing a wrong answer but producing no answer while consuming an epic's worth of time
**When** the developer works this story
**Then** the following **three staged gates** bound the work, each a hard stop requiring a Commander decision to pass:

| Gate | Scope | Budget | Stop condition |
|---|---|---|---|
| **A** | AC #1 — dependency probe (Task 1) | **1 hour** | At 1 hour, record whichever verdict the evidence supports and move on. An unresolved resolver fight **is** a COLLIDE-SEPARABLE verdict — do not debug it further. |
| **B** | AC #2 + AC #5 — decomposition + chunk sweep (Tasks 2, 5) | **1 working day** | This is the **delivery floor**. Gate B alone answers "is the talker the problem, and how much does chunk sizing buy?" — the story must not close without it. |
| **C** | AC #3 + AC #4 — third-party benchmark + mode parity (Tasks 3, 4) | **1 working day** | Hard ceiling. Whatever is measured at the bell is what gets reported. Partial 1.7B data with the gaps named beats complete data a week late. |

**And** the total spike ceiling is **2 working days plus a half-day for Tasks 6 + 7** (verdict, routing, cleanup, regression sweep) — the write-up is inside the budget, not after it
**And** **Gate B is unconditionally mandatory; Gate C is skippable.** If AC #1 returns COLLIDE-FATAL, or if Gate B's decomposition shows the talker is **not** the dominant TTFA segment, the developer **stops after Gate B**, writes the verdict from Gate A + B data alone, and routes to Winston. In that case a REJECT-or-PORT verdict is reached without ever running `faster-qwen3-tts` — which is a **successful** spike outcome, not an incomplete one
**And** if any gate's budget is exhausted with the question still open, the developer **surfaces that to Commander rather than silently extending** — an overrun is itself a finding (it means the technique is harder to reach on Windows than the sources imply, which is decision-relevant input to the ADOPT / PORT-a / PORT-b / REJECT call)

## Tasks / Subtasks

- [x] **Task 1 — Dependency-coexistence probe** (AC: #1, #7)
  - [x] 1.1 Create a throwaway venv on Python 3.10.11 outside the repo tree. Do **not** use `python310/`. **DEVIATION (recorded, evidence 1.1): built on CPython 3.10.20, not 3.10.11.** The side-location 3.10.11 Story 18.5 installed at `I:\Python310Inst\` no longer exists on disk (stale `py -0p` entry); the uv-managed 3.10.20 was used instead. Same minor version, same `cp310-win_amd64` wheels. `python310/` untouched.
  - [x] 1.2 `pip install faster-qwen3-tts`; capture the full resolver output including the `qwen-tts-hf` / `transformers` version demands.
  - [x] 1.3 Attempt a single-interpreter import of both our pinned `qwen_tts` fork and `faster_qwen3_tts`. Record the exact failure if it fails.
  - [x] 1.4 Assign the COEXIST / COLLIDE-SEPARABLE / COLLIDE-FATAL verdict; write evidence §1. **On COLLIDE-FATAL, mark Tasks 3 + 4 NOT APPLICABLE and continue.**
  - [x] 1.5 Independently confirm the MIT license claim from the repo's `LICENSE` file (Mary's memo flags this as README-reported, not verified). A license that is not MIT changes the PORT-vs-ADOPT calculus.

- [x] **Task 2 — TTFA decomposition** (AC: #2, #7)
  - [x] 2.1 Add the four segment-boundary metrics. Mirror the Story 18.1 tag schema (`session_id`, `chunk_index`) so the CSV joins cleanly.
  - [x] 2.2 Confirm instrumentation overhead against the Story 18.1 ≤ 100 µs/call gate before capturing any timing data.
  - [x] 2.3 Capture ≥10 runs on the canonical Sarira-F long-form utterance, warm cache, RTX 5090, at the shipping config (`tts_compile="auto"`, bf16).
  - [x] 2.3b **(AC #2b Phase 1)** Capture both RTX 5090 cells — {short, long} × static-watermark, ≥10 runs each. Mandatory for Gate B.
  - [x] 2.3c **(AC #2b Phase 2 — no hardware needed)** Using `T_a` from Phase 1 and the formula at `streaming_chunk_buffer.py:192-203` (`τ_min = T_a × (1/P − 1)`, clamped, zero when `P ≥ 1.0`), solve for the producer rate `P` at which the adaptive cushion overtakes the talker segment. Report the threshold per utterance class in evidence §2, labelled **derived, not observed**.
  - [~] 2.3d **DEFERRED (not a Gate B/C blocker). (AC #2b Phase 3 — deferred; 3060 unavailable near-term)** When the transfer to the second PC is convenient: confirm the 3060's observed `P` against Phase 2's threshold using the **already-shipped** Story 18.1 capture — set `MYVOICE_PROGRESSIVE_PLAYBACK_CSV`, run a few short + long generations, copy the CSV back, compute `P` as the `progressive_chunk_emit_ms` / `progressive_chunk_audio_duration_ms` ratio. No spike instrumentation and no fresh build required. Verify the capture module is present in the bundled artifact first; fall back to a source-tree run on that host if not. **Not a Gate B or Gate C blocker.**
  - [x] 2.4 Compute median + p95 per segment; verify the ±10 % sum-reconciliation; name any residual.
  - [x] 2.5 Write the one-sentence talker-bound-fraction statement **per matrix cell**. **These are the story's headline numbers.** If the sub-16 GiB cells show the adaptive pre-buffer dominating segment 4, flag it prominently for §6 — it redirects the epic.

- [x] **Task 3 — 1.7B benchmark** (AC: #3; gated on Task 1.4)
  - [x] 3.1 Run `faster-qwen3-tts` 0.6B on RTX 5090 as a published-number sanity check.
  - [x] 3.2 Run 1.7B; record RTF + TTFA, same utterance class as Task 2.3 where the API permits.
  - [x] 3.3 Tabulate against Branch-A 5,929.4 ms as a ratio. Investigate any >2× deviation from scaled expectation rather than reporting the mean.

- [x] **Task 4 — Mode parity probe** (AC: #4; gated on Task 1.4)
  - [x] 4.1 Probe CustomVoice, VoiceDesign, and voice-clone; record works / works-with-changes / unsupported. **PARTIAL (recorded, evidence 4.1): only Base/voice-clone was instantiated and generated through.** CustomVoice and VoiceDesign are recorded as **unverified** — the probe checked `callable(getattr(...))` on the class and never loaded those checkpoints, and `FasterQwen3TTS.generate()` is the counter-example proving attribute presence is not evidence of function.
  - [x] 4.2 Determine the fate of Story 17.2's `<voice>.pt` cache: consumable / regenerable / incompatible.
  - [x] 4.3 Isolate the 0.5 s reference-padding technique and note whether it is liftable independently.

- [x] **Task 5 — Chunk-size sweep** (AC: #5, #7)
  - [x] 5.1 Sweep `chunk_size` ∈ {5, 10, 15, 25}, `lookahead=5` fixed, via direct module-constant edit per the file's documented tuning path — reverted before close. **DEVIATION (recorded, evidence 5.1): no module-constant edit was made.** `_generate_true_stream` builds `CodecTokenStreamer()` with no args, so the geometry comes from `__init__`'s default arguments; the harness rebinds `CodecTokenStreamer.__init__.__defaults__` in-process instead. Behaviourally identical to the documented edit, and it leaves nothing in the source tree to revert — a strict improvement on the literal instruction for AC #7 hygiene. **Also extended beyond the task text: the sweep was run on the SHORT utterance class as well (review B1), which is what turned Follow-up B from inference into measurement.**
  - [x] 5.2 Record `first_chunk_latency_ms` + producer emit/drain ratio per point.
  - [x] 5.3 Run the sweep under `tts_compile="auto"` per Winston's D-25 ruling (the assertion does not fire — `model_registry.py:591` passes no explicit `decode_window_frames`). Budget ~22.5 s of cold compile per sweep point against Gate B; if it does not fit, drop points rather than dropping the `"auto"` regime, and state what was dropped.
  - [x] 5.4 Plot/tabulate the TTFA-vs-ratio curve. Do not commit a new default.

- [x] **Task 6 — Verdict + routing** (AC: #6)
  - [x] 6.1 Write evidence §6 with the ADOPT / PORT-a / PORT-b / REJECT verdict and its justifying numbers. Cost PORT-a and PORT-b separately.
  - [x] 6.1b State what replaces `_probe_compile_engaged`'s signal under any talker-graph path (collision surface #1, second-order effect).
  - [x] 6.2 Address the Story 16.8 forward-hook collision explicitly in the verdict.
  - [x] 6.3 Draft the one-paragraph architecture-pass scope sketch for Winston.
  - [x] 6.4 On any verdict, enumerate Findings 2/3/4 as independently shippable follow-up stories.

- [x] **Task 7 — Regression sweep + cleanup** (AC: #7)
  - [x] 7.1 Revert all sweep constants; confirm `git status` shows no unintended source-tree edits.
  - [x] 7.2 Run the Story 18.1 regression surface; confirm zero regressions.
  - [x] 7.3 Confirm `requirements.txt`, `build_tools/*`, and `python310/` are untouched.

## Dev Notes

### What this story is

A **decision-support spike**. It ships an evidence file and a routed recommendation. Structural precedent: Story 18.1, which closed as instrumentation-only after its measurement invalidated two of its three candidate fixes.

The spike exists because the alternative — going straight to an architecture pass — would commit Winston to redesigning an audited dispatch chain against a third party's **0.6B** benchmark table measured on **different hardware** with **no published 1.7B numbers**. That is exactly the kind of unverified premise this project's review discipline exists to catch.

### What this story is NOT

- **Not an adoption.** No dependency migration, no dispatch rework, no production behavior change. AC #7 is the fence.
- **Not a re-litigation of Epic 18.** The producer-bottleneck result (0.670×) stands. This story is about the *other* metric Epic 18 explicitly deferred.
- **Not a perceptual audition.** No NFR3 gate here — nothing ships to a user's ears. An ADOPT or PORT verdict *would* trigger one in its own story.
- **Not a chunk-size retune.** AC #5 produces a curve, not a new default. Committing the default is a follow-up story with its own D-25 ruling.
- **Not a Nari Labs evaluation.** `nari-qwen3-tts` is H100-SXM-only, Linux x86_64, CUDA 13, Docker. It is unreachable for a Windows consumer-GPU desktop app. Its value to us was the published technique list, already harvested into Mary's memo. **Do not spend spike time on it.**

### Named collision surfaces

Any ADOPT or PORT path must survive these. Naming them here so the spike reports on them rather than discovering them during implementation. **Reviewed and re-graded by Winston 2026-08-31** — one was overstated, one was understated, and one has a second-order effect the original draft missed.

1. **Story 16.8 forward-hook — CONFIRMED, and it has a second-order effect.** The hook on `model.model.talker.forward` captures multi-codebook `codec_ids`; it does not survive `torch.compile` wrapping, which is precisely why `compile_talker=False` today (`torch_runtime.py:365-395`). `faster-qwen3-tts` avoids this by hand-rolling the decode loop instead of using HF `generate()` + `BaseStreamer` — so it has no hook to break, but adopting it means replacing our audited dispatch chain wholesale.
   **Second-order effect:** `_probe_compile_engaged` deliberately walks `talker.code_predictor.model.forward` *because* `compile_talker=False` makes a talker-targeted probe always return False (`torch_runtime.py:365-378`). Any path that compiles or graph-captures the talker does not merely replace the hook — it **invalidates the probe's signal contract**, and that probe is what gates the NFR7 graceful-degradation fallback. A verdict that solves the hook but leaves the probe reporting a stale signal is incomplete.
2. **D-25 decode-window assertion — OVERSTATED; not a blocker.** `torch_runtime.py:637` asserts only when a caller passes an explicit `decode_window_frames`; the sole production call site (`model_registry.py:591`) does not, so it takes the inherit-from-streamer branch and the assertion never fires. Variable or ramped chunk sizing does **not** collide with D-25. The genuine constraint is that `decode_window_frames` participates in the compile cache key (`compile_cache.py:88,155`), so each distinct window shape pays a ~22.5 s cold compile. See the ruling under AC #5.
3. **Story 16.1 pin trip-wire — UNDERSTATED; this is a user-facing migration, not a dev-tree one.** `_QWEN_TTS_PIN_HASH = "3fdb4682"` (`qwen_tts_service.py:1163`) plus the import-attribute test pin us to the `dffdeeq/Qwen3-TTS-streaming` fork. A `qwen-tts-hf` migration invalidates both — **and** the `qwen_tts_pin` field is embedded in Story 17.2's cached `<voice>.pt` metadata (`:1543`, `:1688`), which is checked on load and invalidates the cache on mismatch. The consequence: **every shipped user's cloned-voice prompts silently regenerate on first use after the update**, on a product that ships publicly via myvoicetts.com. That is a real, user-visible cost on the ADOPT path and must be priced there explicitly — not deferred to "probably fine."

### Why the PORT options may dominate — and why they are two options, not one

`StaticCache` ships in transformers 4.57.3 — the version already in our bundled tree. `torch.cuda.CUDAGraph` is plain torch, no Triton required. So the *technique* is reachable inside our existing fork without any dependency migration at all, which makes AC #1's likely COLLIDE verdict much less consequential than it first appears. Task 6.1 must weigh both PORT paths on their own merits rather than treating them as ADOPT's consolation prize.

**Winston, 2026-08-31 — why the split matters.** PORT-a (vendor their `talker_graph.py` / `predictor_graph.py` under MIT attribution) and PORT-b (build `StaticCache` + `CUDAGraph` ourselves) look similar on a slide and are not similar in practice:

| | PORT-a — vendor | PORT-b — build |
|---|---|---|
| Build cost | Lower — working code exists | Higher — we write and debug it |
| Standing cost | We own a fork of an actively-developed repo (336 commits); their improvements stop being free | None beyond normal maintenance |
| Failure mode | Silent drift from upstream; their bug fixes need manual re-vendoring | Our own bugs, but in code shaped like the rest of our tree |
| Fit with dispatch chain | Their loop assumes their surrounding API; adapting it to Story 16.8's contract is the real work | Built against our contract from the start |
| Licence exposure | MIT attribution obligations enter our distribution (**Task 1.5 must verify MIT from `LICENSE`, not the README**) | None |

Rule-of-three applies here: we have exactly one use for this technique. That argues against building an abstraction and mildly favours PORT-a *if* the adaptation cost proves small — which is precisely what the spike should measure rather than assume.

Note also that a PORT path would need **none** of the Story 18.5 Triton-on-Windows bundling machinery — `faster-qwen3-tts` explicitly reports using "just `torch.cuda.CUDAGraph`", no Triton, no Flash Attention, no vLLM. Whether that means Story 18.5's bundle components become removable is an interesting downstream question but is **out of scope here**; note it, do not chase it.

### Contradicted bets to confirm, not re-test

`faster-qwen3-tts` reports testing and rejecting: attention backends (SDPA / FA2 — "no RTF difference; attention not bottleneck") and custom CUDA kernels (8.4× isolated → **1.25× end-to-end**). These corroborate our existing decision to skip megakernel fusion (research P2.B) and argue for deprioritizing the FA2 runtime-verification story named in `architecture-streaming-acceleration-and-lightning-tier.md`. **The spike should not spend time re-testing these** — it should note in evidence §6 that two independent sources put the ceiling near zero, and let Commander retire the FA2 story on that basis.

### Latest Tech Information

- **`faster-qwen3-tts`** — MIT (README-reported; Task 1.5 verifies), 1.3k stars, 336 commits, Windows-native supported, Python 3.10+, PyTorch 2.5.1+ (we are on 2.10.0+cu128 — satisfied). Depends on `qwen-tts-hf` (Transformers 5). Optional GGML backend. Core files: `talker_graph.py`, `predictor_graph.py`, `streaming.py`, `model.py`. Public streaming API takes `chunk_size` as a first-class argument.
- **Published single-stream results (0.6B):** RTX 4090 RTF 1.34 → 5.56 (4.1×), TTFA 3.0× better (152 ms). H100 RTF 0.59 → 4.19 (7.1×). Jetson AGX Orin 0.175 → 1.57 (9.0×). Per-component on Jetson: talker 75 ms → 12 ms per step.
- **Chunk-size sensitivity (their Jetson 0.6B data):** `chunk_size=2` → TTFA 266 ms / RTF 1.042; `chunk_size=8` → TTFA 556 ms / RTF 1.384. A 4× chunk increase roughly doubled TTFA — the shape AC #5 is testing for in our own code.
- **`StaticCache`** — pre-allocated fixed KV tensors with in-place `index_copy_` updates; fixed shapes are the precondition for CUDA-graph capture. Present in transformers 4.57.3. Their reported parity caveat: outputs are not bit-identical vs. `DynamicCache` due to differing kernel reduction orders under BF16/TF32 — relevant to any future audition, not to this spike.
- **Our host:** RTX 5090 Blackwell, Win11, torch 2.10.0+cu128, transformers 4.57.3, portable `python310` (3.10.11). Ship-target also covers RTX 30xx/40xx (Ampere+) per `memory/hardware_setup.md`.

### Project Context Reference

- Test interpreter: run pytest via the bundled `python310\python.exe` — system Python lacks deps (`memory/test_interpreter_portable_python310.md`).
- DLL ordering: torch must import before PyQt6 (`memory/torch_pyqt6_dll_ordering.md`); coverage runs need the inline torch-first preamble via `tools/run_cov.py` (`memory/torch_before_coverage_dll_ordering.md`).
- Git state: V2 is canonical since 2026-05-05; `_bmad-output/` is gitignored (`memory/git_repo_state.md`).
- Production state: ships publicly via myvoicetts.com as a Windows .exe with bundled portable python310; installer size is a known pain point — an ADOPT verdict must account for dependency-size impact (`memory/production_release_state.md`).
- Review discipline: HIGH/MEDIUM fix tests must mirror the exact bug class; run review twice after non-trivial auto-fixes (`memory/code_review_regression_test_exact_class.md`).

## References

- `_bmad-output/planning-artifacts/research/technical-qwen3-tts-ttfa-optimization-2026-08-31.md` — Mary's findings memo; the six ranked findings this spike operationalizes.
- `_bmad-output/implementation-artifacts/18-4-torch-compile-decoder-persistent-cache-evidence.md` — Branch A/B/C medians, the −7.46 % TTFA result, the `compile_talker=False` causal statement, the `9 distinct sizes` CUDA-graph warning, and the Task 8.6 override rationale.
- `_bmad-output/implementation-artifacts/18-1-underrun-gap-mitigation.md` — structural precedent for a measurement-only story closure; instrumentation-overhead gate; CSV-capture infrastructure.
- `_bmad-output/planning-artifacts/architecture-streaming-acceleration-and-lightning-tier.md` — D-21/D-22/D-25 context; the FA2 verification story this spike recommends retiring.
- https://github.com/andimarafioti/faster-qwen3-tts + `/blob/main/BLOG.md`
- https://github.com/QwenLM/Qwen3-TTS/discussions/358 + https://nari-labs.com/blog/qwen3-tts-speed-cost-frontier/ — origin of the enquiry; technique list only.

## Dev Agent Record

### Agent Model Used

claude-opus-5[1m]

### Debug Log References

- 2026-08-31 - Task 2.2 instrumentation-overhead probe (bundled `python310\python.exe`, RTX 5090): N=1000 basic tags 1.03 us/call; N=5000 extended tags 1.06 us/call; N=1000 with the Story 18.1 CSV listener attached 4.28 us/call. All PASS against the 100 us/call gate (~23x headroom, worst case). Run BEFORE any timing capture.
- 2026-08-31 - Gate A: `pip install faster-qwen3-tts` into a throwaway venv (uv CPython 3.10.20, session scratchpad, outside the repo tree). 89 packages; resolver demands `transformers<6,>=5.15.1` -> 5.16.1, `huggingface-hub<2.0,>=1.5.0` -> 1.29.0, `qwen-tts-hf<0.2` -> 0.1.1.post1, `torch>=2.5.1` -> 2.13.0+cpu. Verdict COLLIDE-SEPARABLE.
- 2026-08-31 - Gate A Task 1.3: pinned fork on `sys.path` under transformers 5.16.1 fails at import with `TypeError: check_model_inputs() missing 1 required positional argument: 'func'` (`modeling_qwen3_tts_tokenizer_v2.py:498`). `import faster_qwen3_tts` succeeds standalone.
- 2026-08-31 - Gate A Task 1.5: MIT verified from `faster_qwen3_tts-0.4.0.dist-info/licenses/LICENSE` ("MIT License / Copyright (c) 2026 Andres Marafioti") plus `License-Expression: MIT` in METADATA. `qwen-tts-hf` is Apache-2.0 - a new obligation on the ADOPT path.
- 2026-08-31 - Gate B first pass DISCARDED: the long-form cell and the cs5/cs10/cs15 sweep were captured while a 3 GB `pip` download ran concurrently and showed a 2.3x slowdown (TTFA 2,100-3,400 ms, ratio 0.94-1.38 vs 1,662 ms / 0.659 on a quiet host). Every cell re-captured quiet into `implementation-artifacts/clean/`; the contaminated pass is retained for audit.
- 2026-08-31 - Gate B capture (quiet host): long n=10 warm, short n=10 warm, sweep {5,10,15} n=5 warm each, cold/warm cells n=4 each for compile=auto and compile=off. Aggregated by `20-1-aggregate-ttfa.py`.
- 2026-08-31 - Gate C benchmark (throwaway venv, torch 2.11.0+cu128 from the cu128 index, transformers 5.16.1, faster-qwen3-tts 0.4.0, same RTX 5090): 0.6B TTFA 289 ms / RTF 3.97; 1.7B TTFA 310 ms / RTF 3.71; 1.7B with MyVoice's Story 17.2 `.pt` at a 30-frame window TTFA 665 ms / RTF 3.85; 1.7B short-utterance TTFA 304 ms.
- 2026-08-31 - Task 5.4 empirical check: `%LOCALAPPDATA%\MyVoice\torch_compile_cache\` still holds exactly its two pre-existing key directories after the whole sweep. `engage_compile_optimizations`'s hard-coded `streamer_chunk_size=25` default means `decode_window_frames` never varies with the sweep, so no sweep point paid a cold compile.
- 2026-08-31 - Task 7.2 regression sweep (PRE-review; superseded by the 514-test sweep below): 320 passed (Story 18.1 surface) + 186 passed (dispatch / session / models / streaming-buffer) = 506, zero failures. Two failures in the FIRST sweep were verified pre-existing on baseline `3e3e740` (stale exact-call assertions vs the 2026-05-15 `text_length` kwarg) and fixed; see evidence file section 7.4.
- 2026-08-31 (review response) - Gate B RE-CAPTURED end to end on a quiet host after the C1/C5 code fixes, into `implementation-artifacts/clean2/`: long n=10, short n=10, long sweep {5,10,15} n=5, **short sweep {5,10,15} n=5 (B1)**, cold/warm n=4 x2. Per-cell timestamps in `20-1-recapture-timeline.txt`; total 12m38s (18:21:25 -> 18:34:03). Evidence sections 2 and 5 are POOLED across `clean/` + `clean2/`, with the between-pass spread published (11 % long, 37 % short).
- 2026-08-31 (review response) - Gate C headline re-run in the quiet window immediately after (18:34:28 -> 18:35:14): 1.7B + MyVoice `.pt` + 30-frame window = TTFA 664.5 ms / RTF 3.849 vs the original 665.3 / 3.851 - reproduces within 0.1 %, so the Gate C headline is not host-contamination-sensitive. Its 5-run spread was 662-674 ms (+/-1 %) against MyVoice's +/-11-37 %, which is what establishes that the variance in section 2.4 is ours, not the host's. Also confirmed `loaded_via_myvoice_stub: False` (evidence 4.2, review A7).
- 2026-08-31 (review response) - Adaptive-cushion simulation against the shipped `StreamingChunkBuffer` with an injected clock: `20-1-adaptive-cushion-sim.py` -> `20-1-adaptive-cushion-sim.txt`. Shows `MAX_PRE_DELAY_SECONDS` (escape 3), not tau_min (escape 5), is the binding constraint for every P <~ 0.78, and that at P=0.5 the effective wait is 12.5 s because the cap is only evaluated inside `push` (review A4).
- 2026-08-31 (review response) - Scanned all 24 `.pt` files under `voice_files/`: every one pickles `qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem`, never MyVoice's wrapper class - so the Gate C probe's `_install_myvoice_stub()` was dead code on every file measured, and the 'consumable as-is' verdict rests on the library-class path rather than on a shim (review A7).
- 2026-08-31 (review response) - Post-fix regression sweep: **328 passed** (Story 18.1 surface + the 8 new retained-surface tests) + **186 passed** (dispatch / session / models / streaming-buffer) = **514, zero failures**. Two production instrumentation defects were found by writing those tests and fixed before this run: `prefill_forward_calls` reported 2 for a single prefill (the counter is incremented before the call), and the threshold emit branch carried no `path` tag so 'absent' had to be read as 'threshold'.

### Completion Notes List

_All figures below are the post-review, pooled two-pass numbers. MyVoice TTFA on
this host carries a +/-20% run-to-run and session-to-session spread (evidence
2.4), so every speedup is stated as a range._

- **Verdict: PORT-b (build), staged behind three cheaper wins.** Not ADOPT (namespace collision, transformers 4->5 for the whole app, 89 transitive packages, and it does not fix the cold start). Not PORT-a (the adaptation cost measured low enough that vendoring's "working code exists" advantage does not pay for owning a fork of a 336-commit repo). Full costing in evidence section 6.
- **Headline correction to Epic 18.** Story 18.4's 5,929.4 ms Branch-A median is a *first-generation-of-process* number. Steady-state TTFA on the RTX 5090 is **1,785 ms** (pooled n=20), 90.0 % of it talker-bound. The A-vs-B null (-7.46 %) is explained: the compile branch pays a ~4.0 s one-time in-process inductor-cache reload that almost exactly cancels its ~2.4 s per-generation win. Reconstructing the 18.4 measurement conditions brackets **both** 18.4 medians (Branch A 5,767-6,042 vs 5,929.4; Branch B 4,975-5,788 vs 5,517.8). In steady state compile is **2.4-2.8x faster** on TTFA, not 7 % slower - 18.4's causal claim ("first-chunk latency reflects talker speed (unchanged)") is wrong, because the code predictor runs inside the talker's per-frame forward and *is* compiled.
- **TRUE_STREAM degenerates to batch on short utterances.** `chunk_size + lookahead = 30` frames = 2.5 s of audio; **11 of 20** short runs never reached the threshold and emitted only the terminal residual flush. TTFA 1,651 ms vs 1,701 ms generation wall = **97 %**. Clear Comms is an interjection feature, so this is the utterance class that matters most.
- **B1 (added on review): `chunk_size = 10` fixes it, measured rather than inferred.** On the short fixture, all **5/5** runs take the threshold path at cs=10 and produce 3 chunks; TTFA falls 1,651 -> 921 ms (**-44 %**) and drops from 97 % of generation time to 50 %. On the long fixture the same setting gives 1,785 -> 875 ms (**-51 %**) with no measurable throughput cost (ratio 0.665 -> 0.676). The optimum is NOT the smallest value: at cs=5 the 417 ms/chunk falls under the 500 ms static watermark and the consumer hands back 263-277 ms.
- **AC #2b Phase 2 - re-derived by SIMULATION against the shipped buffer, and the first draft was wrong about which constraint binds.** `_adaptive_ready_to_dispatch` checks `elapsed >= MAX_PRE_DELAY_SECONDS` (10 s) *before* it ever consults tau_min, so for every `P <~ 0.78` the **cap** is the binding escape, not the formula. Because the cap is only evaluated inside `push`, the effective wait at the 3060's documented `P ~ 0.5` is **12.5 s** against a 5.0 s talker segment (2.5x). At `chunk_size = 10` the ratio worsens to **4.0x** - Follow-ups B and C are coupled and must move together.
- **T_a estimator overshoot recorded as its own finding.** `text_length * 0.08 s/char` gives 27.92 s against a measured 18.88 s (**+45 %**) and feeds tau_min directly. Simulated impact: at P=0.80 a perfect estimator would cut segment 4 from 7.81 s to 5.21 s; at P=0.50 it changes **nothing**, because the cap dominates.
- **AC #3 answers the 1.7B question affirmatively.** 0.6B -> 1.7B costs ~7 % on both TTFA and RTF on this host, so the published 0.6B tables transfer. Like-for-like (same `.pt` prompt, same 30-frame window, same host): 665 ms / RTF 3.85 vs our 1,785 ms / RTF 1.42 = **2.5-2.8x TTFA, 2.71x RTF**. On the SHORT/Clear-Comms class the gap is **4.8-6.7x** (304 ms vs 1,651 ms).
- **Gate C confounds stated and partly closed.** Row 3 was re-run in the quiet window immediately after the Gate B re-capture: 664.5 ms vs the original 665.3 ms - reproduces within **0.1 %**, so the headline is not host-contamination-sensitive. The **runtime boundary is NOT removable**: their side runs torch 2.11.0+cu128 / transformers 5.16.1 against our 2.10.0+cu128 / 4.57.3, and the COLLIDE-SEPARABLE verdict makes holding it fixed impossible. Treat 2.5-2.8x as an **upper bound on what PORT-b could recover on our stack**, not a target.
- **Their variance is 20x tighter than ours.** 5 runs on the same quiet host in the same window: faster-qwen3-tts +/-1 %, MyVoice +/-11-37 % between sessions with a 2x outlier inside one session. A robustness argument for fixed-shape graph capture, independent of the mean speedup.
- **AC #4: the Story 17.2 `.pt` cache is CONSUMABLE AS-IS**, and the stub contradiction is resolved. All **24** `.pt` files under `voice_files/` pickle the LIBRARY class (`qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem`), which resolves to qwen-tts-hf's identically named dataclass, so `torch.load` succeeds with **no shim** - the probe's `_install_myvoice_stub()` was dead code on every file measured (now instrumented and reported as `loaded_via_myvoice_stub: False`). Collision surface #3 still fires, but from *our own* `qwen_tts_pin` metadata check (`qwen_tts_service.py:1543`), which a migration shim can suppress - not from format incompatibility. **CustomVoice and VoiceDesign are downgraded to "unverified"** - never instantiated.
- **Task 6.1b:** `_probe_compile_engaged` returns False under any graph-capture path (no dynamo sentinels), which would disengage NFR7's gate on the very acceleration it certifies. Proposed replacement: a positive `TalkerGraph.captured` / `PredictorGraph.captured` assertion plus a one-shot numerical-parity check against an eager forward (tolerance, not equality - StaticCache is not bit-identical to DynamicCache under BF16/TF32).
- **PORT-b's central assumption, flagged as the highest-risk item for Winston:** `TalkerGraph` captures the INNER `talker.model.forward`, not the OUTER `talker.forward` that Story 16.8's hook patches - so PORT-b may preserve the audited dispatch chain rather than replace it. Verified against the object graph; NOT verified empirically. Its failure would flip the verdict to PORT-a or REJECT.
- **`transformers 4.57.3` already ships the whole `StaticCache` surface the port needs.** The only API delta across the entire port surface is `lazy_initialization(key_states)` (4.57.3, 1 arg) vs 2 args in transformers 5 - a one-line adaptation. Our pinned fork exposes every attribute `TalkerGraph` and `PredictorGraph` reach for.
- **Latent D-25 trap recorded (evidence 5.5).** `engage_compile_optimizations` hard-codes `streamer_chunk_size=25`, so `decode_window_frames` is pinned at 30 regardless of the streamer. The sweep was therefore free of cold compiles (empirically confirmed: no new cache keys), but a follow-up story that commits a new `DEFAULT_CHUNK_SIZE` would silently violate the very invariant D-25's assertion exists to protect.
- **Measurement-quality corrections applied on review.** Every "p95" in the first draft was the MAXIMUM (`round(0.95*(n-1)) = n-1` for n <= 10) - now an interpolated type-7 quantile with `max` alongside. The AC #2 sum-reconciliation is an **identity** (the segments telescope), so its "0.0000 % error" proves ordering and completeness, not accuracy; the +/-10 % gate as specified was unfalsifiable. A genuine `perf_counter` bracket taken outside the metric stream was added as the real check: slack median +0.16/+0.66 ms, range -0.19 to +1.21 ms across 20 runs. `first_chunk_latency_ms` is a **restatement** of TTFA(post), not corroboration.
- **Two corrections to Mary's memo** (evidence 6.6): Finding 5's stated cause is wrong (`EmotionProfile.repetition_penalty` is an orphaned field - no call site passes it; the library default 1.05 is what we actually take), and Epic 18's causal statement is wrong as above. Finding 6's recommendation stands: **recommend Commander retire the FA2 verification story**, citing this file plus research P2.B.
- **Recommended sequencing (evidence 6.4):** (A) prime the compile cache on the warm path too, -4.0 s on the first generation after every launch; (B) retune `chunk_size` 25 -> 10, -51 % long / -44 % short, **moving `MAX_PRE_DELAY_SECONDS` in the same change**; (C) re-scope the sub-16 GiB cushion around that cap (a product decision, not the one-liner the first draft proposed); (D) lift the 0.5 s reference padding (perceptual benefit unaudited); then (E) re-measure and scope PORT-b against the post-A+B baseline (~875 ms), not against today's 1,785 ms.
- **Zero production BEHAVIOR change shipped; observational surface RETAINED by architect ruling 2026-08-31** (AC #7 as amended): 218 additive lines in `src/`, all `metrics.record` calls, one-shot guards, comments, and one optional observational `session_id=None` kwarg. Evidence 2.8 makes the deferred AC #2b Phase 3 capture depend on these metrics, so reverting them would break it - which is why they now carry 8 tests and documented contracts. `torch_runtime.py`, `codec_token_streamer.py` and `streaming_chunk_buffer.py` are untouched; `audio_coordinator.py`'s watermark constants are untouched; `requirements.txt`, `build_tools/*` and `python310/` are untouched.
- **Writing the retained-surface tests found two real instrumentation defects**: `prefill_forward_calls` reported 2 for a single prefill (counter incremented before the call), and the threshold emit path carried no `path` tag so "absent" had to be read as "threshold". Both fixed.
- **AC #2b Phase 3 remains deferred** (RTX 3060 on a second PC). Confirmed executable as specified and now *more* valuable than planned: the spike's six boundaries were added to the same env-var-gated CSV surface, so a 3060 run yields the full four-segment decomposition, not just `P`. The standing constraint was honoured - the adaptive path was NOT simulated by forcing the VRAM threshold on the 5090; the section-2.7 simulation drives the real buffer class with an injected clock instead.

### File List

**Source - additive instrumentation, RETAINED by architect ruling 2026-08-31:**

- `src/myvoice/services/qwen_tts_service.py` (+95) - four one-shot TTFA boundary metrics: `ttfa_generation_start_ms` (in `_generate_true_stream`) and `ttfa_talker_thread_start_ms` + `ttfa_first_decode_step_ms` + `ttfa_first_chunk_emit_ms` (in `_build_true_stream_talker`). Session id is read from the existing `self._current_session_id` rather than widening `_build_true_stream_talker`'s signature, which the integration suite monkeypatches with a 3-positional fake. The residual-flush emission sits **outside** the guarded try (review C1) so an instrumentation call never shares a failure domain with the audio dispatch it measures. Both emit paths carry an explicit `path` tag. No control-flow change.
- `src/myvoice/services/tts_streaming/streaming_decoder.py` (+20) - `ttfa_first_decode_complete_ms` (first decoded chunk only) plus its one-shot instance flag. `decode_chunk_latency_ms` unchanged.
- `src/myvoice/services/audio_coordinator.py` (+65) - `from myvoice.observability import metrics`; `ttfa_first_playback_write_ms` fired once per streaming session immediately before the first `_dispatch_chunk_to_services`; optional observational `session_id: Optional[str] = None` keyword on `start_streaming_session` plus its two state fields, **re-armed on teardown as well as on open** (review C2); Args block now documents both `session_id` and the previously undocumented `text_length` (review C4). Watermark / crossfade / adaptive-pre-buffer constants and logic unchanged.
- `src/myvoice/app.py` (+6) - passes `session_id=chunk.session_id` to `start_streaming_session`.
- `src/myvoice/observability/progressive_playback_csv_capture.py` (+32/-4) - the six `ttfa_*` names added to `_CAPTURED_METRIC_NAMES`; module docstring corrected (it said "three metrics" and listed four while the frozenset holds ten) and the unchanged-CSV-header trade documented (review C4).

**Tests (+8 new, all covering the retained surface):**

- `tests/unit/observability/test_progressive_playback_csv_capture.py` (+73/-2) - all six names in the closed-set assertion; new row-layout test.
- `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py` (+61) - `ttfa_generation_start_ms` one-shot, wall-clock-bracketed, shares t0 with `first_chunk_latency_ms`.
- `tests/integration/test_streaming_tts_smoke.py` (+173) - the three talker boundaries against the **real** `_run_talker` (one-shot, monotonic ordering, `path`/`frames` tags) **and a `step_count=20` case pinning the `path="residual_flush"` branch**, which no prior fixture exercised.
- `tests/unit/services/tts_streaming/test_streaming_decoder.py` (+81) - `ttfa_first_decode_complete_ms` once per session while `decode_chunk_latency_ms` stays per chunk; second worker re-arms.
- `tests/unit/services/test_audio_coordinator.py` (+112) - segment-4 boundary once across two chunks; **re-arms with a fresh session id after `stop_streaming_session`** (the C2 regression); `session_id=None` default for legacy callers.
- `tests/unit/test_app_progressive_playback_instrumentation.py` (+9) - session id actually reaches the coordinator (asserted against a non-None value on purpose).
- `tests/unit/test_app_progressive_playback.py` (+12/-2) - **pre-existing breakage fixed, out of scope, flagged.** Two `assert_awaited_once_with` calls were already stale against the `text_length` kwarg added 2026-05-15; verified failing on untouched baseline `3e3e740`.

**New tracked scratch drivers (AC #7 sanctions `tools/`; 962 lines total):**

- `tools/ttfa_spike_harness.py` (648) - headless four-segment decomposition + chunk-sweep driver over the production TRUE_STREAM path, with the independent `perf_counter` bracket, drain-before-unsubscribe, and try/finally run loop (review A2, C5).
- `tools/ttfa_spike_faster_qwen3_probe.py` (314) - Gate C benchmark + mode-parity probe (throwaway venv only; imports nothing from `src/myvoice/`), with interpolated quantiles and a guarded RTF (review A1, C5).

**Evidence (gitignored; force-add per `memory/git_repo_state.md`):**

- `20-1-ttfa-spike-faster-qwen3-tts-evidence.md`
- `20-1-aggregate-ttfa.py` + `20-1-aggregate-output.txt`
- `20-1-adaptive-cushion-sim.py` + `20-1-adaptive-cushion-sim.txt` (section 2.7)
- `clean/` and `clean2/` - both clean capture passes, all cells + logs
- `clean2/20-1-sweep-short-cs{5,10,15}.csv` - the B1 short sweep
- `20-1-recapture-timeline.txt` - per-cell timestamps for the gate accounting
- `gatec/20-1-fq3-*.json` + `gatec2/…-requiet.json` + logs
- `20-1-regression-sweep.log`
- `20-1-*.csv` + `20-1-run-*.log` (discarded contaminated first pass, retained for audit)

**Confirmed untouched (AC #7 / Task 7.3):** `requirements.txt`, `build_tools/requirements-production.txt`, `build_tools/myvoice.spec`, all other `build_tools/*`, the bundled `python310/` tree, `src/myvoice/services/tts_streaming/torch_runtime.py`, `src/myvoice/services/tts_streaming/codec_token_streamer.py`, `src/myvoice/services/streaming_chunk_buffer.py`.

**Out-of-scope working-tree file surfaced by `git status`:** `_bmad-output/implementation-artifacts/sprint-status.yaml` (Epic 20 registration; pre-dates this session's work).

## Change Log

- 2026-08-31 — Story drafted by Mary (analyst) from the same-day TTFA research memo, following Commander's direction to investigate before committing to an architecture pass. Registered as the first story of Epic 20 (First-Audio Latency); Epic 19 remains reserved for the architected-but-unregistered Lightning Tier scope.
- 2026-08-31 — AC #8 (staged timebox) added at Commander's direction, resolving Open Question #1. Gate A = 1 h, Gate B = 1 working day (mandatory delivery floor), Gate C = 1 working day (skippable), + half-day for verdict/cleanup. Routed to Winston (architect) for pre-read review ahead of the Epic 20 architecture pass.
- 2026-08-31 — **Winston's architecture pre-read applied (3 amendments).** (1) **D-25 re-graded from blocker to non-issue** — the assertion at `torch_runtime.py:637` fires only on an explicit `decode_window_frames`, which the sole production call site (`model_registry.py:591`) never passes; AC #5's sweep therefore runs under the shipping `tts_compile="auto"` regime, with the real constraint being ~22.5 s cold compile per sweep point (`decode_window_frames` is in the compile cache key). Resolves Open Question #3. (2) **PORT split into PORT-a (vendor) / PORT-b (build)** — AC #6 is now a four-way verdict; the two paths have different build costs, maintenance profiles, failure modes, and licence exposure, and costing them as one hid the trade-off. (3) **AC #2b added — measure the product's worst case.** The original decomposition sampled the best case on both axes (long-form utterance, RTX 5090 static watermark). TTFA matters most for *short* utterances (Clear Comms is an interjection feature) and on *sub-16 GiB* hosts, where `audio_coordinator.py:64-90`'s adaptive pre-buffer (`MAX_PRE_DELAY_SECONDS = 10.0`) may dominate TTFA instead of the talker — a finding that would redirect the epic toward a much cheaper consumer-side fix. Now a 2 × 2 matrix with a stated-limitation fallback if no sub-16 GiB host is reachable (new Open Question #4). Also raised: collision surface #1 gains the `_probe_compile_engaged` signal-contract side effect (the probe walks the code predictor precisely *because* the talker is eager, so any talker-graph path invalidates the NFR7 degradation gate); collision surface #3 re-graded from dev-tree to **user-facing** — the `qwen_tts_pin` embedded in Story 17.2's `<voice>.pt` metadata means an ADOPT migration silently regenerates every shipped user's cloned-voice prompts.
- 2026-08-31 — **AC #2b three-phased after Commander confirmed the RTX 3060 is not reachable near-term** (second PC, no hot-swap). Rather than degrade the cell to an untested limitation, the adaptive-cushion question is now answered analytically first: `streaming_chunk_buffer.py:192-203` implements `τ_min = T_a × (1/P − 1)` with `P` observed at runtime, so Phase 2 solves for the **break-even producer rate** at which the cushion overtakes the talker segment — a derived threshold the 3060 later either clears or does not. Phase 3 (the physical confirmation) turns out to need **no spike instrumentation and no fresh build**: Story 18.1's env-var-gated CSV capture already ships (`app.py:241-245`) and yields `P` directly as the `progressive_chunk_emit_ms` / `progressive_chunk_audio_duration_ms` ratio, so it is an env var plus a CSV copy whenever the transfer is convenient. Gate B is no longer blocked on hardware. Resolves Open Question #4. Standing constraint recorded: do not simulate the adaptive path by forcing the VRAM threshold on the 5090 — that reproduces the code path but not the physics.

- 2026-08-31 - **Spike executed and closed for review (Gates A + B + C all closed inside budget).** AC #1 = **COLLIDE-SEPARABLE** (`qwen-tts` and `qwen-tts-hf` both claim the top-level `qwen_tts` import name - a namespace conflict, not merely a transformers 4-vs-5 version conflict; our pinned fork does not import under transformers 5 at all). MIT verified from the shipped `LICENSE`. AC #2's decomposition reframes the epic: Epic 18's 5,929.4 ms is a first-generation-of-process number that reconciles to within 1.9 %; **steady-state TTFA is 1,662 ms, 90.0 % talker-bound**, and compile is **2.78x faster** in steady state rather than 7.46 % slower - the null was a 3,961 ms one-time inductor-cache reload cancelling a 2.9 s per-generation win. AC #2b found TRUE_STREAM degenerating to batch on short (Clear Comms) utterances, and derived the sub-16 GiB adaptive-cushion break-even at `P < 0.87`. AC #5 found the TTFA optimum at `chunk_size = 10`, not the smallest value, because the 500 ms watermark bites below ~6. AC #3/#4 measured 1.7B at **2.50x TTFA / 2.57x RTF** like-for-like and found the Story 17.2 `.pt` cache **consumable as-is**. **Verdict: PORT-b (build), staged behind three cheaper reversible wins**, routed to Winston. Zero production behavior change; 506 tests green.

- 2026-08-31 - **Architect review response applied (no loopback; measurements stand).** Sixteen claim corrections, one missing experiment, five code fixes, one ruling. Headline numbers were re-derived from a second clean capture pass and pooled (n=20 per Phase-1 cell): steady-state TTFA **1,785 ms** long / **1,651 ms** short, compile advantage **2.4-2.8x**, third-party ratio **2.5-2.8x** long / **4.8-6.7x** short. Every "p95" in the first draft was the maximum and is now an interpolated quantile with `max` alongside; the AC #2 sum-reconciliation is recorded as an **identity** (the +/-10 % gate as specified was unfalsifiable) and replaced by a genuine `perf_counter` bracket taken outside the metric stream. **Follow-up C was re-scoped**: simulation against the shipped `StreamingChunkBuffer` shows `MAX_PRE_DELAY_SECONDS`, not tau_min, is the binding escape below `P ~ 0.78`, so the proposed one-line change does not help at the operating point that justified it. **B1 added**: the short-utterance chunk-size sweep now measures what Follow-up B previously inferred - cs=10 pulls short utterances off the residual-flush path in 5/5 runs for -44 % TTFA. CustomVoice/VoiceDesign downgraded to **unverified**. Gate C confounds stated; its headline re-run quiet and reproduced within 0.1 %. **Architect ruling recorded**: the six metrics stay as permanent observational surface (AC #2b Phase 3 depends on them), so they now carry 8 tests and documented contracts - which surfaced two real instrumentation defects. 514 tests green.

## Open Questions for Dev Agent (deferred per workflow guidance)

1. ~~**Timebox.**~~ **RESOLVED 2026-08-31 by Commander** — see AC #8. Staged gates: A = 1 h (dependency probe), B = 1 working day (decomposition + chunk sweep, **mandatory floor**), C = 1 working day (third-party benchmark, **skippable**), plus a half-day for verdict + cleanup. Stopping after Gate B on a COLLIDE-FATAL or a non-talker-dominant decomposition is an explicitly successful outcome.
2. **Canonical utterance for AC #3.** `faster-qwen3-tts`'s API may not accept our Story 17.2 `<voice>.pt` prompt format, in which case its 1.7B TTFA is measured on a *different* conditioning path than our Branch-A baseline. If so, say so plainly and treat the comparison as indicative rather than like-for-like — do not silently compare across conditioning regimes.
3. ~~**Does AC #5's sweep belong under `tts_compile="off"` or `"auto"`?**~~ **RESOLVED 2026-08-31 by Winston** — `"auto"`. The D-25 assertion never fires from the production call path (`model_registry.py:591` passes no explicit `decode_window_frames`), so no waiver is needed. The live constraint is the ~22.5 s cold compile per sweep point, since `decode_window_frames` is in the compile cache key. Trade sweep *points* for the shipping regime, never the reverse.
4. ~~**Is a sub-16 GiB host reachable for AC #2b?**~~ **RESOLVED 2026-08-31 — no, not near-term.** Commander confirmed the 3060 is on a second PC with no hot-swap; reaching it means a transfer. AC #2b is therefore three-phased: Phase 1 (5090 cells) and Phase 2 (derived break-even threshold) run now and close Gate B; Phase 3 (3060 confirmation) is deferred and needs only the already-shipped Story 18.1 CSV capture plus an env var — no spike instrumentation, no fresh build. Standing constraint: **do not simulate the adaptive path by forcing the VRAM threshold on the 5090.** The cell's value is real producer timing on slow hardware; a forced flag reproduces the code path but not the physics, and would yield a confidently wrong `P`.

## Suggested Review Order

**The deliverable — read this first**

- Verdict, the five findings, and the corrected ranges; everything else is supporting evidence.
  [`20-1-...-evidence.md`](./20-1-ttfa-spike-faster-qwen3-tts-evidence.md)

**Producer-side boundaries (segments 1-3)**

- t0 for the whole decomposition; every other boundary is measured against this.
  [`qwen_tts_service.py:4330`](../../src/myvoice/services/qwen_tts_service.py#L4330)

- Talker thread start — separates prompt-encode from the autoregressive loop.
  [`qwen_tts_service.py:3927`](../../src/myvoice/services/qwen_tts_service.py#L3927)

- First decode step, inside the forward hook; `prefill_forward_calls` off-by-one fixed here.
  [`qwen_tts_service.py:4007`](../../src/myvoice/services/qwen_tts_service.py#L4007)

- Threshold emit path — now carries an explicit `path` tag rather than relying on absence.
  [`qwen_tts_service.py:4044`](../../src/myvoice/services/qwen_tts_service.py#L4044)

- Residual-flush emit — the short-utterance class; moved out of the audio-critical try.
  [`qwen_tts_service.py:4133`](../../src/myvoice/services/qwen_tts_service.py#L4133)

- Decode completion; segment 3 closes here.
  [`streaming_decoder.py:211`](../../src/myvoice/services/tts_streaming/streaming_decoder.py#L211)

**Consumer-side boundary (segment 4) and its session join**

- The observational `session_id` keyword — retained surface per architect ruling D1.
  [`audio_coordinator.py:1128`](../../src/myvoice/services/audio_coordinator.py#L1128)

- First playback write — the metric AC #2b Phase 3 depends on for the RTX 3060 run.
  [`audio_coordinator.py:1354`](../../src/myvoice/services/audio_coordinator.py#L1354)

- Per-session re-arm on stop as well as start; without this only generation 1 emits.
  [`audio_coordinator.py:1855`](../../src/myvoice/services/audio_coordinator.py#L1855)

- Producer session id threaded to the coordinator so the CSV joins by session, not position.
  [`app.py:3003`](../../src/myvoice/app.py#L3003)

**Shipped capture surface**

- The six names Phase 3 reads with no rebuild; docstring contract corrected to match.
  [`progressive_playback_csv_capture.py:105`](../../src/myvoice/observability/progressive_playback_csv_capture.py#L105)

**Supporting**

- Measurement drivers (throwaway scaffolding, not product code).
  [`ttfa_spike_harness.py`](../../tools/ttfa_spike_harness.py)

- Tests pinning all six boundaries, the residual-flush branch, and the session-id join.
  [`test_progressive_playback_csv_capture.py`](../../tests/unit/observability/test_progressive_playback_csv_capture.py)
