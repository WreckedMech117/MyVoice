# Story 20.1 Evidence — TTFA Spike: `faster-qwen3-tts` Adopt / Port / Reject

Story file: `_bmad-output/implementation-artifacts/20-1-ttfa-spike-faster-qwen3-tts.md`
Research memo: `_bmad-output/planning-artifacts/research/technical-qwen3-tts-ttfa-optimization-2026-08-31.md`
Branch: `spike/20-1-ttfa` · baseline commit `3e3e740`
Host: RTX 5090 (31.8 GiB, cap 12.0) · Win11 · portable Python 3.10.11 · torch 2.10.0+cu128 · transformers 4.57.3 · qwen-tts pin `3fdb4682`
Date: 2026-08-31 · **Revised 2026-08-31 after architect review** (see §10 for the change list)

Force-added per the Story 16.9 / 17.2 / 17.3 / 18.1 / 18.4 evidence-file precedent
(`_bmad-output/` is gitignored — `memory/git_repo_state.md`).

---

## §0. Answer first

> **Epic 18's 5,929 ms TTFA baseline is a first-generation-of-process number.
> On the second and every later generation of a session, TTFA on the RTX 5090 is
> ≈ 1,785 ms — and 90 % of that really is the talker loop, exactly where Mary's
> Finding 1 pointed. Epic 18's "compile is −7.46 % on TTFA" conclusion is an
> artifact of measuring only cold generations: in steady state compile is
> **2.4–2.8× faster**, not 7 % slower.**

Six findings reshape Epic 20:

1. **Epic 18's A-vs-B null is explained, and it inverts.** The compile branch
   pays a ~3.9 s one-time in-process inductor-cache reload on the first
   generation, which almost exactly cancels its ~2.4 s per-generation win.
   Every Story 18.4 sample was a first generation. Steady state: compile
   1,706 ms vs eager 4,044 ms (§2.6).
2. **The one-time reload is avoidable today.** `warmup_compile_async` runs a
   priming generation **only on a cold cache** — on the warm-cache path it
   returns early and hands the reload to the user's first utterance
   (`qwen_tts_service.py:1918-1935`). Priming on the warm path too is a ~3.9 s
   win on the utterance that matters most (§6.4 Follow-up A).
3. **TRUE_STREAM degenerates to batch on short utterances.** Anything under
   `chunk_size + lookahead = 30` frames (2.5 s of audio) never reaches the
   streamer's first-emit threshold; the whole utterance arrives as the terminal
   residual flush. Measured across 20 runs: **11/20 took the residual-flush
   path**, and TTFA (1,651 ms) is **97 % of total generation time**
   (1,701 ms). Clear Comms, an *interjection* feature, lives entirely in that
   class (§2.5).
4. **`chunk_size = 10` fixes that, and it is measured, not inferred.** At
   `chunk_size = 10` the short utterance takes the threshold path in **5/5**
   runs and TTFA drops 1,651 → 921 ms (**−44 %**), with real streaming (3
   chunks instead of 1). On the long utterance the same setting gives
   1,785 → 875 ms (**−51 %**). The optimum is *not* the smallest value: at
   `chunk_size = 5` the 500 ms consumer watermark starts to bite and hands back
   ~277 ms (§5).
5. **On sub-16 GiB hosts the dominant TTFA term is our own consumer cushion —
   and the binding constraint is `MAX_PRE_DELAY_SECONDS`, not the τ_min
   formula.** Simulated against the shipped `StreamingChunkBuffer`: at the
   3060's documented `P ≈ 0.5` the release is triggered by the 10 s cap at
   **t = 12.5 s**, against a 5.0 s talker segment — a 2.5× cushion-to-talker
   ratio. The cap is the binding escape for every `P ≲ 0.78` (§2.7).
6. **Our TTFA is noisy; theirs is not.** On the same quiet host in the same
   window, `faster-qwen3-tts` varied ±1 % across 5 runs while MyVoice varied
   ±11–37 % between capture sessions and threw a 2× outlier inside one session
   (§2.4). That is a robustness argument for fixed-shape graph capture that is
   independent of the mean speedup.

Measured third-party comparison (AC #3), like-for-like on the same host, same
voice prompt, same 30-frame window:

| | MyVoice (steady state) | `faster-qwen3-tts` 1.7B | ratio |
|---|---:|---:|---:|
| TTFA, long utterance | 1,785 ms | 665 ms | **2.68×** |
| TTFA, **short / Clear Comms** utterance | **1,651 ms** | **304 ms** | **5.43×** |
| RTF (audio s / wall s) | 1.42 | 3.85 | **2.71×** |

**Verdict: PORT-b (build), staged behind three cheaper wins. Not ADOPT.**
Full reasoning and per-path costing in §6.

---

## §1. AC #1 — Dependency-coexistence probe (Gate A)

### 1.1 Method

Throwaway venv created **outside the repo tree**, in this session's scratchpad:

```
C:\Users\AL301\AppData\Local\Temp\claude\I--MyVoiceV2\<session>\scratchpad\spike201\venv
```

**Deviation from Task 1.1, recorded:** the story specifies Python 3.10.11. The
side-location full CPython 3.10.11 that Story 18.5 installed at
`I:\Python310Inst\` **no longer exists on disk** (the `py -0p` launcher entry is
stale and resolves to a missing path). The venv was therefore built from the
uv-managed **CPython 3.10.20**
(`C:\Users\AL301\AppData\Roaming\uv\python\cpython-3.10.20-windows-x86_64-none`).
Same minor version, same ABI tag; every wheel resolved as `cp310-win_amd64`.
The bundled `python310/` tree was **not** used and **not** modified.

### 1.2 Resolver output (Task 1.2)

`pip install faster-qwen3-tts` resolves **faster-qwen3-tts 0.4.0**:

```
Collecting faster-qwen3-tts
  Using cached faster_qwen3_tts-0.4.0-py3-none-any.whl.metadata (28 kB)
Collecting qwen-tts-hf<0.2,>=0.1.1.post1 (from faster-qwen3-tts)
  Using cached qwen_tts_hf-0.1.1.post1-py3-none-any.whl.metadata (62 kB)
Collecting transformers<6,>=5.15.1 (from faster-qwen3-tts)
  Using cached transformers-5.16.1-py3-none-any.whl.metadata (32 kB)
Collecting torch>=2.5.1 (from faster-qwen3-tts)
  Using cached torch-2.13.0-cp310-cp310-win_amd64.whl.metadata (39 kB)
Collecting huggingface-hub<2.0,>=1.5.0 (from faster-qwen3-tts)
```

**89 packages** installed in total — it pulls `gradio 6.26.0`, `onnxruntime`,
`librosa`, `numba`/`llvmlite`, `pandas`, `fastapi`, `uvicorn` transitively.
Directly relevant to the installer-size constraint in
`memory/production_release_state.md`.

Pin-by-pin against the bundled tree:

| package | MyVoice (bundled `python310`) | `faster-qwen3-tts` demand | resolved | verdict |
|---|---|---|---|---|
| `transformers` | **4.57.3** | `<6,>=5.15.1` | 5.16.1 | **major-version collision** |
| `huggingface-hub` | **0.36.0** | `<2.0,>=1.5.0` | 1.29.0 | **major-version collision** |
| `tokenizers` | 0.22.2 | (via transformers 5) | 0.23.1 | collision |
| `torch` | **2.10.0+cu128** | `>=2.5.1` | 2.13.0 **+cpu** | satisfiable — see §1.5 |
| TTS engine | `qwen-tts` 0.0.4 @ `dffdeeq/Qwen3-TTS-streaming@3fdb4682` | `qwen-tts-hf<0.2` | 0.1.1.post1 | **import-name collision** |

### 1.3 The collision is worse than a version conflict — it is a namespace conflict

```
$ cat venv/Lib/site-packages/qwen_tts_hf-0.1.1.post1.dist-info/top_level.txt
qwen_tts
$ cat python310/Lib/site-packages/qwen_tts-0.0.4.dist-info/top_level.txt
qwen_tts
```

Two different distributions — `qwen-tts` (ours) and `qwen-tts-hf` (theirs) —
both claim the top-level import name **`qwen_tts`**. They cannot be installed
into one interpreter at all: not with a version pin, not with careful install
ordering. One overwrites the other's files.

This is the most consequential AC #1 finding and neither the story nor Mary's
memo anticipated it; both framed the risk as a `transformers` major-version
conflict alone.

### 1.4 Single-interpreter import attempt (Task 1.3)

Our pinned fork's package tree was copied to a temp directory and prepended to
`sys.path` inside the venv (so it shadows `qwen-tts-hf`'s `qwen_tts`), then
imported under transformers 5.16.1:

```
transformers 5.16.1
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "...\forkpkg\qwen_tts\__init__.py", line 21, in <module>
    from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
  ...
  File "...\forkpkg\qwen_tts\core\tokenizer_12hz\modeling_qwen3_tts_tokenizer_v2.py",
        line 498, in Qwen3TTSTokenizerV2DecoderTransformerModel
    @check_model_inputs()
TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

Transformers 5 turned `check_model_inputs` from a decorator **factory**
(`@check_model_inputs()`) into a bare decorator (`@check_model_inputs`). Our
pinned fork therefore fails at **class-definition time**, before any model is
touched. This is not a hot-path incompatibility a shim could paper over; it is
an import-time break in the tokenizer module.

Conversely, `faster_qwen3_tts` **imports cleanly on its own**:

```
torch 2.13.0+cpu   cuda_available False   transformers 5.16.1
faster_qwen3_tts OK 0.4.0
```

### 1.5 Windows PyPI torch is CPU-only — a real friction point for ADOPT

The resolver's `torch 2.13.0` is the PyPI Windows wheel, which is **CPU-only**
(`torch.cuda.is_available() → False`). GPU benchmarking required a second step:

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
→ torch 2.11.0+cu128   cuda True
```

`torch>=2.5.1` is satisfied and `faster_qwen3_tts` + `transformers 5.16.1` still
import against it. Recorded because an ADOPT path would have to pin the CUDA
index in `build_tools/requirements-production.txt`, and because **`torch 2.13`
— the version the resolver naturally picks — has no cu128 build published** on
`download.pytorch.org` at time of measurement (the cu128 index tops out at
2.11.0).

### 1.6 Verdict

> ### **COLLIDE-SEPARABLE**
>
> They cannot share an interpreter — the `qwen_tts` top-level name is claimed by
> both distributions, and our pinned fork does not even import under
> transformers 5. But `faster-qwen3-tts` runs standalone in its own venv on
> Windows + Python 3.10 + CUDA 12.8, which is sufficient for benchmarking.
> AC #3 and AC #4 therefore proceed and are **not** marked NOT APPLICABLE.

### 1.7 Licence verification (Task 1.5) — MIT CONFIRMED from `LICENSE`, not the README

```
$ cat venv/Lib/site-packages/faster_qwen3_tts-0.4.0.dist-info/licenses/LICENSE
MIT License

Copyright (c) 2026 Andres Marafioti
...
$ grep License venv/Lib/site-packages/faster_qwen3_tts-0.4.0.dist-info/METADATA
License-Expression: MIT
License-File: LICENSE
```

**MIT is verified** from the licence text shipped inside the distribution — the
artifact we would actually be vendoring, not the README. PORT-a's attribution
obligation is a one-line notice: cheap.

**A licence obligation Mary's memo did not surface:** `qwen-tts-hf` declares
`License-Expression: Apache-2.0`. An **ADOPT** path (which takes `qwen-tts-hf`
as a runtime dependency) therefore adds an Apache-2.0 NOTICE obligation to the
distribution. **PORT does not** — it touches only the MIT-licensed files.

---

## §2. AC #2 + AC #2b Phase 1 — Decomposing our own TTFA (Gate B)

### 2.1 Instrumentation (Task 2.1) — and the ruling that it stays

Six additive one-shot `metrics.record` boundaries were added to the TRUE_STREAM
path, all valued as absolute wall-clock ms (`time.time() * 1000.0`) so they join
the Story 18.1 CSV columns by subtraction rather than clock reconciliation:

| metric | site | closes / opens |
|---|---|---|
| `ttfa_generation_start_ms` | `qwen_tts_service.py` — `_generate_true_stream`, publishes `start_time` | t0 |
| `ttfa_talker_thread_start_ms` | `qwen_tts_service.py` — first statement of `_run_talker` | splits segment 1 into MyVoice dispatch overhead vs. model prompt-encode |
| `ttfa_first_decode_step_ms` | `qwen_tts_service.py` — `_streaming_forward`, first non-`None` `codec_ids` | closes segment 1, opens segment 2 |
| `ttfa_first_chunk_emit_ms` | `qwen_tts_service.py` — before the first `streamer.queue.put` (`path="threshold"`) **and** in `_flush_residual_and_eos` (`path="residual_flush"`) | closes segment 2, opens segment 3 |
| `ttfa_first_decode_complete_ms` | `streaming_decoder.py` — `_decode_and_post`, after `decode_fn` returns, first chunk only | closes segment 3, opens segment 4 |
| `ttfa_first_playback_write_ms` | `audio_coordinator.py` — `play_audio_chunk`, before the first `_dispatch_chunk_to_services` | closes segment 4 |

All six were added to `progressive_playback_csv_capture.py`'s
`_CAPTURED_METRIC_NAMES`, so the existing `MYVOICE_PROGRESSIVE_PLAYBACK_CSV`
env-var surface and `01_Run_MyVoice_With_CSV_Capture.bat` capture them with no
new plumbing (Story 18.2 Task 4.1 precedent). The CSV header is unchanged, so
the `path` / `frames` / `pcm_samples` / `text_length` / `buffer_mode` tags are
visible in the structured `myvoice.metrics` log but not in CSV columns —
deliberate, to avoid churning the schema Stories 18.1–18.4 already consume.

`AudioCoordinator.start_streaming_session` gained one optional keyword,
`session_id: Optional[str] = None`, threaded from
`MyVoiceApp._handle_progressive_chunk_async` so the consumer-side boundary
carries the same session tag the producer-side boundaries do. It is
observational only; no behavior reads it. It is re-armed on
`start_streaming_session` **and cleared on `stop_streaming_session`**, so a
multi-generation playback session emits the boundary once per session rather
than once per process.

> **Architect ruling, 2026-08-31 — these six metrics, the two coordinator state
> fields, and the `session_id` keyword STAY as permanent product surface.**
> §2.8 makes the deferred AC #2b Phase 3 (RTX 3060) capture depend on them;
> reverting them would break it. AC #7's fence is therefore restated as **"zero
> production *behavior* change; observational surface retained by architect
> decision"** — the fence held on behavior, and the retention is deliberate
> rather than accidental. Because the surface is retained, it carries tests
> (§7.4) and documented contracts (`start_streaming_session`'s Args block, the
> capture module's docstring).

**Residual-flush variant — why it exists.** The first capture attempt lost 6 of
11 short-utterance runs to a missing `ttfa_first_chunk_emit_ms`. That was not an
instrumentation bug; it was the finding in §2.5. The metric is now emitted from
both paths, each tagged, so the short class is measurable instead of silently
dropped — and both branches are pinned by tests.

### 2.2 Instrumentation overhead (Task 2.2 — gate: ≤ 100 µs/call)

Measured through the bundled `python310\python.exe` on the RTX 5090 host,
mirroring the Story 18.1 §1 method, **before** any timing data was captured:

| configuration | N | mean per call | verdict |
|---|---:|---:|:---:|
| basic tags (`session_id` + `frames`), no listeners | 1,000 | **1.03 µs** | PASS |
| extended tags (4 kwargs), no listeners | 5,000 | **1.06 µs** | PASS |
| basic tags, Story 18.1 CSV listener attached | 1,000 | **4.28 µs** | PASS |

~23× headroom in the worst case, and each of the six boundaries fires **once per
generation** (one-shot flags), not once per chunk.

**Clock resolution.** `time.time()` steps at **0.5 ms** on this host. Every
table below marks values at or below that step as `<=res` rather than printing
them as measurements — a "0.3 ms" segment-4 median means *below the clock's
ability to resolve*, not "measured 0.3 ms".

### 2.3 Method (Tasks 2.3 / 2.3b)

`tools/ttfa_spike_harness.py` drives the **production** dispatch path —
`QwenTTSService._generate_true_stream`, the real `CodecTokenStreamer`, the real
`StreamingDecoderWorker`, the real Story 16.8 forward-hook — with no Qt, no
`SessionRegistry` and no `AudioCoordinator`. Voice = Sarira-F, loaded from the
Story 17.2 `voice_files/Sarira-F.quality.pt` cache (warm). Precision `auto`
(→ bf16 on Ampere+), `tts_compile="auto"` — the shipping regime.

**Harness validity cross-check.** The harness reproduces Epic 18's independently
measured steady-state producer emit/drain ratios:
Branch A (bf16+compile) 18.4 = **0.670×**, harness = **0.665×**;
Branch B (bf16+eager) 18.4 = **1.663×**, harness = **1.633×**.
The §2.6 cold-start reconciliation brackets both 18.4 medians. The harness is
measuring the same system Epic 18 measured.

**Two honest limitations, stated up front:**

1. **Segment 4 is harness-replayed, not device-measured.** The harness mirrors
   `MyVoiceApp._handle_progressive_chunk_async` exactly — sync trampoline,
   `run_coroutine_threadsafe` onto the loop, float32→int16 conversion, then a
   push into a **real** `StreamingChunkBuffer` built with the production
   constants (500 ms watermark, 64-sample crossfade, 24 kHz mono int16). What it
   does not reproduce is the PyAudio `start_streaming_session` device-open on
   chunk 0 and the blocking `stream.write`. Story 17.3 §4.1 estimates the
   device-open at ~50–100 ms. **That term is named as an unattributed residual
   rather than folded into segment 4.** The shipped
   `ttfa_first_playback_write_ms` metric measures it for real on any future GUI
   capture run.
2. **Utterance-length stochasticity.** The talker's EOS timing varies run to
   run, so the short fixture lands sometimes above and sometimes below the
   30-frame first-emit threshold. That variance is itself the §2.5 finding, not
   noise to be averaged away.

Fixtures:

* **long** — the canonical Story 17.3 §4.1 step-3 paragraph (349 chars), the
  same string every Epic 18 measurement used. ~19.0 s of audio.
* **short** — `"Hold on a second, say that again."` (33 chars, ~2.3 s of audio).
  Clear Comms is a voice-chat **interjection** feature
  (`memory/clear_comms_purpose_framing.md`).

**Two capture passes, pooled.** Every cell was captured twice — pass 1 into
`clean/`, pass 2 into `clean2/` after the review-response code fixes (which
touch only instrumentation placement and the harness's drain ordering, never
producer timing). Unless stated otherwise the tables below are **pooled across
both passes**, and the between-pass spread is reported (§2.4) rather than hidden
by picking one.

### 2.4 Measurement quality — what the reconciliation actually proves

Three separate things get called "the check" in this story; they are not
equally strong, and the review was right to press on it.

**(a) The segment sum is an IDENTITY, not a check.** Segments 1+2+3+residual are
defined as consecutive differences of the same five timestamps, so they
telescope to `t_post − t0` by construction. The "0.0000 % reconciliation error"
in every cell below is arithmetic. **AC #2's ±10 % gate as written was
unfalsifiable** — a spec defect, recorded here as a limitation rather than
claimed as a passed gate. It does confirm one thing worth having: that no
boundary is emitted out of order and none is missing.

**(b) `first_chunk_latency_ms` is a RESTATEMENT, not corroboration.** It is
measured from the same `start_time` to the same `_wrapped_post` call. It agrees
to the last digit because it is the same interval, not because two methods
converged. Listed in the tables as "restatement" for that reason.

**(c) The independent bracket IS a check, and it passes.** The harness stamps
`perf_counter()` immediately before awaiting the dispatch, and again in the
synchronous chunk trampoline on the decoder-worker thread — both **outside the
metric stream, on a different clock**. Comparing that interval to the
metric-derived total can fail (e.g. if `t0` were taken after model load, slack
would be ≈ −4,000 ms):

| cell | n | bracket median | TTFA(post) median | slack median | slack range |
|---|---:|---:|---:|---:|---|
| A — long | 10 | 1,849.4 ms | 1,849.1 ms | **+0.66 ms** | −0.19 … +1.21 ms |
| B — short | 10 | 1,472.4 ms | 1,472.1 ms | **+0.16 ms** | −0.14 … +0.83 ms |

Slack stays inside ~2 clock steps on 19 of 20 runs (one long run reached
1.21 ms). The metric-derived total is the interval it claims to be.

**(d) Run-to-run and session-to-session variance is large, and it is ours.**

| cell | pass 1 median | pass 2 median | pooled median | spread |
|---|---:|---:|---:|---:|
| long, cs25 | 1,662 ms | 1,849 ms | **1,785 ms** | 11 % |
| short, cs25 | 2,020 ms | 1,472 ms | **1,651 ms** | 37 % |
| long, cs5 | 776 ms | 594 ms | 673 ms | 31 % |
| long, cs10 | 1,014 ms | 778 ms | 875 ms | 30 % |
| long, cs15 | 1,157 ms | 1,185 ms | 1,171 ms | 2 % |

Within pass 2's long cell one run reached 3,649 ms against a 1,785 ms median —
which is why the tables report an interpolated p95 **and** the max separately.

This is **not** host contention: `faster-qwen3-tts` re-benchmarked on the same
host in the window immediately after this capture (§3.7) varied by **1 %** over
5 runs (662–674 ms). The variance is intrinsic to our pipeline — stochastic
sampling changes token content and utterance length, and the CUDA-graph
re-recording the 18.4 evidence noted ("we have observed 9 distinct sizes")
makes per-step cost shape-dependent. **Treat every MyVoice TTFA figure in this
document as ±20 %, and every speedup ratio as a range.**

### 2.5 The 2 × 2 matrix, Phase 1 (both RTX 5090 cells)

Vocabulary, fixed for the whole document (the review found three competing
definitions):

* **TTFA(post)** = t0 → first PCM handed to the consumer. This is what
  `first_chunk_latency_ms` measures.
* **TTFA(release)** = TTFA(post) + segment 4 (the consumer cushion). This is
  what a listener waits through, minus the device.
* **"first audible sample"** as AC #2 words it = TTFA(release) + the PyAudio
  device-open and device buffer latency, which this harness does not measure
  (§2.3 limitation 1).

#### Cell A — long utterance × RTX 5090 static watermark (pooled n = 20)

| segment | n | median (ms) | p95 interp (ms) | max (ms) |
|---|---:|---:|---:|---:|
| **1 — prefill / prompt-encode** | 20 | **104.7** | 131.5 | 213.8 |
| &nbsp;&nbsp;1a — MyVoice dispatch overhead | 20 | 1.0 | 1.6 | 3.0 |
| &nbsp;&nbsp;1b — model prompt-encode | 20 | 103.4 | 129.9 | 210.8 |
| **2 — talker to 30-frame chunk** | 20 | **1,606.5** | 1,930.4 | 3,343.5 |
| **3 — first decode (codec → PCM)** | 20 | **77.1** | 95.6 | 101.3 |
| **4 — consumer cushion** | 20 | **`<=res`** | 1.1 | 1.5 |
| residual (decode-complete → post) | 20 | `<=res` | 1.0 | 1.0 |
| **TTFA(post)** | 20 | **1,785.1** | 2,135.7 | 3,648.7 |
| restatement: `first_chunk_latency_ms` | 20 | 1,785.1 | 2,135.7 | 3,648.7 |

* **TTFA(release) = 1,785.4 ms** (segment 4 is below clock resolution here: at
  `chunk_size=25` one chunk carries 2,083 ms of audio, four times the 500 ms
  watermark, so the buffer releases on the first push).
* Producer emit/drain ratio **0.665×** (P = 1.50× real-time); T_a = 19,043 ms;
  generation wall 13,409 ms → **RTF 1.42**.
* First-emit path: `threshold` on 20/20 runs.

> **AC #2 headline, cell A: 90.0 % of TTFA(release) on a long utterance on the
> RTX 5090 in steady state is talker-bound (segment 2).**

#### Cell B — short utterance × RTX 5090 static watermark (pooled n = 20)

| segment | n | median (ms) | p95 interp (ms) | max (ms) |
|---|---:|---:|---:|---:|
| **1 — prefill / prompt-encode** | 20 | **92.8** | 125.0 | 128.0 |
| **2 — talker to first token chunk** | 20 | **1,500.6** | 1,950.9 | 1,978.8 |
| **3 — first decode** | 20 | **44.7** | 78.4 | 80.2 |
| **4 — consumer cushion** | 20 | **`<=res`** | 1.0 | 1.0 |
| **TTFA(post)** | 20 | **1,650.6** | 2,133.2 | 2,185.0 |

* T_a = 2,297 ms; generation wall = **1,701 ms**; **TTFA is 97.0 % of total
  generation time.**
* First-emit path: **`residual_flush` 11/20**, `threshold` 9/20.

> **AC #2 headline, cell B: 90.9 % talker-bound — but the number is misleading
> on its own, because on this utterance class TTFA equals generation time.
> TRUE_STREAM contributes essentially nothing.**

**The structural finding.** `DEFAULT_CHUNK_SIZE = 25` + `DEFAULT_LOOKAHEAD = 5`
means the streamer's first-emit threshold is **30 codec frames = 2.5 s of audio
at 12 Hz**. Eleven of twenty short runs never reached it: the talker hit EOS
first, and the only token chunk that ever left the forward-hook was the terminal
residual flush in `_flush_residual_and_eos`. On those runs TRUE_STREAM is
**batch generation with extra machinery**. The other nine crossed 30 frames only
just, so their "streaming" gain is one sub-chunk.

Clear Comms is an interjection feature; short utterances are its entire purpose.
**The streaming architecture does not currently serve the feature whose latency
matters most**, and no amount of talker acceleration changes that — only the
chunk geometry does, which §5 now measures directly rather than inferring.

#### Cells C and D — UNMEASURED

The AC #2b matrix's other two cells — **C (long × sub-16 GiB adaptive)** and
**D (short × sub-16 GiB adaptive)** — were **not measured**. No sub-16 GiB host
was reachable (RTX 3060 on a second PC, no hot-swap). §2.7 substitutes a
derivation and a simulation against the shipped buffer; §2.8 leaves the physical
confirmation deferred. Per the story's standing constraint, the adaptive path
was **not** simulated by forcing the VRAM threshold on the 5090.

### 2.6 The cold/warm split — and the reconciliation with Epic 18

Four generations per process, `--warmup 0` so the first is recorded (pass 2):

**`tts_compile="auto"`** (Epic 18 Branch A analogue)

| run | TTFA(post) | 1a model load | 1b first-forward | 2 talker | 3 decode | gen wall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (cold) | **10,366** | 4,599 | **4,008** | 1,587 | 172 | 22,378 |
| 1 | 1,706 | 1 | 92 | 1,523 | 89 | 13,337 |
| 2 | 1,697 | 1 | 97 | 1,514 | 85 | 13,426 |
| 3 | 1,754 | 0 | 95 | 1,573 | 86 | 12,712 |

**`tts_compile="off"`** (Epic 18 Branch B analogue)

| run | TTFA(post) | 1a model load | 1b first-forward | 2 talker | 3 decode | gen wall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (cold) | **8,875** | 3,900 | 1,151 | 3,664 | 160 | 34,737 |
| 1 | 4,044 | 1 | 162 | 3,791 | 90 | 31,894 |
| 2 | 4,297 | 1 | 160 | 4,069 | 66 | 30,206 |
| 3 | 4,003 | 1 | 160 | 3,763 | 79 | 32,806 |

**Reconciliation.** Story 18.4's `.bat` harness launched a fresh process per
sample and generated exactly one utterance, so **every 18.4 sample is a cold
run**. In the GUI the BASE model is preloaded at startup (`app.py:607-618`;
Sarira-F is CLONED → `QwenModelType.BASE` per
`voice_profile_service.py:1383-1398`), so segment 1a is already paid. Dropping
1a from the cold rows and taking both capture passes as a range:

| branch | 1b + 2 + 3 (this spike, both passes) | Story 18.4 measured median | verdict |
|---|---:|---:|---|
| A — bf16 + compile | **5,767 – 6,042 ms** | **5,929.4 ms** | 18.4's median lands **inside** the range |
| B — bf16 + eager | **4,975 – 5,788 ms** | **5,517.8 ms** | 18.4's median lands **inside** the range |

Epic 18's numbers reproduce, and the mechanism is now visible:

* The **eager** branch's TTFA is dominated by a slow talker loop (segment 2 =
  3,763–4,069 ms).
* The **compile** branch's talker loop is ~2.4× faster (segment 2 =
  1,514–1,573 ms) but it pays a **~4.0 s one-time in-process inductor-cache
  reload + CUDA-graph record** on the first forward (segment 1b), which almost
  exactly cancels the gain.

That cancellation *is* the −7.46 % result. Story 18.4's causal statement —
*"first-chunk latency reflects talker speed (unchanged)"* — is **incorrect**:
`compile_talker=False` leaves the talker's own transformer eager, but the
**code predictor runs inside the talker's per-frame forward**
(`modeling_qwen3_tts.py:1671`) and *is* compiled.

| regime | TTFA(post) steady state | producer ratio |
|---|---:|---:|
| compile | **1,706 ms** (pass 2) / 1,770 ms (pass 1) | 0.665–0.671× |
| eager | **4,044 ms** (pass 2) / 4,620 ms (pass 1) | 1.633–1.90× |
| **compile advantage** | **2.4 – 2.8×** | **2.5 – 2.9×** |

**Where the ~4 s first-forward cost comes from, and why it is fixable.**
`warmup_compile_async` (`qwen_tts_service.py:1918-1935`) runs a priming
generation **only when `compile_cache.is_warm(key)` is False**. On a warm cache
it emits a `cache_hit` breadcrumb and returns — deliberately, per its own
docstring: *"the inductor cache reloads from disk lazily on first user-facing
generation."* Every launch after the first-ever therefore hands a ~4 s bill to
the user's first utterance. Priming on the warm path too (silently, with the
audio callback detached — the gate at `:1982-1998` already documents why the
callback must not fire) removes it. See §6.4, Follow-up A.

### 2.7 AC #2b Phase 2 — DERIVED, and then SIMULATED against the shipped buffer

The first draft of this section derived a break-even from the τ_min formula
alone and drew the wrong conclusion about which constraint binds. Corrected here
by driving the **production `StreamingChunkBuffer`** with an injected clock and
synthetic chunk arrivals (`20-1-adaptive-cushion-sim.py`, output in
`20-1-adaptive-cushion-sim.txt`).

`_adaptive_ready_to_dispatch` (`streaming_chunk_buffer.py:261-307`) has five
escapes **in priority order**, and the τ_min comparison is the **last**:

1. `is_final`
2. `_chunks_held >= max_hold_chunks` (16)
3. `elapsed >= max_pre_delay_seconds` (**10.0 s**)
4. observed `P >= 1.0`
5. `audio_buffered_seconds >= τ_min`

Simulation, long fixture, `chunk_size=25`, `T_a_est = 349 × 0.08 = 27.92 s`
(what the code actually uses):

| P | segment 4 (release offset) | released by | talker segment | cushion / talker |
|---:|---:|---|---:|---:|
| 0.50 | **12.50 s** | escape 3 — the 10 s cap | 5.00 s | **2.50×** |
| 0.70 | 11.90 s | escape 3 — the 10 s cap | 3.57 s | 3.33× |
| 0.75 | 11.11 s | escape 3 — the 10 s cap | 3.33 s | 3.33× |
| 0.80 | 7.81 s | escape 5 — τ_min 6.98 s | 3.12 s | 2.50× |
| 0.85 | 4.90 s | escape 5 — τ_min 4.93 s | 2.94 s | 1.67× |
| 0.90 | 2.31 s | escape 5 — τ_min 3.10 s | 2.78 s | 0.83× |
| 0.95 | 2.19 s | escape 5 — τ_min 1.47 s | 2.63 s | 0.83× |

> **The binding constraint at the ship-target operating point is
> `MAX_PRE_DELAY_SECONDS = 10.0` itself, not the τ_min formula.** For every
> `P ≲ 0.78` the 10 s cap fires before τ_min is ever consulted. And because the
> cap is only evaluated inside `push`, never on a timer, the effective wait is
> the first chunk arrival at or after 10 s — **12.5 s at P = 0.5**, not 10 s.
>
> `streaming_chunk_buffer.py:14-20` records the RTX 3060 12 GB at an observed
> producer rate of **~0.5×**. Sub-16 GiB users can therefore wait **up to
> ~12.5 s** for first audio **by design**, against a 5 s talker segment. That is
> the AC #2b redirect, and it is a different — and cheaper — fix class than
> CUDA-graphing the talker.

Closed forms, for completeness (`A_c = chunk_size/12`, `W = 30/12 = 2.5 s`,
`M = ⌈W/A_c⌉·A_c = 4.167 s`), **labelled DERIVED, not observed**:

| condition | closed form | long, T_a_est 27.92 s | long, T_a measured 18.88 s | short, T_a 2.30 s |
|---|---|---:|---:|---:|
| cushion adds any delay past chunk 0 | `P < T_a/(T_a + A_c)` | P < 0.931 | P < 0.901 | P < 0.523 |
| cushion exceeds the talker segment | `P < T_a/(T_a + M)` | P < 0.870 | P < 0.819 | P < 0.354 |
| τ_min reaches the 10 s clamp | `P < T_a/(T_a + 10)` | P < 0.736 | P < 0.654 | P < 0.186 |

**An interaction Follow-up B must account for.** Re-simulated at
`chunk_size = 10`, the cushion gets *relatively worse*, because the talker
segment shrinks while the 10 s cap does not:

| chunk_size | P | segment 4 | released by | talker seg | ratio |
|---:|---:|---:|---|---:|---:|
| 25 | 0.50 | 12.50 s | 10 s cap | 5.00 s | 2.50× |
| 10 | 0.50 | 10.00 s | 10 s cap | 2.50 s | **4.00×** |
| 10 | 0.75 | 10.00 s | 10 s cap | 1.67 s | 6.00× |
| 10 | 0.90 | 2.78 s | τ_min 3.10 s | 1.39 s | 2.00× |

Follow-ups B and C are therefore **coupled**: retuning `chunk_size` without
touching the cap improves TTFA on ≥16 GiB hosts and leaves sub-16 GiB hosts
pinned at the cap.

**Finding: the T_a estimator overshoots by ~45 %** (review A5). The runtime
estimate `text_length × 0.08 s/char` (`audio_coordinator.py:89`) gives 27.92 s
for the long fixture against a measured 18.88 s. It feeds τ_min directly.
Simulated impact:

| P | segment 4 with T_a_est 27.92 s | segment 4 with a perfect estimator (18.88 s) |
|---:|---:|---:|
| 0.50 | 12.50 s (10 s cap) | **12.50 s (10 s cap — no change)** |
| 0.80 | 7.81 s | **5.21 s (−33 %)** |

So the estimator matters **only in the band where τ_min binds** (roughly
`0.78 < P < 0.90`). At the 3060's documented `P ≈ 0.5` a perfect estimator
changes nothing, because the cap dominates. Worth its own small ticket; not the
lever for the ship target.

### 2.8 AC #2b Phase 3 — deferred, unblocked, no new build required

Confirmed executable as specified: `progressive_playback_csv_capture.py` is
wired at `app.py:241-245` via `maybe_enable_from_env`, and `P` is recoverable
directly as `progressive_chunk_audio_duration_ms / Δ progressive_chunk_emit_ms`.
The spike's new boundaries were added to the same capture set, so a 3060 run
captures the **full four-segment decomposition** too, not just `P` — a free
upgrade to Phase 3's value, and the reason the six metrics are retained
(§2.1 ruling).

**Not run** (RTX 3060 is on a second PC, no hot-swap). Not a Gate B or Gate C
blocker.

**One caveat for whoever runs Phase 3:** verify `progressive_playback_csv_capture`
is present in the bundled artifact before trusting a shipped-exe run; otherwise
fall back to a source-tree run on that host.

---

## §3. AC #3 — 1.7B benchmark parity (Gate C)

### 3.1 Method

`tools/ttfa_spike_faster_qwen3_probe.py`, run with the **throwaway venv**
interpreter on the same RTX 5090, bf16, `attn_implementation="sdpa"`. TTFA is
the wall-clock interval from the `generate_voice_clone_streaming(...)` call to
the **first yielded audio chunk**. One discarded warmup run per configuration
(their CUDA-graph capture happens on first call), then n = 5.

### 3.2 Results

| # | model | prompt source | chunk (frames) | text | TTFA median | TTFA max | RTF median | first-call TTFA |
|---|---|---|---:|---|---:|---:|---:|---:|
| 1 | 0.6B Base | `ref_audio` re-encode | 12 | long | **289 ms** | 294 ms | **3.97** | 12,302 ms |
| 2 | 1.7B Base | `ref_audio` re-encode | 12 | long | **310 ms** | 317 ms | **3.71** | 2,647 ms |
| 3 | **1.7B Base** | **MyVoice Story 17.2 `.pt`** | **30** | long | **665 ms** | 674 ms | **3.85** | 1,733 ms |
| 3′ | 1.7B Base — **quiet-window re-run** | MyVoice `.pt` | 30 | long | **664.5 ms** | 673.5 ms | **3.85** | 2,933 ms |
| 4 | 1.7B Base | `ref_audio` re-encode | 12 | short | **304 ms** | 306 ms | **3.26** | 2,345 ms |

Model load: 0.6B 8,178 ms; 1.7B 5,833 / 3,763 ms.

### 3.3 The 1.7B question is answered: the 0.6B numbers transfer

Row 1 vs row 2, same host, same config: TTFA 289 → 310 ms (**+7 %**),
RTF 3.97 → 3.71 (**−7 %**). **Going from 0.6B to 1.7B costs ~7 %, not a
generation.** Mary's central caveat — *"every published benchmark table is
0.6B"* — is resolved: the technique's benefit is essentially size-invariant on
this hardware, because CUDA-graph replay removes launch overhead rather than
compute, and both models are launch-bound at batch 1.

### 3.4 Published-number sanity check (Task 3.1)

| | published (RTX 4090, 0.6B) | measured (RTX 5090, 0.6B) | ratio |
|---|---:|---:|---:|
| RTF | 5.56 | 3.97 | 0.71× |
| TTFA | 152 ms | 289 ms | 1.90× |

Both are **within the AC #3 2× investigation threshold**, so no formal
investigation is triggered, but the direction is worth stating rather than
averaging away: a 5090 should beat a 4090, and it does not here. Three
plausible, non-exclusive causes, none of which change the verdict:

1. **Chunk geometry.** Our run uses `chunk_size = 12` (their library default);
   their published TTFA table is chunk-size-sensitive (their own Jetson data:
   `chunk_size = 2` → 266 ms, `chunk_size = 8` → 556 ms), and the headline
   152 ms is not stated against a chunk size.
2. **Reference re-encoding inside the timed window.** Rows 1, 2 and 4 pass
   `ref_audio` + `ref_text`, so `create_voice_clone_prompt` runs inside the
   measured interval. Row 3 (precomputed prompt) removes that.
3. **Windows.** Every published figure is Linux.

### 3.5 The like-for-like comparison (Open Question #2, resolved)

Open Question #2 asked whether the comparison would have to cross conditioning
regimes. **It does not.** Row 3 is a true like-for-like: the same MyVoice
Story 17.2 `Sarira-F.quality.pt` prompt object, the same 30-frame first-emit
window as our `chunk_size=25 + lookahead=5`, the same 1.7B Base checkpoint, the
same host, the same utterance.

| metric | MyVoice (steady state, pooled n=20) | `faster-qwen3-tts` 1.7B | ratio |
|---|---:|---:|---:|
| TTFA, long | **1,785 ms** (pass range 1,662–1,849) | **665 ms** | **2.50 – 2.78×** |
| TTFA, **short** (§3.2 row 4 vs cell B) | **1,651 ms** (pass range 1,472–2,020) | **304 ms** | **4.84 – 6.65×** |
| RTF | 1.42 | 3.85 | **2.71×** |
| talker-loop time to 30 frames | 1,606 ms | ≈ 600 ms (665 − prefill) | ≈ 2.7× |

Against Epic 18's published Branch-A median of **5,929.4 ms** the ratio is
**8.9×** — but that is a *cold-vs-warm* comparison and is reported here only
because AC #3 asks for it explicitly. **The honest numbers are 2.5–2.8× on
long-form and 4.8–6.7× on the Clear Comms interjection class.**

### 3.6 A finding that constrains ADOPT: they have the same cold-start problem

Their first call after load costs **12,302 ms (0.6B)** / **1,733–2,933 ms
(1.7B)** — the CUDA-graph capture. That is the same shape of one-time in-process
cost that §2.6 shows dominating our own first generation. **ADOPT does not fix
the cold start; it relocates it.** Any path — ADOPT, PORT-a, PORT-b — still
needs the startup-priming fix from §6.4 Follow-up A.

### 3.7 Confounds on the Gate C headline (review A8)

Gate B discarded an entire capture pass for host contamination; Gate C is held
to the same standard here.

**(a) Host load — checked, not assumed.** §3.4 notes LM Studio was resident
during the original Gate C runs. Row 3 was therefore **re-run in the quiet
window immediately after the Gate B re-capture** (row 3′, 18:34–18:35, nothing
else running): **664.5 ms / RTF 3.85** against the original **665.3 ms /
RTF 3.85** — reproduces to within **0.1 %**. Gate C's headline is not
contamination-sensitive. (It also demonstrates that the ±20 % spread in §2.4 is
ours, not the host's.)

**(b) The runtime boundary the comparison crosses — NOT removable.** The two
sides of the 2.5–2.8× figure do not run on the same stack:

| | MyVoice | `faster-qwen3-tts` |
|---|---|---|
| torch | 2.10.0+cu128 | 2.11.0+cu128 |
| transformers | 4.57.3 | 5.16.1 |
| TTS package | `qwen-tts` 0.0.4 @ `3fdb4682` | `qwen-tts-hf` 0.1.1.post1 |
| Python | 3.10.11 (bundled) | 3.10.20 (uv) |

This is **inherent to the COLLIDE-SEPARABLE verdict** — the two cannot share an
interpreter, so no experiment can hold the runtime fixed. Some unknown fraction
of the measured gap is attributable to torch 2.11 vs 2.10 and transformers 5 vs
4 rather than to CUDA-graph capture. **State the 2.5–2.8× as an upper bound on
what PORT-b could recover**, not as a PORT-b target. PORT-b runs on *our* stack
and would have to re-establish its own number.

**(c) Only the voice-clone mode was benchmarked**, and only the long and short
fixtures. CustomVoice and VoiceDesign were not instantiated (§4.1).

---

## §4. AC #4 — Feature-parity probe on our three shipped modes

### 4.1 Mode parity (Task 4.1) — API-verified is not works

| MyVoice mode | `faster-qwen3-tts` entrypoint | streaming variant | verdict |
|---|---|---|---|
| Base / voice-clone (`…-Base`) | `generate_voice_clone` ✓ | `generate_voice_clone_streaming` ✓ | **works — instantiated and measured end-to-end (§3)** |
| CustomVoice (`…-CustomVoice`) | `generate_custom_voice` ✓ | `generate_custom_voice_streaming` ✓ | **unverified — attribute present, checkpoint never loaded** |
| VoiceDesign (`…-VoiceDesign`) | `generate_voice_design` ✓ | `generate_voice_design_streaming` ✓ | **unverified — attribute present, checkpoint never loaded** |

**Downgraded from "works" on review.** `_probe_modes` only evaluates
`callable(getattr(FasterQwen3TTS, name, None))` on the class; it never
instantiates the CustomVoice or VoiceDesign checkpoints. This document itself
supplies the counter-example that proves attribute presence is not evidence of
function: `FasterQwen3TTS.generate` is also `callable`, and raises
`NotImplementedError` on the first line of its body. Only the Base checkpoint
was loaded and generated through; the other two are **unverified**, and closing
them costs two model loads and ~5 minutes if Winston wants it before the
architecture pass.

### 4.2 Story 17.2 `<voice>.pt` cache: CONSUMABLE AS-IS (Task 4.2)

This was named as "the single largest hidden migration cost" and it turns out to
be small.

The dataclass is **field-identical** across both packages:

```python
# python310/.../qwen_tts/inference/qwen3_tts_model.py:41-51   (our pinned fork)
# venv/.../qwen_tts/inference/qwen3_tts_model.py:41-51        (qwen-tts-hf)
@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None
```

`voice_files/Sarira-F.quality.pt` unpickled **directly** under `qwen-tts-hf`:

```
MyVoice 17.2 prompt loaded: {'ref_code': (122, 16), 'ref_spk_embedding': (2048,),
 'source_class': 'qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem',
 'loaded_via_myvoice_stub': False,
 'icl_mode': True, 'x_vector_only_mode': False}
```

…and **generated correct-length audio through it** (row 3 of §3.2: 19–21 s of
audio from the 349-char fixture, matching our own 19.0 s).

**Resolving the stub contradiction (review A7).** The probe's
`_install_myvoice_stub()` originally claimed the `.pt` files pickle
`myvoice.services.qwen_tts_service.VoiceClonePromptItem`. That claim was wrong,
and it contradicted the observed `source_class`. A scan of **all 24 `.pt` files
under `voice_files/` (12 voices × quality/small)** shows every one pickles
`qwen_tts.inference.qwen3_tts_model.…` — the library class, because
`QwenTTSService._normalize_voice_clone_prompt` converts to it before persisting.
Under the venv that module path resolves to *qwen-tts-hf's* identically named
dataclass, so `torch.load` succeeds with no shim.

> **The stub was dead code on every file measured.** The probe now records
> `loaded_via_myvoice_stub`, which reported `False`, and its docstring says so.
> The "consumable as-is" verdict rests on the library-class path, not on a shim.

**Collision surface #3 still fires — for a different, and cheaper, reason.**
`Sarira-F.quality.pt.meta.json` carries `"qwen_tts_pin": "3fdb4682"`, and
`qwen_tts_service.py:1543` invalidates the cache on any mismatch. On an ADOPT
migration the pin string changes, so **every shipped user's cloned-voice prompts
would silently regenerate on first use** — exactly as the story predicted. The
correction is that this is caused by **our own metadata check, not by format
incompatibility**, so it is avoidable: a migration shim that accepts the legacy
pin value for `.pt` files whose payload validates would suppress the
regeneration entirely.

**One hazard to record:** because the pickled class path is identical in both
packages, a `.pt` written by one and read by the other resolves silently. If
qwen-tts-hf ever changes that dataclass's field order or semantics, the load
will succeed and produce wrong audio rather than raising. Any ADOPT path needs a
shape assertion at load (e.g. `ref_spk_embedding.shape == (2048,)`,
`ref_code.shape[-1] == 16`) — cheap, and the trip-wire discipline Story 16.1
already established.

### 4.3 The 0.5 s reference padding is liftable on its own (Task 4.3)

`FasterQwen3TTS._load_ref_audio_with_silence` (model.py:278-294) is **12 lines,
self-contained, and depends on nothing but `soundfile` + `numpy`**:

```python
audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if silence_secs > 0:
    silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
    audio = np.concatenate([audio, silence])
```

Its rationale (verbatim from their docstring): *"The ICL voice-cloning prompt
ends with the last codec token of the reference audio, so the model's first
generated token is conditioned on whatever phoneme the reference ends with.
Appending a short silence makes the last tokens encode silence instead,
preventing that phoneme from bleeding into the start of the generated speech."*

> **Liftable independently of every other decision here.** It lands on
> MyVoice's `create_voice_clone_prompt_for_tier` path and would need a
> `schema_version` bump in `.pt.meta.json` (currently `"1.1"`) so existing
> caches regenerate once. Worth taking **on a REJECT verdict too** — see §6.4
> Follow-up D. **Not perceptually validated by this spike** — the claim is
> theirs, and an NFR3 audition belongs in its own story.

---

## §5. AC #5 — Chunk-size sensitivity sweep (Gate B) + B1

### 5.1 Method

`chunk_size ∈ {5, 10, 15, 25}`, `lookahead = 5` held fixed, **`tts_compile="auto"`
— the shipping regime**, per Winston's D-25 ruling. Long-utterance points pooled
across both capture passes (n = 10 each; n = 20 at cs25). Short-utterance points
(**B1**, added on review) captured in pass 2 only, n = 5 each; the cs25 row is
the pooled n = 20 cell-B figure.

**Mechanism:** the harness rebinds
`CodecTokenStreamer.__init__.__defaults__` in-process. `_generate_true_stream`
constructs `CodecTokenStreamer()` with no arguments, so the geometry comes from
the `__init__` default arguments, which Python bound to the module constants at
class-definition time. Rebinding `__defaults__` is exactly equivalent to the
module-constant edit the class docstring documents as the tuning path, and
leaves **no source-tree edit to revert** — a deliberate deviation from Task 5.1's
literal wording, improving on it for AC #7 hygiene.

### 5.2 The long-utterance curve (pooled)

| `chunk_size` | window | audio / chunk | seg 2 talker | seg 4 cushion | **TTFA(post)** | **TTFA(release)** | ratio | chunks | gen wall | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 417 ms | 500 | **277.1** | 673 | **951** | 0.760 | 48 | 14,849 | 10 |
| **10** | 15 | 833 ms | 706 | `<=res` | **875** | **875** ← min | 0.676 | 24 | 12,801 | 10 |
| 15 | 20 | 1,250 ms | 995 | `<=res` | 1,171 | 1,172 | 0.677 | 16 | 13,223 | 10 |
| **25** (committed) | 30 | 2,083 ms | 1,606 | `<=res` | 1,785 | **1,785** | 0.665 | 10 | 13,409 | 20 |

### 5.3 B1 — the short-utterance curve, measured (not inferred)

The first draft argued that `chunk_size = 10` "is what actually fixes the
short-utterance degeneration" by extrapolating from the long-form sweep. The
review was right that this was the spike's top recommendation resting on
inference. It is now measured:

| `chunk_size` | window | seg 2 | seg 4 | **TTFA(post)** | **TTFA(release)** | chunks | **first-emit path** | gen wall | n |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 5 | 10 | 473 | **263.4** | 635 | 899 | 5 | **threshold 5/5** | 1,682 | 5 |
| **10** | 15 | 744 | `<=res` | **921** | **921** | 3 | **threshold 5/5** | 1,846 | 5 |
| 15 | 20 | 1,000 | `<=res` | 1,174 | 1,174 | 2 | **threshold 5/5** | 1,609 | 5 |
| 25 (committed) | 30 | 1,501 | `<=res` | 1,651 | 1,651 | 1–2 | **`residual_flush` 11/20** | 1,701 | 20 |

> **B1 answer: yes — the 15-frame threshold does pull short utterances off the
> residual-flush path.** At `chunk_size = 10` all 5/5 short runs took the
> threshold path and produced 3 chunks; TTFA fell 1,651 → 921 ms (**−44 %**),
> and TTFA went from **97 % of generation time** (batch-equivalent) to **50 %**
> (genuine streaming). Every sweep point at 15 frames or below cleared the
> threshold in 5/5 runs.

### 5.4 What the curves say

* **Segment 2 tracks the window almost linearly on both fixtures** (long: 500 /
  706 / 995 / 1,606 ms for 10 / 15 / 20 / 30 frames), which independently
  corroborates the §2.5 decomposition — the cross-check AC #5 was designed to
  provide.
* **The optimum is `chunk_size = 10`, not the smallest value.** At
  `chunk_size = 5` each chunk carries 417 ms of audio, **below the 500 ms static
  watermark** (`audio_coordinator.py:61`), so the consumer holds two chunks and
  hands back 263–277 ms — wiping out most of the producer-side gain on both
  fixtures. **`chunk_size ≥ 6` keeps the watermark a no-op.**
* **The throughput cost is small and the OFR-E gate survives.** Long-form
  generation wall goes 13,409 → 12,801 ms from 25 → 10 (within the ±20 % noise
  band; i.e. **no measurable throughput penalty**), and the producer emit/drain
  ratio goes 0.665 → 0.676 — comfortably under the architecture's `< 1.0×`
  sustained target. At `chunk_size = 5` the ratio reaches 0.760: still passing,
  with a thinner margin.
* **Best single move: `chunk_size = 10`.** −51 % TTFA on long-form, −44 % on
  short, the short class moves off the residual-flush path, ratio still 0.676.

**No new default is committed here** (AC #5). This is the curve a follow-up
story uses — and per §2.7 that story must move `MAX_PRE_DELAY_SECONDS` in the
same change, or sub-16 GiB hosts get a *worse* cushion-to-talker ratio.

### 5.5 The D-25 cost model in the story is wrong — the sweep is free

Winston's D-25 ruling correctly stated the assertion never fires. Its *cost*
model — "every sweep point is a distinct key that pays a ~22.5 s cold compile" —
does not hold, for a reason worth recording:

`engage_compile_optimizations` (`torch_runtime.py:515-523`) takes
`streamer_chunk_size: int = 25, streamer_lookahead: int = 5` as **hard-coded
defaults**, and the sole production call site (`model_registry.py:591`) passes
neither. `decode_window_frames` therefore resolves to **30 regardless of the
streamer's actual geometry**, so the `compile_cache` key
(`compile_cache.py:88,155`) never varies with `chunk_size`.

Confirmed empirically: after every sweep point ran,
`%LOCALAPPDATA%\MyVoice\torch_compile_cache\` still contains exactly the **two
pre-existing key directories** (dated 2026-05-11 and 2026-05-14). Zero new keys,
zero extra cold compiles.

> **Latent trap for the follow-up story that commits a new default.** Because
> `decode_window_frames` is pinned at 30 independent of the streamer, retuning
> `DEFAULT_CHUNK_SIZE` **silently violates the very D-25 invariant the assertion
> exists to protect**. It is harmless *today* only because (a) the fork skips
> its manual `capture_cuda_graph` under `compile_mode="reduce-overhead"`
> (`modeling_qwen3_tts_tokenizer_v2.py:966-971`) and (b) MyVoice's decode path
> calls `speech_tokenizer.decode(...)` directly rather than
> `stream_generate_pcm`, so `decode_window_frames` never reaches a runtime shape
> decision. A chunk-size retune must thread the real geometry into
> `engage_compile_optimizations`, or the invariant is decorative.

---

## §6. AC #6 — Verdict and routing

### 6.1 Verdict: **PORT-b (build) — staged, and third in line**

> **PORT-b: implement `StaticCache` + `torch.cuda.CUDAGraph` over the talker and
> code-predictor decode loops against `transformers 4.57.3`, inside our own
> tree, behind the existing `tts_compile` gate. But do it *after* three cheaper
> wins that together deliver more than PORT-b does, and re-measure before
> committing to it.**

**Why not REJECT.** The gain is real, reproduces on 1.7B, and is large:
**2.5–2.8× TTFA on long-form and 4.8–6.7× on the Clear Comms class**,
like-for-like on our own host with our own voice prompt (§3.5). The 1.7B
question that motivated the spike is answered affirmatively (§3.3). Their
run-to-run spread is also 20× tighter than ours (§2.4).

**Why not ADOPT.** Four measured costs, any one of which would be tolerable and
which together are not:

1. **Namespace collision, not just a version conflict** (§1.3).
2. **Transformers 4 → 5 for the whole app** (§1.4) — our pinned fork does not
   import under transformers 5 at all.
3. **89 transitive packages** including `gradio`, `onnxruntime`, `numba`,
   `pandas` (§1.2), against a product whose installer size is a known pain point.
4. **It does not fix the thing that actually hurts** — their first call after
   load costs 1.7–12.3 s (§3.6).

Plus the Apache-2.0 NOTICE obligation `qwen-tts-hf` brings (§1.7) and the `.pt`
regeneration event on every shipped user's machine (§4.2).

**Why PORT-b over PORT-a.** The spike measured the adaptation cost rather than
assuming it, and it came out low, which favours PORT-b:

* `TalkerGraph` uses `transformers.StaticCache` and **the model's own forward**;
  it reimplements nothing (`talker_graph.py:1-14`).
* `PredictorGraph` reaches for `code_predictor.small_to_mtp_projection`,
  `.model`, `.model.codec_embedding`, `.lm_head`, `config.num_code_groups`.
* **Our pinned fork exposes every one of those attributes**, verified by source
  read (`modeling_qwen3_tts.py:1104, 1241, 1246`).
* **`transformers 4.57.3` already ships the entire `StaticCache` surface they
  use** — `StaticCache(config=…, max_cache_len=…)`, `.layers`,
  `StaticLayer.lazy_initialization`, `is_initialized`. The **only** API delta
  found across the whole port surface is
  `lazy_initialization(key_states)` (4.57.3, one argument) vs
  `lazy_initialization(key_states, value_states)` (transformers 5, two).

So the technique is reachable in our pinned tree today, and PORT-a's "working
code exists" advantage is worth less than it looks, while its standing cost (a
fork of a 336-commit, actively developed repo) is permanent. Rule of three
applies: we have exactly one use for this.

**One caveat on the size of the prize.** §3.7(b): the 2.5–2.8× was measured
across a runtime boundary (torch 2.11 / transformers 5 vs our 2.10 / 4.57.3)
that the COLLIDE-SEPARABLE verdict makes impossible to close experimentally.
Treat it as an **upper bound on what PORT-b could recover on our stack**.

### 6.2 PORT-a vs PORT-b, costed separately

| | **PORT-a — vendor** | **PORT-b — build** |
|---|---|---|
| Surface | `talker_graph.py` (216 lines) + `predictor_graph.py` (214) + `sampling.py` (66) ≈ **496 lines** vendored, plus adapting their `streaming.py` (359) decode loop to Story 16.8's contract | Same technique, ~300–450 lines written against our contract; `StaticCache` comes from transformers 4.57.3 |
| Build cost | **Lower** — working code exists; one-line `lazy_initialization` arity fix; their loop assumes their surrounding `_prepare_generation` API, which is the real adaptation work | **Higher** — we write and debug the graph capture, the mask table, the cache-position plumbing |
| Standing cost | We own a fork of an actively developed repo (336 commits); their fixes need manual re-vendoring | None beyond normal maintenance |
| Failure mode | Drift from upstream; a fix we do not notice | Our own bugs — in code shaped like the rest of our tree, covered by our suites |
| Fit with the Story 16.8 dispatch chain | Their hand-rolled loop **replaces** the HF-`generate()`-plus-forward-hook design wholesale | Built against the existing contract from the start |
| Licence exposure | MIT attribution enters our distribution (**verified from `LICENSE`**, §1.7) | None |
| Triton / Story 18.5 bundling | Not required — "just `torch.cuda.CUDAGraph`" | Not required |
| **Verdict** | Viable fallback if PORT-b stalls | **Recommended** |

### 6.3 The Story 16.8 forward-hook collision, and what replaces `_probe_compile_engaged`

**The hook (collision surface #1).** Story 16.8 patches
`model.model.talker.forward` to capture multi-codebook `codec_ids` from
`Qwen3TTSTalkerOutputWithPast.hidden_states[1]`. `torch.compile` wrapping that
forward breaks the capture — which is why `compile_talker=False` today
(`torch_runtime.py:365-395`).

`faster-qwen3-tts` avoids the collision by construction: it hand-rolls the decode
loop and never uses `generate()` + `BaseStreamer`. **Any ADOPT or PORT-a path
therefore replaces our audited dispatch chain wholesale.**

**PORT-b's answer is different and is the reason to prefer it.** `TalkerGraph`
captures `talker.model.forward` — the **inner transformer backbone** — not
`talker.forward`, the outer wrapper the hook patches. So a PORT-b implementation
may graph-capture the inner backbone while leaving the outer `talker.forward`
eager and hookable, **preserving Story 16.8's contract intact**. This must be
verified empirically in the architecture pass — it is the single highest-risk
assumption in this verdict — but the object graph supports it.

**Second-order effect: `_probe_compile_engaged` (Task 6.1b).**
`_probe_compile_engaged` (`torch_runtime.py:348-409`) walks
`talker.code_predictor.model.forward` and sniffs for the dynamo sentinels
`_torchdynamo_orig_callable` / `__wrapped__` / `_orig_mod`, *because*
`compile_talker=False` makes a talker-targeted probe always return False. Under
any graph-capture path those sentinels are **absent** — the acceleration comes
from a hand-rolled `torch.cuda.CUDAGraph`, not from dynamo. The probe would
return False, `engage_compile_optimizations` would report
`reason="probe_failed"`, and the **NFR7 graceful-degradation gate would disengage
the very acceleration it is meant to certify.**

> **Replacement signal.** Replace the attribute sniff with a *positive capability
> assertion on the graph objects*: after warm-up, assert `TalkerGraph.captured`
> and `PredictorGraph.captured` are True **and** run a one-shot numerical-parity
> check (one graph replay vs one eager forward, within a bf16 tolerance). This is
> strictly stronger than today's contract — it verifies the graph actually
> replays and produces sane output. `faster-qwen3-tts` documents the parity
> caveat (`StaticCache` outputs are not bit-identical to `DynamicCache` under
> BF16/TF32 because kernel reduction orders differ), so the tolerance must be a
> tolerance, not an equality.

### 6.4 The staged plan — three cheaper wins first

The spike's strongest result is that **PORT-b is not the top of the list.**

| # | Follow-up | Measured / derived value | Cost | Reversible |
|---|---|---|---|---|
| **A** | **Prime the compile cache on the warm path too.** `warmup_compile_async` currently primes only on a cold cache (§2.6); the warm path hands a **~4.0 s** in-process inductor reload to the user's first utterance. Prime it at startup with the audio callback detached. | **−4.0 s on the first generation after every launch** — the largest single term in the number Epic 18 measured | Small; the priming machinery and its audio-callback gate already exist (`qwen_tts_service.py:1982-1998`) | Yes (env-var / settings gate) |
| **B** | **Retune `chunk_size` 25 → 10** (`lookahead=5`). | **−51 % TTFA long-form** (1,785 → 875 ms) and **−44 % short** (1,651 → 921 ms); short utterances leave the residual-flush path in 5/5 runs — **measured, §5.3** | No measurable throughput cost; ratio 0.665 → 0.676. **Must thread the real geometry into `engage_compile_optimizations` (§5.5) AND move with Follow-up C (§2.7) or sub-16 GiB hosts get a worse cushion ratio** | Yes (module constant) |
| **C** | **Re-scope the sub-16 GiB cushion around `MAX_PRE_DELAY_SECONDS`.** Simulation (§2.7) shows the 10 s cap — not τ_min — is the binding escape for every `P ≲ 0.78`, and that because it is only evaluated inside `push` the effective wait at `P = 0.5` is **12.5 s**. Fix class: lower/tier the cap, evaluate it on a timer rather than on arrival, and reconsider whether a 10 s silent wait is the right product answer at all. | Derived: cushion is 2.5× the talker at the 3060's documented `P ≈ 0.5`; 4.0× if Follow-up B lands first | Larger than the one-line change the first draft proposed — this is a product decision plus a 3060 confirmation (AC #2b Phase 3) | Yes |
| **D** | **Lift the 0.5 s reference-audio silence padding** (§4.3). | Quality, not speed. **Unaudited claim** — needs its own NFR3 audition | 12 lines + a `.pt` `schema_version` bump | Yes |
| **E** | **PORT-b.** | Upper bound 2.5–2.8× on long-form, 4.8–6.7× on short — measured across a runtime boundary (§3.7b), so treat as a ceiling | Weeks; touches the audited dispatch chain; needs the §6.3 probe replacement + an NFR3 audition | No |

A + B together take the first generation after launch from ~5.8 s to ~1.0 s and
the steady state from 1,785 ms to ~875 ms — **without touching the dispatch
chain.** PORT-b's remaining marginal value should be re-measured against that
new baseline.

### 6.5 Findings 2 / 3 / 4 as independently shippable stories (Task 6.4)

* **Finding 2 (chunk sizing)** → Follow-up B. Now backed by a measured
  four-point curve **on both utterance classes**, an identified optimum, the
  §5.5 D-25 trap, and the §2.7 coupling to the cushion cap.
* **Finding 3 (leading-silence trim)** → still un-attempted and still cheap.
  Nari reports ~80 ms with no inference change. **Re-price it downward:** at
  1,785 ms TTFA it is 4.5 %; after Follow-ups A + B it is ~9 %. Worth doing,
  worth doing last.
* **Finding 4 (0.5 s reference padding)** → Follow-up D. Verified as a 12-line,
  dependency-free lift (§4.3); perceptual benefit unverified.

### 6.6 Corrections to the research memo, and one retirement

1. **Finding 5's rationale is wrong in one clause.** Mary wrote *"this is live
   for us — `emotion_profile.py` ships `repetition_penalty` values of 1.2–1.5 on
   every emotion preset."* Grep of `src/myvoice/services/` and `src/myvoice/ui/`
   finds **no call site that passes `repetition_penalty` to any generation
   entrypoint**; `EmotionProfile.repetition_penalty` is an orphaned field. We do
   take the penalty path, but at the qwen-tts library default of **1.05**
   (`qwen3_tts_model.py:380`). The finding's *conclusion* stands; its stated
   cause does not. The orphaned field is worth its own small ticket.
2. **Epic 18's causal statement is wrong** (§2.6): compile is +2.4–2.8× on
   steady-state TTFA, not −7.46 %.
3. **Retire the FA2 runtime-verification story.** `faster-qwen3-tts` tested and
   rejected SDPA/FA2 ("no RTF difference; attention not bottleneck") and custom
   CUDA kernels (8.4× isolated → 1.25× end-to-end) — a second independent source
   agreeing with research P2.B. **Recommend Commander retire the FA2
   verification story** named in
   `architecture-streaming-acceleration-and-lightning-tier.md`, citing this
   file. No spike time was spent re-testing either.

Out of scope but noted, not chased: a PORT path needs none of Story 18.5's
Triton-on-Windows bundling machinery.

### 6.7 Routing to Winston — Epic 20 architecture-pass scope sketch

> **Winston:** the Epic 20 architecture pass should decide, in this order,
> (1) whether Follow-ups A + B are taken as a single "TTFA quick wins" story
> before any architectural work — the measured case is that they deliver more
> than PORT-b and touch nothing audited; (2) whether `chunk_size` becomes a
> *ramped schedule* (small first chunk growing to 25) or a single retuned
> constant, and — load-bearing — how the real streamer geometry gets threaded
> into `engage_compile_optimizations` so the D-25 invariant stops being
> decorative (§5.5), **and** how `MAX_PRE_DELAY_SECONDS` moves in the same
> change so sub-16 GiB hosts do not get a worse cushion-to-talker ratio (§2.7);
> (3) whether a 10 s silent pre-buffer is the right product answer for the
> RTX 30xx tier at all, or whether that tier should degrade differently;
> (4) whether PORT-b's central assumption holds — that `TalkerGraph` can capture
> the **inner** `talker.model.forward` while leaving the **outer**
> `talker.forward` eager and hookable — this is the one assumption whose failure
> would flip the verdict to PORT-a or REJECT; and (5) what replaces
> `_probe_compile_engaged`'s signal contract so NFR7's gate keeps meaning
> something under graph capture (§6.3). Scope the pass against the **post-A+B
> baseline (~875 ms TTFA)**, not against today's 1,785 ms, and treat the
> 2.5–2.8× third-party figure as a **ceiling measured across a runtime
> boundary** (§3.7b), not a target.

---

## §7. AC #7 — Spike hygiene

### 7.1 Zero production BEHAVIOR change; observational surface retained by ruling

`git diff --numstat` at close (`_bmad-output/` is gitignored and excluded):

```
  6   0  src/myvoice/app.py
 32   4  src/myvoice/observability/progressive_playback_csv_capture.py
 65   0  src/myvoice/services/audio_coordinator.py
 95   0  src/myvoice/services/qwen_tts_service.py
 20   0  src/myvoice/services/tts_streaming/streaming_decoder.py
--------
218   4  src/  (the 4 deletions are the CSV module's rewritten docstring header)

173   0  tests/integration/test_streaming_tts_smoke.py
 73   2  tests/unit/observability/test_progressive_playback_csv_capture.py
112   0  tests/unit/services/test_audio_coordinator.py
 61   0  tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py
 81   0  tests/unit/services/tts_streaming/test_streaming_decoder.py
 12   2  tests/unit/test_app_progressive_playback.py
  9   0  tests/unit/test_app_progressive_playback_instrumentation.py
```

Plus **two new `tools/` drivers, 962 lines total**
(`ttfa_spike_harness.py` 648, `ttfa_spike_faster_qwen3_probe.py` 314).

Every `src/` addition is a `metrics.record` call, a one-shot guard, a comment,
or the single optional observational `session_id=None` keyword. **Per the §2.1
architect ruling this surface is retained, not reverted** — AC #7's fence is
"zero production *behavior* change", and it held.

Specifically **not** touched:

* `qwen_tts_service.py`'s dispatch chain — no control-flow edit.
* `torch_runtime.py` — **untouched**, so compile gating is unchanged.
* `codec_token_streamer.py` — **untouched**; `DEFAULT_CHUNK_SIZE` is still 25.
  The AC #5 sweep rebound `__init__.__defaults__` in-process from
  `tools/ttfa_spike_harness.py`, so there is nothing to revert (Task 7.1).
* `audio_coordinator.py`'s watermark — `_DEFAULT_STREAMING_WATERMARK_MS`,
  `_DEFAULT_STREAMING_CROSSFADE_SAMPLES`, `MAX_PRE_DELAY_SECONDS` and the
  adaptive gate all unchanged.
* `streaming_chunk_buffer.py` — **untouched**, despite §2.7 identifying a
  product-level problem in it. Routed as Follow-up C, not applied.

### 7.2 Instrumentation gate

≤ 100 µs/call verified before any timing capture — 1.03–4.28 µs (§2.2). All six
boundaries are one-shot per generation.

### 7.3 Third-party experimentation is quarantined

All `faster-qwen3-tts` work ran in the throwaway venv under the session
scratchpad. Nothing under `src/myvoice/` imports it. The two scratch drivers
live in `tools/` per AC #7; the Gate C probe installs a three-line stub module
rather than importing `myvoice`, precisely so the venv never pulls our pinned
fork onto its path (and §4.2 records that the stub proved to be dead code).

### 7.4 Tests — the retained surface is now pinned

Eight new tests, one per emission site plus the branches that were previously
unexercised:

| file | coverage added |
|---|---|
| `test_progressive_playback_csv_capture.py` | all six names in the closed-set assertion; a row-layout test pinning that `ttfa_*` rows share the unchanged header |
| `test_qwen_tts_service_true_stream_instrumentation.py` | `ttfa_generation_start_ms` fires once, is absolute wall-clock inside the dispatch window, and shares t0 with `first_chunk_latency_ms` |
| `test_streaming_tts_smoke.py` | the three talker boundaries against the **real** `_run_talker`: one-shot each, monotonic ordering, `path="threshold"`, `frames=30`; **and a `step_count=20` case pinning the `path="residual_flush"` branch** — the branch whose absence cost 6 of 11 runs on the first capture pass and which no prior fixture exercised |
| `test_streaming_decoder.py` | `ttfa_first_decode_complete_ms` fires once per session while `decode_chunk_latency_ms` still fires per chunk; a second worker re-arms (per-instance, not module, state) |
| `test_audio_coordinator.py` | `ttfa_first_playback_write_ms` fires once across two `play_audio_chunk` calls; **re-arms with a fresh session id after `stop_streaming_session`** (the C2 regression); defaults to `session_id=None` for legacy callers |
| `test_app_progressive_playback_instrumentation.py` | the session id actually propagates to the coordinator (`== "sid-arrival"`, deliberately non-None so hard-coding None cannot pass) |

**Two production defects were found by writing these tests**, which is the
argument for having them:

1. `ttfa_first_decode_step_ms` tagged `prefill_forward_calls` with the raw
   invocation counter, which is incremented *before* the call — so it reported
   **2** for a single prefill. Fixed to report the actual prefill count.
2. The threshold emit path carried **no `path` tag** — "absent" had to be read
   as "threshold". Both paths now tag explicitly.

### 7.5 Regression sweep — zero regressions

```
python310\python.exe -m pytest -q  (Story 18.1 sweep surface + the 8 new tests)
→ 328 passed in 26.63s

python310\python.exe -m pytest -q  (dispatch / session / models / streaming-buffer)
→ 186 passed in 3.29s
```

**514 tests, zero failures.** The Story 16.1 pin trip-wire
(`tests/test_qwen_tts_internals.py`) passes.

**One pre-existing failure fixed, out of scope, flagged.** The first sweep showed
2 failures in `test_app_progressive_playback.py`. **They reproduce identically on
the untouched baseline** — verified by restoring `git show HEAD:src/myvoice/app.py`
and re-running: the assertions were already stale against the `text_length=None`
kwarg that the 2026-05-15 adaptive-pre-buffer change added without updating them.
Both assertions now name both kwargs explicitly. This is a fix to someone else's
pre-existing breakage.

### 7.6 Untouched build/dependency surface

`git status` shows **no modification** to `requirements.txt`,
`build_tools/requirements-production.txt`, `build_tools/myvoice.spec`, any other
`build_tools/*` file, or the bundled `python310/` tree.

---

## §8. AC #8 — Gate accounting, with a clock

AC #8 makes the timebox load-bearing, so "closed inside budget" needs numbers.
These are **agent wall-clock, reconstructed from artifact mtimes** — not
human working days, and therefore not directly comparable to the budget's units.
The honest reading is that **no gate was budget-constrained and no gate's
question was left open for time reasons**.

| Gate | Scope | Budget (human) | Elapsed (agent wall-clock, reconstructed) | Outcome |
|---|---|---|---|---|
| **A** | AC #1 dependency probe | 1 hour | **≈ 8 min** (17:09 venv → 17:17 licence verified) | COLLIDE-SEPARABLE; MIT verified from `LICENSE` |
| **B** | AC #2 + AC #5 | 1 working day (mandatory floor) | **≈ 31 min** (17:18 instrumentation → 17:49 last clean CSV), + **13 min** for the review-response re-capture incl. B1 (18:21 → 18:34) | Both 5090 cells n=20 pooled, Phase 2 derived + simulated, four-point sweep on **both** utterance classes |
| **C** | AC #3 + AC #4 | 1 working day (skippable) | **≈ 9 min** (5 min cu128 torch install + 4 min benchmarks, 17:48 → 17:52), + 1 min quiet re-run (18:34 → 18:35) | 0.6B sanity + 1.7B + like-for-like + short + quiet re-run; mode parity partially verified (§4.1) |

Verdict, write-up, regression and review response are outside these three
figures. No gate overran; no budget was silently extended.

**One measurement-integrity note.** A first pass of the long-form cell and the
`chunk_size=5/10/15` sweep was captured while a 3 GB `pip` download was running
concurrently, and showed a 2.3× slowdown (TTFA 2,100–3,400 ms, ratio 0.94–1.38).
Those runs were **discarded**; the contaminated first pass is retained at
`implementation-artifacts/20-1-*.csv` for audit. The two *clean* passes
(`clean/`, `clean2/`) are what §2 and §5 report, pooled.

---

## §9. Artifacts

Force-added per `memory/git_repo_state.md`:

| path | contents |
|---|---|
| `clean/` and `clean2/` `20-1-ttfa-rtx5090-{long,short}-cs25.csv` | AC #2b Phase 1 cells A and B, two passes (n=10 + warmup each) |
| `clean/`, `clean2/` `20-1-sweep-long-cs{5,10,15}.csv` | AC #5 long sweep, two passes (n=5 each) |
| `clean2/20-1-sweep-short-cs{5,10,15}.csv` | **B1** short sweep (n=5 each) |
| `clean*/20-1-coldwarm-long-cs25-{eager,compile}.csv` | §2.6 Epic 18 reconciliation (n=4 each, first run recorded) |
| `clean*/*.log`, `20-1-recapture-timeline.txt` | stdout/stderr and per-cell timestamps |
| `gatec/20-1-fq3-*.json`, `gatec2/…-requiet.json` | AC #3 + AC #4 raw records, incl. the quiet-window re-run |
| `20-1-aggregate-ttfa.py`, `20-1-aggregate-output.txt` | the aggregator behind every table in §2 and §5 |
| `20-1-adaptive-cushion-sim.py`, `20-1-adaptive-cushion-sim.txt` | §2.7 simulation against the shipped `StreamingChunkBuffer` |
| `20-1-regression-sweep.log` | AC #7 regression evidence |
| `20-1-*.csv`, `20-1-run-*.log` (non-`clean*`) | the discarded contaminated first pass, retained for audit |
| `tools/ttfa_spike_harness.py`, `tools/ttfa_spike_faster_qwen3_probe.py` | tracked source (new) |

---

## §10. Review-response change log (2026-08-31)

Architect review returned no loopback on the measurements; the corrections were
to framing, two recommendations, and missing coverage. What changed:

**Claim corrections.** Every "p95" was actually the maximum
(`round(0.95·(n−1))` = `n−1` for n ≤ 10) — replaced with an interpolated
type-7 quantile in both drivers and the aggregator, with `max` reported
alongside (A1). The AC #2 sum-reconciliation is now stated as an **identity**
and the ±10 % gate recorded as an unfalsifiable spec defect, with a genuine
`perf_counter` bracket added as the real check (A2, §2.4). `first_chunk_latency_ms`
is labelled a **restatement** (A3). Segment-4 / "perceived TTFA" collapsed into
one vocabulary — TTFA(post) / TTFA(release) / "first audible sample" (A10).
Sub-clock-resolution values print `<=res` (A11). Per-key n published (A12).
Gate elapsed times added (A13, §8). Hygiene stat now includes the `tools/`
drivers (A15). AC #2b cells C and D named as unmeasured (A16).

**Recommendations that did not survive their own evidence.** Follow-up C was
re-scoped: simulation against the shipped buffer shows `MAX_PRE_DELAY_SECONDS`,
not τ_min, is the binding escape for every `P ≲ 0.78`, so the proposed
`τ_min·P` one-liner does not address the operating point that justified it
(A4, §2.7). The T_a estimator's ~45 % overshoot is now its own enumerated
finding, with the caveat that it changes nothing at `P = 0.5` (A5). CustomVoice
and VoiceDesign downgraded from "works" to **unverified** — the probe only
checked `callable(getattr(...))`, and `generate()` proves that is not evidence
(A6, §4.1). Gate C confounds stated and partly closed by a quiet-window re-run
(A8, §3.7). The short-utterance comparison promoted into the headline (A9, §0).

**The missing experiment (B1).** The short-utterance chunk-size sweep now
measures what was previously inferred: `chunk_size = 10` pulls short utterances
off the residual-flush path in 5/5 runs and cuts TTFA 44 % (§5.3).

**Code.** Residual-flush metric moved out of the guarded try (C1);
segment-4 state re-armed on `stop_streaming_session` (C2); eight new tests
covering all six metrics and both emit paths (C3); docstrings corrected (C4);
harness drain-before-unsubscribe, try/finally around the run loop, and a guarded
RTF (C5). Writing C3 surfaced two real instrumentation defects (§7.4).

**Ruling recorded.** The six metrics, the two coordinator fields and the
`session_id` keyword are permanent product surface by architect decision, which
is why they now carry tests and documented contracts (D1, §2.1).
