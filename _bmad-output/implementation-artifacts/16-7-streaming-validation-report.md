# Story 16.7 — Streaming-Default Validation Report

**Date:** 2026-05-08
**Hardware:** RTX 5090 Blackwell + Win11 + torch 2.10.0+cu128 (per `memory/hardware_setup.md`)
**qwen-tts pin:** 0.0.4
**Maintainer:** Commander (`memory/user_role.md` not yet present; sole contributor)

---

## 1. Executive summary

**RECOMMENDATION: `FAIL-UPSTREAM-STREAMING` — DO NOT FLIP THE STREAMING DEFAULT.**

The empirical validation gate Story 16.7 was created to run revealed two
independent failure modes that together block the streaming-default flag flip:

1. **TRUE_STREAM is structurally broken.** Story 16.6's `_build_true_stream_talker`
   calls `model.model.generate(streamer=streamer)` with no text/speaker/language
   conditioning (`qwen_tts_service.py:2522`). The wrapper raises
   immediately on every call; the talker thread silently swallows the exception
   and exits. **51 of 51 TRUE_STREAM measurements failed with 0 chunks emitted
   (100% failure rate).** This was hidden in production by the existing
   exception-swallow in `_run_talker` and surfaced by Story 16.7's first
   empirical run against the real qwen-tts model (the smoke tests at
   `tests/integration/test_streaming_tts_smoke.py:559,674,...` all monkey-patched
   the talker, so the real-model path had never been exercised).
2. **SENTENCE_STREAM does NOT meet NFR1 across the input set on either GPU or
   CPU.** Architecture line 802 framed NFR1 satisfaction as "GPU: meets via
   TRUE_STREAM (~1.5–1.8s estimated). CPU: meets via inherited SENTENCE_STREAM."
   The empirical reality on this host with qwen-tts 0.0.4 is:
   - **GPU SENTENCE_STREAM: p95 = 18.143s** (9.07× the 2.000s NFR1 ceiling).
     Even short-class utterances (≤30 chars) hit p95 = 6.169s; only 7/16 of
     them met NFR1 individually.
   - **CPU SENTENCE_STREAM: p95 = 4.593s** (2.30× the ceiling). All 10 short-class
     baseline measurements were over the 2s ceiling.

**Code change implied (one-line PR):** None directly from this report — but
two follow-ups needed:
- **Story 16.8 (TRUE_STREAM real wire-up):** replicate the wrapper's
  preprocessing in `_run_talker` so `model.talker.generate(inputs_embeds=...,
  streamer=streamer, ...)` is callable with full conditioning, OR wait for
  upstream qwen-tts to forward the `streamer` kwarg through
  `Qwen3TTSForConditionalGeneration.generate`'s `**kwargs` to the inner talker
  at `qwen_tts/core/models/modeling_qwen3_tts.py:2272-2278`.
- **Story 16.9 (NFR1 reconciliation):** investigate why SENTENCE_STREAM on GPU
  yields p95 = 18s instead of the architecture's projected sub-2s. Possible
  causes (in priority order to investigate): (a) qwen-tts 0.0.4 is materially
  slower than the version the architecture's estimates were based on; (b) the
  CUSTOM_VOICE model is slower than projected and the model-tier fallback to
  0.6B is the route to NFR1 compliance; (c) `_generate_streaming`'s
  sentence-split granularity is too coarse for short Discord-call patter; (d)
  NFR1 itself was always overly optimistic and the contract needs revision.

**This story does NOT recommend flipping any default flag.** The Story 16.6
graceful-degradation guard added in this story (`qwen_tts_service.py:2845-2861`)
ensures CUDA users get SENTENCE_STREAM-rendered audio (not silence) until
Story 16.8 lands; that is the production-safe state and should remain.

---

## 2. Methodology

### 2.1 Hardware + software pin

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell) |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.10.11 (bundled portable `python310/python.exe`) |
| torch | 2.10.0+cu128 |
| CUDA toolkit | v12.8 |
| qwen-tts pin | 0.0.4 |
| MyVoice branch | epic-16 (Story 16.7 in-progress) |

### 2.2 Input set

`_bmad-output/implementation-artifacts/16-7-input-set.csv` — 51 utterances:

- **17 short** (≤30 chars): Discord-call patter ("Got it.", "Hold on.",
  "Mic check.") + 4 perceptual-difficult tongue-twisters
  ("Bit, bat, bot, but, bet.", "Sip ship sap shop sup.")
- **17 medium** (30–150 chars): full sentences from Discord-meeting context
  + 4 perceptual-difficult sibilant-rich items
  ("She sells seashells by the seashore.")
- **17 long** (150+ chars): multi-sentence narrative from
  Discord-standup / post-mortem / planning context + 2 sustained-sibilant
  long items
- **10 perceptual-difficult** total (4 short + 4 medium + 2 long)

### 2.3 Measurement protocol

`scripts/validate_streaming_default.py` — standalone harness invoking the
production code path with mocked AudioCoordinator sinks (no audio actually
played to devices) so wall-clock measures dispatch + decode latency only.
For each utterance:

1. Construct `QwenTTSRequest(text=..., language="English",
   model_type=CUSTOM_VOICE, speaker="Ryan", streaming=True)`.
2. Call the appropriate generator method directly (per the AC #5 Decision
   recorded in this story's Change Log — direct generator calls give clean
   apples-to-apples; the no-override path uses the public dispatch and emits
   `streaming_mode` / `streaming_mode_fallback` metrics).
3. Read `response.first_chunk_latency` (seconds; `None` if no chunks
   emitted), `response.audio_data` (numpy float32 PCM at
   `response.sample_rate`), and `response.mode`.
4. Record one `MeasurementResult` row to the CSV.

The harness explicitly refuses TRUE_STREAM on a CPU host (`SystemExit` per
AC #3 / D-9 / NFR12 protection).

### 2.4 Perceptual A/B audition protocol

**NOT EXECUTED** — Task 6 deferred per AC #2's defer condition. After the
silent-TRUE_STREAM finding (section 3.1), running the audition would have
compared two SENTENCE_STREAM-fallback renditions instead of TRUE_STREAM vs
SENTENCE_STREAM. With both A and B drawn from the same code path the audition
provides no signal on overlap-add seam quality (the architectural concern at
NFR3). The audition is deferred to Story 16.8 once real TRUE_STREAM streaming
is wired.

---

## 3. GPU latency results

### 3.1 TRUE_STREAM (`16-7-gpu-latency-measurements.csv`, n=51)

| Metric | Value |
|---|---|
| Rows with `error_flag == ""` | 0 |
| Rows with `error_flag == "RuntimeError('TRUE_STREAM produced 0 audio chunks ...')"` | 51 |
| Fallback rate (talker failure) | 100% |
| p50 first-chunk latency | N/A (no successful measurements) |
| p95 first-chunk latency | N/A |
| **NFR1 GATE** | **FAIL — gate cannot evaluate; TRUE_STREAM dispatch is structurally broken** |

Every utterance hit the empty-chunks guard added in this story
(`qwen_tts_service.py:2845-2861`). The fix was deliberately surfacing the
silent-failure mode that Story 16.6's wire-up created; the harness's role
here was to confirm that the silent-failure path triggers reliably across all
input classes (it does — short, medium, and long all fail identically). For
the architectural framing this is the empirical answer to architecture line
905's "Medium for Phase ⊥ (streaming) — the only meaningful uncertainty is
empirical": the answer is "TRUE_STREAM as Story 16.6 shipped does not
function against the real qwen-tts model; a follow-up to wire conditioning +
streamer propagation is required."

### 3.2 SENTENCE_STREAM apples-to-apples (`16-7-gpu-sentence_stream-comparison.csv`, n=50)

| Class | n | p50 | p95 | max | NFR1 (p95<2s)? |
|---|---|---|---|---|---|
| short | 17 | 2.031s | 6.169s | 6.834s | **FAIL** |
| medium | 17 | 5.782s | 10.087s | 11.002s | **FAIL** |
| long | 16 | 14.260s | 22.157s | 25.253s | **FAIL** |
| **Overall** | **50** | **6.136s** | **18.143s** | **25.253s** | **FAIL** |

Top 5 slowest measurements (all long-class, 158–212 chars):

| utterance_id | chars | first_chunk_latency_seconds |
|---|---|---|
| l-007 | 212 | 25.253 |
| l-005 | 197 | 21.125 |
| l-014 | 158 | 18.182 |
| l-010 | 207 | 18.095 |
| l-015 | 190 | 16.778 |

**Per-class compliance with NFR1 (p95 < 2.000s):**
- short: 7 of 16 individual measurements met NFR1 (44%, after dropping the
  s-001 warmup outlier at 6.003s).
- medium: 0 of 17 measurements met NFR1.
- long: 0 of 16 measurements met NFR1.

**Note on the harness's `error_flag` column:** every row in this CSV is
flagged `fallback_occurred` due to a harness classifier bug (false positive)
where `GenerationMode.STREAMING.value` ("streaming") was compared against
`StreamingMode.SENTENCE_STREAM.value` ("sentence_stream") and the mismatch
inferred a fallback. The latency numbers themselves are valid — these are
real successful direct `_generate_streaming` calls. The harness fix landed
in the same Story 16.7 cycle (`scripts/validate_streaming_default.py`
`_classify_dispatched_mode`) so future runs produce clean CSVs; the existing
CSV is preserved as-committed for reproducibility per AC #7.

### 3.3 No-override resolver path (not run)

The harness was not invoked without `--mode-override` against the production
resolver in this round; the silent-TRUE_STREAM finding made the resolver
path's measurement redundant (it would record N rows where TRUE_STREAM falls
back through the now-graceful chain to SENTENCE_STREAM, identical to section
3.2's data with one extra ~ms per row for the resolver pass).

---

## 4. Perceptual A/B results

**STATUS: DEFERRED to Story 16.8 (per AC #2's defer condition).**

The fixture builder (`scripts/build_streaming_perceptual_ab_fixture.py`) ran
on the maintainer's host and produced 10 paired WAV files. The user's first
audition pass observed every "A" file (canonically the TRUE_STREAM rendition)
to be silent and every "B" file (SENTENCE_STREAM) to play normally. That
finding is what surfaced the Sev-1 silent-audio bug documented in section 3.1
and is also what made the audition itself non-informative for the
architecturally-named gate (NFR3 — overlap-add seam detection). Without a
working TRUE_STREAM path producing audible output, A/B comparison cannot
test seams.

The audition is deferred until Story 16.8 lands real TRUE_STREAM streaming;
at that point the fixture builder can be re-run and the audition repeated
under the existing `LISTENING-INSTRUCTIONS.md` protocol.

---

## 5. CPU baseline confirmation

`16-7-cpu-baseline-measurements.csv`, n=10, all short-class.

| Metric | Value | NFR1 (<2s)? |
|---|---|---|
| min | 2.041s | FAIL |
| p50 | 2.739s | FAIL |
| p95 | 4.593s | FAIL |
| max | 4.897s | FAIL |
| mean | 3.037s | FAIL |

**0 of 10 short-class measurements** met NFR1 on CPU. The architecture's
"NFR1 satisfaction on CPU is therefore inherited [from V2 baseline], not
promised by streaming" framing is contradicted by this measurement —
SENTENCE_STREAM on CPU does not deliver sub-2s first audio with qwen-tts
0.0.4 + the CUSTOM_VOICE model + Discord-call-length inputs.

This is a SEPARATE finding from the TRUE_STREAM gate but is a **release
blocker** for the streaming-default flag flip even after Story 16.8 lands —
flipping the default to TRUE_STREAM only helps GPU users; CPU users still
need NFR1 compliance via SENTENCE_STREAM and that compliance is not
empirically verified.

**Note on sample size:** the CPU run was 10 utterances (per AC #3's "≥10
records" floor), all short-class. medium- and long-class CPU measurements
are not yet captured. A follow-up CPU run extending to medium + long would
strengthen the inheritance-violation finding but is not gating; the failure
is already empirically demonstrated for short-class which is the easiest
case for SENTENCE_STREAM to satisfy.

---

## 6. Recommendation

**`FAIL-UPSTREAM-STREAMING` — leave streaming default unchanged.**

### 6.1 The streaming-default flag flip is blocked

Story 16.7 was scoped to produce evidence that would inform the streaming-
default-flag flip (the one-line edit at `streaming_mode.py:54-56` or the
settings UI's first-launch initializer). The evidence produced is
unambiguous: **DO NOT flip.**

Reasons:

1. **TRUE_STREAM dispatch is structurally broken** — every measurement on
   the maintainer's RTX 5090 host failed with the empty-chunks guard
   triggered. Flipping the default would route 100% of CUDA users into
   the fallback chain (now graceful per the Story 16.7 fix) but provides
   no actual streaming benefit until Story 16.8 lands.
2. **SENTENCE_STREAM does not meet NFR1 on either path** — flipping the
   default cannot fix this; it's a separate problem that needs its own
   investigation.

### 6.2 Production safety net stays in place

The `_generate_true_stream` empty-chunks guard added in this story
(`qwen_tts_service.py:2845-2861` + regression tests at
`tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure`)
ensures any production CUDA user whose `streaming_mode_override` resolves to
TRUE_STREAM (default behavior on CUDA per `streaming_mode.py:54-56`) hears
SENTENCE_STREAM-rendered audio rather than silence. The architecturally-named
fallback chain (NFR7) is now the production-safe path. **No further code
change is needed in this story.**

### 6.3 Follow-up stories named

| Story | Scope | Priority |
|---|---|---|
| **16.8 — Real TRUE_STREAM wire-up** | Replicate the qwen-tts wrapper's preprocessing in `_run_talker` (build `talker_input_embeds`, `trailing_text_hiddens`, etc. from `request.text` + `request.speaker` + `request.language`); call `model.talker.generate(inputs_embeds=..., streamer=streamer, ...)` directly. Re-run Story 16.7's harness on the result. Re-run perceptual A/B audition. | High — gates the streaming-default flag flip |
| **16.9 — NFR1 reconciliation** | Investigate why SENTENCE_STREAM on GPU yields p95 = 18s. Profile the existing `_generate_streaming` path (sentence split, model invocation, decode); compare against the architecture's projected ~1.5–1.8s ceiling. Decide if NFR1 needs revision OR if the implementation has a regression. | High — release blocker |

These two stories are independent and can be worked in parallel. Story 16.8
unblocks the architectural Phase ⊥ track; Story 16.9 unblocks the entire
release of the streaming optimization pass.

### 6.4 What this story did NOT recommend

- **No flag flip** — the only literal code change implied by the gate failure
  is "leave `streaming_mode.py:54-56` as-is" (it already defaults to
  TRUE_STREAM on CUDA, which now graciously falls back).
- **No tighten-chunk-size** — Story 16.6's `codec_token_streamer.py`
  `DEFAULT_CHUNK_SIZE = 25` / `DEFAULT_LOOKAHEAD = 5` are not exercised in
  this report's measurements because TRUE_STREAM never produced any chunks
  to overlap-add.
- **No D-8 dedicated CUDA stream** — the architecture's D-8 follow-up
  ("dedicated `torch.cuda.Stream` for the decoder") is moot until TRUE_STREAM
  produces tokens to decode.

---

## 7. Reproducibility

### 7.1 Exact commands run (this report's source data)

```cmd
REM --- GPU TRUE_STREAM measurement (Task 4) ---
python310\python.exe scripts\validate_streaming_default.py ^
    --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv ^
    --output-dir _bmad-output\implementation-artifacts\ ^
    --mode-override true_stream

REM --- GPU SENTENCE_STREAM apples-to-apples (Task 4) ---
python310\python.exe scripts\validate_streaming_default.py ^
    --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv ^
    --output-dir _bmad-output\implementation-artifacts\ ^
    --mode-override sentence_stream

REM --- CPU SENTENCE_STREAM baseline (Task 5) ---
set CUDA_VISIBLE_DEVICES=-1
python310\python.exe scripts\validate_streaming_default.py ^
    --input-set _bmad-output\implementation-artifacts\16-7-input-set.csv ^
    --output-dir _bmad-output\implementation-artifacts\ ^
    --mode-override sentence_stream --utterance-count 10
set CUDA_VISIBLE_DEVICES=
```

### 7.2 Exact source artifacts

- `scripts/validate_streaming_default.py` (committed) — the harness
- `scripts/build_streaming_perceptual_ab_fixture.py` (committed) — the
  fixture builder (not-fully-exercised due to TRUE_STREAM bug)
- `_bmad-output/implementation-artifacts/16-7-input-set.csv` (committed) —
  fixed 51-utterance input set
- `_bmad-output/implementation-artifacts/16-7-gpu-latency-measurements.csv`
  (committed) — TRUE_STREAM run output
- `_bmad-output/implementation-artifacts/16-7-gpu-sentence_stream-comparison.csv`
  (committed) — apples-to-apples GPU SENTENCE_STREAM output
- `_bmad-output/implementation-artifacts/16-7-cpu-baseline-measurements.csv`
  (committed) — CPU baseline output

### 7.3 Hardware reproducibility

The reported latency numbers depend on:
- **GPU model**: RTX 5090 Blackwell. Re-running on Ampere (RTX 30xx) or
  Ada Lovelace (RTX 40xx) is expected to be slower for SENTENCE_STREAM
  (architecture's framing assumed this is roughly hardware-agnostic at the
  >2s level, which is now suspect — Story 16.9 should re-measure on at least
  one Ampere card to scope the regression).
- **CUDA toolkit**: v12.8 to match the bundled torch wheel (cu128).
  Mismatched CUDA versions may exhibit different latency.
- **qwen-tts pin**: 0.0.4. The next public release of qwen-tts may ship
  the streamer-aware wrapper that Story 16.8 needs; at that point this
  report should be re-run against the new pin and Story 16.6's
  trip-wire test (`tests/test_qwen_tts_internals.py`) updated accordingly.

### 7.4 Software reproducibility

To re-run from scratch on a clean RTX 5090 + Win11 + bundled python310:

```cmd
git checkout epic-16
git pull
REM (commit 16-7 has the harness + input set; uncommitted changes apply
REM the empty-chunks guard fix and the regression tests)
python310\python.exe -m pytest tests\integration\test_streaming_tts_smoke.py ^
    tests\unit\services\test_qwen_tts_service_dispatch.py
REM (expect 52 passed)
REM Then run the three commands in section 7.1.
```

The committed regression tests at
`tests/integration/test_streaming_tts_smoke.py::TestSilentTalkerSurfacesAsFailure`
are the canonical guard against the silent-audio bug regressing in any
future Story 16.8 / 16.9 work.

---

*Report authored 2026-05-08 by claude-opus-4-7[1m] for Story 16.7 Task 7.
Source CSVs are the authoritative data; the report's tables are derived
from them via the script in this story's Change Log entry.*
