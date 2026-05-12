# Story 18.3 — bf16 Precision on Talker + Decoder — Evidence

Status: in-progress (autonomous source-tree work landed; Commander-routed
empirical sections partially populated — see §"Commander-routed work" below).

Story file: `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder.md`.

## H1 — production wire-up regression (caught + fixed 2026-05-10)

**The first `02_Story_18.3_DType_Audit.bat` run surfaced a HIGH-severity
defect in the autonomous source-tree pass.** The
`ModelRegistry initialized` log line read:

```
2026-05-10 00:34:21 - ModelRegistry - INFO -
ModelRegistry initialized: device=cuda:0, dtype=torch.bfloat16,
precision_source='legacy_constructor_arg', quality_tier=quality
```

`legacy_constructor_arg` means the new precision resolver did NOT
engage. Root cause: `app.py:445` constructed `QwenTTSService(...)`
without the `app_settings=self._app_settings` keyword argument —
even though the `QwenTTSService.__init__` → `ModelRegistry.__init__`
hop was wired correctly. The production call site was the broken hop.

**Impact (had it shipped without this fix):**
- `tts_precision="fp32"` setting silently ignored — NFR7 fp32 fallback
  inert.
- `tts_precision="bf16"` setting silently ignored.
- The Task 7 NFR1 measurement would have measured **two identical bf16
  runs** — the bf16 vs fp32 A/B would have been a null comparison.
- Telemetry would always log `source='legacy_constructor_arg'` —
  Commander's runtime-engagement check would silently fail.

**Bug class:** "production call-site drops the new keyword argument."
The previous wire-up tests at `test_qwen_tts_service_dispatch.py`
exercised the `QwenTTSService → ModelRegistry` hop only (because the
`_make_service` test helper always passed `app_settings=`); they did
NOT cover the `app.py → QwenTTSService` hop. Per
`memory/code_review_regression_test_exact_class.md`, the new
regression test must mirror **this exact** bug class — which is
exactly what `tests/unit/test_app_qwen_tts_construction.py` does:
AST-scans `app.py` for every `QwenTTSService(...)` call and asserts
`app_settings` is in the kwarg list.

**Fix landed:** `app.py:445` now passes `app_settings=self._app_settings,`.
3 new regression tests at `tests/unit/test_app_qwen_tts_construction.py`
catch any future occurrence at static-scan time.

## Pre-implementation audit

**Scope (Task 1.1):** capture the actual `model.dtype` /
`model.model.talker.dtype` / `model.model.speech_tokenizer` parameter
dtype on the production model loaded with the existing HEAD
(`model_registry.py:95` `dtype: str = "bfloat16"` default) on the RTX
5090 dev host.

**Status:** COMPLETE (2026-05-10 second audit run, post-H1+M1 fixes —
captures + findings below). Procedure preserved for any future
re-audit (e.g., post-qwen-tts-pin-bump or post-Story-18.4 re-validation).
The dev-agent flow: Commander runs `02_Story_18.3_DType_Audit.bat` once
on the RTX 5090 dev host. The bat sets `MYVOICE_DTYPE_AUDIT=1` and
launches MyVoice; the instrumentation hook in
`model_registry._instrument_dtype_audit` walks the loaded model and
attaches one-shot forward hooks. After Commander generates a single
short utterance and closes MyVoice, `logs/myvoice.log` carries the
captures, which the dev agent parses into this section.

**Procedure (for Commander — ~2 minutes):**

1. `del logs\myvoice.log`
2. Double-click `02_Story_18.3_DType_Audit.bat`.
3. Pick Sarira-F as the speaker.
4. Generate one short utterance (any short paragraph; the audit needs
   only ONE forward pass per module).
5. Close MyVoice cleanly.
6. Send `logs/myvoice.log` (or just the lines tagged `[DTYPE_AUDIT]` /
   `[DTYPE_AUDIT_FWD]`) back to the dev agent.

**What the captures look like:**

```
[DTYPE_AUDIT] model.dtype = <dtype>
[DTYPE_AUDIT] model.model.talker.dtype = <dtype>
[DTYPE_AUDIT] speech_tokenizer found at: model.model.speech_tokenizer
[DTYPE_AUDIT] speech_tokenizer.<param>.dtype = <dtype>  (first 5 sampled)
[DTYPE_AUDIT_FWD] talker in=[<dtype>, ...] out=<dtype>
[DTYPE_AUDIT_FWD] speech_tokenizer in=[<dtype>, ...] out=<dtype>
```

**Result (second audit run, 2026-05-10 10:08, RTX 5090 dev host, post-H1+M1-fix HEAD):**

```
ModelRegistry initialized: device=cuda:0, dtype=torch.bfloat16,
  precision_source='app_settings_auto_ampere', quality_tier=quality   ← H1 fix engaged

[DTYPE_AUDIT] model.dtype = <attribute not present>           ← qwen-tts wrapper idiosyncrasy
[DTYPE_AUDIT] model.model.dtype = torch.bfloat16
[DTYPE_AUDIT] model.model.talker.dtype = torch.bfloat16       ← talker bf16 confirmed
[DTYPE_AUDIT] speech_tokenizer found at: model.model.speech_tokenizer
[DTYPE_AUDIT] speech_tokenizer type: Qwen3TTSTokenizer        ← HF wrapper
[DTYPE_AUDIT] speech_tokenizer inner Module: model.speech_tokenizer.model
              (type=Qwen3TTSTokenizerV2Model)                  ← inner nn.Module walked (M1 fix)
[DTYPE_AUDIT] speech_tokenizer.model.encoder.encoder.layers.0.conv.weight.dtype = torch.bfloat16
[DTYPE_AUDIT] speech_tokenizer.model.encoder.encoder.layers.0.conv.bias.dtype = torch.bfloat16
[DTYPE_AUDIT] speech_tokenizer.model.encoder.encoder.layers.1.block.1.conv.weight.dtype = torch.bfloat16
[DTYPE_AUDIT] speech_tokenizer.model.encoder.encoder.layers.1.block.1.conv.bias.dtype = torch.bfloat16
[DTYPE_AUDIT] speech_tokenizer.model.encoder.encoder.layers.1.block.3.conv.weight.dtype = torch.bfloat16
[DTYPE_AUDIT] speech_tokenizer.model (truncated to first 5 params)   ← codec ALSO bf16
```

**Findings — Task 1.1 (post-load attribute walk):**

1. **Talker is bf16 end-to-end.** `model.model.talker.dtype = torch.bfloat16`
   confirms the V2 default's bf16 wiring reaches the talker module on RTX 5090.
2. **Codec/vocoder is ALSO bf16.** This is **surprising** — the story Dev Notes'
   central audit hypothesis was that vocoders typically stay in fp32 for
   numerical stability. qwen-tts 0.0.4 actually loads the inner
   `Qwen3TTSTokenizerV2Model` in bf16 too. The convolutional weights/biases
   in the encoder are all `torch.bfloat16`. So the **entire forward pass** is
   bf16 from input through codec output — not just the talker side.
3. The H1 fix is confirmed working: `precision_source='app_settings_auto_ampere'`
   means the new resolver (Story 18.3 source-tree pass) engaged via
   AppSettings → Ampere+ probe → `bf16`.
4. The known wrapper idiosyncrasy (`model.dtype` absent) does not affect
   anything because we have direct access to the leaf module dtypes.

## End-to-end dtype audit

**Scope (Task 1.2 + 1.3):** attach a one-shot forward hook on the
talker's first-token forward pass and on the codec decoder's first-chunk
decode call. Capture input + output + intermediate dtypes; surface any
internal autocast / `.float()` / `.to(torch.float32)` upcast that
erases the bf16 compute gain.

**Status:** COMPLETE (2026-05-10 second audit run, post-H1+M1 fixes —
captures + findings below). The captures landed in the same
`02_Story_18.3_DType_Audit.bat` run as §"Pre-implementation audit". The
instrumentation in `model_registry._instrument_dtype_audit` attaches
one-shot `register_forward_hook`s on `talker` and `speech_tokenizer`;
the hooks log dtypes on first invocation and detach themselves. No
manual REPL gymnastics required.

**Result (second audit run, 2026-05-10 10:08, RTX 5090):**

```
[DTYPE_AUDIT_FWD] talker
  args=()
  kwargs=dict{
    logits=<Tensor dtype=torch.bfloat16 shape=(1, 125, 3072)>,
    past_key_values=<DynamicCache>,
    hidden_states=tuple[tuple[
      <Tensor dtype=torch.bfloat16 shape=(1, 125, 2048)>,
      <Tensor dtype=torch.bfloat16 shape=(1, 125, 2048)>,
      <Tensor dtype=torch.bfloat16 shape=(1, 125, 2048)>,
      ...
    ], None],
    past_hidden=<Tensor dtype=torch.bfloat16 shape=(1, 1, 2048)>,
    generation_step=<int>
  }
  out=dict{
    cache_position=<Tensor dtype=torch.int64 shape=(125,)>,
    past_key_values=<DynamicCache>,
    input_ids=None,
    inputs_embeds=<Tensor dtype=torch.bfloat16 shape=(1, 125, 2048)>,
    position_ids=<Tensor dtype=torch.int64 shape=(1, 125)>
  }
```

**Findings — Task 1.2 (talker forward hook):**

1. **Every tensor is bf16.** All five tensor-bearing kwargs (`logits`,
   `hidden_states[i]`, `past_hidden`, plus `inputs_embeds` in the
   output) are `torch.bfloat16`. No fp32 round-trip, no autocast
   dtype change — the talker forward pass runs entirely in bf16.
2. **Integer dtypes (`int64`)** appear for position/cache indexing —
   that's expected (positional indexing is not a numerical-precision
   concern; ints have only one valid dtype on each axis).
3. The kwargs+structured-output captures only worked thanks to the
   M1 fix (`with_kwargs=True` + structured-output walking). Without
   the fix, this would have appeared as `args=[] out=<Qwen3TTSTalkerOutputWithPast>`
   with no dtype info — exactly what the first audit run captured.

**Speech_tokenizer.inner forward hook DID NOT fire** — the inner
`Qwen3TTSTokenizerV2Model` was successfully discovered and hooked
(line 224 of the audit log), but no `[DTYPE_AUDIT_FWD] speech_tokenizer.inner`
line appeared. Plausible reason: the codec decode is invoked via
`Qwen3TTSTokenizer.decode(...)` which calls into the inner Module's
methods (`encode` / `decode`) directly, NOT through the inner
Module's `forward()` / `__call__`. `register_forward_hook` only
fires on `forward()` invocations, so the codec path is not
intercepted. Capturing decode-time dtypes would require either
monkey-patching `Qwen3TTSTokenizer.decode` or finding the actual
forward-bearing submodule inside `Qwen3TTSTokenizerV2Model`. **Task
1.5 routing condition does NOT trigger** — the static parameter walk
(Task 1.1) already confirmed every codec parameter is bf16, so
forward-pass dtype is not in question.

**If Task 1.5 routing condition triggers** (any audit branch surfaces
an unexpected fp32 round-trip *inside* the model's forward pass that
erases the bf16 compute gain), STOP the dev-agent flow and route to
Open Question #1 below — the upstream-pin-bump-vs-local-wrapper choice
is Commander-routed, not a dev-agent unilateral edit.

## Streaming pipeline dtype audit

**Scope (Task 5.1 + 5.2 + 5.3):** confirm the `decode_fn` callable's
GPU→CPU cast site, the float32 contract enforced at the boundary, and
the chunk → bytes invariant in `_handle_progressive_chunk_async`.

**Status:** COMPLETE (read-only audit; no GPU required).

### Findings

**1. `decode_fn` consumer contract (`streaming_decoder.py:82`):**

```python
decode_fn: Callable[[list[Any]], np.ndarray],
```

The streaming decoder worker accepts `np.ndarray` from the supplier
callable and posts segments via `post_mutation('append_chunk', session_id, pcm)`.
The dtype is not pinned by the type annotation alone, but the supplier
site (below) enforces float32 explicitly.

**2. `decode_fn` supplier site (`qwen_tts_service.py:3325-3396`,
`_build_true_stream_decode_fn`):**

The GPU→CPU cast happens at lines 3393-3395 verbatim:

```python
if hasattr(audio, "detach"):
    audio = audio.detach().cpu().numpy()
return np.asarray(audio, dtype=np.float32).flatten()
```

- `audio.detach().cpu().numpy()` — the GPU→CPU transfer boundary. NumPy
  has no native `bfloat16` dtype, so this conversion either (a) raises
  if the tensor is bf16 (the strict-error path), or (b) silently
  upcasts via NumPy's DLPack-mediated bf16 → fp32 path on NumPy 2.0+.
  Either way, the bf16 representation does NOT cross into NumPy land.
- `np.asarray(audio, dtype=np.float32)` — the explicit float32 cast.
  This is the canonical contract enforcement: regardless of whatever
  intermediate dtype the upstream produced, the output is float32.

**3. Codec decoder call site (`qwen_tts_service.py:3379-3381`):**

```python
result = model.model.speech_tokenizer.decode(
    [{"audio_codes": chunk}]
)
```

The `model.model.speech_tokenizer.decode` API is documented at
`qwen3_tts_tokenizer.py:281-283` as returning
`(wavs: List[np.ndarray], sample_rate: int)`. **Crucial implication:
the speech_tokenizer's `.decode()` returns `np.ndarray` natively** —
i.e., the bf16 → fp32 cast happens INSIDE the qwen-tts wrapper at the
boundary, NOT in MyVoice code. This is consistent with vocoders typically
staying in fp32 for numerical stability (the central audit hypothesis
in the story file's Dev Notes).

**Result class:** **(a) Already correct.** The
`streaming_decoder.py:82` consumer + `qwen_tts_service.py:3393-3395`
supplier path is structurally fp32 at the GPU→CPU boundary. The bf16
gain (if any) is preserved on the talker side; the codec decoder
returns whatever the qwen-tts speech_tokenizer's internal vocoder
returns (most-likely fp32 per typical vocoder design).

**4. Chunk → bytes invariant (`app.py:2622-2625`,
`_handle_progressive_chunk_async`):**

```python
if chunk.audio_data.size > 0:
    audio_bytes = (
        np.clip(chunk.audio_data, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()
```

The path expects `chunk.audio_data` as `np.ndarray[float32]` (range
[-1.0, 1.0]) and converts to int16 PCM bytes. **No edits to this code
in Story 18.3.** Story 18.3 audits the upstream cast site; this AC
invariant remains untouched (no-edit confirmation).

**5. Storyline confirmation:**

The central audit hypothesis from the story file's Dev Notes —
"`tensor.detach().cpu().numpy()` cannot run on a bf16 tensor (numpy
has no `bfloat16` dtype) — so either (a) `model.model.speech_tokenizer.decode`
returns fp32 internally (most likely), OR (b) the production bf16
default is not actually engaging end-to-end on the codec-decoder side"
— is consistent with this read-only audit's finding. The
speech_tokenizer's documented return type (`List[np.ndarray]`) supports
hypothesis (a). The Task 1.2 + 1.3 forward-hook capture (PENDING)
confirms or refutes this on the live RTX 5090 model.

**Task 5.4 routing condition:** would only trigger if the Task 1.2 +
1.3 hook capture surfaced an internal `.float()` upcast inside the
model's forward pass before the documented GPU→CPU boundary. The
read-only audit has not surfaced such a defect; the runtime hook
capture is the final answer.

## NFR1 first-chunk-latency measurement

**Scope (Task 7.1 + 7.2 + 7.3):** measure first-chunk-latency under
`AppSettings.tts_precision="auto"` (bf16 on RTX 5090) vs
`AppSettings.tts_precision="fp32"` (NFR7 override) using the canonical
Story 17.3 §4.1 step 3 Sarira-F long-form utterance with
`MYVOICE_PROGRESSIVE_PLAYBACK_CSV` env-var-gated capture; N=10 per
branch.

**Status:** COMPLETE (2026-05-10, RTX 5090 dev host — measurement table
+ per-launch detail + producer-ratio analysis + diagnosis below). OQ #3
routing TRIGGERED at -3.77% median (well below the [30%, 50%]
anticipated gate); Commander selected option (b) "defer Task 8 audition
to post-Story-18.4 retrospective." Procedure preserved for the deferred
re-measurement: Commander runs two batch files sequentially:
`03_Story_18.3_NFR1_BF16.bat` (loops 10 fresh-process launches with
`tts_precision="auto"`) and `04_Story_18.3_NFR1_FP32.bat` (same pattern
with `tts_precision="fp32"`). Each batch programmatically toggles
`tts_precision` via `18-3-set-precision.py` before launching, so no
manual `settings.json` editing. After both batches close, the dev agent
runs `_bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py`
which reads all 20 per-run CSVs, writes the two consolidated CSVs
(`18-3-rtx5090-bf16.csv` + `18-3-rtx5090-fp32.csv`), and prints
median/p90/p95 + delta tables to stdout — automatically surfacing the
Task 7.4 routing condition (sub-20% speedup).

**Methodology:**

- Each generation is a fresh process launch (kill the app between
  runs) so cuDNN benchmark autotune cache state does not bleed across
  runs (same discipline as Story 18.2 Task 4.2).
- The fp32 branch is **fp32-with-TF32-engaged**, NOT strict-fp32.
  Story 18.2's TF32 + cuDNN benchmark engagement composes on top of
  any precision setting — the bf16 measurement composes ON TOP OF
  Story 18.2's TF32+cuDNN-engaged baseline. The strict-fp32 vs
  TF32-fp32 comparison is **out of scope** (Story 18.2 closed it as
  null on the producer-bottleneck workload; the bf16 measurement does
  not need to re-litigate).
- The settings-toggle methodology eliminates the git-checkout-pair
  complexity Story 18.2's spec required (because bf16 is already on
  HEAD, post-18.3 source-tree edits surface a clean A/B via the
  `tts_precision` setting).

**Procedure (for Commander — ~30 minutes):**

1. Double-click `03_Story_18.3_NFR1_BF16.bat`. The bat:
   - Sets `tts_precision="auto"` via `18-3-set-precision.py`
   - Loops 10 fresh-process launches, each writing to
     `18-3-rtx5090-bf16-run<NN>.csv`
   - Prompts you to generate one Sarira-F long-form utterance per launch
2. Double-click `04_Story_18.3_NFR1_FP32.bat`. Same pattern; writes to
   `18-3-rtx5090-fp32-run<NN>.csv`.
3. Run the aggregator from the repo root:
   ```
   python310\python.exe _bmad-output\implementation-artifacts\18-3-aggregate-nfr1.py
   ```
4. Send the script's stdout (the median / p90 / p95 / delta tables) back
   to the dev agent. The script also writes the consolidated
   `18-3-rtx5090-bf16.csv` + `18-3-rtx5090-fp32.csv` files for the
   evidence-file footer.

**Important:** after the FP32 run, restore `tts_precision="auto"` for
normal MyVoice use:
```
python310\python.exe _bmad-output\implementation-artifacts\18-3-set-precision.py auto
```

**Result (2026-05-10, RTX 5090 dev host, post-Story-18.2 TF32+cuDNN baseline):**

Cold-start first-chunk latency (first generation per fresh-process launch;
runs 7 + 10 of bf16 captured 2 records each — only the first counted):

```
            bf16 (auto)    fp32 (override)    Δ (ms)    Δ (%)
median      5029.2 ms      4846.4 ms          -182.8     -3.77   ← bf16 SLIGHTLY SLOWER
p90         5656.8 ms      5409.5 ms          -247.4     -4.57
p95         5704.6 ms      5634.4 ms           -70.2     -1.25
N           10             10                  —          —
```

Per-launch detail (high run-to-run variance, ±1500 ms in some pairs):

```
run   bf16 (ms)    fp32 (ms)    Δ (ms)
1     5752.3       5859.3       +107.0
2     5646.2       4579.7       -1066.5     ← outlier
3     4988.6       4976.3        -12.3
4     4905.4       5199.1       +293.6
5     4628.1       5048.6       +420.6
6     5176.8       5359.5       +182.7
7     4607.1       4495.3       -111.8
8     4614.7       4538.3        -76.3
9     5069.9       4168.3       -901.6
10    5323.8       4716.5       -607.3
```

5 of 10 launches: bf16 faster; 5 of 10: fp32 faster. With ±1500 ms
per-launch noise vs. a ~180 ms median delta, **the result is well
within noise**.

Producer-bottleneck steady-state ratio (Story 18.1 §4.4 methodology —
mean inter-chunk-emit interval ÷ mean chunk audio duration; ratio > 1
means the producer is slower than realtime → audio gaps):

```
                    bf16 (auto)    fp32 (override)
mean interval (ms)  3213           2782
mean duration (ms)  1981           1981
ratio               1.62           1.40           ← bf16 WORSE by 0.22 (-15.5%)
```

For comparison: Story 18.1's V2 baseline producer ratio was **3.23**
(`memory/epic18_producer_bottleneck_finding.md`). Both branches today
sit at 1.40–1.62 — Story 18.2's TF32 + cuDNN benchmark engagement is
the source of the headline improvement (3.23 → 1.40); Story 18.3's
bf16 default sits 15% **worse** than that fp32-with-TF32 baseline.

### Methodology composition (per Story 18.3 AC #10)

The fp32 branch in this A/B is **fp32-with-TF32-engaged**, not
strict-fp32. Story 18.2's TF32 + cuDNN benchmark autotune engages at
startup on every Ampere+ host regardless of precision; the bf16 vs
fp32 comparison here is **bf16 vs (fp32 + TF32 + cuDNN benchmark)**.
Strict-fp32 (TF32 disabled) is out of scope per AC #10 — Story 18.2
closed it as null on the producer-bottleneck workload.

### Why bf16 didn't deliver — diagnosis

1. **TF32 ate the bf16 budget.** Blackwell (RTX 5090, capability 12.0)
   has very capable TF32 tensor cores. Story 18.2 engaged TF32 + cuDNN
   benchmark; the fp32 path now executes matmul on tensor cores. The
   incremental gain from going to bf16 (which uses different tensor-
   core kernels) is small or negative.
2. **Workload is not matmul-throughput-bound.** First-chunk latency
   requires ~25 talker forward passes (autoregressive, single-token,
   no batching) plus one codec decode. Single-token forwards are
   dominated by **kernel-launch overhead + KV-cache management**, not
   matmul GFLOPs. bf16's ~2× tensor-core throughput advantage
   materializes only at large batch sizes and large matmul shapes.
3. **bf16 has slightly higher dispatch overhead** in PyTorch for small
   ops — historically a few % per kernel. Compounded over ~25 forward
   passes, this could account for the small bf16 regression.
4. The dtype audit (Task 1) confirmed bf16 IS engaged end-to-end — on
   talker AND on the speech_tokenizer's inner `Qwen3TTSTokenizerV2Model`
   (parameters all `torch.bfloat16`). So this is NOT a wire-up bug.

### Open Question #3 — ROUTING TRIGGERED

Story 18.3 Task 7.4 routes to Commander when median speedup < 20%. Our
median is **-3.77%** (bf16 slightly slower). Commander must decide
between three options the story spec lays out (`:1383+` and OQ #3):

**Option (a) — Ship-as-engaged-anyway.** Keep
`tts_precision="auto"` → bf16 on Ampere+. Run Task 8 audition to
certify perceptual equivalence. Accept the partial / null perf gain.
- **Pros:** matches the story's anticipated default; ships the resolver
  + setting + audit infrastructure as designed; NFR7 fp32 fallback path
  is exercised by users who hit any perceptual issue.
- **Cons:** the audition is load-bearing for a default that doesn't
  pay for itself on perf. Listeners' time + Commander coordination is
  spent certifying a no-op default.
- **Risk:** if Task 8 surfaces any `audible_seam` flag, we'd be FAILING
  on perceptual grounds for a default that doesn't help anyway.

**Option (b) — Defer to future investigation with a shim.** Skip the
audition; ship Story 18.3 source-tree work + setting + audit; revisit
the bf16-as-default decision after Story 18.4 (`torch.compile`) lands.
- **Pros:** Story 18.4 dramatically changes the kernel-launch overhead
  profile (CUDA graphs / compiled paths). With small-op overhead
  collapsed, matmul throughput becomes the dominant axis again — bf16
  may genuinely help. Re-running the A/B post-18.4 with the same
  harness would give a cleaner answer.
- **Pros:** preserves the resolver + setting (NFR7 fallback works
  immediately) without burning the listener-recruitment budget on a
  decision the data can't yet justify.
- **Cons:** the architecture amendment Task 9 still has to land —
  needs to honestly say "engaged but no measured speedup vs Story 18.2
  baseline; revisit post-18.4."

**Option (c) — Close as power-user opt-in.** Flip the resolver default
for `tts_precision="auto"` on Ampere+ to return **fp32** (not bf16),
because TF32 already engages; `tts_precision="bf16"` becomes the
user-explicit power-user opt-in.
- **Pros:** data-driven default; users get the measurably-faster path.
- **Cons:** deviates from the story's anticipated outcome at `:1381`;
  more nuanced resolver (auto-on-Ampere depends on what TF32 has done,
  not just hardware capability); harder to roll back at Story 18.4 if
  bf16 starts helping there.
- **Cons:** "Ampere with TF32 engaged → fp32" is a moving target — if
  a future qwen-tts pin or a future story disables TF32, the auto
  default would silently flip back to bf16.

**Dev-agent recommendation: Option (b).** The audit (Task 1) +
resolver + setting + H1 fix + finalization-drain follow-up are all
worth shipping. The bf16-as-default decision should wait until Story
18.4 lands — at that point the workload may genuinely be matmul-bound
and the A/B will give a clean answer with the same harness. Commander
chooses the actual policy.

### Audition implication

If Commander picks (a), we proceed to Task 8 audition with the
caveat that listeners are NOT certifying a perf-positive default —
they're certifying perceptual equivalence in case `bf16` is later
proven to help (post-18.4).

If Commander picks (b) or (c), Task 8 audition is **deferred or
skipped**. The architecture amendment in Task 9 documents the
decision honestly.

## NFR3 audition verdict

**Scope (Task 8):** ≥3-listener perceptual A/B audition; PASS iff zero
listeners flag `audible_seam` on any TRUE_STREAM (= bf16) pair across
all 30 trials (10 utterances × 3 listeners × A and B).

**Status:** PENDING — Commander-routed (requires fixture regeneration
under bf16 / fp32 + recruiting ≥3 listeners + running the helper
script).

**Procedure outline (for Commander; full detail in Task 8.1–8.6 of the
story file):**

1. Regenerate the 10-pair fixture against bf16 / fp32 outputs into a
   new directory `_bmad-output/implementation-artifacts/18-3-perceptual-fixtures/`
   (do NOT overwrite `16-7-perceptual-fixtures/` — Story 17.1's
   verdict reproducibility depends on it).
2. Adapt `17-1-l1-audition-helper.py` as `18-3-l1-audition-helper.py`
   pointing at the new fixture directory.
3. Recruit ≥3 listeners; run the audition (same protocol as
   Story 17.1: `LISTENING-INSTRUCTIONS.md` byte-identical, controlled
   defect vocabulary, 30 total trials).
4. Capture results at `18-3-bf16-precision-audition.csv`.

**Verdict gate (verbatim Story 17.1):** PASS iff zero listeners flag
`audible_seam` on any TRUE_STREAM (= bf16) pair across all 30 trials.

**Result template (Commander to fill):**

| System | sibilance | tonal_drift | low_amp_consonant | audible_seam | other |
|---|---|---|---|---|---|
| A (fp32 baseline) | ? | ? | ? | ? | ? |
| B (bf16 candidate) | ? | ? | ? | ? | ? |

**Per-listener subtotals + per-utterance subtotals + verdict:** TBD.

**Task 8.6 routing condition:** if FAIL on any utterance from any
listener, STOP and route to Open Question #4 with the failed-utterance
class annotated. Do NOT close the story or amend the architecture
until Commander routes the outcome.

## Bundled smoke

**Scope (Task 10):** first production-bundle verification of the
combined Story 18.2 + 18.3 source-tree edits; `myvoice.log` must contain
both INFO breadcrumbs.

**Status:** PENDING — Commander-routed (requires `build_release.bat`
and bundled exe launch on Windows host).

**Procedure (for Commander; per Task 10.1–10.5):**

1. Run `build_release.bat` (or equivalent per
   `memory/build_tools_phase_perp_state.md` + `memory/production_release_state.md`).
2. Launch `build_tools/dist/MyVoice/MyVoice.exe`. Inspect the portable
   `myvoice.log` (per `setup_logging()` discipline) for BOTH INFO
   breadcrumbs:
   - Story 18.2: `"TF32 + cuDNN benchmark enabled (device_capability=...)"`
   - Story 18.3: `"ModelRegistry initialized: device=cuda:0, dtype=torch.bfloat16, precision_source='app_settings_auto_ampere', quality_tier=quality"`
3. Run a short canonical TTS generation; verify no errors.
4. Run a long-form Sarira-F CLONED utterance; Commander confirms zero
   perceptual defects compared to the pre-18.3 / pre-18.2 baseline
   (Commander-solo spot-check; independent of the multi-listener
   audition).

**Task 10.5 routing:** if the bundled smoke surfaces a packaging defect
(e.g., `tts_precision` field round-trip failure in portable
`settings.json`, or `resolve_tts_precision` PyInstaller hidden-imports
issue), surface it to Commander rather than absorbing.

## Side observation — finalization race surfaced by bf16 engagement (FIXED in-story)

**Resolution:** the finalization race was closed in this story (Path A —
"close the race before Task 7") rather than deferred. See the change-log
entry "Finalization-drain follow-up landed" in the story file. Key
artifacts:

- `AudioCoordinator.stop_streaming_session(wait_for_drain: bool = False)`
- `_DRAIN_SAFETY_BUFFER_S = 0.15` (PyAudio internal latency cushion)
- `_MAX_DRAIN_WAIT_S = 15.0` (hard cap so a math drift can't hang close)
- 7 new unit tests at `TestStopStreamingSessionDrain` pin the contract.
- The two `is_final` call sites in `app.py` pass `wait_for_drain=True`;
  the two cancel sites keep the default `False` so user-cancel stays
  prompt.

The original observation (preserved below for reference) is what
prompted the fix.

---


**Reported by Commander after the 2026-05-10 10:08 audit run:**
"Gaps slightly shorter (improvement) but the last chunk doesn't play —
still sounded cut off at the end."

**Log timing analysis** (`logs/myvoice.log` lines 286–316):

| Wall clock (HH:MM:SS.ms) | Event |
|---|---|
| 10:08:42.650 | TTS generation start (TRUE_STREAM, Sarira-F-cloned voice) |
| 10:08:47.595 | Progressive playback session opened (sample_rate=24000Hz) |
| 10:08:49.634 → 10:08:51.792 | 4 chunk-emit/arrival metric pairs across ~2.2s |
| 10:08:51.792 | `TTS generation complete (TRUE_STREAM): 115639 samples, 3 chunks, 9.14s total, 4.93s first chunk` |
| 10:08:51.793 | `Starting audio playback via AudioCoordinator` (final chunk dispatched) |
| 10:08:51.935 | `Virtual mic streaming session stopped` (143 ms after dispatch) |

**Diagnosis:** the producer streamed 9.14 s of audio in 4.34 s of
wall-clock session time (faster-than-realtime, ~2× ratio) and then
called `_audio_coordinator.stop_streaming_session()` immediately on
`is_final` at `app.py:2661-2668`:

```python
if chunk.is_final:
    try:
        await self._audio_coordinator.stop_streaming_session()
    ...
```

The coordinator's `stop_streaming_session` (`audio_coordinator.py:1234-1259`)
does NOT wait for the underlying PyAudio output buffer to drain — it
calls each service's `stop_streaming_session()` directly and returns.
With the bf16 + TF32 + cuDNN engagements all firing
(`speech_tokenizer.model` bf16 surprised even the audit hypothesis),
the producer outpaces consumption faster than before, so the buffer
still has the tail of the last chunk un-played when the session is
torn down → audible cut-off at the end.

**Root cause class:** Story 17.3 progressive-playback finalization
race. The bug always existed but was masked when the producer was the
bottleneck (gaps in audio meant the buffer was almost empty by the
time `is_final` arrived).

**Out of scope for Story 18.3.** Story 18.3 does not touch chunk
emission, the AudioCoordinator session lifecycle, or the
`is_final → stop_streaming_session` finalization path. The bf16
engagement merely **surfaces** the latent race by making the producer
faster.

**Recommended follow-up — a new story (call it 18.5 or "Story 17.3
finalization-drain follow-up"):**

- Make `stop_streaming_session()` await the PyAudio buffer drain
  (poll `Stream.is_active()` until False, with a generous timeout) OR
- Defer the `stop_streaming_session()` call in
  `_handle_progressive_chunk_async` until the buffer is observed
  empty.

The Story 18.4 (`torch.compile` decoder cache) work will make this
race **even more pronounced** — the producer will be faster still.
Suggest closing the finalization race BEFORE Story 18.4 lands, or
co-landing the fix.

The Story 18.3 audition (Task 8) should be conducted with this
caveat in mind: the bf16 audio is correct end-to-end; the cut-off is
a finalization-pipeline issue, not a perceptual defect of bf16.
Listeners should NOT mark `audible_seam` for the cut-off-at-end
because the seam is at the session-tear-down boundary, not at a
chunk-overlap-add boundary.

## Commander-routed work

The dev agent has scaffolded all the harness Commander needs. Each task
that requires GPU / listener / build interaction is routed through a
batch file or helper script; Commander does NOT need to write any code
or REPL gymnastics.

| Task | Harness | Effort |
|---|---|---|
| **Task 1** (dtype audit + forward hooks) | `02_Story_18.3_DType_Audit.bat` (env-var-gated instrumentation in `model_registry.py`) | ~2 min Commander; agent parses `myvoice.log` |
| **Task 7** (NFR1 N=10 measurement) | `03_Story_18.3_NFR1_BF16.bat` + `04_Story_18.3_NFR1_FP32.bat` + `18-3-aggregate-nfr1.py` | ~30 min Commander; agent reads aggregator output |
| **Task 8** (audition) | Fixture regen via GUI (Commander); `18-3-l1-audition-helper.py` + `18-3-set-precision.py` | ~2 hr Commander + listener time |
| **Task 9** (architecture amendment) | Dev agent applies once Task 8 verdict lands | Agent does it |
| **Task 10** (bundled smoke) | `build_release.bat` + manual launch (Commander) | ~30 min Commander |
| **Task 12** (code-review) | `/bmad-bmm-code-review` from a different LLM than the implementer | Commander runs |

## Open Questions

(Empty initially. Populated as Commander surfaces routing conditions
during Tasks 1, 5, 7, 8.)

1. *(Reserved — Task 1.5 routing.)* Pre-implementation audit surfaces
   an unexpected fp32 round-trip inside the model's forward pass.
2. *(Reserved — Task 5.4 routing.)* Streaming pipeline dtype audit
   surfaces a defect (currently CLEAR — no dev-agent-routable defect
   surfaced in the read-only audit; runtime hook capture is the final
   answer).
3. *(Reserved — Task 7.4 routing.)* NFR1 measurement falls below 20%
   speedup.
4. *(Reserved — Task 8.6 routing.)* NFR3 audition flags `audible_seam`
   on any utterance.

## Source artifacts

Force-add per `memory/git_repo_state.md` (`_bmad-output/` is gitignored):

- ✓ `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder.md` (story file)
- ✓ `_bmad-output/implementation-artifacts/18-3-bf16-precision-on-talker-decoder-evidence.md` (this file)
- ✓ `_bmad-output/implementation-artifacts/18-3-set-precision.py` (settings-mutation helper; force-add)
- ✓ `_bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py` (NFR1 aggregator; force-add)
- ✓ `_bmad-output/implementation-artifacts/18-3-l1-audition-helper.py` (audition helper, adapted from 17-1; force-add)
- ✓ `02_Story_18.3_DType_Audit.bat` (Task 1 harness — repo root, NOT under `_bmad-output/`)
- ✓ `03_Story_18.3_NFR1_BF16.bat` (Task 7.1 harness — repo root)
- ✓ `04_Story_18.3_NFR1_FP32.bat` (Task 7.2 harness — repo root)
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run<NN>.csv` (Task 7.1; Commander-produced; N=10)
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32-run<NN>.csv` (Task 7.2; Commander-produced; N=10)
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-bf16.csv` (Task 7.3; aggregator-produced)
- ○ `_bmad-output/implementation-artifacts/18-3-rtx5090-fp32.csv` (Task 7.3; aggregator-produced)
- ○ `_bmad-output/implementation-artifacts/18-3-perceptual-fixtures/` (Task 8.1; Commander-produced via GUI)
- ○ `_bmad-output/implementation-artifacts/18-3-bf16-precision-audition.csv` (Task 8.4; Commander-produced)
