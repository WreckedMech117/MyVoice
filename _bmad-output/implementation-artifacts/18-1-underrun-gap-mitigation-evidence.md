# Story 18.1 Evidence — Underrun-Gap Mitigation (Phase ⊥-Polish-2)

This file is the empirical record for Story 18.1's Tasks 1, 2/3, 4, and 6.
Force-added per the Story 16.9 / 17.1 / 17.2 / 17.3 evidence-file precedent
(`_bmad-output/` is gitignored per `memory/git_repo_state.md`).

## §1. Instrumentation overhead (Task 1.0)

**Goal.** Confirm `metrics.record` overhead is ≤ 100 µs/call before adding
inline metric calls to the timing-sensitive producer (`_wrapped_post`
inside `_generate_true_stream`) and consumer (`_handle_progressive_chunk_async`)
paths. Instrumentation that itself perturbs the ratio Task 1.4 measures
would produce self-invalidating data.

**Method.** Tight loop of N=1000 and N=5000 `metrics.record(...)` calls
on the RTX 5090 dev host through the project's bundled `python310/python.exe`.
Tags varied to reflect realistic per-chunk emission (session_id +
chunk_index for the basic shape; +hardware/model_type for the extended
shape).

**Result (run 2026-05-09 on RTX 5090 / Win11 / Python 3.10.11):**

| Run                              |       N | Mean per call | Verdict |
| -------------------------------- | ------: | ------------: | :-----: |
| Basic tags (session_id, chunk_index) |   1000 |    2.40 µs |  PASS   |
| Extended tags (4 kwargs)         |   5000 |    1.35 µs |  PASS   |

Both well under the 100 µs threshold (~50× headroom). **Decision: inline
`metrics.record` calls are safe in the producer-side `_wrapped_post`
chunk-emit branch (qwen_tts_service.py) AND the consumer-side
`_handle_progressive_chunk_async` post-`play_audio_chunk` block (app.py).**
No per-session list buffering / CSV-flush-at-close fallback path was
needed.

The metric module's own thread-safe + listener-snapshot logic
(`metrics.record` at `src/myvoice/observability/metrics.py:77`) is the
established hot-path-safe contract — Story 11.3's design held up.

## §2. Per-chunk metric emission points (Tasks 1.1, 1.2, 1.3)

Three new metrics were added under the existing `metrics.record(...)`
single-chokepoint helper:

### 2.1 `progressive_chunk_emit_ms` (Task 1.1)

- **Site**: `src/myvoice/services/qwen_tts_service.py` —
  `_wrapped_post` `append_chunk` branch inside `_generate_true_stream`.
  Fires per chunk emitted by the TRUE_STREAM producer, AFTER the
  `chunk_count_box` increment and BEFORE the
  `_audio_chunk_ready_callback` invocation.
- **Value**: `time.time() * 1000.0` — absolute wall-clock ms. Wall-clock
  (vs elapsed-since-start) so the CSV can join by `chunk_index` against
  the consumer-side `progressive_chunk_playback_arrival_ms` without a
  clock-base reconciliation step.
- **Tags**: `session_id` (registry-issued, str) + `chunk_index` (0-based int).
  Mirrors first_chunk_latency_ms's structural precedent
  (`qwen_tts_service.py:3122`).
- **Scope**: TRUE_STREAM only — SENTENCE_STREAM is not in this story's AC #1
  scope (the underrun gap is a TRUE_STREAM-on-Blackwell observation per
  Story 17.3 evidence §4.4). The synthetic terminal `finalize` post does
  NOT emit this metric (only the data-bearing `append_chunk` posts do).

### 2.2 `progressive_chunk_playback_arrival_ms` (Task 1.2)

- **Site**: `src/myvoice/app.py` — inside `_handle_progressive_chunk_async`,
  AFTER `await self._audio_coordinator.play_audio_chunk(...)` returns and
  inside the `chunk.audio_data.size > 0` branch.
- **Value**: `time.time() * 1000.0` — absolute wall-clock ms (joinable
  with `progressive_chunk_emit_ms` by `chunk_index`).
- **Tags**: `chunk_index` + `is_final` (bool) + `audio_data_size`
  (per-chunk frame count, int) + `session_id` (registry-issued, str
  — added by code-review pass M1 for cross-CSV joinability when
  multiple generations are captured in one run).
- **`is_final` semantics by streaming mode** (worth flagging for the
  Task 1.4 CSV analyst): on TRUE_STREAM, every data chunk has
  `is_final=False` because the producer emits a SEPARATE synthetic
  terminal `AudioChunk(is_final=True, audio_data.size==0)` that is
  filtered by the `size > 0` gate — so a TRUE_STREAM CSV will have
  `is_final=False` on every consumer-side row. SENTENCE_STREAM, by
  contrast, sets `is_final=True` on its LAST data chunk (real audio +
  terminal flag fused) and that row WILL appear in the CSV. Use
  `chunk_index = max(chunk_index)` to find the last data row of a
  session in TRUE_STREAM CSVs.

### 2.3 `progressive_chunk_audio_duration_ms` (Task 1.3)

- **Site**: same block as 2.2 (immediately after the arrival metric).
- **Value**: `(audio_data.size / sample_rate) * 1000.0` — the canonical
  "drain time" for the consumer side.
- **Tags**: `chunk_index`.
- **Guard**: `sample_rate > 0` (defensive — production AudioChunks always
  carry a non-zero sample_rate, but the guard prevents a ZeroDivisionError
  if a pathological synthetic chunk slips through).

### 2.4 Co-location semantics (zero-size terminal handling)

Both consumer-side metrics live INSIDE the `if chunk.audio_data.size > 0:`
branch of `_handle_progressive_chunk_async`. TRUE_STREAM's synthetic
terminal `AudioChunk(is_final=True, audio_data.size==0)` therefore emits
NEITHER consumer-side metric — it skips the play_audio_chunk call (the
existing Story 17.3 contract) and the metric emission alongside it.
Tested by `test_no_metrics_emitted_for_zero_size_terminal_chunk` in
`tests/unit/test_app_progressive_playback_instrumentation.py`.

This co-location matters for Task 1.4's CSV analysis: a zero-size row
with `audio_duration_ms == 0.0` would skew the median/p95 statistics
on the duration metric. Keeping both metrics inside the data-bearing
branch keeps the CSV clean.

## §3. Tests added (Task 1)

| File                                                                    | Tests | Status |
| ----------------------------------------------------------------------- | ----: | :----: |
| `tests/unit/test_app_progressive_playback_instrumentation.py`           |     4 |  PASS  |
| `tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py` |     3 |  PASS  |
| `tests/unit/observability/test_progressive_playback_csv_capture.py`     |    13 |  PASS  |
| **Total Story 18.1 new tests**                                          |    20 |  PASS  |

Regression sweep on Story 17.3's 32 progressive-playback tests
(`test_qwen_tts_service_true_stream_callback.py` ×3,
`test_app_progressive_playback.py` ×22, `test_app_progressive_playback_cancel.py` ×3,
`test_progressive_playback_dispatch_skip.py` ×3, etc.) all pass
unchanged. Metrics observability tests (`tests/unit/observability/`)
pass 45/45.

## §4. Mitigation choice (Tasks 1.4 + 1.5)

### 4.0 Verdict (2026-05-09)

**Producer-side bottleneck. Options 1 and 2 are mathematically ruled
out as full mitigations. The fix class is Option 3 (decode-rate
investigation), and that work belongs in Stories 18.3 (bf16 precision)
and 18.4 (`torch.compile` + persistent cache) per the Epic 18 plan.
Story 18.1 closes shipping the instrumentation + CSV-capture
infrastructure; no mitigation source-tree edits beyond Task 1 are
shipped.**

### 4.1 Procedure for the Task 1.4 measurement run

This is a hand-off to Commander to execute on RTX 5090 hardware. The
CSV writes itself — no log-grepping required. Steps 4–5 below assume
the env-var-gated capture wired into ``MyVoiceApp.__init__``
(``src/myvoice/observability/progressive_playback_csv_capture.py``).

**Step 1 — Engage CSV capture.** Double-click
``01_Run_MyVoice_With_CSV_Capture.bat`` from the repo root. The .bat
mirrors ``01_Run_MyVoice.bat`` (preserving the
torch-before-PyQt6 DLL ordering invariant from
``memory/torch_pyqt6_dll_ordering.md`` — ``main.py`` runs directly,
NOT as ``python -m myvoice``) and additionally:

  * Sets ``MYVOICE_PROGRESSIVE_PLAYBACK_CSV`` to the spec'd path
    (``_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv``).
  * Pre-creates the ``_bmad-output/implementation-artifacts/`` folder
    so the listener's ``open()`` call succeeds.
  * On exit, prints the captured row count so you can verify the
    file was populated before reporting back.

Manual PowerShell equivalent (two separate commands, NOT one line):

```powershell
$env:MYVOICE_PROGRESSIVE_PLAYBACK_CSV = "I:\MyVoiceV2\_bmad-output\implementation-artifacts\18-1-instrumentation-rtx5090-longform.csv"
.\python310\python.exe .\src\myvoice\main.py
```

The CSV header (``metric_name, value, session_id, chunk_index,
is_final, audio_data_size``) writes immediately on launch — verify
the file exists and is non-empty BEFORE generating to confirm the
env-var took effect.

**Step 2 — Confirm Sarira-F warm cache.** Story 17.2 should have populated
``voice_files/Sarira-F.quality.pt``. If absent, Story 17.2's lazy-
precompute path will populate it on first generation; allow that
pre-roll to complete (and discard its CSV rows) before the canonical run.

**Step 3 — Generate the canonical long-form utterance.** Use the verbatim
≥250-character paragraph from Story 17.3 §4.1 step 3 of
``_bmad-output/implementation-artifacts/17-3-progressive-audio-playback-during-true-stream-evidence.md``.
Speaker: Sarira-F. The TRUE_STREAM dispatch path is the active path
on RTX 5090 (D-9 / NFR12 hardware-aware default). Audition the playback
in real-time and confirm the ~1-s gaps are present in this run (so the
data is anchored to the same defect class Commander observed in
Story 17.3 §4.4).

**Step 4 — Close the app cleanly.** ``MyVoiceApp._on_about_to_quit``
flushes + closes the CSV before service cleanup. Verify the file size
is non-trivial (≥1 KB for a long-form utterance — typically ~3 rows
per chunk at ~10–30 chunks).

**Step 5 — Compute the ratio** (per AC #3 / Task 1.4):
   - **Inter-chunk-emit interval**: ``progressive_chunk_emit_ms[n] - progressive_chunk_emit_ms[n-1]`` for n=1..N-1.
     Report median + p95.
   - **Chunk audio duration**: ``progressive_chunk_audio_duration_ms`` directly.
     Report median + p95.
   - **Ratio**: ``inter-emit-interval / audio-duration``.
     - **Ratio < 1** (emit-interval < drain-time) ⇒ consumer starves
       (Option 1 OR Option 2 is the correct fix class).
     - **Ratio > 1** (emit-interval > drain-time) ⇒ consumer is sufficient;
       the underrun has a different cause (re-examine Option 3 path).

**Step 6 — Report verdict.** Populate this section (§4) with the
verdict — which Option was chosen, what the data showed, why the
cheapest viable mitigation was sufficient OR why a more expensive
option was required. Per AC #3, do not start with Option 3 unless
Options 1 + 2 have been empirically ruled out (a 3-chunk pre-buffer
leaves a measurable inter-chunk gap > 250 ms AND ``frames_per_buffer = 4096``
ALSO leaves a measurable gap by the same criterion, with the raw data
captured here).

### 4.2 Captured measurement (canonical Sarira-F long-form, 2026-05-09)

**Run details.** Commander executed the canonical Story 17.3 §4.1 step 3
paragraph (354 chars, ~22 s of speech) via
``01_Run_MyVoice_With_CSV_Capture.bat`` on the RTX 5090 dev host with
the Sarira-F warm-cache. Generation completed naturally; the app was
closed cleanly. CSV at
``_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv``.

**Capture summary:** 11 producer-side ``progressive_chunk_emit_ms``
records (chunks 0–10), 10 consumer-side records (chunks 0–9). Chunk 10
is the talker's end-of-stream flush emitting ~82 ms after chunk 9 — the
talker calls ``streamer.end()`` and the worker decodes any remaining
buffered tokens immediately (this is structurally distinct from the
steady-state cadence and is excluded from the steady-state stats below).

**Per-chunk timeline:**

| chunk | emit_rel (s) | audio_dur (ms) | emit_interval (ms) | ratio | silent_gap (ms) |
| ----: | -----------: | -------------: | -----------------: | ----: | --------------: |
| 0     | 0.00         | 1980.7         | —                  | —     | —               |
| 1     | 5.34         | 1980.7         | 5,338              | 2.70  | 3,358           |
| 2     | 10.62        | 1980.7         | 5,280              | 2.67  | 3,300           |
| 3     | 16.71        | 1980.7         | 6,093              | 3.08  | 4,112           |
| 4     | 23.59        | 1980.7         | 6,877              | 3.47  | 4,896           |
| 5     | 30.00        | 1980.7         | 6,415              | 3.24  | 4,434           |
| 6     | 36.37        | 1980.7         | 6,362              | 3.21  | 4,382           |
| 7     | 42.61        | 1980.7         | 6,244              | 3.15  | 4,263           |
| 8     | 49.08        | 1980.7         | 6,473              | 3.27  | 4,493           |
| 9     | 55.76        | 1980.7         | 6,678              | 3.37  | 4,697           |
| 10    | 55.84        | (n/a — EOS flush) | 83             | 0.04  | (n/a)           |

**Steady-state aggregates** (chunks 1→2 through 8→9, dropping the
chunk-0→1 warm-up and the chunk-9→10 EOS-flush):

| Stat                          | Value  |
| ----------------------------- | -----: |
| Inter-chunk-emit interval — median | 6,389 ms |
| Inter-chunk-emit interval — p95    | 6,678 ms |
| Chunk audio duration — median      | 1,981 ms (constant — fixed batch size 47,537 samples / 24,000 Hz) |
| **Ratio (emit / duration) — median** | **3.23×** |
| **Ratio — p95**                       | **3.37×** |
| Silent gap — median                | 4,408 ms |
| Silent gap — p95                   | 4,697 ms |
| Steady-state gaps > 250 ms         | **8 / 8 (100%)** |
| Steady-state gaps > 1,000 ms       | 8 / 8 (100%) |

**Producer:consumer real-time ratio:** the producer is generating audio
at **31% of real-time** (1 / 3.23 = 0.31). For every 1 second of audio
the consumer can play, the producer takes 3.23 s of wall-clock to
produce. Real-time deficit = 4.4 s / chunk in steady state.

### 4.3 Mitigation choice — Option 3 (decode-rate investigation)

**Option 1 (pre-buffer N=3 chunks at consumer) — mathematically ruled out.**
A 3-chunk pre-buffer holds 5.94 s of audio (3 × 1.981 s). At a
steady-state real-time deficit of 4.4 s/chunk, the buffer is exhausted
after ~1.4 chunks of post-flush playback. For this 11-chunk utterance:
chunks 0–2 would play seamlessly through ~t=16.6 s, chunk 3's small
~100 ms gap could land under 250 ms (best case), and chunks 4 through 9
would each have ~4.5 s silent gaps before them — **6 of the 8
post-pre-buffer inter-chunk gaps still > 250 ms.** AC #3's "Option 1
ruled out only if a 3-chunk pre-buffer leaves a measurable inter-chunk
gap exceeding 250 ms" is satisfied with overwhelming margin.

To eliminate gaps for an 11-chunk utterance via pre-buffer alone, N
would need to be ≥ 24 chunks (≥ 48 s of pre-buffer = ~127 s wait at
6.4 s/chunk emit). That defeats progressive playback's whole point and
violates AC #2's "first-chunk-to-audible within ~50–100 ms PyAudio
buffer-fill bound" by ~1,000×.

**Option 2 (frames_per_buffer increase) — structurally wrong fix class.**
PyAudio's ``frames_per_buffer`` controls per-stream output-buffer fill
pacing. It does not change inter-chunk arrival rate at the consumer —
chunks still emit from the producer at 6.4 s intervals regardless of
how PyAudio buffers within a single chunk. AC #3's structural rule
("HIGH/MEDIUM-fix regression tests must mirror the exact bug class")
applies in reverse here: a buffer-size knob does not address a
producer-throughput defect.

**Option 3 (decode-rate investigation) — the data points here.** The
talker model (Qwen3-TTS) on RTX 5090 is producing tokens at ~31% of
real-time consumption. This is exactly what Stories 18.3 (bf16
precision on the talker decoder, per
``epics-optimization-pass.md`` lines 1322+) and 18.4 (``torch.compile``
+ persistent compiled-decoder cache) target. The Epic 18 plan
(``epics-optimization-pass.md`` lines 228–250) sequences 18.3 + 18.4
as the producer-side throughput uplifts; this story's data confirms
that sequencing is the correct one.

### 4.4 AC #3 amendment — empirical ruling-out via captured + math

AC #3's strict reading wants Option 1 to be implemented and re-measured
to confirm gaps > 250 ms persist. The captured data plus deterministic
math eliminate this round-trip:

  * The captured 6,389 ms median emit-interval is fixed by the producer.
    It is not a property of the consumer-side code path.
  * No consumer-side change (pre-buffer size, frames_per_buffer,
    handler reordering) changes the producer-side emit-interval.
  * A 3-chunk pre-buffer is a 5.94 s audio cushion. The deficit per
    chunk is 4.4 s. After 5.94 / 4.4 ≈ 1.35 chunks of playback past
    pre-buffer drain, gaps return to 4.4 s — the SAME gap class the
    raw producer data already exhibits.

The story spec also says: *"The data was inconclusive" is not a
sufficient ruling-out — the dev agent surfaces ambiguity to Commander
rather than escalating to option 3 unilaterally.* The data is **not
inconclusive**. The 3.23× steady-state ratio with 100% of gaps >250 ms
across 8 steady-state samples is overwhelming. The escalation to
Option 3 is empirical (data-driven) and procedural (Commander
2026-05-09 sign-off via the dev-story workflow's mitigation-gate
question).

**Amendment note:** AC #3's "Option 1 ruled out only if a 3-chunk
pre-buffer leaves a measurable inter-chunk gap exceeding 250 ms" is
read as satisfied by the captured raw producer-side cadence + the
deterministic math above, without ceremonial Option 1 implementation.
This amendment is procedurally authorized by Commander's 2026-05-09
choice during the dev-story mitigation-gate question.

### 4.5 What this story ships (revised scope)

**Source-tree edits — instrumentation only.**

  * ``src/myvoice/services/qwen_tts_service.py`` — ``progressive_chunk_emit_ms``
    metric in TRUE_STREAM ``_wrapped_post`` ``append_chunk`` branch.
  * ``src/myvoice/app.py`` — ``progressive_chunk_playback_arrival_ms`` +
    ``progressive_chunk_audio_duration_ms`` metrics in
    ``_handle_progressive_chunk_async`` (``size > 0`` branch);
    env-var-gated CSV capture wiring in ``__init__`` and clean-stop in
    ``_on_about_to_quit``.
  * ``src/myvoice/observability/progressive_playback_csv_capture.py`` — new
    module: env-var-gated CSV listener with idempotent stop-callable
    and per-record flush.

**Convenience launcher.**

  * ``01_Run_MyVoice_With_CSV_Capture.bat`` — engages CSV capture +
    runs ``main.py`` directly per the established DLL-ordering invariant.

**Tests.**

  * ``tests/unit/test_app_progressive_playback_instrumentation.py`` (4 tests).
  * ``tests/unit/services/test_qwen_tts_service_true_stream_instrumentation.py`` (3 tests).
  * ``tests/unit/observability/test_progressive_playback_csv_capture.py`` (13 tests).

**Evidence.**

  * This file (``18-1-underrun-gap-mitigation-evidence.md``).
  * ``18-1-instrumentation-rtx5090-longform.csv`` (Task 1.4 raw data).

**What this story does NOT ship.**

  * No consumer-side pre-buffer state machine (Option 1) — empirically + mathematically ruled out.
  * No ``streaming_chunk_size`` config knob (Option 2) — wrong fix class.
  * No bundled-mode smoke audition (Task 4) — no mitigation source-tree edits to validate via bundled smoke; Story 17.3's existing bundled-smoke gate covers the progressive-playback dispatch surface, which 18.1's instrumentation does not change.
  * No NFR1 spot-check (Task 6) — the instrumentation does not change first-chunk-latency; spot-check is N/A for this scope.

### 4.6 Hand-off to Stories 18.3 + 18.4

The CSV-capture infrastructure built here stays useful for measuring
18.3 + 18.4 improvements. Engaging
``MYVOICE_PROGRESSIVE_PLAYBACK_CSV=1`` on a post-18.3 build and re-
running the canonical paragraph will produce a comparable CSV; the
ratio (now expected ~1.5× → ~1.0× → < 1.0×) is the canonical empirical
gate for whether 18.3/18.4 closed the producer bottleneck. Stories
18.3 + 18.4's evidence files SHOULD reference this CSV-capture path as
their measurement gate so the empirical comparison is anchored to the
same instrumentation surface.

## §5. Mitigation implementation (Tasks 2 / 3) — NOT APPLICABLE

Per §4.0 verdict: Options 1 + 2 are empirically + mathematically ruled
out (§4.3 + §4.4); the fix class is Option 3 (decode-rate
investigation), which belongs in Stories 18.3 + 18.4. This story
ships no consumer-side mitigation source-tree edits beyond Task 1's
instrumentation.

## §6. Bundled audition smoke (Task 4) — NOT APPLICABLE

Per §4.5 revised scope: there is no mitigation source-tree edit for
this story to validate via a bundled smoke. Story 17.3's existing
bundled-smoke gate covers the progressive-playback dispatch surface;
Story 18.1's Task 1 instrumentation lives on the same surface but does
not alter dispatch behavior — so the bundled artifact's user-visible
behavior is unchanged from Story 17.3's already-validated build.

The Task 5 regression sweep (§3) covers the dispatch-path-equivalence
guarantee (Story 17.3's 32 progressive-playback tests pass unchanged).

## §7. NFR1 spot-check (Task 6) — NOT APPLICABLE

Per §4.5 revised scope: the instrumentation does not touch the
``first_chunk_latency_ms`` emission path or the dispatch state
machine. Sarira-F warm-cache baseline from Story 17.2 evidence §4.3.2
remains **3.93–4.94 s p95** on the GPU short class — Story 18.1
preserves this contract by construction (zero edits to the latency-
measurement path). Stories 18.3 + 18.4 will run the NFR1 spot-check
when their producer-side throughput uplifts land.
