# Story 20.12 — Evidence: the pre-buffer release line

Date: 2026-09-14. Branch `story-20-12-release-reason-log`.
Change confined to `src/myvoice/services/streaming_chunk_buffer.py` and
`tests/unit/services/test_streaming_chunk_buffer.py`; `audio_coordinator.py`
untouched (it already logs the enable decision).

## 1. The line

Logger `myvoice.services.streaming_chunk_buffer`, level INFO, emitted once
per session at the moment the watermark queue is first flushed — from
`push` (the normal release) or `flush_remaining` (the coordinator's
stop/abort drain). Both set `_watermark_filled`, so neither can fire twice;
`reset()` opens a new session and logs again.

```
Streaming buffer: pre-buffer RELEASED — reason=<token> held=<s>s waited=<s>s P=<rate|n/a> cushion=<s>s|n/a mode=adaptive|static
```

| field | meaning |
|---|---|
| `reason` | adaptive: `last_release_reason` (`producer_keeps_up`, `gapless_feasible`, `gapless_unreachable`, `is_final`, `max_hold_chunks`, `max_pre_delay`); static: `static_watermark` or `is_final`; either mode: `flush_remaining` when the teardown drain was the first release |
| `held` | seconds of audio in the flushed payload (before crossfade; same length) |
| `waited` | wall-clock seconds from the first non-empty push to the flush (`0.00` if nothing was ever pushed) |
| `P` | worst observed producer rate — the value the cushion was sized against. `n/a` on the static path (never measured) and before chunk 2 exists |
| `cushion` | what the policy required at release: the static watermark on the static path and in the unreachable regime, τ_gapless (clamped by `max_pre_delay`) in the feasible regime, `0.00s` when the producer keeps up, `n/a` when P was never measured. On a guardrail escape it is still the policy's number, so `max_pre_delay … cushion=3.00s` reads as "wanted 3 s, never got it" |
| `mode` | `adaptive` (< 16 GiB, coordinator enabled it) / `static` (≥ 16 GiB default) |

The static path logs **without** setting `last_release_reason` — that stays
`None`, as `TestStaticWatermarkPathUntouched` pins. `_cushion_decision` is a
pure function of P and the worst-rate tracker is read, not re-observed, so the
line cannot move a release.

## 2. One example per regime

Produced by driving the real class with an injected clock (scratch script;
values are exact for the fixture geometry shown).

| regime | fixture | line |
|---|---|---|
| producer keeps up | 1 s chunks, chunk 2 at +0.5 s | `… reason=producer_keeps_up held=2.00s waited=0.50s P=2.00 cushion=0.00s mode=adaptive` |
| gapless feasible | the 3060 case from 20.10 §1: cs10 0.8 s chunks, P = 0.9, T_a = 9 s → τ = 1.0 s ≤ 2.0 s budget | `… reason=gapless_feasible held=1.60s waited=0.89s P=0.90 cushion=1.00s mode=adaptive` |
| gapless unreachable | P = 0.5 on the 349-char long fixture (T_a 19.695 s) → τ = 19.7 s ≫ budget, falls back to the 500 ms watermark | `… reason=gapless_unreachable held=1.60s waited=1.60s P=0.50 cushion=0.50s mode=adaptive` |
| is_final (adaptive) | stream ends on chunk 1 | `… reason=is_final held=1.00s waited=0.00s P=n/a cushion=n/a mode=adaptive` |
| max_hold_chunks | 3 × 100 ms with no clock movement, `max_hold_chunks=3` | `… reason=max_hold_chunks held=0.30s waited=0.00s P=n/a cushion=n/a mode=adaptive` |
| max_pre_delay | P = 0.5 measured on chunk 2, cap 3 s fires on chunk 3 | `… reason=max_pre_delay held=2.10s waited=3.00s P=0.50 cushion=3.00s mode=adaptive` |
| static watermark | ≥ 16 GiB default, 5 × 100 ms at 100 ms cadence | `… reason=static_watermark held=0.50s waited=0.40s P=n/a cushion=0.50s mode=static` |
| is_final (static) | one 100 ms chunk, final | `… reason=is_final held=0.10s waited=0.00s P=n/a cushion=0.50s mode=static` |
| flush_remaining | session torn down 0.3 s after chunk 1, before any release | `… reason=flush_remaining held=0.10s waited=0.30s P=n/a cushion=n/a mode=adaptive` |

Reading the 20.10 §1 log against this: the 3060 first dispatch (`held` ≈
1.58 s, two cs10 chunks, P ≈ 0.85–0.95 on a 155-char T_a ≈ 9 s) would now print
`reason=gapless_feasible … cushion≈0.5–1.6s mode=adaptive`, answering the F6
question that §4 raised.

## 3. Tests (AC #2) — `tests/unit/services/test_streaming_chunk_buffer.py::TestReleaseReasonLog`

All use the existing `_FakeClock` / `_adaptive_buf` / `_drive` fixtures and
`caplog` scoped to the buffer's logger; each asserts exactly one line and
checks the parsed `key=value` fields.

1. `test_producer_keeps_up_logs_zero_cushion`
2. `test_gapless_feasible_logs_tau_as_the_cushion`
3. `test_gapless_unreachable_logs_the_watermark_as_the_cushion`
4. `test_is_final_before_a_rate_exists_logs_na`
5. `test_max_hold_chunks_guardrail_is_named`
6. `test_max_pre_delay_guardrail_still_reports_the_policy_cushion`
7. `test_static_watermark_release_is_not_silent` (also re-pins `last_release_reason is None`)
8. `test_static_is_final_below_the_watermark_is_named`
9. `test_flush_remaining_before_release_is_the_first_release`
10. `test_line_is_emitted_exactly_once_per_session` (pass-through, final push and drain stay silent; `reset()` logs again once)
11. `test_flush_remaining_on_an_empty_buffer_stays_silent`

## 4. Results (AC #3)

- `tests/unit/services/test_streaming_chunk_buffer.py` + `test_consumer_crossfade_scoping.py`: **71 passed** (60 pre-existing, unchanged, + 11 new).
- Full suite, `I:\MyVoiceV2\python310\python.exe -m pytest tests -q -p no:cacheprovider`:

  ```
  3003 passed, 1 xfailed in 120.44s (0:02:00)
  ```

  Baseline on `main` was 2,992 passed, 1 xfailed; the delta is exactly the
  11 new rows. No existing test was modified.
