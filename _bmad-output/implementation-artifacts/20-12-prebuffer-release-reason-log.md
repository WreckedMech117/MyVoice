# Story 20.12: Log the Pre-Buffer Release Regime

Status: done - 2026-09-14 — one INFO line per session (`Streaming buffer: pre-buffer RELEASED — reason=… held=… waited=… P=… cushion=… mode=…`) from `StreamingChunkBuffer`; 11 regime tests; full suite 3003 passed, 1 xfailed. Evidence: `20-12-prebuffer-release-reason-log-evidence.md`.

<!-- Source: Story 20.10 evidence §4, observability gap raised from the RTX 3060 log. -->
<!-- Risk: LOW. One INFO line per streaming session; no behaviour change. -->

## Story

As **the maintainer reading a user's log from a sub-16 GiB card**,
I want **one line per streaming session saying which release regime opened playback and how much audio was held**,
so that **Epic 20 F6 questions are answered from the log instead of by inferring from dispatched byte counts**.

## Context

`StreamingChunkBuffer` (`streaming_chunk_buffer.py`) decides when to release
the first audio: `is_final`, `max_hold_chunks`, `max_pre_delay`,
`producer_keeps_up`, `gapless_feasible`, or `gapless_unreachable` (static
watermark). It records the decision in `last_release_reason`, which is exposed
for tests only. The 3060 log shows the adaptive path engaged
(`detected_vram=12.0 GiB`, `cushion_budget=2.0s`) and the first dispatch
carrying two chunks, but the regime had to be inferred.

## Acceptance Criteria

### AC #1 — One INFO line at release
**Then** when the watermark queue is first flushed, the buffer logs one INFO
line naming: the release reason, the seconds of audio held, the wall-clock
seconds since the first chunk arrived, the worst observed producer rate `P`
(or `n/a`), the cushion the policy required, and the mode (static / adaptive)
**And** the line is emitted once per session — a session that releases on the
static watermark logs it too, so the ≥16 GiB path is not silent.

### AC #2 — Test-pinned
**Then** a unit test per regime asserts the line is emitted exactly once with
the right reason token, using the existing `StreamingChunkBuffer` test
fixtures and an injected clock.

### AC #3 — No behaviour change
**Then** every existing `StreamingChunkBuffer` and `audio_coordinator` test
passes unchanged; the log line is the only diff on the release path.

## Dev Notes

Keep the change inside `streaming_chunk_buffer.py`; `audio_coordinator.py`
already logs the enable decision and does not need to change. Run pytest with
`I:\MyVoiceV2\python310\python.exe -m pytest` (the worktree has no interpreter
of its own; never modify or copy `python310`).
