"""Story 18.3 Task 7.3 — aggregate N=10 CSV captures and compute the
bf16-vs-fp32 first_chunk_latency_ms delta.

Reads:
  - _bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run01.csv ... run10.csv
  - _bmad-output/implementation-artifacts/18-3-rtx5090-fp32-run01.csv ... run10.csv

Each input CSV has the 6-column schema from progressive_playback_csv_capture.py:
    metric_name,value,session_id,chunk_index,is_final,audio_data_size

The first_chunk_latency_ms metric is recorded once per generation. For each
run, we filter rows where metric_name == "first_chunk_latency_ms" and
collect the value(s); typically 1 per CSV.

Outputs:
  - 18-3-rtx5090-bf16.csv  (consolidated bf16 first-chunk-latency table)
  - 18-3-rtx5090-fp32.csv  (consolidated fp32 first-chunk-latency table)
  - Stdout: median + p90 + p95 per branch + absolute and percent deltas

Usage (from repo root):
    python310/python.exe _bmad-output/implementation-artifacts/18-3-aggregate-nfr1.py

Working file — gitignored under ``_bmad-output/``; not committed.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import List


ARTIFACTS = Path(__file__).resolve().parent
BF16_RUNS = sorted(ARTIFACTS.glob("18-3-rtx5090-bf16-run*.csv"))
FP32_RUNS = sorted(ARTIFACTS.glob("18-3-rtx5090-fp32-run*.csv"))
BF16_CONSOLIDATED = ARTIFACTS / "18-3-rtx5090-bf16.csv"
FP32_CONSOLIDATED = ARTIFACTS / "18-3-rtx5090-fp32.csv"

CONSOLIDATED_HEADER = ("run_index", "first_chunk_latency_ms", "session_id")


def _extract_first_chunk_latency_values(per_run_csvs: List[Path]) -> List[tuple[int, float, str]]:
    """For each input CSV, pull the FIRST first_chunk_latency_ms row only.

    Cold-start discipline: each launch's measurement is the first generation
    after the process started. If a CSV captured multiple
    first_chunk_latency_ms records (because Commander generated twice during
    one launch), the SECOND generation benefits from cuDNN benchmark autotune
    cache + allocator warmup + first-token-kernel-JIT warmup, so its timing
    reflects warmed-pipeline state, not the cold-start A/B Story 18.3
    measures. Take only the first record; warn on extras.

    Returns a list of (run_index, value_ms, session_id) tuples.
    """
    rows: List[tuple[int, float, str]] = []
    for path in per_run_csvs:
        # Extract run index from filename suffix (e.g., "run07.csv" --> 7).
        stem = path.stem  # "18-3-rtx5090-bf16-run07"
        run_idx = int(stem.split("-run")[-1])
        run_records: List[tuple[float, str]] = []
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row.get("metric_name") == "first_chunk_latency_ms":
                    try:
                        value = float(row["value"])
                    except (TypeError, ValueError):
                        print(f"  WARN: non-numeric first_chunk_latency_ms in {path.name}: {row}",
                              file=sys.stderr)
                        continue
                    run_records.append((value, row.get("session_id", "")))
        if not run_records:
            print(f"  WARN: no first_chunk_latency_ms record in {path.name}", file=sys.stderr)
            continue
        if len(run_records) > 1:
            extras = [f"{v:.1f}ms" for v, _s in run_records[1:]]
            print(
                f"  WARN: {path.name} has {len(run_records)} first_chunk records; "
                f"keeping only the cold-start record (first={run_records[0][0]:.1f}ms; "
                f"discarded warmed-pipeline records: {extras})",
                file=sys.stderr,
            )
        # Cold-start discipline: take the first record only.
        value, session_id = run_records[0]
        rows.append((run_idx, value, session_id))
    rows.sort()
    return rows


def _write_consolidated(path: Path, rows: List[tuple[int, float, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CONSOLIDATED_HEADER)
        for r in rows:
            writer.writerow(r)
    print(f"  wrote {path.name} ({len(rows)} rows)")


def _quantiles(values: List[float]) -> tuple[float, float, float]:
    """Return (median, p90, p95) for the values list."""
    if not values:
        return (0.0, 0.0, 0.0)
    median = statistics.median(values)
    if len(values) >= 10:
        # statistics.quantiles divides into n equal-sized intervals;
        # n=10 with method='inclusive' yields the 9 deciles (P10..P90).
        # P95 is interpolated separately because deciles top out at P90.
        deciles = statistics.quantiles(values, n=10, method="inclusive")
        p90 = deciles[8]  # 9th decile = P90
    else:
        p90 = max(values)
    sorted_vals = sorted(values)
    if len(sorted_vals) >= 2:
        # Linear interpolation of P95.
        rank = 0.95 * (len(sorted_vals) - 1)
        lo = int(rank)
        frac = rank - lo
        if lo + 1 < len(sorted_vals):
            p95 = sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])
        else:
            p95 = sorted_vals[-1]
    else:
        p95 = sorted_vals[0] if sorted_vals else 0.0
    return (median, p90, p95)


def _extract_steady_state_emit_intervals(per_run_csvs: List[Path]) -> dict:
    """Compute the producer-side steady-state ratio per Story 18.1's evidence
    file section4.4 methodology.

    For each CSV (one per fresh-process launch):
      * Read every ``progressive_chunk_emit_ms`` record (chronological — these
        are emitted by ``_record_progressive_chunk_emit`` in
        qwen_tts_service.py at the moment the producer hands a chunk to the
        consumer).
      * Drop the first chunk (cold-start decoder warmup; not steady-state).
      * Read the matching ``progressive_chunk_audio_duration_ms`` records to
        get each chunk's audio duration.
      * Compute per-chunk emit_interval_ms (gap between consecutive emits).
      * steady_state_ratio = mean(emit_interval) / mean(audio_duration). A
        ratio > 1 means the producer is slower than realtime --> audio gaps.

    Returns aggregated dict with the per-launch ratios + the across-launches
    median.
    """
    per_launch_ratios: List[tuple[int, float, float, float]] = []  # (run, mean_interval, mean_duration, ratio)
    for path in per_run_csvs:
        run_idx = int(path.stem.split("-run")[-1])
        emit_times_ms: List[float] = []
        durations_ms: List[float] = []
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                name = row.get("metric_name")
                try:
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                # progressive_chunk_emit_ms records the producer-side
                # wall-clock at chunk emission. progressive_chunk_audio_duration_ms
                # is the chunk's playback duration. Both have chunk_index tag.
                if name == "progressive_chunk_emit_ms":
                    emit_times_ms.append(value)
                elif name == "progressive_chunk_audio_duration_ms":
                    durations_ms.append(value)
        # Need at least 3 chunks (drop first; need 2+ intervals).
        if len(emit_times_ms) < 3 or len(durations_ms) < 2:
            continue
        # Drop the first chunk (cold-start). Sort defensively.
        emit_sorted = sorted(emit_times_ms)
        durations_sorted = sorted(durations_ms)[1:]
        intervals = [emit_sorted[i] - emit_sorted[i - 1] for i in range(2, len(emit_sorted))]
        if not intervals or not durations_sorted:
            continue
        mean_interval = sum(intervals) / len(intervals)
        mean_duration = sum(durations_sorted) / len(durations_sorted)
        if mean_duration <= 0:
            continue
        ratio = mean_interval / mean_duration
        per_launch_ratios.append((run_idx, mean_interval, mean_duration, ratio))
    if not per_launch_ratios:
        return {"per_launch": [], "median_ratio": 0.0, "median_interval": 0.0, "median_duration": 0.0, "n": 0}
    ratios = sorted(r for (_i, _mi, _md, r) in per_launch_ratios)
    intervals = sorted(mi for (_i, mi, _md, _r) in per_launch_ratios)
    durations = sorted(md for (_i, _mi, md, _r) in per_launch_ratios)
    return {
        "per_launch": per_launch_ratios,
        "median_ratio": statistics.median(ratios),
        "median_interval": statistics.median(intervals),
        "median_duration": statistics.median(durations),
        "n": len(per_launch_ratios),
    }


def main() -> int:
    if not BF16_RUNS:
        print(f"FATAL: no bf16 run CSVs found at {ARTIFACTS}/18-3-rtx5090-bf16-run*.csv",
              file=sys.stderr)
        return 2
    if not FP32_RUNS:
        print(f"FATAL: no fp32 run CSVs found at {ARTIFACTS}/18-3-rtx5090-fp32-run*.csv",
              file=sys.stderr)
        return 2

    print(f"BF16 input CSVs: {len(BF16_RUNS)}")
    for p in BF16_RUNS:
        print(f"  - {p.name}")
    print(f"FP32 input CSVs: {len(FP32_RUNS)}")
    for p in FP32_RUNS:
        print(f"  - {p.name}")
    print()

    bf16_rows = _extract_first_chunk_latency_values(BF16_RUNS)
    fp32_rows = _extract_first_chunk_latency_values(FP32_RUNS)

    print(f"BF16 first_chunk_latency_ms records: {len(bf16_rows)}")
    print(f"FP32 first_chunk_latency_ms records: {len(fp32_rows)}")
    print()

    # Per-launch first-chunk-latency dump for spread visibility.
    print("Per-launch first_chunk_latency_ms (cold-start):")
    print(f"  {'run':<5} {'bf16 (ms)':<12} {'fp32 (ms)':<12} {'delta (ms)':<12}")
    bf16_by_run = {i: v for (i, v, _s) in bf16_rows}
    fp32_by_run = {i: v for (i, v, _s) in fp32_rows}
    all_runs = sorted(set(bf16_by_run.keys()) | set(fp32_by_run.keys()))
    for run in all_runs:
        b = bf16_by_run.get(run)
        f = fp32_by_run.get(run)
        delta = (f - b) if (b is not None and f is not None) else None
        b_str = f"{b:.1f}" if b is not None else "-"
        f_str = f"{f:.1f}" if f is not None else "-"
        d_str = f"{delta:+.1f}" if delta is not None else "-"
        print(f"  {run:<5} {b_str:<12} {f_str:<12} {d_str:<12}")
    print()

    _write_consolidated(BF16_CONSOLIDATED, bf16_rows)
    _write_consolidated(FP32_CONSOLIDATED, fp32_rows)
    print()

    bf16_values = [v for (_i, v, _s) in bf16_rows]
    fp32_values = [v for (_i, v, _s) in fp32_rows]

    bf16_med, bf16_p90, bf16_p95 = _quantiles(bf16_values)
    fp32_med, fp32_p90, fp32_p95 = _quantiles(fp32_values)

    def _pct(fp32_val: float, bf16_val: float) -> float:
        if fp32_val == 0:
            return 0.0
        return (fp32_val - bf16_val) / fp32_val * 100.0

    print("=" * 70)
    print("NFR1 first_chunk_latency_ms summary (cold-start; first record per launch)")
    print("=" * 70)
    print(f"{'':<10} {'bf16 (auto)':<18} {'fp32 (override)':<18} {'delta (ms)':<14} {'delta (%)':<10}")
    print(f"{'median':<10} {bf16_med:<18.3f} {fp32_med:<18.3f} {fp32_med - bf16_med:<14.3f} {_pct(fp32_med, bf16_med):<10.2f}")
    print(f"{'p90':<10} {bf16_p90:<18.3f} {fp32_p90:<18.3f} {fp32_p90 - bf16_p90:<14.3f} {_pct(fp32_p90, bf16_p90):<10.2f}")
    print(f"{'p95':<10} {bf16_p95:<18.3f} {fp32_p95:<18.3f} {fp32_p95 - bf16_p95:<14.3f} {_pct(fp32_p95, bf16_p95):<10.2f}")
    print(f"N (bf16): {len(bf16_values)}")
    print(f"N (fp32): {len(fp32_values)}")
    print()

    # Story 18.1 producer-bottleneck steady-state ratio comparison. First-chunk
    # latency is dominated by the small-batch autoregressive forward kernel-
    # launch overhead (not matmul throughput); the producer-bottleneck metric
    # captures the steady-state ratio that Story 18.1 pinned at 3.23×.
    print("=" * 70)
    print("Producer-bottleneck steady-state ratio (Story 18.1 section4.4 methodology)")
    print("=" * 70)
    bf16_ss = _extract_steady_state_emit_intervals(BF16_RUNS)
    fp32_ss = _extract_steady_state_emit_intervals(FP32_RUNS)
    print(f"{'':<10} {'bf16 (auto)':<18} {'fp32 (override)':<18} {'delta':<14}")
    if bf16_ss["n"] > 0 and fp32_ss["n"] > 0:
        print(
            f"{'mean interval (ms)':<10} "
            f"{bf16_ss['median_interval']:<18.3f} "
            f"{fp32_ss['median_interval']:<18.3f} "
            f"{fp32_ss['median_interval'] - bf16_ss['median_interval']:<14.3f}"
        )
        print(
            f"{'mean duration (ms)':<10} "
            f"{bf16_ss['median_duration']:<18.3f} "
            f"{fp32_ss['median_duration']:<18.3f} "
            f"{fp32_ss['median_duration'] - bf16_ss['median_duration']:<14.3f}"
        )
        print(
            f"{'ratio (>1=gaps)':<10} "
            f"{bf16_ss['median_ratio']:<18.3f} "
            f"{fp32_ss['median_ratio']:<18.3f} "
            f"{fp32_ss['median_ratio'] - bf16_ss['median_ratio']:<14.3f}"
        )
        ratio_pct = _pct(fp32_ss["median_ratio"], bf16_ss["median_ratio"])
        print(f"{'ratio delta (%)':<10} {ratio_pct:<14.2f}")
        print(f"N launches with sufficient chunks: bf16={bf16_ss['n']}, fp32={fp32_ss['n']}")
        print()
        print(f"Story 18.1 baseline ratio: 3.23 (memory/epic18_producer_bottleneck_finding.md)")
        if bf16_ss["median_ratio"] < fp32_ss["median_ratio"]:
            print(f"--> bf16 IMPROVES the producer ratio by {fp32_ss['median_ratio'] - bf16_ss['median_ratio']:.3f} ({ratio_pct:.1f}%)")
        else:
            print(f"--> bf16 does NOT improve the producer ratio on this workload.")
    else:
        print(
            f"WARN: insufficient steady-state data "
            f"(bf16 n={bf16_ss['n']}, fp32 n={fp32_ss['n']}); "
            f"need >=3 progressive_chunk_emit_ms records per launch."
        )
    print()

    # Task 7.4 routing condition surfacing
    speedup_pct = _pct(fp32_med, bf16_med)
    print(f"Median speedup: {speedup_pct:.2f}%")
    if speedup_pct < 20.0:
        print()
        print("WARN: Median speedup < 20% — Story 18.3 Task 7.4 routing condition triggered.")
        print("      Anticipated gate at epics-optimization-pass.md:1381 = [30%, 50%].")
        print("      Route to Open Question #3 BEFORE running the audition (Task 8).")
    elif 20.0 <= speedup_pct < 30.0:
        print()
        print("NOTE: Median speedup in [20%, 30%) — below the lower bound of the")
        print("      anticipated gate but above the OQ #3 threshold. Story may proceed")
        print("      to audition; surface the partial gain to Commander in closure.")
    else:
        print()
        print(f"OK: Median speedup in the anticipated range — proceed to Task 8 audition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
