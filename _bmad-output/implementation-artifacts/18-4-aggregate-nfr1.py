"""Story 18.4 Task 8.3 — aggregate N=10 CSVs per branch (3-way A/B/C) and
compute the bf16+compile vs bf16-eager vs fp32-eager deltas.

Reads three sets of per-iteration CSVs:
  - _bmad-output/implementation-artifacts/18-4-rtx5090-bf16-compile-run01.csv ... run10.csv   (BRANCH A)
  - _bmad-output/implementation-artifacts/18-4-rtx5090-bf16-eager-run01.csv   ... run10.csv   (BRANCH B)
  - _bmad-output/implementation-artifacts/18-4-rtx5090-fp32-eager-run01.csv   ... run10.csv   (BRANCH C)

Each input CSV has the 6-column schema from progressive_playback_csv_capture.py:
    metric_name,value,session_id,chunk_index,is_final,audio_data_size

Outputs three consolidated CSVs + computes:
  * Median + p90 + p95 per branch (cold-compile run #1 discarded from BRANCH A)
  * A vs B (compile gain over bf16-eager)
  * A vs C (compounded gain over fp32-eager)
  * B vs C (bf16-only re-validation)
  * Producer-bottleneck steady-state ratio per branch (Story 18.1 §4.4 methodology)
  * Open Question #1 routing surface (sub-20% A-vs-B speedup OR branch-A ratio
    not below 1.0× sustained)

Cold-compile discipline (architecture D-23 + Story 18.4 Task 8.4):
  Branch A's RUN #1 is cold-compile (~10-30s additional startup); the
  aggregator DISCARDS run #1 from the median calculation and reports it
  separately. Branch B and Branch C have no cold-compile cost; all 10 runs
  participate in their medians.

Usage (from repo root):
    python310/python.exe _bmad-output/implementation-artifacts/18-4-aggregate-nfr1.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import List


ARTIFACTS = Path(__file__).resolve().parent
A_RUNS = sorted(ARTIFACTS.glob("18-4-rtx5090-bf16-compile-run*.csv"))  # BRANCH A
B_RUNS = sorted(ARTIFACTS.glob("18-4-rtx5090-bf16-eager-run*.csv"))    # BRANCH B
C_RUNS = sorted(ARTIFACTS.glob("18-4-rtx5090-fp32-eager-run*.csv"))    # BRANCH C
A_CONSOLIDATED = ARTIFACTS / "18-4-rtx5090-bf16-compile.csv"
B_CONSOLIDATED = ARTIFACTS / "18-4-rtx5090-bf16-eager.csv"
C_CONSOLIDATED = ARTIFACTS / "18-4-rtx5090-fp32-eager.csv"

CONSOLIDATED_HEADER = ("run_index", "first_chunk_latency_ms", "session_id", "is_cold_compile")


def _extract_first_chunk_latency_values(per_run_csvs: List[Path]) -> List[tuple[int, float, str]]:
    """For each input CSV, pull the FIRST first_chunk_latency_ms row only.

    Cold-start discipline (Story 18.3 precedent): each launch's measurement is
    the first generation after the process started. If a CSV captured multiple
    first_chunk_latency_ms records (Commander generated twice during one
    launch), the SECOND generation benefits from cuDNN benchmark cache +
    allocator warmup + first-token-kernel-JIT warmup, so its timing reflects
    warmed-pipeline state, not the cold-start. Take only the first record;
    warn on extras.

    Returns a list of (run_index, value_ms, session_id) tuples.
    """
    rows: List[tuple[int, float, str]] = []
    for path in per_run_csvs:
        stem = path.stem
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
        value, session_id = run_records[0]
        rows.append((run_idx, value, session_id))
    rows.sort()
    return rows


def _write_consolidated(path: Path, rows: List[tuple[int, float, str]], cold_compile_run: int = -1) -> None:
    """Write the consolidated CSV. Branch A marks run #1 as cold-compile.

    The is_cold_compile column lets a future reader filter out the cold-
    compile outlier without re-deriving the discipline from comments.
    """
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CONSOLIDATED_HEADER)
        for r in rows:
            run_idx, value, session_id = r
            is_cold = "yes" if run_idx == cold_compile_run else "no"
            writer.writerow((run_idx, value, session_id, is_cold))
    print(f"  wrote {path.name} ({len(rows)} rows)")


def _quantiles(values: List[float]) -> tuple[float, float, float]:
    """Return (median, p90, p95) for the values list."""
    if not values:
        return (0.0, 0.0, 0.0)
    median = statistics.median(values)
    if len(values) >= 10:
        deciles = statistics.quantiles(values, n=10, method="inclusive")
        p90 = deciles[8]
    else:
        p90 = max(values)
    sorted_vals = sorted(values)
    if len(sorted_vals) >= 2:
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


def _extract_steady_state_emit_intervals(per_run_csvs: List[Path], skip_run: int = -1) -> dict:
    """Compute the producer-side steady-state ratio per Story 18.1's §4.4 methodology.

    Optionally skips ``skip_run`` (cold-compile discipline for branch A's run 1).
    """
    per_launch_ratios: List[tuple[int, float, float, float]] = []
    for path in per_run_csvs:
        run_idx = int(path.stem.split("-run")[-1])
        if run_idx == skip_run:
            continue
        emit_times_ms: List[float] = []
        durations_ms: List[float] = []
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                name = row.get("metric_name")
                try:
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                if name == "progressive_chunk_emit_ms":
                    emit_times_ms.append(value)
                elif name == "progressive_chunk_audio_duration_ms":
                    durations_ms.append(value)
        if len(emit_times_ms) < 3 or len(durations_ms) < 2:
            continue
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


def _pct(baseline: float, treatment: float) -> float:
    """Percent change from baseline to treatment. Positive = treatment is faster."""
    if baseline == 0:
        return 0.0
    return (baseline - treatment) / baseline * 100.0


def main() -> int:
    if not A_RUNS or not B_RUNS or not C_RUNS:
        print("FATAL: missing input CSVs:", file=sys.stderr)
        print(f"  Branch A (bf16+compile): {len(A_RUNS)} found", file=sys.stderr)
        print(f"  Branch B (bf16+eager):   {len(B_RUNS)} found", file=sys.stderr)
        print(f"  Branch C (fp32+eager):   {len(C_RUNS)} found", file=sys.stderr)
        return 2

    print(f"Branch A (bf16+compile) input CSVs: {len(A_RUNS)}")
    print(f"Branch B (bf16+eager)   input CSVs: {len(B_RUNS)}")
    print(f"Branch C (fp32+eager)   input CSVs: {len(C_RUNS)}")
    print()

    a_rows_full = _extract_first_chunk_latency_values(A_RUNS)
    b_rows = _extract_first_chunk_latency_values(B_RUNS)
    c_rows = _extract_first_chunk_latency_values(C_RUNS)

    # Cold-compile discipline: branch A's run #1 is discarded from median.
    a_cold_run = 1
    a_cold_value = None
    a_rows = []
    for r in a_rows_full:
        if r[0] == a_cold_run:
            a_cold_value = r[1]
        else:
            a_rows.append(r)

    print(f"Branch A records: {len(a_rows_full)} total; "
          f"run #1 cold-compile = {a_cold_value:.1f}ms (DISCARDED from median); "
          f"warm-cache N = {len(a_rows)}" if a_cold_value is not None else f"Branch A records: {len(a_rows_full)} (no cold-compile data)")
    print(f"Branch B records: {len(b_rows)}")
    print(f"Branch C records: {len(c_rows)}")
    print()

    # Per-launch dump for spread visibility (all three branches side-by-side).
    a_by_run = {i: v for (i, v, _s) in a_rows_full}
    b_by_run = {i: v for (i, v, _s) in b_rows}
    c_by_run = {i: v for (i, v, _s) in c_rows}
    all_runs = sorted(set(a_by_run.keys()) | set(b_by_run.keys()) | set(c_by_run.keys()))
    print("Per-launch first_chunk_latency_ms (cold-start record):")
    print(f"  {'run':<5} {'A (bf16+compile)':<20} {'B (bf16+eager)':<18} {'C (fp32+eager)':<18}")
    for run in all_runs:
        a = a_by_run.get(run)
        b = b_by_run.get(run)
        c = c_by_run.get(run)
        marker = " (cold-compile)" if run == a_cold_run and a is not None else ""
        a_str = f"{a:.1f}{marker}" if a is not None else "-"
        b_str = f"{b:.1f}" if b is not None else "-"
        c_str = f"{c:.1f}" if c is not None else "-"
        print(f"  {run:<5} {a_str:<20} {b_str:<18} {c_str:<18}")
    print()

    _write_consolidated(A_CONSOLIDATED, a_rows_full, cold_compile_run=a_cold_run)
    _write_consolidated(B_CONSOLIDATED, b_rows)
    _write_consolidated(C_CONSOLIDATED, c_rows)
    print()

    a_values = [v for (_i, v, _s) in a_rows]  # warm-cache only
    b_values = [v for (_i, v, _s) in b_rows]
    c_values = [v for (_i, v, _s) in c_rows]

    a_med, a_p90, a_p95 = _quantiles(a_values)
    b_med, b_p90, b_p95 = _quantiles(b_values)
    c_med, c_p90, c_p95 = _quantiles(c_values)

    print("=" * 78)
    print("NFR1 first_chunk_latency_ms summary (cold-start per launch; branch A warm-cache only)")
    print("=" * 78)
    print(f"{'':<10} {'A (bf16+compile)':<20} {'B (bf16+eager)':<18} {'C (fp32+eager)':<18}")
    print(f"{'median':<10} {a_med:<20.3f} {b_med:<18.3f} {c_med:<18.3f}")
    print(f"{'p90':<10} {a_p90:<20.3f} {b_p90:<18.3f} {c_p90:<18.3f}")
    print(f"{'p95':<10} {a_p95:<20.3f} {b_p95:<18.3f} {c_p95:<18.3f}")
    print(f"N (warm A / full B / full C): {len(a_values)} / {len(b_values)} / {len(c_values)}")
    print()

    print("Pairwise deltas (positive = treatment faster than baseline):")
    print(f"  A vs B (compile gain over bf16-eager):  "
          f"median Δ={b_med - a_med:+.1f}ms ({_pct(b_med, a_med):+.2f}%)")
    print(f"  A vs C (compounded gain over fp32-eager): "
          f"median Δ={c_med - a_med:+.1f}ms ({_pct(c_med, a_med):+.2f}%)")
    print(f"  B vs C (bf16-only re-validation):       "
          f"median Δ={c_med - b_med:+.1f}ms ({_pct(c_med, b_med):+.2f}%)")
    print()

    # Producer-bottleneck steady-state ratio per branch.
    print("=" * 78)
    print("Producer-bottleneck steady-state ratio (Story 18.1 §4.4 methodology)")
    print("=" * 78)
    a_ss = _extract_steady_state_emit_intervals(A_RUNS, skip_run=a_cold_run)
    b_ss = _extract_steady_state_emit_intervals(B_RUNS)
    c_ss = _extract_steady_state_emit_intervals(C_RUNS)
    print(f"  Branch A (bf16+compile) ratio: {a_ss['median_ratio']:.3f}  (N={a_ss['n']})")
    print(f"  Branch B (bf16+eager)   ratio: {b_ss['median_ratio']:.3f}  (N={b_ss['n']})")
    print(f"  Branch C (fp32+eager)   ratio: {c_ss['median_ratio']:.3f}  (N={c_ss['n']})")
    print()
    print(f"  Story 18.1 baseline ratio: 3.23 (memory/epic18_producer_bottleneck_finding.md)")
    print(f"  Story 18.2 closed to:      1.40 (fp32+TF32+cuDNN benchmark)")
    print(f"  Story 18.3 measured:       1.62 (bf16+TF32; net null over 18.2)")
    print(f"  Story 18.4 target (OFR-E): <1.0× sustained on Branch A")
    print()
    if a_ss["n"] > 0 and a_ss["median_ratio"] < 1.0:
        print(f"  ✓ Branch A ratio < 1.0 — OFR-E target achieved.")
    elif a_ss["n"] > 0:
        print(f"  ✗ Branch A ratio >= 1.0 — OFR-E NOT achieved on this measurement.")
    else:
        print(f"  ? Branch A steady-state data insufficient (need ≥3 emit records per launch).")
    print()

    # Task 8.6 routing condition surfacing
    speedup_pct = _pct(b_med, a_med)  # A vs B compile gain
    print("=" * 78)
    print("Task 8.6 routing condition (Open Question #1)")
    print("=" * 78)
    print(f"Median A-vs-B speedup (compile gain): {speedup_pct:.2f}%")
    print(f"Branch A producer ratio: {a_ss['median_ratio']:.3f}")
    print()
    if speedup_pct < 20.0:
        print("⚠ Median speedup < 20% — Story 18.4 Task 8.6 routing condition TRIGGERED.")
        print("  Route to Open Question #1 BEFORE running the joint audition (Task 9).")
        print("  Anticipated gate at story:1402 = 1.5–3× warm-cache decode speedup.")
        print("  Sub-20% suggests compile didn't apply to the talker's autoregressive")
        print("  loop where Story 18.1's bottleneck lives. Commander decides next step.")
    elif a_ss["n"] > 0 and a_ss["median_ratio"] >= 1.0:
        print("⚠ Branch A ratio >= 1.0 — Story 18.4 OFR-E NOT met.")
        print("  Route to Open Question #1: producer-bottleneck not closed by compile.")
    else:
        print("✓ Speedup ≥ 20% and producer ratio < 1.0 — proceed to Task 9 audition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
