"""verify_reload_compile.py — bundled-exe verifier for the compile-disengage spec.

Host-side driver script. Locates ``dist/MyVoice/MyVoice.exe``, runs the
``--reload-compile-repro`` flag once per ``MYVOICE_RELOAD_COMPILE_FIX`` value,
parses each run's ``RELOAD_COMPILE_REPRO:`` marker line, collects the
``reload_compile_diagnostic`` metric entries from the fresh ``myvoice.log``
slice, prints the 5-row verdict table per TD-8 of the tech-spec, and writes
``verify_reload_compile_verdict.md`` as the canonical handoff artifact.

Exit codes:
    0 — at least one non-off fix produced a PASS row
    1 — no fix passed (post-merge follow-up required)
    3 — invariant violation: at least one row's ``initial`` column is not
        ``engaged_cold_compile`` / ``engaged_warm_cache`` (a fix broke the
        initial-compile path)

Usage::

    python build_tools/verify_reload_compile.py [--bundle-dir PATH]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUNDLE_DIR = _REPO_ROOT / "dist" / "MyVoice"
_VERDICT_FILE = Path(__file__).resolve().parent / "verify_reload_compile_verdict.md"

_FIX_VALUES = ("off", "post_gen", "rthook", "triton", "all")
_INITIAL_INVARIANT_OK = ("engaged_cold_compile", "engaged_warm_cache")
_RELOAD_PASS = ("engaged_cold_compile", "engaged_warm_cache")

_MARKER_RE = re.compile(
    r"^RELOAD_COMPILE_REPRO:\s+fix=(\S+)\s+initial=(\S+)\s+reload=(\S+)\s*$",
    re.MULTILINE,
)


def _locate_log_file(bundle_dir: Path) -> Optional[Path]:
    """The bundled app writes to ``<bundle_dir>/logs/myvoice.log`` via
    ``portable_paths.get_logs_path()``. Locate the file (creating the dir
    if missing) and return the path.
    """
    log_path = bundle_dir / "logs" / "myvoice.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _truncate_log(log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    try:
        log_path.write_text("", encoding="utf-8")
    except Exception as exc:
        print(f"WARNING: could not truncate {log_path}: {exc}", file=sys.stderr)


def _read_log(log_path: Optional[Path]) -> str:
    if log_path is None:
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        print(f"WARNING: could not read {log_path}: {exc}", file=sys.stderr)
        return ""


def _extract_diagnostic_records(log_text: str) -> List[Dict[str, str]]:
    """Pull every ``reload_compile_diagnostic`` metric record out of myvoice.log.

    The logging.basicConfig format in ``main.py:setup_logging`` is::

        %(asctime)s - %(name)s - %(levelname)s - %(message)s

    Metric records are emitted via ``_logger.info("metric", extra=...)`` so the
    message is literally ``"metric"`` and the tag payload lands in the
    LogRecord's ``extra`` dict. Because logging.Formatter does NOT serialize
    ``extra`` by default, the on-disk lines for metrics show only the
    ``"metric"`` string. Our diagnostic emission attaches tags via kwargs to
    ``metrics.record``; with the default formatter those tags don't reach the
    file.

    F9 fix: precision-match on the on-disk shape of relevant log lines
    rather than substring-grepping. The three buckets we want:

      * ``myvoice.metrics - INFO - metric`` — the tag payload lives in
        LogRecord.extra and is missing from the formatted line, but the
        presence of the line confirms emission count.
      * ``torch_runtime.*compile.+(cuda_unavailable|engaged_)`` — the gate
        result INFO lines from ``engage_compile_optimizations``.
      * ``ModelRegistry.+model loaded.+compile_engaged=`` — the post-load
        INFO from ``_load_model_sync``.

    The previous substring match (``"reload_compile" in line``) caught
    unrelated dispatch breadcrumbs and user-named voice profiles.
    """
    pattern = re.compile(
        r"(?:- myvoice\.metrics - INFO - metric\b)"
        r"|(?:torch_runtime[^-]*INFO[^|]*torch\.compile\b)"
        r"|(?:ModelRegistry[^-]*INFO[^|]*compile_engaged=)"
    )
    records: List[Dict[str, str]] = []
    for line in log_text.splitlines():
        if pattern.search(line):
            records.append({"line": line.strip()})
    return records


def _run_one(
    exe: Path,
    fix: str,
    bundle_dir: Path,
    timeout: float = 600.0,
) -> Tuple[int, str, str]:
    """Run the bundled exe once with the given fix value.

    Returns (return_code, stdout, stderr).
    """
    env = dict(os.environ)
    env["MYVOICE_RELOAD_COMPILE_FIX"] = fix
    env["MYVOICE_DISABLE_COMPILE_WARMUP"] = "1"
    # Force HF offline so the bundled exe's huggingface_hub does not hit the
    # network — the bundled Python's cert store does not validate
    # huggingface.co's chain (2026-05-18 first run: SSLCertVerificationError).
    # Models are already in the HF cache via the installer's predownload.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    cmd = [str(exe), "--reload-compile-repro", f"--fix={fix}"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(bundle_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"
    except Exception as exc:
        return 125, "", f"subprocess.run raised: {type(exc).__name__}: {exc}"


def _parse_marker(stdout: str, fix: str) -> Tuple[str, str]:
    """Extract (initial_reason, reload_reason) from the marker line."""
    for m in _MARKER_RE.finditer(stdout):
        if m.group(1) == fix:
            return m.group(2), m.group(3)
    # Fallback: any marker line at all (in case --fix= parsing differs)
    m = _MARKER_RE.search(stdout)
    if m is not None:
        return m.group(2), m.group(3)
    return "no_marker", "no_marker"


def _verdict_for_row(initial: str, reload: str) -> str:
    if initial not in _INITIAL_INVARIANT_OK:
        return "INVARIANT_VIOLATED"
    if reload in _RELOAD_PASS:
        return "PASS"
    if reload == "cuda_unavailable":
        return "FAIL_BUG_CONFIRMED"
    if reload == "not_evaluated":
        return "FAIL_INITIAL_INVARIANT"
    return f"FAIL_{reload}"


def _format_verdict_table(rows: List[Dict[str, str]]) -> str:
    """5-row verdict table per TD-8."""
    lines = []
    lines.append(
        f"{'fix':<10} {'initial':<24} {'reload':<24} VERDICT"
    )
    lines.append("-" * 80)
    for row in rows:
        lines.append(
            f"{row['fix']:<10} {row['initial']:<24} {row['reload']:<24} {row['verdict']}"
        )
    return "\n".join(lines)


def _write_verdict_file(rows: List[Dict[str, str]], summary_line: str) -> None:
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    body = []
    body.append("# Reload-Compile Verify Verdict")
    body.append("")
    body.append(f"**Generated:** {timestamp}")
    body.append("")
    body.append(f"**Summary:** {summary_line}")
    body.append("")
    body.append("## Verdict Table")
    body.append("")
    body.append("```")
    body.append(_format_verdict_table(rows))
    body.append("```")
    body.append("")
    body.append("## Per-Row Detail")
    body.append("")
    for row in rows:
        body.append(f"### fix = `{row['fix']}`")
        body.append("")
        body.append(f"- **verdict:** {row['verdict']}")
        body.append(f"- **initial reason:** `{row['initial']}`")
        body.append(f"- **reload reason:** `{row['reload']}`")
        body.append(f"- **exe exit code:** {row['exit_code']}")
        if row.get("log_excerpt"):
            body.append("")
            body.append("**Log excerpt (first 20 matching lines):**")
            body.append("")
            body.append("```")
            for line in row["log_excerpt"][:20]:
                body.append(line)
            body.append("```")
        if row.get("stderr_tail"):
            body.append("")
            body.append("**Stderr tail:**")
            body.append("")
            body.append("```")
            body.append(row["stderr_tail"][-1500:])
            body.append("```")
        body.append("")
    _VERDICT_FILE.write_text("\n".join(body), encoding="utf-8")
    print(f"Verdict file written: {_VERDICT_FILE}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=_DEFAULT_BUNDLE_DIR,
        help=f"Path to the bundled MyVoice directory (default: {_DEFAULT_BUNDLE_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-run subprocess timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--fixes",
        type=str,
        default=",".join(_FIX_VALUES),
        help=f"Comma-separated list of fix values to test (default: {','.join(_FIX_VALUES)})",
    )
    args = parser.parse_args(argv)

    bundle_dir = args.bundle_dir.resolve()
    exe = bundle_dir / "MyVoice.exe"
    if not exe.exists():
        print(f"ERROR: bundle exe not found at {exe}", file=sys.stderr)
        print("Build the bundle first: python build_tools/build_portable.py", file=sys.stderr)
        return 2

    log_path = _locate_log_file(bundle_dir)
    fixes = [f.strip() for f in args.fixes.split(",") if f.strip()]
    unknown = [f for f in fixes if f not in _FIX_VALUES]
    if unknown:
        print(f"WARNING: unknown fix values will still be sent verbatim: {unknown}", file=sys.stderr)

    rows: List[Dict[str, str]] = []
    print(f"Running verifier against {exe}")
    print(f"Per-run timeout: {args.timeout}s")
    print()
    for fix in fixes:
        print(f"=== fix={fix} ===", flush=True)
        _truncate_log(log_path)
        t0 = time.time()
        rc, stdout, stderr = _run_one(exe, fix, bundle_dir, timeout=args.timeout)
        elapsed = time.time() - t0
        initial, reload = _parse_marker(stdout, fix)
        verdict = _verdict_for_row(initial, reload)
        log_text = _read_log(log_path)
        log_excerpt = [r["line"] for r in _extract_diagnostic_records(log_text)]
        row = {
            "fix": fix,
            "initial": initial,
            "reload": reload,
            "verdict": verdict,
            "exit_code": str(rc),
            "elapsed_sec": f"{elapsed:.1f}",
            "log_excerpt": log_excerpt,
            "stderr_tail": stderr,
        }
        rows.append(row)
        print(
            f"  -> initial={initial}, reload={reload}, verdict={verdict} "
            f"(exit={rc}, {elapsed:.1f}s)",
            flush=True,
        )
        print()

    print()
    table = _format_verdict_table(rows)
    print(table)
    print()

    # Determine exit code per AC.
    invariant_violated = any(row["initial"] not in _INITIAL_INVARIANT_OK for row in rows)
    non_off_passes = [row for row in rows if row["fix"] != "off" and row["verdict"] == "PASS"]

    # F5 Heisenbug detection: the spec's pre-mortem risk #4 anticipates
    # "instrumentation altered failure mode". If fix=off (which runs zero
    # H1/H2/H3 fix logic) PASSes, the bug did NOT reproduce — either the
    # always-on diagnostic probe is unintentionally re-initialising CUDA
    # state, or the failure is non-deterministic. Flag this in the summary
    # rather than reporting "fix=off is the winning fix" which would
    # mislead investigators.
    off_row = next((r for r in rows if r["fix"] == "off"), None)
    bug_did_not_reproduce = (
        off_row is not None
        and off_row["verdict"] == "PASS"
        and off_row["reload"] != "cuda_unavailable"
    )

    if invariant_violated:
        summary = (
            f"INVARIANT VIOLATED — at least one row's initial column is not "
            f"engaged_*; a fix broke the working initial-compile path. "
            f"exit_code=3."
        )
        exit_code = 3
    elif bug_did_not_reproduce:
        summary = (
            "PASS (bug did NOT reproduce) — fix=off shows reload="
            f"{off_row['reload']!r}, but the spec's failing trace requires "
            "reload=cuda_unavailable to confirm the bug exists in this build. "
            "Either (a) the always-on diagnostic probe is unintentionally "
            "fixing the bug (pre-mortem risk #4 — Heisenbug), or (b) the "
            "failure is non-deterministic and didn't fire on this run. "
            "Manual smoke gate (GUI voice-switch + generate) remains the "
            "canonical acceptance test. exit_code=0."
        )
        exit_code = 0
    elif non_off_passes:
        winning = non_off_passes[0]["fix"]
        summary = (
            f"PASS — winning fix is `{winning}` (lowest-impact non-off PASS). "
            f"exit_code=0."
        )
        exit_code = 0
    else:
        summary = (
            "FAIL — no non-off fix produced a PASS row. Follow-up: spec a "
            "deeper investigation of torch.cuda.is_available()'s C-side "
            "implementation in this torch build. exit_code=1."
        )
        exit_code = 1

    print(summary)
    _write_verdict_file(rows, summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
