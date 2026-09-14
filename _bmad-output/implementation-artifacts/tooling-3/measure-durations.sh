#!/usr/bin/env bash
# Story tooling-3 Task 2 — measure the per-test duration distribution.
#
# Runs every test directory in its own pytest process (same non-recursive
# explicit-file-list shape as 20-8-suite-with-hang-guard.sh, so the numbers
# are comparable to Story 20.8's per-directory evidence) under a PROVISIONAL
# --timeout=900 thread guard. Per-test durations stream to a JSONL file via
# the durations_jsonl plugin, so a directory that hits the guard still yields
# the durations of every test that completed before the abort; the aborted
# directory is then re-run file by file so the tests after the hang are
# measured too.
#
# Usage: bash measure-durations.sh <label>
set -u
LABEL="${1:?usage: measure-durations.sh <label>}"
cd "$(dirname "$0")/../../.." || exit 1
PY="python310/python.exe"
HERE="_bmad-output/implementation-artifacts/tooling-3"
LOG="$HERE/measure-$LABEL.log"
export TOOLING3_DURATIONS="$HERE/measure-$LABEL.jsonl"
GUARD="${GUARD:-900}"
: > "$LOG"; : > "$TOOLING3_DURATIONS"
echo "started $(date -Is) guard ${GUARD}s" | tee -a "$LOG"

run_one () {  # $@ = files
    $PY -u "$HERE/run_pytest.py" "$@" -v -rfE --timeout="$GUARD" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && tail -c 4000 "$LOG" | grep -q "+++ Timeout +++"; then
        return 124
    fi
    return $rc
}

DIRS=$(find tests -name "test_*.py" | sed 's|/[^/]*$||' | sort -u)
for d in $DIRS; do
    files=$(ls "$d"/test_*.py 2>/dev/null)
    [ -n "$files" ] || continue
    echo "--- dir: $d ---" | tee -a "$LOG"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    run_one $files; rc=$?
    echo "    dir rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$LOG"
    if [ $rc -eq 124 ]; then
        echo "DIR HIT GUARD — re-running file by file" | tee -a "$LOG"
        for f in $files; do
            echo "--- file: $f ---" | tee -a "$LOG"
            t1=$(date +%s)
            run_one "$f"; rc=$?
            echo "    file rc=$rc wall=$(( $(date +%s) - t1 ))s" | tee -a "$LOG"
            [ $rc -eq 124 ] && echo "TIMEOUT in $f" | tee -a "$LOG"
        done
    fi
done
echo "finished $(date -Is)" | tee -a "$LOG"
