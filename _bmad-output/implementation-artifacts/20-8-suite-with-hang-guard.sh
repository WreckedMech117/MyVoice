#!/usr/bin/env bash
# Story 20.8 AC #4 — run the whole suite under a wall-clock guard.
#
# WHY THIS EXISTS
# ---------------
# On current `main` (c57adfd) several tests HANG rather than fail: CPU pins at
# a constant value and the run never returns. Confirmed, each reproducing in
# isolation on a clean tree with no Story 20.8 change applied:
#
#   tests/settings/test_reset_to_defaults.py::TestResetQuickSpeak::test_reset_quick_speak_entries
#   tests/ui/test_close_to_tray_toggle.py::TestInterfaceTabCloseBehaviorControl::test_control_exists_and_is_a_two_option_combo
#   ...and at least one more inside the bulk of `tests/`, around the 90 % mark.
#
# They are dialog/UI tests; none touches streaming geometry. `pytest-timeout`
# is not installed in the portable interpreter, so the guard must be external —
# and a plain `pytest tests/` cannot complete at all.
#
# A hang is not a result the AC #4 comparison can score. So:
#
#   1. every test DIRECTORY runs in its own guarded process;
#   2. a directory that does not return has its partial output DISCARDED and is
#      re-run FILE BY FILE, also guarded, so one hanging file costs one file
#      rather than a directory;
#   3. a file that still does not return is named as HUNG.
#
# Discarding the timed-out directory's partial log matters: pytest is killed
# before printing its summary line, so those tests would otherwise be counted
# once from the partial output and again from the fallback.
#
# The same script produces the before and the after run, so the comparison is
# like-for-like — which is the property AC #4 actually needs ("the pre-existing
# failure set unchanged in count and identity"), and one a partial run cannot
# have.
#
# Usage:  bash 20-8-suite-with-hang-guard.sh <label>
#         -> 20-8-regression-<label>.log      (kept pytest output)
#            20-8-regression-<label>.summary  (counts + sorted FAILED ids + HUNG)
#
# Working file — gitignored under `_bmad-output/`; force-add per
# `memory/git_repo_state.md`.

set -u
LABEL="${1:?usage: 20-8-suite-with-hang-guard.sh <label>}"
cd "$(dirname "$0")/../.." || exit 1
PY="python310/python.exe"
ART="_bmad-output/implementation-artifacts"
LOG="$ART/20-8-regression-$LABEL.log"
SUM="$ART/20-8-regression-$LABEL.summary"
HUNGF="$ART/20-8-regression-$LABEL.hung"
TMP="$ART/.20-8-guard-tmp-$LABEL.log"
DIR_TIMEOUT="${DIR_TIMEOUT:-900}"
FILE_TIMEOUT="${FILE_TIMEOUT:-300}"

: > "$LOG"
: > "$HUNGF"

echo "=== Story 20.8 regression run: $LABEL ===" >> "$LOG"
echo "started $(date -Is)  dir timeout ${DIR_TIMEOUT}s  file timeout ${FILE_TIMEOUT}s" >> "$LOG"

run_files_in () {   # $1 = directory
    for f in "$1"/test_*.py; do
        [ -e "$f" ] || continue
        echo "--- file: $f ---" >> "$LOG"
        timeout "$FILE_TIMEOUT" $PY -u -m pytest "$f" -p no:randomly -q -rf \
            >> "$LOG" 2>&1
        if [ $? -eq 124 ]; then
            echo "$f" >> "$HUNGF"
            echo "HUNG after ${FILE_TIMEOUT}s" >> "$LOG"
            printf 'HUNG  %s\n' "$f"
        fi
    done
}

# Every directory that directly contains test files, deepest first is
# irrelevant — each is run with --ignore of nothing and non-recursively via an
# explicit file list, so no test is counted twice.
DIRS=$(find tests -name "test_*.py" | sed 's|/[^/]*$||' | sort -u)

for d in $DIRS; do
    files=$(ls "$d"/test_*.py 2>/dev/null)
    [ -n "$files" ] || continue
    echo "--- dir: $d ---" >> "$LOG"
    : > "$TMP"
    # shellcheck disable=SC2086
    timeout "$DIR_TIMEOUT" $PY -u -m pytest $files -p no:randomly -q -rf \
        > "$TMP" 2>&1
    if [ $? -eq 124 ]; then
        echo "DIR TIMED OUT after ${DIR_TIMEOUT}s — partial output DISCARDED, re-running file by file" >> "$LOG"
        printf 'dir timeout, splitting: %s\n' "$d"
        run_files_in "$d"
    else
        cat "$TMP" >> "$LOG"
    fi
done
rm -f "$TMP"

{
    echo "=== $LABEL ==="
    echo "PASSED      $(grep -oE '[0-9]+ passed' "$LOG" | awk '{s+=$1} END {print s+0}')"
    echo "FAILED      $(grep -oE '[0-9]+ failed' "$LOG" | awk '{s+=$1} END {print s+0}')"
    echo "ERRORS      $(grep -oE '[0-9]+ error' "$LOG" | awk '{s+=$1} END {print s+0}')"
    echo "HUNG FILES  $(wc -l < "$HUNGF" | tr -d ' ')"
    echo
    echo "--- HUNG files ---"
    cat "$HUNGF"
    echo
    echo "--- FAILED node ids (sorted, for the identity diff) ---"
    grep -E '^FAILED ' "$LOG" | sed 's/ - .*//' | sort -u
} > "$SUM"

cat "$SUM"
echo "finished $(date -Is)" >> "$LOG"
