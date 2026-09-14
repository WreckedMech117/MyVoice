#!/usr/bin/env bash
# Story tooling-3 Task 3 — run `pytest tests/` to completion under pytest.ini's
# timeout, enumerating every hang.
#
# pytest-timeout's thread method (the only one that works on Windows) ends the
# pytest process with os._exit(1) when a test exceeds its budget, after naming
# the test (-v puts its id on the "+++ Timeout +++" line) and dumping every
# thread's stack. One run therefore surfaces ONE hang. This driver loops:
# run, harvest the hung id, --deselect it, run again — until a run ends
# normally. Every iteration's full console output is kept; per-test durations
# stream to JSONL so nothing completed before an abort is lost.
#
# Usage: bash suite-until-complete.sh <label> [extra pytest args]
set -u
LABEL="${1:?usage: suite-until-complete.sh <label>}"; shift
cd "$(dirname "$0")/../../.." || exit 1
PY="python310/python.exe"
HERE="_bmad-output/implementation-artifacts/tooling-3"
HUNG="$HERE/suite-$LABEL.hung"
MAXITER="${MAXITER:-80}"
# Resume: an existing <label>.hung seeds the deselect list so a stopped loop
# continues from where it left off instead of re-discovering the same hangs.
DESELECT=()
START=1
if [ -s "$HUNG" ]; then
    while read -r _ id _; do DESELECT+=("--deselect=$id"); done < <(grep '^TIMEOUT: ' "$HUNG")
    START=$(( $(grep -c '^TIMEOUT: ' "$HUNG") + 1 ))
    echo "resuming: ${#DESELECT[@]} hangs already recorded, starting at iteration $START" | tee -a "$HUNG"
else
    : > "$HUNG"
fi
for i in $(seq "$START" "$MAXITER"); do
    LOG="$HERE/suite-$LABEL-iter$i.log"
    export TOOLING3_DURATIONS="$HERE/suite-$LABEL-iter$i.jsonl"
    : > "$TOOLING3_DURATIONS"
    echo "=== iteration $i start $(date -Is) deselected=${#DESELECT[@]} ===" | tee "$LOG"
    t0=$(date +%s)
    $PY -u "$HERE/run_pytest.py" tests/ -v -rfE "${DESELECT[@]}" "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== iteration $i end $(date -Is) rc=$rc wall=$(( $(date +%s) - t0 ))s ===" | tee -a "$LOG"
    hung=$(grep -E '^tests/.* [+]{3,} Timeout [+]{3,}' "$LOG" | tail -1 | sed -E 's/ +[+]{3,} Timeout [+]{3,}.*$//')
    if [ -n "$hung" ]; then
        echo "TIMEOUT: $hung (iteration $i)" | tee -a "$HUNG"
        DESELECT+=("--deselect=$hung")
        continue
    fi
    if ! grep -qE '^=+ .*(passed|failed|error).* in [0-9.]+s' "$LOG"; then
        echo "iteration $i ended without a Timeout block AND without a pytest summary (rc=$rc) -- aborted externally? not counting it" | tee -a "$HUNG"
        exit 3
    fi
    echo "completed normally on iteration $i" | tee -a "$HUNG"
    exit 0
done
echo "gave up after $MAXITER iterations" | tee -a "$HUNG"
exit 2
