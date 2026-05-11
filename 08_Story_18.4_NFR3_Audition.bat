@echo off
REM ========================================
REM Story 18.4 Task 9.4 - NFR3 Joint Audition Session
REM ========================================
REM Drives the per-listener audition helper that walks you through 10
REM paired A/B trials from the Story 18.4 perceptual fixture. Single
REM audition certifies three dimensions simultaneously:
REM   - bf16 vs fp32             (closes Story 18.3 OQ #3 deferred audition)
REM   - compile-engaged vs eager (closes Story 18.4 primary perceptual gate)
REM   - new pin vs old pin       (closes D-22 Branch B re-audition obligation)
REM
REM Usage:
REM   08_Story_18.4_NFR3_Audition.bat          (defaults to L1 - Commander)
REM   08_Story_18.4_NFR3_Audition.bat L2       (organic listener 2)
REM   08_Story_18.4_NFR3_Audition.bat L3       (organic listener 3)
REM
REM What you'll do (~10 min per listener):
REM   1. Helper plays "trial A" end-to-end, then "trial B" end-to-end.
REM   2. You can press [r] to replay both, [Enter] to continue to entry.
REM   3. Pick from the defect vocabulary for each trial:
REM        none / audible_seam / clipping / phase_artifact /
REM        tonal_distortion / other_describe_in_notes
REM   4. Pick a_or_b_preferred: A / B / equivalent
REM   5. Optional free-text notes.
REM   6. Loop repeats for all 10 utterances (s-014 through l-014).
REM
REM Verdict gate (verbatim Story 17.1):
REM   PASS iff zero listeners flag `audible_seam` on any B (= bf16+compile)
REM   trial across all 30 trials.
REM
REM Blinding: the helper intentionally does NOT print the underlying WAV
REM filename. The per-listener truth-table at
REM _bmad-output/implementation-artifacts/18-4-perceptual-fixtures/_perlistener_truthtable.json
REM randomizes which actual mode (fp32_eager or bf16_compile) lands as
REM trial A vs trial B per listener.
REM
REM Output CSV (per-listener rows accumulate):
REM   _bmad-output/implementation-artifacts/18-4-bf16-compile-pinbump-audition.csv
REM
REM Re-running is safe - rows already present for the same listener are
REM skipped automatically. To restart fresh, delete the existing rows
REM manually first.
REM ========================================

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Enable delayed expansion for variables
setlocal EnableDelayedExpansion

REM Listener id - default L1 (Commander), override via first CLI arg.
set "LISTENER_ID=%~1"
if "%LISTENER_ID%"=="" set "LISTENER_ID=L1"

REM Check if portable Python exists
if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found!
    echo.
    echo Please ensure the python310 folder is present.
    echo.
    pause
    exit /b 1
)

REM Check if the audition helper exists
set "HELPER=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-l1-audition-helper.py"
if not exist "%HELPER%" (
    echo [ERROR] Audition helper not found at:
    echo   %HELPER%
    echo.
    pause
    exit /b 1
)

REM Check that the fixture directory + truth-table exist (the helper will
REM also check, but a clean pre-check surfaces the missing-fixture error
REM before the helper prints its session banner).
set "FIXTURE_DIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-perceptual-fixtures"
set "TRUTH_TABLE=%FIXTURE_DIR%\_perlistener_truthtable.json"
if not exist "%TRUTH_TABLE%" (
    echo [ERROR] Truth-table not found at:
    echo   %TRUTH_TABLE%
    echo.
    echo Did you regenerate the fixture first?
    echo   python310\python.exe _bmad-output\implementation-artifacts\18-4-regen-fixture.py both
    echo   python310\python.exe _bmad-output\implementation-artifacts\18-4-generate-truthtable.py
    echo.
    pause
    exit /b 1
)

REM Change to script directory to ensure relative paths work
cd /d "%SCRIPT_DIR%"

echo.
echo ========================================
echo MyVoice V2 - Story 18.4 NFR3 Joint Audition
echo ========================================
echo Listener id: %LISTENER_ID%
echo Fixture dir: %FIXTURE_DIR%
echo.
echo Use headphones if you have them. Comfortable Discord-call volume;
echo do not crank past your normal listening level.
echo.
echo The helper plays trial A and trial B for each of 10 utterances
echo (s-014, s-015, s-016, s-017, m-011, m-012, m-013, m-014, l-013, l-014).
echo Pick exactly ONE defect from the fixed vocabulary per trial. The
echo `audible_seam` flag is the verdict-gate marker - flag it if you hear
echo chunk-boundary discontinuity in either trial. Other defects are
echo observational.
echo.
echo Press [r] to replay both trials before recording your answer.
echo Press [Enter] at the playback prompt to continue to defect entry.
echo ========================================
echo.

REM Run the helper. It accepts the listener id as a positional CLI arg.
"%SCRIPT_DIR%\python310\python.exe" "%HELPER%" %LISTENER_ID%

REM Capture exit code
set "EXIT_CODE=%errorlevel%"

echo.
echo ========================================
echo Audition session for %LISTENER_ID% closed
echo ========================================
echo.

set "CANONICAL_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-bf16-compile-pinbump-audition.csv"

if !EXIT_CODE! neq 0 (
    echo [WARNING] Helper exited with code: !EXIT_CODE!
    echo If you aborted partway through, rows already entered are saved.
    echo.
) else (
    echo Session completed cleanly.
    echo.
)

if exist "%CANONICAL_CSV%" (
    echo Audition CSV at:
    echo   %CANONICAL_CSV%
    echo.
    echo Row count:
    for /f %%c in ('find /c /v "" ^< "%CANONICAL_CSV%"') do echo   %%c lines incl. header
    echo.
    echo Report back to the dev-story workflow for verdict computation.
    echo Per Story 17.1 verbatim: PASS iff zero `audible_seam` flags on B
    echo (= bf16+compile) trials across all listeners.
) else (
    echo [INFO] No audition CSV produced yet - session aborted before any
    echo row was recorded.
)

REM Pause so the row-count + status remain visible after the helper closes.
pause
exit /b 0
