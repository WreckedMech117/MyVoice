@echo off
REM ========================================
REM Story 20.6 AC #4 - NFR3 lookahead-retirement audition (Commander solo)
REM ========================================
REM   reference arm = what ships today: codec state caching + the gated
REM                   (0-sample) consumer crossfade + the 5-frame lookahead
REM   candidate arm = the same, with the lookahead RETIRED
REM
REM ONE VARIABLE. Both arms carry codec state caching, chunk_size 25 and
REM the same gated consumer crossfade. The Story 20.4 seam blend is a
REM DEPENDENT of the lookahead, not a second arm - it cross-fades the
REM retained lookahead tail into the next chunk's head, and with no
REM lookahead there is no tail.
REM
REM BOTH FILES IN EVERY PAIR COME FROM ONE TALKER RUN. The chunking change
REM is downstream of generation, so the fixture captures one run per pair
REM and re-slices the recovered frame sequence per arm. Wording, prosody
REM and duration are identical to the sample; there is no take-to-take
REM variance to average over within a pair.
REM
REM 16 trials: the seven epic-standard utterances x two takes, plus a
REM zero-seam control (ctl-020) whose two arms are BYTE-IDENTICAL. The
REM control is not a trick - it sets the round's noise floor.
REM
REM THE EXPECTED ANSWER IS "equivalent" ON MOST TRIALS. Offline the two
REM arms measure about -70 dB apart with identical length on every pair.
REM What this round asks is whether that measurement is right about what
REM the ear does; twice in this epic an offline metric was not.
REM
REM BLOCKING: any chunk-boundary defect on a candidate trial that its
REM paired reference does not also carry. If it lands at an INTERIOR seam
REM that is prediction P4 - the diagnosis is wrong, and the retirement gets
REM REVERTED rather than tuned.
REM
REM The 32-WAV fixture is generated ahead of time by
REM   20-6-regen-audition-fixture.py
REM under _bmad-output. If it is missing this script says so and stops - it
REM does NOT silently regenerate, because that takes GPU time.
REM
REM ~20 minutes.
REM
REM Usage:
REM   14_Story_20.6_AC4_Audition.bat        (L1 - Commander)
REM   14_Story_20.6_AC4_Audition.bat L2     (a second listener, same machine)
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

set "LISTENER_ID=%~1"
if "%LISTENER_ID%"=="" set "LISTENER_ID=L1"
set "ROUND_ID=%~2"
if "%ROUND_ID%"=="" set "ROUND_ID=r1"

set "ARTIFACTS=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "FIXTURE=%ARTIFACTS%\20-6-perceptual-fixtures"

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    echo Expected: %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%FIXTURE%\_perlistener_truthtable.json" (
    echo [ERROR] Audition fixture not found.
    echo Expected: %FIXTURE%\_perlistener_truthtable.json
    echo.
    echo Generate it first - this needs the GPU and takes a few minutes:
    echo   python310\python.exe %ARTIFACTS%\20-6-regen-audition-fixture.py
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

echo.
echo ========================================
echo Story 20.6 AC #4 audition - listener %LISTENER_ID%
echo ========================================
echo.
echo Use headphones. Normal Discord-call volume.
echo.
echo Both files in a pair are the SAME generation decoded two ways. The
echo words, the timing and the delivery are identical. You are judging
echo SEAMS only - clicks, discontinuities, prosody that resets mid-phrase,
echo smeared consonants at a boundary.
echo.
echo "equivalent" is the PREDICTED answer here, not a cop-out. Two of the
echo sixteen trials are byte-identical on purpose.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%ARTIFACTS%\20-6-l1-audition-helper.py" %LISTENER_ID% %ROUND_ID%
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] helper exited with code !EXIT_CODE!
echo.
echo Results CSV:
echo   %ARTIFACTS%\20-6-lookahead-audition.csv
echo.
echo Hand the verdict block above back to the session so it lands in the
echo Story 20.6 evidence file section 6.
echo.

pause
exit /b 0
