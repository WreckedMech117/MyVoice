@echo off
REM ========================================
REM Story 20.4 AC #5 - NFR3 chunk-boundary audition (Commander solo)
REM ========================================
REM ROUND 2. Round 1 FAILED - m-020 was clean at chunk_size=25 and carried
REM tonal_distortion at chunk_size=10, and l-020/l-021 carried seam defects
REM on BOTH arms. The follow-up analysis found the cause was NOT the
REM consumer crossfade: every decoder chunk boundary was deleting 15-19 ms
REM of real speech (a splice-alignment bug, present at chunk_size=25 too),
REM and the two independent decodes either side of a boundary differ by
REM about 35 percent. Both are now fixed in streaming_decoder.py.
REM
REM This round asks whether the fix worked.
REM   reference arm = chunk_size 25, shipped pre-fix stitching. These are
REM                   round 1's EXACT files, so your round-1 calls on them
REM                   act as a calibration anchor.
REM   candidate arm = chunk_size 10 WITH the seam fix.
REM
REM The 14-WAV round-3 fixture is generated ahead of time by
REM   20-4-regen-audition-fixture-r4.py
REM under _bmad-output, which drives the production TRUE_STREAM path
REM (streamer, streaming-decoder overlap-add, StreamingChunkBuffer
REM crossfade) on the CLONED Sarira-F voice. If the fixture is missing this
REM script says so and stops - it does NOT silently regenerate, because
REM that takes GPU time.
REM
REM Round 1's fixture, truth table and results CSV are preserved untouched.
REM
REM ~10 minutes. Seven utterances: three short, two medium, two long.
REM Trials are blinded; the helper unblinds and prints the verdict at the
REM end.
REM
REM Usage:
REM   12_Story_20.4_AC5_Audition.bat        (L1 - Commander, round 3)
REM   12_Story_20.4_AC5_Audition.bat L2     (a second listener, same machine)
REM   12_Story_20.4_AC5_Audition.bat L1 r1  (re-run round 1)
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

set "LISTENER_ID=%~1"
if "%LISTENER_ID%"=="" set "LISTENER_ID=L1"
set "ROUND_ID=%~2"
if "%ROUND_ID%"=="" set "ROUND_ID=r4"

set "ARTIFACTS=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "FIXTURE=%ARTIFACTS%\20-4-perceptual-fixtures-r4"
if "%ROUND_ID%"=="r1" set "FIXTURE=%ARTIFACTS%\20-4-perceptual-fixtures"
if "%ROUND_ID%"=="r2" set "FIXTURE=%ARTIFACTS%\20-4-perceptual-fixtures-r2"
if "%ROUND_ID%"=="r3" set "FIXTURE=%ARTIFACTS%\20-4-perceptual-fixtures-r3"

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    echo Expected: %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%FIXTURE%\_perlistener_truthtable.json" (
    echo [ERROR] Audition fixture not found for round %ROUND_ID%.
    echo Expected: %FIXTURE%\_perlistener_truthtable.json
    echo.
    echo Generate it first - this needs the GPU and takes a few minutes:
    echo   python310\python.exe %ARTIFACTS%\20-4-regen-audition-fixture-r4.py
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

echo.
echo ========================================
echo Story 20.4 AC #5 audition - listener %LISTENER_ID%, round %ROUND_ID%
echo ========================================
echo.
echo Use headphones. Normal Discord-call volume.
echo You are judging SEAMS, not voice quality - the two takes are
echo different samples, so wording rhythm will differ. Listen for clicks,
echo discontinuities, and prosody that resets mid-phrase.
echo.
echo Round 2 note: one arm of every pair is round 1's exact cs25 file.
echo If l-020 and l-021 draw the same calls you gave them last time, the
echo session is internally consistent - that is why they were reused.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%ARTIFACTS%\20-4-l1-audition-helper.py" %LISTENER_ID% %ROUND_ID%
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] helper exited with code !EXIT_CODE!
echo.
echo Results CSV:
echo   %ARTIFACTS%\20-4-chunk-retune-audition-%ROUND_ID%.csv
echo   round 1's remains at 20-4-chunk-retune-audition.csv, untouched
echo.
echo Hand the verdict block above back to the session so it lands in the
echo Story 20.4 evidence file section 5.
echo.

pause
exit /b 0
