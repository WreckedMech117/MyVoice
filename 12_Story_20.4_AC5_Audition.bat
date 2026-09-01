@echo off
REM ========================================
REM Story 20.4 AC #5 - NFR3 chunk-boundary audition (Commander solo)
REM ========================================
REM The retune moves the streamer's chunk boundary from 30 frames to 15,
REM and the streaming decoder trims a lookahead-sized tail per chunk, so
REM chunk-stitching behaviour changes for EVERY generation. AC #5 makes an
REM audible chunk-boundary artefact a BLOCKING finding.
REM
REM The 14-WAV fixture is generated ahead of time by
REM   _bmad-output\implementation-artifacts\20-4-regen-audition-fixture.py
REM which drives the production TRUE_STREAM path (streamer -> streaming
REM decoder overlap-add -> StreamingChunkBuffer crossfade) at both
REM geometries on the CLONED Sarira-F voice. If the fixture is missing this
REM script says so and stops - it does NOT silently regenerate, because
REM regeneration takes several minutes of GPU time.
REM
REM ~10 minutes. Seven utterances: three short, two medium, two long.
REM Trials are blinded; the helper unblinds and prints the verdict at the
REM end.
REM
REM Usage:
REM   12_Story_20.4_AC5_Audition.bat        (defaults to L1 - Commander)
REM   12_Story_20.4_AC5_Audition.bat L2     (a second listener, same machine)
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

set "LISTENER_ID=%~1"
if "%LISTENER_ID%"=="" set "LISTENER_ID=L1"

set "ARTIFACTS=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "FIXTURE=%ARTIFACTS%\20-4-perceptual-fixtures"

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
    echo   python310\python.exe %ARTIFACTS%\20-4-regen-audition-fixture.py
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

echo.
echo ========================================
echo Story 20.4 AC #5 audition - listener %LISTENER_ID%
echo ========================================
echo.
echo Use headphones. Normal Discord-call volume.
echo You are judging SEAMS, not voice quality - the two takes are
echo different samples, so wording rhythm will differ. Listen for clicks,
echo discontinuities, and prosody that resets mid-phrase.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%ARTIFACTS%\20-4-l1-audition-helper.py" %LISTENER_ID%
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] helper exited with code !EXIT_CODE!
echo.
echo Results CSV:
echo   %ARTIFACTS%\20-4-chunk-retune-audition.csv
echo.
echo Hand the verdict block above back to the session so it lands in the
echo Story 20.4 evidence file section 5.
echo.

pause
exit /b 0
