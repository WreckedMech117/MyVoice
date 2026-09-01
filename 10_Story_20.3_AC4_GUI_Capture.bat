@echo off
REM ========================================
REM Story 20.3 AC #4 - GUI first-generation TTFA capture
REM ========================================
REM Confirms through the REAL app that compile priming now runs and that
REM the user's first generation is fast. Stories 20.1/20.2 measured this
REM with a headless harness; the GUI path was dead until Story 20.3.
REM
REM This launches MyVoice 5 times in a row, one CSV per launch.
REM
REM WHAT TO DO IN EACH LAUNCH:
REM   1. Make sure a CLONED voice is the active profile so BASE is the
REM      resident model. This is the common user path Story 20.3 targets.
REM   2. WAIT for the "Preparing TTS engine" indicator to disappear.
REM      THIS MATTERS: priming holds the request semaphore, so generating
REM      while it is up measures queueing, not first-forward. The number
REM      would look like a regression and would not be one.
REM   3. Generate ONE utterance. Same text every launch.
REM   4. Close the app cleanly - window close, NOT Ctrl-C, and not while
REM      it is still generating.
REM   5. The next launch starts automatically.
REM
REM Note: cmd.exe parses a literal close-paren inside a for /L body as the
REM end of the block - that bug silently cost Story 18.4 a whole run. This
REM script uses a goto loop and paren-free echoes to stay clear of it.
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    echo Expected: %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\src\myvoice\main.py" (
    echo [ERROR] MyVoice application files not found.
    echo Expected: %SCRIPT_DIR%\src\myvoice\main.py
    pause
    exit /b 1
)

"%SCRIPT_DIR%\python310\python.exe" -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [WARNING] Dependencies not installed.
    echo Please run 00_Install_Dependencies.bat first.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "OUTDIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "TOTAL_RUNS=5"

echo.
echo ========================================
echo Story 20.3 AC #4 - GUI capture, %TOTAL_RUNS% launches
echo ========================================
echo.
echo Per launch:
echo   - active profile must be a CLONED voice, so BASE is resident
echo   - WAIT for "Preparing TTS engine" to clear before generating
echo   - generate ONE utterance, same text each time
echo   - close the app cleanly
echo.
echo Output CSVs:
echo   %OUTDIR%\20-3-gui-r01.csv ... r0%TOTAL_RUNS%.csv
echo.
echo Press a key to begin.
pause >nul

set /a RUN=1

:RUN_LOOP
set "CSV=%OUTDIR%\20-3-gui-r0!RUN!.csv"
set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=!CSV!"

echo.
echo ========================================
echo LAUNCH !RUN! of %TOTAL_RUNS%
echo ========================================
echo CSV: !CSV!
echo.
echo Reminder: wait for "Preparing TTS engine" to clear, generate once,
echo then close the app cleanly.
echo.

REM Run main.py directly - NOT as a module - to preserve the
REM torch-before-PyQt6 DLL-ordering invariant.
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] Launch !RUN! exited with code !EXIT_CODE! - check logs\myvoice.log

if exist "!CSV!" (
    for /f %%c in ('find /c /v "" ^< "!CSV!"') do echo Launch !RUN! captured %%c lines including header
) else (
    echo [WARNING] No CSV written for launch !RUN!.
    echo The env-var may not have taken effect, or no generation was run.
)

set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto RUN_LOOP

echo.
echo ========================================
echo All %TOTAL_RUNS% launches complete
echo ========================================
echo.
echo Row counts:
set /a RUN=1
:COUNT_LOOP
set "CSV=%OUTDIR%\20-3-gui-r0!RUN!.csv"
if exist "!CSV!" (
    for /f %%c in ('find /c /v "" ^< "!CSV!"') do echo   r0!RUN!: %%c lines
) else (
    echo   r0!RUN!: MISSING
)
set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto COUNT_LOOP

echo.
echo Priming telemetry from the most recent log - expect primed_warm,
echo and NOT no_model_loaded:
echo.
findstr /C:"tts_compile_warmup_priming" /C:"Compile cache hit" /C:"warmup primed" /C:"warmup skipped" "%SCRIPT_DIR%\logs\myvoice.log" 2>nul | more
echo.
echo If the lines above say no_model_loaded, Story 20.3's ordering fix
echo did not take effect and the numbers below are the OLD behavior.
echo.
echo Hand these CSVs back to the session to populate evidence section 4.
echo.

pause
exit /b 0
