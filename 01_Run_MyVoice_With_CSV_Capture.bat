@echo off
REM ========================================
REM Story 18.2 Task 4.2 + 4.5 — TF32 + cuDNN Benchmark Single-Shot Capture
REM ========================================
REM Story 18.1 baseline preserved at:
REM   _bmad-output\implementation-artifacts\18-1-instrumentation-rtx5090-longform.csv
REM
REM This run writes to a SEPARATE Story 18.2 path so the 18.1 baseline
REM is intact and can serve as the implicit TF32-OFF reference for the
REM producer-side metrics (progressive_chunk_emit_ms inter-arrival
REM cadence — the canonical 18.1 verdict was 3.23x steady-state ratio /
REM 31% real-time talker decode).
REM
REM What's NEW vs Story 18.1's run:
REM   - Story 18.2 Task 4.1 widened the CSV-capture filter to include
REM     `first_chunk_latency_ms` — so this CSV will carry one extra
REM     metric_name not present in 18.1's CSV (single-sample first-chunk
REM     latency for the engaged TF32 path).
REM   - Story 18.2 wired `enable_tf32_and_cudnn_benchmark()` into main()
REM     so this run engages TF32 matmul + cuDNN benchmark autotune at
REM     startup. The single INFO breadcrumb
REM         "TF32 + cuDNN benchmark enabled (device_capability=10.0)"
REM     should appear exactly once in logs\myvoice.log.
REM
REM Procedure (Commander):
REM   1. CLEAR logs first so the new run's lines stand alone:
REM        del logs\myvoice.log
REM   2. Run this .bat.
REM   3. Confirm the CSV file shows up at the OUTPUT path below with
REM      the header row only (proves env-var took effect).
REM   4. Pick Sarira-F as the speaker.
REM   5. Generate the canonical Story 17.3 Section 4.1 step 3 long-form
REM      paragraph (>=250 chars, ~22s of speech). Audition normally.
REM   6. Close the app cleanly (NOT Ctrl-C, NOT taskbar close-while-
REM      generating). Wait for "Application shutting down" in the log.
REM   7. Verify the CSV is non-trivially populated. Then come back to
REM      the dev-story workflow — the agent reads both CSVs + the log
REM      and populates evidence file Sec 4.2 + Sec 4.5.
REM ========================================

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Enable delayed expansion for variables
setlocal EnableDelayedExpansion

REM Check if portable Python exists
if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found!
    echo.
    echo Please ensure the python310 folder is present.
    echo.
    pause
    exit /b 1
)

REM Check if source directory exists
if not exist "%SCRIPT_DIR%\src\myvoice\main.py" (
    echo [ERROR] MyVoice application files not found!
    echo Expected: %SCRIPT_DIR%\src\myvoice\main.py
    echo.
    pause
    exit /b 1
)

REM Check if dependencies are installed
"%SCRIPT_DIR%\python310\python.exe" -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [WARNING] Dependencies not installed!
    echo.
    echo Please run "00_Install_Dependencies.bat" first.
    echo.
    pause
    exit /b 1
)

REM Change to script directory to ensure relative paths work
cd /d "%SCRIPT_DIR%"

REM Add src directory to Python path
set "PYTHONPATH=%SCRIPT_DIR%\src"

REM ========================================
REM Story 18.2 Task 4.2: engage CSV capture (TF32 ON branch)
REM ========================================
REM Output path is pinned to the Story 18.2 evidence-file location so
REM the 18.1 baseline at 18-1-instrumentation-rtx5090-longform.csv stays
REM untouched and the dev agent can compute the producer-emit-interval
REM delta (Story 18.1 = TF32-off implicit baseline; this CSV = TF32 on).

set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-2-rtx5090-tf32-on.csv"

REM Make sure the parent directory exists (the capture module also
REM does this defensively, but a clean pre-create surfaces permission
REM issues here at launch instead of inside the listener).
if not exist "%SCRIPT_DIR%\_bmad-output\implementation-artifacts" (
    mkdir "%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
)

echo.
echo ========================================
echo MyVoice V2 - Story 18.2 Task 4.2 + 4.5 Run
echo ========================================
echo CSV target:
echo   %MYVOICE_PROGRESSIVE_PLAYBACK_CSV%
echo.
echo TF32 wire-up should fire at startup. After launch, look for the
echo INFO breadcrumb in logs\myvoice.log:
echo     "TF32 + cuDNN benchmark enabled (device_capability=10.0)"
echo.
echo If you have NOT cleared logs\myvoice.log first, do so before
echo running this BAT so the breadcrumb is unambiguously from this run.
echo.
echo Then run the canonical Sarira-F long-form utterance and close
echo the app cleanly.
echo ========================================
echo.

REM Run main.py directly (NOT as module) to preserve the torch-before-PyQt6
REM DLL-ordering invariant captured in memory/torch_pyqt6_dll_ordering.md.
REM Mirrors 01_Run_MyVoice.bat:84 verbatim except for the env-var above.
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"

REM Capture exit code
set "EXIT_CODE=%errorlevel%"

echo.
echo ========================================
echo MyVoice has closed
echo ========================================
echo.

if !EXIT_CODE! neq 0 (
    echo [ERROR] MyVoice exited with error code: !EXIT_CODE!
    echo Check the logs folder for details.
    echo.
) else (
    echo MyVoice closed normally.
    echo.
    if exist "%MYVOICE_PROGRESSIVE_PLAYBACK_CSV%" (
        echo CSV captured at:
        echo   %MYVOICE_PROGRESSIVE_PLAYBACK_CSV%
        echo.
        echo Row count:
        for /f %%c in ('find /c /v "" ^< "%MYVOICE_PROGRESSIVE_PLAYBACK_CSV%"') do echo   %%c lines (incl. header)
        echo.
        echo Report this back to the dev-story workflow to proceed
        echo with Story 18.2 Task 4.4 delta computation + Task 5/6.
    ) else (
        echo [WARNING] CSV file was NOT created at the expected path.
        echo Either the env-var did not take effect, or the listener
        echo could not open the file. Check the myvoice.log for
        echo "progressive-playback CSV capture enabled" or the
        echo "Failed to enable progressive-playback CSV capture" line.
    )
)

REM Pause so the row-count + status remain visible after the app closes.
pause
exit /b 0
