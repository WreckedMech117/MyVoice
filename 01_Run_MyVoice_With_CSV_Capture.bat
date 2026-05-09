@echo off
REM ========================================
REM Story 18.1 Task 1.4 — Underrun-Gap Mitigation Measurement Run
REM ========================================
REM Launches MyVoice with progressive-playback metric CSV capture
REM enabled. The capture is fully automated:
REM   1. CSV header writes immediately on launch (you can verify the
REM      file exists with just the header before generating).
REM   2. Three metrics emit per chunk during TRUE_STREAM playback:
REM        - progressive_chunk_emit_ms        (producer / qwen_tts_service)
REM        - progressive_chunk_playback_arrival_ms  (consumer / app.py)
REM        - progressive_chunk_audio_duration_ms    (consumer / app.py)
REM   3. Closing the app cleanly (X button or File->Quit) flushes +
REM      closes the CSV file via _on_about_to_quit.
REM
REM Procedure:
REM   1. Run this .bat.
REM   2. Confirm the CSV file shows up at the OUTPUT path below with
REM      the header row only (proves env-var took effect).
REM   3. Pick Sarira-F as the speaker.
REM   4. Generate the canonical Story 17.3 Section 4.1 step 3 long-form
REM      paragraph (>=250 chars). Audition in real-time and confirm
REM      the ~1-second silent gaps are present (anchors the data to
REM      the same defect class).
REM   5. Close the app cleanly (NOT Ctrl-C, NOT taskbar close-while-
REM      generating). Wait for "Application shutting down" in the log.
REM   6. Verify the CSV is non-trivially populated (~3 rows per chunk
REM      times 10-30 chunks for a long-form utterance).
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
REM Story 18.1 Task 1.4: engage CSV capture
REM ========================================
REM Output path is pinned to the story-spec'd location verbatim so the
REM evidence file references the same artifact path the .bat writes.
REM If you'd rather use the default logs-dir path, replace the value
REM below with literal "1" — the capture module then writes to
REM <logs_dir>/18-1-instrumentation-rtx5090-longform.csv automatically.

set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-1-instrumentation-rtx5090-longform.csv"

REM Make sure the parent directory exists (the capture module also
REM does this defensively, but a clean pre-create surfaces permission
REM issues here at launch instead of inside the listener).
if not exist "%SCRIPT_DIR%\_bmad-output\implementation-artifacts" (
    mkdir "%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
)

echo.
echo ========================================
echo MyVoice V2 - Story 18.1 Task 1.4 Run
echo ========================================
echo CSV target:
echo   %MYVOICE_PROGRESSIVE_PLAYBACK_CSV%
echo.
echo After launch, confirm the CSV exists with just the header row
echo before you generate. Then run the canonical Sarira-F long-form
echo utterance and close the app cleanly.
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
        echo with Task 1.5 (mitigation choice) + Task 2 OR Task 3.
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
