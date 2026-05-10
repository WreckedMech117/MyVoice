@echo off
REM ========================================
REM Story 18.3 Task 7.2 — NFR1 FP32 override branch capture (N=10)
REM ========================================
REM Programmatically sets AppSettings.tts_precision="fp32" (NFR7 override
REM path — forces fp32 even on Ampere+ RTX 5090), then launches MyVoice.
REM
REM This is the comparison branch for 03_Story_18.3_NFR1_BF16.bat. The
REM fp32 branch composes ON TOP OF Story 18.2's TF32+cuDNN-engaged baseline
REM (i.e., the fp32 here is fp32-with-TF32-engaged, not strict-fp32) per
REM Story 18.3 AC #10.
REM
REM IMPORTANT — fresh-process discipline (per Story 18.3 AC #9):
REM   The story spec requires N=10 fresh-process launches per branch. Each
REM   iteration launches MyVoice; you generate ONE Sarira-F long-form
REM   utterance (Story 17.3 §4.1 step 3 canonical paragraph, >=250 chars);
REM   close MyVoice cleanly. The loop iterates 10 times.
REM
REM Each iteration's CSV writes to a uniquely-numbered path:
REM   _bmad-output/implementation-artifacts/18-3-rtx5090-fp32-run01.csv ... run10.csv
REM
REM IMPORTANT — restore tts_precision="auto" after the run:
REM   This bat leaves settings.json with tts_precision="fp32". When you're
REM   done with the measurement, run:
REM     python310\python.exe _bmad-output\implementation-artifacts\18-3-set-precision.py auto
REM   to restore the default. (The bf16 bat does this automatically as
REM   step 1 if you run it again.)
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found!
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

REM Story 18.3 measurement-mode bypass: clicking the window X button performs
REM an immediate clean close (no tray-minimize, no confirm dialog) so the
REM 10-launch loop advances without manual right-click-tray-Exit per iter.
REM Production close behavior is unchanged when this env var is unset.
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

REM Step 1 — set tts_precision=fp32 (NFR7 override path).
echo.
echo ========================================
echo Step 1: setting AppSettings.tts_precision = "fp32" (NFR7 override)
echo ========================================
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-set-precision.py" fp32
if errorlevel 1 (
    echo [ERROR] Failed to update settings.json
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\_bmad-output\implementation-artifacts" (
    mkdir "%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
)

REM Step 2 — N=10 fresh-process launch loop.
echo.
echo ========================================
echo Step 2: starting N=10 fresh-process measurement loop
echo ========================================
echo.
echo For each iteration:
echo   1. Wait for MyVoice to finish loading.
echo   2. Look for the Story 18.3 INFO line in logs:
echo      "ModelRegistry initialized: ... dtype=torch.float32, precision_source='app_settings_override'..."
echo   3. Pick Sarira-F as the speaker.
echo   4. Generate the canonical Sarira-F long-form paragraph
echo      (Story 17.3 §4.1 step 3; ^>=250 chars / ~22s of speech) —
echo      use the EXACT SAME utterance as the bf16 branch so the A/B
echo      methodology is clean.
echo   5. Close MyVoice cleanly. The loop continues to the next run.
echo.

for /L %%I in (1,1,10) do (
    set "RUN_NUM=0%%I"
    set "RUN_NUM=!RUN_NUM:~-2!"
    set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-rtx5090-fp32-run!RUN_NUM!.csv"
    echo.
    echo ===== Run %%I of 10 =====
    echo CSV: !MYVOICE_PROGRESSIVE_PLAYBACK_CSV!
    echo.
    "%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
    if errorlevel 1 (
        echo [WARN] Run %%I exited with non-zero code; continuing.
    )
)

echo.
echo ========================================
echo FP32 capture loop complete.
echo ========================================
echo.
echo Captured CSVs at:
dir /B "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-rtx5090-fp32-run*.csv" 2>nul
echo.
echo IMPORTANT — restore tts_precision="auto" before normal MyVoice use:
echo   python310\python.exe _bmad-output\implementation-artifacts\18-3-set-precision.py auto
echo.
echo Then report back to the dev-story workflow with both N=10 captures.
echo.

pause
exit /b 0
