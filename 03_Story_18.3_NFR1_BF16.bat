@echo off
REM ========================================
REM Story 18.3 Task 7.1 — NFR1 BF16 branch capture (N=10)
REM ========================================
REM Programmatically sets AppSettings.tts_precision="auto" (bf16 on RTX 5090
REM per the new resolve_tts_precision contract), then launches MyVoice.
REM
REM IMPORTANT — fresh-process discipline (per Story 18.3 AC #9):
REM   The story spec requires N=10 fresh-process launches per branch. Each
REM   iteration launches MyVoice; you generate ONE Sarira-F long-form
REM   utterance (>=250 chars / ~22s of speech — Story 17.3 §4.1 step 3
REM   canonical paragraph); close MyVoice cleanly. The loop iterates 10 times.
REM
REM Each iteration's CSV writes to a uniquely-numbered path so all 10 are
REM preserved for the dev agent's median/p90/p95 aggregation:
REM   _bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run01.csv
REM   _bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run02.csv
REM   ...
REM   _bmad-output/implementation-artifacts/18-3-rtx5090-bf16-run10.csv
REM
REM After the loop completes, the dev agent reads all 10 CSVs and computes
REM the first_chunk_latency_ms aggregate vs the fp32 branch (run 04_*.bat).
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

REM Step 1 — set tts_precision=auto (bf16 default on Ampere+).
echo.
echo ========================================
echo Step 1: setting AppSettings.tts_precision = "auto" (bf16 on RTX 5090)
echo ========================================
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-set-precision.py" auto
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
echo      "ModelRegistry initialized: ... dtype=torch.bfloat16, precision_source='app_settings_auto_ampere'..."
echo   3. Pick Sarira-F as the speaker.
echo   4. Generate the canonical Sarira-F long-form paragraph
echo      (Story 17.3 §4.1 step 3; ^>=250 chars / ~22s of speech).
echo   5. Close MyVoice cleanly. The loop continues to the next run.
echo.

for /L %%I in (1,1,10) do (
    set "RUN_NUM=0%%I"
    set "RUN_NUM=!RUN_NUM:~-2!"
    set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-rtx5090-bf16-run!RUN_NUM!.csv"
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
echo BF16 capture loop complete.
echo ========================================
echo.
echo Captured CSVs at:
dir /B "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-3-rtx5090-bf16-run*.csv" 2>nul
echo.
echo Next: run 04_Story_18.3_NFR1_FP32.bat for the fp32 override branch.
echo Then report back to the dev-story workflow.
echo.

pause
exit /b 0
