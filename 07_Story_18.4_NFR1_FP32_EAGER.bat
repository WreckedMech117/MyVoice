@echo off
REM ========================================
REM Story 18.4 Task 8.1 — NFR1 Branch C: FP32 + EAGER (N=10)
REM ========================================
REM Programmatically sets AppSettings.tts_precision="fp32" + tts_compile="off"
REM (the strict-eager baseline; TF32 + cuDNN benchmark still engage from
REM Story 18.2). Launches MyVoice N=10 times in fresh-process loop. Each
REM iteration generates ONE Sarira-F long-form utterance; per-iteration
REM CSV captures first_chunk_latency_ms.
REM
REM Per-iteration CSVs land at:
REM   _bmad-output/implementation-artifacts/18-4-rtx5090-fp32-eager-run01.csv
REM   ...
REM   _bmad-output/implementation-artifacts/18-4-rtx5090-fp32-eager-run10.csv
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
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

REM Step 1 — set tts_precision=fp32 + tts_compile=off (branch C).
echo.
echo ========================================
echo Step 1: AppSettings.tts_precision="fp32" + tts_compile="off" (BRANCH C)
echo ========================================
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-set-precision-and-compile.py" fp32 off
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
echo Step 2: starting N=10 fresh-process measurement loop (FP32+EAGER)
echo ========================================
echo.
echo For each iteration:
echo   1. Wait for MyVoice to finish loading.
echo   2. Look for the Story 18.4 INFO lines in logs:
echo      - "ModelRegistry initialized: ... dtype=torch.float32, precision_source='app_settings_override'"
echo      - "torch.compile skipped: user_disabled"
echo   3. Pick Sarira-F; generate the canonical long-form paragraph.
echo   4. Close MyVoice cleanly. Loop continues.
echo.

for /L %%I in (1,1,10) do (
    set "RUN_NUM=0%%I"
    set "RUN_NUM=!RUN_NUM:~-2!"
    set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-rtx5090-fp32-eager-run!RUN_NUM!.csv"
    echo.
    echo ===== Run %%I of 10 FP32+EAGER =====
    echo CSV: !MYVOICE_PROGRESSIVE_PLAYBACK_CSV!
    echo.
    "%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
    if errorlevel 1 (
        echo [WARN] Run %%I exited with non-zero code; continuing.
    )
)

echo.
echo ========================================
echo FP32+EAGER capture loop complete (BRANCH C).
echo ========================================
echo.
echo Captured CSVs at:
dir /B "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-rtx5090-fp32-eager-run*.csv" 2>nul
echo.
echo All three branches captured. Run aggregate at:
echo   python310/python.exe _bmad-output/implementation-artifacts/18-4-aggregate-nfr1.py
echo.

pause
exit /b 0
