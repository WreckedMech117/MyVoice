@echo off
REM ========================================
REM Story 18.4 Task 8.1 — NFR1 Branch A: BF16 + COMPILE (N=10)
REM ========================================
REM Programmatically sets AppSettings.tts_precision="bf16" + tts_compile="on"
REM (the production target — the bf16+compile branch). Launches MyVoice
REM N=10 times in fresh-process loop. Each iteration generates ONE Sarira-F
REM long-form utterance (Story 17.3 §4.1 step 3 canonical paragraph; >=250
REM chars / ~22s of speech); per-iteration CSV captures
REM `metrics.first_chunk_latency_ms` for the 3-way A/B/C aggregator.
REM
REM Cold-compile distinction (architecture D-23 + Story 18.4 Task 8.4):
REM   Run 1 is COLD-COMPILE (slow startup, ~10-30s additional latency).
REM   Runs 2-10 hit the warm cache. The aggregator script DISCARDS run #1
REM   from the median calculation and reports it separately.
REM
REM Per-iteration CSVs land at:
REM   _bmad-output/implementation-artifacts/18-4-rtx5090-bf16-compile-run01.csv
REM   ...
REM   _bmad-output/implementation-artifacts/18-4-rtx5090-bf16-compile-run10.csv
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
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

REM Step 1 — set tts_precision=bf16 + tts_compile=on (branch A).
echo.
echo ========================================
echo Step 1: AppSettings.tts_precision="bf16" + tts_compile="on" (BRANCH A)
echo ========================================
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-set-precision-and-compile.py" bf16 on
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
echo Step 2: starting N=10 fresh-process measurement loop (BF16+COMPILE)
echo ========================================
echo.
echo For each iteration:
echo   1. Wait for MyVoice to finish loading.
echo   2. Look for the Story 18.4 INFO lines in logs:
echo      - "ModelRegistry initialized: ... compile_engaged='deferred'"
echo      - "torch.compile + CUDA Graph engaged (decode_window_frames=30, ..., cache=cold|warm)"
echo      - "Compile cache hit; skipping warmup priming" OR
echo        "Compile warmup primed cache successfully (duration=...ms)"
echo   3. Pick Sarira-F as the speaker.
echo   4. Generate the canonical Sarira-F long-form paragraph (Story 17.3
echo      §4.1 step 3; ^>=250 chars / ~22s of speech).
echo   5. Close MyVoice cleanly. Loop continues to next run.
echo.
echo NOTE: Run #1 is COLD-COMPILE (slow startup). The aggregator script
echo discards run #1 from the median. Runs #2-10 are warm-cache and form
echo the canonical speedup measurement.
echo.

for /L %%I in (1,1,10) do (
    set "RUN_NUM=0%%I"
    set "RUN_NUM=!RUN_NUM:~-2!"
    set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-rtx5090-bf16-compile-run!RUN_NUM!.csv"
    echo.
    echo ===== Run %%I of 10 (BF16+COMPILE) =====
    echo CSV: !MYVOICE_PROGRESSIVE_PLAYBACK_CSV!
    echo.
    "%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
    if errorlevel 1 (
        echo [WARN] Run %%I exited with non-zero code; continuing.
    )
)

echo.
echo ========================================
echo BF16+COMPILE capture loop complete (BRANCH A).
echo ========================================
echo.
echo Captured CSVs at:
dir /B "%SCRIPT_DIR%\_bmad-output\implementation-artifacts\18-4-rtx5090-bf16-compile-run*.csv" 2>nul
echo.
echo Next: run 06_Story_18.4_NFR1_BF16_EAGER.bat for BRANCH B (bf16 + eager).
echo.

pause
exit /b 0
