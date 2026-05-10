@echo off
REM ========================================
REM Story 18.3 Task 1 — Dtype Audit (one-shot)
REM ========================================
REM Sets MYVOICE_DTYPE_AUDIT=1 so model_registry._instrument_dtype_audit
REM fires after the model loads. The audit:
REM   - Logs every relevant dtype attribute on model / talker / speech_tokenizer
REM   - Attaches one-shot forward hooks on talker + speech_tokenizer
REM   - Detaches hooks after the first generation's forward pass
REM
REM Procedure (Commander):
REM   1. CLEAR logs first so the audit lines stand alone:
REM        del logs\myvoice.log
REM   2. Run this .bat.
REM   3. Pick Sarira-F as the speaker.
REM   4. Generate ANY short utterance (e.g., the canonical Story 17.3 §4.1
REM      step 1 short paragraph — does not need to be the long-form one;
REM      the audit only needs ONE forward pass on each module).
REM   5. Close the app cleanly.
REM   6. The audit lines in logs\myvoice.log are tagged [DTYPE_AUDIT] and
REM      [DTYPE_AUDIT_FWD]. The dev agent reads them and populates
REM      evidence file §"Pre-implementation audit" + §"End-to-end dtype audit".
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found!
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\src\myvoice\main.py" (
    echo [ERROR] MyVoice application files not found!
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

REM Engage the dtype audit. Production has zero overhead when this env var
REM is unset; setting it to 1 routes through model_registry._instrument_dtype_audit.
set "MYVOICE_DTYPE_AUDIT=1"

echo.
echo ========================================
echo MyVoice V2 - Story 18.3 Task 1 Dtype Audit
echo ========================================
echo MYVOICE_DTYPE_AUDIT=1 (one-shot dtype + forward-hook capture)
echo.
echo Look for [DTYPE_AUDIT] and [DTYPE_AUDIT_FWD] lines in
echo logs\myvoice.log AFTER you generate a single short utterance and
echo close MyVoice cleanly.
echo ========================================
echo.

"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"

set "EXIT_CODE=%errorlevel%"

echo.
echo ========================================
echo MyVoice has closed (exit code: !EXIT_CODE!)
echo ========================================
echo.
echo Next steps:
echo   1. Inspect logs\myvoice.log for [DTYPE_AUDIT] and [DTYPE_AUDIT_FWD] lines.
echo   2. Report the captures back to the dev-story workflow — the agent
echo      will fold them into the evidence file.
echo.

pause
exit /b 0
