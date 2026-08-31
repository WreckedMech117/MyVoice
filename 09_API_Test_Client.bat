@echo off
REM ========================================
REM MyVoice Local TTS API - Test Client
REM ========================================
REM Small standalone GUI to call /v1/audio/speech and play the audio back.
REM Enable the API first in MyVoice: Settings > API Access.
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found at %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\tools\api_test_client.py" (
    echo [ERROR] tools\api_test_client.py not found.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

echo.
echo ========================================
echo MyVoice - API Test Client
echo ========================================
echo Make sure the API is enabled in Settings ^> API Access.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\tools\api_test_client.py"

if errorlevel 1 (
    echo.
    echo [ERROR] The test client exited with an error.
    pause
)
