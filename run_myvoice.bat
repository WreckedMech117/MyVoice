@echo off
REM ========================================
REM MyVoice Application Launcher
REM ========================================
REM This script launches the MyVoice application
REM using the virtual environment.
REM ========================================

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Enable delayed expansion for variables
setlocal EnableDelayedExpansion

echo.
echo ========================================
echo MyVoice V2 - Starting
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please set up the environment first:
    echo   1. python -m venv .venv
    echo   2. .venv\Scripts\activate
    echo   3. pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo   4. pip install -r requirements.txt
    echo   5. pip install -e .
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
"%SCRIPT_DIR%\.venv\Scripts\python.exe" -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [WARNING] Dependencies not installed!
    echo.
    echo Please run the following in your virtual environment:
    echo   pip install -r requirements.txt
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Add src directory to Python path
set "PYTHONPATH=%SCRIPT_DIR%\src"

echo Starting MyVoice Application...
echo.

REM Launch the application
"%SCRIPT_DIR%\.venv\Scripts\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"

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
    pause
) else (
    echo MyVoice closed normally.
    echo.
)

exit /b 0
