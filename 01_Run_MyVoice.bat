@echo off
REM ========================================
REM MyVoice Application Launcher (V2)
REM ========================================
REM This script launches the MyVoice application
REM using the bundled portable Python environment.
REM V2 uses embedded Qwen3-TTS (no external services).
REM NO PYTHON INSTALLATION REQUIRED!
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
    echo Have you extracted all files correctly?
    echo.
    pause
    exit /b 1
)

REM Check if source directory exists
if not exist "%SCRIPT_DIR%\src\myvoice\main.py" (
    echo [ERROR] MyVoice application files not found!
    echo Expected: %SCRIPT_DIR%\src\myvoice\main.py
    echo.
    echo Please ensure all application files are present.
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

echo.
echo ========================================
echo MyVoice V2 - Starting
echo ========================================
echo.

REM Display splash screen during initialization
powershell -WindowStyle Hidden -ExecutionPolicy Bypass -Command "Start-Process powershell -ArgumentList '-WindowStyle Hidden -ExecutionPolicy Bypass -File \"%SCRIPT_DIR%\src\install_files\show_splash.ps1\" \"%SCRIPT_DIR%\src\icon\MyVoice_Splash.png\"' -PassThru | Select-Object -ExpandProperty Id | Out-File '%TEMP%\myvoice_splash.pid' -Encoding ASCII"

REM Brief pause to show splash
timeout /t 1 /nobreak >nul

REM Close splash screen before launching app
if exist "%TEMP%\myvoice_splash.pid" (
    for /f %%p in (%TEMP%\myvoice_splash.pid) do (
        taskkill /PID %%p /F >nul 2>&1
    )
    del "%TEMP%\myvoice_splash.pid" >nul 2>&1
)

REM Launch the application
echo Starting MyVoice Application...
echo.

REM Run main.py directly (NOT as module) to ensure DLL directories are registered
REM before any package imports occur. Using -m would load __init__.py first,
REM which imports app.py and PyQt6, causing torch DLL loading conflicts.
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"

REM Capture exit code
set "EXIT_CODE=%errorlevel%"

echo.
echo ========================================
echo MyVoice has closed
echo ========================================
echo.

REM If application exits with error, show message
if !EXIT_CODE! neq 0 (
    echo [ERROR] MyVoice exited with error code: !EXIT_CODE!
    echo Check the logs folder for details.
    echo.
) else (
    echo MyVoice closed normally.
    echo.
)

REM Cleanup: Remove splash PID file if it exists
if exist "%TEMP%\myvoice_splash.pid" (
    del "%TEMP%\myvoice_splash.pid" >nul 2>&1
)

REM Auto-close terminal when app exits
exit /b 0