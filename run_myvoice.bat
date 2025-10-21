@echo off
REM MyVoice Launcher Script
REM Runs MyVoice application from the virtual environment

echo ========================================
echo MyVoice - Voice Cloning Application
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: MyVoice is not installed!
    echo.
    echo Please run install_myvoice.bat first to set up MyVoice.
    echo.
    pause
    exit /b 1
)

REM Quick check if myvoice package is installed
.venv\Scripts\python.exe -c "import myvoice" 2>nul
if errorlevel 1 (
    echo ERROR: MyVoice package not found!
    echo.
    echo Please run install_myvoice.bat to complete installation.
    echo.
    pause
    exit /b 1
)

REM Run MyVoice
echo Starting MyVoice...
echo.
.venv\Scripts\python.exe -m myvoice.main

REM Pause if there was an error
if errorlevel 1 (
    echo.
    echo MyVoice exited with an error.
    pause
)
