@echo off
REM FFmpeg Setup Script for MyVoice
REM Downloads and installs FFmpeg binaries

echo ========================================
echo MyVoice - FFmpeg Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ first
    echo.
    pause
    exit /b 1
)

REM Run the Python setup script
python setup_ffmpeg.py

pause
