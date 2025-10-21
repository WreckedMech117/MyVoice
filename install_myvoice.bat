@echo off
REM MyVoice Installation Script
REM Sets up virtual environment, installs all dependencies, and prepares MyVoice for use

echo ========================================
echo MyVoice - Installation Script
echo ========================================
echo.
echo This script will:
echo   1. Create Python virtual environment
echo   2. Download and install FFmpeg
echo   3. Install Python dependencies
echo   4. Install MyVoice package
echo.
pause

REM Check if Python is available
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ first from https://www.python.org/
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment if it doesn't exist
echo [2/4] Setting up virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists, skipping creation.
)
echo.

REM Setup FFmpeg
echo [3/4] Setting up FFmpeg...
if not exist "ffmpeg\ffmpeg.exe" (
    echo FFmpeg not found. Running setup script...
    call setup_ffmpeg.bat
    if errorlevel 1 (
        echo ERROR: FFmpeg setup failed
        echo You can manually download FFmpeg later if needed.
        set /p continue="Continue installation anyway? (y/N): "
        if /i not "!continue!"=="y" exit /b 1
    )
) else (
    echo FFmpeg already installed, skipping.
)
echo.

REM Upgrade pip
echo [4/4] Installing dependencies...
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1

REM Install PyTorch first (CPU version)
echo Installing PyTorch (CPU version - this may take a while)...
.venv\Scripts\pip.exe install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch installation had issues, trying requirements.txt instead
)

REM Install requirements
echo Installing other dependencies from requirements.txt...
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Common issues:
    echo   - PyAudio requires Microsoft Visual C++ Build Tools
    echo   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    pause
    exit /b 1
)

REM Install MyVoice package in editable mode
echo Installing MyVoice package...
.venv\Scripts\pip.exe install -e .
if errorlevel 1 (
    echo ERROR: Failed to install MyVoice package
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Make sure GPT-SoVITS is running on http://localhost:9880
echo      Download from: https://github.com/RVC-Boss/GPT-SoVITS
echo.
echo   2. Run MyVoice using:
echo      run_myvoice.bat
echo.
echo Troubleshooting:
echo   - If audio doesn't work, install VB-Audio Cable or VoiceMeeter
echo   - Check logs in the 'logs' folder for errors
echo   - Settings are stored in 'config/settings.json'
echo.
pause
