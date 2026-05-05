@echo off
REM MyVoice Build Script
REM Builds the application using PyInstaller and sets up the distribution
REM All build outputs go to build_tools/build/ and build_tools/dist/

echo ========================================
echo MyVoice Build Script
echo ========================================
echo.

REM Store the original directory and switch to build_tools
set ORIGINAL_DIR=%CD%
cd /d "%~dp0"

echo [1/3] Running PyInstaller...
..\python310\python.exe -m PyInstaller myvoice.spec --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed!
    cd /d "%ORIGINAL_DIR%"
    exit /b 1
)

echo.
echo [2/3] Copying voice_files to distribution root...
if exist "dist\MyVoice\_internal\voice_files" (
    xcopy /E /I /Y "dist\MyVoice\_internal\voice_files" "dist\MyVoice\voice_files" >nul
    echo Voice files copied successfully
) else (
    echo WARNING: voice_files not found in _internal, skipping copy
)

echo.
echo [3/3] Creating config directory...
if not exist "dist\MyVoice\config" (
    mkdir "dist\MyVoice\config"
    echo Config directory created
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Output: build_tools\dist\MyVoice\MyVoice.exe
echo Size:
dir "dist\MyVoice" | find "bytes"
echo.

cd /d "%ORIGINAL_DIR%"
