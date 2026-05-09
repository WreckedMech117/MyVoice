@echo off
REM ============================================================================
REM MyVoice Release Build Script
REM Complete automated build process: Clean -> Build -> Installer -> Checksums
REM ============================================================================

setlocal enabledelayedexpansion

REM Change to build_tools directory
cd /d "%~dp0"

REM Set portable Python paths
set PYTHON_DIR=%~dp0..\python310
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set SCRIPTS_DIR=%PYTHON_DIR%\Scripts

echo.
echo ============================================================================
echo MyVoice Release Builder
echo ============================================================================
echo.

REM ============================================================================
REM Pre-Build Checks
REM ============================================================================

echo [Pre-Build Checks]
echo.

REM Check Python installation
if not exist "%PYTHON_EXE%" (
    echo ERROR: Portable Python not found at %PYTHON_EXE%
    echo Please ensure python310 directory exists
    pause
    exit /b 1
)

echo + Python found:
"%PYTHON_EXE%" --version

REM Check PyInstaller
if not exist "%SCRIPTS_DIR%\pyinstaller.exe" (
    echo ERROR: PyInstaller not found in %SCRIPTS_DIR%
    echo Install with: %PYTHON_EXE% -m pip install pyinstaller
    pause
    exit /b 1
)

echo + PyInstaller found:
"%PYTHON_EXE%" -m PyInstaller --version

REM Check Inno Setup
set INNO_SETUP="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %INNO_SETUP% (
    echo ERROR: Inno Setup 6 not found at %INNO_SETUP%
    echo Please install from: https://jrsoftware.org/isdl.php
    pause
    exit /b 1
)

echo + Inno Setup found

REM Check required files
if not exist "myvoice.spec" (
    echo ERROR: myvoice.spec not found
    pause
    exit /b 1
)

if not exist "installer.iss" (
    echo ERROR: installer.iss not found
    pause
    exit /b 1
)

echo + Build files found
echo.

REM ----------------------------------------------------------------------------
REM Pin verification — qwen-tts must match Story 16.1 / D-12 commit hash
REM (tooling-2 AC #2). Halts the build at "Pre-Build Checks" on mismatch.
REM ----------------------------------------------------------------------------

echo [Pin Verification]
echo.
"%PYTHON_EXE%" verify_qwen_tts_pin.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: qwen-tts pin verification failed!
    echo ============================================================================
    echo.
    echo See above for the expected/actual hashes and restoration commands.
    echo.
    pause
    exit /b 1
)
echo.

REM ============================================================================
REM Get/Update Version
REM ============================================================================

echo [Version Management]
echo.

REM Display current version
"%PYTHON_EXE%" version.py

echo.
set /p INCREMENT_BUILD="Increment build number? (y/N): "
if /i "!INCREMENT_BUILD!"=="y" (
    "%PYTHON_EXE%" version.py increment-build
    echo.
)

REM ----------------------------------------------------------------------------
REM Sync version constants across files (tooling-2 AC #5). Propagates the
REM possibly-just-incremented VERSION_MAJOR/MINOR/PATCH/BUILD from
REM version.py into src/myvoice/__init__.py and installer.iss so the
REM produced installer's Add/Remove Programs entry, registry Version key,
REM registry Build key, and OutputBaseFilename all agree with version.py.
REM Closes the gap noted in Story tooling-2 AC #5.
REM ----------------------------------------------------------------------------
echo [Version Sync]
"%PYTHON_EXE%" version.py update-all
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: version.py update-all failed!
    echo ============================================================================
    echo.
    pause
    exit /b 1
)
echo.

REM ============================================================================
REM Step 1: Clean Previous Builds
REM ============================================================================

echo [Step 1/5] Cleaning previous builds...
echo.

if exist "build" (
    echo Removing build\ directory...
    rd /s /q "build" 2>nul
)

if exist "dist" (
    echo Removing dist\ directory...
    rd /s /q "dist" 2>nul
)

cd ..
if exist "installer_output" (
    echo Removing installer_output\ directory...
    rd /s /q "installer_output" 2>nul
)
cd build_tools

echo + Cleanup complete
echo.

REM ============================================================================
REM Step 2: Build Executable with PyInstaller
REM ============================================================================

echo [Step 2/5] Building executable with PyInstaller...
echo.
echo This may take 5-15 minutes depending on your system...
echo.

set BUILD_START=%time%

"%PYTHON_EXE%" -m PyInstaller myvoice.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: PyInstaller build failed!
    echo ============================================================================
    echo.
    echo Check the error messages above for details.
    echo Common issues:
    echo   - Missing dependencies: pip install -r requirements.txt
    echo   - Import errors: Check hidden imports in myvoice.spec
    echo   - Path issues: Verify file paths in spec file
    echo.
    pause
    exit /b 1
)

set BUILD_END=%time%

echo.
echo + Executable build complete
echo.

REM Verify output
if not exist "dist\MyVoice\MyVoice.exe" (
    echo ERROR: MyVoice.exe not found in dist\MyVoice\
    pause
    exit /b 1
)

REM Display executable info
echo Executable information:
dir "dist\MyVoice\MyVoice.exe"
echo.

REM Calculate size
for /f %%A in ('powershell -command "((Get-ChildItem -Path 'dist\MyVoice' -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB).ToString('0.00')"') do set SIZE=%%A
echo Total size: !SIZE! MB
echo.

REM ============================================================================
REM Step 3: Build Installer with Inno Setup
REM ============================================================================

echo [Step 3/5] Building installer with Inno Setup...
echo.

%INNO_SETUP% installer.iss

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: Inno Setup build failed!
    echo ============================================================================
    echo.
    echo Check the error messages above for details.
    echo Common issues:
    echo   - Missing source files: Verify paths in installer.iss
    echo   - Icon file issues: Check icon paths exist
    echo   - Syntax errors: Review installer.iss for typos
    echo.
    pause
    exit /b 1
)

echo.
echo + Installer build complete
echo.

cd ..

REM Verify installer output
if not exist "installer_output\MyVoice-Setup-v*.exe" (
    echo ERROR: Installer not found in installer_output\
    cd build_tools
    pause
    exit /b 1
)

REM Display installer info
echo Installer information:
dir "installer_output\MyVoice-Setup-v*.exe"
echo.

REM ============================================================================
REM Step 4: Generate Checksums
REM ============================================================================

echo [Step 4/5] Generating checksums...
echo.

cd installer_output

for %%F in (MyVoice-Setup-v*.exe) do (
    echo Generating SHA256 for %%F...
    certutil -hashfile "%%F" SHA256 > "%%F.sha256.txt"

    echo Generating MD5 for %%F...
    certutil -hashfile "%%F" MD5 > "%%F.md5.txt"

    echo.
    echo + Checksums saved:
    echo   - %%F.sha256.txt
    echo   - %%F.md5.txt
)

echo.

REM ============================================================================
REM Step 5: Organize Release Artifacts
REM ============================================================================

echo [Step 5/5] Organizing release artifacts...
echo.

REM Get version from version.py
cd ..\build_tools
for /f "tokens=*" %%a in ('"%PYTHON_EXE%" -c "import version; print(version.VERSION)"') do set VERSION=%%a
cd ..\installer_output

REM Create versioned release folder
set RELEASE_DIR=MyVoice-v!VERSION!
if exist "!RELEASE_DIR!" (
    echo Removing existing release folder...
    rd /s /q "!RELEASE_DIR!"
)

echo Creating release folder: !RELEASE_DIR!
mkdir "!RELEASE_DIR!"

REM Copy installer and checksums
for %%F in (MyVoice-Setup-v*.exe) do (
    echo Copying %%F...
    copy "%%F" "!RELEASE_DIR!\" >nul
)

for %%F in (*.sha256.txt) do (
    echo Copying %%F...
    copy "%%F" "!RELEASE_DIR!\" >nul
)

for %%F in (*.md5.txt) do (
    echo Copying %%F...
    copy "%%F" "!RELEASE_DIR!\" >nul
)

REM Copy documentation
if exist "..\resources\readme.txt" (
    echo Copying readme.txt...
    copy "..\resources\readme.txt" "!RELEASE_DIR!\README.txt" >nul
)

if exist "..\LICENSE" (
    echo Copying LICENSE...
    copy "..\LICENSE" "!RELEASE_DIR!\LICENSE.txt" >nul
)

echo.
echo + Release artifacts organized in: !RELEASE_DIR!\
echo.

REM ============================================================================
REM Build Summary
REM ============================================================================

echo ============================================================================
echo BUILD COMPLETE!
echo ============================================================================
echo.
echo Build Summary:
echo   Version:           !VERSION!
echo   Executable Size:   !SIZE! MB
echo   Build Time:        %BUILD_START% - %BUILD_END%
echo.
echo Output Locations:
echo   Executable:        build_tools\dist\MyVoice\
echo   Installer:         installer_output\!RELEASE_DIR!\
echo.
echo Release Artifacts:
dir "!RELEASE_DIR!\" /b
echo.
echo ============================================================================
echo.
echo Next Steps:
echo   1. Test the executable: build_tools\dist\MyVoice\MyVoice.exe
echo   2. Test the installer: installer_output\!RELEASE_DIR!\MyVoice-Setup-v*.exe
echo   3. Run validation: build_tools\test_release.bat
echo   4. Review release checklist: build_tools\RELEASE_CHECKLIST.md
echo   5. (Optional) Code sign installer: See EXE3.2 documentation
echo.
echo ============================================================================
echo.

cd ..\build_tools

endlocal
pause
