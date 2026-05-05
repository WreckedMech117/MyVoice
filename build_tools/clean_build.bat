@echo off
REM ============================================================================
REM MyVoice Build Cleanup Script
REM Removes all build artifacts and temporary files
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo MyVoice Build Cleanup
echo ========================================================================
echo.

REM Change to build_tools directory
cd /d "%~dp0"

set ERROR_COUNT=0

REM ============================================================================
REM Clean PyInstaller Build Artifacts
REM ============================================================================

echo [1/5] Cleaning PyInstaller build artifacts...

if exist "build" (
    echo   - Removing build\ directory...
    rd /s /q "build" 2>nul
    if exist "build" (
        echo   ! Warning: Could not remove build\ directory
        set /a ERROR_COUNT+=1
    ) else (
        echo   + build\ removed
    )
) else (
    echo   - build\ not found (already clean)
)

if exist "dist" (
    echo   - Removing dist\ directory...
    rd /s /q "dist" 2>nul
    if exist "dist" (
        echo   ! Warning: Could not remove dist\ directory
        set /a ERROR_COUNT+=1
    ) else (
        echo   + dist\ removed
    )
) else (
    echo   - dist\ not found (already clean)
)

REM Clean PyInstaller spec warnings log
if exist "warn-myvoice.txt" (
    echo   - Removing PyInstaller warnings log...
    del /q "warn-myvoice.txt" 2>nul
    echo   + warn-myvoice.txt removed
)

echo.

REM ============================================================================
REM Clean Installer Output
REM ============================================================================

echo [2/5] Cleaning installer output...

cd ..
if exist "installer_output" (
    echo   - Removing installer_output\ directory...
    rd /s /q "installer_output" 2>nul
    if exist "installer_output" (
        echo   ! Warning: Could not remove installer_output\ directory
        set /a ERROR_COUNT+=1
    ) else (
        echo   + installer_output\ removed
    )
) else (
    echo   - installer_output\ not found (already clean)
)

cd build_tools
echo.

REM ============================================================================
REM Clean Python Cache Files
REM ============================================================================

echo [3/5] Cleaning Python cache files...

cd ..\src

set CACHE_COUNT=0

REM Remove __pycache__ directories
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        rd /s /q "%%d" 2>nul
        set /a CACHE_COUNT+=1
    )
)

REM Remove .pyc files
for /r %%f in (*.pyc) do (
    if exist "%%f" (
        del /q "%%f" 2>nul
        set /a CACHE_COUNT+=1
    )
)

REM Remove .pyo files
for /r %%f in (*.pyo) do (
    if exist "%%f" (
        del /q "%%f" 2>nul
        set /a CACHE_COUNT+=1
    )
)

echo   + Removed !CACHE_COUNT! cache files/directories

cd ..\build_tools
echo.

REM ============================================================================
REM Clean Temporary Build Files
REM ============================================================================

echo [4/5] Cleaning temporary files...

set TEMP_COUNT=0

REM Clean Inno Setup temporary files
if exist "Output" (
    rd /s /q "Output" 2>nul
    set /a TEMP_COUNT+=1
    echo   + Removed Output\ directory
)

REM Clean any .log files in build_tools
for %%f in (*.log) do (
    del /q "%%f" 2>nul
    set /a TEMP_COUNT+=1
)

REM Clean any .tmp files
for %%f in (*.tmp) do (
    del /q "%%f" 2>nul
    set /a TEMP_COUNT+=1
)

if !TEMP_COUNT! EQU 0 (
    echo   - No temporary files found
) else (
    echo   + Removed !TEMP_COUNT! temporary files
)

echo.

REM ============================================================================
REM Verify Cleanup
REM ============================================================================

echo [5/5] Verifying cleanup...

set REMAINING=0

if exist "build" set /a REMAINING+=1
if exist "dist" set /a REMAINING+=1
if exist "..\installer_output" set /a REMAINING+=1

if !REMAINING! EQU 0 (
    echo   + All build artifacts removed successfully
) else (
    echo   ! Warning: Some directories could not be removed
    echo   ! This may be because files are in use
    echo   ! Try closing applications and running cleanup again
    set /a ERROR_COUNT+=1
)

echo.
echo ========================================================================

if !ERROR_COUNT! EQU 0 (
    echo Cleanup completed successfully!
    echo Build environment is clean and ready for fresh build.
) else (
    echo Cleanup completed with !ERROR_COUNT! warning(s)
    echo Some files may still be in use. Close applications and try again.
)

echo ========================================================================
echo.

endlocal
pause
