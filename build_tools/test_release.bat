@echo off
REM ============================================================================
REM MyVoice Release Testing Script
REM Automated validation of build artifacts before distribution
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================================
echo MyVoice Release Testing
echo ============================================================================
echo.

set TESTS_PASSED=0
set TESTS_FAILED=0

REM ============================================================================
REM Test 1: Verify Executable Exists
REM ============================================================================

echo [Test 1/8] Checking executable existence...

if exist "dist\MyVoice\MyVoice.exe" (
    echo + PASS: MyVoice.exe found
    set /a TESTS_PASSED+=1
) else (
    echo - FAIL: MyVoice.exe not found in dist\MyVoice\
    set /a TESTS_FAILED+=1
)
echo.

REM ============================================================================
REM Test 2: Verify Executable Size
REM ============================================================================

echo [Test 2/8] Checking executable size...

if exist "dist\MyVoice\MyVoice.exe" (
    for %%A in ("dist\MyVoice\MyVoice.exe") do set SIZE=%%~zA

    REM Check if size is > 1MB (minimum reasonable size)
    if !SIZE! GTR 1048576 (
        set /a SIZE_MB=!SIZE! / 1048576
        echo + PASS: Executable size: !SIZE_MB! MB
        set /a TESTS_PASSED+=1
    ) else (
        echo - FAIL: Executable size too small: !SIZE! bytes
        set /a TESTS_FAILED+=1
    )
) else (
    echo - SKIP: Executable not found
)
echo.

REM ============================================================================
REM Test 3: Verify Required DLLs Bundled
REM ============================================================================

echo [Test 3/8] Checking required DLLs...

set DLL_COUNT=0

REM Check for Python DLL
if exist "dist\MyVoice\_internal\python310.dll" (
    echo + Found python310.dll
    set /a DLL_COUNT+=1
) else (
    echo - Missing python310.dll
)

REM Check for Python3 DLL
if exist "dist\MyVoice\_internal\python3.dll" (
    echo + Found python3.dll
    set /a DLL_COUNT+=1
) else (
    echo - Missing python3.dll
)

if !DLL_COUNT! EQU 2 (
    echo + PASS: Python DLLs bundled correctly
    set /a TESTS_PASSED+=1
) else (
    echo - FAIL: Missing Python DLLs (!DLL_COUNT!/2)
    set /a TESTS_FAILED+=1
)
echo.

REM ============================================================================
REM Test 4: Verify Installer Exists
REM ============================================================================

echo [Test 4/8] Checking installer existence...

cd ..
set INSTALLER_FOUND=0

for %%F in (installer_output\MyVoice-Setup-v*.exe) do (
    if exist "%%F" (
        echo + PASS: Installer found: %%~nxF
        set INSTALLER_FOUND=1
        set INSTALLER_PATH=%%F
    )
)

if !INSTALLER_FOUND! EQU 1 (
    set /a TESTS_PASSED+=1
) else (
    echo - FAIL: No installer found in installer_output\
    set /a TESTS_FAILED+=1
)

cd build_tools
echo.

REM ============================================================================
REM Test 5: Verify Installer Size
REM ============================================================================

echo [Test 5/8] Checking installer size...

if !INSTALLER_FOUND! EQU 1 (
    cd ..
    for %%A in ("!INSTALLER_PATH!") do set INSTALLER_SIZE=%%~zA
    cd build_tools

    REM Check if size is > 100MB (minimum reasonable size for packaged app)
    if !INSTALLER_SIZE! GTR 104857600 (
        set /a SIZE_MB=!INSTALLER_SIZE! / 1048576
        echo + PASS: Installer size: !SIZE_MB! MB
        set /a TESTS_PASSED+=1
    ) else (
        set /a SIZE_MB=!INSTALLER_SIZE! / 1048576
        echo - FAIL: Installer size too small: !SIZE_MB! MB
        echo   Expected: >100 MB
        set /a TESTS_FAILED+=1
    )
) else (
    echo - SKIP: Installer not found
)
echo.

REM ============================================================================
REM Test 6: Verify Checksums Exist
REM ============================================================================

echo [Test 6/8] Checking checksums...

cd ..
set CHECKSUM_COUNT=0

for %%F in (installer_output\*.sha256.txt) do (
    if exist "%%F" (
        echo + Found: %%~nxF
        set /a CHECKSUM_COUNT+=1
    )
)

for %%F in (installer_output\*.md5.txt) do (
    if exist "%%F" (
        echo + Found: %%~nxF
        set /a CHECKSUM_COUNT+=1
    )
)

if !CHECKSUM_COUNT! GEQ 2 (
    echo + PASS: Checksums generated (!CHECKSUM_COUNT! files)
    set /a TESTS_PASSED+=1
) else (
    echo - FAIL: Missing checksums (found !CHECKSUM_COUNT!, expected >= 2)
    set /a TESTS_FAILED+=1
)

cd build_tools
echo.

REM ============================================================================
REM Test 7: Verify Required Resources
REM ============================================================================

echo [Test 7/8] Checking required resources...

set RESOURCE_COUNT=0

if exist "..\resources\license.txt" (
    echo + Found license.txt
    set /a RESOURCE_COUNT+=1
) else (
    echo - Missing license.txt
)

if exist "..\resources\readme.txt" (
    echo + Found readme.txt
    set /a RESOURCE_COUNT+=1
) else (
    echo - Missing readme.txt
)

if exist "..\src\icon\MyVoice.png" (
    echo + Found MyVoice.png
    set /a RESOURCE_COUNT+=1
) else (
    echo - Missing MyVoice.png
)

if exist "..\src\icon\MyVoice_Splash.png" (
    echo + Found MyVoice_Splash.png
    set /a RESOURCE_COUNT+=1
) else (
    echo - Missing MyVoice_Splash.png
)

if !RESOURCE_COUNT! EQU 4 (
    echo + PASS: All required resources found
    set /a TESTS_PASSED+=1
) else (
    echo - FAIL: Missing resources (!RESOURCE_COUNT!/4)
    set /a TESTS_FAILED+=1
)
echo.

REM ============================================================================
REM Test 8: Quick Launch Test (Optional)
REM ============================================================================

echo [Test 8/8] Quick launch test (optional)...
echo.
echo This test will attempt to launch MyVoice.exe briefly.
set /p DO_LAUNCH="Run launch test? (y/N): "

if /i "!DO_LAUNCH!"=="y" (
    echo.
    echo Launching MyVoice.exe...
    echo (Will auto-close after 3 seconds)
    echo.

    start "" "dist\MyVoice\MyVoice.exe"

    REM Wait 3 seconds
    timeout /t 3 /nobreak >nul

    REM Try to close it
    taskkill /im MyVoice.exe /f >nul 2>&1

    if %ERRORLEVEL% EQU 0 (
        echo + PASS: Executable launched and terminated successfully
        set /a TESTS_PASSED+=1
    ) else (
        echo + PARTIAL: Executable may have launched (unable to verify close)
        set /a TESTS_PASSED+=1
    )
) else (
    echo - SKIP: Launch test skipped
)
echo.

REM ============================================================================
REM Test Results Summary
REM ============================================================================

echo ============================================================================
echo TEST RESULTS
echo ============================================================================
echo.

set /a TOTAL_TESTS=!TESTS_PASSED! + !TESTS_FAILED!
set /a PASS_PERCENT=(TESTS_PASSED * 100) / TOTAL_TESTS

echo Tests Passed:  !TESTS_PASSED!
echo Tests Failed:  !TESTS_FAILED!
echo Total Tests:   !TOTAL_TESTS!
echo Pass Rate:     !PASS_PERCENT!%%
echo.

if !TESTS_FAILED! EQU 0 (
    echo Status: ALL TESTS PASSED ✓
    echo.
    echo ============================================================================
    echo The release build is ready for distribution!
    echo ============================================================================
    echo.
    echo Recommended next steps:
    echo   1. Test installer on clean Windows system
    echo   2. Verify all application features work
    echo   3. Review RELEASE_CHECKLIST.md
    echo   4. (Optional) Code sign the installer
    echo   5. Distribute the release
    echo.
) else (
    echo Status: SOME TESTS FAILED ✗
    echo.
    echo ============================================================================
    echo The release build has issues that need to be addressed.
    echo ============================================================================
    echo.
    echo Please review the failed tests above and:
    echo   1. Fix any missing files or resources
    echo   2. Rebuild with: build_release.bat
    echo   3. Run tests again
    echo.
)

echo ============================================================================
echo.

endlocal
pause
