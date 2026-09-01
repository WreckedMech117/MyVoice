@echo off
REM ========================================
REM Story 20.4 AC #6 - GUI TTFA capture after the chunk-size retune
REM ========================================
REM Successor to 10_Story_20.3_AC4_GUI_Capture.bat. Same mechanics, three
REM deliberate differences, all of them required by Story 20.4 AC #6:
REM
REM   1. Writes 20-4-gui-r0N.csv, NOT 20-3-gui-r0N.csv. Story 20.3's
REM      captures are the BASELINE this story is measured against
REM      (1b 192 ms / TOTAL 1,353 ms) - overwriting them would destroy
REM      the comparison.
REM   2. SIX launches, not five. Launch 1 is a throwaway: the retune
REM      changes decode_window_frames 30 -> 15, which is one of the seven
REM      compile-cache key dimensions, so the first launch after this
REM      ships pays exactly ONE cold compile. Launches 2-6 are the five
REM      warm launches that compare like-for-like against Story 20.3.
REM   3. TWO generations per launch - a LONG one then a SHORT one. The
REM      short class is the one chunk_size=10 changes most (Story 20.1
REM      SS5.3 measured it moving off the residual-flush dispatch path in
REM      5/5 runs), and AC #6 asks for both classes.
REM
REM ========================================
REM WHAT TO DO IN EACH LAUNCH
REM ========================================
REM   1. Make sure a CLONED voice is the active profile so BASE is the
REM      resident model. Same profile every launch.
REM   2. WAIT for the "Preparing TTS engine" indicator to disappear.
REM      THIS MATTERS: priming holds the request semaphore, so generating
REM      while it is up measures queueing, not first-forward.
REM      ON LAUNCH 1 ONLY this may take ~20-30 s longer than usual - that
REM      is the one expected cold compile. Wait it out.
REM   3. Generate the LONG utterance. Text is in
REM      _bmad-output\implementation-artifacts\20-4-gui-utterances.txt
REM      Use the SAME text every launch.
REM   4. LET IT FINISH PLAYING. Do not close or re-generate mid-playback.
REM      Story 20.3's captures stopped after chunk 0 because the app was
REM      closed early, which is why they carry no producer-ratio data.
REM   5. Generate the SHORT utterance from the same file. Let it finish.
REM   6. Close the app with the X. MYVOICE_AUTO_QUIT_ON_CLOSE=1 is set
REM      below, so X really quits instead of minimizing to tray. Do NOT
REM      use Ctrl-C.
REM   7. The next launch starts automatically.
REM
REM Note: cmd.exe parses a literal close-paren inside a for /L body as the
REM end of the block - that bug silently cost Story 18.4 a whole run. This
REM script uses a goto loop and paren-free echoes to stay clear of it.
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    echo Expected: %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\src\myvoice\main.py" (
    echo [ERROR] MyVoice application files not found.
    echo Expected: %SCRIPT_DIR%\src\myvoice\main.py
    pause
    exit /b 1
)

"%SCRIPT_DIR%\python310\python.exe" -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [WARNING] Dependencies not installed.
    echo Please run 00_Install_Dependencies.bat first.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "OUTDIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
REM Story 18.3 measurement-mode bypass. Without this the X button minimizes
REM to tray -- minimize_to_tray defaults True -- and closing from the taskbar
REM leaves the process alive, so this script never regains control.
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM Preflight 1: compile must be engaged or there is nothing to prime and
REM the whole measurement is moot.
"%SCRIPT_DIR%\python310\python.exe" -c "import json,sys;d=json.load(open(r'%SCRIPT_DIR%\config\settings.json'));sys.exit(0 if d.get('tts_compile')=='auto' else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] config\settings.json does not have tts_compile set to auto.
    echo Compile priming cannot engage, so this measurement would be meaningless.
    echo Set tts_compile to auto and re-run.
    pause
    exit /b 1
)

REM Preflight 2: confirm the committed streamer geometry really is 10 + 5.
REM If someone reverted the retune, every number below would be a
REM re-measurement of the OLD build under a Story 20.4 filename.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import codec_token_streamer as c;sys.exit(0 if (c.DEFAULT_CHUNK_SIZE,c.DEFAULT_LOOKAHEAD)==(10,5) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] CodecTokenStreamer geometry is not the Story 20.4 committed
    echo 10 + 5. The retune is missing or was reverted.
    pause
    exit /b 1
)

set "TOTAL_RUNS=6"

echo.
echo ========================================
echo Story 20.4 AC #6 - GUI capture, %TOTAL_RUNS% launches
echo ========================================
echo.
echo Per launch:
echo   - active profile must be a CLONED voice, so BASE is resident
echo   - WAIT for "Preparing TTS engine" to clear before generating
echo   - generate the LONG utterance, LET IT FINISH PLAYING
echo   - generate the SHORT utterance, LET IT FINISH PLAYING
echo   - close with the X - auto-quit is enabled, so it really exits
echo.
echo Utterance texts:
echo   %OUTDIR%\20-4-gui-utterances.txt
echo.
echo LAUNCH 1 IS A THROWAWAY - it pays the one expected cold compile for
echo the new decode-window cache key. Its "Preparing TTS engine" will take
echo noticeably longer. Still do both generations; the analysis drops it.
echo.
echo Output CSVs:
echo   %OUTDIR%\20-4-gui-r01.csv ... r0%TOTAL_RUNS%.csv
echo.
echo Press a key to begin.
pause >nul

set /a RUN=1

:RUN_LOOP
set "CSV=%OUTDIR%\20-4-gui-r0!RUN!.csv"
set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=!CSV!"

echo.
echo ========================================
echo LAUNCH !RUN! of %TOTAL_RUNS%
echo ========================================
echo CSV: !CSV!
echo.
if !RUN! equ 1 echo THIS IS THE COLD-KEY LAUNCH - expect a long "Preparing TTS engine".
echo Reminder: wait for "Preparing TTS engine" to clear, generate LONG,
echo let it finish, generate SHORT, let it finish, then close with the X.
echo.

REM Run main.py directly - NOT as a module - to preserve the
REM torch-before-PyQt6 DLL-ordering invariant.
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] Launch !RUN! exited with code !EXIT_CODE! - check logs\myvoice.log

if exist "!CSV!" (
    for /f %%c in ('find /c /v "" ^< "!CSV!"') do echo Launch !RUN! captured %%c lines including header
) else (
    echo [WARNING] No CSV written for launch !RUN!.
    echo The env-var may not have taken effect, or no generation was run.
)

set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto RUN_LOOP

echo.
echo ========================================
echo All %TOTAL_RUNS% launches complete
echo ========================================
echo.
echo Row counts:
set /a RUN=1
:COUNT_LOOP
set "CSV=%OUTDIR%\20-4-gui-r0!RUN!.csv"
if exist "!CSV!" (
    for /f %%c in ('find /c /v "" ^< "!CSV!"') do echo   r0!RUN!: %%c lines
) else (
    echo   r0!RUN!: MISSING
)
set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto COUNT_LOOP

echo.
echo Compile telemetry from the most recent log - launch 1 should show a
echo cold compile for the new decode-window key, launches 2-6 warm:
echo.
findstr /C:"tts_compile_warmup_priming" /C:"Compile cache hit" /C:"warmup primed" /C:"decode_window_frames" "%SCRIPT_DIR%\logs\myvoice.log" 2>nul | more
echo.
echo Now aggregate - session_id grouping is mandatory, see Story 20.3 SS4.1a:
echo   python310\python.exe %OUTDIR%\20-4-aggregate-gui.py --skip-first-launch
echo.
echo Hand the CSVs and that output back to the session to populate the
echo Story 20.4 evidence file section 4.
echo.

pause
exit /b 0
