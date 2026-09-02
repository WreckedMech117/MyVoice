@echo off
REM ========================================
REM Story 20.6 follow-up - KILL-SWITCH BASELINE (the other arm)
REM ========================================
REM WHY THIS EXISTS
REM
REM Story 20.6's GUI capture measured TOTAL 1,364 ms against Story 20.3's
REM 1,353 ms and found segment 2 flat where retiring the lookahead should
REM have cut it by roughly a sixth. That comparison CANNOT BE ATTRIBUTED:
REM there is no post-20.5, pre-20.6 GUI baseline. Story 20.5 verified
REM headless, and the last GUI capture predates codec state caching. So
REM 1,353 -^> 1,364 spans two stories and an unknown amount of driver, OS
REM and pin drift.
REM
REM MYVOICE_CODEC_STATE_CACHE=0 gives the pre-20.5 geometry - stateless
REM decode, lookahead 5, the post-decode trim and the Story 20.4 seam
REM blend - on TODAY'S code, TODAY'S machine, TODAY'S driver. That is the
REM clean control the 20.3 comparison cannot be. This launcher captures it.
REM
REM ========================================
REM THIS IS THE *OTHER* ARM. IT DOES NOT REPLACE 13_.
REM ========================================
REM   13_Story_20.6_AC3_GUI_Capture.bat  -^>  20-6-gui-r0N.csv        (arm B)
REM   15_Story_20.6_KillSwitch_Baseline  -^>  20-6-killswitch-r0N.csv (arm A)
REM
REM Different glob on purpose. The 20-6-gui-r*.csv files are the other arm
REM of this experiment and MUST NOT be touched. 13_'s preflight asserts
REM resolve_streamer_geometry() == (25, 0) and would refuse to run here
REM anyway; this one asserts the OPPOSITE, so an operator cannot capture
REM the same arm twice and compare a run against itself.
REM
REM ========================================
REM THE AUDIO WILL SOUND SLIGHTLY WORSE. THAT IS EXPECTED.
REM ========================================
REM This arm runs the pre-20.5 decode: every chunk from a cold codec state,
REM masked by the Story 20.4 trim and seam blend. That is what shipped
REM before Story 20.5 and it is the point of the control. Do NOT report the
REM seams as a regression - we are measuring LATENCY here, not audio.
REM
REM ========================================
REM WHAT TO DO IN EACH LAUNCH
REM ========================================
REM   1. Make sure a CLONED voice is the active profile so BASE is the
REM      resident model. SAME PROFILE as the 13_ capture.
REM
REM   2. *** WAIT FOR "Preparing TTS engine" TO DISAPPEAR. ***
REM      This is the single thing that spoiled two of ten generations in
REM      the 13_ capture. Priming holds the request semaphore; generating
REM      while the indicator is up measures QUEUEING, not first-forward,
REM      and puts 840-1,383 ms of somebody else's work inside the number.
REM      After every launch this script prints a CHECK line telling you
REM      whether that launch survived. If it says CONTAMINATED, the launch
REM      is spoiled - keep going, but wait longer on the next one.
REM
REM      ON LAUNCH 1 the wait is LONGER than usual. The kill switch moves
REM      decode_window_frames 25 -^> 30, which is one of the seven compile-
REM      cache key dimensions, so this arm pays its own cold compile.
REM
REM   3. Generate the LONG utterance. Text is in
REM      _bmad-output\implementation-artifacts\20-4-gui-utterances.txt
REM      SAME text as the 13_ capture. Let it FINISH PLAYING.
REM   4. Generate the SHORT utterance from the same file. Let it finish.
REM   5. Close the app with the X. MYVOICE_AUTO_QUIT_ON_CLOSE=1 is set
REM      below, so X really quits. Do NOT use Ctrl-C.
REM   6. The next launch starts automatically.
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
    pause
    exit /b 1
)

"%SCRIPT_DIR%\python310\python.exe" -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [WARNING] Dependencies not installed.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "OUTDIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

REM ===== THE VARIABLE UNDER TEST =====
REM Set for this shell, so it is inherited by every launch below AND by
REM the preflight, which therefore verifies the same environment the app
REM will actually see.
set "MYVOICE_CODEC_STATE_CACHE=0"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM Preflight 1: compile must be engaged, same as the other arm.
"%SCRIPT_DIR%\python310\python.exe" -c "import json,sys;d=json.load(open(r'%SCRIPT_DIR%\config\settings.json'));sys.exit(0 if d.get('tts_compile')=='auto' else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] config\settings.json does not have tts_compile set to auto.
    echo The two arms must share every setting except the kill switch.
    pause
    exit /b 1
)

REM Preflight 2: the committed constants are 25 + 5, same as the other arm.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import codec_token_streamer as c;sys.exit(0 if (c.DEFAULT_CHUNK_SIZE,c.DEFAULT_LOOKAHEAD)==(25,5) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] CodecTokenStreamer constants are not 25 + 5.
    echo Both arms must share the chunk size or this is not an A/B.
    pause
    exit /b 1
)

REM Preflight 3: THE OPPOSITE OF 13_. The kill switch must be live and the
REM geometry must resolve to (25, 5) - the pre-20.5 arm. Without this an
REM operator could capture the shipping arm twice under a killswitch
REM filename and compare a run against itself, which would silently
REM "prove" a null result.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import resolve_streamer_geometry as g;sys.exit(0 if g()==(25,5) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] Resolved streamer geometry is not 25 + 5.
    echo The kill switch is NOT taking effect, so this would capture the
    echo SHIPPING arm under a kill-switch filename - a run compared against
    echo itself. Check that MYVOICE_CODEC_STATE_CACHE is not being
    echo overridden by a user or machine environment variable.
    pause
    exit /b 1
)

REM Provenance: record the geometry actually resolved in this environment,
REM next to the CSVs it describes. 20-6-compare-arms.py reads this and
REM refuses to run if it disagrees with the arm declared on its command
REM line - so the comparison cannot be made against a mis-labelled arm.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys,json,os,time;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import resolve_streamer_geometry as g;cs,la=g();json.dump({'arm':'kill-switch (pre-20.5 geometry)','env_MYVOICE_CODEC_STATE_CACHE':os.environ.get('MYVOICE_CODEC_STATE_CACHE'),'resolved_chunk_size':cs,'resolved_lookahead':la,'decode_window_frames':cs+la,'captured_utc':time.strftime('%%Y-%%m-%%dT%%H:%%M:%%SZ',time.gmtime()),'glob':'20-6-killswitch-r*.csv'},open(r'%OUTDIR%\20-6-killswitch-manifest.json','w'),indent=2)"
if errorlevel 1 (
    echo [WARNING] Could not write the capture manifest. The comparison will
    echo fall back to the arm declared on its command line.
)

set "TOTAL_RUNS=6"

echo.
echo ========================================
echo Story 20.6 follow-up - KILL-SWITCH BASELINE, %TOTAL_RUNS% launches
echo ========================================
echo.
echo Kill switch CONFIRMED live: MYVOICE_CODEC_STATE_CACHE=0
echo   geometry resolves to chunk_size 25, lookahead 5 - the PRE-20.5 arm
echo   decode_window_frames 30, so launch 1 pays its own cold compile
echo.
echo This is arm A. Arm B is already captured in 20-6-gui-r0N.csv and is
echo NOT touched by this run.
echo.
echo THE AUDIO WILL HAVE THE OLD CHUNK SEAMS. That is the control working,
echo not a regression. We are measuring latency here.
echo.
echo ****************************************************************
echo ** WAIT FOR "Preparing TTS engine" TO CLEAR BEFORE GENERATING **
echo ** Two of ten generations were lost to this last time.         **
echo ** After each launch this script tells you whether that        **
echo ** launch survived. Launch 1 will take noticeably longer.      **
echo ****************************************************************
echo.
echo Per launch:
echo   - active profile must be the SAME CLONED voice as the 13_ capture
echo   - WAIT for "Preparing TTS engine" to clear
echo   - generate the LONG utterance, LET IT FINISH PLAYING
echo   - generate the SHORT utterance, LET IT FINISH PLAYING
echo   - close with the X
echo.
echo Utterance texts - use the SAME ones as the 13_ capture:
echo   %OUTDIR%\20-4-gui-utterances.txt
echo.
echo Output CSVs:
echo   %OUTDIR%\20-6-killswitch-r01.csv ... r0%TOTAL_RUNS%.csv
echo.
echo Press a key to begin.
pause >nul

set /a RUN=1
set /a SPOILED=0

:RUN_LOOP
set "CSVNAME=20-6-killswitch-r0!RUN!.csv"
set "CSV=%OUTDIR%\!CSVNAME!"
set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=!CSV!"

echo.
echo ========================================
echo LAUNCH !RUN! of %TOTAL_RUNS%   [kill-switch arm]
echo ========================================
echo CSV: !CSV!
echo.
if !RUN! equ 1 echo THIS IS THE COLD-KEY LAUNCH - expect a LONG "Preparing TTS engine".
echo WAIT for "Preparing TTS engine" to clear. Then LONG, let it finish;
echo SHORT, let it finish; close with the X.
echo.

REM Run main.py directly - NOT as a module - to preserve the
REM torch-before-PyQt6 DLL-ordering invariant.
"%SCRIPT_DIR%\python310\python.exe" "%SCRIPT_DIR%\src\myvoice\main.py"
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] Launch !RUN! exited with code !EXIT_CODE! - check logs\myvoice.log

REM Per-launch contamination check. The operator finds out NOW, not after
REM all six launches are spent.
"%SCRIPT_DIR%\python310\python.exe" "%OUTDIR%\20-6-compare-arms.py" --check "!CSVNAME!"
if errorlevel 2 set /a SPOILED+=1

set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto RUN_LOOP

echo.
echo ========================================
echo All %TOTAL_RUNS% launches complete
echo ========================================
echo.
if !SPOILED! gtr 0 echo [NOTE] !SPOILED! launch^(es^) flagged as semaphore-contaminated. The
if !SPOILED! gtr 0 echo comparison excludes those generations and names them.
echo.
echo Geometry telemetry from the most recent log - every TRUE_STREAM line
echo should read lookahead=5 with carries_codec_state=False, and the codec
echo state cache should report itself OFF:
echo.
findstr /C:"TRUE_STREAM geometry" /C:"codec state cache" /C:"decode_window_frames" "%SCRIPT_DIR%\logs\myvoice.log" 2>nul | more
echo.
echo ========================================
echo THE COMPARISON
echo ========================================
"%SCRIPT_DIR%\python310\python.exe" "%OUTDIR%\20-6-compare-arms.py"
echo.
echo Hand the block above back to the session so it lands in the Story 20.6
echo evidence file, section 11.
echo.

pause
exit /b 0
