@echo off
REM ========================================
REM Story 20.8 AC #3 - GUI TTFA capture at chunk_size 25 (arm A, the control)
REM ========================================
REM WHY THIS EXISTS
REM
REM Story 20.6 SS12 established that a cross-session GUI comparison already
REM produced one false conclusion in this epic - the same code measured two
REM months apart read 46.66 vs 38.25 ms/frame, and the conclusion drawn from
REM it had to be retracted. So the cs10 capture in 16_ cannot be scored
REM against Story 20.6's cs25 figure. It needs a cs25 arm captured on the
REM SAME machine in the SAME sitting. This launcher is that arm.
REM
REM ========================================
REM HOW THE GEOMETRY IS FORCED BACK - AND WHY NOTHING IS EDITED
REM ========================================
REM After the retune the committed constant is 10, so this arm cannot just
REM launch the app. It goes through
REM
REM   _bmad-output\implementation-artifacts\20-8-run-myvoice-at-cs25.py
REM
REM which rebinds CodecTokenStreamer's chunk-size default IN-PROCESS and
REM then runs main.py. That is the same mechanism Story 20.1 SS5.1 used for
REM its sweep and Story 20.4 used for its round-4 fixture: exactly
REM equivalent to the module-constant edit the class docstring documents,
REM and it LEAVES NO SOURCE-TREE EDIT TO REVERT.
REM
REM That matters here more than usual. The alternative is asking an operator
REM to hand-edit a shipping constant between launches and hand-edit it back,
REM which is a way to ship the wrong number by accident.
REM
REM The runner writes 20-8-cs25-manifest.json recording the geometry the
REM process ACTUALLY resolved, and 20-8-compare-gui-arms.py refuses to score
REM if that disagrees with the declared arm. A flag is not provenance.
REM
REM ========================================
REM RUN THIS AFTER 16_, IN THE SAME SITTING
REM ========================================
REM   16_Story_20.8_AC3_GUI_Capture.bat  -^>  20-8-gui-r0N.csv   (arm B, cs10)
REM   17_Story_20.8_CS25_Baseline.bat    -^>  20-8-cs25-r0N.csv  (arm A, cs25)
REM
REM SAME cloned voice profile. SAME utterance texts. Back to back.
REM
REM ========================================
REM THIS ARM IS SLOWER TO FIRST AUDIO. THAT IS THE POINT.
REM ========================================
REM cs25 is what ships today: first emit waits 25 frames rather than 10, so
REM expect roughly 600 ms more before audio starts. Do NOT report that as a
REM regression - it is the quantity being measured.
REM
REM ========================================
REM WHAT TO DO IN EACH LAUNCH
REM ========================================
REM   1. Same CLONED voice profile as arm B.
REM   2. *** WAIT FOR "Preparing TTS engine" TO DISAPPEAR. ***
REM      ON LAUNCH 1 the wait is longer: this arm resolves
REM      decode_window_frames back to 25, a DIFFERENT compile-cache key
REM      from arm B's, so it pays its own one cold compile. Launch 1 is a
REM      declared throwaway.
REM   3. Generate the LONG utterance, LET IT FINISH PLAYING.
REM   4. Generate the SHORT utterance, LET IT FINISH PLAYING.
REM   5. Close with the X. Do NOT use Ctrl-C.
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "OUTDIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "RUNNER=%OUTDIR%\20-8-run-myvoice-at-cs25.py"
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

if not exist "%RUNNER%" (
    echo [ERROR] Reference-arm runner not found:
    echo   %RUNNER%
    pause
    exit /b 1
)

REM Preflight 1: compile engaged.
"%SCRIPT_DIR%\python310\python.exe" -c "import json,sys;d=json.load(open(r'%SCRIPT_DIR%\config\settings.json'));sys.exit(0 if d.get('tts_compile')=='auto' else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] config\settings.json does not have tts_compile set to auto.
    pause
    exit /b 1
)

REM Preflight 2: the OPPOSITE assertion to 16_. The committed constant must
REM be the retune, because this arm exists to compare AGAINST it. If the
REM committed value were already 25 there would be no experiment - both
REM launchers would capture the same arm.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import codec_token_streamer as c;sys.exit(0 if c.DEFAULT_CHUNK_SIZE==10 else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] The committed chunk size is not 10.
    echo This launcher forces the geometry back to 25 to act as the CONTROL
    echo for the retune. If the retune is not committed, both launchers
    echo would capture the same arm and the comparison would be a run
    echo against itself.
    pause
    exit /b 1
)

REM Preflight 3: the rebind actually takes, in a throwaway process, before
REM any launch is spent on it.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import codec_token_streamer as c;c.DEFAULT_CHUNK_SIZE=25;c.CodecTokenStreamer.__init__.__defaults__=(25,c.DEFAULT_LOOKAHEAD,c.DEFAULT_QUEUE_MAX_FACTOR,None);from myvoice.services.tts_streaming import resolve_streamer_geometry as g;sys.exit(0 if g()==(25,0) and c.CodecTokenStreamer().chunk_size==25 else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] The in-process chunk-size rebind did not take.
    echo Refusing to capture an arm whose geometry is not what it claims.
    pause
    exit /b 1
)

set "TOTAL_RUNS=5"

echo.
echo ========================================
echo Story 20.8 AC #3 - GUI capture, ARM A, %TOTAL_RUNS% launches
echo ========================================
echo.
echo Geometry forced to: chunk_size 25, lookahead RETIRED to 0.
echo   This is what ships today, on today's machine, in today's sitting.
echo   decode_window_frames resolves to 25 - a different cache key from
echo   arm B's 10, so launch 1 pays its own cold compile.
echo.
echo EXPECT THIS ARM TO BE SLOWER TO FIRST AUDIO by roughly 600 ms.
echo That is the measurement, not a fault.
echo.
echo Utterance texts - THE SAME ONES ARM B USED:
echo   %OUTDIR%\20-4-gui-utterances.txt
echo.
echo Output CSVs:
echo   %OUTDIR%\20-8-cs25-r01.csv ... r0%TOTAL_RUNS%.csv
echo.
echo Press a key to begin.
pause >nul

set /a RUN=1

:RUN_LOOP
set "CSV=%OUTDIR%\20-8-cs25-r0!RUN!.csv"
set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=!CSV!"

echo.
echo ========================================
echo ARM A - LAUNCH !RUN! of %TOTAL_RUNS%
echo ========================================
echo CSV: !CSV!
echo.
if !RUN! equ 1 echo THIS IS THE COLD-KEY LAUNCH - expect a long "Preparing TTS engine".
echo Reminder: wait for "Preparing TTS engine" to clear, generate LONG,
echo let it finish, generate SHORT, let it finish, then close with the X.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%RUNNER%"
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] Launch !RUN! exited with code !EXIT_CODE! - check logs\myvoice.log

"%SCRIPT_DIR%\python310\python.exe" "%OUTDIR%\20-8-compare-gui-arms.py" --check "20-8-cs25-r0!RUN!.csv"

set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto RUN_LOOP

echo.
echo ========================================
echo Both arms captured. Score them:
echo ========================================
echo.
echo   python310\python.exe %OUTDIR%\20-8-compare-gui-arms.py
echo.
echo That prints, per class, segment by segment, arm B minus arm A - plus
echo the per-frame talker cost for each arm and the cross-check that says
echo whether the saving really is the 15 fewer frames the geometry predicts.
echo.
echo Hand the CSVs and that output back to the session.
echo.
pause
