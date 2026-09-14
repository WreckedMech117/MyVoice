@echo off
REM ========================================
REM Story 20.8 AC #3 - GUI TTFA capture at the RETUNED geometry (arm B)
REM ========================================
REM WHY THIS EXISTS
REM
REM Phase 1 measured the chunk-size curve HEADLESSLY and found cs10 beats
REM cs25 by 645 ms. The harness reproduces everything up to and including
REM StreamingChunkBuffer.push but NOT the PyAudio device open, the Qt event
REM loop or the real AudioCoordinator. The shipped TTFA claim is a GUI-path
REM claim, so it needs a GUI-path measurement.
REM
REM ========================================
REM THIS IS ONE ARM OF TWO. RUN BOTH, IN ONE SITTING.
REM ========================================
REM   16_Story_20.8_AC3_GUI_Capture.bat  -^>  20-8-gui-r0N.csv   (arm B, cs10)
REM   17_Story_20.8_CS25_Baseline.bat    -^>  20-8-cs25-r0N.csv  (arm A, cs25)
REM
REM Story 20.6 SS12 is why the second arm is not optional: the same code
REM measured in two sessions two months apart differed by 46.66 vs 38.25
REM ms/frame, and that cross-session comparison produced a false conclusion
REM that had to be retracted. Quoting Story 20.6's cs25 GUI figure here
REM would repeat it exactly. Run 16_ and 17_ back to back, same machine,
REM same profile, same afternoon.
REM
REM Different glob on purpose. Each launcher's preflight asserts ITS OWN
REM geometry and refuses to run under the other, so an operator cannot
REM capture the same arm twice and compare a run against itself.
REM
REM ========================================
REM WHAT TO DO IN EACH LAUNCH
REM ========================================
REM   1. Make sure a CLONED voice is the active profile so BASE is the
REM      resident model. SAME PROFILE for all launches of BOTH arms.
REM
REM   2. *** WAIT FOR "Preparing TTS engine" TO DISAPPEAR. ***
REM      This is the single thing that spoiled two of ten generations in
REM      the Story 20.6 capture. Priming holds the request semaphore, so
REM      generating while the indicator is up measures QUEUEING, not
REM      first-forward, and puts 840-1,383 ms of somebody else's work
REM      inside the number. After every launch this script prints a CHECK
REM      line telling you whether that launch survived.
REM
REM      ON LAUNCH 1 the wait is LONGER. The retune moves
REM      decode_window_frames 25 -^> 10, a new compile-cache key, so this
REM      arm pays exactly ONE cold compile - about 19 s on the Phase 1
REM      measurements. Launch 1 is a declared throwaway; the analysis
REM      drops it.
REM
REM   3. Generate the LONG utterance. Text is in
REM      _bmad-output\implementation-artifacts\20-4-gui-utterances.txt
REM      SAME text every launch, both arms.
REM   4. LET IT FINISH PLAYING. Do not close or re-generate mid-playback -
REM      the producer emit/drain ratio needs the whole stream.
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
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "OUTDIR=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "MYVOICE_AUTO_QUIT_ON_CLOSE=1"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM Preflight 1: compile must be engaged or there is nothing to prime and
REM the whole measurement is moot.
"%SCRIPT_DIR%\python310\python.exe" -c "import json,sys;d=json.load(open(r'%SCRIPT_DIR%\config\settings.json'));sys.exit(0 if d.get('tts_compile')=='auto' else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] config\settings.json does not have tts_compile set to auto.
    echo Compile priming cannot engage, so this measurement would be meaningless.
    pause
    exit /b 1
)

REM Preflight 2: this is the CANDIDATE arm. The committed constant must be
REM the retune, and the lookahead constant must be untouched at 5 - Story
REM 20.8 changes chunk size only, and DEFAULT_LOOKAHEAD is the STATELESS
REM path's lookahead.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import codec_token_streamer as c;sys.exit(0 if (c.DEFAULT_CHUNK_SIZE,c.DEFAULT_LOOKAHEAD)==(10,5) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] CodecTokenStreamer constants are not 10 + 5.
    echo This launcher captures the RETUNED arm. If the retune is not
    echo committed there is nothing here to measure.
    pause
    exit /b 1
)

REM Preflight 3: the geometry actually RESOLVED for this process. This is
REM the check that matters - if MYVOICE_CODEC_STATE_CACHE is set to a
REM disabling value, the lookahead comes back and every number below would
REM describe a different build.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import resolve_streamer_geometry as g;sys.exit(0 if g()==(10,0) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] Resolved streamer geometry is not 10 + 0.
    echo Most likely MYVOICE_CODEC_STATE_CACHE is set to a disabling value.
    echo Clear it and re-run.
    pause
    exit /b 1
)

REM Provenance: record what this process resolved, so the comparison cannot
REM be scored against a mis-declared arm.
"%SCRIPT_DIR%\python310\python.exe" -c "import sys,json;sys.path.insert(0,r'%SCRIPT_DIR%\src');from myvoice.services.tts_streaming import resolve_streamer_geometry as g;cs,la=g();json.dump({'arm':'candidate','resolved_chunk_size':cs,'resolved_lookahead':la,'decode_window_frames':cs+la,'mechanism':'committed constant'},open(r'%OUTDIR%\20-8-gui-manifest.json','w'),indent=2)"

set "TOTAL_RUNS=5"

echo.
echo ========================================
echo Story 20.8 AC #3 - GUI capture, ARM B, %TOTAL_RUNS% launches
echo ========================================
echo.
echo Geometry confirmed: chunk_size 10, lookahead RETIRED to 0.
echo   decode_window_frames 25 -^> 10, a NEW compile-cache key, so launch 1
echo   pays exactly one cold compile - about 19 s.
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
echo the new decode-window cache key. Still do both generations.
echo.
echo Output CSVs:
echo   %OUTDIR%\20-8-gui-r01.csv ... r0%TOTAL_RUNS%.csv
echo.
echo AFTER THIS, RUN 17_Story_20.8_CS25_Baseline.bat IN THE SAME SITTING.
echo A cs10 capture on its own cannot be compared to anything.
echo.
echo Press a key to begin.
pause >nul

set /a RUN=1

:RUN_LOOP
set "CSV=%OUTDIR%\20-8-gui-r0!RUN!.csv"
set "MYVOICE_PROGRESSIVE_PLAYBACK_CSV=!CSV!"

echo.
echo ========================================
echo ARM B - LAUNCH !RUN! of %TOTAL_RUNS%
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

"%SCRIPT_DIR%\python310\python.exe" "%OUTDIR%\20-8-compare-gui-arms.py" --check "20-8-gui-r0!RUN!.csv"

set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto RUN_LOOP

echo.
echo ========================================
echo Arm B complete - %TOTAL_RUNS% launches
echo ========================================
echo.
echo Row counts:
set /a RUN=1
:COUNT_LOOP
set "CSV=%OUTDIR%\20-8-gui-r0!RUN!.csv"
if exist "!CSV!" (
    for /f %%c in ('find /c /v "" ^< "!CSV!"') do echo   r0!RUN!: %%c lines
) else (
    echo   r0!RUN!: MISSING
)
set /a RUN+=1
if !RUN! leq %TOTAL_RUNS% goto COUNT_LOOP

echo.
echo Compile telemetry - launch 1 should show a COLD compile for the new
echo decode-window key and launches 2-6 warm, and the TRUE_STREAM geometry
echo line should read chunk_size=10 lookahead=0:
echo.
findstr /C:"tts_compile_warmup_priming" /C:"Compile cache hit" /C:"warmup primed" /C:"decode_window_frames" /C:"TRUE_STREAM geometry" "%SCRIPT_DIR%\logs\myvoice.log" 2>nul | more
echo.
echo ========================================
echo NOW RUN 17_Story_20.8_CS25_Baseline.bat - the other arm, same sitting.
echo ========================================
echo.
pause
