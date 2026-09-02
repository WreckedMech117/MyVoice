@echo off
REM ========================================
REM Story 20.8 AC #3 - NFR3 chunk-size audition (Commander solo)
REM ========================================
REM   reference arm = what ships today: chunk_size 25
REM   candidate arm = the committed retune: chunk_size 10
REM
REM ONE VARIABLE. Both arms carry codec state caching (Story 20.5), the
REM retired lookahead (Story 20.6) and the same gated 0-sample consumer
REM crossfade. The ONLY difference is how many pieces the stream is cut
REM into: 25-frame chunks -^> 10-frame chunks, about 2.5x as many joins.
REM
REM ========================================
REM THIS GEOMETRY FAILED THIS TEST BEFORE. THAT IS THE POINT.
REM ========================================
REM Story 20.4 auditioned chunk_size 10 and it FAILED twice - blocking seam
REM defects in rounds 2 and 3. It was reverted, and Story 20.4 SS17 named
REM the one thing that would reopen it: removing the seam residual at the
REM CAUSE rather than masking it with a blend.
REM
REM Story 20.5 did that - head NRMSE 0.406 -^> 0.0078, lag jitter 0 samples
REM on every seam, edge loss 555 -^> 0. Story 20.6 then retired the
REM lookahead. This round asks whether that was enough.
REM
REM So a blocking defect here is NOT a small setback. It would mean the
REM mechanism argument is wrong and the geometry question closes for good -
REM not retuned to 15, closed. Prediction P4 in the evidence file says so
REM in advance, which is the point of writing it down first.
REM
REM ========================================
REM BOTH FILES IN EVERY PAIR COME FROM ONE TALKER RUN
REM ========================================
REM Story 20.4 could not do this - its arms were different takes, and its
REM round 4 was unresolvable because the SAME configuration flagged
REM differently across two takes.
REM
REM Story 20.8 SS7 tested the assumption behind that instead of repeating
REM it: at a fixed seed, live cs25 / cs10 / cs7 runs emit BIT-IDENTICAL
REM token streams, and an offline re-chunk rendered through the real
REM decoder worker and the real consumer buffer is bit-for-bit what the
REM live run at that geometry produces. So the fixture captures ONE
REM generation per pair - in a process running at the CANDIDATE geometry,
REM because a build at a different chunk size draws a different stream -
REM and re-cuts it two ways. Wording, prosody, pauses and duration are
REM identical to the sample. Anything you hear is the cutting.
REM
REM ========================================
REM WHAT TO EXPECT
REM ========================================
REM 16 trials: the seven epic-standard utterances x two takes, plus TWO
REM byte-identical control trials (ctl-020). The controls are not a trick -
REM each is one arm rendered twice, asserted identical at generation time,
REM and they set the round's noise floor. If you express a preference on
REM one of them, the round is discarded and re-run rested; that decision is
REM already recorded so it cannot be argued about afterwards.
REM
REM "equivalent" is the PREDICTED answer on most trials, not a cop-out.
REM
REM BLOCKING: any chunk-boundary defect on a candidate trial that its
REM paired reference does not also carry. A defect on BOTH files of a pair
REM is upstream of the geometry - demonstrably, since they are the same
REM take - so it is recorded, not blocking.
REM
REM The 32-WAV fixture is generated ahead of time by
REM   20-8-regen-audition-fixture.py
REM under _bmad-output. If it is missing this script says so and stops - it
REM does NOT silently regenerate, because that takes GPU time.
REM
REM ~25 minutes.
REM
REM Usage:
REM   18_Story_20.8_AC3_Audition.bat        (L1 - Commander)
REM   18_Story_20.8_AC3_Audition.bat L2     (a second listener, same machine)
REM ========================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

setlocal EnableDelayedExpansion

set "LISTENER_ID=%~1"
if "%LISTENER_ID%"=="" set "LISTENER_ID=L1"
set "ROUND_ID=%~2"
if "%ROUND_ID%"=="" set "ROUND_ID=r1"

set "ARTIFACTS=%SCRIPT_DIR%\_bmad-output\implementation-artifacts"
set "FIXTURE=%ARTIFACTS%\20-8-perceptual-fixtures"

if not exist "%SCRIPT_DIR%\python310\python.exe" (
    echo [ERROR] Portable Python not found.
    echo Expected: %SCRIPT_DIR%\python310\python.exe
    pause
    exit /b 1
)

if not exist "%FIXTURE%\_perlistener_truthtable.json" (
    echo [ERROR] Audition fixture not found.
    echo Expected: %FIXTURE%\_perlistener_truthtable.json
    echo.
    echo Generate it first - this needs the GPU and takes a few minutes:
    echo   python310\python.exe %ARTIFACTS%\20-8-regen-audition-fixture.py
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%\src"

echo.
echo ========================================
echo Story 20.8 AC #3 audition - listener %LISTENER_ID%
echo ========================================
echo.
echo Use headphones. Normal Discord-call volume.
echo.
echo Both files in a pair are the SAME generation re-cut two ways. The
echo words, the timing and the delivery are identical. You are judging
echo SEAMS only - clicks, discontinuities, prosody that resets mid-phrase,
echo smeared consonants at a boundary.
echo.
echo One arm has about 2.5x as many joins as the other. Whether that is
echo audible now that each join carries real codec state is the whole
echo question - and the last time this geometry was auditioned, it was not
echo good enough.
echo.
echo "equivalent" is the PREDICTED answer here, not a cop-out. Two of the
echo sixteen trials are byte-identical on purpose.
echo.

"%SCRIPT_DIR%\python310\python.exe" "%ARTIFACTS%\20-8-l1-audition-helper.py" %LISTENER_ID% %ROUND_ID%
set "EXIT_CODE=!errorlevel!"

echo.
if !EXIT_CODE! neq 0 echo [WARNING] helper exited with code !EXIT_CODE!
echo.
echo Results CSV:
echo   %ARTIFACTS%\20-8-chunksize-audition.csv
echo.
echo Hand the verdict block above back to the session so it lands in the
echo Story 20.8 evidence file section 9.
echo.
echo THEN run the GUI capture - BOTH arms, same sitting:
echo   16_Story_20.8_AC3_GUI_Capture.bat    then    17_Story_20.8_CS25_Baseline.bat
echo.

pause
exit /b 0
