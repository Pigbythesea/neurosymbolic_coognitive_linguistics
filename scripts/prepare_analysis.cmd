@echo off
setlocal
cd /d "%~dp0.."
if not exist artifacts mkdir artifacts
set "PREP_LOG=%CD%\artifacts\prepare-analysis.log"
> "%PREP_LOG%" echo Complete real-corpus analysis preparation
echo [1/6] Compiling accepted annotations. Details: %PREP_LOG%
.venv-analysis\Scripts\python.exe -B -u scripts\compile_semantics.py --require-complete >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo [2/6] Verifying complete accepted corpus and occurrence/query links.
.venv-analysis\Scripts\python.exe -B -u scripts\verify_reviewed_semantics.py >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo [3/6] Verifying matched content and temporal support across all stories.
.venv-analysis\Scripts\python.exe -B -u scripts\verify_encoding_support.py >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo [4/6] Verifying real-query numerics and experiment definitions.
.venv-analysis\Scripts\python.exe -B -u scripts\verify_analysis.py >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo [5/6] Verifying compute equivalence and the complete fit job inventory.
.venv-analysis\Scripts\python.exe -B -u scripts\verify_compute.py --device cpu >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo [6/6] Packaging and checking the complete bundle.
.venv-analysis\Scripts\python.exe -B -u scripts\package_analysis.py >> "%PREP_LOG%" 2>&1
if errorlevel 1 goto failed
echo ANALYSIS PACKAGE READY. No upload or cluster command was executed.
endlocal
exit /b 0

:failed
echo PREPARATION STOPPED. Read artifacts\prepare-analysis.log and return the error to the implementation thread.
echo The existing ZIP, if any, is not certified for this changed code.
endlocal
exit /b 1
