@echo off
setlocal
cd /d "%~dp0.."
.venv-analysis\Scripts\python.exe -B scripts\compile_semantics.py --require-complete
if errorlevel 1 exit /b 1
.venv-analysis\Scripts\python.exe -B scripts\verify_reviewed_semantics.py
if errorlevel 1 exit /b 1
.venv-analysis\Scripts\python.exe -B scripts\verify_encoding_support.py
if errorlevel 1 exit /b 1
.venv-analysis\Scripts\python.exe -B scripts\verify_analysis.py
if errorlevel 1 exit /b 1
.venv-analysis\Scripts\python.exe -B scripts\package_analysis.py
if errorlevel 1 exit /b 1
echo ANALYSIS PACKAGE READY. No upload or cluster command was executed.
endlocal
