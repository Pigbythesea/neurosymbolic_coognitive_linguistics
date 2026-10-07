@echo off
setlocal
cd /d "%~dp0.."
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if not exist artifacts mkdir artifacts
echo Comparing compact traces and geometry against the real fitted audit.
echo Details: artifacts\trace-qualification-local.log
.venv-analysis\Scripts\python.exe -B -u scripts\verify_trace_tables.py --geometry > artifacts\trace-qualification-local.log 2>&1
if errorlevel 1 goto failed
echo TRACE AND GEOMETRY QUALIFIED. Preparing the integrated analysis package.
call scripts\prepare_analysis.cmd
if errorlevel 1 goto failed
endlocal
exit /b 0
:failed
echo STOPPED. Return artifacts\trace-qualification-local.log and artifacts\prepare-analysis.log if preparation started.
echo Do not transfer the previous ZIP or launch experiments.
endlocal
exit /b 1
