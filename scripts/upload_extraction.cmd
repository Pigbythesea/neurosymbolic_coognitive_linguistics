@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "SETUP_ARGUMENT="
if /I "%~1"=="--source-only" set "SETUP_ARGUMENT= --source-only"
".venv\Scripts\python.exe" -B scripts\validate_extraction.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -B scripts\package_extraction.py
if errorlevel 1 exit /b 1
scp artifacts\extraction-source.zip artifacts\install_extraction_bundle.py artifacts\setup_extraction.sbatch zzhan330@login.arch.jhu.edu:/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/artifacts/
if errorlevel 1 exit /b 1
echo.
echo Transfer complete. In your EXISTING SSH session, submit:
echo sbatch /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/artifacts/setup_extraction.sbatch%SETUP_ARGUMENT%
echo.
if /I "%~1"=="--source-only" (
    echo The job updates source on a CPU compute node, using the existing environment and models.
) else (
    echo The job installs and downloads on a CPU compute node.
)
echo No remote command was executed by this upload script.
endlocal
