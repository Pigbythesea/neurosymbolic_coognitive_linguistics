@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "ANALYSIS_UV=%USERPROFILE%\.local\bin\uv.exe"
set "ANALYSIS_PYTHON=%CD%\.venv-analysis\Scripts\python.exe"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_PYTHON_DOWNLOADS=never"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "VIRTUAL_ENV="

if /I "%~1"=="--check" goto check
if not "%~1"=="" goto usage
if not exist "%ANALYSIS_UV%" goto missing_uv
if not exist ".venv\Scripts\python.exe" goto missing_python

if exist "%ANALYSIS_PYTHON%" goto install
if exist ".venv-analysis" goto incomplete_environment
"%ANALYSIS_UV%" venv --no-config --python "%CD%\.venv\Scripts\python.exe" .venv-analysis
if errorlevel 1 goto failed

:install
echo Installing PyTorch CPU for local analysis development.
echo This step downloads packages; the annotation environment is separate.
"%ANALYSIS_UV%" pip install --no-config --only-binary :all: --python "%ANALYSIS_PYTHON%" --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
if errorlevel 1 goto failed
"%ANALYSIS_UV%" pip install --no-config --only-binary :all: --python "%ANALYSIS_PYTHON%" --index-url https://pypi.org/simple -r requirements\analysis.txt
if errorlevel 1 goto failed
"%ANALYSIS_UV%" pip check --no-config --python "%ANALYSIS_PYTHON%"
if errorlevel 1 goto failed

:check
if not exist "%ANALYSIS_PYTHON%" goto environment_not_ready
"%ANALYSIS_PYTHON%" -B scripts\check_analysis_environment.py
if errorlevel 1 goto failed
popd
exit /b 0

:missing_uv
echo STOP: uv.exe was not found at "%ANALYSIS_UV%".
goto failed
:missing_python
echo STOP: the existing Python interpreter .venv\Scripts\python.exe is missing.
goto failed
:incomplete_environment
echo STOP: .venv-analysis exists without its Python executable. No files were removed.
goto failed
:environment_not_ready
echo ANALYSIS ENVIRONMENT NOT READY: .venv-analysis has not been installed.
echo Run: call scripts\setup_analysis.cmd
popd
exit /b 2
:usage
echo Usage: call scripts\setup_analysis.cmd [--check]
popd
exit /b 2
:failed
echo ANALYSIS SETUP STOPPED. Paste the error above; no later step was executed.
popd
exit /b 1
