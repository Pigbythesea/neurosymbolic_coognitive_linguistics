@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "UV_CACHE_DIR=%CD%\.uv-cache"
if not exist ".venv\Scripts\python.exe" (
    echo STOP: The project Python environment is missing.
    exit /b 1
)
".venv\Scripts\python.exe" -B -u scripts\annotate_corpus.py
set "ANNOTATION_EXIT=%ERRORLEVEL%"
if "%ANNOTATION_EXIT%"=="2" (
    echo Deferred stories need review. The run continued through independent stories; accepted annotations are saved.
    echo Run .venv\Scripts\python.exe scripts\annotation_status.py to see the report.
    exit /b 2
)
if not "%ANNOTATION_EXIT%"=="0" exit /b %ANNOTATION_EXIT%
".venv\Scripts\python.exe" -B -u scripts\audit_annotations.py
if errorlevel 1 exit /b 1
echo Default annotation pass and structural audit are complete. Human semantic review remains pending.
endlocal
