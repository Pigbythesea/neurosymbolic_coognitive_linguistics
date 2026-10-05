@echo off
setlocal
cd /d "%~dp0.."
.venv-analysis\Scripts\python.exe -B scripts\package_analysis.py
if errorlevel 1 exit /b 1
scp artifacts\analysis-source.zip artifacts\install_analysis_bundle.py artifacts\setup_analysis.sbatch zzhan330@login.arch.jhu.edu:/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/artifacts/
if errorlevel 1 exit /b 1
echo Transfer complete. No remote command or job was executed.
echo In your existing cluster session, the setup submission is:
echo sbatch /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/artifacts/setup_analysis.sbatch
endlocal
