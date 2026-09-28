@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python run_validation.py --render
if errorlevel 1 (
  echo Validation failed. Read results\validation.json and the console error.
  pause
  exit /b 1
)
echo Open results\report.html and results\pinocchio_validation.mp4
pause
