@echo off
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python run_pinocchio.py --render --workers 3
if errorlevel 1 (
  echo Validation failed. Inspect results_pinocchio\validation.json.
  exit /b 1
)
echo Done. Open results_pinocchio\report.html or results_pinocchio\Go2_Pinocchio.mp4.
