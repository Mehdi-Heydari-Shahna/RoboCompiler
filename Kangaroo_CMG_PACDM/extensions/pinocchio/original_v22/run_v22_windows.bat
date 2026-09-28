@echo off
setlocal
cd /d "%~dp0"
set "OPENBLAS_NUM_THREADS=1"
python run_kangaroo_v22.py %*
if errorlevel 1 (
  echo Validation failed. Read the failed gates in results.
  exit /b 1
)
echo All required model validation gates passed.
endlocal
