@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
call conda run --no-capture-output -n kangaroo_pinocchio python run_pinocchio.py %*
set "result=%ERRORLEVEL%"
if not "%result%"=="0" echo Run failed. Read the error above, results\validation.json and README.md.
exit /b %result%
