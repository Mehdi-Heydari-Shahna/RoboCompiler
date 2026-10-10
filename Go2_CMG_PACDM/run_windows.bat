@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python run_go2.py --render --workers 3 %*
exit /b %errorlevel%
