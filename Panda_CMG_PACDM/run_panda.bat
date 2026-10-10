@echo off
setlocal
cd /d "%~dp0"
call conda activate panda_cmg_pacdm
if errorlevel 1 exit /b %errorlevel%
set OPENBLAS_NUM_THREADS=1
python run_panda.py --render
pause
