@echo off
cd /d "%~dp0"
python -c "import numpy, scipy, pinocchio as pin; print('Pinocchio',pin.__version__)"
if errorlevel 1 exit /b 1
python run_franka.py --profile full --stages native --native required --out results_native
pause
