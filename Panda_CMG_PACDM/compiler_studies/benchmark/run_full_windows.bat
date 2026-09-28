@echo off
cd /d "%~dp0"
python run_franka.py --profile full --native required --out results_local
pause
