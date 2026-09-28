@echo off
cd /d "%~dp0"
python run_stewart.py --profile full --native off --out results_local
if errorlevel 1 exit /b 1
