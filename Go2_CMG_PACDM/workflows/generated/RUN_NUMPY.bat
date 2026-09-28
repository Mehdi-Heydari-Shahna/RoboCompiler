@echo off
cd /d "%~dp0"
set "OUT=results_local_%RANDOM%%RANDOM%"
python run_go2.py --profile full --native off --out "%OUT%"
if errorlevel 1 (echo Run failed. Read the error above.) else (echo Results are in %OUT%)
pause
