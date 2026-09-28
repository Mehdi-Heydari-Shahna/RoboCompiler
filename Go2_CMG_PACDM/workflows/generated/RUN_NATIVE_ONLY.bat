@echo off
cd /d "%~dp0"
set "OUT=results_native_%RANDOM%%RANDOM%"
echo Activate the Pinocchio environment before running this file.
python run_go2.py --profile full --stages native --native required --out "%OUT%"
if errorlevel 1 (echo Run failed. Read the error above.) else (echo Results are in %OUT%)
pause
