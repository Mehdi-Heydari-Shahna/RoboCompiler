@echo off
cd /d "%~dp0"
python run_comparison.py --profile full --native required --out results_local
set "RUN_ERROR=%ERRORLEVEL%"
if not "%RUN_ERROR%"=="0" echo Benchmark stopped. See the error above and README.md.
exit /b %RUN_ERROR%
