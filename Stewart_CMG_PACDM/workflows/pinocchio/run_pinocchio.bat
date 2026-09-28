@echo off
setlocal
cd /d "%~dp0"
call conda run --no-capture-output -n stewart_pinocchio python run_pinocchio.py %*
set "result=%ERRORLEVEL%"
if not "%result%"=="0" echo Run failed. Read the error above and README.md.
exit /b %result%
