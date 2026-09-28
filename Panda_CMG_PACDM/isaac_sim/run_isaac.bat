@echo off
setlocal
cd /d "%~dp0"
python run_isaac.py %*
set RESULT=%ERRORLEVEL%
exit /b %RESULT%
