@echo off
cd /d "%~dp0"
python run_stewart.py --render %*
exit /b %ERRORLEVEL%

