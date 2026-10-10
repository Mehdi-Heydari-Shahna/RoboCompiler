@echo off
setlocal
python "%~dp0launch.py" %*
exit /b %errorlevel%
