@echo off
cd /d "%~dp0"
set "OPENBLAS_NUM_THREADS=1"
python run_kangaroo_v20.py
set "KANGAROO_RUN_STATUS=%ERRORLEVEL%"
echo.
echo Full validation returned %KANGAROO_RUN_STATUS%. Read results\full_validation.json.
pause
exit /b %KANGAROO_RUN_STATUS%
