@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 2
call :run
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Kangaroo stopped. Read the failed check and the printed results path above.
popd
endlocal & exit /b %RC%

:run
if not defined CONDA_PREFIX (
    echo Open Miniforge Prompt and run: conda activate isaac61
    exit /b 2
)
python -c "import sys; from kangaroo_isaac.version import VERSION; print('Kangaroo Isaac Sim '+VERSION); sys.exit(0 if VERSION=='23.0.5' else 2)"
if errorlevel 1 exit /b %ERRORLEVEL%
python tools\verify_package.py
if errorlevel 1 exit /b %ERRORLEVEL%
python run_isaac.py --preflight
if errorlevel 1 exit /b %ERRORLEVEL%
python run_isaac.py --offline-test
if errorlevel 1 exit /b %ERRORLEVEL%
python run_isaac.py --verify-native --headless --no-visuals
exit /b %ERRORLEVEL%
