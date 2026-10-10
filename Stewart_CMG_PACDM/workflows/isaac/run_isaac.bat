@echo off
setlocal
cd /d "%~dp0"
if not defined ISAAC_SIM_PATH goto pipenv
if not exist "%ISAAC_SIM_PATH%\python.bat" (
  echo ISAAC_SIM_PATH must be the Isaac Sim folder containing python.bat.
  exit /b 2
)
call "%ISAAC_SIM_PATH%\python.bat" "%~dp0run_validation.py" %*
exit /b %errorlevel%
:pipenv
python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('isaacsim') else 1)"
if errorlevel 1 (
  echo Activate an Isaac Sim Python environment, or set ISAAC_SIM_PATH to its standalone installation folder.
  exit /b 2
)
python "%~dp0run_validation.py" %*
exit /b %errorlevel%
