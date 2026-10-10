@echo off
setlocal DisableDelayedExpansion
if not defined CONDA_PREFIX (
    echo ERROR: Open Miniforge Prompt and run conda activate isaac61 first.
    exit /b 2
)
if not exist "%CONDA_PREFIX%\python.exe" (
    echo ERROR: The active environment has no Python interpreter.
    exit /b 2
)
if not exist "%~dp0tools\repair_miniforge.py" (
    echo ERROR: Extract the complete ZIP first. Keep this launcher with the project files.
    exit /b 2
)
"%CONDA_PREFIX%\python.exe" "%~dp0tools\repair_miniforge.py" %*
exit /b %errorlevel%
