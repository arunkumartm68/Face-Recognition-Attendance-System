@echo off
setlocal
cd /d "%~dp0"
title Face Recognition Attendance System - Setup
echo ===================================================
echo   Face Recognition Attendance System - Setup
echo ===================================================
echo.

if exist "venv\Scripts\python.exe" goto :install

:: Find Python 3.10 - 3.13 (numpy and dlib have no wheels for newer versions yet)
set "PY="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.10 - 3.13 was not found.
    echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run setup.bat again.
    goto :failed
)
echo Creating the virtual environment with: %PY%
%PY% -m venv venv || goto :failed

:install
echo Installing Python packages (this can take a few minutes)...
venv\Scripts\python.exe -m pip install --upgrade pip || goto :failed
venv\Scripts\python.exe -m pip install -r requirements.txt || goto :failed

echo.
echo Downloading the face recognition models (about 85 MB)...
venv\Scripts\python.exe download_models.py || goto :failed

echo.
echo Setup complete. Double-click start.bat to launch the system.
pause
exit /b 0

:failed
echo.
echo Setup did not finish - see the messages above.
pause
exit /b 1
