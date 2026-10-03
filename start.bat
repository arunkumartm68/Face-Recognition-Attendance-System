@echo off
setlocal
title Face Recognition Attendance System
echo ===================================================
echo   Face Recognition Attendance System Launcher
echo ===================================================
echo.

set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

if not exist "%PROJECT%\venv\Scripts\python.exe" (
    echo The Python environment was not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: dlib cannot open files in very long folder paths on Windows, so run the
:: project from a short virtual drive (X:) when that drive letter is free.
set "RUNDIR=%PROJECT%"
set "MAPPED="
if exist "X:\start.bat" if exist "X:\common.py" (
    rem X: still points at this project from an earlier run - remap it
    subst X: /D >nul 2>&1
)
if not exist X:\ (
    echo Mounting project to X: drive...
    subst X: "%PROJECT%" >nul 2>&1 && set "RUNDIR=X:" && set "MAPPED=1"
)
cd /d "%RUNDIR%\"

if not exist "data\data_dlib\shape_predictor_68_face_landmarks.dat" goto :models
if not exist "data\data_dlib\dlib_face_recognition_resnet_model_v1.dat" goto :models
goto :run

:models
echo Face recognition models not found - downloading them (about 85 MB)...
venv\Scripts\python.exe download_models.py || goto :failed

:run
:: Open the dashboard once the server has had a few seconds to start
echo Opening dashboard in your browser...
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start "" http://127.0.0.1:5000"

echo Starting the web server (press Ctrl+C to stop)...
venv\Scripts\python.exe app.py

if defined MAPPED subst X: /D >nul 2>&1
pause
exit /b 0

:failed
echo.
echo Could not download the face recognition models - check your internet connection.
if defined MAPPED subst X: /D >nul 2>&1
pause
exit /b 1
