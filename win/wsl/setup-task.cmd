@echo off
rem setup-task.cmd - register logon task for wsl-fix-net.cmd
setlocal EnableExtensions

net session >nul 2>&1
if errorlevel 1 (
    echo [info] need admin, relaunching...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -Wait"
    exit /b 0
)

set "TASKNAME=WslFixNet"
set "SCRIPT=%~dp0wsl-fix-net.cmd"

echo ======================================================================
echo   register task: %TASKNAME%
echo   script       : %SCRIPT%
echo   trigger      : ONLOGON + 20s delay (current user, highest)
echo ======================================================================
echo.

schtasks /Query /TN "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [1/3] delete existing %TASKNAME% ...
    schtasks /Delete /TN "%TASKNAME%" /F >nul 2>&1
)

echo [2/3] create %TASKNAME% ...
rem WSL distros are per-user: do NOT use /RU SYSTEM
rem /DELAY format is mmmm:ss (minutes:seconds), NOT hh:mm:ss
schtasks /Create /TN "%TASKNAME%" /TR "\"%SCRIPT%\" nopause" /SC ONLOGON /DELAY 0000:20 /RL HIGHEST /F
if not "%errorlevel%"=="0" (
    echo [error] create task failed
    pause
    exit /b 1
)

echo [3/3] query ...
schtasks /Query /TN "%TASKNAME%" /FO LIST /V
echo.
echo done. recommended: run "%SCRIPT%" once now
echo.
pause
endlocal