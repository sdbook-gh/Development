@echo off
rem uninstall-task.cmd - remove WslFixNet task
setlocal EnableExtensions

net session >nul 2>&1
if errorlevel 1 (
    echo [info] need admin, relaunching...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -Wait"
    exit /b 0
)

set "TASKNAME=WslFixNet"

echo ======================================================================
echo   uninstall %TASKNAME%
echo ======================================================================
echo.

schtasks /Query /TN "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [1/2] delete task %TASKNAME% ...
    schtasks /Delete /TN "%TASKNAME%" /F >nul 2>&1
    if not errorlevel 1 (
        echo         deleted
    ) else (
        echo         [error] delete failed
    )
) else (
    echo [1/2] task not found
)

echo [2/2] optional restore auto WSL network:
echo        powershell -NoProfile -Command "Get-NetNat -Name WslFixNat17219 -ErrorAction SilentlyContinue | Remove-NetNat -Confirm:$false"
echo        netsh interface ipv4 set address name="vEthernet (WSL)" dhcp
echo        wsl --shutdown
echo        then start WSL again
echo.
pause
endlocal