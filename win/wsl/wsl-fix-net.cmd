@echo off
rem ============================================================================
rem  wsl-fix-net.cmd  -  pin WSL2 LAN to 172.19.16.0/20
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

if /i "%~1"=="nopause" set "NOPAUSE=1"

set "IFACE=vEthernet (WSL)"
set "HOST_IP=172.19.16.1"
set "MASK=255.255.240.0"
set "GUEST_IP=172.19.16.2"
set "GUEST_MASK=20"
set "NAT_PREFIX=172.19.16.0/20"
set "NAT_NAME=WslFixNat17219"
set "WSL_DISTRO=Ubuntu-2204"
set "HOST_PS1=%~dp0wsl-host-net.ps1"
set "GUEST_WIN=%~dp0wsl-guest-fix.sh"

net session >nul 2>&1
if errorlevel 1 (
    echo [info] need admin, relaunching...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList 'nopause' -Verb RunAs -Wait"
    exit /b 0
)

echo.
echo ======================================================================
echo   WSL static LAN
echo   host %IFACE%  -^> %HOST_IP% / %MASK%
echo   guest eth0    -^> %GUEST_IP% / %GUEST_MASK%
echo   NAT prefix    -^> %NAT_PREFIX%
echo ======================================================================
echo.

echo [1/5] start WSL distro %WSL_DISTRO% ...
wsl -d %WSL_DISTRO% -u root -- echo wsl-alive >nul 2>&1
if errorlevel 1 (
    echo [error] cannot start distro %WSL_DISTRO%
    echo         run: wsl --list --verbose
    goto :end
)
ping 127.0.0.1 -n 3 >nul

echo [2/5] host IP + forwarding + NAT ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%HOST_PS1%" -Iface "%IFACE%" -HostIp "%HOST_IP%" -Mask "%MASK%" -NatPrefix "%NAT_PREFIX%" -NatName "%NAT_NAME%"
if errorlevel 1 (
    echo [error] host-side PowerShell failed
    goto :end
)
ping 127.0.0.1 -n 3 >nul

echo [3/5] resolve guest script path ...
set "GUEST_SCRIPT_SH="
for /f "usebackq delims=" %%I in (`wsl -d %WSL_DISTRO% -u root -- wslpath -a "%GUEST_WIN%"`) do set "GUEST_SCRIPT_SH=%%I"
if not defined GUEST_SCRIPT_SH (
    echo [error] wslpath failed for "%GUEST_WIN%"
    goto :end
)
echo         %GUEST_SCRIPT_SH%

echo [4/5] configure guest eth0 ...
wsl -d %WSL_DISTRO% -u root -- bash "%GUEST_SCRIPT_SH%"
if errorlevel 1 (
    echo [warn] guest script failed
)

echo [5/5] verify ...
netsh interface ipv4 show config name="%IFACE%"
wsl -d %WSL_DISTRO% -u root -- ip -4 addr show dev eth0
wsl -d %WSL_DISTRO% -u root -- ip -4 route show default
echo.
echo ping gateway:
wsl -d %WSL_DISTRO% -u root -- ping -c 2 -W 2 %HOST_IP%
echo ping 8.8.8.8:
wsl -d %WSL_DISTRO% -u root -- ping -c 2 -W 2 8.8.8.8

:end
echo.
if not defined NOPAUSE pause
endlocal