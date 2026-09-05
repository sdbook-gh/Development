@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "PYTHONIOENCODING=utf-8"

:: ============================================
:: Python embeddable-zip installer (Windows)
:: Downloads the official "Windows embeddable package"
:: (a plain zip), extracts it into the install dir,
:: enables site-packages, bootstraps pip, supplements
:: venv+ensurepip from the official NuGet package, and
:: updates PATH so it takes effect immediately.
::
:: Usage: win_install.cmd [install_dir] [python_version]
:: Example: win_install.cmd D:\Python 3.12.5
::
:: NOTE (embeddable package limitations):
::   - no tkinter / IDLE
::   - pip is bootstrapped via get-pip.py (step 7)
::   - venv + ensurepip are supplemented from the official
::     NuGet package (step 6), so "python -m venv" works
::   - step 5 writes sitecustomize.py so venvs keep real
::     isolation instead of inheriting the base python paths
:: ============================================

:: Default config
set "DEFAULT_INSTALL_DIR=C:\Python312"
set "DEFAULT_VERSION=3.12.5"

:: Parse args
set "INSTALL_DIR=%~1"
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%DEFAULT_INSTALL_DIR%"

set "PY_VERSION=%~2"
if "%PY_VERSION%"=="" set "PY_VERSION=%DEFAULT_VERSION%"

:: Extract short version for file naming (e.g. 3.12.5 -> 312)
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do set "VER_SHORT=%%a%%b"

set "PYTHON_HOME=%INSTALL_DIR%"
set "PYTHON_SCRIPTS=%INSTALL_DIR%\Scripts"

echo ============================================
echo   Python %PY_VERSION% embeddable-zip installer
echo   Install dir: %INSTALL_DIR%
echo ============================================
echo.

:: Detect existing install (re-run is safe: download/extract skipped)
set "ALREADY_INSTALLED=0"
if exist "%INSTALL_DIR%\python.exe" set "ALREADY_INSTALLED=1"

:: [1/9] Create install dir
if "%ALREADY_INSTALLED%"=="1" (
    echo [1/9] python.exe already present, keeping existing install
) else if not exist "%INSTALL_DIR%" (
    echo [1/9] Creating install dir...
    mkdir "%INSTALL_DIR%" >nul 2>&1
) else (
    echo [1/9] Install dir already exists
)

:: Download link / filename
set "ZIP_NAME=python-%PY_VERSION%-embed-amd64.zip"
set "DOWNLOAD_URL=https://www.python.org/ftp/python/%PY_VERSION%/%ZIP_NAME%"
set "ZIP_PATH=%TEMP%\%ZIP_NAME%"

:: [2/9] Download zip
if "%ALREADY_INSTALLED%"=="1" (
    echo [2/9] Already installed, skipping download
) else if exist "%ZIP_PATH%" (
    echo [2/9] Zip already downloaded, skipping: %ZIP_PATH%
) else (
    echo [2/9] Downloading Python %PY_VERSION% embeddable zip...
    echo       Source: %DOWNLOAD_URL%
    powershell -Command "try { Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%ZIP_PATH%' -UseBasicParsing } catch { exit 1 }"
    if !errorlevel! neq 0 (
        echo [ERROR] Download failed. Check version or network.
        echo         You can manually place the zip at: %ZIP_PATH%
        pause
        exit /b 1
    )
    echo       Download finished
)

:: [3/9] Extract zip into install dir
if "%ALREADY_INSTALLED%"=="1" (
    echo [3/9] Already installed, skipping extraction
) else (
    echo [3/9] Extracting zip to %INSTALL_DIR% ...
    powershell -NoProfile -Command "try { Expand-Archive -LiteralPath '%ZIP_PATH%' -DestinationPath '%INSTALL_DIR%' -Force } catch { exit 1 }"
    if !errorlevel! neq 0 (
        echo [ERROR] Extraction failed.
        pause
        exit /b 1
    )
)

:: [4/9] Rewrite pythonXXX._pth: enable site-packages + Lib
:: The embeddable zip ships with site-packages disabled and no pip;
:: "Lib" is added so the supplemented venv/ensurepip can be imported.
:: IMPORTANT: do NOT add an explicit "Lib\site-packages" line. The ._pth
:: is also inherited by python.exe inside venvs, so a base site-packages
:: entry would leak in there: ensurepip would then find the base pip and
:: skip seeding, leaving fresh venvs without their own pip. site.py adds
:: the base site-packages by itself as soon as that directory exists.
echo [4/9] Enabling site-packages + Lib in python%VER_SHORT%._pth ...
> "%INSTALL_DIR%\python%VER_SHORT%._pth" (
    echo python%VER_SHORT%.zip
    echo .
    echo Lib
    echo import site
)

:: [5/9] sitecustomize.py: put the active prefix' site-packages first
:: The ._pth entries (base root + base Lib) are inherited by venv pythons,
:: so without this a package installed inside a venv could be shadowed by
:: a same-named file living next to the base interpreter.
:: No effect when the base python itself runs.
:: NOTE: the python source below must not contain a '!' character, because
:: this script runs with EnableDelayedExpansion and cmd would eat it.
echo [5/9] Writing sitecustomize.py for venv isolation ...
powershell -NoProfile -Command ^
  "[IO.File]::WriteAllLines('%PYTHON_HOME%\sitecustomize.py', @('import sys, site','','if sys.prefix == sys.base_prefix:','    pass','else:','    front = [p for p in site.getsitepackages() if p.startswith(sys.prefix)]','    if front:','        sys.path[:] = front + [p for p in sys.path if p not in front]'))"
if !errorlevel! neq 0 echo [WARN] sitecustomize.py write failed, venv isolation may be incomplete
"%PYTHON_HOME%\python.exe" -c "import sitecustomize" >nul 2>&1
if !errorlevel! neq 0 echo [WARN] sitecustomize.py is not valid python, venv isolation may be incomplete

:: [6/9] Supplement venv + ensurepip (stripped from the embeddable zip;
:: files are taken from the official NuGet package of the same version)
if exist "%INSTALL_DIR%\Lib\venv\__init__.py" (
    echo [6/9] venv already present, skipping supplement
) else (
    set "NUGET_ZIP=%TEMP%\python-%PY_VERSION%-nuget.zip"
    set "NUGET_DIR=%TEMP%\python-%PY_VERSION%-nuget"
    if not exist "!NUGET_ZIP!" (
        echo       Downloading NuGet package python %PY_VERSION% ...
        powershell -Command "try { Invoke-WebRequest -Uri 'https://www.nuget.org/api/v2/package/python/%PY_VERSION%/' -OutFile '!NUGET_ZIP!' -UseBasicParsing } catch { exit 1 }"
        if !errorlevel! neq 0 (
            echo [WARN] NuGet download failed, skipping venv supplement
            echo        python + pip still work, only "python -m venv" is unavailable
            goto pipstep
        )
    )
    echo [6/9] Extracting venv + ensurepip from NuGet package ...
    powershell -NoProfile -Command "try { Expand-Archive -LiteralPath '!NUGET_ZIP!' -DestinationPath '!NUGET_DIR!' -Force } catch { exit 1 }"
    if !errorlevel! neq 0 (
        echo [WARN] NuGet extraction failed, skipping venv supplement
        goto pipstep
    )
    if not exist "!NUGET_DIR!\tools\Lib\venv\__init__.py" (
        echo [WARN] venv not found in NuGet package, skipping supplement
        goto pipstep
    )
    xcopy /e /i /y "!NUGET_DIR!\tools\Lib\venv" "%INSTALL_DIR%\Lib\venv" >nul
    xcopy /e /i /y "!NUGET_DIR!\tools\Lib\ensurepip" "%INSTALL_DIR%\Lib\ensurepip" >nul
    echo       venv + ensurepip supplemented
)

:pipstep
:: [7/9] Bootstrap pip via get-pip.py
if exist "%PYTHON_SCRIPTS%\pip.exe" (
    echo [7/9] pip already installed, skipping bootstrap
) else (
    set "GETPIP_PATH=%TEMP%\get-pip.py"
    if not exist "!GETPIP_PATH!" (
        echo       Downloading get-pip.py ...
        powershell -Command "try { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '!GETPIP_PATH!' -UseBasicParsing } catch { exit 1 }"
        if !errorlevel! neq 0 (
            echo [WARN] get-pip.py download failed, skipping pip bootstrap
            echo        Later you can run: python %%TEMP%%\get-pip.py
            goto verify
        )
    )
    echo [7/9] Bootstrapping pip ...
    "%PYTHON_HOME%\python.exe" "!GETPIP_PATH!" --no-warn-script-location
    if !errorlevel! neq 0 echo [WARN] pip bootstrap returned non-zero code
)

:verify
:: [8/9] Verify
echo [8/9] Verifying...
if not exist "%PYTHON_HOME%\python.exe" (
    echo [ERROR] python.exe not found, install may have failed
    pause
    exit /b 1
)
echo       Python installed:
"%PYTHON_HOME%\python.exe" --version
if exist "%PYTHON_SCRIPTS%\pip.exe" (
    "%PYTHON_SCRIPTS%\pip.exe" --version
) else (
    echo       pip NOT installed - see WARN above
)
if not exist "%PYTHON_HOME%\Lib\venv\__init__.py" (
    echo       venv NOT supplemented - see WARN above
) else (
    "%PYTHON_HOME%\python.exe" -c "import venv, ensurepip" >nul 2>&1
    if !errorlevel! equ 0 echo       venv available: python -m venv envname
)
if not exist "%PYTHON_HOME%\sitecustomize.py" (
    echo       sitecustomize.py missing - see WARN above
)

:: [9/9] Set USER PATH (prepend) and PYTHONIOENCODING
echo [9/9] Setting environment variables...
powershell -NoProfile -Command ^
  "$u=[Environment]::GetEnvironmentVariable('Path','User');" ^
  "$p='%PYTHON_HOME%'; $s='%PYTHON_SCRIPTS%';" ^
  "$items = $u -split ';' | Where-Object { $_.Trim() -ne $p -and $_.Trim() -ne $s -and $_ -ne '' };" ^
  "$new = ($p + ';' + $s + ';' + ($items -join ';')).TrimEnd(';');" ^
  "[Environment]::SetEnvironmentVariable('Path',$new,'User');" ^
  "Write-Host '       PATH updated: python dirs prepended to front'"
:: Persist PYTHONIOENCODING; setx also broadcasts WM_SETTINGCHANGE so that
:: Explorer refreshes and newly opened windows pick up the new PATH at once.
setx PYTHONIOENCODING utf-8 >nul

:finish
echo.
echo ============================================
echo   Installation finished!
echo   Python:  %PYTHON_HOME%
echo   Scripts: %PYTHON_SCRIPTS%
echo ============================================
echo.
echo NOTE: This writes to the CURRENT USER PATH.
echo DO NOT run as Administrator (it would write to the
echo Administrator account, invisible to your normal login).
echo NOTE: Embeddable package has NO tkinter/IDLE.
echo       venv + pip are supplemented by this script.
echo       To reinstall from scratch: delete the install
echo       dir and run this script again.
echo ============================================
echo.

:: Make PATH take effect IMMEDIATELY in the current window:
:: %vars% are expanded at parse time (while setlocal is still active), then
:: endlocal restores the caller's scope, and set writes the values into it.
endlocal & set "PATH=%PYTHON_HOME%;%PYTHON_SCRIPTS%;%PATH%" & set "PYTHONIOENCODING=utf-8"

:: Optional: cleanup downloaded files
:: del /f /q "%ZIP_PATH%" "%TEMP%\get-pip.py" "%TEMP%\python-%PY_VERSION%-nuget.zip" >nul 2>&1
:: rmdir /s /q "%TEMP%\python-%PY_VERSION%-nuget" >nul 2>&1

pause
