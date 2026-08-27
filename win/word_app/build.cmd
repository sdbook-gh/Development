@echo off
setlocal
cd /d "%~dp0"

rem =======================================================
rem WordEmbedDemo build script (pure ASCII, no codepage issues)
rem   1) Search csc.exe in the .NET Framework directory (Windows 10 ships it).
rem   2) Fallback: search msbuild in common VS install paths.
rem   3) If neither found, show clear guidance.
rem =======================================================

set "PROJ_DIR=WordEmbedDemo"

rem ---- try to locate msbuild.exe -------------------------
set "MSBUILD="
rem Probe common VS install paths directly (no for /r full recursion).
for %%D in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
    "D:\Program Files\Microsoft Visual Studio\2022\Community"
    "D:\Program Files\Microsoft Visual Studio\2022\Professional"
    "D:\Program Files\Microsoft Visual Studio\2022\Enterprise"
) do (
    if exist "%%~D\MSBuild\Current\Bin\msbuild.exe" set "MSBUILD=%%~D\MSBuild\Current\Bin\msbuild.exe"
)
where msbuild >nul 2>nul
if "%ERRORLEVEL%"=="0" set "MSBUILD=msbuild"

rem ---- try to locate csc.exe (shipped in .NET Framework dir) -----
rem Windows 10 ships .NET Framework 4.x with csc.exe (C#5 compiler),
rem use it directly; no Visual Studio needed.
set "CSC="
where csc >nul 2>nul
if "%ERRORLEVEL%"=="0" set "CSC=csc"
if not defined CSC (
    rem 64-bit first, then 32-bit (v4.0.30319 = .NET Framework 4.x compiler dir)
    for %%d in (
        "%systemroot%\Microsoft.NET\Framework64\v4.0.30319"
        "%systemroot%\Microsoft.NET\Framework\v4.0.30319"
    ) do (
        if exist "%%~d\csc.exe" if not defined CSC set "CSC=%%~d\csc.exe"
    )
)

echo.
set "OUT=%PROJ_DIR%\bin_csc"

rem ---- Primary: use the .NET Framework csc.exe bundled with Windows 10
if defined CSC (
    echo [build] using CSC: %CSC%
    if not exist "%OUT%" mkdir "%OUT%"
    "%CSC%" /nologo /target:winexe /win32manifest:"%PROJ_DIR%\app.manifest" ^
        /r:System.dll /r:System.Core.dll /r:System.Drawing.dll /r:System.Windows.Forms.dll ^
        /out:"%OUT%\WordEmbedDemo.exe" ^
        "%PROJ_DIR%\NativeMethods.cs" "%PROJ_DIR%\WordProcessHost.cs" "%PROJ_DIR%\MainForm.cs" "%PROJ_DIR%\Program.cs" "%PROJ_DIR%\Properties\AssemblyInfo.cs"
    if "%ERRORLEVEL%"=="0" (
        echo.
        echo [OK] compiled: %cd%\%OUT%\WordEmbedDemo.exe
        echo [OK] run it:  "%cd%\%OUT%\WordEmbedDemo.exe"
    ) else (
        echo [FAILED] see compiler output above.
    )
    goto :done
)

rem ---- Fallback: MSBuild (requires VS Build Tools)
if defined MSBUILD (
    echo [build] using MSBuild: %MSBUILD%
    "%MSBUILD%" "%PROJ_DIR%\WordEmbedDemo.csproj" /p:Configuration=Release /nologo
    if "%ERRORLEVEL%"=="0" (
        echo.
        echo [OK] BUILD OK: %cd%\%PROJ_DIR%\bin\Release\WordEmbedDemo.exe
        echo [OK] run it:  "%cd%\%PROJ_DIR%\bin\Release\WordEmbedDemo.exe"
    ) else (
        echo [FAILED] MSBuild returned code %ERRORLEVEL%.
    )
    goto :done
)

:no_tool
echo.
echo [ERROR] csc.exe was not found. It should ship with .NET Framework 4.x:
echo         %systemroot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
echo.
echo To compile this project:
echo   A) Install/repair ".NET Framework 4.8 Developer Pack" (includes csc.exe),
echo      then re-run build.cmd.
echo   B) Install Microsoft Visual Studio Build Tools (provides msbuild.exe).
echo   C) Run the app directly from the existing exe:
echo        "%cd%\%PROJ_DIR%\bin_test\WordEmbedDemo.exe"
echo      (prebuilt binary still works, but does NOT include the latest fix)
echo.

:done
echo.
endlocal
timeout /t 5 /nobreak >nul 2>nul