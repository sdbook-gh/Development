@echo off
setlocal
cd /d "%~dp0"

rem =======================================================
rem WordEmbedDemo build script (pure ASCII, no codepage issues)
rem   1) Search msbuild in common VS install paths.
rem   2) Search csc in the .NET Framework directory.
rem   3) If neither found, show clear guidance.
rem =======================================================

set "PROJ_DIR=WordEmbedDemo"

rem ---- try to locate msbuild.exe -------------------------
set "MSBUILD="
for %%D in ( "C:\Program Files" "C:\Program Files (x86)" "D:\Program Files" "D:\Program Files (x86)" ) do (
    if exist "%%~D\Microsoft Visual Studio" (
        for /r "%%~D\Microsoft Visual Studio" %%F in (msbuild.exe) do (
            if not defined MSBUILD set "MSBUILD=%%F"
        )
    )
)
where msbuild >nul 2>nul
if "%ERRORLEVEL%"=="0" set "MSBUILD=msbuild"

rem ---- try to locate csc.exe (shipped in .NET Framework dir) -----
set "CSC="
where csc >nul 2>nul
if "%ERRORLEVEL%"=="0" set "CSC=csc"
if not defined CSC (
    rem .NET Framework ships one csc.exe per version, e.g. v4.0.30319\csc.exe
    rem (avoid for /r here: it can return a path that does not actually exist).
    for %%d in (
        "%systemroot%\Microsoft.NET\Framework\v4.0.30319"
        "%systemroot%\Microsoft.NET\Framework64\v4.0.30319"
        "%systemroot%\Microsoft.NET\Framework\v3.5"
        "%systemroot%\Microsoft.NET\Framework\v2.0.50727"
    ) do (
        if exist "%%~d\csc.exe" if not defined CSC set "CSC=%%~d\csc.exe"
    )
    if not defined CSC (
        rem last-resort: ask where to at least list a real csc.exe path
        for /f "tokens=* delims=" %%F in ('where /r "%systemroot%\Microsoft.NET\Framework" csc.exe 2^>nul') do (
            if not defined CSC set "CSC=%%F"
        )
    )
)

echo.
if defined MSBUILD (
    echo [build] using MSBuild: %MSBUILD%
    "%MSBUILD%" "%PROJ_DIR%\WordEmbedDemo.csproj" /p:Release /nologo
    if "%ERRORLEVEL%"=="0" (
        echo.
        echo [OK] BUILD OK: %cd%\%PROJ_DIR%\bin\Release\WordEmbedDemo.exe
        echo [OK] run it:  "%cd%\%PROJ_DIR%\bin\Release\WordEmbedDemo.exe"
    ) else (
        echo [FAILED] MSBuild returned code %ERRORLEVEL%.
    )
    goto :done
)

set "OUT=%PROJ_DIR%\bin_csc"
if defined CSC (
    echo [build] using CSC: %CSC%
    if not exist "%OUT%" mkdir "%OUT%"
    "%CSC%" /nologo /target:exe /out:"%OUT%\WordEmbedDemo.exe" "%PROJ_DIR%\NativeMethods.cs" "%PROJ_DIR%\WordProcessHost.cs" "%PROJ_DIR%\MainForm.cs" "%PROJ_DIR%\Program.cs" "%PROJ_DIR%\Properties\AssemblyInfo.cs"
    if "%ERRORLEVEL%"=="0" (
        echo.
        echo [OK] compiled: %cd%\%OUT%\WordEmbedDemo.exe
        echo [note] this is a console-subsystem exe ^(may show a black window^).
    ) else (
        echo [FAILED] see compiler output above.
    )
    goto :done
)

:no_tool
echo.
echo [ERROR] Neither MSBuild nor CSC (csc.exe) was found on this machine.
echo.
echo To compile this project you need a .NET (C#) compiler. Options:
echo   A) Install Microsoft Visual Studio Build Tools (VS 2019/2022/Community),
echo      which provides msbuild.exe, then re-run build.cmd.
echo   B) Install a .NET Framework SDK (v4.x), which includes csc.exe.
echo   C) Run the app directly from the existing exe:
echo        "%cd%\%PROJ_DIR%\bin_testWordEmbedDemo.exe"
echo      (prebuilt binary still works, but does NOT include the latest fix)
echo.

:done
echo.
endlocal
timeout /t 5 /nobreak >nul 2>nul