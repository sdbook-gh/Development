@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem llmproxy 环境初始化：创建 venv 并安装 Python 依赖（幂等）
setlocal
cd /d "%~dp0"

set VENV_DIR=venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ^>^>^> 创建虚拟环境 ^(%VENV_DIR%^) ...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo 错误: python -m venv 失败，请确认 PATH 中有 Python 3.x
        exit /b 1
    )
    echo ^>^>^> 虚拟环境创建完成
) else (
    echo ^>^>^> 虚拟环境已存在，跳过创建
)

echo ^>^>^> 安装依赖 ...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo 错误: 依赖安装失败
    exit /b 1
)
echo ^>^>^> 依赖安装完成
echo ^>^>^> 环境就绪，启动: run.cmd
endlocal
