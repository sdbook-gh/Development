@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem llmproxy 启动脚本：激活 venv 并启动服务（默认读项目根 config.yaml）
rem
rem 用法:
rem   run.cmd                     使用 config.yaml
rem   run.cmd --port 4500         透传 python -m llmproxy 的参数
rem
rem 注意: config.yaml 中 api_key 直接写入
setlocal
cd /d "%~dp0"


venv\Scripts\python.exe -m llmproxy %*
endlocal
