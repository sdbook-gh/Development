@echo off
rem llmproxy 启动脚本：激活 venv 并启动服务（默认读项目根 config.yaml）
rem
rem 用法:
rem   run.cmd                     使用 config.yaml
rem   run.cmd --port 4500         透传 python -m llmproxy 的参数
rem
rem 注意: config.yaml 中 api_key 使用 ${ENV_VAR} 引用。
rem   启动前自动加载 .env（本地密钥，不入库）；也可手动 set 变量。
setlocal
cd /d "%~dp0"

rem 加载本地密钥（.env 已被 .gitignore 忽略）
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

venv\Scripts\python.exe -m llmproxy %*
endlocal
