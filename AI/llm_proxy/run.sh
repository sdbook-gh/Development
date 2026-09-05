#!/usr/bin/env bash
# llmproxy 启动脚本：激活 venv 并启动服务（默认读项目根 config.yaml）
#
# 用法:
#   ./run.sh                     # 使用 config.yaml
#   ./run.sh --port 4500         # 透传 python -m llmproxy 的参数
#
# 注意: config.yaml 中 api_key 使用 ${ENV_VAR} 引用。
#   启动前自动加载 .env（本地密钥，不入库）；也可手动导出:
#   export SHANGTANG_API_KEY=... RUNINFRA_API_KEY=...
set -euo pipefail
cd "$(dirname "$0")"

# 加载本地密钥（.env 已被 .gitignore 忽略）
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ ! -x venv/bin/python ]; then
    echo "错误: venv 不存在。先执行:" >&2
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

exec venv/bin/python -m llmproxy "$@"
