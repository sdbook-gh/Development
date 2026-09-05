#!/usr/bin/env bash
# llmproxy 启动脚本：激活 venv 并启动服务（默认读项目根 config.yaml）
#
# 用法:
#   ./run.sh                     # 使用 config.yaml
#   ./run.sh --port 4500         # 透传 python -m llmproxy 的参数
#
# 注意: config.yaml 中 api_key 直接写入
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "错误: venv 不存在。先执行:" >&2
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

exec venv/bin/python -m llmproxy "$@"
