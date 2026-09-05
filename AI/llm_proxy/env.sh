#!/usr/bin/env bash
# llmproxy 环境初始化：创建 venv 并安装 Python 依赖（幂等）
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo ">>> 创建虚拟环境 ($VENV_DIR) ..."
    python3 -m venv "$VENV_DIR"
    echo ">>> 虚拟环境创建完成"
else
    echo ">>> 虚拟环境已存在，跳过创建"
fi

echo ">>> 安装依赖 ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r requirements.txt -q
echo ">>> 依赖安装完成"
echo ">>> 环境就绪，启动: ./run.sh"
