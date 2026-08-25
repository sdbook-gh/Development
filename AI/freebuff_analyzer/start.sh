#!/usr/bin/env bash
# 启动 freebuff 流量记录代理
# 用法:
#   ./start.sh           # 默认: 正向代理模式 127.0.0.1:8888
#   ./start.sh fallback  # 备用: hosts 劫持 + reverse 模式(需 sudo, 供 freebuff 不走代理时用)
#   ./start.sh stop      # 停止已启动的代理(8888; 若 443 有 fallback 代理也一并停)
set -euo pipefail
cd "$(dirname "$0")"

MITMDUMP="$PWD/.venv/bin/mitmdump"
PORT="${PORT:-8888}"
CA_DIR="$PWD/ca"

if [ ! -x "$MITMDUMP" ]; then
  echo "未找到 mitmdump，请先执行: .venv/bin/pip install mitmproxy"
  exit 1
fi
mkdir -p "$CA_DIR"

if [ "${1:-}" = "stop" ]; then
  killed=0
  for p in "$PORT" 443; do
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$p\$"; then
      echo "端口 $p 有监听进程，停止中..."
      fuser -k "$p/tcp" 2>/dev/null || sudo fuser -k "$p/tcp"
      killed=1
    else
      echo "端口 $p 无监听，跳过"
    fi
  done
  if [ "$killed" = 1 ]; then echo "已停止"; else echo "没有正在运行的 mitmdump"; fi
  exit 0
fi

if [ "${1:-}" = "fallback" ]; then
  IP="$(getent ahostsv4 codebuff.com | awk '{print $1}' | head -1)"
  if [ -z "$IP" ]; then
    echo "无法解析 codebuff.com（可能已在 /etc/hosts 里被改过，请先还原再运行）"
    exit 1
  fi
  echo "codebuff.com 真实 IP: $IP"
  echo "请先手动执行（需要 sudo，本脚本不自动改 /etc/hosts）:"
  echo "  sudo sh -c 'echo \"127.0.0.1 codebuff.com\" >> /etc/hosts'"
  read -r -p "改好后按 Enter 继续启动 reverse 代理(端口 443, 用 sudo)..."
  echo "启动中: $MITMDUMP -p 443 --mode reverse://$IP:443"
  echo "另一个终端里运行 freebuff 前同样需要:"
  echo "  export NODE_EXTRA_CA_CERTS=$CA_DIR/mitmproxy-ca.pem"
  echo "  export SSL_CERT_FILE=$CA_DIR/mitmproxy-ca.pem"
  sudo "$MITMDUMP" -p 443 \
    --mode "reverse://$IP:443" \
    --set confdir="$CA_DIR" \
    --set ssl_insecure=true \
    -s freebuff_logger.py
else
  echo "启动正向代理: 127.0.0.1:$PORT (CA 目录: $CA_DIR)"
  echo ""
  echo "启动后，另一个终端运行 freebuff 前请设置:"
  echo "  export HTTP_PROXY=http://127.0.0.1:$PORT"
  echo "  export HTTPS_PROXY=http://127.0.0.1:$PORT"
  echo "  export NO_PROXY=localhost,127.0.0.1"
  echo "  export NODE_EXTRA_CA_CERTS=$CA_DIR/mitmproxy-ca.pem"
  echo "  export SSL_CERT_FILE=$CA_DIR/mitmproxy-ca.pem"
  echo "  freebuff"
  echo ""
  "$MITMDUMP" -p "$PORT" \
    --mode regular \
    --set confdir="$CA_DIR" \
    --set ssl_insecure=true \
    -s freebuff_logger.py
fi
