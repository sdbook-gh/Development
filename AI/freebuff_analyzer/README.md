# freebuff_analyzer

只读分析 freebuff 与远程模型通信的本地 HTTP(S) 抓包工具（基于 mitmproxy，不修改任何流量）。

## 背景结论（二进制分析）

- `freebuff`（npm 包）→ 下载 Bun 编译核心二进制到 `~/.config/manicode/freebuff`
- LLM 请求**不直连 OpenRouter**，而是发给 codebuff.com 自有 AI 网关（`/api/chat/completions` 等），由后端转发上游
- 鉴权：`~/.config/manicode/credentials.json` 里的 `authToken`（预计作为 header 发送）
- 遥测：PostHog（us.i.posthog.com）

## 快速开始

```bash
# 1. 首次: 建 venv + 安装 mitmproxy（已完成可跳过）
python3 -m venv .venv
.venv/bin/pip install mitmproxy

# 2. 终端 A: 启动代理
./start.sh

# 3. 终端 B: 让 freebuff 走代理并信任本工具 CA
export HTTP_PROXY=http://127.0.0.1:8888
export HTTPS_PROXY=http://127.0.0.1:8888
export NO_PROXY=localhost,127.0.0.1
export NODE_EXTRA_CA_CERTS="$PWD/ca/mitmproxy-ca.pem"
export SSL_CERT_FILE="$PWD/ca/mitmproxy-ca.pem"
freebuff
```

发一个测试 prompt 后，观察终端 A 的实时输出 + `logs/session_*/` 里的文件。

## 备用模式（freebuff 不认 HTTPS_PROXY 时）

Bun 二进制若不支持代理环境变量，可用 hosts 劫持 + reverse 模式：

```bash
./start.sh fallback
```

会提示你手动加 `127.0.0.1 codebuff.com` 到 `/etc/hosts`（需要 sudo），然后 reverse 代理到 codebuff 真实 IP。分析完记得删掉 hosts 那行。

## 日志说明

每次启动代理生成 `logs/session_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `session.log` | 完整文本日志（与终端一致，含打码） |
| `NNN_host_METHOD.json` | 每请求一个 JSON：**未打码** headers + 完整请求/响应体 |
| `sse_NNN_host.txt` | SSE 流式响应全文（chat completions 在此） |

控制台会对 `Authorization`/`x-api-key`/`Cookie` 等打码（只显示前 8 位）；JSON 文件为本地分析保留**完整凭据**，注意不要外传，分析完可删。

## 分析要点

拿到日志后重点看：

1. **鉴权**：`*.json` 里 `codebuff.com` 请求的 `Authorization` / `x-api-key` header —— 确认 authToken 如何携带
2. **模型路由**：请求体 `model` 字段（如 `stealth/ox-alpha`）与响应 `providerMetadata.codebuff`（含 costDollars/upstreamInferenceCost）
3. **LLM 通信协议**：`/api/chat/completions` 的 SSE 事件结构（Vercel AI SDK data stream 格式）
4. **其他接口**：`/api/usage`、`/api/web_search`、`/api/ads`、`/api/logs/ingest` 等

## 清理

```bash
rm -rf ca/ logs/     # 删除 CA 与所有日志（含凭据）
```
