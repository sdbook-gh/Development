================================================================================
                              freebuff_analyzer
================================================================================

只读分析 freebuff 与远程模型通信的本地 HTTP(S) 抓包工具
（基于 mitmproxy，不修改任何流量）。


--------------------------------------------------------------------------------
一、背景结论（二进制分析）
--------------------------------------------------------------------------------

- freebuff（npm 包）会下载 Bun 编译核心二进制到 ~/.config/manicode/freebuff
- LLM 请求 *不直连* OpenRouter，而是发给 codebuff.com 自有 AI 网关
  （/api/chat/completions 等），由后端转发上游
- 鉴权：~/.config/manicode/credentials.json 里的 authToken（预计作为 header 发送）
- 遥测：PostHog（us.i.posthog.com）


--------------------------------------------------------------------------------
二、环境准备（venv + pip 安装依赖）
--------------------------------------------------------------------------------

本工具仅依赖 mitmproxy。推荐使用 Python 虚拟环境（venv）隔离安装，
避免污染系统 Python 环境。

【Linux / macOS / WSL】

  # 1. 进入项目根目录
  cd /path/to/freebuff_analyzer

  # 2. 创建虚拟环境（命名为 .venv，已被 .gitignore 忽略）
  python3 -m venv .venv

  # 3. （可选但推荐）升级 pip 到最新
  .venv/bin/pip install --upgrade pip

  # 4. 安装依赖（两种方式任选其一）

     # 方式 A：通过 requirements.txt 安装（推荐）
     .venv/bin/pip install -r requirements.txt

     # 方式 B：直接安装 mitmproxy
     .venv/bin/pip install mitmproxy

  # 5. 验证安装
  .venv/bin/mitmdump --version

【Windows（PowerShell / CMD）】

  # 1. 进入项目根目录
  cd D:\path\to\freebuff_analyzer

  # 2. 创建虚拟环境
  python -m venv .venv

  # 3. （可选）升级 pip
  .venv\Scripts\python.exe -m pip install --upgrade pip

  # 4. 安装依赖
  .venv\Scripts\pip install -r requirements.txt

  # 5. 验证安装
  .venv\Scripts\mitmdump.exe --version

【依赖说明】

  requirements.txt 内容仅为：
      mitmproxy

  mitmproxy 会自动携带其全部子依赖（tornado、pyOpenSSL、cryptography 等），
  无需手动逐个安装。


--------------------------------------------------------------------------------
三、快速开始
--------------------------------------------------------------------------------

  # 0. 前置：已完成上方“环境准备”（.venv 内已装好 mitmproxy）

  # 1. 终端 A：启动代理
  ./start.sh

  # 2. 终端 B：让 freebuff 走代理并信任本工具 CA
  export HTTP_PROXY=http://127.0.0.1:8888
  export HTTPS_PROXY=http://127.0.0.1:8888
  export NO_PROXY=localhost,127.0.0.1
  export NODE_EXTRA_CA_CERTS="$PWD/ca/mitmproxy-ca.pem"
  export SSL_CERT_FILE="$PWD/ca/mitmproxy-ca.pem"
  freebuff

发一个测试 prompt 后，观察终端 A 的实时输出 + logs/session_*/ 里的文件。


--------------------------------------------------------------------------------
四、备用模式（freebuff 不认 HTTPS_PROXY 时）
--------------------------------------------------------------------------------

Bun 二进制若不支持代理环境变量，可用 hosts 劫持 + reverse 模式：

  ./start.sh fallback

会提示你手动加 “127.0.0.1 codebuff.com” 到 /etc/hosts（需要 sudo），
然后 reverse 代理到 codebuff 真实 IP。
分析完记得删掉 hosts 那行。


--------------------------------------------------------------------------------
五、日志说明
--------------------------------------------------------------------------------

每次启动代理生成 logs/session_<时间戳>/ 目录：

  文件                          内容
  ----------------------------  -----------------------------------------------
  session.log                   完整文本日志（与终端一致，含打码）
  NNN_host_METHOD.json          每请求一个 JSON：未打码 headers +
                                完整请求/响应体
  sse_NNN_host.txt              SSE 流式响应全文（chat completions 在此）

控制台会对 Authorization / x-api-key / Cookie 等打码（只显示前 8 位）；
JSON 文件为本地分析保留 *完整凭据*，注意不要外传，分析完可删。


--------------------------------------------------------------------------------
六、分析要点
--------------------------------------------------------------------------------

拿到日志后重点看：

  1. 鉴权：*.json 里 codebuff.com 请求的 Authorization / x-api-key header ——
     确认 authToken 如何携带
  2. 模型路由：请求体 model 字段（如 stealth/ox-alpha）与响应
     providerMetadata.codebuff（含 costDollars/upstreamInferenceCost）
  3. LLM 通信协议：/api/chat/completions 的 SSE 事件结构
     （Vercel AI SDK data stream 格式）
  4. 其他接口：/api/usage、/api/web_search、/api/ads、/api/logs/ingest 等


--------------------------------------------------------------------------------
七、清理
--------------------------------------------------------------------------------

  rm -rf ca/ logs/     # 删除 CA 与所有日志（含凭据）


--------------------------------------------------------------------------------
八、文件结构
--------------------------------------------------------------------------------

  freebuff_analyzer/
  ├── freebuff_logger.py   mitmproxy addon：流量记录核心逻辑
  ├── start.sh             启动脚本（正向代理 / fallback / stop）
  ├── requirements.txt     Python 依赖清单（mitmproxy）
  ├── readme.txt           本说明文件
  ├── .gitignore
  ├── .venv/               虚拟环境（不入库）
  ├── ca/                  mitmproxy CA 证书目录（运行时生成，不入库）
  └── logs/                抓包日志（运行时生成，不入库）


--------------------------------------------------------------------------------
                                — 文档结束 —
--------------------------------------------------------------------------------
