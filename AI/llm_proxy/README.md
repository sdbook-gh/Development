# llm_proxy

用 Python 实现的类 LiteLLM 代理服务：对外提供**虚拟模型**，后台按策略在多个真实模型（upstream）间做**检测与切换**，供 pi 等多 agent 并行使用。

## 功能特性

- **虚拟模型**：客户端请求 `model: agent-auto`，由代理映射到真实 upstream 转发
- **多种负载均衡策略**：轮流（round_robin）、按负载（least_load）、故障切换（failover）、加权随机（weighted_random）
- **故障检测与切换**：429 / 400 / 408 / 5xx / 超时 / 连接错误自动切换下一个 upstream，支持熔断冷却与半开恢复
- **每 provider 独立网络策略**：直连或走指定代理
- **每 upstream 独立频率控制**：不控制 / 最小间隔 / 每分钟配额，避免 429
- **多 agent 并行**：全异步（FastAPI + httpx），按 in-flight 并发数做负载统计
- **日志**：console + 滚动文件，记录每次选择、切换原因、错误详情、熔断/恢复事件
- **pi extension 适配**：兼容 LiteLLM 协议（`/model/info` + `x-litellm-*` 响应头），现有 `pi_llmproxy` extension 零逻辑改动

## 目录结构

```
llm_proxy/
├── venv/                      # Python 虚拟环境（env.sh / env.cmd 创建）
├── llmproxy/                  # 代理服务包
│   ├── __init__.py
│   ├── __main__.py            # python -m llmproxy 启动入口
│   ├── config.py              # 配置加载 + 校验（api_key 支持 ${ENV_VAR}）
│   ├── upstream.py            # upstream 客户端（httpx，直连/代理、连接池、流式透传）
│   ├── ratelimit.py           # 每 upstream 频率控制
│   ├── health.py              # 熔断冷却、Retry-After 尊重、半开恢复
│   ├── router.py              # 负载均衡策略 + 故障切换
│   ├── api.py                 # FastAPI 路由（OpenAI 兼容 + LiteLLM 兼容头）
│   ├── status.py              # /status 状态聚合
│   └── logging_setup.py       # 日志：console + 滚动文件
├── pi_llmproxy/               # pi extension（显示实际在用哪个模型）
├── config.yaml                # 主配置文件
├── requirements.txt
├── env.sh                     # Linux/WSL 环境初始化（venv + 依赖）
├── env.cmd                    # Win10 原生环境初始化（venv + 依赖）
├── run.sh                     # Linux/WSL 启动
├── run.cmd                    # Win10 原生启动
└── scripts/smoke_test.sh      # 冒烟测试（多 agent 并发、强制切换演示）
```

## 配置文件（config.yaml）

```yaml
server:
  host: 127.0.0.1
  port: 4400
  log_level: info
  log_file: logs/llmproxy.log

# 可选：出站代理定义，供 upstream 按名字引用
proxies:
  home: http://172.19.16.1:4067

upstreams:
  - name: glm-free
    base_url: https://api.z.ai/api/paas/v4
    api_key: ${GLM_API_KEY}          # 支持环境变量引用，避免明文入库
    model: coding-glm-5.3-free
    proxy: home                      # none=直连 / 代理名=走代理
    headers:                         # 可选：扩展 HTTP 头（可覆盖 User-Agent 等默认头）
      x-opencode-project: vscode
    rate_limit:
      mode: min_interval             # none | min_interval | rpm
      min_interval_ms: 1000
    weight: 1
  - name: gemini-flash
    base_url: https://...
    api_key: ...
    model: gemini-3.7-flash-free
    proxy: none
    rate_limit: { mode: none }

virtual_models:
  agent-auto:                        # 虚拟模型名，pi 里就选这个
    strategy: failover               # round_robin | least_load | failover | weighted_random
    upstreams: [glm-free, gemini-flash]
    retry:
      max_attempts: 3
      retryable_status: [408, 429, 500, 502, 503, 504, 400]
      cooldown_seconds: 30           # 连续失败 N 次后的熔断时长
      failure_threshold: 3
```

### 配置项说明

| 配置项 | 说明 |
|--------|------|
| `server.port` | 监听端口，默认 **4400**；命令行 `--port` 可覆盖 |
| `proxies` | 命名代理列表，upstream 通过名字引用 |
| `upstreams[].proxy` | `none`=直连；代理名=走该代理 |
| `upstreams[].rate_limit.mode` | `none` 不控制 / `min_interval` 最小间隔 / `rpm` 每分钟配额 |
| `upstreams[].api_key` | 支持 `${ENV_VAR}` 环境变量引用 |
| `upstreams[].headers` | 可选，扩展 HTTP 头（字符串映射，值支持 `${ENV_VAR}`）；合并顺序在默认头之后，可覆盖 `User-Agent` 等默认头；仅该 upstream 生效 |
| `virtual_models.<name>.strategy` | 该虚拟模型的负载均衡策略 |
| `retry.retryable_status` | 触发切换的状态码；400 默认包含（部分 provider 对超长上下文返回 400），可移除 |
| `retry.cooldown_seconds` | 连续失败达阈值后的熔断冷却时长 |
| `retry.failure_threshold` | 触发熔断的连续失败次数 |

## 核心行为

### 负载均衡策略

| 策略 | 行为 |
|------|------|
| `round_robin` | 依次轮流分配 |
| `least_load` | 选当前 in-flight 并发最少的 upstream（多 agent 并行的关键） |
| `failover` | 按配置顺序，失败切换下一个 |
| `weighted_random` | 按 weight 加权随机 |

### 故障切换与熔断

- **可切换错误**：429（尊重 `Retry-After` 设置冷却）、408/500/502/503/504、超时、连接错误、（可选）400
- **切换**：失败即标记并换下一个 upstream 重试，直到 `max_attempts` 或全部尝试完
- **熔断**：连续失败达 `failure_threshold` 后冷却 `cooldown_seconds`，期间跳过该 upstream
- **半开恢复**：冷却结束后放一个探测请求，成功即恢复

### 流式请求

`stream: true` 全程透传；重试只发生在**首字节之前**，之后出错如实记日志并透传给客户端。

### 多 agent 并行

- FastAPI 全异步 + httpx 连接池，天然支持多 pi agent 并发
- 每 upstream 独立计数器（asyncio 安全）支撑 `least_load`
- 客户端可用 `x-client-id` 头自报身份，日志按 agent 区分；每请求有 `rid` 贯穿全链路日志

## 日志

- console + `logs/llmproxy.log`（按大小滚动，保留 3 份）
- 每请求一行：时间、rid、client、虚拟模型、选中 upstream、尝试次数、最终状态码、耗时、**切换原因**（如 `glm-free 429 → failover gemini-flash`）
- 后台事件：健康探测结果、熔断进入/解除、限速等待
- 失败时截取响应体片段入日志，便于排查 400 之类

## pi extension 适配

服务器响应头同时返回：

- `x-llmproxy-upstream` / `x-llmproxy-model-id`（自有头）
- `x-litellm-model-id` / `x-litellm-model-group`（兼容别名）
- `/model/info` 返回 LiteLLM 同构 JSON

现有 `pi_llmproxy/index.ts` **逻辑零改动**，仅将 `pi_llmproxy/config.json` 的 `proxyUrl` 指向本服务。

## 观测端点

| 端点 | 用途 |
|------|------|
| `GET /health` | 存活检查 |
| `GET /status` | 每 upstream 的健康/冷却/并发/失败数/最近错误/限速状态（JSON） |
| `GET /v1/models` | OpenAI 兼容模型列表（含虚拟模型） |
| `GET /model/info` | LiteLLM 兼容模型信息 |
| `POST /v1/chat/completions` | OpenAI 兼容对话接口（含流式） |

## 快速开始

### WSL / Linux

```bash
# 1. 环境初始化（创建 venv + 安装依赖，幂等）
./env.sh

# 2. 配置
cp config.yaml config.yaml   # 按需修改；api_key 建议用环境变量放 .env

# 3. 启动
./run.sh                     # 或 ./venv/bin/python -m llmproxy

# 4. pi 侧
pi -e ./pi_llmproxy/index.ts
```

### Win10 原生

```batch
:: 1. 环境初始化（创建 venv + 安装依赖，幂等）
env.cmd

:: 2. 配置
copy config.yaml config.yaml   :: 按需修改；api_key 建议用环境变量放 .env

:: 3. 启动
run.cmd                        :: 或 venv\Scripts\python.exe -m llmproxy

:: 4. pi 侧
pi -e ./pi_llmproxy/index.ts
```

## 已确认决策

| 决策点 | 结论 |
|--------|------|
| 配置格式 | YAML |
| 监听端口 | 4400（命令行可覆盖） |
| 状态展示 | 仅日志 + `/status` JSON，不做 Web 页 |
| 版本管理 | git init |
| pip 下载 | 不走代理（直连） |
| 运行环境 | WSL（Python 3.10.12）/ Win10 原生（Python 3.x），venv 在项目根 `venv/`（两平台共用目录名，各自初始化时覆盖创建） |
| 环境初始化 | `./env.sh`（Linux）/ `env.cmd`（Win10） |
| 启动脚本 | `./run.sh`（Linux）/ `run.cmd`（Win10） |

## TODO

- [x] venv 创建与依赖安装（走代理；执行前二次确认）
- [x] `llmproxy/config.py` — YAML 配置加载与校验
- [x] `llmproxy/ratelimit.py` — 每 upstream 频率控制
- [x] `llmproxy/health.py` — 熔断与半开恢复
- [x] `llmproxy/router.py` — 负载均衡与故障切换
- [x] `llmproxy/upstream.py` — httpx 上游客户端（直连/代理、流式）
- [x] `llmproxy/api.py` — FastAPI 端点与 LiteLLM 兼容响应头
- [x] `llmproxy/logging_setup.py` + `llmproxy/status.py` — 日志与状态聚合
- [x] `llmproxy/__main__.py` — 启动入口
- [x] `config.yaml` 配置样例
- [x] `pi_llmproxy/config.json` 指向新端口 4400
- [x] `run.sh` + `scripts/smoke_test.sh`
- [x] 启动服务自测并汇报
