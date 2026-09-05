# 方案：/switch-llmproxy 强制切换 llmproxy 当前模型

> 状态：方案已确认，等待执行。
> 决策记录：① pin 默认尊重熔断，命令加 `--force` 旁路；② pin 为内存态，重启失效；③ 管理端点不鉴权（依赖 127.0.0.1 绑定，与现有 `/status` 一致）。

## 一、现状分析

### 服务端 `llmproxy/`（Python FastAPI）
- `router.py`：`candidates()` 按策略（round_robin / least_load / failover / weighted_random）从可用池排候选顺序 → `dispatch()` 逐个候选准入（熔断 claim + 限速等待 + in-flight 计数）并调用 `send` 回调。
- `api.py` 现有端点：`POST /v1/chat/completions`、`GET /v1/models`、`GET /model/info`、`GET /health`、`GET /status` —— **没有任何运行时管理端点**，无法在不重启的情况下改变路由行为。
- 隐式直连路由：未定义同名虚拟模型的 upstream 会自动建单 upstream 路由（`model` 填 upstream 名即直达）。

### 扩展端 `pi_llmproxy/index.ts`（pi 扩展）
- 仅展示 footer：轮询 `/model/info` + 读响应头（`x-llmproxy-*` / `x-litellm-*`）判定实际服务的 upstream。
- **未注册任何命令**。pi 支持 `pi.registerCommand(name, { description, getArgumentCompletions, handler })`，可实现 `/switch-llmproxy`。

## 二、服务端设计（llmproxy/）

### 1. `llmproxy/router.py`（局部修改）
- `_Route` 增加 `pinned: str | None` 字段。
- `Router` 新增方法：
  - `set_pin(vm_name, upstream)` —— 校验 upstream 属于该路由（含隐式直连路由），否则抛 `KeyError`/`ValueError`；
  - `clear_pin(vm_name)` —— 清除 pin，恢复正常策略路由；
  - `get_pin(vm_name)` —— 查询当前 pin。
- `candidates()`：有 pin 时**仅返回 pinned upstream 一个候选**，仍正常走 `rt.enter()` 的熔断/限速准入；若该 upstream 处于熔断冷却，由 dispatch 按现有语义返回 503（提示冷却剩余时间）。`force` 旁路见下。
- 熔断旁路：`UpstreamRuntime.enter()` / `HealthTracker.claim()` 增加可选 bypass 参数，仅当请求显式携带 `force=true` 时生效；熔断统计照常记录。
- `route_snapshot()` 输出增加 `pinned` 字段（供 `/status` 与 `/admin/routes` 消费）。

### 2. `llmproxy/api.py`（新增管理端点，局部插入）

| 端点 | 方法 | 请求体 / 说明 | 响应 |
|------|------|----------------|------|
| `/admin/pin` | POST | `{"virtual_model": "agent-auto", "upstream": "sensenova", "force": false}` | 200 `{ok, virtual_model, upstream, force}`；模型不存在 404；upstream 不在该路由 400 |
| `/admin/unpin` | POST | `{"virtual_model": "agent-auto"}` | 200 `{ok, virtual_model}`（未 pin 时幂等成功） |
| `/admin/routes` | GET | — | 每个虚拟模型：候选 upstream 列表（含名称、真实模型、健康状态/冷却剩余、in-flight）、当前 `pinned` |

- pin/unpin 操作记 `logger.info` 日志。
- 不鉴权（与 `/status` 一致），依赖 `server.host: 127.0.0.1` 绑定仅本机访问。

### 3. pin 生命周期
- **内存态，进程重启后自动失效**，恢复 config.yaml 中的策略路由。
- `/admin/routes` 与 `/status` 随时可查当前 pin。

## 三、扩展端设计（pi_llmproxy/index.ts）

### 4. 命令注册 `pi.registerCommand("switch-llmproxy", ...)`
- `/switch-llmproxy`（无参）：
  1. `GET /admin/routes`，取**当前 pi 模型对应的虚拟模型**（由 `ctx.model.id` 解析，如 `agent-auto`；若为直连 upstream 名则取该隐式路由）；
  2. `ctx.ui.select` 列出候选 upstream（标注 健康/冷却中、当前 pin），列表末尾含 `auto（取消固定）` 项；
  3. 选中后 `POST /admin/pin`。
- `/switch-llmproxy <upstream名>`：直接 pin；`<upstream名> --force` 旁路熔断准入。
- `/switch-llmproxy auto`（或 `clear` / `off`）：unpin，恢复正常策略路由。
- `getArgumentCompletions` 提供 upstream 名 + `auto` 补全。
- 成功后 `ctx.ui.notify` 提示，并立即刷新 footer；失败（代理不可达 / upstream 非法）给出明确错误提示。

### 5. footer 增强
- 新增 `fetchRoutes()` 拉取 `/admin/routes`，与现有 `/model/info` 轮询同频（`pollMs`）。
- pin 生效时展示：`🤖 agent-auto ⏸ sensenova (pinned) · via llmproxy`；其余展示逻辑不变。

## 四、文档与测试

### 6. 文档
- `pi_llmproxy/README.md`：新增 `/switch-llmproxy` 用法、pinned footer 展示说明。
- 根 `README.md`：新增 `/admin/pin`、`/admin/unpin`、`/admin/routes` 端点说明与示例。

### 7. `scripts/smoke_test.sh` 追加 pin 场景
- pin 某一 upstream → 多次请求校验响应头 `x-llmproxy-upstream` 恒为 pinned；
- unpin → 恢复原有策略（可观察到轮询/切换行为）。

## 五、验证（执行阶段，运行前再次确认）

8. 运行 `./scripts/smoke_test.sh`（自带 mock upstream，不耗真实 key；临时目录 `/mnt/e/temp`）。
9. 手工 curl 验证：`POST /admin/pin` → `POST /v1/chat/completions` 查看响应头 → `POST /admin/unpin`。
10. `pi -e ./pi_llmproxy/index.ts` 实测 `/switch-llmproxy` 交互与 footer。
11. `git diff` 自检变更范围与预期一致。

## 六、修改文件清单（均局部修改，不整文件重写）

| 文件 | 变更 |
|------|------|
| `llmproxy/router.py` | pin 状态 + candidates 逻辑 + force 旁路 + snapshot |
| `llmproxy/api.py` | 新增 3 个 `/admin/*` 端点 |
| `pi_llmproxy/index.ts` | `/switch-llmproxy` 命令 + footer pinned 展示 + routes 轮询 |
| `pi_llmproxy/README.md` | 命令与配置文档 |
| `README.md` | admin 端点文档 |
| `scripts/smoke_test.sh` | pin 冒烟场景 |
