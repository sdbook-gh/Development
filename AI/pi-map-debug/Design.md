# map-debug 扩展分析报告

> 分析对象:`.pi/extensions/map-debug/`(pi 自定义扩展 `pi-map-debug`)
> 报告生成基于对源码的静态分析,涵盖架构、实现、依赖与风险评估。

---

## 1. 项目概述与定位

`map-debug` 是一个 **pi 编码助手的自定义扩展**,核心目标是建立一条
**「浏览器端报错 → pi(LLM)」** 的反馈通道。

### 背景
pi 自身无法看到浏览器控制台。而离线地图项目(`map-offline-navigation/map_style`)
的渲染依赖 `tile_server.py`(本地瓦片服务)+ `style_localmap.html`(MapLibre 渲染页)。
当渲染异常或瓦片出错时,LLM 缺乏浏览器侧的诊断信息。

### 解决方案
扩展注册了一个工具 **`map_debug_test`**,通过 **Playwright 驱动 headless Chromium**
打开待测页面,自动抓取浏览器端全部报错与网络统计,并以结构化 JSON 返回给 LLM,
使其能够据此定位 `tile_server.py` / 样式 / 瓦片的问题并验证修复。

### 一句话定位
> 一个「无头浏览器探针」pi 扩展:用 Playwright 打开本地地图页,
> 把 JS 异常、console、网络失败、HTTP 错误、MapLibre 事件、瓦片 z/x/y 统计
> 汇总成 JSON 诊断报告,供 LLM 闭环调试。

---

## 2. 目录结构与文件清单

```
map-debug/
├── .pi/                     # pi 扩展元数据目录
├── browsers/                # 本地下载的 Chromium(不占用 ~/.cache/ms-playwright)
│   ├── chrome-headless-shell-linux64/
│   └── chrome-linux64/
├── node_modules/            # 依赖(已安装)
├── ambient.d.ts             # 本地类型声明(仅类型检查用,运行时被 pi 注入覆盖)
├── e2e.mjs                  # 端到端验证脚本(脱离 pi,手工复现同样抓取)
├── index.ts                 # 扩展主入口(注册 map_debug_test 工具)
├── package.json             # 包配置 + pi 扩展声明 + 依赖
├── package-lock.json        # 锁文件
└── tsconfig.json            # TypeScript 编译/检查配置
```

| 文件 | 行数/大小 | 作用 |
|---|---|---|
| `index.ts` | ~11 KB | 扩展核心:注册工具、注入 Hook、抓取报错、生成报告 |
| `e2e.mjs` | ~4 KB | 镜像 `index.ts` 逻辑的独立 E2E 脚本(无需 pi) |
| `package.json` | 436 B | 包名 `pi-map-debug`,声明 `pi.extensions:["./index.ts"]` |
| `ambient.d.ts` | 250 B | 本地 `ExtensionAPI` 类型垫片 |
| `tsconfig.json` | 304 B | ES2022 / strict / noEmit 类型检查配置 |
| `readme.md`(上级) | 2.3 KB | 安装、激活、参数、抓取内容、预期说明 |

---

## 3. 核心架构与端到端工作流

### 3.1 pi 扩展机制
- `package.json` 中 `"pi": { "extensions": ["./index.ts"] }` 声明入口。
- `index.ts` 以 **ESM 默认导出函数** `export default function (pi: ExtensionAPI)` 形式被 pi 加载。
- 通过 `pi.registerTool({...})` 注册一个 LLM 可调用的工具 `map_debug_test`。
- 运行时 `ExtensionAPI` 由 pi **注入**;`ambient.d.ts` 仅为本地 `tsc` 类型检查提供垫片,
  其 `import type` 在运行时被擦除。

### 3.2 端到端工作流(`map_debug_test.execute`)

```
LLM 调用 map_debug_test(url, zoomRange, ...)
        │
        ▼
[1] 可选健康检查  ── fetch http://127.0.0.1:3000/health (4s 超时)
        │            └─ 记录 report.server = "ok" / "http_xxx" / "down (...)"
        ▼
[2] 启动浏览器    ── chromium.launch(executablePath=browsers/..., --no-sandbox)
        │            └─ 优先 chrome-headless-shell,回退 chrome
        ▼
[3] 注入 Hook     ── page.addInitScript(MAP_HOOK_SCRIPT)
        │            └─ 在页面任何脚本前包裹 maplibregl.Map
        ▼
[4] 挂载监听      ── pageerror / console / requestfailed / response
        │            └─ 实时分流到收集器(瓦片请求按 /tiles/{z}/{x}/{y} 归类)
        ▼
[5] 打开页面      ── page.goto(url, waitUntil:"load", timeout)
        │            └─ 失败则记录 gotoError,不中断
        ▼
[6] 等待渲染      ── waitForTimeout(waitMs)
        ▼
[7] zoom 扫描     ── 仅当 zoomRange.length>1:page.evaluate 控制 window.map.setZoom
        │            └─ 逐级 setZoom + 等待 "idle"(单级最多 2.5s)
        ▼
[8] 再等待 + 取数 ── waitForTimeout(waitMs) + 读取 window.__mapErrors
        ▼
[9] 截图          ── .pi/tmp/map-debug-{timestamp}.png
        ▼
[10] 汇总判定     ── 组装 report,计算 verdict,JSON 截断 8000B
        │            └─ CLEAN / ERRORS(n) / UNKNOWN
        ▼
[11] 关闭浏览器   ── finally: browser.close()
        │
        ▼
返回 { content: text, details: report } 给 LLM
```

---

## 4. 核心文件详解

### 4.1 `index.ts` —— 扩展核心

#### (a) 路径解析
```ts
const HERE = dirname(fileURLToPath(import.meta.url));   // 扩展自身目录
const BROWSERS_DIR = resolve(HERE, "browsers");
const HEADLESS_SHELL = resolve(BROWSERS_DIR, "chrome-headless-shell-linux64", "chrome-headless-shell");
const FULL_CHROME    = resolve(BROWSERS_DIR, "chrome-linux64", "chrome");
```
- Chromium **随扩展分发**(`browsers/`),与系统/全局缓存解耦,避免污染 `~/.cache/ms-playwright`。
- 运行时按存在性优先选用 headless-shell,回退完整 chrome。

#### (b) `MAP_HOOK_SCRIPT` —— MapLibre 拦截注入(最关键技术点)
一段在页面脚本**之前**执行的 IIFE,目的:在 MapLibre 实例化前包裹其构造函数,
从而能监听 `error` / `tileerror` 事件(这些事件不会出现在 console / pageerror 中)。

机制:
1. `window.__mapErrors = []` —— 全局错误收集桶。
2. `Object.defineProperty(window, "maplibregl", { get/set })` —— 拦截对 `window.maplibregl` 的赋值。
3. 当页面加载 MapLibre 并赋值 `window.maplibregl = ...` 时,`set` 触发:
   - 标记 `v.__mapHooked = true`(防重复包裹);
   - 保存原始构造函数 `Orig = v.Map`;
   - 定义 `Wrapped` 构造函数:用 `Reflect.construct(Orig, args, newTarget)` 正确构造实例,
     随即为该实例注册 `m.on("error")` 与 `m.on("tileerror")`,错误推入 `__mapErrors`;
   - `Wrapped.prototype = Object.create(Orig.prototype)` + 重置 `constructor`,维持原型链;
   - `v.Map = Wrapped` —— 替换对外暴露的构造函数。
4. 全程 `try/catch` 包裹,任何 hook 失败都不影响页面正常渲染(容错优先)。

> 注意:`addInitScript` 保证此脚本在页面所有 `<script>` 之前执行,
> 因此能先于 MapLibre 库的赋值完成 `defineProperty` 拦截。

#### (c) `truncate(text, 8000)`
对最终 JSON 做字节级截断,附带 `truncated` 标志,防止超长报告撑爆 LLM 上下文。

#### (d) 工具注册 `registerTool`
- `name`: `map_debug_test`,`label`: `Map Debug Test`。
- `description`: 说明用 file:// 打开页面、依赖 `tile_server.py` 的 CORS 头、抓取项与 zoom 扫描。
- `promptSnippet` / `promptGuidelines`:引导 LLM 在「渲染异常 / 瓦片问题」时使用,
  并界定边界——浏览器侧用本工具,`tile_server.py` 的启停/日志用 bash。
- `parameters`(TypeBox `Type.Object`):
  | 参数 | 类型 | 默认 | 说明 |
  |---|---|---|---|
  | `url` | string? | `file://{cwd}/style_localmap.html` | 待测页面 |
  | `zoomRange` | number[]? | `[10]` | 多级缩放扫描(需页面暴露 `window.map`) |
  | `skipServerCheck` | boolean? | `false` | 跳过 3000 /health 检查 |
  | `waitMs` | integer? | `1500` | 每级缩放等待毫秒 |
  | `screenshot` | boolean? | `true` | 截图到 `.pi/tmp/` |
  | `timeoutMs` | integer? | `60000` | 页面加载超时 |
  | `headless` | boolean? | `true` | 无头模式 |

#### (e) `execute` 抓取逻辑要点
- **健康检查**:4s 超时的 `AbortController` + `fetch("/health")`,仅诊断、不阻断。
- **事件分流**:
  - `pageerror` → 记录 `name: message` + stack 前 6 行。
  - `console` → 仅收 `error`/`warning` 级别。
  - `requestfailed` → URL 正则 `\/tiles\/(\d+)\/(\d+)\/(\d+)` 命中归 `tiles(tile-failed, ok:false, reason)`,
    否则归 `reqFailed` + `otherFailed`。
  - `response` → `status>=400` 归 `httpErrs`(URL 经 `absSnippet` 截断 160 字符);
    命中瓦片正则归 `tiles(tile-resp, ok, status)`。
- **zoom 扫描**(仅 `zoomRange.length>1` 触发):
  在 `page.evaluate` 内操作 `window.map.setZoom(z)`,每级用 `m.once("idle")` 等待渲染稳定
  (兜底 2.5s 超时),返回 `{ swept, reached }`。若页面未暴露 `window.map` 则返回提示而不报错。
- **截图**:路径 `.pi/tmp/map-debug-{Date.now()}.png`,`mkdirSync` 递归建目录。
- **汇总字段**:`pageerrors` / `console`(≤150) / `requestFailed`(≤150) / `httpErrors`(≤150)
  / `mapErrors` / `tiles`(≤300) / `otherFailed`(≤80) / `tilesFetched` / `failedTileCount` / `screenshot` / `zoom`。
- **verdict 判定**:
  `errCount = pageerrors + reqFailed + httpErrs + mapErrors`
  → `0` ⇒ `CLEAN(未见明显报错)`;`>0` ⇒ `ERRORS(n)`。
- **取消响应**:检查 `signal.aborted`,返回带 `cancelled:true` 的 details。
- **资源释放**:`try/finally` 确保 `browser.close()` 必被执行。

### 4.2 `e2e.mjs` —— 独立端到端脚本
- **定位**:开发期脱离 pi 手工复现同样抓取,便于快速验证页面/服务状态。
- **差异**(相对 `index.ts`):
  - 无 pi、无参数、无健康检查、无 zoom 扫描。
  - `url` 基于 `process.cwd()`(`index.ts` 基于 `ctx.cwd`,等价但来源不同)。
  - 固定 `waitForTimeout(3000)`;固定截图名 `map-debug-e2e.png`。
  - 收集上限更小(console≤60、tiles≤60、otherFailed≤30)。
  - 结果 `console.log(JSON.stringify)` 直接打印。
- **运行约束**:**必须从项目根** `map_style/` 运行(`readme.md` 明确要求),
  否则 `style_localmap.html` 路径解析错误。

### 4.3 `package.json`
```jsonc
{
  "name": "pi-map-debug", "private": true, "version": "1.0.0", "type": "module",
  "scripts": { "build": "echo 'nothing to build'", "check": "echo 'nothing to check'" },
  "pi": { "extensions": ["./index.ts"] },
  "dependencies":      { "playwright": "^1.62.1", "typebox": "^1.3.11" },
  "devDependencies":   { "@types/node": "^26.1.2", "tsx": "^4.23.10", "typescript": "^7.0.2" }
}
```
- `type: module` + pi 直接加载 `index.ts`(经 `tsx` 运行时编译),**无需预构建**
  (`build`/`check` 均为空 echo)。
- 运行时依赖:`playwright`(浏览器自动化)、`typebox`(工具参数 schema)。
- 开发依赖:`tsx`(TS 运行时)、`typescript` + `@types/node`(类型检查)。

### 4.4 `ambient.d.ts`
```ts
declare module "@earendil-works/pi-coding-agent" {
  export interface ExtensionAPI {
    registerTool(x: unknown): void;
    [k: string]: unknown;
  }
}
```
- **仅本地类型检查用途**。真实 `ExtensionAPI` 由 pi 在运行时注入,
  `import type` 在编译后擦除,不会引入运行时依赖。

### 4.5 `tsconfig.json`
- `target: ES2022`、`module: ESNext`、`moduleResolution: bundler`、`strict: true`、`noEmit: true`。
- `lib: ES2022, DOM`(DOM 用于 `MAP_HOOK_SCRIPT` 中浏览器 API 的类型推断)。
- `include: index.ts, ambient.d.ts`。
- `readme.md` 强调:**类型检查必须用 `node_modules/.bin/tsc -p tsconfig.json`**,
  裸 `npx tsc` 会拉到全局错误版本。

---

## 5. 关键技术点

| 技术点 | 实现方式 | 价值 |
|---|---|---|
| **MapLibre 事件捕获** | `addInitScript` + `Object.defineProperty(window,'maplibregl')` 拦截赋值,包裹 `Map` 构造函数注册 `error`/`tileerror` | 捕获 console/pageerror 抓不到的渲染层错误 |
| **瓦片 z/x/y 精确统计** | URL 正则 `\/tiles\/(\d+)\/(\d+)\/(\d+)` 在 `requestfailed`/`response` 双通道归类 | 精确定位哪一级哪一瓦片失败 |
| **错误四分类** | pageerror / console / network(reason) / HTTP≥400 / mapEvent | 覆盖浏览器端几乎所有报错来源 |
| **zoom 扫描** | `page.evaluate` 调 `window.map.setZoom` + `once('idle')` | 触发多层级瓦片加载,发现 zoom 相关问题 |
| **上下文保护** | 各列表 slice 上限 + JSON 8000B 截断 + `truncated` 标志 | 防止超长报告污染 LLM 上下文 |
| **本地浏览器分发** | Chromium 放 `browsers/`,按存在性选 headless-shell/chrome | 与系统缓存解耦,可移植 |
| **容错优先** | Hook 与各步骤全程 `try/catch`,`goto` 失败不中断 | 探针自身尽量不因页面问题而崩溃 |
| **资源安全** | `try/finally` + `browser.close()` + abort 信号 | 杜绝浏览器进程泄漏 |

---

## 6. 依赖关系与运行环境

### 6.1 依赖图
```
index.ts
  ├── @earendil-works/pi-coding-agent  (类型,运行时由 pi 注入)
  ├── typebox        → Type.Object 参数 schema
  ├── playwright     → chromium 浏览器自动化
  └── node:fs/path/url

e2e.mjs
  └── playwright + node:fs/path/url   (无 typebox / 无 pi)

运行时外部依赖:
  └── tile_server.py  (需在 127.0.0.1:3000 提供 /tiles 与 /health,并返回 CORS 头)
```

### 6.2 运行前提
1. pi 已加载扩展(`/reload` 或重启),工具 `map_debug_test` 可用。
2. `tile_server.py` 已运行(除非 `skipServerCheck=true`)。
3. 待测页面 `style_localmap.html` 存在于 `cwd`。
4. `browsers/` 下 Chromium 可执行(已随扩展下载)。
5. `e2e.mjs` 必须从项目根 `map_style/` 运行。

### 6.3 正常预期(readme 定义)
- `pageerrors=0`、`mapErrors=[]`、`httpErrors=[]`、瓦片 0 失败。
- console 仅可能出现 headless 的 `GL Driver Message … GPU stall due to ReadPixels` 告警
  (无头渲染噪声,可忽略)。

---

## 7. 优点 / 风险与局限分析

### 7.1 优点
- **闭环设计优秀**:把 LLM 看不到的浏览器黑盒打开,形成
  「调工具 → 看报错 → 改 tile_server/样式 → 再调工具验证」的完整闭环。
- **Hook 技巧巧妙**:`defineProperty` 拦截库赋值 + 构造函数包裹,是非侵入式捕获
  MapLibre 内部事件的有效手段,且全程容错。
- **报告结构化且防御性强**:字段分明、上限截断、verdict 一目了然。
- **零构建**:pi 直接跑 `index.ts`(`tsx`),`build`/`check` 为空,部署简单。
- **可移植**:Chromium 随扩展走,不依赖系统安装。

### 7.2 风险与局限
1. **`verdict` 计算口径偏严/有遗漏**
   - `errCount` 未计入 `tiles` 中的失败瓦片与 `otherFailed`,
     故「瓦片大量失败但无 JS/HTTP/map 错误」时仍可能被判 `CLEAN`,
     与 `failedTileCount` 不一致,易误导 LLM。
   - `console` warning 也未计入 errCount(仅 error 级网络/HTTP 计入),口径需明确。
2. **`MAP_HOOK_SCRIPT` 依赖赋值时机**
   - 仅当页面以 `window.maplibregl = ...` 方式赋值时拦截生效;
     若页面通过 ES Module `import` 直接持有引用(不经 `window.maplibregl`),Hook 失效。
   - 依赖 `v.Map` 存在;若 MapLibre API 结构变更(如改命名导出)需同步维护。
3. **`e2e.mjs` 与 `index.ts` 逻辑重复**
   - 两份 `MAP_HOOK_SCRIPT` 几乎相同,维护时易出现两边不一致
     (目前已在 `Wrapped` 原型处理等细节上完全一致,但属隐患)。
4. **zoom 扫描前置条件强**
   - 必须页面暴露 `window.map` 且为 MapLibre 实例;`zoomRange` 默认 `[10]`(长度 1)
     意味着**默认不扫描**,需 LLM 主动传多值才触发,易被忽略。
5. **`waitMs` 固定等待而非事件驱动**
   - 首屏用 `waitForTimeout(waitMs)` 而非 `networkidle`/`load` 之外的渲染完成信号,
     弱网或大瓦片量下可能抓取过早;`idle` 等待仅用于 zoom 扫描阶段。
6. **截图无失败处理反馈**
   - 截图失败时 `shot = undefined` 静默吞掉,LLM 可能误以为有图但路径为空。
7. **`absSnippet` 仅用于部分字段**
   - `httpErrs` 用了截断,但 `reqFailed` 存原始 URL(无截断),超长 URL 仍可能进报告
     (虽有 slice 上限,但单条可能很长)。
8. **平台耦合**
   - `browsers/` 路径硬编码 `linux64`,仅适配 Linux;跨平台需额外处理。
9. **`report.browser` 字符串拼接缺陷**
   - `report.browser = headless ? "headless " : "" + (...)` 中 `"" + (...)` 因运算符优先级
     实际为 `"" + existsSync(...)?...`,当 `headless=false` 时结果为空串前缀拼接,
     语义与字面意图略有偏差(非致命,但可读性/准确性受损)。

---

## 8. 改进建议

### 8.1 正确性
- **统一 `verdict` 口径**:将 `failedTileCount` 与 `otherFailed.length` 纳入 errCount,
  或额外输出 `tileVerdict`,避免「瓦片全挂却 CLEAN」的误判。
- **修复 `report.browser` 拼接**:用模板字符串
  `` `${headless ? "headless " : ""}${existsSync(HEADLESS_SHELL) ? "chromium-headless-shell" : "chromium"}` ``
  明确表达意图。
- **`reqFailed` URL 一致截断**:与 `httpErrs` 一样走 `absSnippet`。

### 8.2 健壮性
- **Hook 失效兜底**:在读取 `__mapErrors` 后,若数组为空且未检测到 `window.maplibregl.__mapHooked`,
  在报告中加 `mapHook:"not-applied"` 提示,帮助 LLM 判断 Hook 是否生效。
- **首屏等待策略**:可选支持 `waitUntil:"networkidle"` 或显式等待 `map.on('load')`,
  替代纯 `waitForTimeout`,提升抓取稳定性。
- **截图失败可见化**:截图失败时在 report 中显式置 `screenshot:"failed"` 而非 `undefined`。

### 8.3 可维护性
- **抽取共享 Hook**:将 `MAP_HOOK_SCRIPT` 提到独立 `.mjs`/字符串常量模块,
  供 `index.ts` 与 `e2e.mjs` 共同 import,消除重复。
- **默认 zoom 扫描**:考虑将默认 `zoomRange` 设为 `[10, 14, 15]` 之类多值,
  或在 description 中更显著提示「单值不扫描」,降低误用。
- **跨平台浏览器路径**:按 `process.platform` 选择 `browsers/` 子目录,提升可移植性。

### 8.4 可观测性
- **分阶段 `onUpdate`**:当前仅在启动时发一次 `onUpdate`,可在 goto / zoom / 截图各阶段
  推送进度,改善长任务的可视反馈。
- **暴露原始(未截断)报告路径**:将完整 JSON 落盘到 `.pi/tmp/`,details 中给路径,
  便于 LLM 在需要时回看全量数据。

---

## 9. 结论

`map-debug` 是一个**设计目标清晰、实现技巧扎实**的 pi 扩展,
通过「Playwright 无头浏览器 + MapLibre 构造函数 Hook + 瓦片正则归类」三件套,
有效打通了 LLM 调试离线地图渲染的「最后一公里」。
其容错优先、上下文保护、零构建、本地浏览器分发的工程取舍均合理。

主要待改进点集中在 **`verdict` 口径一致性**、**Hook 生效可观测性**、
**`index.ts`/`e2e.mjs` 逻辑去重**与**首屏等待策略**上,
均为增量优化,不影响当前核心可用性。

---

*报告完*
