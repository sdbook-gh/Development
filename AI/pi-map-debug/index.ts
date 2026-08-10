import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { chromium } from "playwright";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * map_debug_test — 用 headless Chromium 打开 style_localmap.html，
 * 抓取浏览器端全部报错，供 LLM 据此调试 tile_server.py 并验证修复。
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSERS_DIR = resolve(HERE, "browsers");
const HEADLESS_SHELL = resolve(BROWSERS_DIR, "chrome-headless-shell-linux64", "chrome-headless-shell");
const FULL_CHROME = resolve(BROWSERS_DIR, "chrome-linux64", "chrome");

/** 在页面脚本之前包裹 maplibregl.Map，收集 map error / tileerror 事件 */
const MAP_HOOK_SCRIPT = `(() => {
  window.__mapErrors = [];
  try {
    let real = undefined;
    Object.defineProperty(window, "maplibregl", {
      configurable: true,
      enumerable: true,
      get() { return real; },
      set(v) {
        real = v;
        try {
          if (v && v.Map && !v.__mapHooked) {
            v.__mapHooked = true;
            const Orig = v.Map;
            const Wrapped = function (...args) {
              if (!(this instanceof Wrapped)) return new Orig(...args);
              const m = Reflect.construct(Orig, args, new.target && new.target.prototype ? new.target : Wrapped);
              try {
                m.on("error", (e) => window.__mapErrors.push({
                  type: "map-error",
                  msg: (e && e.error && (e.error.message || e.error)) || JSON.stringify(e),
                }));
                m.on("tileerror", (e) => window.__mapErrors.push({
                  type: "tileerror",
                  tile: (e && e.tileID && e.tileID.canonical) ? String(e.tileID.canonical) : "",
                }));
              } catch (_) {}
              return m;
            };
            try { Wrapped.prototype = Object.create(Orig.prototype); } catch (_) {}
            try { Object.defineProperty(Wrapped.prototype, "constructor", { value: Wrapped, writable: true, configurable: true }); } catch (_) {}
            v.Map = Wrapped;
          }
        } catch (_) {}
      },
    });
  } catch (_) {}
})();`;

function truncate(text: string, maxBytes = 8000): { content: string; truncated: boolean } {
  const truncated = text.length > maxBytes;
  return {
    content: truncated ? text.slice(0, maxBytes) + `\n...[truncated]` : text,
    truncated,
  };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "map_debug_test",
    label: "Map Debug Test",
    description:
      "用 headless Chromium 打开 style_localmap.html（file://，依赖 tile_server.py 已运行，因其返回 CORS 头），抓取浏览器端报错：JS 异常、console error/warning、失败网络请求(reason)、HTTP>=400 响应、maplibre map error/tileerror 事件；并统计 /tiles/{z}/{x}/{y} 请求成败与具体 z/x/y。可选 zoom 扫描触发多层级瓦片。返回截断后的 JSON 诊断报告。",
    promptSnippet: "抓取离线地图页浏览器端报错、瓦片请求成败与 z/x/y 统计",
    promptGuidelines: [
      "当 style_localmap.html 渲染异常或 tile_server.py 瓦片出问题时，使用 map_debug_test 获取浏览器端错误与瓦片/字体/精灵请求统计，据此定位是哪个 z/x/y、哪个 source-layer 或请求失败。",
      "map_debug_test 只负责浏览器侧信息；tile_server.py 的启动/重启/日志读取用 bash 完成。",
    ],
    parameters: Type.Object({
      url: Type.Optional(Type.String({ description: "页面 URL，默认 file:// 打开项目内 style_localmap.html" })),
      zoomRange: Type.Optional(Type.Array(Type.Number(), { description: "逐级缩放级别数组，如 [14,15,4,10]；需要 style_localmap.html 已暴露 window.map（否则自动跳过并提示）。默认 [10] 不扫描。" })),
      skipServerCheck: Type.Optional(Type.Boolean({ description: "true 则跳过 tile_server /health 检查，默认 false" })),
      waitMs: Type.Optional(Type.Integer({ description: "后每级缩放等待毫秒数，默认 1500" })),
      screenshot: Type.Optional(Type.Boolean({ description: "是否截图到 .pi/tmp/，默认 true" })),
      timeoutMs: Type.Optional(Type.Integer({ description: "页面加载/会话超时毫秒数，默认 60000" })),
      headless: Type.Optional(Type.Boolean({ description: "headless，默认 true" })),
    }),

    async execute(_toolCallId: string, params: any, signal: { aborted?: boolean } | undefined, onUpdate: ((u: any) => void) | undefined, ctx: { cwd: string }) {
      const url = params.url ?? `file://${resolve(ctx.cwd, "style_localmap.html")}`;
      const zoomRange: number[] = params.zoomRange && params.zoomRange.length ? params.zoomRange : [10];
      const waitMs = params.waitMs ?? 1500;
      const wantShot = params.screenshot !== false;
      const timeoutMs = params.timeoutMs ?? 60000;
      const headless = params.headless !== false;

      const report: Record<string, unknown> = { url, zoomCmd: zoomRange };

      // 1) 可选服务器健康检查
      if (!params.skipServerCheck) {
        try {
          const ac = new AbortController();
          const t = setTimeout(() => ac.abort(), 4000);
          const res = await fetch("http://127.0.0.1:3000/health", { signal: ac.signal });
          clearTimeout(t);
          report.server = res.ok ? "ok" : `http_${res.status}`;
        } catch (e) {
          report.server = `down (${(e as Error).message})`;
        }
      } else {
        report.server = "skipped";
      }

      onUpdate?.({ content: [{ type: "text", text: `启动浏览器: ${url}\nserver=${report.server}` }], details: { step: "launch" } });

      const exe = existsSync(HEADLESS_SHELL) ? HEADLESS_SHELL : FULL_CHROME;
      report.browser = headless ? "headless " : "" + (existsSync(HEADLESS_SHELL) ? "chromium-headless-shell" : "chromium");

      const browser = await chromium.launch({ headless, executablePath: exe, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
      try {
        const page = await browser.newPage();

        const pageerrors: string[] = [];
        const consoleMsgs: Array<{ level: string; text: string }> = [];
        const reqFailed: Array<{ url: string; reason: string }> = [];
        const httpErrs: Array<{ status: number; url: string }> = [];
        const tiles: Array<{ kind: string; z: number; x: number; y: number; ok?: boolean; status?: number; reason?: string }> = [];
        const otherFailed: string[] = [];

        await page.addInitScript(MAP_HOOK_SCRIPT);

        page.on("pageerror", (e) =>
          pageerrors.push(`${e.name || "Error"}: ${e.message}${e.stack ? "\n" + e.stack.split("\n").slice(0, 6).join("\n") : ""}`),
        );
        page.on("console", (m) => {
          if (m.type() === "error" || m.type() === "warning") consoleMsgs.push({ level: m.type(), text: m.text() });
        });
        page.on("requestfailed", (req) => {
          const reason = req.failure()?.errorText ?? "failed";
          const mt = req.url().match(/\/tiles\/(\d+)\/(\d+)\/(\d+)/);
          if (mt) tiles.push({ kind: "tile-failed", z: +mt[1], x: +mt[2], y: +mt[3], ok: false, reason });
          else { reqFailed.push({ url: req.url(), reason }); otherFailed.push(`failed: ${absSnippet(req.url())} (${reason})`); }
        });
        page.on("response", (resp) => {
          const u = resp.url();
          if (resp.status() >= 400) httpErrs.push({ status: resp.status(), url: absSnippet(u) });
          const mt = u.match(/\/tiles\/(\d+)\/(\d+)\/(\d+)/);
          if (mt) tiles.push({ kind: "tile-resp", z: +mt[1], x: +mt[2], y: +mt[3], ok: resp.ok(), status: resp.status() });
        });

        // 2) 打开页面
        try {
          await page.goto(url, { timeout: timeoutMs, waitUntil: "load" });
        } catch (e) {
          report.gotoError = (e as Error).message;
        }
        await page.waitForTimeout(waitMs);

        // 3) zoom 扫描（需要 window.map 已在 html 暴露）
        if (zoomRange.length > 1) {
          const swept = await page.evaluate(async (zs: number[]) => {
            const m = (window as any).map;
            if (!m || typeof m.setZoom !== "function" || !m.once) {
              return { swept: false, note: "style_localmap.html 未暴露 window.map，无法控制缩放" };
            }
            const reached: number[] = [];
            for (const z of zs) {
              m.setZoom(z);
              await new Promise<void>((res) => {
                const t = setTimeout(res, 2500);
                try { m.once("idle", () => { clearTimeout(t); res(); }); } catch (_) { clearTimeout(t); res(); }
              });
              reached.push(m.getZoom());
            }
            return { swept: true, reached };
          }, zoomRange);
          report.zoom = swept;
        } else {
          report.zoom = { swept: false, why: "zoomRange 长度=1" };
        }
        await page.waitForTimeout(waitMs);

        // 4) 读取 Map hook 收集的错误
        const mapErrors = await page.evaluate(() => (window as any).__mapErrors || []);

        // 5) 截图
        let shot: string | undefined;
        if (wantShot) {
          mkdirSync(resolve(ctx.cwd, ".pi/tmp"), { recursive: true });
          shot = resolve(ctx.cwd, ".pi/tmp", `map-debug-${Date.now()}.png`);
          try { await page.screenshot({ path: shot }); } catch { shot = undefined; }
        }

        report.pageerrors = pageerrors;
        report.console = consoleMsgs.slice(0, 150);
        report.requestFailed = reqFailed.slice(0, 150);
        report.httpErrors = httpErrs.slice(0, 150);
        report.mapErrors = mapErrors;
        report.tiles = tiles.slice(0, 300);
        report.otherFailed = otherFailed.slice(0, 80);
        report.tilesFetched = tiles.length;
        report.failedTileCount = tiles.filter((t) => t.kind === "tile-failed" || (t.kind === "tile-resp" && t.ok === false)).length;
        report.screenshot = shot;

        // 汇总判定
        const errCount = pageerrors.length + reqFailed.length + httpErrs.length + mapErrors.length;
        report.verdict =
          errCount === 0 ? "CLEAN(未见明显报错)" :
          errCount > 0 ? `ERRORS(${errCount})` : "UNKNOWN";

        const json = JSON.stringify(report, null, 2);
        const tr = truncate(json);

        if (signal?.aborted) {
          return { content: [{ type: "text", text: "map_debug_test 已取消" }], details: { ...report, cancelled: true } };
        }

        return {
          content: [{ type: "text", text: `map_debug_test 完成${tr.truncated ? "（结果截断）" : ""}\n\n${tr.content}` }],
          details: { ...report, truncated: tr.truncated },
        };
      } finally {
        await browser.close();
      }
    },
  });
}

/** 缩短 URL 便于展示 */
function absSnippet(u: string): string {
  return u.length > 160 ? u.slice(0, 160) + "..." : u;
}