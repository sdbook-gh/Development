// 端到端验证：镜像 index.ts 的浏览器抓取逻辑，打开真实页面
import { chromium } from "playwright";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const MAP_HOOK_SCRIPT = `(() => {
  window.__mapErrors = [];
  try {
    let real = undefined;
    Object.defineProperty(window, "maplibregl", {
      configurable: true, enumerable: true,
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
                m.on("error", (e) => window.__mapErrors.push({ type: "map-error", msg: (e && e.error && (e.error.message || e.error)) || JSON.stringify(e) }));
                m.on("tileerror", (e) => window.__mapErrors.push({ type: "tileerror", tile: (e && e.tileID && e.tileID.canonical) ? String(e.tileID.canonical) : "" }));
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

const BROWSERS_DIR = resolve(dirname(fileURLToPath(import.meta.url)));
const SHELL = resolve(BROWSERS_DIR, "browsers/chrome-headless-shell-linux64/chrome-headless-shell");
const CHROME = resolve(BROWSERS_DIR, "browsers/chrome-linux64/chrome");
const exe = existsSync(SHELL) ? SHELL : CHROME;

const url = `file://${resolve(process.cwd(), "style_localmap.html")}`;
const report = { url, browser: "chromium-headless-shell" };

const browser = await chromium.launch({ headless: true, executablePath: exe, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
try {
  const page = await browser.newPage();
  const pageerrors = [], consoleMsgs = [], reqFailed = [], httpErrs = [], tiles = [], otherFailed = [];
  await page.addInitScript(MAP_HOOK_SCRIPT);
  page.on("pageerror", (e) => pageerrors.push(`${e.name || "Error"}: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") consoleMsgs.push({ level: m.type(), text: m.text() }); });
  page.on("requestfailed", (req) => {
    const reason = req.failure()?.errorText ?? "failed";
    const mt = req.url().match(/\/tiles\/(\d+)\/(\d+)\/(\d+)/);
    if (mt) tiles.push({ kind: "tile-failed", z: +mt[1], x: +mt[2], y: +mt[3], reason });
    else { reqFailed.push({ url: req.url(), reason }); otherFailed.push(`failed: ${req.url()} (${reason})`); }
  });
  page.on("response", (resp) => {
    const u = resp.url();
    if (resp.status() >= 400) httpErrs.push({ status: resp.status(), url: u.slice(0, 160) });
    const mt = u.match(/\/tiles\/(\d+)\/(\d+)\/(\d+)/);
    if (mt) tiles.push({ kind: "tile-resp", z: +mt[1], x: +mt[2], y: +mt[3], ok: resp.ok(), status: resp.status() });
  });

  try { await page.goto(url, { timeout: 60000, waitUntil: "load" }); } catch (e) { report.gotoError = e.message; }
  await page.waitForTimeout(3000);

  const mapErrors = await page.evaluate(() => window.__mapErrors || []);
  mkdirSync(resolve(process.cwd(), ".pi/tmp"), { recursive: true });
  const shot = resolve(process.cwd(), ".pi/tmp/map-debug-e2e.png");
  try { await page.screenshot({ path: shot }); } catch { }

  report.pageerrors = pageerrors;
  report.console = consoleMsgs.slice(0, 60);
  report.requestFailed = reqFailed.slice(0, 60);
  report.httpErrors = httpErrs.slice(0, 60);
  report.mapErrors = mapErrors;
  report.tiles = tiles.slice(0, 60);
  report.otherFailed = otherFailed.slice(0, 30);
  report.tilesFetched = tiles.length;
  report.screenshot = shot;

  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}