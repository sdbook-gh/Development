"""
freebuff_logger.py — mitmproxy addon：只读记录 freebuff 全部流量。

- 控制台：实时彩色输出（敏感头打码）
- 文件：logs/session_*/  下 session.log（完整文本日志）+ 每请求一个完整 JSON
      （含未打码 headers / 完整 body / SSE 全文，仅供本地分析）

用法（由 start.sh 启动）:
    mitmdump -p 8888 --mode regular --set confdir=ca -s freebuff_logger.py
"""
import json
import os
import time
from datetime import datetime

from mitmproxy import http

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONSOLE_BODY_LIMIT = 2000
SSE_LINE_LIMIT = 500

# ANSI
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"

TAG_COLORS = {
    "CODEBUFF-API": BOLD + MAGENTA,
    "OPENROUTER": BOLD + GREEN,
    "POSTHOG": DIM,
    "SENTRY": DIM,
    "LOCAL": DIM,
    "OTHER": CYAN,
}

SENSITIVE_HEADERS = {
    "authorization", "proxy-authorization", "x-api-key",
    "cookie", "set-cookie", "x-auth-token", "x-codebuff-token",
}


def classify(host: str) -> str:
    h = (host or "").lower()
    if "codebuff" in h:
        return "CODEBUFF-API"
    if "openrouter" in h:
        return "OPENROUTER"
    if "posthog" in h:
        return "POSTHOG"
    if "sentry" in h:
        return "SENTRY"
    if h in ("127.0.0.1", "localhost", "::1"):
        return "LOCAL"
    return "OTHER"


def mask_header(name: str, value: str) -> str:
    if name.lower() in SENSITIVE_HEADERS:
        v = value.strip()
        return v[:8] + "***" if len(v) > 8 else "***"
    return value


class FreebuffLogger:
    def __init__(self):
        self.seq = 0
        self._seq_by_fid = {}
        self._req_start = {}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(SCRIPT_DIR, "logs", f"session_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.log_path = os.path.join(self.session_dir, "session.log")
        self._logf = open(self.log_path, "a", encoding="utf-8")
        print(f"{BOLD}=== freebuff_logger ==={RESET}")
        print(f"会话目录: {self.session_dir}")
        print(f"文本日志: {self.log_path}")
        print()

    def _log(self, text: str, color: str = ""):
        line = f"[{datetime.now():%H:%M:%S.%f}[:-3]] {text}"
        print(f"{color}{line}{RESET}" if color else line)
        self._logf.write(line + "\n")
        self._logf.flush()

    def _body_pretty(self, raw: bytes | None) -> str | None:
        if not raw:
            return None
        text = raw.decode("utf-8", errors="replace")
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            return text

    def _print_body(self, pretty: str | None):
        if pretty is None:
            self._log("    (body 为空)", DIM)
            return
        if len(pretty) <= CONSOLE_BODY_LIMIT:
            shown = pretty
        else:
            shown = pretty[:CONSOLE_BODY_LIMIT] + f"\n    ... [截断，完整内容 {len(pretty)} 字符见 JSON 文件]"
        for ln in shown.splitlines():
            self._log("    " + ln, DIM)

    def _log_headers(self, headers):
        for k, v in headers.items():
            self._log(f"    {k}: {mask_header(k, v)}", DIM)

    # ---- hooks -------------------------------------------------------------

    def request(self, flow: http.HTTPFlow):
        self.seq += 1
        seq = self.seq
        self._seq_by_fid[flow.id] = seq
        self._req_start[flow.id] = time.time()
        req = flow.request
        tag = classify(req.pretty_host)
        color = TAG_COLORS.get(tag, CYAN)
        self._log(f"#{seq:03d} >>> REQUEST  [{tag}] {req.method} {req.pretty_url}", color)
        self._log_headers(req.headers)
        self._log("    -- request body --", DIM)
        raw = req.get_content()
        pretty = self._body_pretty(raw)
        self._print_body(pretty)
        # 高亮 model 字段
        if pretty:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "model" in obj:
                    self._log(f"    ** model: {obj['model']} **", BOLD + YELLOW)
            except Exception:
                pass
        self._log("", color)

    def response(self, flow: http.HTTPFlow):
        seq = self._seq_by_fid.get(flow.id, 0)
        start = self._req_start.pop(flow.id, None)
        elapsed = round(time.time() - start, 3) if start else None
        req, resp = flow.request, flow.response
        tag = classify(req.pretty_host)
        color = TAG_COLORS.get(tag, CYAN)
        size = len(resp.raw_content or b"")
        ct = resp.headers.get("content-type", "")
        self._log(f"#{seq:03d} <<< RESPONSE [{tag}] {resp.status_code} {resp.reason or ''}"
                  f" | {elapsed}s | {size}B | {ct}", color)
        self._log_headers(resp.headers)
        self._log("    -- response body --", DIM)
        raw = resp.get_content()
        if "event-stream" in ct:
            self._log_sse(seq, req.pretty_host, raw)
        else:
            self._print_body(self._body_pretty(raw))
        self._save_flow(flow, tag, seq, elapsed)
        self._log("", color)

    def _log_sse(self, seq: int, host: str, raw: bytes | None):
        text = (raw or b"").decode("utf-8", errors="replace")
        events, cur = [], {}
        for ln in text.splitlines():
            if ln.startswith("data:"):
                cur.setdefault("data", []).append(ln[5:].lstrip())
            elif ln.startswith("event:"):
                cur["event"] = ln[6:].strip()
            elif ln.startswith("id:"):
                cur["id"] = ln[3:].strip()
            elif ln.strip() == "" and cur:
                events.append(cur)
                cur = {}
        if cur:
            events.append(cur)
        self._log(f"    ** SSE 流: {len(events)} 个事件 **", BOLD + YELLOW)
        for i, ev in enumerate(events):
            data = "\n".join(ev.get("data", []))
            name = ev.get("event", "message")
            shown = data if len(data) <= SSE_LINE_LIMIT else data[:SSE_LINE_LIMIT] + "…"
            self._log(f"    [evt {i + 1}/{len(events)}] {name}: {shown}", YELLOW)
        sse_path = os.path.join(self.session_dir, f"sse_{seq:03d}_{host.replace('/', '_')}.txt")
        with open(sse_path, "w", encoding="utf-8") as f:
            f.write(text)

    def error(self, flow: http.HTTPFlow):
        seq = self._seq_by_fid.get(flow.id, 0)
        tag = classify(flow.request.pretty_host)
        self._log(f"#{seq:03d} !!! ERROR [{tag}] {flow.error}", RED)

    def done(self):
        self._logf.close()
        print(f"\n{BOLD}=== 会话结束，日志在: {self.session_dir} ==={RESET}")

    def _save_flow(self, flow: http.HTTPFlow, tag: str, seq: int, elapsed):
        req, resp = flow.request, flow.response
        record = {
            "seq": seq,
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "request": {
                "method": req.method,
                "url": req.pretty_url,
                "headers": {k: v for k, v in req.headers.items()},
                "body_text": req.get_text() or "",
            },
            "response": {
                "status": resp.status_code,
                "reason": resp.reason,
                "elapsed_s": elapsed,
                "content_type": resp.headers.get("content-type"),
                "headers": {k: v for k, v in resp.headers.items()},
                "body_text": resp.get_text() or "",
            },
        }
        host = req.pretty_host.replace("/", "_").replace(":", "_")
        path = os.path.join(self.session_dir, f"{seq:03d}_{host}_{req.method}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


addons = [FreebuffLogger()]
