#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pi_agent.py — 自动读取 pi agent 配置文件，用最少的 token 验证指定 LLM 是否可用。

用法示例:
    python3 test_pi_agent.py --provider opencode --model hy3-free \\
        --thinking high --api-key sk-xxxx

读取位置（按顺序自动探测）:
    - 配置文件 models.json : ~/.pi/agent/models.json
    - 密钥文件 auth.json  : ~/.pi/agent/auth.json
    - 设置文件 settings.json : ~/.pi/agent/settings.json
也可用 --config / --auth 显式指定路径。

校验原理:
    用 max_tokens=1、prompt="ping" 的最小请求打一次 {baseUrl}/chat/completions，
    返回 HTTP 200 且响应包含合法 choices 即判定可用。
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from urllib.parse import urljoin

DEFAULT_CONFIG = os.path.expanduser("~/.pi/agent/models.json")
DEFAULT_AUTH = os.path.expanduser("~/.pi/agent/auth.json")
DEFAULT_SETTINGS = os.path.expanduser("~/.pi/agent/settings.json")


def log(msg):
    print(msg, file=sys.stderr)


def resolve_value(value, errors):
    """解析 apiKey/headers 的值: 支持 !cmd / $ENV / ${ENV} / 字面量。"""
    if not isinstance(value, str):
        return value
    # `!command` 执行命令取 stdout
    if value.startswith("!") and not value.startswith(("$$", "$!")):
        try:
            out = subprocess.run(
                value[1:], shell=True, capture_output=True, text=True, timeout=20
            )
            if out.returncode == 0:
                return out.stdout.strip()
            errors.append("命令执行失败: %s" % value)
            return None
        except Exception as e:
            errors.append("命令执行异常: %s (%s)" % (value, e))
            return None
    # `$$` -> 字面量 `$`
    if value.startswith("$$"):
        return value[1:]
    # `$!` -> 字面量 `!`
    if value.startswith("$!"):
        return value[1:]
    # ${ENV} / $ENV 环境变量插值
    if "$" in value:
        def _expand(m):
            return os.environ.get(m.group(1), "")
        import re
        v = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: _expand(m), value)
        return v
    return value


def load_json(path, name):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log("[warn] 未找到%s: %s" % (name, path))
        return None
    except Exception as e:
        log("[warn] 读取%s失败: %s (%s)" % (name, path, e))
        return None


class AuthSet:
    """采集所有候选密钥来源并确定最终可用密钥。"""

    def __init__(self, args, provider, model_cfg, provider_cfg):
        self.errors = []
        self.provider = provider
        self.api_key = None
        self.api_key_source = "未设置"

        candidates = []

        # 1) --api-key 最高优先级
        if args.api_key:
            candidates.append((args.api_key, "命令行 --api-key"))

        # 2) auth.json 中该 provider 的密钥
        auth = load_json(args.auth, "auth.json")
        if auth and provider in auth:
            entry = auth[provider]
            key = entry.get("key") or entry.get("access")
            if isinstance(key, str) and key:
                candidates.append((key, "auth.json[%s]" % provider))

        # 3) models.json 中 provider / model 的 apiKey（含 env/命令展开）
        for cfg, name in ((provider_cfg, "provider"), (model_cfg, "model")):
            if cfg and isinstance(cfg, dict) and cfg.get("apiKey"):
                resolved = resolve_value(cfg["apiKey"], self.errors)
                if resolved:
                    candidates.append((resolved, "models.json[%s].apiKey" % name))

        # 4) 常见环境变量兜底
        for env in ("OPENAI_API_KEY", "%s_API_KEY" % provider.upper()):
            if os.environ.get(env):
                candidates.append((os.environ[env], "环境变量 %s" % env))
                break

        for key, src in candidates:
            if isinstance(key, str) and key and "none" != key.lower():
                self.api_key = key
                self.api_key_source = src
                return  # 取第一个有效项

    def header(self):
        return {"Authorization": "Bearer %s" % self.api_key}


def build_request_body(model, api):
    """按 api 类型构造最简请求体。"""
    if api == "anthropic-messages":
        return {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    if api == "google-generative-ai":
        return {
            "model": model,
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }
    if api == "openai-responses":
        return {
            "model": model,
            "input": "ping",
            "max_output_tokens": 1,
        }
    # 默认 openai-completions
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }


def build_url(base_url, api):
    if api == "google-generative-ai":
        # 让其指向 generateContent 端点
        endpoint = "models/%s:generateContent" % "{{MODEL}}"  # 稍后替换模式
        return base_url.rstrip("/") + "/models/:generateContent"
    if api == "anthropic-messages":
        return urljoin(base_url.rstrip("/") + "/", "v1/messages")
    return urljoin(base_url.rstrip("/") + "/", "chat/completions")


def do_request(url, headers, body, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        status = resp.status
    return status, raw


def parse_result(response_text, api):
    try:
        obj = json.loads(response_text)
    except Exception as e:
        return False, None, "响应不是合法 JSON: %s" % e
    if api == "anthropic-messages":
        content = obj.get("content") or []
        ok = bool(content)
        snippet = json.dumps(content)[:120] if content else ""
        return ok, "", snippet
    if api == "openai-responses" or api == "openai-completions":
        choices = obj.get("choices") or []
        ok = bool(choices)
        snippet = ""
        if choices:
            msg = choices[0].get("message") or {}
            snippet = json.dumps(msg.get("content") or msg)[:120]
        return ok, "", snippet
    if api == "google-generative-ai":
        candidates = obj.get("candidates") or []
        ok = bool(candidates)
        return ok, "", json.dumps(candidates)[:120] if candidates else ""
    return False, "", "未知 api 类型: %s" % api


def main():
    ap = argparse.ArgumentParser(
        description="用最少的 token 验证 pi agent 配置的 LLM 是否可用"
    )
    ap.add_argument("--provider", dest="provider", help="provider 名称（默认取 settings.json 的 defaultProvider）")
    ap.add_argument("--model", dest="model", help="模型 ID（默认取 settings.json 的 defaultModel）")
    ap.add_argument("--thinking", dest="thinking", default=None, help="思考级别 high/medium/low")
    ap.add_argument("--api-key", dest="api_key", help="API 密钥（最高优先级）")
    ap.add_argument("--config", dest="config", default=DEFAULT_CONFIG, help="models.json 路径")
    ap.add_argument("--auth", dest="auth", default=DEFAULT_AUTH, help="auth.json 路径")
    ap.add_argument("--settings", dest="settings", default=DEFAULT_SETTINGS, help="settings.json 路径")
    ap.add_argument("--timeout", dest="timeout", type=int, default=30, help="请求超时秒数")
    ap.add_argument("--models", dest="models_opt", action="store_true", help="仅列出已配置的 provider/model 后退出")
    args = ap.parse_args()

    # 读取配置文件
    config = load_json(args.config, "models.json")
    settings = load_json(args.settings, "settings.json")

    if args.models_opt:
        if config and config.get("providers"):
            for pname, pc in config["providers"].items():
                base = pc.get("baseUrl", "?")
                models = [m.get("id") for m in (pc.get("models") or [])]
                print("%s -> %s %s" % (pname, base, models))
        return 0

    if not config or not config.get("providers"):
        print("❌ 错误: 无法读取有效的 models.json（%s）" % args.config)
        return 1

    providers = config["providers"]

    # 确定 provider
    provider = args.provider or (settings or {}).get("defaultProvider")
    if not provider:
        print("❌ 错误: 未指定 --provider，且 settings.json 无 defaultProvider")
        return 1
    if provider not in providers:
        print("❌ 错误: provider '%s' 不在 models.json 中。可用: %s" % (
            provider, list(providers.keys())))
        return 1
    provider_cfg = providers[provider]

    # 确定 model
    available_models = provider_cfg.get("models") or []
    model = args.model or (settings or {}).get("defaultModel") or \
        (available_models[0]["id"] if available_models else None)
    if not model:
        print("❌ 错误: 未指定 --model，且无法从配置推断 model")
        return 1

    model_cfg = None
    for m in available_models:
        if m.get("id") == model or m.get("id") == provider + "/" + model:
            model_cfg = m
            break
    # 模型不在配置中也允许验证（直接使用传入的 --model 发起请求）
    if model_cfg is None:
        print("   [提示] model '%s' 不在 provider '%s' 配置中，将直接使用该 ID 验证。" % (
            model, provider))

    # api 类型
    api = provider_cfg.get("api") or (model_cfg or {}).get("api") or "openai-completions"

    # baseUrl
    base_url = provider_cfg.get("baseUrl")
    if not base_url:
        print("❌ 错误: provider '%s' 未配置 baseUrl。请先配置 models.json。" % provider)
        return 1

    # 密钥
    auths = AuthSet(args, provider, model_cfg, provider_cfg)

    # 组装请求
    body = build_request_body(model, api)
    url = build_url(base_url, api)

    if not auths.api_key:
        print("❌ 错误: 未找到 %s 的有效 API 密钥（--api-key / auth.json / models.json / 环境变量均未命中）" % provider)
        return 1

    headers = {"Content-Type": "application/json"}
    headers.update(auths.header())
    # 附带 UA
    headers.setdefault("User-Agent", "test_pi_agent/1.0")

    if True:
        print("🔎 目标: provider=%s model=%s api=%s" % (provider, model, api))
        print("   baseUrl = %s" % base_url)
        print("   url     = %s" % url)
        print("   apiKey 来源 = %s" % auths.api_key_source)
        print("   thinking = %s" % (args.thinking or "未指定"))

    # 发起请求
    try:
        status, raw = do_request(url, headers, body, args.timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        print("❌ 错误: HTTP %s %s" % (e.code, e.reason))
        if detail:
            print("   响应体: %s" % detail)
        return 1
    except urllib.error.URLError as e:
        print("❌ 错误: 连接失败 - %s" % e.reason)
        return 1
    except Exception as e:
        print("❌ 错误: 请求异常 - %s" % e)
        return 1

    text = raw.decode("utf-8", "replace")
    if status == 200:
        ok, _, snippet = parse_result(text, api)
        if ok:
            print("✅ 正确: LLM 可用 (provider=%s model=%s, HTTP 200)" % (provider, model))
            if snippet:
                print("   回复片段: %s" % snippet)
            return 0
        else:
            print("❌ 错误: HTTP 200 但响应无有效 choices/content（可能是密钥或模型名错误且服务返回空）")
            print("   响应体: %s" % text[:300])
            return 1
    else:
        print("❌ 错误: HTTP %s" % status)
        print("   响应体: %s" % text[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
