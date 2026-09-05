"""配置加载与校验。

从 YAML 文件读取配置（见 README.md「配置文件」章节），支持：

- ``${ENV_VAR}`` 形式的环境变量引用（api_key 等避免明文入库）
- 类型/取值校验与引用完整性校验（upstream / 代理名 / 策略枚举）
- 未知键报错，防止配置项拼写错误静默失效

用法::

    cfg = load_config("config.yaml")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "ServerConfig",
    "RateLimitConfig",
    "UpstreamConfig",
    "RetryConfig",
    "VirtualModelConfig",
    "AppConfig",
    "load_config",
]

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: 429 等错误默认可切换；400 默认包含（部分 provider 对超长上下文返回 400）
DEFAULT_RETRYABLE_STATUS = [408, 429, 500, 502, 503, 504, 400]

STRATEGIES = ("round_robin", "least_load", "failover", "weighted_random")
RATE_LIMIT_MODES = ("none", "min_interval", "rpm")
LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(ValueError):
    """配置文件缺失、格式错误或校验失败。"""


def _substitute_env(value: Any, where: str) -> Any:
    """递归展开 ``${VAR}`` 环境变量引用。未设置的变量视为配置错误。"""
    if isinstance(value, str):
        def _repl(m: re.Match[str]) -> str:
            var = m.group(1)
            env = os.environ.get(var)
            if env is None:
                raise ConfigError(f"{where}: 环境变量 ${{{var}}} 未设置")
            return env

        return _ENV_VAR_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v, where) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v, where) for v in value]
    return value


def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: 应为 mapping，实际为 {type(value).__name__}")
    return value


def _check_unknown_keys(data: dict, known: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where}: 未知配置项 {unknown}（已知: {list(known)}）")


def _get_int(data: dict, key: str, where: str, default: int | None = None,
             minimum: int | None = None) -> int:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"{where}: 缺少必填项 '{key}'")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: '{key}' 应为整数，实际为 {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: '{key}' 不能小于 {minimum}，实际为 {value}")
    return value


def _parse_headers(data: Any, where: str) -> dict[str, str]:
    """解析 upstream 扩展头：mapping，键为非空头名字符串，值为字符串。

    允许值引用 ``${ENV_VAR}``（由 ``_substitute_env`` 递归展开），也允许空字符串值。
    """
    headers = _require_mapping(data, where)
    out: dict[str, str] = {}
    for k, v in headers.items():
        if not isinstance(k, str) or not k.strip() or k != k.strip():
            raise ConfigError(
                f"{where}: 头名应为非空且无首尾空白的字符串，实际为 {k!r}"
            )
        if any(ch.isspace() or ord(ch) < 0x20 for ch in k):
            raise ConfigError(f"{where}: 头名 {k!r} 含空白/控制字符，非法")
        if not isinstance(v, str):
            raise ConfigError(f"{where}: 头 '{k}' 的值应为字符串，实际为 {v!r}")
        out[k] = v
    return out


def _get_str(data: dict, key: str, where: str, default: str | None = None,
             allowed: tuple[str, ...] | None = None, allow_empty: bool = False) -> str:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"{where}: 缺少必填项 '{key}'")
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ConfigError(f"{where}: '{key}' 应为非空字符串，实际为 {value!r}")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"{where}: '{key}' 应为 {list(allowed)} 之一，实际为 {value!r}")
    return value


# ---------------------------------------------------------------- server ----

@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 4400  # 默认监听端口；命令行 --port 可覆盖
    log_level: str = "info"
    log_file: str = "logs/llmproxy.log"

    @classmethod
    def from_dict(cls, data: Any, where: str = "server") -> "ServerConfig":
        data = _require_mapping(data, where)
        _check_unknown_keys(data, ("host", "port", "log_level", "log_file"), where)
        host = _get_str(data, "host", where, default="127.0.0.1")
        port = _get_int(data, "port", where, default=4400, minimum=1)
        if port > 65535:
            raise ConfigError(f"{where}: 'port' 不能大于 65535，实际为 {port}")
        log_level = _get_str(data, "log_level", where, default="info", allowed=LOG_LEVELS)
        log_file = _get_str(data, "log_file", where, default="logs/llmproxy.log")
        return cls(host=host, port=port, log_level=log_level, log_file=log_file)


# ------------------------------------------------------------ rate_limit ----

@dataclass
class RateLimitConfig:
    """每 upstream 独立频率控制。"""

    mode: str = "none"           # none | min_interval | rpm
    min_interval_ms: int = 0     # mode=min_interval 时必填
    rpm: int = 0                 # mode=rpm 时必填

    @classmethod
    def from_dict(cls, data: Any, where: str = "rate_limit") -> "RateLimitConfig":
        data = _require_mapping(data, where)
        _check_unknown_keys(data, ("mode", "min_interval_ms", "rpm"), where)
        mode = _get_str(data, "mode", where, default="none", allowed=RATE_LIMIT_MODES)
        cfg = cls(mode=mode)
        if mode == "min_interval":
            cfg.min_interval_ms = _get_int(data, "min_interval_ms", where, minimum=1)
        elif mode == "rpm":
            cfg.rpm = _get_int(data, "rpm", where, minimum=1)
        return cfg


# ------------------------------------------------------------- upstreams ----

@dataclass
class UpstreamConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    proxy: str = "none"          # none=直连；代理名=走该代理
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    weight: int = 1
    headers: dict[str, str] = field(default_factory=dict)   # 扩展头（后合并，可覆盖默认头）

    @classmethod
    def from_dict(cls, data: Any, where: str) -> "UpstreamConfig":
        data = _require_mapping(data, where)
        _check_unknown_keys(
            data, ("name", "base_url", "api_key", "model", "proxy", "rate_limit", "weight",
                  "headers"), where
        )
        rate_limit = RateLimitConfig.from_dict(
            data.get("rate_limit", {}), f"{where}.rate_limit"
        )
        return cls(
            name=_get_str(data, "name", where),
            base_url=_get_str(data, "base_url", where),
            api_key=_get_str(data, "api_key", where, default="", allow_empty=True),
            model=_get_str(data, "model", where),
            proxy=_get_str(data, "proxy", where, default="none"),
            rate_limit=rate_limit,
            weight=_get_int(data, "weight", where, default=1, minimum=1),
            headers=_parse_headers(data.get("headers", {}), f"{where}.headers"),
        )


# -------------------------------------------------------- virtual_models ----

@dataclass
class RetryConfig:
    max_attempts: int = 3
    retryable_status: tuple[int, ...] = tuple(DEFAULT_RETRYABLE_STATUS)
    cooldown_seconds: float = 30.0
    failure_threshold: int = 3
    # 非 429 失败使熔断进入 OPEN 后，用最小请求（"ping", max_tokens=probe_max_tokens）
    # 主动探测 upstream 是否恢复：成功立即闭合熔断；失败维持原冷却且 0 token 消耗
    active_probe: bool = True
    probe_max_tokens: int = 3   # 探测请求输出上限；默认 3 以满足要求 max_tokens>2 的 provider

    @classmethod
    def from_dict(cls, data: Any, where: str = "retry") -> "RetryConfig":
        data = _require_mapping(data, where)
        _check_unknown_keys(
            data, ("max_attempts", "retryable_status", "cooldown_seconds", "failure_threshold",
                   "active_probe", "probe_max_tokens"),
            where,
        )
        cfg = cls()
        cfg.max_attempts = _get_int(data, "max_attempts", where, default=3, minimum=1)
        raw_status = data.get("retryable_status", list(DEFAULT_RETRYABLE_STATUS))
        if not isinstance(raw_status, list) or not raw_status:
            raise ConfigError(f"{where}: 'retryable_status' 应为非空整数列表，实际为 {raw_status!r}")
        statuses: list[int] = []
        for s in raw_status:
            if isinstance(s, bool) or not isinstance(s, int) or not (100 <= s <= 599):
                raise ConfigError(f"{where}: 'retryable_status' 含非法状态码 {s!r}（应为 100-599 整数）")
            statuses.append(s)
        cfg.retryable_status = tuple(statuses)
        # cooldown_seconds 允许小数（秒）：与 RetryConfig.cooldown_seconds: float 类型一致
        raw_cooldown = data.get("cooldown_seconds", 30)
        if isinstance(raw_cooldown, bool) or not isinstance(raw_cooldown, (int, float)):
            raise ConfigError(
                f"{where}: 'cooldown_seconds' 应为数字（秒，可为小数），实际为 {raw_cooldown!r}"
            )
        if raw_cooldown < 0:
            raise ConfigError(f"{where}: 'cooldown_seconds' 不能为负，实际为 {raw_cooldown}")
        cfg.cooldown_seconds = float(raw_cooldown)
        cfg.failure_threshold = _get_int(
            data, "failure_threshold", where, default=3, minimum=1
        )
        raw_probe = data.get("active_probe", True)
        if not isinstance(raw_probe, bool):
            raise ConfigError(f"{where}: 'active_probe' 应为布尔值，实际为 {raw_probe!r}")
        cfg.active_probe = raw_probe
        cfg.probe_max_tokens = _get_int(
            data, "probe_max_tokens", where, default=3, minimum=1
        )
        return cfg


@dataclass
class VirtualModelConfig:
    name: str                    # 与 virtual_models 的键一致
    strategy: str = "failover"
    upstreams: list[str] = field(default_factory=list)
    retry: RetryConfig = field(default_factory=RetryConfig)

    @classmethod
    def from_dict(cls, data: Any, name: str) -> "VirtualModelConfig":
        where = f"virtual_models.{name}"
        data = _require_mapping(data, where)
        _check_unknown_keys(data, ("strategy", "upstreams", "retry"), where)
        raw_upstreams = data.get("upstreams")
        if not isinstance(raw_upstreams, list) or not raw_upstreams:
            raise ConfigError(f"{where}: 'upstreams' 应为非空列表，实际为 {raw_upstreams!r}")
        upstreams: list[str] = []
        for u in raw_upstreams:
            if not isinstance(u, str) or not u:
                raise ConfigError(f"{where}: 'upstreams' 应为非空字符串列表，实际为 {raw_upstreams!r}")
            upstreams.append(u)
        return cls(
            name=name,
            strategy=_get_str(data, "strategy", where, default="failover", allowed=STRATEGIES),
            upstreams=upstreams,
            retry=RetryConfig.from_dict(data.get("retry", {}), f"{where}.retry"),
        )


# ------------------------------------------------------------ top-level ----

@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    proxies: dict[str, str] = field(default_factory=dict)          # 代理名 -> URL
    upstreams: list[UpstreamConfig] = field(default_factory=list)
    virtual_models: dict[str, VirtualModelConfig] = field(default_factory=dict)
    # 加载来源（load_config 填写）；/status 据此检测“改了配置但未重启”
    config_path: str | None = None
    config_mtime: float | None = None

    def get_upstream(self, name: str) -> UpstreamConfig:
        for u in self.upstreams:
            if u.name == name:
                return u
        raise ConfigError(f"upstream '{name}' 不存在")

    def get_proxy_url(self, proxy_ref: str) -> str | None:
        """解析 upstream 的代理引用：``none`` 返回 None（直连），其余查 proxies 表。"""
        if proxy_ref == "none":
            return None
        try:
            return self.proxies[proxy_ref]
        except KeyError:
            raise ConfigError(
                f"upstream 引用的代理 '{proxy_ref}' 未在 proxies 中定义"
                f"（已定义: {list(self.proxies)}）"
            ) from None

    @classmethod
    def from_dict(cls, data: Any) -> "AppConfig":
        data = _require_mapping(data, "（根节点）")
        _check_unknown_keys(data, ("server", "proxies", "upstreams", "virtual_models"), "（根节点）")

        cfg = cls()
        cfg.server = ServerConfig.from_dict(data.get("server", {}))

        raw_proxies = data.get("proxies", {})
        raw_proxies = _require_mapping(raw_proxies, "proxies")
        for name, url in raw_proxies.items():
            if not isinstance(url, str) or not url:
                raise ConfigError(f"proxies.{name}: 应为非空字符串 URL，实际为 {url!r}")
            cfg.proxies[name] = url

        raw_upstreams = data.get("upstreams")
        if not isinstance(raw_upstreams, list) or not raw_upstreams:
            raise ConfigError("'upstreams' 应为非空列表")
        seen: set[str] = set()
        for i, item in enumerate(raw_upstreams):
            where = f"upstreams[{i}]"
            up = UpstreamConfig.from_dict(item, where)
            if up.name in seen:
                raise ConfigError(f"{where}: upstream 名称 '{up.name}' 重复")
            seen.add(up.name)
            cfg.upstreams.append(up)

        # 校验 upstream 的代理引用可解析
        for up in cfg.upstreams:
            if up.proxy != "none" and up.proxy not in cfg.proxies:
                raise ConfigError(
                    f"upstream '{up.name}' 引用的代理 '{up.proxy}' 未在 proxies 中定义"
                    f"（已定义: {list(cfg.proxies)}）"
                )

        raw_vms = data.get("virtual_models")
        raw_vms = _require_mapping(raw_vms, "virtual_models")
        if not raw_vms:
            raise ConfigError("'virtual_models' 不能为空")
        for name, item in raw_vms.items():
            vm = VirtualModelConfig.from_dict(item, name)
            for ref in vm.upstreams:
                if ref not in seen:
                    raise ConfigError(
                        f"virtual_models.{name}: 引用的 upstream '{ref}' 不存在"
                        f"（已定义: {sorted(seen)}）"
                    )
            cfg.virtual_models[name] = vm

        return cfg


def load_config(path: str | Path) -> AppConfig:
    """加载并校验配置文件。文件缺失 / YAML 语法错误 / 校验失败均抛 :class:`ConfigError`。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"配置文件 YAML 语法错误 ({path}): {e}") from e
    if raw is None:
        raise ConfigError(f"配置文件为空: {path}")
    raw = _substitute_env(raw, str(path))
    cfg = AppConfig.from_dict(raw)
    cfg.config_path = str(path)
    cfg.config_mtime = path.stat().st_mtime
    return cfg
