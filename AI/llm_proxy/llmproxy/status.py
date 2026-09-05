"""/status JSON 聚合。

每 upstream 输出：健康状态（state/连续失败/累计失败/恢复次数/冷却剩余/
最近错误）+ 并发（inflight）+ 限速状态（mode/窗口占用/等待数）。
另附全局 summary（各状态计数、总并发、熔断中列表）与路由表。
"""

from __future__ import annotations

import os
import time

from .config import AppConfig
from .router import Router

__all__ = ["aggregate_status"]


def _config_state(cfg: AppConfig) -> dict:
    """配置文件加载信息与过期检测（磁盘 mtime 晚于加载时刻 → stale）。"""
    out: dict = {"config_path": cfg.config_path}
    if cfg.config_mtime is not None:
        out["loaded_mtime"] = round(cfg.config_mtime, 3)
    if cfg.config_path is None or cfg.config_mtime is None:
        return out
    try:
        disk_mtime = os.stat(cfg.config_path).st_mtime
    except OSError:
        out["config_stale"] = None  # 文件已不可访问，无法判断
        return out
    out["disk_mtime"] = round(disk_mtime, 3)
    out["config_stale"] = disk_mtime > cfg.config_mtime + 1e-6
    return out


def aggregate_status(cfg: AppConfig, router: Router, started_at: int) -> dict:
    """组装 /status 响应（纯聚合，不做 IO）。"""
    upstreams = router.snapshot()

    state_counts: dict[str, int] = {}
    inflight_total = 0
    cooling: list[str] = []
    for name, snap in upstreams.items():
        state = snap["health"]["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
        inflight_total += snap["inflight"]
        if state == "open":
            cooling.append(name)

    return {
        "server": {
            "host": cfg.server.host,
            "port": cfg.server.port,
            "log_level": cfg.server.log_level,
            "log_file": cfg.server.log_file,
            "started_at": started_at,
            "uptime_s": int(time.time() - started_at),
            "pid": os.getpid(),
        },
        "config": _config_state(cfg),
        "proxies": dict(cfg.proxies),
        "summary": {
            "upstreams_total": len(upstreams),
            "health_states": state_counts,
            "cooling": cooling,
            "inflight_total": inflight_total,
            "virtual_models": len(router.route_snapshot()),
        },
        "upstreams": upstreams,
        "routes": router.route_snapshot(),
    }
