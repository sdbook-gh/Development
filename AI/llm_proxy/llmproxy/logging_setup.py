"""日志初始化：控制台 + 滚动文件（保留 3 份备份）。

- ``llmproxy.*`` 各模块 logger 统一输出到两处
- 滚动文件：``server.log_file``（默认 ``logs/llmproxy.log``），默认 5MB/份
- 第三方库降噪：httpx / httpcore 降到 WARNING，避免每请求刷连接日志

另提供 :func:`attempts_chain`：把一次派发的尝试轨迹压成一行
（如 ``bad:500(failover: 500) -> good:ok``），供每请求日志使用。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from .config import ServerConfig

__all__ = ["setup_logging", "attempts_chain"]

# 每份日志文件大小上限；超过滚动，保留 LOG_BACKUPS 份备份
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(cfg: ServerConfig, max_bytes: int = LOG_MAX_BYTES) -> None:
    """按 server 配置初始化 llmproxy 日志（可重复调用，幂等）。

    Args:
        cfg: server 配置（log_level / log_file）
        max_bytes: 滚动文件单份大小上限（测试用小值可验证滚动）
    """
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    root = logging.getLogger("llmproxy")
    root.setLevel(level)
    # 幂等：清掉旧 handler 再挂新的（避免重复加载时重复输出）
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)

    # 第三方库降噪（每次请求都会发 HTTP，INFO 会淹没业务日志）
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("日志初始化完成: level=%s file=%s (单份 %.2fMB, 保留 %d 份)",
              cfg.log_level, log_path, max_bytes / (1024 * 1024), LOG_BACKUPS)


def attempts_chain(attempts: list) -> str:
    """尝试轨迹压成一行：``bad:http_error:500(failover: 500) -> good:ok:200``。

    被跳过的候选也保留（``bad:skipped(熔断准入未通过)``），便于排查切换原因。
    """
    parts: list[str] = []
    for a in attempts:
        seg = f"{a.upstream}:{a.outcome}"
        if a.status is not None:
            seg += f":{a.status}"
        if a.note:
            seg += f"({a.note})"
        parts.append(seg)
    return " -> ".join(parts) if parts else "-"
