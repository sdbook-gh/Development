"""每 upstream 独立频率控制。

三种模式（对应 config.yaml 的 ``rate_limit``）：

- ``none``         不控制
- ``min_interval`` 最小间隔：相邻两次放行至少间隔 N 毫秒，超出即排队等待（FIFO）
- ``rpm``          滑动窗口配额：任意 60 秒内最多放行 N 次

设计要点：

- 所有方法 asyncio 安全；**每个 upstream 一个独立实例**，切换 upstream 不受其他
  upstream 的限速影响
- ``acquire()`` 在锁内只做"预约槽位"，实际等待在锁外 sleep，不阻塞其他协程预约
- ``snapshot()`` 输出当前状态，供 ``/status`` 聚合展示
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque

from .config import RateLimitConfig

__all__ = [
    "AsyncRateLimiter",
    "NoLimit",
    "MinIntervalLimiter",
    "RpmLimiter",
    "create_rate_limiter",
]


class AsyncRateLimiter(ABC):
    """限速器基类。"""

    mode: str = "none"

    @abstractmethod
    async def acquire(self) -> None:
        """获取一次调用许可；必要时异步等待。"""

    @abstractmethod
    def snapshot(self) -> dict:
        """当前状态快照（供 /status 输出）。"""


class NoLimit(AsyncRateLimiter):
    """不做任何控制。"""

    mode = "none"

    async def acquire(self) -> None:
        return None

    def snapshot(self) -> dict:
        return {"mode": self.mode}


class MinIntervalLimiter(AsyncRateLimiter):
    """最小间隔限速：相邻两次放行至少 ``min_interval_ms`` 毫秒，FIFO 排队。

    实现：锁内为每个调用者预约一个槽位时刻（``max(now, 上一个槽位) + interval``），
    锁外 sleep 到自己的槽位。保证并发调用者按到达顺序获得间隔均匀的放行时刻。
    """

    mode = "min_interval"

    def __init__(self, min_interval_ms: int) -> None:
        if min_interval_ms <= 0:
            raise ValueError(f"min_interval_ms 必须为正数，实际为 {min_interval_ms}")
        self._interval = min_interval_ms / 1000.0
        self._min_interval_ms = min_interval_ms
        self._next_slot = 0.0  # monotonic；下一个可用槽位
        self._waiting = 0  # 正在等待槽位的调用者数
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
            self._waiting += 1
        try:
            delay = slot - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        finally:
            self._waiting -= 1

    def snapshot(self) -> dict:
        next_free_ms = max(0.0, (self._next_slot - time.monotonic()) * 1000.0)
        return {
            "mode": self.mode,
            "min_interval_ms": self._min_interval_ms,
            "next_free_in_ms": round(next_free_ms, 1),
            "waiting": self._waiting,
        }


class RpmLimiter(AsyncRateLimiter):
    """滑动窗口配额限速：任意 ``window_seconds``（默认 60s）内最多 ``rpm`` 次。

    实现：deque 保存已放行/已预约的槽位时刻。锁内先清理过期槽位；若窗口内已满，
    则预约"最早槽位过期后"的时刻（``slots[F] + window``，F 为窗口外的未来预约数），
    锁外 sleep 到槽位。保证滑动窗口内放行数不超过 rpm，多调用者并发安全。
    """

    mode = "rpm"

    def __init__(self, rpm: int, window_seconds: float = 60.0) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm 必须为正数，实际为 {rpm}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds 必须为正数，实际为 {window_seconds}")
        self._rpm = rpm
        self._window = window_seconds
        self._slots: deque[float] = deque()  # 已预约槽位时刻（monotonic，非降序）
        self._waiting = 0
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        # 过期判据用 <=：槽位恰好在过期时刻被复用，与预约逻辑一致
        while self._slots and self._slots[0] <= cutoff:
            self._slots.popleft()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._evict(now)
            if len(self._slots) < self._rpm:
                slot = now
            else:
                # 窗口已满：第 F 个未来预约对应槽位 slots[F] 的过期时刻；
                # 若多个历史槽位时刻相同（同刻放行），过期后同时腾出多个名额，
                # 复用同一时刻是正确的。
                future = sum(1 for s in self._slots if s > now)
                slot = self._slots[future] + self._window
            self._slots.append(slot)
            self._waiting += 1
        try:
            delay = slot - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        finally:
            self._waiting -= 1

    def snapshot(self) -> dict:
        now = time.monotonic()
        used = sum(1 for s in self._slots if s > now - self._window)
        return {
            "mode": self.mode,
            "rpm": self._rpm,
            "window_seconds": self._window,
            "used_in_window": used,
            "waiting": self._waiting,
        }


def create_rate_limiter(cfg: RateLimitConfig) -> AsyncRateLimiter:
    """根据配置创建限速器（每 upstream 独立实例）。"""
    if cfg.mode == "none":
        return NoLimit()
    if cfg.mode == "min_interval":
        return MinIntervalLimiter(cfg.min_interval_ms)
    if cfg.mode == "rpm":
        return RpmLimiter(cfg.rpm)
    raise ValueError(f"未知限速模式: {cfg.mode!r}")
