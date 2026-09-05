"""被动健康检测与熔断（每 upstream 一个独立实例）。

状态机::

    CLOSED ──连续失败达 failure_threshold──▶ OPEN ──冷却结束──▶ HALF_OPEN ──探测成功──▶ CLOSED
      ▲                                    │                   │
      └─────────────成功清零────────────────┴────探测失败──────▶ OPEN（重新冷却）

另外支持**主动探测**（report_probe_success）：路由层在非 429 失败触发熔断后，
会用最小请求实测 upstream；若 upstream 已恢复，直接闭合熔断，不等冷却结束。

- **CLOSED**：正常放行；每次失败累加连续失败计数，成功清零
- **OPEN**：冷却期内拒绝准入（路由层跳过该 upstream）；429 等
  ``immediate`` 失败或带 ``Retry-After`` 时用其值作冷却时长（封顶 MAX_COOLDOWN）
- **HALF_OPEN**：冷却结束自动进入；只放行一个探测请求，成功即闭合恢复，
  失败则重新进入 OPEN（时长不叠加，避免无限惩罚）

所有方法 asyncio 安全；每 upstream 独立实例，互不影响。
"""

from __future__ import annotations

import asyncio
import logging
import time

__all__ = ["CircuitState", "HealthTracker"]

logger = logging.getLogger("llmproxy.health")

# 熔断状态
CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

# Retry-After 冷却时长上限（秒），防止异常大的值
MAX_COOLDOWN = 3600.0


class HealthTracker:
    """单 upstream 的被动健康跟踪器（熔断器）。"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold 必须 >= 1，实际为 {failure_threshold}")
        if cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds 必须 >= 0，实际为 {cooldown_seconds}")
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0  # 进入 OPEN 时的 monotonic 时刻
        self._cooldown = cooldown_seconds  # 本次 OPEN 的冷却时长
        self._half_open_granted = False  # 半开探测名额是否已发出
        self._lock = asyncio.Lock()

        # 统计
        self.total_failures = 0
        self.total_recoveries = 0
        self.last_failure_at: float | None = None  # monotonic
        self.last_error: str | None = None
        self.last_error_status: int | None = None

    # ------------------------------------------------------------- 查询 ----

    @property
    def state(self) -> str:
        """当前状态；OPEN 冷却结束自动转 HALF_OPEN（无锁读，仅供快照参考）。"""
        if (
            self._state == OPEN
            and time.monotonic() - self._opened_at >= self._cooldown
        ):
            return HALF_OPEN
        return self._state

    def available(self) -> bool:
        """是否可接收请求（供路由层快速预筛；真实准入以 claim 为准）。"""
        return self.state != OPEN

    # ------------------------------------------------------------- 准入 ----

    async def claim(self) -> bool:
        """请求放行。半开态只放行一个探测请求，其余拒绝。"""
        async with self._lock:
            now = time.monotonic()
            if self._state == OPEN:
                if now - self._opened_at >= self._cooldown:
                    self._state = HALF_OPEN
                    self._half_open_granted = False
                    logger.info("[%s] 冷却结束，OPEN -> HALF_OPEN（放行探测）", self.name)
                else:
                    return False
            if self._state == HALF_OPEN and self._half_open_granted:
                return False
            if self._state == HALF_OPEN:
                self._half_open_granted = True
                logger.info("[%s] 半开探测放行", self.name)
            return True

    async def abort_claim(self) -> None:
        """回滚一次已成功的 claim（请求未实际发出即被取消时调用）。

        仅在 HALF_OPEN 且名额已被本次 claim 占用时释放，让下一个请求可重新
        探测；CLOSED 态的 claim 无名额语义，本方法无副作用。**不计失败**、
        不改变状态机（取消 ≠ upstream 故障）。
        """
        async with self._lock:
            if self._state == HALF_OPEN and self._half_open_granted:
                self._half_open_granted = False
                logger.info("[%s] 请求取消，释放半开探测名额", self.name)

    # --------------------------------------------------------- 结果回报 ----

    async def report_success(self) -> None:
        """请求成功：计数清零；半开探测成功则闭合恢复。"""
        async with self._lock:
            self._consecutive_failures = 0
            self._half_open_granted = False
            if self._state == HALF_OPEN:
                self._state = CLOSED
                self._cooldown = self.cooldown_seconds
                self.total_recoveries += 1
                logger.info("[%s] 探测成功，HALF_OPEN -> CLOSED（累计恢复 %d 次）",
                            self.name, self.total_recoveries)

    async def report_probe_success(self) -> None:
        """主动探测成功：立即闭合熔断（OPEN/HALF_OPEN -> CLOSED），失败计数清零。

        供路由层在非 429 失败触发熔断后的最小请求探测使用；若探测时熔断
        已被被动流程闭合（HALF_OPEN 探测成功等），本方法幂等无副作用。
        """
        async with self._lock:
            self._consecutive_failures = 0
            self._half_open_granted = False
            if self._state != CLOSED:
                prev = self._state
                self._state = CLOSED
                self._cooldown = self.cooldown_seconds
                self.total_recoveries += 1
                logger.info("[%s] 主动探测成功，%s -> CLOSED（累计恢复 %d 次）",
                            self.name, prev, self.total_recoveries)

    async def report_failure(
        self,
        status: int | None = None,
        error: str | None = None,
        retry_after: float | None = None,
        immediate: bool = False,
    ) -> bool:
        """请求失败：累加计数；达到阈值（或 immediate / 半开探测失败）则进入 OPEN。

        Args:
            status: HTTP 状态码（网络错误等无状态码时为 None）
            error: 错误描述（记入 last_error）
            retry_after: 429 等响应的 ``Retry-After`` 秒数（由调用方解析）
            immediate: 立即熔断（如 429 限流，不等连续失败阈值）

        Returns:
            True 表示本次失败触发了"进入熔断"事件（供日志记录）。
        """
        async with self._lock:
            self.total_failures += 1
            self._consecutive_failures += 1
            self.last_failure_at = time.monotonic()
            self.last_error = error
            self.last_error_status = status

            was_open = self._state == OPEN
            opened = False
            trigger_note = f"连续失败 {self._consecutive_failures} 次"

            if self._state == HALF_OPEN:
                # 探测失败：重新冷却（时长不叠加，仍用标准冷却）
                self._state = OPEN
                self._opened_at = time.monotonic()
                self._cooldown = self.cooldown_seconds
                self._consecutive_failures = 0
                opened = True
            elif immediate or self._consecutive_failures >= self.failure_threshold:
                # 429 立即熔断，或连续失败达阈值
                self._state = OPEN
                self._opened_at = time.monotonic()
                self._cooldown = self._resolve_cooldown(retry_after)
                self._consecutive_failures = 0
                opened = True

            # OPEN 态中的后续失败只刷新统计，不重复触发事件、不延长冷却
            if opened and not was_open:
                reason = "immediate" if immediate else trigger_note
                logger.warning("[%s] 熔断 OPEN 冷却 %.0fs（原因: %s, status=%s, error=%s）",
                               self.name, self._cooldown, reason, status, error)
            return opened and not was_open

    @property
    def current_cooldown(self) -> float:
        """当前生效的冷却时长（本次 OPEN 可能被 429 Retry-After 覆盖默认值）。"""
        return self._cooldown

    def _resolve_cooldown(self, retry_after: float | None) -> float:
        """冷却时长：优先 Retry-After（封顶 MAX_COOLDOWN），否则用配置的冷却时长。"""
        if retry_after is not None:
            return max(0.0, min(float(retry_after), MAX_COOLDOWN))
        return self.cooldown_seconds

    # ------------------------------------------------------------- 快照 ----

    def snapshot(self) -> dict:
        """当前状态快照（供 /status 输出）。"""
        now = time.monotonic()
        state = self.state
        snap: dict = {
            "state": state,
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self.total_failures,
            "total_recoveries": self.total_recoveries,
            "failure_threshold": self.failure_threshold,
        }
        if state == OPEN:
            snap["cooldown_remaining_ms"] = round(
                max(0.0, (self._opened_at + self._cooldown - now) * 1000.0), 1
            )
        if self.last_failure_at is not None:
            snap["last_failure_ago_ms"] = round((now - self.last_failure_at) * 1000.0, 1)
        if self.last_error is not None:
            snap["last_error"] = self.last_error
            snap["last_error_status"] = self.last_error_status
        return snap
