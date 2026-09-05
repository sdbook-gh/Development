"""负载均衡与故障切换编排。

串联 ratelimit（频率控制）与 health（熔断），对每个虚拟模型提供：

- ``candidates()``：按策略生成候选 upstream 顺序
  （round_robin / least_load / failover / weighted_random）
- ``dispatch()``：完整尝试循环 —— 逐个候选准入（熔断 claim + 限速等待 +
  in-flight 计数），调用 ``send`` 回调，按结果决定重试/切换/返回

可切换判定（见 README）：

- HTTP 状态码在 ``retry.retryable_status``（默认含 400/408/429/500/502/503/504）
- 网络层错误（超时 / 连接失败等，无状态码）
- 429 额外触发立即熔断并尊重 ``Retry-After``

不可切换（如 401/403/404）立即返回失败，不浪费其余尝试次数。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig, RetryConfig, UpstreamConfig
from .health import HealthTracker
from .ratelimit import AsyncRateLimiter, create_rate_limiter
from .upstream import UpstreamClients

__all__ = [
    "AttemptLog",
    "DispatchResult",
    "NoUpstreamAvailableError",
    "Router",
]

logger = logging.getLogger("llmproxy.router")


class NoUpstreamAvailableError(Exception):
    """虚拟模型下没有可用的 upstream（全部处于熔断冷却）。"""


# ---------------------------------------------------------------- runtime ----

class UpstreamRuntime:
    """单个 upstream 的运行时状态：配置 + 健康 + 限速 + in-flight 并发。"""

    def __init__(self, cfg: UpstreamConfig) -> None:
        self.cfg = cfg
        self.health = HealthTracker(
            name=cfg.name,
            failure_threshold=3,  # 由虚拟模型 retry 配置在 Router 中覆盖
            cooldown_seconds=30.0,
        )
        self.limiter: AsyncRateLimiter = create_rate_limiter(cfg.rate_limit)
        self.inflight = 0  # asyncio 单线程，无 await 的自增足够安全

    async def enter(self, rid: str = "-") -> bool:
        """尝试占用该 upstream：熔断准入 → 限速排队 → in-flight +1。

        Returns:
            False 表示未获准入（熔断冷却/半开名额被占），调用方应换下一个候选。
        """
        if not await self.health.claim():
            return False
        t0 = time.monotonic()
        try:
            await self.limiter.acquire()
        except asyncio.CancelledError:
            # 客户端断连取消限速等待：回滚半开名额，不计为 upstream 故障，原样上抛
            await self.health.abort_claim()
            raise
        except Exception as exc:
            await self.health.report_failure(status=None, error=repr(exc))
            raise
        waited_ms = (time.monotonic() - t0) * 1000
        if waited_ms >= 1.0:  # 首个调用者无需等待，不刷日志
            logger.info("rid=%s upstream=%s 限速等待 %.0fms（mode=%s waiting=%d）",
                        rid, self.cfg.name, waited_ms,
                        self.limiter.mode, self.limiter.snapshot().get("waiting", 0))
        self.inflight += 1
        return True

    def leave(self) -> None:
        """释放一次占用（in-flight -1）。"""
        self.inflight = max(0, self.inflight - 1)

    def snapshot(self) -> dict:
        return {
            "name": self.cfg.name,
            "model": self.cfg.model,
            "proxy": self.cfg.proxy,
            "headers": sorted(self.cfg.headers),   # 仅键名，不泄露值
            "inflight": self.inflight,
            "rate_limit": self.limiter.snapshot(),
            "health": self.health.snapshot(),
        }


# ----------------------------------------------------------------- result ----

@dataclass
class AttemptLog:
    """单次尝试的记录（含被跳过的候选）。"""

    upstream: str
    outcome: str            # ok | skipped | http_error | network_error | exception
    status: int | None = None
    error: str | None = None
    duration_ms: float | None = None
    note: str | None = None  # 附加说明（如 failover 原因、熔断事件）

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DispatchResult:
    """dispatch() 的最终结果。"""

    ok: bool
    upstream: str | None = None       # 实际服务的 upstream 名
    model: str | None = None          # 实际服务的 upstream 模型 id
    status: int | None = None         # 最终 HTTP 状态码（网络错误为 None）
    response: Any = None              # upstream 客户端产物（响应体/流迭代器）
    error: str | None = None
    attempts: list[AttemptLog] = field(default_factory=list)
    total_ms: float = 0.0
    retry_after: float | None = None  # 最后一次失败响应的 Retry-After（秒，供 api 层透传）


@dataclass
class _Route:
    """虚拟模型的路由表。"""

    name: str
    strategy: str
    runtimes: list[UpstreamRuntime]   # 配置顺序
    retry: RetryConfig
    _rr: Iterator[int] = field(default_factory=itertools.count)

    def next_start(self) -> int:
        return next(self._rr)


# ----------------------------------------------------------------- router ----

class Router:
    """管理全部 upstream 运行时与虚拟模型路由。"""

    def __init__(self, app_config: AppConfig,
                 clients: UpstreamClients | None = None) -> None:
        self._runtimes: dict[str, UpstreamRuntime] = {}
        for up in app_config.upstreams:
            self._runtimes[up.name] = UpstreamRuntime(up)
        self._routes: dict[str, _Route] = {}
        self._clients = clients
        # 在途主动探测任务（按 upstream 名去重，跨路由共享，防止探测风暴）
        self._probe_tasks: dict[str, asyncio.Task] = {}
        self._closed = False
        for name, vm in app_config.virtual_models.items():
            runtimes = [self._runtimes[ref] for ref in vm.upstreams]
            retry = vm.retry
            for rt in runtimes:
                # 健康参数跟随虚拟模型的 retry 配置（同 upstream 多路由时取首见）
                rt.health.failure_threshold = retry.failure_threshold
                rt.health.cooldown_seconds = retry.cooldown_seconds
            self._routes[name] = _Route(
                name=name, strategy=vm.strategy, runtimes=runtimes, retry=retry
            )

    # --------------------------------------------------------- 候选生成 ----

    def candidates(self, vm_name: str) -> list[UpstreamRuntime]:
        """按策略排出候选顺序，并过滤掉熔断冷却中的 upstream。

        若全部处于冷却，抛 :class:`NoUpstreamAvailableError`。
        """
        route = self._routes.get(vm_name)
        if route is None:
            raise KeyError(f"虚拟模型 '{vm_name}' 未定义")
        pool = [rt for rt in route.runtimes if rt.health.available()]
        if not pool:
            states = ", ".join(
                f"{rt.cfg.name}={rt.health.state}" for rt in route.runtimes
            )
            raise NoUpstreamAvailableError(
                f"虚拟模型 '{vm_name}' 无可用 upstream（{states}）"
            )
        return self._apply_strategy(route, pool)

    def _apply_strategy(self, route: _Route, pool: list[UpstreamRuntime]) -> list[UpstreamRuntime]:
        s = route.strategy
        if s == "failover":
            return pool  # 保持配置顺序
        if s == "round_robin":
            n = len(pool)
            start = route.next_start() % n
            return pool[start:] + pool[:start]
        if s == "least_load":
            # 稳定排序：负载相同时保持配置顺序
            return sorted(pool, key=lambda rt: rt.inflight)
        if s == "weighted_random":
            weights = [rt.cfg.weight for rt in pool]
            start = random.choices(range(len(pool)), weights=weights, k=1)[0]
            return pool[start:] + pool[:start]
        raise ValueError(f"未知策略: {s!r}")

    # ------------------------------------------------------------- 派发 ----

    async def dispatch(
        self,
        vm_name: str,
        send: Callable[[UpstreamRuntime], Awaitable[tuple[int | None, Any, str | None, float | None]]],
        rid: str = "-",
    ) -> DispatchResult:
        """对虚拟模型执行一次带切换的完整派发。

        Args:
            vm_name: 虚拟模型名
            send: 异步回调 ``(runtime) -> (status, response, error, retry_after)``。
                正常响应返回 ``(status, response_obj, None, None)``；失败返回
                ``(status, None, error_desc, retry_after_or_None)``；网络层错误
                status 传 None。
            rid: 请求关联 id，用于日志追踪。

        Returns:
            :class:`DispatchResult` —— ok=True 时携带 response；
            否则携带最后一次失败的 status/error（供 api 层转换为 HTTP 错误）。
        """
        route = self._routes[vm_name]
        started = time.monotonic()
        attempts_left = route.retry.max_attempts
        trail: list[AttemptLog] = []
        last_failure: AttemptLog | None = None
        last_retry_after: float | None = None

        try:
            pool = self.candidates(vm_name)
        except NoUpstreamAvailableError as e:
            logger.warning("rid=%s vm=%s dispatch 失败: %s", rid, vm_name, e)
            return DispatchResult(
                ok=False, error=str(e), status=503,
                attempts=trail, total_ms=(time.monotonic() - started) * 1000,
            )

        for rt in pool:
            if attempts_left <= 0:
                break
            if not await rt.enter(rid):
                trail.append(AttemptLog(upstream=rt.cfg.name, outcome="skipped",
                                        note="熔断准入未通过"))
                continue
            attempts_left -= 1
            t0 = time.monotonic()
            try:
                status, response, error, retry_after = await send(rt)
            except asyncio.CancelledError:
                # 客户端断连/服务关闭：释放占用与半开名额后如实上抛；
                # 取消不等于 upstream 故障，不计失败、不消费剩余尝试次数
                rt.leave()
                await rt.health.abort_claim()
                raise
            except Exception as exc:  # 防御：send 实现自身抛出的意外异常
                dur = (time.monotonic() - t0) * 1000
                rt.leave()
                await rt.health.report_failure(status=None, error=repr(exc))
                last_failure = AttemptLog(upstream=rt.cfg.name, outcome="exception",
                                          error=repr(exc), duration_ms=round(dur, 1))
                trail.append(last_failure)
                logger.error("rid=%s vm=%s upstream=%s send 异常: %r",
                             rid, vm_name, rt.cfg.name, exc)
                continue  # 视为网络层错误，可切换

            dur = (time.monotonic() - t0) * 1000
            rt.leave()

            if error is None:
                await rt.health.report_success()
                trail.append(AttemptLog(upstream=rt.cfg.name, outcome="ok",
                                        status=status, duration_ms=round(dur, 1)))
                logger.info(
                    "rid=%s vm=%s upstream=%s model=%s 成功 status=%s attempts=%d/%d %.0fms",
                    rid, vm_name, rt.cfg.name, rt.cfg.model, status,
                    len(trail), route.retry.max_attempts, dur,
                )
                return DispatchResult(
                    ok=True, upstream=rt.cfg.name, model=rt.cfg.model,
                    status=status, response=response, attempts=trail,
                    total_ms=round((time.monotonic() - started) * 1000, 1),
                )

            # ---- 失败 ----
            immediate = status == 429
            last_retry_after = retry_after
            opened = await rt.health.report_failure(
                status=status, error=error, retry_after=retry_after, immediate=immediate,
            )
            note = f"failover: {status or 'network'}" + ("; 熔断进入 OPEN" if opened else "")
            if retry_after is not None:
                note += f" (Retry-After {retry_after}s)"
            last_failure = AttemptLog(upstream=rt.cfg.name,
                                      outcome="http_error" if status else "network_error",
                                      status=status, error=error,
                                      duration_ms=round(dur, 1), note=note)
            trail.append(last_failure)
            logger.warning(
                "rid=%s vm=%s upstream=%s 失败 status=%s error=%s %.0fms%s",
                rid, vm_name, rt.cfg.name, status, error, dur,
                f" -> 触发熔断(冷却{rt.health.current_cooldown:.0f}s)" if opened else " -> failover",
            )

            # ---- 主动探测（非 429 失败且本次失败使熔断进入 OPEN 时）----
            # 429 限流探测无意义且会加剧限流，仍走被动冷却；其余错误用最小请求
            # 实测 upstream 是否恢复，成功则立即闭合熔断，不等冷却结束。
            if self._clients is not None and opened and status != 429 and route.retry.active_probe:
                self._start_probe(rt, vm_name, rid, route.retry.probe_max_tokens)

            if status is not None and status not in route.retry.retryable_status:
                logger.info("rid=%s vm=%s status=%s 不可切换，停止重试", rid, vm_name, status)
                break  # 非可切换错误：不再浪费剩余尝试

        result = DispatchResult(
            ok=False, attempts=trail,
            total_ms=round((time.monotonic() - started) * 1000, 1),
        )
        if last_failure is not None:
            result.upstream = last_failure.upstream
            result.status = last_failure.status
            result.error = last_failure.error
            result.retry_after = last_retry_after
        return result

    # ------------------------------------------------------- 主动探测 ----

    def _start_probe(self, rt: UpstreamRuntime, vm_name: str, rid: str,
                     max_tokens: int) -> None:
        """为刚进入 OPEN 的 upstream 发起一次最小请求探测。

        - 按 upstream 名去重（跨路由共享），同一时间至多一个在途探测
        - 探测请求不走 dispatch/round-robin，不占 attempt，不影响限速与计数
        - 探测成功 → 熔断立即闭合；失败 → 维持原冷却时长（不延长、不重复惩罚）
        """
        if self._closed or rt.cfg.name in self._probe_tasks:
            return
        name = rt.cfg.name
        body = {
            "model": rt.cfg.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": max_tokens,   # 最小化 token 消耗（默认 3，满足 max_tokens>2 校验）
            "stream": False,
        }
        clients = self._clients
        health = rt.health

        async def _run() -> None:
            try:
                await asyncio.sleep(1.0)  # 短退避：避开紧随失败的瞬时窗口
                st, _payload, err, _ra = await clients.get(name).chat(body)
                if err is None and st == 200:
                    await health.report_probe_success()
                    logger.info("[probe] rid=%s vm=%s upstream=%s 主动探测成功 status=200（熔断闭合）",
                                rid, vm_name, name)
                else:
                    # 探测失败不计入熔断统计：维持原冷却，不延长、不触发新事件
                    logger.info("[probe] rid=%s vm=%s upstream=%s 主动探测失败 status=%s error=%s（维持冷却）",
                                rid, vm_name, name, st, (err or "")[:200])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 探测自身异常不外溢、不影响主流程
                logger.warning("[probe] rid=%s vm=%s upstream=%s 主动探测异常: %r",
                               rid, vm_name, name, exc)
            finally:
                self._probe_tasks.pop(name, None)

        self._probe_tasks[name] = asyncio.create_task(
            _run(), name=f"llmproxy-probe-{name}")

    # ------------------------------------------------------------- 快照 ----

    def snapshot(self) -> dict:
        """全部 upstream 运行时快照（供 /status）。"""
        return {name: rt.snapshot() for name, rt in self._runtimes.items()}

    async def aclose(self) -> None:
        """取消全部在途主动探测任务（服务关闭时调用，先于 upstream 客户端释放）。"""
        self._closed = True
        tasks = [t for t in self._probe_tasks.values() if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._probe_tasks.clear()

    def route_snapshot(self) -> dict:
        """虚拟模型路由快照（供 /status）。"""
        return {
            name: {"strategy": r.strategy,
                   "upstreams": [rt.cfg.name for rt in r.runtimes],
                   "max_attempts": r.retry.max_attempts,
                   "retryable_status": list(r.retry.retryable_status)}
            for name, r in self._routes.items()
        }

    def get_runtime(self, name: str) -> UpstreamRuntime:
        return self._runtimes[name]
