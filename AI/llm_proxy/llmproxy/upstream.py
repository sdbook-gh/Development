"""upstream 客户端：httpx.AsyncClient 封装，每 upstream 一个独立实例。

职责：

- 按配置走**直连**或**命名代理**（proxies 表 → httpx proxy URL）
- 连接池（每 upstream 独立，互不干扰）
- OpenAI ``/chat/completions`` 请求转发，支持流式（``stream: true``）透传
- 失败时截取响应体片段供日志 / api 层使用
- **重试边界**：非流式在响应返回前失败可切换 upstream；流式一旦开始向客户端
  发送（首块产生）即不可切换，错误如实上抛

网络层错误统一包装为 :class:`UpstreamNetworkError`（status=None 语义）。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from .config import AppConfig, UpstreamConfig

__all__ = [
    "UpstreamNetworkError",
    "UpstreamClient",
    "UpstreamClients",
]

logger = logging.getLogger("llmproxy.upstream")

DEFAULT_TIMEOUT = 300.0          # 单请求总超时（流式长响应需要较长时间）
CONNECT_TIMEOUT = 10.0           # 连接建立超时（坏 upstream / 不通代理快速失败并切换）
ERROR_BODY_SNIPPET = 500         # 失败响应体截取长度（字符）


class UpstreamNetworkError(Exception):
    """网络层错误（超时 / 连接失败 / 代理失败等），无 HTTP 状态码。"""

    def __init__(self, upstream: str, detail: str) -> None:
        super().__init__(f"[{upstream}] {detail}")
        self.upstream = upstream
        self.detail = detail


def _snippet(body: bytes | str, limit: int = ERROR_BODY_SNIPPET) -> str:
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    return text if len(text) <= limit else text[:limit] + f"...（截断，共 {len(text)} 字符）"


class UpstreamClient:
    """单个 upstream 的异步客户端（独立 httpx.AsyncClient = 独立连接池/代理）。"""

    def __init__(self, cfg: UpstreamConfig, proxy_url: str | None,
                 timeout: float | None = None) -> None:
        self.cfg = cfg
        self._proxy_url = proxy_url
        # 默认头在前，扩展头（cfg.headers）在后：同名可覆盖默认头（如 User-Agent）
        headers: dict[str, str] = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(cfg.headers)
        # Gemini 的 OpenAI 兼容层用 x-goog-api-key / key= 均可，Bearer 已兼容。
        # trust_env=False：代理只来自配置文件的 proxies 表，不受系统环境变量干扰。
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            proxy=proxy_url,          # None = 直连
            timeout=httpx.Timeout(
                timeout if timeout is not None else DEFAULT_TIMEOUT,
                connect=CONNECT_TIMEOUT,
            ),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            trust_env=False,
        )

    @property
    def proxy_url(self) -> str | None:
        return self._proxy_url

    async def chat(
        self,
        body: dict[str, Any],
        on_first_byte: Callable[[], None] | None = None,
    ) -> tuple[int, Any, str | None, float | None]:
        """转发一次 chat/completions。

        Args:
            body: OpenAI 请求体（模型名已由 api 层替换为 upstream 真实模型）
            on_first_byte: 流式模式下首块数据到达时回调（api 层用于标记
                "已对客户端开始发送，之后不可再切换 upstream"）

        Returns:
            ``(status, payload, error, retry_after)``：

            - 成功：``(status, dict 或 async 迭代器, None, None)``；
              流式时 payload 为 ``async iterator[bytes]``（SSE 原始字节流）
            - HTTP 失败：``(status, None, error_desc, retry_after_or_None)``
            - 网络错误：``(None, None, error_desc, None)``
        """
        stream = bool(body.get("stream"))
        t0 = time.monotonic()
        try:
            if stream:
                return await self._chat_stream(body, on_first_byte)
            return await self._chat_json(body)
        except UpstreamNetworkError as e:
            logger.warning("[%s] 网络错误 %.0fms: %s",
                           self.cfg.name, (time.monotonic() - t0) * 1000, e.detail)
            return (None, None, str(e), None)

    async def _chat_json(self, body: dict[str, Any]) -> tuple[int, Any, str | None, float | None]:
        """非流式请求。"""
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise UpstreamNetworkError(self.cfg.name, f"timeout: {e!r}") from e
        except httpx.TransportError as e:
            raise UpstreamNetworkError(self.cfg.name, f"transport: {e!r}") from e

        if resp.status_code >= 400:
            return (resp.status_code, None, _snippet(resp.content), self._retry_after(resp))
        data = self._parse_json(resp.content)
        if data is None:
            return (resp.status_code, None,
                    f"响应非 JSON: {_snippet(resp.content)}", None)
        return (resp.status_code, data, None, None)

    async def _chat_stream(
        self,
        body: dict[str, Any],
        on_first_byte: Callable[[], None] | None,
    ) -> tuple[int, Any, str | None, float | None]:
        """流式请求。返回 (status, async_iterator, error, retry_after)。

        首块到达即调用 ``on_first_byte()``（此后上游错误无法再切换，
        由迭代器内如实上抛，api 层透传给客户端）。
        """
        req = self._client.build_request("POST", "/chat/completions", json=body)
        try:
            resp = await self._client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise UpstreamNetworkError(self.cfg.name, f"timeout: {e!r}") from e
        except httpx.TransportError as e:
            raise UpstreamNetworkError(self.cfg.name, f"transport: {e!r}") from e

        if resp.status_code >= 400:
            # 流式失败响应也要读完整体（错误详情），再关闭
            try:
                await resp.aread()
                err = _snippet(resp.content)
                retry_after = self._retry_after(resp)
            finally:
                await resp.aclose()
            return (resp.status_code, None, err, retry_after)

        notified = False

        async def iterator():
            nonlocal notified
            try:
                async for chunk in resp.aiter_bytes():
                    if not notified:
                        notified = True
                        if on_first_byte is not None:
                            on_first_byte()
                    yield chunk
                if not notified:
                    # 空流也算"开始"（api 层至少发一个空 SSE 事件收尾）
                    if on_first_byte is not None:
                        on_first_byte()
            finally:
                await resp.aclose()

        return (resp.status_code, iterator(), None, None)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        """解析 Retry-After 头（秒）。仅 429 场景有意义，其他场景返回值忽略。"""
        v = resp.headers.get("retry-after")
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None  # HTTP-date 形式不支持，忽略

    @staticmethod
    def _parse_json(content: bytes) -> dict | None:
        try:
            data = json.loads(content)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()


class UpstreamClients:
    """全部 upstream 客户端的管理器（启动时创建，关闭时统一释放）。"""

    def __init__(self, app_config: AppConfig) -> None:
        self._clients: dict[str, UpstreamClient] = {}
        for up in app_config.upstreams:
            proxy_url = app_config.get_proxy_url(up.proxy)
            self._clients[up.name] = UpstreamClient(up, proxy_url)
            logger.info("upstream '%s' -> %s (%s, 扩展头: %s)", up.name, up.base_url,
                        f"via {proxy_url}" if proxy_url else "直连",
                        sorted(up.headers) if up.headers else "无")

    def get(self, name: str) -> UpstreamClient:
        return self._clients[name]

    async def aclose(self) -> None:
        for c in self._clients.values():
            await c.aclose()
