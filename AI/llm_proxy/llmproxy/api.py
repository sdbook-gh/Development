"""FastAPI 应用：OpenAI 兼容端点 + LiteLLM 兼容响应头。

端点：

- ``POST /v1/chat/completions`` —— 主转发端点（含流式）
- ``GET  /v1/models`` —— OpenAI 风格模型列表（虚拟模型，含隐式 upstream 直连名）
- ``GET  /model/info`` —— LiteLLM 同构 JSON（pi extension 消费）
- ``GET  /health`` —— 存活探针
- ``GET  /status`` —— upstream / 路由运行时聚合

响应头（成功时；``x-litellm-*`` 为兼容别名）：

- ``x-llmproxy-upstream``    实际服务的 upstream 名
- ``x-llmproxy-model``       实际服务的真实模型 id
- ``x-llmproxy-model-id``    部署 id（``{虚拟模型}@{upstream}``，与 /model/info 的
  ``model_info.id`` 一致），别名 ``x-litellm-model-id``
- ``x-litellm-model-group``  虚拟模型名

pi extension 解析链：``x-litellm-model-id`` → ``/model/info`` 的 ``model_info.id``
→ ``litellm_params.model``（真实模型 id）。

流式重试边界：dispatch 的 ``send`` 内**提前拉取上游首块**——首块到达前的任何失败
（含 200 后、首块前的连接中断）都以网络错误语义交还 dispatch 正常切换；
``StreamingResponse`` 开始发送后不可再切换，流中断如实上抛。
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AppConfig, VirtualModelConfig
from .logging_setup import attempts_chain
from .router import DispatchResult, Router
from .status import aggregate_status
from .upstream import UpstreamClients

__all__ = ["create_app"]

logger = logging.getLogger("llmproxy.api")


async def _prepend_first(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """把提前拉取的首块与剩余流拼接（保证首字节前可 failover）。"""
    if first:
        yield first
    async for chunk in rest:
        yield chunk


def _error_json(status: int, message: str, err_type: str,
                code: str | None = None) -> JSONResponse:
    """OpenAI 风格错误响应。"""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type,
                           "param": None, "code": code}},
    )


def _served_headers(vm_name: str, result: DispatchResult) -> dict[str, str]:
    upstream = result.upstream or ""
    model_id = f"{vm_name}@{upstream}"
    return {
        "x-llmproxy-upstream": upstream,
        "x-llmproxy-model": result.model or "",
        "x-llmproxy-model-id": model_id,
        "x-litellm-model-id": model_id,
        "x-litellm-model-group": vm_name,
    }


def create_app(cfg: AppConfig) -> FastAPI:
    """组装 FastAPI 应用（Router + UpstreamClients + 端点）。"""
    # 直连便利：未定义同名虚拟模型的 upstream，自动建隐式单 upstream 路由，
    # 使 `model: "<upstream 名>"` 请求直达该 upstream（仍享受熔断/限速）。
    for up in cfg.upstreams:
        if up.name not in cfg.virtual_models:
            cfg.virtual_models[up.name] = VirtualModelConfig(
                name=up.name, strategy="failover", upstreams=[up.name]
            )

    clients = UpstreamClients(cfg)
    router = Router(cfg, clients)
    started_at = int(time.time())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("llmproxy 启动: %d upstream / %d 虚拟模型（含隐式直连路由）",
                    len(cfg.upstreams), len(cfg.virtual_models))
        yield
        await router.aclose()   # 先取消在途主动探测（其引用 upstream 连接池）
        await clients.aclose()
        logger.info("llmproxy 关闭: upstream 连接已释放")

    app = FastAPI(title="llmproxy", version="0.1.0", lifespan=lifespan)

    # ---------------------------------------------- chat completions ----

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        rid = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        client_id = request.headers.get("x-client-id", "-")

        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return _error_json(400, "请求体不是合法 JSON", "invalid_request_error")
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str) or not model:
            return _error_json(400, "缺少 'model' 字段", "invalid_request_error",
                               code="invalid_model")
        if model not in cfg.virtual_models:
            return _error_json(
                404, f"模型 '{model}' 不存在（可用: {sorted(cfg.virtual_models)}）",
                "invalid_request_error", code="model_not_found",
            )

        stream = bool(body.get("stream"))

        async def send(rt):
            """dispatch 每次尝试的回调：替换真实模型名 → 转发 upstream。"""
            st, payload, err, ra = await clients.get(rt.cfg.name).chat(
                {**body, "model": rt.cfg.model}
            )
            if err is None and stream and hasattr(payload, "__anext__"):
                # 提前拉首块：此处失败 → 网络错误语义，dispatch 可继续切换
                try:
                    first = await payload.__anext__()
                except StopAsyncIteration:
                    first = b""
                except Exception as exc:
                    return (None, None, f"stream 首块读取失败: {exc!r}", None)
                return (st, _prepend_first(first, payload), None, None)
            return (st, payload, err, ra)

        result = await router.dispatch(model, send, rid=rid)
        duration_ms = round((time.monotonic() - t0) * 1000, 1)

        if result.ok:
            logger.info("rid=%s client=%s vm=%s upstream=%s model=%s stream=%s %.0fms attempts=%d [%s]",
                        rid, client_id, model, result.upstream, result.model,
                        stream, duration_ms, len(result.attempts),
                        attempts_chain(result.attempts))
            headers = _served_headers(model, result)
            if stream:
                # 响应头已随 StreamingResponse 定格；此后流中断如实透传给客户端
                return StreamingResponse(result.response,
                                         media_type="text/event-stream",
                                         headers=headers)
            # OpenAI 语义：响应体 model 统一为请求的虚拟模型名（真实模型见响应头）
            if isinstance(result.response, dict) and "model" in result.response:
                result.response["model"] = model
            return JSONResponse(content=result.response, headers=headers)

        # 失败：透传最后一次错误状态码；网络层错误（无状态码）→ 502
        status = result.status if result.status is not None else 502
        # 429 等错误若上游给了 Retry-After，透传给客户端做退避（RFC 6585：非负整秒）
        fail_headers: dict[str, str] = {}
        if result.retry_after is not None:
            fail_headers["Retry-After"] = str(max(1, int(result.retry_after)))
        logger.warning("rid=%s client=%s vm=%s 派发失败 status=%s error=%s %.0fms attempts=%d [%s]",
                       rid, client_id, model, result.status, result.error,
                       duration_ms, len(result.attempts), attempts_chain(result.attempts))
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": result.error or "派发失败",
                "type": "llmproxy_upstream_error",
                "param": None,
                "code": result.status,
                "llmproxy": {
                    "upstream": result.upstream,
                    "attempts": [a.as_dict() for a in result.attempts],
                },
            }},
            headers=fail_headers or None,
        )

    # -------------------------------------------------- 信息类端点 ----

    @app.get("/v1/models")
    async def list_models():
        data = [{"id": name, "object": "model",
                 "created": started_at, "owned_by": "llmproxy"}
                for name in cfg.virtual_models]
        return {"object": "list", "data": data}

    @app.get("/model/info")
    async def model_info():
        """LiteLLM 同构 JSON：每 (虚拟模型, upstream) 一条部署记录。

        pi extension 用响应头 x-litellm-model-id 匹配 model_info.id，
        取 litellm_params.model 展示实际服务的真实模型。
        """
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for vm_name, vm in cfg.virtual_models.items():
            for up_name in vm.upstreams:
                if (vm_name, up_name) in seen:
                    continue
                seen.add((vm_name, up_name))
                up = cfg.get_upstream(up_name)
                out.append({
                    "model_name": vm_name,
                    "litellm_params": {
                        "model": up.model,
                        "api_base": up.base_url,
                        "custom_llm_provider": "openai",
                    },
                    "model_info": {
                        "id": f"{vm_name}@{up_name}",
                        "upstream": up_name,
                        "mode": "chat",
                    },
                })
        return out

    @app.get("/health")
    async def health():
        return {"status": "ok", "time": int(time.time())}

    @app.get("/status")
    async def status():
        return aggregate_status(cfg, router, started_at)

    return app
