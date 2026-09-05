"""启动入口：``python -m llmproxy [选项]``。

监听端口可通过命令行指定，默认 **4400**（配置文件 ``server.port`` 会被覆盖）::

    python -m llmproxy                     # 使用 config.yaml，端口 4400
    python -m llmproxy --port 4500         # 指定端口
    python -m llmproxy --host 0.0.0.0 --port 4400
    python -m llmproxy -c /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmproxy",
        description="类 LiteLLM 代理服务：虚拟模型 + 多 upstream 负载均衡/故障切换",
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--host", default=None,
        help="监听地址（覆盖配置文件 server.host）",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="监听端口（覆盖配置文件 server.port，默认: 4400）",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["debug", "info", "warning", "error"],
        help="日志级别（覆盖配置文件 server.log_level）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 2

    # 命令行参数优先于配置文件
    if args.host is not None:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.log_level is not None:
        cfg.server.log_level = args.log_level

    import uvicorn

    from .api import create_app
    from .logging_setup import setup_logging

    setup_logging(cfg.server)  # 控制台 + 滚动文件 logs/llmproxy.log（保留 3 份）

    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
