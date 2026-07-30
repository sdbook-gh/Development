#!/usr/bin/env python3
"""
读取 geojson_tile.json 配置，扫描各省 GeoJSON 子目录，
使用 libtippecanoe.so / libtile-join-ext.so 生成瓦片并合并为各省 mbtiles。

通过 os.fork() 隔离 .so 调用：若库内部调用 exit()，仅子进程退出，
父进程通过 waitpid 获取退出码，不影响进程池/线程池。

用法:
    python geojson_to_mbtiles.py --tippecanoe-lib /path/libtippecanoe.so --tile-join-ext-lib /path/libtile-join-ext.so \
        -i ./geojson_output -o ./mbtiles_output -w 8
    python geojson_to_mbtiles.py -p jiangxi -w 1
    python geojson_to_mbtiles.py --dry-run
"""

import argparse
import ctypes
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# 主进程中跟踪 in-flight 任务
_inflight: dict[str, dict] = {}
_inflight_lock = threading.Lock()


# ---------------------------------------------------------------------------
# fork 隔离调用 .so
# ---------------------------------------------------------------------------

def _fork_and_call(lib_path: str, entry_call, need_stderr: bool = True
                   ) -> tuple[int, str]:
    """fork 子进程，加载 .so 并调用 entry_call(lib)，返回 (exit_code, stderr)。

    entry_call(lib) 接收 CDLL 对象，返回 int 退出码。
    若库内部调用 exit()，子进程直接退出，父进程通过 waitpid 获取退出码。
    """
    result_r, result_w = os.pipe()
    stderr_tmp = tempfile.TemporaryFile() if need_stderr else None

    pid = os.fork()
    if pid == 0:
        # ---- 子进程 ----
        os.close(result_r)
        if stderr_tmp:
            os.dup2(stderr_tmp.fileno(), 2)

        try:
            lib = ctypes.CDLL(lib_path)
            code = entry_call(lib)
            os.write(result_w, struct.pack("i", code))
        except BaseException:
            pass
        os._exit(0)

    # ---- 父进程 ----
    os.close(result_w)
    _, status = os.waitpid(pid, 0)

    buf = os.read(result_r, 4)
    os.close(result_r)

    if buf and len(buf) == 4:
        exit_code = struct.unpack("i", buf)[0]
    elif os.WIFEXITED(status):
        exit_code = os.WEXITSTATUS(status)
    else:
        exit_code = -1

    stderr_text = ""
    if stderr_tmp:
        stderr_tmp.seek(0)
        stderr_text = stderr_tmp.read().decode("utf-8", errors="replace")
        stderr_tmp.close()

    return exit_code, stderr_text


# ---------------------------------------------------------------------------
# tippecanoe 调用
# ---------------------------------------------------------------------------

def _tippecanoe_entry(lib: ctypes.CDLL, args: list[str]) -> int:
    """在子进程中调用 tippecanoe_main。"""
    # C 约定: argv[0] = 程序名, argv[1:] = 实际参数
    full_args = ["tippecanoe"] + args
    argc = len(full_args)
    argv_type = ctypes.c_char_p * argc
    argv = argv_type(*[a.encode("utf-8") for a in full_args])
    return lib.tippecanoe_main(argc, argv)


# ---------------------------------------------------------------------------
# tile-join-ext 调用
# ---------------------------------------------------------------------------

# 命令行 flag -> (option_key, option_value)
_TJ_FLAG_MAP = {
    "--force": ("force", "1"),
    "--no-tile-compression": ("no_tile_compression", "1"),
    "--no-tile-size-limit": ("no_tile_size_limit", "1"),
    "--quiet": ("quiet", "1"),
    "--overzoom": ("overzoom", "1"),
    "--if-matched": ("if_matched", "1"),
    "--exclude-all": ("exclude_all", "1"),
    "--boundary-only": ("boundary_only", "1"),
    "--drop-densest-as-needed": ("drop_densest_as_needed", "1"),
}

# 命令行 key-value 参数 -> option_key
_TJ_KV_MAP = {
    "--buffer": "buffer",
    "--attribution": "attribution",
    "--name": "name",
    "--description": "description",
    "--minimum-zoom": "minzoom",
    "--maximum-zoom": "maxzoom",
    "--csv": "csv",
    "--exclude": "exclude",
    "--include": "include",
    "--keep-layer": "keep_layer",
    "--remove-layer": "remove_layer",
    "--rename-layer": "rename_layer",
    "--feature-filter": "feature_filter",
    "--feature-filter-file": "feature_filter_file",
    "--read-from": "read_from",
}


def _tile_join_entry(lib: ctypes.CDLL, inputs: list[str], output: str,
                     options: dict[str, str]) -> int:
    """在子进程中调用 tile-join-ext API。"""
    # 设置函数签名 (ctypes 默认 restype=int 会导致 64 位指针截断)
    lib.tile_join_ext_create.restype = ctypes.c_void_p
    lib.tile_join_ext_create.argtypes = []
    lib.tile_join_ext_add_input.restype = ctypes.c_int
    lib.tile_join_ext_add_input.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tile_join_ext_set_output.restype = ctypes.c_int
    lib.tile_join_ext_set_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tile_join_ext_set_option.restype = ctypes.c_int
    lib.tile_join_ext_set_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.tile_join_ext_run.restype = ctypes.c_int
    lib.tile_join_ext_run.argtypes = [ctypes.c_void_p]
    lib.tile_join_ext_destroy.restype = None
    lib.tile_join_ext_destroy.argtypes = [ctypes.c_void_p]

    ctx = lib.tile_join_ext_create()
    for inp in inputs:
        lib.tile_join_ext_add_input(ctx, inp.encode("utf-8"))
    lib.tile_join_ext_set_output(ctx, output.encode("utf-8"))
    for k, v in options.items():
        lib.tile_join_ext_set_option(ctx, k.encode("utf-8"), v.encode("utf-8"))
    ret = lib.tile_join_ext_run(ctx)
    lib.tile_join_ext_destroy(ctx)
    return ret


def _parse_join_args(join_args: list[str], output: str, inputs: list[str]
                      ) -> dict[str, str]:
    """将命令行 join_args 解析为 tile-join-ext API options dict。"""
    options: dict[str, str] = {}
    i = 0
    while i < len(join_args):
        arg = join_args[i]
        if arg in _TJ_FLAG_MAP:
            k, v = _TJ_FLAG_MAP[arg]
            options[k] = v
        elif arg in _TJ_KV_MAP and i + 1 < len(join_args):
            options[_TJ_KV_MAP[arg]] = join_args[i + 1]
            i += 1
        elif arg == "-o":
            pass  # output 已单独处理
        elif arg.startswith("--") and i + 1 < len(join_args) and not join_args[i + 1].startswith("-"):
            # 通用 fallback: --foo-bar value -> ("foo_bar", value)
            key = arg[2:].replace("-", "_")
            options[key] = join_args[i + 1]
            i += 1
        elif arg.startswith("--"):
            # 通用 fallback: --foo-bar -> ("foo_bar", "1")
            key = arg[2:].replace("-", "_")
            options[key] = "1"
        i += 1
    return options


# ---------------------------------------------------------------------------
# 进度监控
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s"


def _add_inflight(key: str, task: str, province: str, layer: str = ""):
    with _inflight_lock:
        _inflight[key] = {
            "task": task, "start": time.time(),
            "province": province, "layer": layer,
        }


def _remove_inflight(key: str):
    with _inflight_lock:
        _inflight.pop(key, None)


def progress_monitor(active: threading.Event):
    while active.is_set():
        time.sleep(60)
        if not active.is_set():
            break
        now = time.time()
        lines = ["[STATUS] 各 worker 运行状态:"]
        with _inflight_lock:
            if not _inflight:
                lines.append("  (无活跃 worker)")
            for key, info in _inflight.items():
                elapsed = now - info["start"]
                prov = info.get("province", "")
                layer = info.get("layer", "")
                task = info.get("task", "")
                dur = _format_duration(elapsed)
                if layer:
                    lines.append(f"  {key} [{dur}] {task}: {prov}/{layer}")
                else:
                    lines.append(f"  {key} [{dur}] {task}: {prov}")
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# 配置 & 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def scan_subdirs(input_dir: Path) -> list[Path]:
    subdirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and list(d.glob("*.geojson"))
    )
    if not subdirs:
        print(f"[ERROR] {input_dir} 下没有找到包含 .geojson 的子目录", file=sys.stderr)
        sys.exit(1)
    return subdirs


def dedup_names(names: list[str]) -> dict[str, str]:
    if len(names) <= 1:
        return {names[0]: names[0]}
    rev = [n[::-1] for n in names]
    common_len = 0
    for chars in zip(*rev):
        if len(set(chars)) == 1:
            common_len += 1
        else:
            break
    if common_len >= 3:
        return {n: n[:-common_len] if n[:-common_len] else n for n in names}
    return {n: n for n in names}


# ---------------------------------------------------------------------------
# tippecanoe worker (ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_tippecanoe(args_tuple: tuple) -> dict:
    """执行单个 tippecanoe 任务 (一个省份的一个图层)。"""
    (subdir, short_name, layer_name, layer_cfg,
     temp_dir, global_args, dry_run,
     tippecanoe_lib) = args_tuple

    province = subdir.name
    geojson_file = layer_cfg["geojson"]
    geojson_path = subdir / geojson_file

    if not geojson_path.exists():
        return {"province": province, "short_name": short_name,
                "layer": layer_name, "status": "geojson_not_found"}
    if geojson_path.stat().st_size == 0:
        return {"province": province, "short_name": short_name,
                "layer": layer_name, "status": "geojson_empty"}

    mbtiles_path = temp_dir / f"{layer_name}.mbtiles"

    # 构建参数列表（与命令行 tippecanoe 相同）
    cmd_args = list(global_args)
    cmd_args += ["-Z", str(layer_cfg["zoom_min"]), "-z", str(layer_cfg["zoom_max"])]
    cmd_args += ["-l", layer_name]
    cmd_args += ["-o", str(mbtiles_path)]
    cmd_args += layer_cfg.get("tippecanoe_args", [])
    cmd_args += [str(geojson_path)]

    if dry_run:
        display = f"tippecanoe {' '.join(cmd_args)}"
        print(f"[DRY-RUN] {province}/{layer_name}: {display}")
        return {"province": province, "short_name": short_name,
                "layer": layer_name, "status": "dry_run"}

    start = time.time()

    # 使用 libtippecanoe.so (fork 隔离)
    exit_code, stderr = _fork_and_call(
        tippecanoe_lib,
        lambda lib: _tippecanoe_entry(lib, cmd_args),
        need_stderr=True,
    )
    ok = (exit_code == 0)
    err = stderr.strip()[-300:] if not ok else ""

    elapsed = time.time() - start

    if not ok:
        return {"province": province, "short_name": short_name,
                "layer": layer_name, "status": "error",
                "error": err, "elapsed": elapsed}

    size_mb = round(mbtiles_path.stat().st_size / (1024 * 1024), 2)
    return {"province": province, "short_name": short_name,
            "layer": layer_name, "status": "ok", "size_mb": size_mb,
            "elapsed": elapsed, "mbtiles_path": str(mbtiles_path)}


# ---------------------------------------------------------------------------
# tile-join worker (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def tile_join_worker(short_name: str, temp_dir: Path, output_root: Path,
                     join_args: list[str], ok_count: int, total: int,
                     tile_join_lib: str | None = None) -> dict:
    """线程池 worker: 执行 tile-join。"""
    key = f"tj:{short_name}"
    _add_inflight(key, "tile-join", short_name)

    start = time.time()
    mbtiles_files = sorted(temp_dir.glob("*.mbtiles"))
    output_mbtiles = output_root / f"{short_name}.mbtiles"

    if not mbtiles_files:
        _remove_inflight(key)
        return {"province": short_name, "status": "no_files",
                "elapsed": time.time() - start}

    inputs = [str(f) for f in mbtiles_files]
    output = str(output_mbtiles)

    # 使用 libtile-join-ext.so (fork 隔离)
    options = _parse_join_args(join_args, output, inputs)
    exit_code, stderr = _fork_and_call(
        tile_join_lib,
        lambda lib: _tile_join_entry(lib, inputs, output, options),
        need_stderr=True,
    )
    ok = (exit_code == 0)
    err = stderr.strip()[-300:] if not ok else ""

    elapsed = time.time() - start
    _remove_inflight(key)

    if not ok:
        return {"province": short_name, "status": "error",
                "error": err, "elapsed": elapsed}

    size_mb = round(output_mbtiles.stat().st_size / (1024 * 1024), 2)
    return {"province": short_name, "status": "ok",
            "size_mb": size_mb, "elapsed": elapsed,
            "ok_count": ok_count, "total": total}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoJSON -> mbtiles 按省份并发转换 (使用 libtippecanoe.so / libtile-join-ext.so)")
    parser.add_argument("-c", "--config", default="geojson_to_mbtiles.json",
                        help="JSON 配置文件路径, 默认 geojson_to_mbtiles.json")
    parser.add_argument("-i", "--input", default=None,
                        help="输入根目录 (包含各省 GeoJSON 子目录)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出根目录 (mbtiles 文件)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="tippecanoe 并发 worker 数, 默认 4")
    parser.add_argument("-j", "--join-workers", type=int, default=4,
                        help="tile-join 并发 worker 数, 默认 4")
    parser.add_argument("-p", "--province", default=None,
                        help="指定子目录名, 用于测试单省")
    parser.add_argument("--tippecanoe-lib", required=True,
                        help="libtippecanoe.so 路径")
    parser.add_argument("--tile-join-ext-lib", required=True,
                        help="libtile-join-ext.so 路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令, 不执行")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)

    # 输入目录
    if args.input:
        input_dir = Path(args.input).resolve()
    else:
        input_dir = Path(cfg.get("input_dir", "./geojson_output"))
        if not input_dir.is_absolute():
            input_dir = (config_path.parent / input_dir).resolve()

    # 输出目录
    if args.output:
        output_root = Path(args.output).resolve()
    else:
        output_root = Path(cfg.get("output_dir", "./mbtiles_output"))
        if not output_root.is_absolute():
            output_root = (config_path.parent / output_root).resolve()

    # 临时目录
    temp_root = Path(cfg.get("temp_dir", "./mbtiles_temp"))
    if not temp_root.is_absolute():
        temp_root = (config_path.parent / temp_root).resolve()

    global_args = cfg.get("global_tippecanoe_args", ["--force"])
    join_args = cfg.get("tile_join_args", ["--force", "--no-tile-compression"])
    layers_cfg = cfg.get("layers", {})
    text_layers_cfg = cfg.get("text_layers", {})

    subdirs = scan_subdirs(input_dir)

    # 单省测试
    if args.province:
        matched = [d for d in subdirs if d.name == args.province]
        if not matched:
            print(f"[ERROR] 子目录不存在: {args.province}", file=sys.stderr)
            sys.exit(1)
        subdirs = matched

    raw_names = [d.name for d in subdirs]
    name_map = dedup_names(raw_names)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 配置: {config_path}")
    print(f"[INFO] 输入: {input_dir} ({len(subdirs)} 个子目录)")
    print(f"[INFO] 输出: {output_root}")
    print(f"[INFO] 临时: {temp_root}")
    print(f"[INFO] libtippecanoe: {args.tippecanoe_lib}")
    print(f"[INFO] libtile-join-ext: {args.tile_join_lib}")
    print(f"[INFO] 图层: {len(layers_cfg)} 个 ({', '.join(layers_cfg.keys())})")
    if text_layers_cfg:
        print(f"[INFO] 文字图层: {len(text_layers_cfg)} 个 ({', '.join(text_layers_cfg.keys())})")
    print(f"[INFO] tippecanoe 并发: {args.workers} workers")
    print(f"[INFO] tile-join 并发: {args.join_workers} workers")
    if args.dry_run:
        print("[INFO] DRY-RUN 模式")

    # 构建所有图层任务
    all_tasks = []
    for sd in subdirs:
        short_name = name_map[sd.name]
        temp_dir = temp_root / short_name
        if not args.dry_run:
            temp_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, layer_cfg in layers_cfg.items():
            all_tasks.append((sd, short_name, layer_name, layer_cfg,
                              temp_dir, global_args, args.dry_run,
                              args.tippecanoe_lib))
        for layer_name, layer_cfg in text_layers_cfg.items():
            all_tasks.append((sd, short_name, layer_name, layer_cfg,
                              temp_dir, global_args, args.dry_run,
                              args.tippecanoe_lib))

    total_per_province = len(layers_cfg) + len(text_layers_cfg)

    province_done = defaultdict(int)
    province_ok = defaultdict(int)
    province_errors = defaultdict(list)
    province_temp_dir = {}

    workers = min(args.workers, len(all_tasks))

    # 进度监控
    monitor_active = threading.Event()
    monitor_active.set()
    monitor_thread = threading.Thread(target=progress_monitor, args=(monitor_active,), daemon=True)
    if not args.dry_run:
        monitor_thread.start()

    join_executor = ThreadPoolExecutor(max_workers=args.join_workers)
    join_futures = {}

    total_tippecanoe_ok = 0
    total_tippecanoe_err = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_tippecanoe, t): t for t in all_tasks}

        for future, task in futures.items():
            province = task[0].name
            layer_name = task[2]
            _add_inflight(f"tc:{province}:{layer_name}", "tippecanoe", province, layer_name)

        for future in as_completed(futures):
            task = futures[future]
            province = task[0].name
            short_name = task[1]
            layer_name = task[2]
            province_temp_dir[short_name] = task[4]

            _remove_inflight(f"tc:{province}:{layer_name}")

            try:
                r = future.result()
                province_done[province] += 1

                if r["status"] == "ok":
                    province_ok[province] += 1
                    total_tippecanoe_ok += 1
                    elapsed = r.get("elapsed", 0)
                    print(f"  [tippecanoe] {r['province']}/{r['layer']} ok "
                          f"({r['size_mb']} MB, {_format_duration(elapsed)})")
                elif r["status"] == "error":
                    province_errors[province].append(
                        f"{r['layer']}: {r.get('error', '')}")
                    total_tippecanoe_err += 1
                    print(f"  [tippecanoe] {r['province']}/{r['layer']} ERROR: "
                          f"{r.get('error', '')[:100]}")
                elif r["status"] in ("geojson_not_found", "geojson_empty"):
                    print(f"  [tippecanoe] {r['province']}/{r['layer']} "
                          f"skip ({r['status']})")

                # 该省所有图层完成 -> 提交 tile-join
                if province_done[province] == total_per_province:
                    errs = province_errors[province]
                    ok_count = province_ok[province]

                    if errs:
                        print(f"  [{province}] {ok_count}/{total_per_province} ok, "
                              f"{len(errs)} errors (skip tile-join)")
                        for e in errs:
                            print(f"    {e}")
                    elif not args.dry_run:
                        temp_dir = province_temp_dir[short_name]
                        fut = join_executor.submit(
                            tile_join_worker, short_name, temp_dir,
                            output_root, join_args, ok_count, total_per_province,
                            args.tile_join_lib)
                        join_futures[fut] = short_name
                        print(f"  [{province}] {ok_count}/{total_per_province} ok "
                              f"-> tile-join 已提交")
                    else:
                        print(f"  [{province}] {ok_count}/{total_per_province} ok "
                              f"(dry-run, skip tile-join)")

            except Exception as e:
                province = task[0].name
                province_done[province] += 1
                province_errors[province].append(f"exception: {e}")
                print(f"  [{province}] 异常: {e}", file=sys.stderr)

    # 等待所有 tile-join 完成
    total_join_ok = 0
    total_join_err = 0

    for future in as_completed(join_futures):
        short_name = join_futures[future]
        try:
            r = future.result()
            if r["status"] == "ok":
                total_join_ok += 1
                print(f"  [tile-join] {r['province']} ok "
                      f"({r['size_mb']} MB, {_format_duration(r['elapsed'])})")
                temp_dir = province_temp_dir.get(r["province"])
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            elif r["status"] == "error":
                total_join_err += 1
                print(f"  [tile-join] {r['province']} ERROR: "
                      f"{r.get('error', '')[:100]}", file=sys.stderr)
        except Exception as e:
            total_join_err += 1
            print(f"  [tile-join] {short_name} 异常: {e}", file=sys.stderr)

    join_executor.shutdown(wait=True)
    monitor_active.clear()
    if not args.dry_run:
        monitor_thread.join(timeout=5)

    print(f"\n[DONE] {len(subdirs)} 省")
    print(f"  tippecanoe: {total_tippecanoe_ok} ok, {total_tippecanoe_err} errors")
    print(f"  tile-join:  {total_join_ok} ok, {total_join_err} errors")

    if total_tippecanoe_err or total_join_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
