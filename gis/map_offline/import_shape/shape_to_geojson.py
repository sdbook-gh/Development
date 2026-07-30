#!/usr/bin/env python3
"""
读取 shape_to_geojson.json，将各子目录下的 SHP 按 shapefilematch 匹配并转为 GeoJSONSeq。
使用 libgdal.so (通过 Python GDAL 绑定的 gdal.VectorTranslate) 代替 ogr2ogr 子进程。

用法:
    python shape_to_geojson.py --gdal-lib /path/to/libgdal.so -c shape_to_geojson.json -i ./ -o ./geojson_output -w 8
    python shape_to_geojson.py --gdal-lib /path/to/libgdal.so --dry-run -l roads
"""

import argparse
import ctypes
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# 预解析 --gdal-lib，在导入 osgeo 之前加载指定的 libgdal.so
# ---------------------------------------------------------------------------
_gdal_lib_path = None
if "--gdal-lib" in sys.argv:
    idx = sys.argv.index("--gdal-lib")
    if idx + 1 < len(sys.argv):
        _gdal_lib_path = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]

if _gdal_lib_path:
    ctypes.CDLL(_gdal_lib_path, mode=ctypes.RTLD_GLOBAL)

from osgeo import gdal

gdal.UseExceptions()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def scan_subdirs(input_dir: Path) -> list[Path]:
    """扫描 input_dir 下所有包含 .shp 的子目录."""
    subdirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and list(d.glob("*.shp"))
    )
    if not subdirs:
        print(f"[ERROR] {input_dir} 下没有找到包含 .shp 的子目录", file=sys.stderr)
        sys.exit(1)
    return subdirs


def dedup_names(names: list[str]) -> dict[str, str]:
    """从子目录名中提取去重后的短名称.

    例: anhui-latest-free.shp, beijing-latest-free.shp
      -> 公共后缀 "-latest-free.shp"
      -> 去重后: anhui, beijing
    """
    if len(names) <= 1:
        n = names[0]
        return {n: n.replace(".shp", "")}

    # 找最长公共后缀 (反转后找公共前缀)
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


def match_shapefile(subdir: Path, pattern: str, layer_name: str) -> Path | None:
    """用正则匹配子目录下的 shape 文件，处理面/点冲突."""
    regex = re.compile(pattern)
    candidates = sorted(f for f in subdir.glob("*.shp") if regex.search(f.name))

    if not candidates:
        print(f"  [WARN] {subdir.name}/{layer_name}: 无匹配文件, 正则={pattern}",
              file=sys.stderr)
        return None

    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        print(f"  [WARN] {subdir.name}/{layer_name}: 多个匹配文件 [{names}], 正则={pattern}",
              file=sys.stderr)
        return None

    return candidates[0]


def build_sql(layer_cfg: dict, shp_path: str) -> str:
    """构建 SQL 语句."""
    select_fields = layer_cfg["select"]
    layer_name = Path(shp_path).stem
    where = layer_cfg.get("where", "")

    sql = f'SELECT {select_fields} FROM "{layer_name}"'
    if where:
        sql += f" WHERE {where}"
    return sql


def build_options(layer_cfg: dict, global_gdal_args: list | None) -> list[str]:
    """构建 gdal.VectorTranslate options 列表."""
    options = []
    options += [str(a) for a in (global_gdal_args or [])]
    options += [str(a) for a in layer_cfg.get("gdal_args", [])]
    return options


def run_vector_translate(
    sql: str,
    options: list[str],
    output_file: str,
    shp_path: str,
    simplify: float,
    dry_run: bool,
) -> tuple[bool, str]:
    """调用 gdal.VectorTranslate 执行转换.

    返回 (ok, error_msg)
    """
    full_options = list(options)
    full_options += ["-sql", sql]
    if simplify and float(simplify) > 0:
        full_options += ["-simplify", str(simplify)]

    if dry_run:
        # 模拟 ogr2ogr 命令行输出格式
        cmd_str = "ogr2ogr " + " ".join(full_options) + f" {output_file} {shp_path}"
        print(f"[DRY-RUN] {cmd_str}")
        return True, ""

    try:
        result = gdal.VectorTranslate(output_file, shp_path, options=full_options)
        if result is None:
            err = gdal.GetLastErrorMsg()
            return False, err or "Unknown error"
        result = None  # 关闭并刷新输出
        return True, ""
    except Exception as e:
        return False, str(e)


def apply_ogr_result(
    stats: dict,
    layer_key: str,
    output_file: str,
    ok: bool,
    err: str,
    dry_run: bool,
) -> None:
    """写入 layers 统计：成功有数据 / 空输出跳过 / 失败。"""
    if not ok:
        stats["layers"][layer_key] = "error"
        stats["errors"].append(f"{layer_key}: {err}")
        return
    if dry_run:
        stats["layers"][layer_key] = "ok"
        return
    out = Path(output_file)
    size = out.stat().st_size if out.exists() else 0
    if size <= 0:
        if out.exists():
            out.unlink(missing_ok=True)
        print(f"  [WARN] {stats['subdir']}/{layer_key}: 输出为空（无要素），跳过",
              file=sys.stderr)
        stats["layers"][layer_key] = "empty"
        return
    stats["layers"][layer_key] = round(size / (1024 * 1024), 2)


def process_subdir(args_tuple: tuple) -> dict:
    """处理单个子目录: 先处理 layers，再处理 text_layers (如有).

    layers 输出: {layer}.geojson
    text_layers 输出: {layer}_text.geojson
    """
    (subdir, short_name, layers_cfg, text_layers_cfg,
     output_root, global_gdal_args, dry_run) = args_tuple

    subdir_name = subdir.name
    stats = {"subdir": subdir_name, "layers": {}, "errors": []}

    out_dir = output_root / short_name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 处理全量图层
    for layer_name, layer_cfg in layers_cfg.items():
        pattern = layer_cfg["shapefilematch"]
        shp_path = match_shapefile(subdir, pattern, layer_name)

        if shp_path is None:
            stats["layers"][layer_name] = "not_found"
            continue

        output_file = str(out_dir / f"{layer_name}.geojson")
        sql = build_sql(layer_cfg, str(shp_path))
        options = build_options(layer_cfg, global_gdal_args)
        simplify = layer_cfg.get("simplify", 0)

        ok, err = run_vector_translate(
            sql, options, output_file, str(shp_path), simplify, dry_run)
        apply_ogr_result(stats, layer_name, output_file, ok, err, dry_run)

    # 处理文字图层 (如有)
    if text_layers_cfg:
        for layer_name, layer_cfg in text_layers_cfg.items():
            pattern = layer_cfg["shapefilematch"]
            shp_path = match_shapefile(subdir, pattern, layer_name)

            if shp_path is None:
                stats["layers"][f"{layer_name}_text"] = "not_found"
                continue

            output_file = str(out_dir / f"{layer_name}_text.geojson")
            sql = build_sql(layer_cfg, str(shp_path))
            options = build_options(layer_cfg, global_gdal_args)
            simplify = layer_cfg.get("simplify", 0)

            ok, err = run_vector_translate(
                sql, options, output_file, str(shp_path), simplify, dry_run)
            apply_ogr_result(stats, f"{layer_name}_text", output_file, ok, err, dry_run)

    return stats


def main():
    parser = argparse.ArgumentParser(description="SHP -> GeoJSONSeq 按省份分目录转换 (使用 libgdal.so)")
    parser.add_argument("--gdal-lib", required=True,
                        help="指定 libgdal.so 路径 (在导入 osgeo 前预加载)")
    parser.add_argument("-c", "--config", default="shape_to_geojson.json",
                        help="JSON 配置文件路径, 默认 shape_to_geojson.json")
    parser.add_argument("-i", "--input", default=None,
                        help="输入根目录 (包含各省子目录), 默认使用配置文件中的 input_dir")
    parser.add_argument("-o", "--output", default=None,
                        help="输出根目录, 默认使用配置文件中的 output_dir")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="并发 worker 数, 默认 4")
    parser.add_argument("-l", "--layers", nargs="+", default=None,
                        help="指定图层 (空格分隔), 默认全部")
    parser.add_argument("-p", "--province", default=None,
                        help="指定子目录名 (如 jiangxi-latest-free.shp), 用于测试单省")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不执行")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)

    # 输入目录: 命令行 > 配置文件 > 当前目录
    if args.input:
        input_dir = Path(args.input).resolve()
    else:
        input_dir = Path(cfg.get("input_dir", "."))
        if not input_dir.is_absolute():
            input_dir = (config_path.parent / input_dir).resolve()

    # 输出目录: 命令行 > 配置文件
    if args.output:
        output_root = Path(args.output).resolve()
    else:
        output_root = Path(cfg.get("output_dir", "./geojson_output"))
        if not output_root.is_absolute():
            output_root = (config_path.parent / output_root).resolve()

    all_layers = cfg["layers"]
    text_layers = cfg.get("text_layers", {})
    global_gdal_args = cfg.get("global_gdal_args", [])

    if args.layers:
        layers = {k: all_layers[k] for k in args.layers if k in all_layers}
        if not layers:
            print(f"[ERROR] 无效图层: {args.layers}", file=sys.stderr)
            print(f"  可用: {', '.join(all_layers.keys())}", file=sys.stderr)
            sys.exit(1)
        # -l 只影响全量图层，不影响 text_layers
    else:
        layers = all_layers

    subdirs = scan_subdirs(input_dir)

    # 单省测试模式
    if args.province:
        matched = [d for d in subdirs if d.name == args.province]
        if not matched:
            print(f"[ERROR] 子目录不存在: {args.province}", file=sys.stderr)
            sys.exit(1)
        subdirs = matched

    # 子目录名去重
    raw_names = [d.name for d in subdirs]
    name_map = dedup_names(raw_names)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    gdal_ver = gdal.__version__
    print(f"[INFO] GDAL version: {gdal_ver}")
    print(f"[INFO] libgdal: {_gdal_lib_path}")
    print(f"[INFO] 配置: {config_path}")
    print(f"[INFO] 输入: {input_dir} ({len(subdirs)} 个子目录)")
    print(f"[INFO] 输出: {output_root}")
    print(f"[INFO] 全量图层: {len(layers)} 个 ({', '.join(layers.keys())})")
    if text_layers:
        print(f"[INFO] 文字图层: {len(text_layers)} 个 ({', '.join(text_layers.keys())})")
    if global_gdal_args:
        print(f"[INFO] global_gdal_args: {' '.join(str(a) for a in global_gdal_args)}")
    print(f"[INFO] 并发: {args.workers} workers")
    if args.dry_run:
        print("[INFO] DRY-RUN 模式")

    tasks = [
        (sd, name_map.get(sd.name, sd.name), layers, text_layers,
         output_root, global_gdal_args, args.dry_run)
        for sd in subdirs
    ]

    total_stats = []
    workers = min(args.workers, len(tasks))
    total_layer_count = len(layers) + len(text_layers)
    total_layers_ok = 0
    total_layers_err = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_subdir, t): t[0].name for t in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                s = future.result()
                total_stats.append(s)

                # 统计
                ok_count = sum(1 for v in s["layers"].values()
                               if isinstance(v, (int, float)) or v == "ok")
                err_count = len(s["errors"])
                short = name_map.get(name, name)
                total_layers_ok += ok_count
                total_layers_err += err_count

                icon = "OK" if err_count == 0 else "ERR"
                print(f"  [{icon}] {short:<16s} {ok_count}/{total_layer_count} 层成功")
                for e in s["errors"]:
                    print(f"       ERROR: {e}")
            except Exception as e:
                print(f"  [ERR] {name}: {e}")

    print(f"\n{'='*50}")
    print(f"[SUMMARY] 子目录: {len(subdirs)}, 图层/子目录: {total_layer_count}")
    print(f"[SUMMARY] 成功: {total_layers_ok} 层, 失败: {total_layers_err} 层")
    print(f"[SUMMARY] 输出: {output_root}")


if __name__ == "__main__":
    main()
