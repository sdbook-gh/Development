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
import multiprocessing
import os
import re
import signal
import sys
import time
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

# ── GDAL C 层输出转发 ─────────────────────────────────────
# UseExceptions 会把 libgdal.so 的错误/警告静默转为 Python 异常，
# 导致屏幕看不到 GDAL 输出。安装自定义错误处理器，将 C 层输出打印
# 到 stderr，同时保留异常行为（VectorTranslate/OpenEx 失败仍抛异常）。
def _gdal_error_handler(err_class, err_num, err_msg):
    """libgdal.so 的 C 层错误处理器：将 GDAL 输出打印到屏幕."""
    if err_class == gdal.CE_Warning:
        tag = "WARN"
    elif err_class == gdal.CE_Failure:
        tag = "ERROR"
    elif err_class == gdal.CE_Fatal:
        tag = "FATAL"
    elif err_class == gdal.CE_Debug:
        tag = "DEBUG"
    else:
        tag = f"L{err_class}"
    print(f"[GDAL-{tag}] {err_msg}", file=sys.stderr)

gdal.SetErrorHandler(_gdal_error_handler)

# ── 信号处理 ──────────────────────────────────────────────
_interrupted = False

def _force_kill_children():
    """第二次 Ctrl+C: 向所有子进程发送 SIGKILL 并报告状态.

    先报告检测到的子进程并逐个发送 SIGKILL，然后等待最多 2 秒；
    若仍有子进程存活，打印手动 kill 命令列表供用户执行。
    """
    children = multiprocessing.active_children()
    if not children:
        print("[FORCE] 无子进程存活", file=sys.stderr)
        return

    print(f"[FORCE] 检测到 {len(children)} 个子进程，发送 SIGKILL", file=sys.stderr)
    for p in children:
        print(f"  [FORCE] PID {p.pid}: 发送 SIGKILL", file=sys.stderr)
        try:
            os.kill(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            print(f"  [FORCE] PID {p.pid}: 进程不存在（可能已退出）", file=sys.stderr)
        except PermissionError as e:
            print(f"  [FORCE] PID {p.pid}: 无权限杀死: {e}", file=sys.stderr)

    # 等待最多 2 秒确认子进程退出
    print("[FORCE] 等待 2 秒确认子进程退出...", file=sys.stderr)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not any(p.is_alive() for p in children):
            break
        time.sleep(0.2)

    alive = [p for p in children if p.is_alive()]
    if alive:
        print(f"[FORCE] 仍有 {len(alive)} 个子进程未退出:", file=sys.stderr)
        for p in alive:
            print(f"  [FORCE] PID {p.pid} 仍存活", file=sys.stderr)
        print("[FORCE] 请手动执行以下命令强制杀死:", file=sys.stderr)
        for p in alive:
            print(f"  kill -9 {p.pid}", file=sys.stderr)
    else:
        print("[FORCE] 所有子进程均已退出", file=sys.stderr)


def _signal_handler(sig, frame):
    global _interrupted
    if _interrupted:
        # 第二次 Ctrl+C: 强制杀死子进程
        print("\n[FORCE] 再次收到终止信号，强制杀死子进程", file=sys.stderr)
        _force_kill_children()
        print("[FORCE] 主进程退出", file=sys.stderr)
        os._exit(1)
    _interrupted = True
    print("\n[INTERRUPTED] 收到终止信号，正在停止...", file=sys.stderr)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def scan_subdirs(input_dir: Path) -> list[Path]:
    """扫描 input_dir 下所有包含 .shp 的子目录.

    若 input_dir 自身直接包含 .shp 文件，则将其视为单个子目录返回。
    """
    # 若 input_dir 自身直接包含 .shp 文件，视为单省份目录
    if list(input_dir.glob("*.shp")):
        return [input_dir]

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


def match_shapefile(subdir: Path, pattern: str, layer_name: str, config_name: str) -> Path | None:
    """用正则匹配子目录下的 shape 文件，处理面/点冲突."""
    regex = re.compile(pattern)
    candidates = sorted(f for f in subdir.glob("*.shp") if regex.search(f.name))

    if not candidates:
        print(f"  [WARN] [{config_name}] {subdir.name}/{layer_name}: 无匹配文件, 正则={pattern}",
              file=sys.stderr)
        return None

    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        print(f"  [WARN] [{config_name}] {subdir.name}/{layer_name}: 多个匹配文件 [{names}], 正则={pattern}",
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


# ── 文字图层质心转换 ──────────────────────────────────
# text_layers 用于标注，Polygon/MultiPolygon 跨瓦片会产生重复标签。
# 转为质心 Point 后标签只出现在单一瓦片中。

def _ring_centroid(ring: list) -> tuple[list[float] | None, float]:
    """单个闭合环的面积加权质心.

    ring: [[x, y], ...]（首尾点相同）
    返回 (centroid[x, y], abs_area)，退化时 area=0
    """
    n = len(ring)
    if n < 3:
        return None, 0.0
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area = a2 * 0.5
    if abs(area) < 1e-15:
        # 退化（共线/零面积）：回退到 bounding box 中心
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return [sum(xs) / len(xs), sum(ys) / len(ys)], 0.0
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return [cx, cy], abs(area)


def _polygon_centroid(coords: list) -> tuple[list[float] | None, float]:
    """Polygon: coords = [exterior_ring, hole1, ...]. 仅用 exterior ring."""
    if not coords:
        return None, 0.0
    return _ring_centroid(coords[0])


def _multipolygon_centroid(coords: list) -> list[float] | None:
    """MultiPolygon: coords = [poly1, poly2, ...].

    取外环顶点最多的子多边形的质心（主体边界通常最精细）。
    面积加权法在含远距离飞地时质心会偏移（如同名不同区被合并），
    顶点数能可靠区分主体与飞地。
    """
    best_centroid = None
    best_nverts = 0
    for poly in coords:
        if not poly:
            continue
        ext_ring = poly[0]
        nverts = len(ext_ring)
        centroid, _ = _ring_centroid(ext_ring)
        if centroid is not None and nverts > best_nverts:
            best_centroid = centroid
            best_nverts = nverts
    if best_centroid is not None:
        return best_centroid
    # 全部退化：取所有顶点算术平均
    pts = [p for poly in coords if poly for p in poly[0]]
    if pts:
        return [sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts)]
    return None


def convert_text_layer_to_centroid(output_file: str) -> bool:
    """将 text_layer 的 GeoJSONSeq 中 Polygon/MultiPolygon 转为质心 Point.

    Point/LineString 等非面几何保持不变。
    原地重写文件（保持 RS=YES 的 0x1E 前缀格式）。
    返回 True 表示有要素被转换。
    """
    try:
        with open(output_file, "rb") as f:
            data = f.read()
    except OSError:
        return False

    # GeoJSONSeq (RS=YES): 每条记录以 0x1E 开头
    records = data.split(b"\x1e")
    if records and records[0] == b"":
        records = records[1:]

    converted = False
    out_chunks: list[bytes] = []
    for raw in records:
        text = raw.strip()
        if not text:
            continue
        feat = json.loads(text.decode("utf-8"))
        geom = feat.get("geometry")
        if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
            gcoords = geom.get("coordinates")
            if geom["type"] == "Polygon":
                centroid, _ = _polygon_centroid(gcoords)
            else:
                centroid = _multipolygon_centroid(gcoords)
            if centroid is not None:
                feat["geometry"] = {"type": "Point", "coordinates": centroid}
                converted = True
        out_text = json.dumps(feat, ensure_ascii=False, separators=(",", ":"))
        out_chunks.append(b"\x1e" + out_text.encode("utf-8") + b"\n")

    if converted:
        with open(output_file, "wb") as f:
            f.writelines(out_chunks)
    return converted


def count_features(shp_path: str, where: str) -> int:
    """统计满足 where 条件的要素数,用于 fallback 预判与 dry-run 预览。

    返回 -1 表示打开失败(调用方按"可能非空"处理或跳过预查)。
    """
    try:
        ds = gdal.OpenEx(shp_path, gdal.OF_VECTOR)
        if ds is None:
            return -1
        layer = ds.GetLayerByIndex(0)
        if where:
            layer.SetAttributeFilter(where)
        cnt = layer.GetFeatureCount()
        ds = None
        return cnt
    except Exception:
        return -1


def merge_fallback(base_cfg: dict, fb: dict) -> dict:
    """fallback 条目继承主配置,fb 中指定字段覆盖;移除 fallbacks 避免递归。"""
    merged = {k: v for k, v in base_cfg.items() if k != "fallbacks"}
    merged.update(fb)
    return merged


def process_text_layer(
    layer_name: str,
    layer_cfg: dict,
    subdir: Path,
    out_dir: Path,
    global_gdal_args: list | None,
    dry_run: bool,
    config_name: str,
    stats: dict,
) -> None:
    """处理单个 text_layer,支持 fallbacks 链式回退。

    主查询输出为空(0 要素)时,依次尝试 fallbacks;每个 fallback 可覆盖任意字段,
    未指定字段继承主配置。主查询文件未匹配(not_found)不触发回退。
    dry-run 模式下用 count_features 预查要素数,只为非空候选打印命令。
    """
    subdir_name = subdir.name
    fallbacks = layer_cfg.get("fallbacks", []) or []
    candidates = [layer_cfg] + [merge_fallback(layer_cfg, fb) for fb in fallbacks]

    output_file = str(out_dir / f"{layer_name}_text.geojson")
    final_ok = False

    for idx, cfg in enumerate(candidates):
        label = "主查询" if idx == 0 else f"fallback#{idx}"

        pattern = cfg["shapefilematch"]
        shp_path = match_shapefile(subdir, pattern, layer_name, config_name)
        if shp_path is None:
            if idx == 0:
                # 主查询文件未匹配:不触发回退
                stats["layers"][f"{layer_name}_text"] = "not_found"
                return
            print(f"  [WARN] {subdir_name}/{layer_name}_text {label}: 无匹配文件, 跳过",
                  file=sys.stderr)
            continue

        where = cfg.get("where", "")
        cnt = count_features(str(shp_path), where)

        if dry_run:
            if cnt > 0:
                sql = build_sql(cfg, str(shp_path))
                options = build_options(cfg, global_gdal_args)
                simplify = cfg.get("simplify", 0)
                run_vector_translate(sql, options, output_file,
                                     str(shp_path), simplify, dry_run=True)
                final_ok = True
                break
            print(f"  [INFO] {subdir_name}/{layer_name}_text {label}: 预查 {cnt} 要素, 尝试下一回退",
                  file=sys.stderr)
            continue

        # 真实执行:先预查,空则跳过执行以避免写出空文件
        if cnt <= 0:
            print(f"  [INFO] {subdir_name}/{layer_name}_text {label}: 预查 {cnt} 要素, 尝试下一回退",
                  file=sys.stderr)
            continue

        sql = build_sql(cfg, str(shp_path))
        options = build_options(cfg, global_gdal_args)
        simplify = cfg.get("simplify", 0)
        ok, err = run_vector_translate(sql, options, output_file,
                                       str(shp_path), simplify, dry_run=False)
        if not ok:
            stats["layers"][f"{layer_name}_text"] = "error"
            stats["errors"].append(f"{layer_name}_text: {err}")
            return
        # text_layers: 将 Polygon/MultiPolygon 转为质心 Point
        convert_text_layer_to_centroid(output_file)
        out = Path(output_file)
        size = out.stat().st_size if out.exists() else 0
        if size > 0:
            final_ok = True
            break
        if out.exists():
            out.unlink(missing_ok=True)
        print(f"  [INFO] {subdir_name}/{layer_name}_text {label}: 输出为空, 尝试下一回退",
              file=sys.stderr)

    if final_ok:
        if dry_run:
            stats["layers"][f"{layer_name}_text"] = "ok"
        else:
            out = Path(output_file)
            size = out.stat().st_size if out.exists() else 0
            stats["layers"][f"{layer_name}_text"] = round(size / (1024 * 1024), 2)
    else:
        if not dry_run:
            out = Path(output_file)
            if out.exists():
                out.unlink(missing_ok=True)
        print(f"  [WARN] {subdir_name}/{layer_name}_text: 所有查询(含回退)均无要素, 跳过",
              file=sys.stderr)
        stats["layers"][f"{layer_name}_text"] = "empty"


def process_subdir(args_tuple: tuple) -> dict:
    """处理单个子目录: 先处理 layers，再处理 text_layers (如有).

    layers 输出: {layer}.geojson
    text_layers 输出: {layer}_text.geojson
    """
    (subdir, short_name, layers_cfg, text_layers_cfg,
     output_root, global_gdal_args, dry_run, config_name) = args_tuple

    subdir_name = subdir.name
    stats = {"subdir": subdir_name, "layers": {}, "errors": []}

    out_dir = output_root / short_name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 处理全量图层
    for layer_name, layer_cfg in layers_cfg.items():
        pattern = layer_cfg["shapefilematch"]
        shp_path = match_shapefile(subdir, pattern, layer_name, config_name)

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

    # 处理文字图层 (如有,支持 fallbacks 回退)
    if text_layers_cfg:
        for layer_name, layer_cfg in text_layers_cfg.items():
            process_text_layer(
                layer_name, layer_cfg, subdir, out_dir,
                global_gdal_args, dry_run, config_name, stats)

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
         output_root, global_gdal_args, args.dry_run, config_path.name)
        for sd in subdirs
    ]

    total_stats = []
    workers = min(args.workers, len(tasks))
    total_layer_count = len(layers) + len(text_layers)
    total_layers_ok = 0
    total_layers_err = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_subdir, t): t[0].name for t in tasks}
        try:
            for future in as_completed(futures):
                if _interrupted:
                    print("[INFO] 跳过剩余任务", file=sys.stderr)
                    break

                name = futures[future]
                try:
                    s = future.result(timeout=1)
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
        finally:
            if _interrupted:
                # 取消未开始的任务，不等待正在运行的任务
                executor.shutdown(wait=False, cancel_futures=True)
                print("[INFO] 已取消未执行的任务，正在退出...", file=sys.stderr)
            else:
                executor.shutdown(wait=True)

    print(f"\n{'='*50}")
    print(f"[SUMMARY] 子目录: {len(subdirs)}, 图层/子目录: {total_layer_count}")
    print(f"[SUMMARY] 成功: {total_layers_ok} 层, 失败: {total_layers_err} 层")
    print(f"[SUMMARY] 输出: {output_root}")


if __name__ == "__main__":
    main()
