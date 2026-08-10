#!/usr/bin/env python3
"""
读取 shape_to_place.json，将各子目录下的 SHP 按 shapefilematch 匹配并写入 place.db。

与 Android PlacesDbBuilder 使用相同 schema 与字段映射；通过 OGR 要素游标直读 shape（无 GeoJSON 中间文件）。

用法:
    python shape_to_place.py --gdal-lib /path/to/libgdal.so -c shape_to_place.json -i ./beijing_hebei -o ./place.db
    python shape_to_place.py --gdal-lib /path/to/libgdal.so -c shape_to_place.json -i ./ -o ./place.db --merge
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import sqlite3
import sys
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

try:
    from osgeo import ogr, osr
except ImportError:
    print("[ERROR] 需要 GDAL Python 绑定: pip install gdal 或系统安装 python3-gdal", file=sys.stderr)
    sys.exit(1)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS places (
  id INTEGER PRIMARY KEY,
  osm_id TEXT NOT NULL,
  name TEXT NOT NULL,
  layer TEXT NOT NULL,
  fclass TEXT NOT NULL DEFAULT '',
  population INTEGER NOT NULL DEFAULT 0,
  province TEXT NOT NULL DEFAULT '',
  lat REAL NOT NULL,
  lng REAL NOT NULL,
  search_text TEXT NOT NULL,
  UNIQUE(osm_id, layer)
);
CREATE INDEX IF NOT EXISTS idx_places_rank ON places(name, population DESC, id);
"""


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def scan_subdirs(input_dir: Path) -> tuple[bool, list[Path]]:
    root_has = any(input_dir.glob("*.shp"))
    subdirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and list(d.glob("*.shp"))
    )
    return root_has, subdirs


def dedup_names(names: list[str]) -> dict[str, str]:
    if len(names) <= 1:
        n = names[0] if names else ""
        return {n: n.replace(".shp", "")} if n else {}

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
    regex = re.compile(pattern)
    candidates = sorted(f for f in subdir.glob("*.shp") if regex.search(f.name))
    if not candidates:
        print(f"  [WARN] [{config_name}] {subdir.name}/{layer_name}: 无匹配文件, 正则={pattern}", file=sys.stderr)
        return None
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        print(f"  [WARN] [{config_name}] {subdir.name}/{layer_name}: 多个匹配 [{names}], 跳过", file=sys.stderr)
        return None
    return candidates[0]


def ensure_schema(conn: sqlite3.Connection, merge: bool) -> None:
    if not merge:
        conn.executescript("DROP TABLE IF EXISTS places_fts; DROP TABLE IF EXISTS places;")
    conn.executescript(SCHEMA_SQL)
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='places_fts'"
    ).fetchone():
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE places_fts USING fts5(
                  name, search_text, content='places', content_rowid='id',
                  tokenize='trigram'
                )
                """
            )
        except sqlite3.OperationalError:
            conn.execute(
                """
                CREATE VIRTUAL TABLE places_fts USING fts5(
                  name, search_text, content='places', content_rowid='id'
                )
                """
            )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO places_fts(places_fts) VALUES('rebuild')")
    conn.commit()


def feature_lng_lat(geom, transform) -> tuple[float, float] | None:
    if geom is None or geom.IsEmpty():
        return None
    gtype = geom.GetGeometryType()
    flat = ogr.GT_Flatten(gtype)
    if flat == ogr.wkbPoint:
        work = geom
    else:
        work = geom.Centroid()
        if work is None or work.IsEmpty():
            return None
    lng = work.GetX()
    lat = work.GetY()
    if transform is not None:
        work_clone = ogr.Geometry(ogr.wkbPoint)
        work_clone.AddPoint(lng, lat)
        work_clone.Transform(transform)
        lng = work_clone.GetX()
        lat = work_clone.GetY()
    # 纠正轴序颠倒
    if (lat < -90.0 or lat > 90.0) and -90.0 <= lng <= 90.0:
        lng, lat = lat, lng
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return lng, lat


def ingest_shape(
    conn: sqlite3.Connection,
    shp_path: Path,
    layer_key: str,
    layer_cfg: dict,
    province: str,
) -> int:
    select = layer_cfg["select"]
    field_names = [f.strip() for f in select.split(",") if f.strip()]
    if not field_names:
        return 0
    field_index = {name.lower(): i for i, name in enumerate(field_names)}
    name_idx = field_index.get("name")
    if name_idx is None:
        return 0

    ds = ogr.Open(str(shp_path), 0)
    if ds is None:
        raise RuntimeError(f"无法打开 {shp_path}")
    layer = ds.GetLayer(0)
    where = layer_cfg.get("where", "")
    if where:
        layer.SetAttributeFilter(where)

    transform = None
    srs = layer.GetSpatialRef()
    if srs is not None:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not srs.IsSame(wgs84):
            transform = osr.CoordinateTransformation(srs, wgs84)

    count = 0
    insert_sql = """
        INSERT OR REPLACE INTO places(osm_id, name, layer, fclass, population, province, lat, lng, search_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for feat in layer:
        geom = feat.GetGeometryRef()
        coords = feature_lng_lat(geom, transform)
        if coords is None:
            continue
        lng, lat = coords

        fields = []
        for name in field_names:
            idx = feat.GetFieldIndex(name)
            if idx < 0:
                fields.append("")
            else:
                fields.append(feat.GetFieldAsString(idx) or "")

        name = fields[name_idx].strip()
        if not name:
            continue

        osm_id = fields[field_index["osm_id"]].strip() if "osm_id" in field_index else ""
        if not osm_id:
            # Python 内置 hash() 带随机盐、跨进程不稳定，重建/合并库会产生重复行
            osm_id = f"{layer_key}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:12]}"
        fclass = fields[field_index["fclass"]].strip() if "fclass" in field_index else ""
        population = 0
        if "population" in field_index:
            raw = fields[field_index["population"]].strip()
            if raw.isdigit():
                population = int(raw)

        search_parts = [name]
        if province:
            search_parts.append(province)
        if fclass:
            search_parts.append(fclass)
        for i, key in enumerate(field_names):
            if i in {name_idx, field_index.get("osm_id"), field_index.get("fclass"), field_index.get("population")}:
                continue
            val = fields[i].strip()
            if val:
                search_parts.append(val)

        conn.execute(
            insert_sql,
            (osm_id, name, layer_key, fclass, population, province, lat, lng, " ".join(search_parts)),
        )
        count += 1

    ds = None
    conn.commit()
    return count


def collect_tasks(config: dict, input_dir: Path, config_name: str) -> list[tuple[Path, str, str, dict, str]]:
    root_has, subdirs = scan_subdirs(input_dir)
    short_map = dedup_names([d.name for d in subdirs]) if subdirs else {}
    tasks: list[tuple[Path, str, str, dict, str]] = []
    layers = config.get("layers", {})
    dirs = []
    if root_has:
        dirs.append((input_dir, ""))
    for d in subdirs:
        dirs.append((d, short_map.get(d.name, d.name)))

    for subdir, province in dirs:
        for layer_key, layer_cfg in layers.items():
            if not layer_cfg.get("enabled", True):
                continue
            shp = match_shapefile(subdir, layer_cfg["shapefilematch"], layer_key, config_name)
            if shp is not None:
                tasks.append((shp, layer_key, province, layer_cfg, subdir.name))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="从 shapefile 构建 place.db")
    parser.add_argument("--gdal-lib", required=True,
                        help="指定 libgdal.so 路径 (在导入 osgeo 前预加载)")
    parser.add_argument("-c", "--config", default="shape_to_place.json", help="JSON 配置文件")
    parser.add_argument("-i", "--input", default=".", help="含省级子目录或根 .shp 的输入目录")
    parser.add_argument("-o", "--output", required=True, help="输出 place.db 路径")
    parser.add_argument("--merge", action="store_true", help="合并到已有 place.db（INSERT OR REPLACE）")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_dir = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] libgdal: {_gdal_lib_path}")

    tasks = collect_tasks(config, input_dir, Path(args.config).name)
    if not tasks:
        print("[WARN] 无 places 入库任务", file=sys.stderr)
        return 1

    if not args.merge and output.exists():
        output.unlink()

    conn = sqlite3.connect(output)
    try:
        ensure_schema(conn, merge=args.merge and output.exists())
        total_rows = 0
        for shp, layer_key, province, layer_cfg, subdir_name in tasks:
            print(f"[INFO] {subdir_name}/{layer_key}: {shp.name}")
            rows = ingest_shape(conn, shp, layer_key, layer_cfg, province)
            print(f"  -> {rows} 条")
            total_rows += rows
        rebuild_fts(conn)
        print(f"[OK] place.db 共 {total_rows} 条 → {output}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
