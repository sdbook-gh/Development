import argparse
import gzip
import sqlite3
import sys

try:
    import mapbox_vector_tile
except ImportError:
    print("缺少依赖: pip install mapbox-vector-tile", file=sys.stderr)
    sys.exit(1)
def decode_tile(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
def lookup(mbtiles_path, name, layer_name, fclass_filter):
    db = sqlite3.connect(f"file:{mbtiles_path}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles"
    ).fetchall()
    db.close()
    hits = []
    for z, x, y_tms, data in rows:
        try:
            decoded = decode_tile(data)
        except Exception:
            continue
        layer = decoded.get(layer_name)
        if not layer:
            continue
        for f in layer["features"]:
            props = f.get("properties", {}) or {}
            if props.get("name") != name:
                continue
            if fclass_filter and props.get("fclass") != fclass_filter:
                continue
            y_xyz = (1 << z) - 1 - y_tms
            hits.append({
                "z": z,
                "x": x,
                "y_tms": y_tms,
                "y_xyz": y_xyz,
                "fclass": props.get("fclass"),
                "code": props.get("code"),
                "population": props.get("population"),
                "osm_id": props.get("osm_id"),
            })
    return hits
def main():
    parser = argparse.ArgumentParser(description="按完整名称查找 MBTiles 中的要素")
    parser.add_argument("mbtiles", help="MBTiles 文件路径")
    parser.add_argument("name", help="要素的完整名称（name 属性精确匹配）")
    parser.add_argument("layer", nargs="?", default="places", help="矢量图层名，默认 places")
    parser.add_argument("--fclass", default=None, help="可选：限定 fclass，如 region/city/county")
    args = parser.parse_args()
    print(f"MBTiles : {args.mbtiles}")
    print(f"Layer   : {args.layer}")
    print(f"Name    : {args.name}")
    if args.fclass:
        print(f"fclass  : {args.fclass}")
    print("-" * 80)
    hits = lookup(args.mbtiles, args.name, args.layer, args.fclass)
    if not hits:
        print("未找到匹配要素")
        return
    header = f"{'z':>3} {'x':>7} {'y_tms':>7} {'y_xyz':>7}  {'fclass':<10} {'code':>6} {'population':>12}  osm_id"
    print(header)
    print("-" * len(header))
    for h in hits:
        print(f"{h['z']:>3} {h['x']:>7} {h['y_tms']:>7} {h['y_xyz']:>7}  "
              f"{str(h['fclass']):<10} {str(h['code']):>6} {str(h['population']):>12}  {h['osm_id']}")
    print("-" * 80)
    print(f"命中瓦片数: {len(hits)}")
if __name__ == "__main__":
    main()
