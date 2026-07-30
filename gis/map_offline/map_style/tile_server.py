#!/usr/bin/env python3
"""
多 MBTiles 瓦片服务器（已处理 TMS → XYZ 翻转）

用法:
    python3 serve_mbtiles.py [端口] [索引数据库] [兜底mbtiles路径]

默认端口: 3000
默认索引数据库: tile_index.db
默认兜底数据源: /storage/emulated/0/map/data/shape_tile/china.mbtiles
"""
import gzip
import http.server
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import re
import mapbox_vector_tile

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 3000
INDEX_DB_FILE = "./tile_index.db"
FALLBACK_PATH = ""
# FALLBACK_PATH = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FALLBACK_PATH

FALLBACK_SOURCE_ID = 0

CONTENT_TYPES = {
    'pbf': 'application/x-protobuf',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
}

SOURCES = {}
TILE_INDEX = {}
THREAD_LOCAL = threading.local()
MERGED_SOURCE_NAMES = {'merged'}

PERF_LOG_FILE = "performance.log"
PERF_LOG_LOCK = threading.Lock()


def log_performance(z, x, y, tile_size, mbtiles_path, elapsed_ms):
    """记录瓦片查询耗时到 performance.log"""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"z={z} x={x} y={y} size={tile_size} "
        f"elapsed={elapsed_ms:.3f}ms file={mbtiles_path}\n"
    )
    with PERF_LOG_LOCK:
        with open(PERF_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line)


def get_tile_format(db):
    try:
        cursor = db.execute("SELECT value FROM metadata WHERE name='format'")
        row = cursor.fetchone()
        if row:
            fmt = row[0].lower()
            if fmt in CONTENT_TYPES:
                return fmt
    except Exception:
        pass
    return None


def xyz_to_tms(z, y):
    """将 XYZ 的 y 坐标转换为 TMS 的 y 坐标"""
    return (1 << z) - 1 - y


def open_readonly_db(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_source_format(source):
    # source['path'] = re.sub(r'\.\./([^/]*)\.mbtiles', r'/mnt/e/deve/map/mbtiles_output/mbtiles/\1.mbtiles', source['path'])
    # source['path'] = re.sub('boundary\.mbtiles', '/mnt/e/deve/map/mbtiles_output/tile_index/boundary.mbtiles', source['path'])
    if source.get('is_merged'):
        return 'pbf'

    if not os.path.exists(source['path']):
        print(f"警告: 数据源文件不存在: {source['name']} -> {source['path']}")
        return None

    try:
        db = open_readonly_db(source['path'])
        fmt = get_tile_format(db)
        db.close()
        return fmt
    except Exception as e:
        print(f"警告: 读取数据源格式失败: {source['name']} -> {e}")
        return None


def load_index(index_path):
    """加载 tile_index.db 索引。

    sources 表包含所有区域源和 merged 源。
    兜底源（china）不在 sources 表中，单独处理。
    """
    fallback_source = {
        'id': FALLBACK_SOURCE_ID,
        'name': 'fallback',
        'path': FALLBACK_PATH,
        'priority': 999999,
        'is_merged': False,
    }
    sources = {FALLBACK_SOURCE_ID: fallback_source}
    tile_index = {}

    if not os.path.exists(index_path):
        print(f"警告: 找不到索引数据库 {index_path}，所有瓦片将从兜底数据源读取")
        # fallback_source['format'] = load_source_format(fallback_source)
        return sources, tile_index

    db = open_readonly_db(index_path)

    source_columns = [row[1] for row in db.execute("PRAGMA table_info(sources)")]
    has_is_merged = 'is_merged' in source_columns
    source_sql = (
        "SELECT id, name, path, priority, is_merged FROM sources ORDER BY priority"
        if has_is_merged else
        "SELECT id, name, path, priority, 0 AS is_merged FROM sources ORDER BY priority"
    )

    for source_id, name, path, priority, is_merged in db.execute(source_sql):
        sources[source_id] = {
            'id': source_id,
            'name': name,
            'path': path,
            'priority': priority,
            'is_merged': bool(is_merged) or name in MERGED_SOURCE_NAMES,
        }

    for z, x, y_tms, source_id in db.execute(
        "SELECT zoom, tile_column, tile_row, source_id FROM tile_index"
    ):
        tile_index[(z, x, y_tms)] = source_id

    db.close()

    for source in sources.values():
        source['format'] = load_source_format(source)

    return sources, tile_index


def get_thread_connections():
    if not hasattr(THREAD_LOCAL, 'connections'):
        THREAD_LOCAL.connections = {}
    return THREAD_LOCAL.connections


def get_source_db(source):
    connections = get_thread_connections()
    source_id = source['id']
    if source_id not in connections:
        connections[source_id] = open_readonly_db(source['path'])
    return connections[source_id]


def get_source_for_tile(z, x, y_tms):
    source_id = TILE_INDEX.get((z, x, y_tms), FALLBACK_SOURCE_ID)
    return SOURCES.get(source_id, SOURCES[FALLBACK_SOURCE_ID])


def query_tile(source, z, x, y_tms):
    db = get_source_db(source)
    cursor = db.execute(
        "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
        (z, x, y_tms)
    )
    row = cursor.fetchone()
    return row[0] if row else None


class MBTilesHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)
        print(f'path:${path}')

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'MBTiles Server is running')
            return

        if path == '/search':
            q = query_params.get('q', [''])[0].strip()
            if not q:
                self._send_json([])
                return

            print(f"收到搜索请求: {q}")
            results = self._search_live(q)
            print(f"搜索返回结果数: {len(results)}")
            for res in results:
                print(f"  - {res['name']} ({res.get('layer', 'unknown')}): [{res['lat']:.9f}, {res['lon']:.9f}]")
            self._send_json(results)
            return

        if path.startswith('/fonts/'):
            parts = path.split('/')
            if len(parts) >= 4:
                fontstack = urllib.parse.unquote(parts[2])
                range_pbf = parts[3]
                font_path = os.path.join('fonts', fontstack, range_pbf)
                if os.path.exists(font_path):
                    with open(font_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-protobuf')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                    return
            self.send_response(404)
            self.end_headers()
            return

        if not path.startswith('/tiles/') and not path.startswith('/data/'):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
            return

        parts = path.split('/')
        try:
            if path.startswith('/tiles/'):
                z = int(parts[2])
                x = int(parts[3])
                y_raw = parts[4]
            elif path.startswith('/data/'):
                z = int(parts[3])
                x = int(parts[4])
                y_raw = parts[5]
            else:
                raise ValueError("Unknown path format")

            y_clean = re.sub(r'\.[^.]+$', '', y_raw)
            y_xyz = int(y_clean)
            y_tms = xyz_to_tms(z, y_xyz)
        except (ValueError, IndexError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request')
            return

        try:
            source = get_source_for_tile(z, x, y_tms)
            print(f"读取 mbtiles: {source['name']} -> {source['path']}  (z={z}, x={x}, y={y_xyz})")
            start_time = time.perf_counter()
            tile_blob = query_tile(source, z, x, y_tms)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if tile_blob is None and source['id'] != FALLBACK_SOURCE_ID:
                # fallback_source = SOURCES[FALLBACK_SOURCE_ID]
                print(f"主数据源无瓦片，回退到: {fallback_source['name']} -> {fallback_source['path']}  (z={z}, x={x}, y={y_xyz})")
                # tile_blob = query_tile(fallback_source, z, x, y_tms)
                # source = fallback_source

            tile_size = len(tile_blob) if tile_blob else 0
            log_performance(z, x, y_xyz, tile_size, source['path'], elapsed_ms)

            if tile_blob:
                self._send_tile(tile_blob, source)
            else:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'Tile not found')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())
            return

    def _send_tile(self, tile_blob, source):
        content_type = CONTENT_TYPES.get(source.get('format'), 'application/octet-stream')
        is_gzip = False
        try:
            is_gzip = isinstance(tile_blob, (bytes, bytearray)) and len(tile_blob) >= 2 and tile_blob[0] == 0x1f and tile_blob[1] == 0x8b
        except Exception:
            is_gzip = False

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        if is_gzip:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('X-MBTiles-Source', source['name'])
        self.send_header('Content-Length', len(tile_blob))
        self.end_headers()
        self.wfile.write(tile_blob)

    def log_message(self, format, *args):
        pass

    def _send_json(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def _search_live(self, query):
        results = []
        seen_names = set()
        query_lower = query.lower()

        for source in sorted(SOURCES.values(), key=lambda item: item['priority']):
            if len(results) >= 20:
                break

            if source.get('is_merged'):
                continue

            try:
                db = get_source_db(source)

                cursor = db.execute("SELECT MAX(zoom_level) FROM tiles")
                max_zoom = cursor.fetchone()[0]
                if max_zoom is None:
                    continue

                cursor = db.execute(
                    "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?",
                    (max_zoom,)
                )

                for z, x, y_tms, data in cursor:
                    if len(results) >= 20:
                        break

                    try:
                        tile_bytes = data
                        if tile_bytes.startswith(b'\x1f\x8b'):
                            tile_bytes = gzip.decompress(tile_bytes)

                        decoded_tile = mapbox_vector_tile.decode(tile_bytes, default_options={'y_coord_down': True})
                        y_xyz = (1 << z) - 1 - y_tms
                        n = 2.0 ** z

                        for layer_name, layer_data in decoded_tile.items():
                            extent = layer_data.get('extent', 4096)
                            for feature in layer_data['features']:
                                name = feature.get('properties', {}).get('name')
                                if not name or query_lower not in name.lower():
                                    continue

                                geom = feature.get('geometry', {})
                                coords = geom.get('coordinates', [])

                                if geom['type'] == 'Point':
                                    px, py = coords
                                elif geom['type'] in ['LineString', 'MultiPoint']:
                                    px, py = coords[0]
                                elif geom['type'] in ['Polygon', 'MultiLineString']:
                                    px, py = coords[0][0]
                                elif geom['type'] == 'MultiPolygon':
                                    px, py = coords[0][0][0]
                                else:
                                    continue

                                lon = (x + px / extent) / n * 360.0 - 180.0
                                y_frac = (y_xyz + py / extent) / n
                                lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_frac)))
                                lat = math.degrees(lat_rad)

                                result_key = (name, round(lat, 6), round(lon, 6))
                                if result_key not in seen_names:
                                    results.append({
                                        "name": name,
                                        "lat": lat,
                                        "lon": lon,
                                        "layer": layer_name,
                                        "source": source['name'],
                                    })
                                    seen_names.add(result_key)
                    except Exception:
                        continue
            except Exception as e:
                print(f"搜索数据源失败: {source['name']} -> {e}")

        return results


if __name__ == '__main__':
    SOURCES, TILE_INDEX = load_index(INDEX_DB_FILE)

    print(f"MBTiles 瓦片服务启动: http://localhost:{PORT}/tiles/{{z}}/{{x}}/{{y}} (已启用 TMS→XYZ 翻转)")
    print(f"索引数据库: {INDEX_DB_FILE}")
    print(f"兜底数据源: {FALLBACK_PATH}")
    print(f"索引瓦片数: {len(TILE_INDEX)}")
    print("数据源:")
    for source in sorted(SOURCES.values(), key=lambda item: item['priority']):
        print(f"  - {source['name']}: {source['path']}  格式: {source.get('format') or '未知'}")

    server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), MBTilesHandler)
    server.daemon_threads = True

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()
