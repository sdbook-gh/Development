#!/usr/bin/env python3
"""
MBTiles 瓦片服务器（已处理 TMS → XYZ 翻转）
用法: python3 serve_mbtiles.py <mbtiles文件> [端口]
默认端口: 3000
"""
import http.server
import sqlite3
import sys
import os
import urllib.parse
import re

MBTILES_FILE = sys.argv[1] if len(sys.argv) > 1 else 'beijing.mbtiles'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

def get_tile_format(db):
    try:
        cursor = db.execute("SELECT value FROM metadata WHERE name='format'")
        row = cursor.fetchone()
        if row:
            fmt = row[0].lower()
            if fmt in ('pbf', 'png', 'jpg', 'jpeg', 'webp'):
                return fmt
    except:
        pass
    return None

db_conn = sqlite3.connect(f"file:{MBTILES_FILE}?mode=ro", uri=True)
TILE_FORMAT = get_tile_format(db_conn)
db_conn.close()

CONTENT_TYPES = {
    'pbf': 'application/x-protobuf',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
}

def xyz_to_tms(z, y):
    """将 XYZ 的 y 坐标转换为 TMS 的 y 坐标"""
    return (1 << z) - 1 - y

import json
import zlib
import sqlite3
import math
import gzip
import re
import mapbox_vector_tile

def xyz_to_latlon(z, x, y_tms):
    """将 MBTiles 的 Z/X/Y (TMS) 转换为经纬度坐标"""
    y_xyz = (1 << z) - 1 - y_tms
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_xyz / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

class MBTilesHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'MBTiles Server is running')
            return

        # 新增搜索接口：实时在数据库中搜索
        if path == '/search':
            q = query_params.get('q', [''])[0].strip()
            if not q:
                self._send_json([])
                return
            
            print(f"收到搜索请求: {q}")
            results = self._search_live(q)
            print(f"搜索返回结果数: {len(results)}")
            self._send_json(results)
            return

        # 新增字体服务接口
        if path.startswith('/fonts/'):
            # 路径格式: /fonts/{fontstack}/{range}.pbf
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

        # 处理 /tiles/z/x/y 或 /data/beijing-260531/z/x/y.pbf
        parts = path.split('/')
        try:
            if path.startswith('/tiles/'):
                # /tiles/z/x/y
                z = int(parts[2])
                x = int(parts[3])
                y_raw = parts[4]
            elif path.startswith('/data/'):
                # /data/beijing-260531/z/x/y.pbf
                z = int(parts[3])
                x = int(parts[4])
                y_raw = parts[5]
            else:
                raise ValueError("Unknown path format")

            y_clean = re.sub(r'\.[^.]+$', '', y_raw)
            y_xyz = int(y_clean)
            # 关键：转换为 TMS 坐标
            y_tms = xyz_to_tms(z, y_xyz)
        except (ValueError, IndexError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request')
            return

        try:
            db = sqlite3.connect(f"file:{MBTILES_FILE}?mode=ro", uri=True)
            cursor = db.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, y_tms)
            )
            row = cursor.fetchone()
            db.close()

            if row:
                tile_blob = row[0]
                content_type = CONTENT_TYPES.get(TILE_FORMAT, 'application/octet-stream')
                # 检测是否为 gzip 压缩
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
                self.send_header('Content-Length', len(tile_blob))
                self.end_headers()
                self.wfile.write(tile_blob)
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
        """
        与 search_mbtiles.py 逻辑完全一致，使用 mapbox-vector-tile 解析。
        """
        results = []
        try:
            db = sqlite3.connect(f"file:{MBTILES_FILE}?mode=ro", uri=True)
            query_lower = query.lower()
            
            # 获取最大级别
            cursor = db.execute("SELECT MAX(zoom_level) FROM tiles")
            max_zoom = cursor.fetchone()[0]
            
            # 搜索最大级别
            cursor = db.execute(
                "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?",
                (max_zoom,)
            )
            
            seen_names = set()
            for z, x, y, data in cursor:
                try:
                    # 解压
                    tile_bytes = data
                    if tile_bytes.startswith(b'\x1f\x8b'):
                        tile_bytes = gzip.decompress(tile_bytes)
                    
                    # 使用 mapbox_vector_tile 解析
                    decoded_tile = mapbox_vector_tile.decode(tile_bytes)
                    
                    for layer_name, layer_data in decoded_tile.items():
                        for feature in layer_data['features']:
                            name = feature.get('properties', {}).get('name')
                            if name and query_lower in name.lower():
                                if name not in seen_names:
                                    lat, lon = xyz_to_latlon(z, x, y)
                                    results.append({"name": name, "lat": lat, "lon": lon})
                                    seen_names.add(name)
                                    
                except: continue
                
                if len(results) >= 20: break
            
            db.close()
        except Exception as e:
            print(f"搜索过程出错: {e}")
        return results

if __name__ == '__main__':
    if not os.path.exists(MBTILES_FILE):
        print(f"错误: 找不到文件 {MBTILES_FILE}")
        sys.exit(1)

    print(f"MBTiles 瓦片服务启动: http://localhost:{PORT}/tiles/{{z}}/{{x}}/{{y}} (已启用 TMS→XYZ 翻转)")
    print(f"数据文件: {MBTILES_FILE}  瓦片格式: {TILE_FORMAT or '未知'}")
    
    server = http.server.HTTPServer(('0.0.0.0', PORT), MBTilesHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()