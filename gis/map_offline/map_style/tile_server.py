#!/usr/bin/env python3
"""
多 MBTiles 瓦片服务器（已处理 TMS → XYZ 翻转）- 架构优化版

用法:
    python3 tile_server.py [--search-dir <目录>]

参数:
    -h, --help          显示帮助信息
    --search-dir <目录> 忽略 sources 表 path 的目录名，从指定目录搜索同名 mbtiles

默认端口: 3000
默认索引数据库: tile_index.db

优化项:
    - SQLite 连接池（线程安全）
    - 瓦片 LRU 内存缓存
    - 搜索倒排索引（启动时预构建）
    - 字体/精灵文件内存缓存
    - 性能日志缓冲批量写入
    - 线程数限制
"""
import gzip
import http.server
import json
import math
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import mapbox_vector_tile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 3000
INDEX_DB_FILE = "./tile_index.db"
FALLBACK_PATH = ""
SEARCH_MBTILES_DIR = ""

USAGE_TEXT = """多 MBTiles 瓦片服务器（已处理 TMS → XYZ 翻转）- 架构优化版

用法:
    python3 tile_server.py [--search-dir <目录>]

参数:
    -h, --help              显示本帮助信息
    --search-dir <目录>     忽略 tile_index.db 中 sources 表 path 字段的目录名，
                            改为从该目录下搜索同名的 mbtiles 文件

默认配置:
    端口:          3000
    索引数据库:    ./tile_index.db
    兜底数据源:    空（可在代码中修改 FALLBACK_PATH 配置）
    搜索目录:      空（不指定 --search-dir 时按 sources 表原始路径读取）

示例:
    python3 tile_server.py
    python3 tile_server.py --search-dir /path/to/mbtiles
"""

SPRITES_DIR = os.path.join(BASE_DIR, "sprites")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

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
MERGED_SOURCE_NAMES = {'merged'}

# ─── 优化配置 ────────────────────────────────────────────────────
MAX_THREADS = 100              # 最大线程数
TILE_CACHE_SIZE = 4096         # 瓦片缓存条目数
FONT_CACHE_SIZE = 256          # 字体文件缓存条目数
SPRITE_CACHE_SIZE = 256        # 精灵文件缓存条目数
LOG_FLUSH_INTERVAL = 1.0       # 日志刷新间隔（秒）
LOG_FLUSH_BATCH = 100          # 日志批量刷新阈值
DB_POOL_SIZE = 20              # 每个数据源的连接池大小

# ─── 线程限制 ────────────────────────────────────────────────────
_thread_semaphore = threading.BoundedSemaphore(MAX_THREADS)


# ═══════════════════════════════════════════════════════════════════
#  SQLite 连接池
# ═══════════════════════════════════════════════════════════════════
class SQLiteConnectionPool:
    """线程安全的 SQLite 只读连接池"""

    def __init__(self, db_path, max_size=DB_POOL_SIZE):
        self._db_path = db_path
        self._max_size = max_size
        self._pool = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created = 0

    def _create_connection(self):
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True
        )
        try:
            conn.execute("PRAGMA query_only=ON")
        except Exception:
            pass
        return conn

    def acquire(self, timeout=5.0):
        """获取一个连接，超时返回 None"""
        # 尝试从池中获取
        try:
            conn = self._pool.get_nowait()
            # 检查连接是否有效
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                # 连接已失效，创建新连接
                with self._lock:
                    self._created -= 1
        except queue.Empty:
            pass

        # 池为空，尝试创建新连接
        with self._lock:
            if self._created < self._max_size:
                self._created += 1
                try:
                    return self._create_connection()
                except Exception:
                    self._created -= 1
                    raise

        # 已达上限，等待池中回收
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, conn):
        """归还连接到池中"""
        if conn is None:
            return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            # 池满，关闭连接
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created -= 1

    def close_all(self):
        """关闭所有连接"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
        with self._lock:
            self._created = 0


# 连接池管理：source_id -> SQLiteConnectionPool
_connection_pools = {}
_pool_lock = threading.Lock()


def get_connection_pool(source):
    """获取指定数据源的连接池"""
    source_id = source['id']
    if source_id not in _connection_pools:
        with _pool_lock:
            if source_id not in _connection_pools:
                _connection_pools[source_id] = SQLiteConnectionPool(
                    source['path'], max_size=DB_POOL_SIZE
                )
    return _connection_pools[source_id]


# ═══════════════════════════════════════════════════════════════════
#  瓦片 LRU 缓存
# ═══════════════════════════════════════════════════════════════════
class TileCache:
    """线程安全的 LRU 瓦片缓存"""

    def __init__(self, maxsize=TILE_CACHE_SIZE):
        self._maxsize = maxsize
        self._cache = {}
        self._order = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # 移到末尾（最近使用）
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._maxsize:
                # 淘汰最久未使用
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[key] = value
            self._order.append(key)

    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
            }

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._hits = 0
            self._misses = 0


# 全局瓦片缓存实例
_tile_cache = TileCache(TILE_CACHE_SIZE)


# ═══════════════════════════════════════════════════════════════════
#  文件缓存（字体/精灵）
# ═══════════════════════════════════════════════════════════════════
class FileCache:
    """LRU 文件内容缓存"""

    def __init__(self, maxsize=256):
        self._maxsize = maxsize
        self._cache = {}
        self._order = []
        self._lock = threading.Lock()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
                self._order.append(path)
                return self._cache[path]
            return None

    def put(self, path, content):
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
            elif len(self._cache) >= self._maxsize:
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[path] = content
            self._order.append(path)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()


_font_cache = FileCache(FONT_CACHE_SIZE)
_sprite_cache = FileCache(SPRITE_CACHE_SIZE)


# ═══════════════════════════════════════════════════════════════════
#  搜索倒排索引
# ═══════════════════════════════════════════════════════════════════
class SearchIndex:
    """预构建的搜索倒排索引，将 name → [(name, lat, lon, layer, source), ...]"""

    def __init__(self):
        self._index = {}
        self._lock = threading.Lock()
        self._built = False

    def build(self, sources):
        """启动时构建搜索索引"""
        print("正在构建搜索索引...")
        start = time.time()
        count = 0

        for source in sorted(sources.values(), key=lambda s: s['priority']):
            if source.get('is_merged') or not source.get('path'):
                continue
            if not os.path.exists(source['path']):
                continue

            try:
                conn = sqlite3.connect(f"file:{source['path']}?mode=ro", uri=True)
                cursor = conn.execute("SELECT MAX(zoom_level) FROM tiles")
                row = cursor.fetchone()
                if not row or row[0] is None:
                    conn.close()
                    continue

                max_zoom = row[0]
                cursor = conn.execute(
                    "SELECT zoom_level, tile_column, tile_row, tile_data "
                    "FROM tiles WHERE zoom_level = ?", (max_zoom,)
                )

                for z, x, y_tms, data in cursor:
                    try:
                        tile_bytes = data
                        if tile_bytes.startswith(b'\x1f\x8b'):
                            tile_bytes = gzip.decompress(tile_bytes)

                        decoded = mapbox_vector_tile.decode(
                            tile_bytes, default_options={'y_coord_down': True}
                        )
                        y_xyz = (1 << z) - 1 - y_tms
                        n = 2.0 ** z

                        for layer_name, layer_data in decoded.items():
                            extent = layer_data.get('extent', 4096)
                            for feature in layer_data.get('features', []):
                                name = feature.get('properties', {}).get('name')
                                if not name:
                                    continue

                                geom = feature.get('geometry', {})
                                coords = geom.get('coordinates', [])

                                if not coords:
                                    continue

                                # 提取坐标
                                geom_type = geom.get('type', '')
                                if geom_type == 'Point':
                                    px, py = coords
                                elif geom_type in ('LineString', 'MultiPoint'):
                                    px, py = coords[0]
                                elif geom_type in ('Polygon', 'MultiLineString'):
                                    px, py = coords[0][0]
                                elif geom_type == 'MultiPolygon':
                                    px, py = coords[0][0][0]
                                else:
                                    continue

                                lon = (x + px / extent) / n * 360.0 - 180.0
                                y_frac = (y_xyz + py / extent) / n
                                lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_frac)))
                                lat = math.degrees(lat_rad)

                                name_lower = name.lower()
                                if name_lower not in self._index:
                                    self._index[name_lower] = []
                                self._index[name_lower].append({
                                    'name': name,
                                    'lat': lat,
                                    'lon': lon,
                                    'layer': layer_name,
                                    'source': source['name'],
                                })
                                count += 1
                    except Exception:
                        continue

                conn.close()
            except Exception as e:
                print(f"  索引构建跳过数据源 {source['name']}: {e}")

        elapsed = time.time() - start
        with self._lock:
            self._built = True
        print(f"搜索索引构建完成: {count} 条记录, {len(self._index)} 个唯一名称, 耗时 {elapsed:.2f}s")

    def search(self, query, limit=20):
        """查询搜索索引"""
        if not self._built:
            return []
        query_lower = query.lower()
        results = self._index.get(query_lower, [])

        # 部分匹配：如果精确匹配无结果，遍历索引做子串匹配
        if not results:
            for name_lower, entries in self._index.items():
                if len(results) >= limit:
                    break
                if query_lower in name_lower:
                    results.extend(entries[:limit - len(results)])

        # 去重
        seen = set()
        unique = []
        for r in results:
            key = (r['name'], round(r['lat'], 6), round(r['lon'], 6))
            if key not in seen:
                seen.add(key)
                unique.append(r)
                if len(unique) >= limit:
                    break

        return unique


# 全局搜索索引实例
_search_index = SearchIndex()


# ═══════════════════════════════════════════════════════════════════
#  性能日志（缓冲批量写入）
# ═══════════════════════════════════════════════════════════════════
class BufferedLogger:
    """缓冲批量写入的性能日志"""

    def __init__(self, filename="performance.log", flush_interval=LOG_FLUSH_INTERVAL,
                 flush_batch=LOG_FLUSH_BATCH):
        self._filename = filename
        self._flush_interval = flush_interval
        self._flush_batch = flush_batch
        self._buffer = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._running = True
        # 后台刷新线程
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def log(self, line):
        with self._lock:
            self._buffer.append(line)
            should_flush = (
                len(self._buffer) >= self._flush_batch or
                (time.time() - self._last_flush) >= self._flush_interval
            )
            if should_flush:
                self._flush_internal()

    def _flush_internal(self):
        """在锁内调用"""
        if not self._buffer:
            return
        try:
            with open(self._filename, 'a', encoding='utf-8') as f:
                f.write('\n'.join(self._buffer) + '\n')
            self._buffer.clear()
            self._last_flush = time.time()
        except Exception as e:
            print(f"日志写入失败: {e}")

    def _flush_loop(self):
        while self._running:
            time.sleep(self._flush_interval)
            with self._lock:
                self._flush_internal()

    def flush(self):
        with self._lock:
            self._flush_internal()

    def stop(self):
        self._running = False
        self.flush()


_perf_logger = BufferedLogger()

# ═══════════════════════════════════════════════════════════════════
#  mbtiles 文件搜索索引（用于 --search-dir）
# ═══════════════════════════════════════════════════════════════════
_mbtiles_index = {}  # basename → 完整路径 的映射

def build_mbtiles_index(search_dir):
    """递归扫描 search_dir，建立 basename → 完整路径 的映射表。

    用于 --search-dir 功能：当 tile_index.db 中 sources 表存储的是
    原始绝对路径（如 Android 设备路径），而实际 mbtiles 文件位于
    另一个目录（可能含子目录）时，通过 basename 匹配找到正确的文件路径。
    """
    global _mbtiles_index
    _mbtiles_index.clear()
    search_dir = os.path.abspath(search_dir)
    if not os.path.isdir(search_dir):
        print(f"警告: 搜索目录不存在: {search_dir}")
        return
    count = 0
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.mbtiles'):
                full = os.path.join(root, f)
                _mbtiles_index[f] = full
                count += 1
    print(f"mbtiles 搜索索引: 在 {search_dir} 下找到 {count} 个 .mbtiles 文件")

# ═══════════════════════════════════════════════════════════════════
#  原有工具函数
# ═══════════════════════════════════════════════════════════════════
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
    """加载 tile_index.db 索引。"""
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
        if SEARCH_MBTILES_DIR and _mbtiles_index:
            basename = os.path.basename(path)
            if basename in _mbtiles_index:
                path = _mbtiles_index[basename]
            else:
                print(f"警告: 在搜索目录中未找到 {basename}，使用原始路径")
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


def get_source_for_tile(z, x, y_tms):
    source_id = TILE_INDEX.get((z, x, y_tms), FALLBACK_SOURCE_ID)
    return SOURCES.get(source_id, SOURCES[FALLBACK_SOURCE_ID])


def query_tile(source, z, x, y_tms):
    """查询瓦片数据（带缓存）"""
    # 先查缓存
    cache_key = (source['id'], z, x, y_tms)
    cached = _tile_cache.get(cache_key)
    if cached is not None:
        return cached

    # 从数据库查询
    pool = get_connection_pool(source)
    conn = pool.acquire()
    if conn is None:
        return None

    try:
        cursor = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (z, x, y_tms)
        )
        row = cursor.fetchone()
        tile_data = row[0] if row else None

        # 放入缓存
        if tile_data is not None:
            _tile_cache.put(cache_key, tile_data)

        return tile_data
    except Exception:
        return None
    finally:
        pool.release(conn)


# ═══════════════════════════════════════════════════════════════════
#  HTTP 请求处理
# ═══════════════════════════════════════════════════════════════════
class MBTilesHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # 线程限制
        acquired = _thread_semaphore.acquire(blocking=True, timeout=30)
        if not acquired:
            self.send_response(503)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Service Unavailable: too many concurrent requests')
            return

        try:
            self._handle_get()
        finally:
            _thread_semaphore.release()

    def _handle_get(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'MBTiles Server is running (optimized)')
            return

        # 健康检查端点
        if path == '/health':
            self._send_json({
                'status': 'ok',
                'cache': _tile_cache.stats(),
                'sources': len(SOURCES),
                'tiles_indexed': len(TILE_INDEX),
            })
            return

        # 搜索端点（使用预构建索引）
        if path == '/search':
            q = query_params.get('q', [''])[0].strip()
            if not q:
                self._send_json([])
                return
            results = _search_index.search(q, limit=20)
            self._send_json(results)
            return

        # 字体端点（带缓存）
        if path.startswith('/fonts/'):
            parts = path.split('/')
            if len(parts) >= 4:
                fontstack = urllib.parse.unquote(parts[2])
                range_pbf = parts[3]
                font_path = os.path.normpath(os.path.join(FONTS_DIR, fontstack, range_pbf))

                # 安全检查
                if not font_path.startswith(FONTS_DIR + os.sep):
                    self.send_response(403)
                    self.end_headers()
                    return

                # 查缓存
                cached = _font_cache.get(font_path)
                if cached is not None:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-protobuf')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', len(cached))
                    self.end_headers()
                    self.wfile.write(cached)
                    return

                # 从磁盘读取并缓存
                if os.path.isfile(font_path):
                    with open(font_path, 'rb') as f:
                        content = f.read()
                    _font_cache.put(font_path, content)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-protobuf')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                    return

            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return

        # 精灵端点（带缓存）
        if path.startswith('/sprites/'):
            rel = urllib.parse.unquote(path[len('/sprites/'):])
            if '..' not in rel and not rel.startswith(('/', '\\')):
                sprite_path = os.path.normpath(os.path.join(SPRITES_DIR, rel))

                # 安全检查
                if not sprite_path.startswith(SPRITES_DIR + os.sep):
                    self.send_response(403)
                    self.end_headers()
                    return

                # 查缓存
                cached = _sprite_cache.get(sprite_path)
                if cached is not None:
                    if sprite_path.endswith('.json'):
                        ctype = 'application/json'
                    elif sprite_path.endswith('.png'):
                        ctype = 'image/png'
                    else:
                        ctype = 'application/octet-stream'
                    self.send_response(200)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', len(cached))
                    self.end_headers()
                    self.wfile.write(cached)
                    return

                # 从磁盘读取并缓存
                if os.path.isfile(sprite_path):
                    with open(sprite_path, 'rb') as f:
                        content = f.read()
                    _sprite_cache.put(sprite_path, content)
                    if sprite_path.endswith('.json'):
                        ctype = 'application/json'
                    elif sprite_path.endswith('.png'):
                        ctype = 'image/png'
                    else:
                        ctype = 'application/octet-stream'
                    self.send_response(200)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                    return

            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return

        # 瓦片端点
        if not path.startswith('/tiles/') and not path.startswith('/data/'):
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
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
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'Bad Request')
            return

        try:
            source = get_source_for_tile(z, x, y_tms)

            # 数据源有效性检查（启动时已验证，运行时只检查 path 是否非空）
            if not source.get('path'):
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return

            start_time = time.perf_counter()
            tile_blob = query_tile(source, z, x, y_tms)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            tile_size = len(tile_blob) if tile_blob else 0

            # 异步记录性能日志
            _perf_logger.log(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"z={z} x={x} y={y_xyz} size={tile_size} "
                f"elapsed={elapsed_ms:.3f}ms file={source['path']}"
            )

            if tile_blob:
                self._send_tile(tile_blob, source)
            else:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'Tile not found')
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _send_tile(self, tile_blob, source):
        content_type = CONTENT_TYPES.get(source.get('format'), 'application/octet-stream')
        is_gzip = False
        try:
            is_gzip = (
                isinstance(tile_blob, (bytes, bytearray)) and
                len(tile_blob) >= 2 and
                tile_blob[0] == 0x1f and tile_blob[1] == 0x8b
            )
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


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 解析命令行参数
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ('-h', '--help'):
            print(USAGE_TEXT)
            sys.exit(0)
        elif arg == '--search-dir':
            if i + 1 < len(argv):
                SEARCH_MBTILES_DIR = argv[i + 1]
                i += 2
                continue
        elif arg.startswith('--search-dir='):
            SEARCH_MBTILES_DIR = arg[len('--search-dir='):]
        i += 1

    # 如果指定了搜索目录，先构建 mbtiles 文件索引
    if SEARCH_MBTILES_DIR:
        build_mbtiles_index(SEARCH_MBTILES_DIR)

    # 加载数据源和索引
    SOURCES, TILE_INDEX = load_index(INDEX_DB_FILE)

    print(f"MBTiles 瓦片服务启动（优化版）: http://localhost:{PORT}/tiles/{{z}}/{{x}}/{{y}}")
    print(f"索引数据库: {INDEX_DB_FILE}")
    print(f"兜底数据源: {FALLBACK_PATH}")
    if SEARCH_MBTILES_DIR:
        print(f"mbtiles 搜索目录: {SEARCH_MBTILES_DIR}")
    print(f"索引瓦片数: {len(TILE_INDEX)}")
    print(f"最大线程数: {MAX_THREADS}")
    print(f"瓦片缓存大小: {TILE_CACHE_SIZE}")
    print(f"连接池大小: {DB_POOL_SIZE}")
    print("数据源:")
    for source in sorted(SOURCES.values(), key=lambda item: item['priority']):
        print(f"  - {source['name']}: {source['path']}  格式: {source.get('format') or '未知'}")

    # 构建搜索索引（启动时预计算）
    # _search_index.build(SOURCES)

    # 启动 HTTP 服务器
    server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), MBTilesHandler)
    server.daemon_threads = True

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        _perf_logger.stop()
        # 关闭所有连接池
        for pool in _connection_pools.values():
            pool.close_all()
        server.server_close()
        print("服务已停止")
