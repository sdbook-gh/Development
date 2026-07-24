#!/usr/bin/env python3
"""
扫描目录下的省级 mbtiles，构建 overview.mbtiles + boundary.mbtiles + tile_index.db。

- overview: 用 tippecanoe 的 tile-join 全量合并 z<=--overview-max-zoom，
  低 zoom 渲染只打开这个小文件，避免扫多个省级大 mbtiles。
- boundary: 通过 libtile-join-ext.so (ctypes) 的 --boundary-only，
  只输出在 >=2 个输入源里出现的瓦片 (z>--overview-max-zoom)。

用法:
  python3 build_tile_index_overview.py --input-dir <dir> [--output-dir <dir>] \\
          [--workers N] [--lib-path <path>]

输出:
  <output-dir>/tile_index.db    sources / tile_sources / tile_index
  <output-dir>/overview.mbtiles  低 zoom 全量合并库 (z<=--overview-max-zoom, 加速低 zoom 渲染)
  <output-dir>/boundary.mbtiles 边界瓦片合并库 (z>--overview-max-zoom)

并发:
  --workers N 控制 Phase 3 boundary (tile-join-ext) 的并发:
    N=1: 单次库调用，库内部用全部 CPU 线程 (默认, I/O 无冗余)
    N>1: zoom 范围按边界瓦片数均分成 N 段, N 个 subprocess 各调一次库,
         每个设 TIPPECANOE_MAX_THREADS = cpu_count // N, 最后 SQL 合并分片.
         代价: N 倍输入 I/O (依赖 OS page cache 抵消).
         对单 zoom 占比过半的数据集 (如本例 z14 占 63%), 收益上限约 2x.
  Phase 2 (扫描省级 mbtiles 建 tile_sources) 始终并发, 与 --workers 无关,
  因为各 worker 读不同文件, 零 I/O 冗余.
"""
import argparse
import ctypes
import os
import sqlite3
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

DEFAULT_LIB_PATH = "/mnt/e/deve/terminal/map/tools/tippecanoe-2.80.0/libtile-join-ext.so"
MERGED_SOURCE_NAME = "merged"
MERGED_SOURCE_FILENAME = "boundary.mbtiles"

# overview: 用 tile-join 全量合并低 zoom 到精简的 overview.mbtiles,
# 低 zoom 渲染时 maplibre 只加载这个小文件, 避免打开大 mbtiles 加速渲染.
OVERVIEW_SOURCE_NAME = "overview"
OVERVIEW_SOURCE_FILENAME = "overview.mbtiles"
DEFAULT_OVERVIEW_MAX_ZOOM = 6

SCHEMA_SQL = """
CREATE TABLE sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    path        TEXT    NOT NULL,
    priority    INTEGER NOT NULL,
    tile_count  INTEGER DEFAULT 0,
    min_zoom    INTEGER,
    max_zoom    INTEGER,
    is_merged   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE tile_index (
    zoom        INTEGER NOT NULL,
    tile_column INTEGER NOT NULL,
    tile_row    INTEGER NOT NULL,
    source_id   INTEGER NOT NULL,
    is_merged   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (zoom, tile_column, tile_row)
) WITHOUT ROWID;
CREATE TABLE tile_sources (
    zoom        INTEGER NOT NULL,
    tile_column INTEGER NOT NULL,
    tile_row    INTEGER NOT NULL,
    source_id   INTEGER NOT NULL,
    PRIMARY KEY (zoom, tile_column, tile_row, source_id)
) WITHOUT ROWID;
"""

FLAT_MBTILES_SCHEMA = """
CREATE TABLE metadata (name text, value text);
CREATE UNIQUE INDEX name ON metadata (name);
CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);
CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);
"""


# ============== libtile-join-ext.so ctypes 绑定 ==============
# 全局变量 (boundary_only/pk/pC 等) 是进程级的, 不能同进程并发;
# 多 worker 必须用 multiprocessing (子进程), 不要用 threading.

_lib_cache = {}


def get_lib(lib_path):
    if lib_path in _lib_cache:
        return _lib_cache[lib_path]
    if not os.path.isfile(lib_path):
        sys.exit(f"找不到 libtile-join-ext.so: {lib_path}")
    try:
        # 必须 RTLD_GLOBAL: .so 链接时没带 -lstdc++ (Makefile 只链 -lsqlite3 -lz),
        # C++ 运行时符号 (operator new 等) 要从宿主进程的全局符号表解析,
        # 默认 RTLD_LOCAL 下会 segfault。
        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        sys.exit(f"加载 {lib_path} 失败: {e}\n  可能缺少 libsqlite3.so / libz.so / libstdc++.so 等依赖")
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
    _lib_cache[lib_path] = lib
    return lib


def call_tile_join_ext(lib_path, input_paths, output_path,
                       min_zoom=None, max_zoom=None):
    """boundary_only 合并 (仅输出 >=2 源重叠瓦片). 仅用于 boundary.mbtiles."""
    lib = get_lib(lib_path)
    ctx = lib.tile_join_ext_create()
    if not ctx:
        raise RuntimeError("tile_join_ext_create() returned NULL")
    try:
        lib.tile_join_ext_set_option(ctx, b"force", b"1")
        lib.tile_join_ext_set_option(ctx, b"no_tile_size_limit", b"1")
        lib.tile_join_ext_set_option(ctx, b"boundary_only", b"1")
        # minzoom/maxzoom 在 .so 里是进程级全局变量 (tile-join-ext.cpp: int minzoom=0; int maxzoom=32;),
        # 同进程多次调用会互相污染. 这里每次都显式写入, None 时用 .so 默认值复位.
        lib.tile_join_ext_set_option(ctx, b"minzoom", str(min_zoom if min_zoom is not None else 0).encode())
        lib.tile_join_ext_set_option(ctx, b"maxzoom", str(max_zoom if max_zoom is not None else 32).encode())
        for p in input_paths:
            if lib.tile_join_ext_add_input(ctx, p.encode()) != 0:
                raise RuntimeError(f"tile_join_ext_add_input 失败: {p}")
        if lib.tile_join_ext_set_output(ctx, output_path.encode()) != 0:
            raise RuntimeError(f"tile_join_ext_set_output 失败: {output_path}")
        rc = lib.tile_join_ext_run(ctx)
        if rc != 0:
            raise RuntimeError(f"tile_join_ext_run 失败, code={rc}")
    finally:
        lib.tile_join_ext_destroy(ctx)
    sys.stdout.flush()


def call_tile_join(input_paths, output_path, min_zoom, max_zoom):
    """标准 tippecanoe tile-join: 全量合并指定 zoom 范围 (用于 overview.mbtiles)."""
    cmd = [
        "tile-join",
        "--force",
        "--no-tile-size-limit",
        "-Z", str(min_zoom),
        "-z", str(max_zoom),
        "-o", output_path,
    ]
    cmd += list(input_paths)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[-500:]
        raise RuntimeError(f"tile-join 失败, code={r.returncode}: {err}")
    sys.stdout.flush()


# ============== multiprocessing worker (必须是顶层函数) ==============

def scan_source_worker(args):
    """子进程: 扫描一个省级 mbtiles 的 tiles 表, 返回 (z,x,y) 列表。"""
    path, = args
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT zoom_level, tile_column, tile_row FROM tiles"
        ).fetchall()
    finally:
        conn.close()
    return rows


def tile_join_ext_worker(args):
    """子进程: 调用 libtile-join-ext.so 处理一个 zoom 段。"""
    lib_path, input_paths, output_path, min_zoom, max_zoom, max_threads = args
    os.environ["TIPPECANOE_MAX_THREADS"] = str(max_threads)
    t0 = time.time()
    call_tile_join_ext(lib_path, input_paths, output_path, min_zoom, max_zoom)
    return (output_path, time.time() - t0)


# ============== phases ==============

def log(phase, msg):
    elapsed = time.time() - log.start if hasattr(log, "start") else 0
    print(f"[{phase}] {msg}" + (f"  (累计 {elapsed:.1f}s)" if elapsed else ""))


def phase1_scan(input_dir):
    log("Phase 1", f"扫描 {input_dir} 下的 *.mbtiles")
    exclude = {MERGED_SOURCE_FILENAME, OVERVIEW_SOURCE_FILENAME}
    files = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".mbtiles"):
            continue
        if name in exclude or name.startswith("boundary.part."):
            continue
        files.append(os.path.join(input_dir, name))
    if not files:
        sys.exit(f"{input_dir} 下没有可用的 .mbtiles 文件")
    log("Phase 1", f"找到 {len(files)} 个省级 mbtiles")
    for f in files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... (共 {len(files)} 个)")
    return files


def phase2_build_index(input_files, output_dir):
    """从零构建 tile_index.db。扫描阶段并发, 与 --workers 无关。"""
    log("Phase 2", "构建 tile_index.db (扫描阶段并发)")
    t0 = time.time()
    db_path = os.path.join(output_dir, "tile_index.db")
    for p in (db_path, db_path + "-wal", db_path + "-shm", db_path + "-journal"):
        if os.path.exists(p):
            os.remove(p)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    for i, path in enumerate(input_files, 1):
        name = os.path.basename(path)[:-len(".mbtiles")]
        try:
            rel = os.path.relpath(path, output_dir)
        except ValueError:
            rel = path
        conn.execute(
            "INSERT INTO sources (id, name, path, priority, is_merged) VALUES (?, ?, ?, ?, 0)",
            (i, name, rel, i),
        )
    conn.commit()

    # 并发扫描: 每个 worker 读不同文件, 零 I/O 冗余
    n_workers = min(cpu_count(), len(input_files))
    with Pool(n_workers) as pool:
        results = pool.map(scan_source_worker, [(p,) for p in input_files])

    for source_id, tiles in enumerate(results, 1):
        if not tiles:
            continue
        conn.executemany(
            "INSERT OR IGNORE INTO tile_sources (zoom, tile_column, tile_row, source_id) VALUES (?, ?, ?, ?)",
            [(z, x, y, source_id) for (z, x, y) in tiles],
        )
    conn.commit()

    # tile_index: 每个 (z,x,y) 取 priority 最小的 source (window function)
    conn.executescript("""
        INSERT INTO tile_index (zoom, tile_column, tile_row, source_id, is_merged)
        SELECT zoom, tile_column, tile_row, source_id, 0
        FROM (
            SELECT ts.zoom, ts.tile_column, ts.tile_row, ts.source_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY ts.zoom, ts.tile_column, ts.tile_row
                       ORDER BY s.priority ASC
                   ) AS rn
            FROM tile_sources ts
            JOIN sources s ON s.id = ts.source_id
        )
        WHERE rn = 1;
        UPDATE sources SET
            tile_count = (SELECT COUNT(*) FROM tile_sources ts WHERE ts.source_id = sources.id),
            min_zoom   = (SELECT MIN(zoom) FROM tile_sources ts WHERE ts.source_id = sources.id),
            max_zoom   = (SELECT MAX(zoom) FROM tile_sources ts WHERE ts.source_id = sources.id);
    """)
    conn.commit()

    ts = conn.execute("SELECT COUNT(*) FROM tile_sources").fetchone()[0]
    ti = conn.execute("SELECT COUNT(*) FROM tile_index").fetchone()[0]
    bdy = conn.execute(
        "SELECT COUNT(*) FROM (SELECT zoom, tile_column, tile_row FROM tile_sources "
        "GROUP BY zoom, tile_column, tile_row HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    conn.close()

    log("Phase 2", f"tile_sources={ts}, tile_index={ti}, boundary={bdy}  耗时 {time.time()-t0:.1f}s")
    return db_path


def partition_zooms_by_boundary(db_path, n_bins, min_zoom=None):
    """把 zoom 范围按边界瓦片数均分成 n_bins 个连续段。

    min_zoom: 只考虑 zoom >= min_zoom 的边界瓦片 (用于把低 zoom 拆给 overview 后,
    boundary 只分高 zoom 段)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    if min_zoom is not None:
        zoom_counts = conn.execute(
            "SELECT zoom, COUNT(*) AS cnt FROM ("
            "  SELECT zoom, tile_column, tile_row FROM tile_sources "
            "  GROUP BY zoom, tile_column, tile_row HAVING COUNT(*) > 1"
            ") WHERE zoom >= ? GROUP BY zoom ORDER BY zoom",
            (min_zoom,),
        ).fetchall()
    else:
        zoom_counts = conn.execute(
            "SELECT zoom, COUNT(*) AS cnt FROM ("
            "  SELECT zoom, tile_column, tile_row FROM tile_sources "
            "  GROUP BY zoom, tile_column, tile_row HAVING COUNT(*) > 1"
            ") GROUP BY zoom ORDER BY zoom"
        ).fetchall()
    conn.close()
    if not zoom_counts:
        return []
    if n_bins <= 1 or len(zoom_counts) == 1:
        return [(zoom_counts[0][0], zoom_counts[-1][0])]

    total = sum(c for _, c in zoom_counts)
    target = total / n_bins
    bins = []
    cur_min = zoom_counts[0][0]
    cur_max = zoom_counts[0][0]
    acc = 0
    for i, (zoom, count) in enumerate(zoom_counts):
        cur_max = zoom
        acc += count
        remaining_zooms = len(zoom_counts) - i - 1
        bins_needed = n_bins - len(bins) - 1
        if acc >= target and bins_needed > 0 and remaining_zooms >= bins_needed:
            bins.append((cur_min, cur_max))
            cur_min = zoom + 1
            acc = 0
    if cur_min <= cur_max:
        bins.append((cur_min, cur_max))
    return bins


def _write_empty_mbtiles(output_path, metadata_src_path):
    """写一个仅含 metadata 的空 mbtiles (扁平 schema), 供无边界瓦片时占位."""
    conn = sqlite3.connect(output_path)
    conn.executescript(FLAT_MBTILES_SCHEMA)
    try:
        src = sqlite3.connect(f"file:{metadata_src_path}?mode=ro", uri=True)
        for row in src.execute("SELECT name, value FROM metadata"):
            conn.execute("INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)", row)
        src.close()
    except sqlite3.Error:
        # 源文件无 metadata 时给个默认 format, 保证服务端 load_source_format 可识别
        conn.execute("INSERT OR REPLACE INTO metadata (name, value) VALUES ('format', 'pbf')")
    conn.commit()
    conn.close()


def phase3_build_overview(input_files, output_dir, overview_max_zoom):
    """构建 overview.mbtiles: 用 tile-join 全量合并 z <= overview_max_zoom.

    把各省低 zoom 瓦片抽到独立小文件, 低 zoom 渲染时 maplibre 只读 overview.mbtiles,
    无需打开省级大文件, 显著减少 I/O 加速首屏. 不用 boundary_only (那是 boundary 的职责).
    """
    log("Phase 3a", f"构建 overview.mbtiles (tile-join, z<={overview_max_zoom})")
    t0 = time.time()
    overview_path = os.path.join(output_dir, OVERVIEW_SOURCE_FILENAME)
    for p in (overview_path, overview_path + "-wal", overview_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)

    call_tile_join(input_files, overview_path, min_zoom=0, max_zoom=overview_max_zoom)

    if not os.path.exists(overview_path):
        # tile-join 在无满足条件瓦片时可能不输出文件
        _write_empty_mbtiles(overview_path, input_files[0])

    conn = sqlite3.connect(f"file:{overview_path}?mode=ro", uri=True)
    count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    zrange = conn.execute(
        "SELECT MIN(zoom_level), MAX(zoom_level) FROM tiles"
    ).fetchone()
    conn.close()
    zinfo = f"z={zrange[0]}~{zrange[1]}" if zrange[0] is not None else "空"
    log("Phase 3a", f"完成, overview 瓦片数={count} ({zinfo})  耗时 {time.time()-t0:.1f}s")
    return overview_path


def phase3_build_boundary(input_files, output_dir, workers, lib_path, db_path,
                          overview_max_zoom):
    """构建 boundary.mbtiles, 只含 z > overview_max_zoom 的边界瓦片。

    低 zoom 部分由 phase3_build_overview (tile-join 全量) 产出到 overview.mbtiles,
    避免 boundary.mbtiles 携带低 zoom 冗余瓦片, 让低 zoom 渲染只读小文件."""
    log("Phase 3", f"构建 boundary.mbtiles (z>{overview_max_zoom}, workers={workers})")
    t0 = time.time()
    boundary_path = os.path.join(output_dir, MERGED_SOURCE_FILENAME)
    for p in (boundary_path, boundary_path + "-wal", boundary_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    for name in os.listdir(output_dir):
        if name.startswith("boundary.part."):
            os.remove(os.path.join(output_dir, name))

    boundary_min_zoom = overview_max_zoom + 1

    if workers <= 1:
        call_tile_join_ext(lib_path, input_files, boundary_path,
                           min_zoom=boundary_min_zoom)
        log("Phase 3", f"完成 (单 worker, 库内部多线程) 耗时 {time.time()-t0:.1f}s")
        return boundary_path

    bins = partition_zooms_by_boundary(db_path, workers, min_zoom=boundary_min_zoom)
    if not bins:
        # 没有高 zoom 边界瓦片: 生成空 boundary.mbtiles (仅 metadata, 供 phase4/5 统一处理)
        print(f"  z>{overview_max_zoom} 无边界瓦片, 生成空 boundary.mbtiles")
        _write_empty_mbtiles(boundary_path, input_files[0])
        log("Phase 3", f"完成 (空 boundary) 耗时 {time.time()-t0:.1f}s")
        return boundary_path
    if len(bins) <= 1:
        print(f"  zoom 范围无法拆成 {workers} 段 (边界瓦片集中度过高), 回退单 worker")
        call_tile_join_ext(lib_path, input_files, boundary_path,
                           min_zoom=boundary_min_zoom)
        log("Phase 3", f"完成 (回退单 worker) 耗时 {time.time()-t0:.1f}s")
        return boundary_path

    threads_per_worker = max(1, cpu_count() // len(bins))
    print(f"  zoom 拆分: {bins}, 每段内部线程 = {threads_per_worker}")

    partials = []
    worker_args = []
    for i, (min_z, max_z) in enumerate(bins):
        p = os.path.join(output_dir, f"boundary.part.{i}.mbtiles")
        partials.append(p)
        worker_args.append((lib_path, input_files, p, min_z, max_z, threads_per_worker))

    with Pool(len(bins)) as pool:
        results = pool.map(tile_join_ext_worker, worker_args)
    for p, elapsed in results:
        print(f"  worker {os.path.basename(p)}: {elapsed:.1f}s")

    # SQL 合并分片 → boundary.mbtiles (扁平 schema, 服务端 query_tile 直接查 tiles 表)
    log("Phase 3", f"合并 {len(partials)} 个分片 → {boundary_path}")
    conn = sqlite3.connect(boundary_path)
    conn.executescript(FLAT_MBTILES_SCHEMA)
    # metadata 从第一个分片拷 (format=pbf 等)
    conn.execute("ATTACH DATABASE ? AS p0", (partials[0],))
    conn.execute("INSERT INTO metadata SELECT * FROM p0.metadata")
    conn.execute("DETACH DATABASE p0")
    for i, p in enumerate(partials):
        alias = f"p{i}"
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (p,))
        conn.execute(f"""
            INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data)
            SELECT zoom_level, tile_column, tile_row, tile_data FROM {alias}.tiles
        """)
        conn.execute(f"DETACH DATABASE {alias}")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    conn.close()

    for p in partials:
        for suffix in ("", "-wal", "-shm"):
            pp = p + suffix
            if os.path.exists(pp):
                os.remove(pp)

    log("Phase 3", f"完成, boundary 瓦片数={count}  耗时 {time.time()-t0:.1f}s")
    return boundary_path


def _register_merged_source(conn, name, filename, tile_count, min_zoom, max_zoom):
    conn.execute(
        """
        INSERT OR REPLACE INTO sources
            (name, path, priority, tile_count, min_zoom, max_zoom, is_merged)
        VALUES (?, ?, 0, ?, ?, ?, 1)
        """,
        (name, filename, tile_count, min_zoom, max_zoom),
    )
    conn.commit()
    return conn.execute("SELECT id FROM sources WHERE name = ?", (name,)).fetchone()[0]


def _mbtiles_stats(path):
    """返回 (tile_count, min_zoom, max_zoom, format)."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    zrange = conn.execute(
        "SELECT MIN(zoom_level), MAX(zoom_level) FROM tiles"
    ).fetchone()
    fmt = conn.execute("SELECT value FROM metadata WHERE name='format'").fetchone()
    conn.close()
    return count, (zrange[0] if zrange[0] is not None else 0), (zrange[1] if zrange[1] is not None else 0), (fmt[0] if fmt else None)


def _mbtiles_valid(path):
    """mbtiles 文件存在且 tiles 表可读 (不要求非空, 空文件也算结构有效)."""
    if not os.path.exists(path):
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM tiles LIMIT 1").fetchone()
        conn.close()
        return True
    except sqlite3.Error:
        return False


def _index_db_ready(db_path):
    """tile_index.db 已建好 (含 tile_sources / tile_index 行), 可供 phase4 直接用."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        ts = conn.execute("SELECT COUNT(*) FROM tile_sources").fetchone()[0]
        ti = conn.execute("SELECT COUNT(*) FROM tile_index").fetchone()[0]
        conn.close()
        return ts > 0 and ti > 0
    except sqlite3.Error:
        return False


def phase4_update_index(db_path, boundary_path, overview_path):
    """注册 overview + boundary 两个 merged 源, 并按各自瓦片集重定向 tile_index.

    低 zoom (z<=overview_max_zoom) 全量瓦片 -> overview 源 (小文件, 加速低 zoom 渲染)
    高 zoom 边界瓦片 -> boundary 源
    用 mbtiles 文件自身作为重定向基准, 保证 文件 = 索引 严格一致."""
    log("Phase 4", "注册 overview/boundary 源并重定向 tile_index")
    t0 = time.time()

    ov_count, ov_min, ov_max, _ = _mbtiles_stats(overview_path)
    bdy_count, bdy_min, bdy_max, _ = _mbtiles_stats(boundary_path)

    if ov_count == 0 and bdy_count == 0:
        print("  overview 与 boundary 均为空 (输入源 < 2 个?), 跳过索引重定向")
        log("Phase 4", f"跳过  耗时 {time.time()-t0:.1f}s")
        return None, None

    conn = sqlite3.connect(db_path)

    overview_id = None
    if ov_count > 0:
        overview_id = _register_merged_source(
            conn, OVERVIEW_SOURCE_NAME, OVERVIEW_SOURCE_FILENAME,
            ov_count, ov_min, ov_max,
        )

    merged_id = None
    if bdy_count > 0:
        merged_id = _register_merged_source(
            conn, MERGED_SOURCE_NAME, MERGED_SOURCE_FILENAME,
            bdy_count, bdy_min, bdy_max,
        )

    # 用各自 mbtiles 自身作为重定向基准, 文件 = 索引严格一致.
    # 注意: Python sqlite3 在 DML 后会持有未提交事务, DETACH 不能在活动事务中执行,
    # 所以必须先 ATTACH + UPDATE 全部完成, 再 commit, 最后 DETACH.
    attached = []
    if overview_id is not None:
        conn.execute("ATTACH DATABASE ? AS ov", (overview_path,))
        attached.append("ov")
        conn.execute(
            """
            UPDATE tile_index
            SET source_id = ?, is_merged = 1
            WHERE (zoom, tile_column, tile_row) IN (
                SELECT zoom_level, tile_column, tile_row FROM ov.tiles
            )
            """,
            (overview_id,),
        )

    if merged_id is not None:
        conn.execute("ATTACH DATABASE ? AS bdy", (boundary_path,))
        attached.append("bdy")
        conn.execute(
            """
            UPDATE tile_index
            SET source_id = ?, is_merged = 1
            WHERE (zoom, tile_column, tile_row) IN (
                SELECT zoom_level, tile_column, tile_row FROM bdy.tiles
            )
            """,
            (merged_id,),
        )

    conn.commit()
    for alias in attached:
        conn.execute(f"DETACH DATABASE {alias}")

    redirected = conn.execute(
        "SELECT COUNT(*) FROM tile_index WHERE is_merged = 1"
    ).fetchone()[0]
    conn.close()

    log("Phase 4",
        f"overview source_id={overview_id} tiles={ov_count}; "
        f"boundary source_id={merged_id} tiles={bdy_count}; "
        f"redirected={redirected}  耗时 {time.time()-t0:.1f}s")
    expected = ov_count + bdy_count
    if redirected != expected:
        sys.exit(f"重定向行数 ({redirected}) 与 overview+boundary 瓦片数 ({expected}) 不一致")
    return merged_id, overview_id


def phase5_verify(db_path, boundary_path, overview_path, merged_id, overview_id):
    log("Phase 5", "校验")
    b_count, _, _, b_fmt = _mbtiles_stats(boundary_path)
    ov_count, _, _, ov_fmt = _mbtiles_stats(overview_path)

    idx_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    redirected = idx_conn.execute(
        "SELECT COUNT(*) FROM tile_index WHERE is_merged = 1"
    ).fetchone()[0]
    src_count = idx_conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]

    def _src_row(sid):
        if sid is None:
            return None
        return idx_conn.execute(
            "SELECT id, name, path, is_merged FROM sources WHERE id = ?", (sid,)
        ).fetchone()

    merged_src = _src_row(merged_id)
    overview_src = _src_row(overview_id)
    idx_conn.close()

    print(f"  overview.mbtiles  瓦片数 : {ov_count}")
    print(f"  overview.mbtiles  format : {ov_fmt or '缺失'}")
    print(f"  boundary.mbtiles  瓦片数 : {b_count}")
    print(f"  boundary.mbtiles  format : {b_fmt or '缺失'}")
    print(f"  tile_index is_merged=1   : {redirected}")
    print(f"  sources 总数             : {src_count}")
    if overview_src:
        print(f"  overview source          : id={overview_src[0]} name={overview_src[1]} path={overview_src[2]} is_merged={overview_src[3]}")
    if merged_src:
        print(f"  merged source            : id={merged_src[0]} name={merged_src[1]} path={merged_src[2]} is_merged={merged_src[3]}")

    ok = (redirected == ov_count + b_count)
    if overview_id is not None and (not ov_fmt or ov_fmt != 'pbf'):
        print("  警告: overview.mbtiles format 不是 pbf, 服务端 load_source_format 会识别失败")
        ok = False
    if merged_id is not None and (not b_fmt or b_fmt != 'pbf'):
        print("  警告: boundary.mbtiles format 不是 pbf, 服务端 load_source_format 会识别失败")
        ok = False
    if not ok:
        sys.exit("校验失败")
    log("Phase 5", "通过")


def main():
    parser = argparse.ArgumentParser(
        description="扫描目录下省级 mbtiles, 构建 overview.mbtiles + boundary.mbtiles + tile_index.db"
    )
    parser.add_argument("-i", "--input", default=None, help="mbtiles 所在目录")
    parser.add_argument("-o", "--output", default=None, help="输出目录 (默认同 input-dir)")
    parser.add_argument("--workers", type=int, default=1,
                        help="tile-join-ext 并发 worker 数; 1=单调用库内部多线程 (默认)")
    parser.add_argument("--lib-path", default=DEFAULT_LIB_PATH, help="libtile-join-ext.so 路径")
    parser.add_argument("--overview-max-zoom", type=int, default=DEFAULT_OVERVIEW_MAX_ZOOM,
                        help=f"overview.mbtiles 收纳的 zoom 上限 (默认 {DEFAULT_OVERVIEW_MAX_ZOOM}); "
                             f"z<=该值由 tile-join 全量合并进 overview, z>该值的边界瓦片归 boundary; "
                             f"低 zoom 渲染只读 overview 小文件以加速 maplibre")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已存在的产物 (tile_index.db / overview.mbtiles / boundary.mbtiles), "
                             "只补跑缺失的 phase; 用于中途失败后断点续跑, 避免重做 1.5h 的 tile-join-ext")
    args = parser.parse_args()

    log.start = time.time()
    print("=" * 60)
    print(f"build_tile_index.py")
    print(f"  input-dir         : {args.input}")
    print(f"  output-dir        : {args.output or args.input}")
    print(f"  workers           : {args.workers}")
    print(f"  lib-path          : {args.lib_path}")
    print(f"  overview-max-zoom : {args.overview_max_zoom}")
    print(f"  resume            : {args.resume}")
    print("=" * 60)

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output or args.input)
    if not os.path.isdir(input_dir):
        sys.exit(f"input-dir 不存在或不是目录: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(args.lib_path):
        sys.exit(f"找不到 libtile-join-ext.so: {args.lib_path}")

    overview_max_zoom = args.overview_max_zoom
    db_path = os.path.join(output_dir, "tile_index.db")
    overview_path = os.path.join(output_dir, OVERVIEW_SOURCE_FILENAME)
    boundary_path = os.path.join(output_dir, MERGED_SOURCE_FILENAME)

    input_files = phase1_scan(input_dir)

    # Phase 2: 索引库 (resume 时若已建好则跳过)
    if args.resume and _index_db_ready(db_path):
        log("Phase 2", f"resume: tile_index.db 已存在且非空, 跳过重建")
    else:
        db_path = phase2_build_index(input_files, output_dir)

    # Phase 3a: overview (resume 时若已存在且非空则跳过)
    if args.resume and _mbtiles_valid(overview_path):
        ov_count, ov_min, ov_max, _ = _mbtiles_stats(overview_path)
        if ov_count > 0:
            log("Phase 3a", f"resume: overview.mbtiles 已存在 (tiles={ov_count}), 跳过重建")
        else:
            ov_count = -1  # 标记需重建
    else:
        ov_count = -1
    if ov_count == -1:
        overview_path = phase3_build_overview(input_files, output_dir, overview_max_zoom)

    # Phase 3: boundary (resume 时若已存在且非空则跳过; 空文件视为损坏需重建)
    b_count = -1
    if args.resume and _mbtiles_valid(boundary_path):
        b_count, _, _, _ = _mbtiles_stats(boundary_path)
        if b_count > 0:
            log("Phase 3", f"resume: boundary.mbtiles 已存在 (tiles={b_count}), 跳过重建")
        else:
            b_count = -1
    if b_count == -1:
        boundary_path = phase3_build_boundary(input_files, output_dir, args.workers,
                                              args.lib_path, db_path, overview_max_zoom)

    merged_id, overview_id = phase4_update_index(db_path, boundary_path, overview_path)
    phase5_verify(db_path, boundary_path, overview_path, merged_id, overview_id)

    print()
    log("Done", f"全部完成，累计 {time.time() - log.start:.1f}s")
    print(f"  overview.mbtiles  : {overview_path}")
    print(f"  boundary.mbtiles  : {boundary_path}")
    print(f"  tile_index.db     : {db_path}")
    print(f"  重启 tile_server.py 即可生效")


if __name__ == "__main__":
    main()
