#!/usr/bin/env python3
"""Convert OSM PBF files to Geofabrik-like Shapefiles via bundle libgdal.so (ctypes).

Features:
  - --lib-dir: load libgdal.so (+ libproj/libsqlite3) from a directory
  - Concurrent (province, layer) tasks via ProcessPoolExecutor
  - SQLite import.db for progress; --resume / --restart
  - import.log for errors and analysis
  - --dry-run: enumerate tasks / verify GDAL load, no conversion
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# GDAL open flags
GDAL_OF_READONLY = 0x00
GDAL_OF_VECTOR = 0x04
GDAL_OF_VERBOSE_ERROR = 0x40

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("osmpbf_to_shp")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# GDAL ctypes
# ---------------------------------------------------------------------------

class GdalApi:
    def __init__(self, lib_dir: Path, osmconf: Optional[Path] = None):
        self.lib_dir = lib_dir.resolve()
        self._gdal = self._load(lib_dir)
        self._bind()
        self._configure(lib_dir, osmconf)
        self._gdal.GDALAllRegister()

    @staticmethod
    def _load(lib_dir: Path) -> ctypes.CDLL:
        lib_dir = lib_dir.resolve()
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"--lib-dir not a directory: {lib_dir}")

        # Ensure dynamic linker finds sibling deps (proj/sqlite3).
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(lib_dir)] + ([prev] if prev else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

        # Preload common deps if present (order matters).
        preload = [
            "libsqlite3.so",
            "libproj.so.25",
            "libproj.so",
            "libz.so",
            "libz.so.1",
        ]
        for name in preload:
            path = lib_dir / name
            if path.is_file():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

        gdal_path = lib_dir / "libgdal.so"
        if not gdal_path.is_file():
            raise FileNotFoundError(f"libgdal.so not found in {lib_dir}")
        return ctypes.CDLL(str(gdal_path), mode=ctypes.RTLD_GLOBAL)

    def _bind(self) -> None:
        g = self._gdal
        g.CPLSetConfigOption.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        g.CPLSetConfigOption.restype = None
        g.CPLGetLastErrorMsg.argtypes = []
        g.CPLGetLastErrorMsg.restype = ctypes.c_char_p
        g.GDALAllRegister.argtypes = []
        g.GDALAllRegister.restype = None
        g.GDALVersionInfo.argtypes = [ctypes.c_char_p]
        g.GDALVersionInfo.restype = ctypes.c_char_p
        g.GDALGetDriverByName.argtypes = [ctypes.c_char_p]
        g.GDALGetDriverByName.restype = ctypes.c_void_p
        g.GDALOpenEx.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        g.GDALOpenEx.restype = ctypes.c_void_p
        g.GDALClose.argtypes = [ctypes.c_void_p]
        g.GDALClose.restype = ctypes.c_int
        g.GDALDatasetGetLayerCount.argtypes = [ctypes.c_void_p]
        g.GDALDatasetGetLayerCount.restype = ctypes.c_int
        g.GDALVectorTranslateOptionsNew.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_void_p,
        ]
        g.GDALVectorTranslateOptionsNew.restype = ctypes.c_void_p
        g.GDALVectorTranslateOptionsFree.argtypes = [ctypes.c_void_p]
        g.GDALVectorTranslateOptionsFree.restype = None
        g.GDALVectorTranslate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        g.GDALVectorTranslate.restype = ctypes.c_void_p

    def _configure(self, lib_dir: Path, osmconf: Optional[Path]) -> None:
        # Prefer share next to lib_dir parent: .../output/share/{gdal,proj}
        output_root = lib_dir.parent
        gdal_data_candidates = [
            output_root / "share" / "gdal",
            lib_dir / "share" / "gdal",
            Path("/usr/share/gdal"),
        ]
        proj_candidates = [
            output_root / "share" / "proj",
            lib_dir / "share" / "proj",
        ]
        for p in gdal_data_candidates:
            if p.is_dir():
                self._gdal.CPLSetConfigOption(b"GDAL_DATA", str(p).encode())
                break
        for p in proj_candidates:
            if p.is_dir():
                self._gdal.CPLSetConfigOption(b"PROJ_LIB", str(p).encode())
                self._gdal.CPLSetConfigOption(b"PROJ_DATA", str(p).encode())
                break
        if osmconf is not None:
            self._gdal.CPLSetConfigOption(
                b"OSM_CONFIG_FILE", str(osmconf.resolve()).encode()
            )

    def last_error(self) -> str:
        msg = self._gdal.CPLGetLastErrorMsg()
        return msg.decode(errors="replace") if msg else ""

    def version(self) -> str:
        v = self._gdal.GDALVersionInfo(b"--version")
        return v.decode() if v else "unknown"

    def has_driver(self, name: str) -> bool:
        return bool(self._gdal.GDALGetDriverByName(name.encode()))

    @staticmethod
    def _cstr_list(args: Sequence[str]):
        arr = (ctypes.c_char_p * (len(args) + 1))()
        for i, a in enumerate(args):
            arr[i] = a.encode()
        arr[len(args)] = None
        return arr

    def translate(
        self,
        src_pbf: Path,
        out_shp: Path,
        sql: str,
        lco: Optional[List[str]] = None,
        dialect: Optional[str] = "SQLITE",
    ) -> None:
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        flags = GDAL_OF_VECTOR | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR
        src = self._gdal.GDALOpenEx(str(src_pbf).encode(), flags, None, None, None)
        if not src:
            raise RuntimeError(f"GDALOpenEx failed: {self.last_error()}")

        argv: List[str] = [
            "-f",
            "ESRI Shapefile",
            "-overwrite",
            "-sql",
            sql,
        ]
        if dialect:
            argv.extend(["-dialect", dialect])
        for item in lco or ["ENCODING=UTF-8"]:
            argv.extend(["-lco", item])

        opts = self._gdal.GDALVectorTranslateOptionsNew(self._cstr_list(argv), None)
        if not opts:
            self._gdal.GDALClose(src)
            raise RuntimeError(
                f"GDALVectorTranslateOptionsNew failed: {self.last_error()}"
            )

        src_arr = (ctypes.c_void_p * 1)(src)
        usage = ctypes.c_int(0)
        try:
            dst = self._gdal.GDALVectorTranslate(
                str(out_shp).encode(),
                None,
                1,
                src_arr,
                opts,
                ctypes.byref(usage),
            )
            if usage.value:
                raise RuntimeError(
                    f"GDALVectorTranslate usage error ({usage.value}): "
                    f"{self.last_error()}"
                )
            if not dst:
                raise RuntimeError(
                    f"GDALVectorTranslate failed: {self.last_error()}"
                )
            self._gdal.GDALClose(dst)
        finally:
            self._gdal.GDALVectorTranslateOptionsFree(opts)
            self._gdal.GDALClose(src)


# ---------------------------------------------------------------------------
# config / tasks
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_osmconf(cfg: dict, config_path: Path, cli_osmconf: Optional[str]) -> Path:
    if cli_osmconf:
        p = Path(cli_osmconf)
    else:
        rel = cfg.get("osmconf", "osmconf_geofabrik.ini")
        p = Path(rel)
        if not p.is_absolute():
            p = config_path.parent / p
    if not p.is_file():
        raise FileNotFoundError(f"osmconf not found: {p}")
    return p.resolve()


def pbf_subdir_name(pbf: Path) -> str:
    name = pbf.name
    for suf in (".osm.pbf", ".pbf"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return pbf.stem


def build_sql(layer: dict, dialect: str = "SQLITE") -> str:
    if "sql" in layer and layer["sql"]:
        sql = layer["sql"]
    else:
        select = layer.get("select", "*")
        source = layer["source_layer"]
        where = layer.get("where") or ""
        # SQLITE dialect drops geometry unless selected explicitly.
        if (
            dialect
            and dialect.upper() == "SQLITE"
            and select.strip() != "*"
            and "geometry" not in select.lower()
        ):
            select = f"{select}, geometry"
        sql = f"SELECT {select} FROM {source}"
        if where.strip():
            sql += f" WHERE {where}"
    return sql


def discover_pbfs(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.osm.pbf"))
    files += sorted(p for p in input_dir.glob("*.pbf") if p not in files)
    return files


def enumerate_tasks(
    pbfs: Sequence[Path],
    layers: Sequence[dict],
    output_dir: Path,
    dialect: str = "SQLITE",
) -> List[dict]:
    tasks = []
    for pbf in pbfs:
        sub = pbf_subdir_name(pbf)
        for layer in layers:
            out_name = layer["output"]
            if not out_name.endswith(".shp"):
                out_name = out_name + ".shp"
            out_shp = output_dir / sub / out_name
            layer_dialect = layer.get("dialect") or dialect
            tasks.append(
                {
                    "src_pbf": str(pbf.resolve()),
                    "layer": layer["name"],
                    "output": layer["output"],
                    "source_layer": layer["source_layer"],
                    "sql": build_sql(layer, layer_dialect),
                    "out_shp": str(out_shp),
                    "lco": layer.get("lco") or ["ENCODING=UTF-8"],
                    "dialect": layer_dialect,
                }
            )
    return tasks


# ---------------------------------------------------------------------------
# import.db
# ---------------------------------------------------------------------------

def db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_pbf TEXT NOT NULL,
            layer TEXT NOT NULL,
            out_shp TEXT NOT NULL,
            sql_text TEXT,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            error_msg TEXT,
            pid INTEGER,
            UNIQUE(src_pbf, layer)
        )
        """
    )
    conn.commit()


def db_reset(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM jobs")
    conn.commit()


def db_upsert_pending(conn: sqlite3.Connection, task: dict) -> None:
    conn.execute(
        """
        INSERT INTO jobs (src_pbf, layer, out_shp, sql_text, status)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(src_pbf, layer) DO UPDATE SET
            out_shp=excluded.out_shp,
            sql_text=excluded.sql_text
        """,
        (
            task["src_pbf"],
            task["layer"],
            task["out_shp"],
            task["sql"],
            STATUS_PENDING,
        ),
    )


def db_get_status(conn: sqlite3.Connection, src_pbf: str, layer: str) -> Optional[str]:
    row = conn.execute(
        "SELECT status FROM jobs WHERE src_pbf=? AND layer=?",
        (src_pbf, layer),
    ).fetchone()
    return row[0] if row else None


def db_mark(
    conn: sqlite3.Connection,
    src_pbf: str,
    layer: str,
    status: str,
    error_msg: Optional[str] = None,
) -> None:
    now = utc_now()
    if status == STATUS_RUNNING:
        conn.execute(
            """
            UPDATE jobs SET status=?, started_at=?, finished_at=NULL,
                            error_msg=NULL, pid=?
            WHERE src_pbf=? AND layer=?
            """,
            (status, now, os.getpid(), src_pbf, layer),
        )
    elif status == STATUS_DONE:
        conn.execute(
            """
            UPDATE jobs SET status=?, finished_at=?, error_msg=NULL, pid=?
            WHERE src_pbf=? AND layer=?
            """,
            (status, now, os.getpid(), src_pbf, layer),
        )
    elif status == STATUS_ERROR:
        conn.execute(
            """
            UPDATE jobs SET status=?, finished_at=?, error_msg=?, pid=?
            WHERE src_pbf=? AND layer=?
            """,
            (status, now, (error_msg or "")[:4000], os.getpid(), src_pbf, layer),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status=? WHERE src_pbf=? AND layer=?",
            (status, src_pbf, layer),
        )
    conn.commit()


def db_reset_stale_running(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "UPDATE jobs SET status=? WHERE status=?",
        (STATUS_ERROR, STATUS_RUNNING),
    )
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _worker_convert(payload: dict) -> dict:
    """Process-pool entry: convert one (pbf, layer)."""
    t0 = time.time()
    result = {
        "src_pbf": payload["src_pbf"],
        "layer": payload["layer"],
        "out_shp": payload["out_shp"],
        "ok": False,
        "error": "",
        "elapsed": 0.0,
    }
    try:
        api = GdalApi(Path(payload["lib_dir"]), Path(payload["osmconf"]))
        if not api.has_driver("OSM"):
            raise RuntimeError("OSM driver not available in libgdal.so")
        if not api.has_driver("ESRI Shapefile"):
            raise RuntimeError("ESRI Shapefile driver not available")
        api.translate(
            Path(payload["src_pbf"]),
            Path(payload["out_shp"]),
            payload["sql"],
            payload.get("lco"),
            payload.get("dialect") or "SQLITE",
        )
        # require .shp exists
        if not Path(payload["out_shp"]).is_file():
            raise RuntimeError("output .shp missing after translate")
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
    result["elapsed"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------

def filter_tasks_for_resume(
    conn: sqlite3.Connection, tasks: List[dict], restart: bool
) -> List[dict]:
    if restart:
        return tasks
    selected = []
    for t in tasks:
        st = db_get_status(conn, t["src_pbf"], t["layer"])
        if st == STATUS_DONE:
            continue
        selected.append(t)
    return selected


def _terminate_pool(pool: ProcessPoolExecutor, logger: logging.Logger) -> None:
    """Force-stop worker processes left by Ctrl+C / SIGTERM."""
    logger.warning("interrupt received: shutting down worker processes...")
    try:
        # Python 3.9+: cancel pending futures
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        pool.shutdown(wait=False)

    procs = []
    try:
        procs = list(getattr(pool, "_processes", {}) or {}.values())
    except Exception:
        procs = []

    for proc in procs:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    deadline = time.time() + 3.0
    for proc in procs:
        try:
            remaining = max(0.0, deadline - time.time())
            proc.join(timeout=remaining)
        except Exception:
            pass

    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        except Exception:
            pass

    # Extra safety: kill any still-tracked children of this process that look like workers
    try:
        import multiprocessing as mp

        for child in mp.active_children():
            try:
                if child.is_alive():
                    child.terminate()
            except Exception:
                pass
        time.sleep(0.3)
        for child in mp.active_children():
            try:
                if child.is_alive():
                    child.kill()
            except Exception:
                pass
    except Exception:
        pass

    logger.warning("worker cleanup finished")


def run_pool(
    payloads: List[dict],
    workers: int,
    conn: sqlite3.Connection,
    logger: logging.Logger,
) -> Tuple[int, int, bool]:
    """Run conversions; return (ok_n, err_n, interrupted)."""
    ok_n = err_n = 0
    interrupted = False
    stop = {"flag": False}
    pool_holder: Dict[str, Optional[ProcessPoolExecutor]] = {"pool": None}

    def _on_signal(signum, _frame):
        stop["flag"] = True
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning("caught %s", name)
        pool = pool_holder["pool"]
        if pool is not None:
            _terminate_pool(pool, logger)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)

    pool = ProcessPoolExecutor(max_workers=max(1, workers))
    pool_holder["pool"] = pool
    try:
        futures = {pool.submit(_worker_convert, pl): pl for pl in payloads}
        for fut in as_completed(futures):
            if stop["flag"]:
                interrupted = True
                break
            pl = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {
                    "src_pbf": pl["src_pbf"],
                    "layer": pl["layer"],
                    "out_shp": pl["out_shp"],
                    "ok": False,
                    "error": str(exc),
                    "elapsed": 0.0,
                }
            if r["ok"]:
                ok_n += 1
                db_mark(conn, r["src_pbf"], r["layer"], STATUS_DONE)
                logger.info(
                    "OK %s/%s (%.1fs) -> %s",
                    Path(r["src_pbf"]).name,
                    r["layer"],
                    r["elapsed"],
                    r["out_shp"],
                )
            else:
                err_n += 1
                db_mark(conn, r["src_pbf"], r["layer"], STATUS_ERROR, r["error"])
                logger.error(
                    "FAIL %s/%s (%.1fs): %s",
                    Path(r["src_pbf"]).name,
                    r["layer"],
                    r["elapsed"],
                    r["error"].splitlines()[0] if r["error"] else "unknown",
                )
                logger.debug("FAIL detail:\n%s", r["error"])
    finally:
        if stop["flag"]:
            interrupted = True
            if pool_holder["pool"] is not None:
                _terminate_pool(pool, logger)
            # Mark unfinished running jobs as error so --resume can retry
            try:
                n = conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, error_msg=? WHERE status=?",
                    (
                        STATUS_ERROR,
                        utc_now(),
                        "interrupted by signal; worker cleaned up",
                        STATUS_RUNNING,
                    ),
                ).rowcount
                conn.commit()
                if n:
                    logger.warning("marked %d running job(s) as error after interrupt", n)
            except Exception as exc:
                logger.error("failed to update running jobs after interrupt: %s", exc)
        else:
            try:
                pool.shutdown(wait=True)
            except Exception:
                _terminate_pool(pool, logger)

        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        pool_holder["pool"] = None

    return ok_n, err_n, interrupted


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSM PBF → Shapefile via libgdal.so (ctypes)"
    )
    p.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "osmpbf_to_shp.json"),
        help="JSON mapping config",
    )
    p.add_argument(
        "--lib-dir",
        required=True,
        help="Directory containing libgdal.so (and libproj/libsqlite3)",
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory of *.osm.pbf files to scan",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Root output dir; per-PBF subdirs created from filename stem",
    )
    p.add_argument("--workers", type=int, default=None, help="Concurrency")
    p.add_argument("--osmconf", default=None, help="Override OSM_CONFIG_FILE")
    p.add_argument(
        "--db",
        default=None,
        help="import.db path (default: <output-dir>/import.db)",
    )
    p.add_argument(
        "--log",
        default=None,
        help="import.log path (default: <output-dir>/import.log)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs marked done; retry error/pending/running",
    )
    mode.add_argument(
        "--restart",
        action="store_true",
        help="Clear progress DB and reconvert all",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/GDAL/tasks only; do not convert or mutate DB",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks (0=all); useful for testing",
    )
    p.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all layer names from config and exit",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer names to process (default: all)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    lib_dir = Path(args.lib_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))
    osmconf = resolve_osmconf(cfg, config_path, args.osmconf)

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else output_dir / "import.db"
    log_path = Path(args.log) if args.log else output_dir / "import.log"
    if args.db:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.log:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path)
    tag = "[DRY-RUN] " if args.dry_run else ""
    logger.info("%sconfig=%s", tag, config_path)
    logger.info("%slib_dir=%s", tag, lib_dir)
    logger.info("%sinput_dir=%s", tag, input_dir)
    logger.info("%soutput_dir=%s", tag, output_dir)
    logger.info("%sosmconf=%s", tag, osmconf)
    logger.info("%sdb=%s log=%s workers=%s", tag, db_path, log_path, workers)

    # Validate GDAL
    try:
        api = GdalApi(lib_dir, osmconf)
    except Exception as exc:
        logger.error("%sfailed to load libgdal: %s", tag, exc)
        return 2

    logger.info("%sGDAL: %s", tag, api.version())
    for drv in ("OSM", "ESRI Shapefile"):
        ok = api.has_driver(drv)
        logger.info("%sdriver %s: %s", tag, drv, "OK" if ok else "MISSING")
        if not ok:
            logger.error("%srequired driver missing: %s", tag, drv)
            return 2

    layers = cfg.get("layers") or []
    if not layers:
        logger.error("%sno layers in config", tag)
        return 2

    pbfs = discover_pbfs(input_dir)
    if not pbfs:
        logger.error("%sno .osm.pbf/.pbf under %s", tag, input_dir)
        return 2
    logger.info("%sfound %d PBF file(s)", tag, len(pbfs))

    tasks = enumerate_tasks(
        pbfs, layers, output_dir, dialect=str(cfg.get("sql_dialect", "SQLITE"))
    )

    if args.list_layers:
        seen = set()
        for t in tasks:
            if t["layer"] not in seen:
                seen.add(t["layer"])
                print(f"{t['layer']:20} | {t['source_layer']:20} | {t['output']}")
        return 0

    if args.layers:
        wanted = {s.strip() for s in args.layers.split(",") if s.strip()}
        tasks = [t for t in tasks if t["layer"] in wanted]
        if not tasks:
            logger.error("%sno tasks match --layers=%s", tag, args.layers)
            return 2
        logger.info("%sfiltered to %d task(s) by --layers", tag, len(tasks))

    logger.info(
        "%stotal tasks=%d (%d pbf x %d layers)",
        tag,
        len(tasks),
        len(pbfs),
        len(layers),
    )

    if args.dry_run:
        # Sample first few tasks
        sample_n = min(5, len(tasks))
        for t in tasks[:sample_n]:
            logger.info(
                "%sTASK %s | %s -> %s | SQL: %s",
                tag,
                Path(t["src_pbf"]).name,
                t["layer"],
                t["out_shp"],
                t["sql"],
            )
        if len(tasks) > sample_n:
            logger.info("%s... %d more tasks omitted", tag, len(tasks) - sample_n)
        # Probe open one PBF
        probe = pbfs[0]
        flags = GDAL_OF_VECTOR | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR
        ds = api._gdal.GDALOpenEx(str(probe).encode(), flags, None, None, None)
        if not ds:
            logger.error("%sprobe open failed: %s", tag, api.last_error())
            return 2
        nlayers = api._gdal.GDALDatasetGetLayerCount(ds)
        api._gdal.GDALClose(ds)
        logger.info("%sprobe open OK: %s (%d layers)", tag, probe.name, nlayers)
        logger.info("%sdry-run complete; no conversion performed", tag)
        return 0

    if not args.resume and not args.restart:
        logger.error("specify --resume or --restart for real conversion")
        return 2

    conn = db_connect(db_path)
    db_init(conn)
    if args.restart:
        logger.info("restart: clearing jobs table")
        db_reset(conn)
    else:
        n = db_reset_stale_running(conn)
        if n:
            logger.warning("marked %d stale running job(s) as error", n)

    for t in tasks:
        db_upsert_pending(conn, t)
    conn.commit()

    todo = filter_tasks_for_resume(conn, tasks, restart=args.restart)
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    logger.info("jobs to run: %d (skipped done handled by --resume)", len(todo))
    if not todo:
        logger.info("nothing to do")
        return 0

    ok_n = err_n = 0
    payloads = []
    for t in todo:
        db_mark(conn, t["src_pbf"], t["layer"], STATUS_RUNNING)
        payloads.append(
            {
                **t,
                "lib_dir": str(lib_dir),
                "osmconf": str(osmconf),
            }
        )

    ok_n, err_n, interrupted = run_pool(payloads, workers, conn, logger)

    conn.close()
    if interrupted:
        logger.warning("interrupted: ok=%d error=%d (use --resume to continue)", ok_n, err_n)
        return 130
    logger.info("finished: ok=%d error=%d", ok_n, err_n)
    return 1 if err_n else 0


if __name__ == "__main__":
    sys.exit(main())
