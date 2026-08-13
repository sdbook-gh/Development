#!/usr/bin/env python3
"""Generate generate_osm_shp_keymap from:
  1) shape_to_geojson.json  (final shapefile field requirements)
  2) osmpbf_to_shp.json     (PBF source layer + SQL key usage)
  3) optional sample PBF    (actual OSM keys found per layer)

Output: an OSM configuration file whose [points]/[lines]/[multipolygons]/...
sections' `attributes=` list is the union of the base template attributes and
all OSM keys collected from the configs / PBF data.

Usage:
  python3 generate_osmconf.py \
    --shape-json .../shape_to_geojson.json \
    --osmpbf-config osmpbf_to_shp.json \
    --pbf-dir province.osm.pbf --lib-dir .../output/x86_64 \
    --output osm_shp_keymap.ini \
    --template /usr/share/gdal/osmconf.ini \
    --lib-dir /mnt/extdisk/map/app/bundle/scripts/output/x86_64
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# SQL identifier extraction
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL", "AS",
    "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE", "CAST", "LIKE",
    "BETWEEN", "EXISTS", "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT",
    "DISTINCT", "INNER", "LEFT", "RIGHT", "JOIN", "ON", "ASC", "ISNOTNULL",
    "OR", "TRUE", "FALSE", "INTEGER", "REAL", "STRING", "DATETIME",
}

# columns produced by the pipeline, not raw OSM tags
_SYSTEM_FIELDS = {
    "osm_id", "osm_way_id", "geometry", "fclass", "ogc_fid", "FID",
    "z_order",
    # OSM layer table names (FROM ...), not attributes
    "points", "lines", "multipolygons", "multilinestrings",
    "other_relations", "polygons", "nodes", "ways", "relations",
    "osm",
}

# Only strip SQL *string* literals (single quotes). Double-quoted tokens are
# SQLITE identifiers (e.g. "natural" for the reserved-word OSM key) and must
# be kept so extract_identifiers can collect them.
_STR_LIT = re.compile(r"'(?:[^'\\]|\\.)*'")
_DQ_IDENT = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_:]*")


def extract_identifiers(sql: str) -> Set[str]:
    """Return non-keyword, non-system identifiers referenced in an SQL string."""
    sql = sql or ""
    out: Set[str] = set()
    # Explicitly collect double-quoted identifiers before stripping strings.
    for m in _DQ_IDENT.finditer(sql):
        tok = m.group(1).replace('\\"', '"')
        if not tok:
            continue
        if tok.upper() in _SQL_KEYWORDS:
            continue
        if tok in _SYSTEM_FIELDS:
            continue
        out.add(tok)
    no_str = _STR_LIT.sub(" ", sql)
    # Remove double-quoted spans so bare _IDENT does not see leftover quotes.
    no_dq = _DQ_IDENT.sub(" ", no_str)
    for tok in _IDENT.findall(no_dq):
        if tok.upper() in _SQL_KEYWORDS:
            continue
        if tok in _SYSTEM_FIELDS:
            continue
        out.add(tok)
    return out

# ---------------------------------------------------------------------------
# minimal ctypes GDAL for PBF field probing
# ---------------------------------------------------------------------------

class _GdalProbe:
    """Load libgdal.so and expose the few functions needed to list OSM layers/fields."""

    def __init__(self, lib_dir: Path, osmconf: Optional[Path] = None):
        lib_dir = lib_dir.resolve()
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [str(lib_dir)] + ([prev] if prev else [])
        )
        for dep in ("libsqlite3.so", "libproj.so.25", "libproj.so"):
            p = lib_dir / dep
            if p.is_file():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        gdal_path = lib_dir / "libgdal.so"
        if not gdal_path.is_file():
            raise FileNotFoundError(f"libgdal.so not found in {lib_dir}")
        self.g = ctypes.CDLL(str(gdal_path), mode=ctypes.RTLD_GLOBAL)
        self._bind()
        # data dirs
        share = lib_dir.parent / "share"
        gdal_data = share / "gdal"
        proj_data = share / "proj"
        if gdal_data.is_dir():
            self.g.CPLSetConfigOption(b"GDAL_DATA", str(gdal_data).encode())
        if proj_data.is_dir():
            self.g.CPLSetConfigOption(b"PROJ_LIB", str(proj_data).encode())
            self.g.CPLSetConfigOption(b"PROJ_DATA", str(proj_data).encode())
        # OSM driver needs an osmconf.ini to open .pbf
        if osmconf is not None and osmconf.is_file():
            self.g.CPLSetConfigOption(
                b"OSM_CONFIG_FILE", str(osmconf.resolve()).encode()
            )
        self.g.GDALAllRegister()

    def _bind(self):
        g = self.g
        g.CPLSetConfigOption.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        g.GDALAllRegister.restype = None
        g.GDALOpenEx.argtypes = [
            ctypes.c_char_p, ctypes.c_uint, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        g.GDALOpenEx.restype = ctypes.c_void_p
        g.GDALClose.argtypes = [ctypes.c_void_p]
        g.GDALClose.restype = ctypes.c_int
        g.GDALDatasetGetLayerCount.argtypes = [ctypes.c_void_p]
        g.GDALDatasetGetLayerCount.restype = ctypes.c_int
        g.GDALDatasetGetLayer.argtypes = [ctypes.c_void_p, ctypes.c_int]
        g.GDALDatasetGetLayer.restype = ctypes.c_void_p
        g.OGR_L_GetName.argtypes = [ctypes.c_void_p]
        g.OGR_L_GetName.restype = ctypes.c_char_p
        g.OGR_L_GetLayerDefn.argtypes = [ctypes.c_void_p]
        g.OGR_L_GetLayerDefn.restype = ctypes.c_void_p
        g.OGR_FD_GetFieldCount.argtypes = [ctypes.c_void_p]
        g.OGR_FD_GetFieldCount.restype = ctypes.c_int
        g.OGR_FD_GetFieldDefn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        g.OGR_FD_GetFieldDefn.restype = ctypes.c_void_p
        g.OGR_Fld_GetNameRef.argtypes = [ctypes.c_void_p]
        g.OGR_Fld_GetNameRef.restype = ctypes.c_char_p

    def fields_per_layer(self, pbf: Path) -> Dict[str, Set[str]]:
        g = self.g
        flags = 0x04 | 0x00 | 0x40  # OF_VECTOR | OF_READONLY | OF_VERBOSE_ERROR
        ds = g.GDALOpenEx(str(pbf).encode(), flags, None, None, None)
        if not ds:
            return {}
        out: Dict[str, Set[str]] = {}
        try:
            n = g.GDALDatasetGetLayerCount(ds)
            for i in range(n):
                lyr = g.GDALDatasetGetLayer(ds, i)
                if not lyr:
                    continue
                name = g.OGR_L_GetName(lyr).decode() or f"layer{i}"
                defn = g.OGR_L_GetLayerDefn(lyr)
                fields: Set[str] = set()
                if defn:
                    fc = g.OGR_FD_GetFieldCount(defn)
                    for j in range(fc):
                        fdef = g.OGR_FD_GetFieldDefn(defn, j)
                        fn = g.OGR_Fld_GetNameRef(fdef)
                        if fn:
                            fields.add(fn.decode())
                out[name] = fields
        finally:
            g.GDALClose(ds)
        return out


# ---------------------------------------------------------------------------
# config parsing helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_from_osmpbf_config(cfg: dict) -> Dict[str, Set[str]]:
    """Group OSM keys used per PBF source layer (from select/where SQL)."""
    result: Dict[str, Set[str]] = {}
    for layer in cfg.get("layers", []):
        src = layer.get("source_layer", "lines")
        keys: Set[str] = set()
        sql = layer.get("sql") or ""
        if not sql:
            select = layer.get("select", "*")
            where = layer.get("where", "")
            src_name = layer.get("source_layer", "")
            sql = f"SELECT {select} FROM {src_name}"
            if where:
                sql += f" WHERE {where}"
        keys |= extract_identifiers(sql)
        result.setdefault(src, set()).update(keys)
    return result


def _extract_shape_keys(lc: dict) -> Set[str]:
    """Fields from one shape layer config, minus derived/system columns."""
    keys: Set[str] = set()
    sel = lc.get("select", "")
    wh = lc.get("where", "")
    if sel:
        keys.update(extract_identifiers(sel))
    if wh:
        keys.update(extract_identifiers(wh))
        # reverse-map fclass IN ('admin_level4') -> admin_level
        for m in re.finditer(r"fclass\s+IN\s*\(([^)]*)\)", wh, re.I):
            vals = [v.strip().strip("'\"") for v in m.group(1).split(",")]
            for v in vals:
                if v.startswith("admin_level"):
                    keys.add("admin_level")
                elif v in ("national_capital", "city", "town", "village",
                           "hamlet", "locality", "suburb", "county"):
                    keys.add("place")
    return keys


_GEOM_SUFFIX = {"_a", "_f", "_point", "_line", "_polygon"}


def _shapefile_to_layer(output_name: str) -> str:
    """gis_osm_roads_free_1 -> roads ; gis_osm_pois_a_free_1 -> pois_a."""
    m = re.match(r"gis_osm_(.*?)_(?:a|f)_free_\d+", output_name)
    if m:
        base = m.group(1)
        if "_a" in output_name.split("gis_osm_")[1]:
            return base + "_a"
        return base
    m = re.match(r"gis_osm_(.*?)_free_\d+", output_name)
    return m.group(1) if m else output_name


def collect_from_shape_json(
    shape_cfg: dict, osmpbf_cfg: dict
) -> Dict[str, Set[str]]:
    """Map shapefile fields to PBF source layers via output-name linkage."""
    # output (gis_osm_roads_free_1) -> source_layer (lines)
    out_to_src: Dict[str, str] = {}
    for layer in osmpbf_cfg.get("layers", []):
        out_to_src[layer.get("output", "")] = layer.get("source_layer", "")

    result: Dict[str, Set[str]] = {}
    groups = [shape_cfg.get("layers", {}), shape_cfg.get("text_layers", {})]
    for grp in groups:
        if not isinstance(grp, dict):
            continue
        for lc in grp.values():
            if not isinstance(lc, dict):
                continue
            match = lc.get("shapefilematch", "")
            keys = _extract_shape_keys(lc)
            # Keep real OSM keys; drop system/derived columns only.
            keys = {k for k in keys if k not in _SYSTEM_FIELDS}
            if not keys:
                continue
            # resolve output name(s) by matching the shapefilematch regex
            # against each known output filename (with .shp appended).
            # shapefilematch is a regex pattern (e.g. ".*_traffic_f.*shp");
            # match it forward against output names instead of trying to
            # reverse-parse a literal "gis_osm_" prefix out of the pattern.
            matched_outputs = []
            if match:
                try:
                    pat = re.compile(match)
                except re.error:
                    pat = None
                if pat is not None:
                    for out in out_to_src:
                        if pat.search(out + ".shp"):
                            matched_outputs.append(out)
            for out in matched_outputs:
                src = out_to_src[out]
                result.setdefault(src, set()).update(keys)
    return result


# ---------------------------------------------------------------------------
# osmconf.ini generation
# ---------------------------------------------------------------------------

_SECTIONS = ["points", "lines", "multipolygons", "multilinestrings",
             "other_relations"]


def parse_ini_sections(path: Path) -> Tuple[Dict[str, List[str]], str]:
    """Parse a config file into {section: [lines-without-section-header]} plus header."""
    text = path.read_text(encoding="utf-8")
    header: List[str] = []
    sections: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1].strip()
            sections.setdefault(cur, [])
            continue
        if cur is None:
            header.append(line)
        else:
            sections[cur].append(line)
    return sections, "\n".join(header)


def attributes_from_lines(lines: List[str]) -> Set[str]:
    for ln in lines:
        if ln.startswith("attributes="):
            return {x.strip() for x in ln[len("attributes="):].split(",") if x.strip()}
    return set()


def merge_attributes(template_attrs: Set[str], extra: Set[str]) -> List[str]:
    merged = template_attrs | extra
    # keep the same ordering as template when possible, then extras sorted
    ordered: List[str] = []
    for a in template_attrs:
        if a in merged:
            ordered.append(a)
    for a in sorted(merged - set(template_attrs)):
        ordered.append(a)
    return ordered


def generate(
    template: Path,
    shape_cfg: dict,
    osmpbf_cfg: dict,
    pbf_fields: Optional[Dict[str, Set[str]]],
) -> str:
    sections, header = parse_ini_sections(template)

    # 1) from osmpbf_to_shp.json, per source layer
    osmpbf_keys = collect_from_osmpbf_config(osmpbf_cfg)
    # 2) from shape_to_geojson.json, mapped to source layers
    shape_keys = collect_from_shape_json(shape_cfg, osmpbf_cfg)
    # 3) from PBF data (if provided): keys = fields - system fields
    pbf_keys: Dict[str, Set[str]] = {}
    if pbf_fields:
        for layer, fields in pbf_fields.items():
            pbf_keys[layer] = {f for f in fields if f not in _SYSTEM_FIELDS
                               and not f.endswith("_tags")}

    out: List[str] = []
    out.append("#")
    out.append("# GDAL OSM config for Geofabrik-like shapefile export.")
    out.append("# Generated by generate_osmconf.py — do not edit by hand.")
    out.append("#")
    out.append("")
    # copy template header lines that are config directives (closed_ways_are_polygons, etc.)
    for ln in header.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        out.append(ln)
    out.append("")

    for sec in _SECTIONS:
        lines = sections.get(sec, [])
        out.append(f"[{sec}]")
        base_attrs = attributes_from_lines(lines)
        extra: Set[str] = set()
        # PBF layer may be named differently (points vs nodes etc.)
        for alias in (sec, sec.rstrip("s")):
            if alias in osmpbf_keys:
                extra |= osmpbf_keys[alias]
            if alias in pbf_keys:
                extra |= pbf_keys[alias]
        # shape-derived keys are already mapped to their specific source layer
        if sec in shape_keys:
            extra |= shape_keys[sec]
        merged = merge_attributes(base_attrs, extra)
        for ln in lines:
            if ln.startswith("attributes="):
                out.append("attributes=" + ",".join(merged))
            else:
                out.append(ln)
        if not lines:
            out.extend(["osm_id=yes", "osm_version=no", "osm_timestamp=no",
                        "osm_uid=no", "osm_user=no", "osm_changeset=no"])
            out.append("attributes=" + ",".join(sorted(extra)))
            out.append("ignore=area,created_by,converted_by,source,time,ele,"
                       "note,todo,openGeoDB:,fixme,FIXME")
        out.append("")
    return "\n".join(out)


def write_probe_osmconf(
    template: Path,
    osmpbf_cfg: dict,
    shape_cfg: dict,
    dest: Path,
) -> Path:
    """Write a temporary osmconf that already declares JSON-required attributes.

    PBF field probing only sees columns listed in OSM_CONFIG_FILE attributes=.
    Using the bare GDAL template creates a circular dependency (e.g. points has
    no `natural`, so probing never discovers it). Seed the probe config from
    osmpbf/shape JSON first so declared keys are visible to GDAL.
    """
    content = generate(template, shape_cfg, osmpbf_cfg, pbf_fields=None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content + "\n", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate osmconf.ini for OSM PBF → shapefile")
    p.add_argument("--shape-json", required=True,
                   help="shape_to_geojson.json (shapefile field requirements)")
    p.add_argument("--osmpbf-config", required=True,
                   help="osmpbf_to_shp.json (PBF layer SQL config)")
    p.add_argument("--pbf-dir", action="append", default=[],
                   help="dir(s) with *.osm.pbf samples to probe (optional)")
    p.add_argument("--lib-dir", default=None,
                   help="dir with libgdal.so (needed only with --pbf-dir)")
    p.add_argument("--output", required=True, help="output osmconf.ini path")
    p.add_argument("--template", default="/usr/share/gdal/osmconf.ini",
                   help="base template (default /usr/share/gdal/osmconf.ini)")
    p.add_argument("--dry-run", action="store_true",
                   help="print generated content, do not write file")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    shape_cfg = load_json(Path(args.shape_json).resolve())
    osmpbf_cfg = load_json(Path(args.osmpbf_config).resolve())
    template = Path(args.template).resolve()

    pbf_fields: Optional[Dict[str, Set[str]]] = None
    probe_osmconf_path: Optional[Path] = None
    if args.pbf_dir:
        if not args.lib_dir:
            print("--lib-dir is required when --pbf-dir is given", file=sys.stderr)
            return 2
        import tempfile

        # Seed probe osmconf from JSON keys so attributes like "natural"
        # are declared before GDAL opens the PBF (breaks template circularity).
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_probe_osmconf.ini", delete=False, encoding="utf-8"
        )
        tmp.close()
        probe_osmconf_path = Path(tmp.name)
        write_probe_osmconf(template, osmpbf_cfg, shape_cfg, probe_osmconf_path)
        print(
            f"[probe] using seeded OSM_CONFIG_FILE={probe_osmconf_path}",
            file=sys.stderr,
        )
        probe = _GdalProbe(Path(args.lib_dir).resolve(), probe_osmconf_path)
        pbf_fields = {}
        try:
            for d in args.pbf_dir:
                for pbf in sorted(Path(d).glob("*.osm.pbf"))[:3]:
                    fp = probe.fields_per_layer(pbf)
                    for layer, fields in fp.items():
                        pbf_fields.setdefault(layer, set()).update(fields)
                    print(
                        f"[probe] {pbf.name}: "
                        + ", ".join(f"{k}({len(v)})" for k, v in fp.items()),
                        file=sys.stderr,
                    )
        finally:
            try:
                probe_osmconf_path.unlink(missing_ok=True)
            except OSError:
                pass

    content = generate(template, shape_cfg, osmpbf_cfg, pbf_fields)
    if args.dry_run:
        print(content)
        return 0
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")
    # summary
    osmpbf_keys = collect_from_osmpbf_config(osmpbf_cfg)
    for sec, keys in sorted(osmpbf_keys.items()):
        print(f"  [{sec}] keys: {', '.join(sorted(keys))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
