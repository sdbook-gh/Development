#!/usr/bin/env python3
"""
MapLibre Glyphs Service

Serves glyph PBF files for MapLibre GL font rendering.

Phase 1: Static PBF file serving (fast path)
Phase 2: Dynamic TTF->PBF generation via fontnik subprocess (fallback)
Phase 2b: Python-native fallback using fonttools (metrics only, no SDF)

PBF wire format (outer glyphs message):
  0x0A <varint:total>
    Field 1 (0x0A): fontstack name (string)
    Field 2 (0x12): range string (string)
    Field 3 (0x1A): glyph data (repeated bytes, with 1+ glyph messages each)

Usage:
    python3 fonts/serve_glyphs.py [port]
"""
import http.server
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'fonts', 'font_config.json')

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

CACHE_DIR = os.path.join(BASE_DIR, CONFIG.get('cache_dir', 'fonts/.cache'))
FONTNIK_SCRIPT = os.path.join(BASE_DIR, CONFIG.get('fontnik_script', 'fonts/generate_range.js'))
STACKS = CONFIG.get('stacks', {})
DEFAULT_PORT = CONFIG.get('port', 8080)

os.makedirs(CACHE_DIR, exist_ok=True)

_compile_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Protobuf wire-format helpers
# ---------------------------------------------------------------------------

def _varint(n):
    result = bytearray()
    while n > 0x7F:
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    result.append(n & 0x7F)
    return bytes(result)


def _signed_varint(n):
    return _varint((n << 1) ^ (n >> 31))


def _read_varint(data, i):
    val = 0
    shift = 0
    while i < len(data) and data[i] & 0x80:
        val |= (data[i] & 0x7F) << shift
        shift += 7
        i += 1
    if i < len(data):
        val |= data[i] << shift
        i += 1
    return val, i


# ---------------------------------------------------------------------------
# PBF parsing - handles the actual glyphs format:
#   outer message: field 1 (fontstack), field 2 (range), field 3 (glyphs)*
#   field 3 contains concatenated glyph messages
# ---------------------------------------------------------------------------

def _parse_glyph_field3(field3_data):
    """Parse concatenated glyph messages from a single field 3 chunk.
    Returns list of (glyph_id, raw_message_bytes).
    """
    glyphs = []
    i = 0
    while i < len(field3_data):
        start = i
        gid = None
        while i < len(field3_data):
            key = field3_data[i]
            i += 1
            fn = key >> 3
            wt = key & 0x7

            if fn == 1 and wt == 0:
                gid, i = _read_varint(field3_data, i)
            elif wt == 0:
                _, i = _read_varint(field3_data, i)
            elif wt == 2:
                length, i = _read_varint(field3_data, i)
                i += length
            elif wt == 1:
                i += 8
            elif wt == 5:
                i += 4
            else:
                break

            if i >= len(field3_data):
                break
            # Check if next is field 1 of a new glyph
            next_key = field3_data[i]
            if (next_key >> 3) == 1 and (next_key & 0x7) == 0:
                break
        if gid is not None:
            glyphs.append((gid, field3_data[start:i]))
    return glyphs


def _parse_glyphs_pbf(data):
    """Parse full PBF, return (fontstack, range, [(gid, raw)]) or None."""
    try:
        i = 0
        if data[i] != 0x0A:
            return None
        i += 1
        total_len, i = _read_varint(data, i)

        fontstack = None
        range_str = None
        all_glyphs = []

        while i < len(data):
            key = data[i]
            i += 1
            fn = key >> 3
            wt = key & 0x7
            if wt != 2:
                break
            length, i = _read_varint(data, i)
            payload = data[i:i + length]
            i += length

            if fn == 1:
                fontstack = payload.decode('utf-8')
            elif fn == 2:
                range_str = payload.decode('utf-8')
            elif fn == 3:
                glyphs = _parse_glyph_field3(payload)
                all_glyphs.extend(glyphs)

        return fontstack, range_str, all_glyphs
    except Exception:
        return None


def _build_glyphs_pbf(fontstack, range_str, glyphs):
    """Build a complete glyphs PBF from parts.
    glyphs: list of (gid, raw_message_bytes)
    """
    body = bytearray()

    # Field 1: fontstack
    fs_bytes = fontstack.encode('utf-8') if fontstack else b''
    body += b'\x0a' + _varint(len(fs_bytes)) + fs_bytes

    # Field 2: range
    r_bytes = range_str.encode('utf-8') if range_str else b''
    body += b'\x12' + _varint(len(r_bytes)) + r_bytes

    # Field 3: glyphs - group multiple glyphs per chunk (like fontnik does)
    for gid, raw in glyphs:
        body += b'\x1a' + _varint(len(raw)) + raw

    return b'\x0a' + _varint(len(body)) + bytes(body)


def _merge_glyph_pbfs(pbf_list, fontstack, range_str):
    """Merge glyphs from multiple PBFs, keeping first occurrence of each ID."""
    seen_ids = set()
    merged_glyphs = []

    for data in pbf_list:
        result = _parse_glyphs_pbf(data)
        if result is None:
            # Pass-through unrecognized format
            if fontstack and range_str:
                merged_glyphs.append(None)
            continue
        _, _, glyphs = result
        for gid, raw in glyphs:
            if gid not in seen_ids:
                seen_ids.add(gid)
                merged_glyphs.append((gid, raw))

    if merged_glyphs:
        return _build_glyphs_pbf(fontstack, range_str, merged_glyphs)
    return pbf_list[0] if pbf_list else None


# ---------------------------------------------------------------------------
# Font resolution & data sources
# ---------------------------------------------------------------------------

def _resolve_font(fontstack):
    fontstack = urllib.parse.unquote(fontstack)
    names = [n.strip() for n in fontstack.split(',')]
    return [STACKS.get(name, {}) for name in names]


def _parse_range(range_name):
    name = range_name.replace('.pbf', '')
    parts = name.split('-')
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _serve_static_pbf(pbf_dir, range_name):
    pbf_path = os.path.join(BASE_DIR, pbf_dir, range_name)
    if os.path.exists(pbf_path) and os.path.getsize(pbf_path) > 50:
        with open(pbf_path, 'rb') as f:
            return f.read()
    return None


def _generate_via_fontnik(ttf_rel, start, end):
    ttf_path = os.path.join(BASE_DIR, ttf_rel)
    if not os.path.exists(ttf_path):
        return None

    cache_name = f"{os.path.splitext(os.path.basename(ttf_rel))[0]}_{start}-{end}.pbf"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return f.read()

    with _compile_lock:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return f.read()

        result = subprocess.run(
            ['node', FONTNIK_SCRIPT, ttf_path, str(start), str(end), cache_path],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return None

        with open(cache_path, 'rb') as f:
            return f.read()


def _generate_fallback_glyphs(glyph_ids, ttf_rel, range_str):
    """Python-native fallback: generate glyphs with metrics but no SDF bitmap."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None

    ttf_path = os.path.join(BASE_DIR, ttf_rel)
    if not os.path.exists(ttf_path):
        return None

    try:
        font = TTFont(ttf_path)
        cmap = font.getBestCmap()
        hmtx = font['hmtx']
        upem = font['head'].unitsPerEm

        glyphs = []
        for gid in sorted(glyph_ids):
            glyph_name = cmap.get(gid)
            if glyph_name is None:
                continue

            advance = hmtx.metrics.get(glyph_name, (0, 0))[0]
            advance_norm = max(1, round(advance * 24 / upem))

            raw = bytearray()
            raw += _varint((1 << 3) | 0) + _varint(gid)
            raw += _varint((2 << 3) | 0) + _varint(24)
            raw += _varint((3 << 3) | 0) + _varint(24)
            raw += _varint((4 << 3) | 0) + _signed_varint(0)
            raw += _varint((5 << 3) | 0) + _signed_varint(0)
            raw += _varint((6 << 3) | 0) + _varint(advance_norm)
            glyphs.append((gid, bytes(raw)))

        # Try to get fontstack name from TTF (before closing)
        try:
            name_table = font['name']
            for record in name_table.names:
                if record.nameID == 1 and record.platformID == 3:
                    fontstack = record.toUnicode()
                    break
            else:
                fontstack = os.path.splitext(os.path.basename(ttf_rel))[0]
        except Exception:
            fontstack = os.path.splitext(os.path.basename(ttf_rel))[0]

        font.close()

        return _build_glyphs_pbf(fontstack, range_str, glyphs)
    except Exception:
        return None


def _get_glyphs_for_font(font_config, start, end, range_name, range_str):
    if not font_config:
        return None

    pbf_dir = font_config.get('pbf_dir')
    if pbf_dir:
        data = _serve_static_pbf(pbf_dir, range_name)
        if data:
            return data

    ttf = font_config.get('ttf')
    if ttf:
        data = _generate_via_fontnik(ttf, start, end)
        if data and len(data) > 50:
            return data

        data = _generate_fallback_glyphs(range(start, end + 1), ttf, range_str)
        if data:
            return data

    return None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class GlyphsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path in ('/', ''):
            self._send_json({
                'service': 'MapLibre Glyphs Service',
                'fonts': list(STACKS.keys()),
                'cache_dir': CACHE_DIR,
            })
            return

        if path.startswith('/fonts/'):
            parts = path.split('/')
            if len(parts) < 4:
                self._send_error(400, 'Invalid glyphs URL')
                return

            fontstack_raw = parts[2]
            range_name = parts[3]
            start, end = _parse_range(range_name)

            if start is None or end is None:
                self._send_error(400, f'Invalid range: {range_name}')
                return

            resolved_fonts = _resolve_font(fontstack_raw)

            if not resolved_fonts or not any(resolved_fonts):
                self._send_error(404, f'Unknown font stack: {fontstack_raw}')
                return

            range_str = f'{start}-{end}'
            pbf_results = []
            for font_config in resolved_fonts:
                if not font_config:
                    continue
                result = _get_glyphs_for_font(font_config, start, end, range_name, range_str)
                if result:
                    pbf_results.append(result)

            if not pbf_results:
                self._send_error(404, f'No glyphs found for {fontstack_raw} range {range_name}')
                return

            if len(pbf_results) == 1:
                data = pbf_results[0]
            else:
                # Multiple fonts in stack - merge by glyph ID
                stack_name = urllib.parse.unquote(fontstack_raw)
                data = _merge_glyph_pbfs(pbf_results, stack_name, range_str)

            if data is None:
                self._send_error(500, 'Failed to generate glyphs')
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/x-protobuf')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_error(404, 'Not Found')

    def do_HEAD(self):
        self.do_GET()

    def _send_json(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, code, msg):
        self.send_response(code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(msg.encode())

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT

    print(f"MapLibre Glyphs Service")
    print(f"Listening on http://localhost:{port}")
    print(f"Font stacks: {list(STACKS.keys())}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Node available: {shutil.which('node') is not None}")

    server = http.server.ThreadingHTTPServer(('0.0.0.0', port), GlyphsHandler)
    server.daemon_threads = True

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
