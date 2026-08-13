#!/usr/bin/env python3
"""
Generate MapLibre Glyph PBF files from a TTF font using pure Python.

Renders at 4x, downsamples coverage, then encodes a real SDF matching
TinySDF / fontnik (cutoff=0.25, radius=8, edge ~192). Bitmap padding stays
3px (MapLibre glyph atlas convention). Variable fonts are pinned to Regular
(wght=400); Noto Sans SC VF defaults to Thin otherwise.

Usage:
    python3 fonts/generate_pbf.py NotoSansSC.ttf fonts/data
    python3 fonts/generate_pbf.py NotoSansSC.ttf fonts/data 0-255
    python3 fonts/generate_pbf.py NotoSansSC.ttf fonts/data 0-255 256-511
"""

import math
import os
import sys
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np

FONT_SIZE = 24
SDF_BORDER = 3          # PBF bitmap padding; MapLibre atlas convention
SDF_RADIUS = 8          # TinySDF / fontnik distance encoding range
SDF_CUTOFF = 0.25
VARIATION_NAME = "Regular"
SCALE = 4
RENDER_SIZE = FONT_SIZE * SCALE  # 96px; must match 24px metrics * SCALE
INF = 1e20


# ---------------------------------------------------------------------------
# Protobuf wire format helpers
# ---------------------------------------------------------------------------

def _varint(n):
    buf = bytearray()
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n & 0x7F)
    return bytes(buf)


def _signed_varint(n):
    return _varint((n << 1) ^ (n >> 31))


def _build_glyph_pbf(gid, bitmap, w, h, left, top, advance):
    """Single glyph in protobuf wire format."""
    buf = bytearray()
    buf += b'\x08' + _varint(gid)           # 1: id
    if bitmap and len(bitmap) > 0:
        buf += b'\x12' + _varint(len(bitmap)) + bitmap  # 2: SDF bitmap
    buf += b'\x18' + _varint(w)             # 3: rendered width (no border)
    buf += b'\x20' + _varint(h)             # 4: rendered height (no border)
    buf += b'\x28' + _signed_varint(left)   # 5: left (signed)
    buf += b'\x30' + _signed_varint(top)    # 6: top (signed)
    buf += b'\x38' + _varint(advance)       # 7: advance
    return bytes(buf)


def _build_range_pbf(fontstack_name, range_str, glyphs):
    """Full range PBF: outer glyphs { stacks } / inner fontstack { name, range, glyphs }."""
    body = bytearray()
    nb = fontstack_name.encode('utf-8')
    body += b'\x0a' + _varint(len(nb)) + nb          # 1: name
    rb = range_str.encode('utf-8')
    body += b'\x12' + _varint(len(rb)) + rb          # 2: range
    for gid, bmp, w, h, l, t, adv in glyphs:
        gp = _build_glyph_pbf(gid, bmp, w, h, l, t, adv)
        body += b'\x1a' + _varint(len(gp)) + gp      # 3: glyphs
    return b'\x0a' + _varint(len(body)) + bytes(body)


# ---------------------------------------------------------------------------
# TinySDF-compatible squared Euclidean distance transform
# ---------------------------------------------------------------------------

def _edt_1d(grid, offset, stride, length, f, v, z):
    v[0] = 0
    z[0] = -INF
    z[1] = INF
    f[0] = grid[offset]
    k = 0
    for q in range(1, length):
        f[q] = grid[offset + q * stride]
        q2 = q * q
        while True:
            r = int(v[k])
            s = (f[q] - f[r] + q2 - r * r) / (2.0 * (q - r))
            if s > z[k]:
                break
            k -= 1
            if k < 0:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    k = 0
    for q in range(length):
        while z[k + 1] < q:
            k += 1
        r = int(v[k])
        qr = q - r
        grid[offset + q * stride] = f[r] + qr * qr


def _edt_2d(grid):
    """In-place squared Euclidean distance transform on a C-contiguous 2D array."""
    h, w = grid.shape
    flat = grid.ravel()
    max_len = max(h, w)
    f = np.empty(max_len, dtype=np.float64)
    v = np.empty(max_len, dtype=np.int32)
    z = np.empty(max_len + 1, dtype=np.float64)
    for x in range(w):
        _edt_1d(flat, x, w, h, f, v, z)
    for y in range(h):
        _edt_1d(flat, y * w, 1, w, f, v, z)


def _coverage_to_sdf(coverage):
    """Encode TinySDF / fontnik SDF. Edge ~192 (cutoff=0.25), radius=SDF_RADIUS."""
    alpha = coverage.astype(np.float64) / 255.0
    grid_outer = np.where(
        alpha >= 1.0, 0.0,
        np.where(alpha <= 0.0, INF, np.maximum(0.0, 0.5 - alpha) ** 2),
    ).astype(np.float64, copy=False)
    grid_inner = np.where(
        alpha >= 1.0, INF,
        np.where(alpha <= 0.0, 0.0, np.maximum(0.0, alpha - 0.5) ** 2),
    ).astype(np.float64, copy=False)
    _edt_2d(grid_outer)
    _edt_2d(grid_inner)
    dist = np.sqrt(grid_outer) - np.sqrt(grid_inner)
    encoded = np.clip(
        np.round(255.0 - 255.0 * (dist / SDF_RADIUS + SDF_CUTOFF)),
        0, 255,
    ).astype(np.uint8)
    return encoded


def _downsample_coverage(hi_res, out_h, out_w):
    """Area-average SCALE x SCALE blocks to target coverage."""
    need_h = out_h * SCALE
    need_w = out_w * SCALE
    canvas = np.zeros((need_h, need_w), dtype=np.float64)
    h = min(hi_res.shape[0], need_h)
    w = min(hi_res.shape[1], need_w)
    canvas[:h, :w] = hi_res[:h, :w]
    return canvas.reshape(out_h, SCALE, out_w, SCALE).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Glyph raster + metrics
# ---------------------------------------------------------------------------

def _metrics_24(font24, char):
    """Baseline-relative metrics at 24px (Pillow anchor='ls')."""
    draw = ImageDraw.Draw(Image.new('L', (1, 1)))
    bbox = draw.textbbox((0, 0), char, font=font24, anchor='ls')
    advance = max(1, int(round(font24.getlength(char))))
    left = int(math.floor(bbox[0]))
    right = int(math.ceil(bbox[2]))
    top_y = int(math.floor(bbox[1]))
    bot_y = int(math.ceil(bbox[3]))
    width = max(0, right - left)
    height = max(0, bot_y - top_y)
    top = -top_y
    return width, height, left, top, advance


def render_glyph_sdf(pil_hi, font24, codepoint):
    """Render one glyph to an SDF bitmap.

    Returns (sdf_bytes, width, height, left, top, advance).
    width/height exclude the 3px SDF border; bitmap includes it.
    """
    char = chr(codepoint)
    width, height, left, top, advance = _metrics_24(font24, char)

    if width <= 0 or height <= 0:
        return None, 0, 0, 0, 0, advance

    out_w = width + SDF_BORDER * 2
    out_h = height + SDF_BORDER * 2
    hr_w = out_w * SCALE
    hr_h = out_h * SCALE

    canvas = Image.new('L', (hr_w, hr_h), 0)
    draw = ImageDraw.Draw(canvas)
    tx = (SDF_BORDER - left) * SCALE
    ty = (SDF_BORDER + top) * SCALE
    draw.text((tx, ty), char, font=pil_hi, fill=255, anchor='ls')

    coverage = _downsample_coverage(np.array(canvas, dtype=np.uint8), out_h, out_w)
    sdf = _coverage_to_sdf(coverage)
    return sdf.tobytes(), width, height, left, top, advance


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _apply_regular_instance(font):
    """Pin a variable font to Regular (wght=400). Default VF instance may be Thin."""
    try:
        names = font.get_variation_names()
    except Exception:
        return None
    if not names:
        return None
    for name in names:
        label = name.decode("utf-8", errors="replace") if isinstance(name, (bytes, bytearray)) else str(name)
        if label == VARIATION_NAME:
            font.set_variation_by_name(name)
            return VARIATION_NAME
    try:
        font.set_variation_by_axes([400])
        return "wght=400"
    except Exception:
        return None


def get_font_name(ttf_path):
    """Extract font name from TTF (family + style)."""
    font = TTFont(ttf_path)
    name = None
    try:
        family = style = ''
        for r in font['name'].names:
            if r.nameID == 1 and r.platformID == 3:
                family = r.toUnicode()
            if r.nameID == 2 and r.platformID == 3:
                style = r.toUnicode()
        name = family
        if style:
            name += ' ' + style
        if not name:
            for r in font['name'].names:
                if r.nameID == 1:
                    name = r.toUnicode()
                    break
        if not name:
            name = os.path.splitext(os.path.basename(ttf_path))[0]
    except Exception:
        name = os.path.splitext(os.path.basename(ttf_path))[0]
    font.close()
    return name


def _parse_range_specs(specs):
    """Parse CLI range args like '0-255' or '256' into (start, end) inclusive."""
    ranges = []
    for spec in specs:
        if '-' in spec:
            a, b = spec.split('-', 1)
            ranges.append((int(a), int(b)))
        else:
            start = int(spec)
            ranges.append((start, start + 255))
    return ranges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_pbf(ttf_path, output_dir, range_specs=None):
    os.makedirs(output_dir, exist_ok=True)

    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    all_codepoints = sorted(set(cmap.keys()))
    fontstack_name = get_font_name(ttf_path)
    font.close()

    pil_hi = ImageFont.truetype(ttf_path, RENDER_SIZE)
    font24 = ImageFont.truetype(ttf_path, FONT_SIZE)
    variation = _apply_regular_instance(pil_hi)
    _apply_regular_instance(font24)

    range_glyphs = {}
    for cp in all_codepoints:
        r = (cp // 256) * 256
        range_glyphs.setdefault(r, []).append(cp)

    if range_specs:
        wanted = set()
        for start, end in range_specs:
            wanted.add((start // 256) * 256)
            wanted.add((end // 256) * 256)
        range_starts = sorted(wanted)
    else:
        max_cp = max(all_codepoints) if all_codepoints else 0
        range_starts = list(range(0, (max_cp // 256) * 256 + 1, 256))

    total_ranges = len(range_starts)
    print(f"Font: {fontstack_name}")
    if variation:
        print(f"Variation: {variation}")
    print(f"Total codepoints: {len(all_codepoints)}")
    print(f"Generating {total_ranges} range(s)")
    print()

    for idx, rs in enumerate(range_starts, 1):
        re = rs + 255
        range_str = f"{rs}-{re}"
        fname = f"{rs}-{re}.pbf"
        out_path = os.path.join(output_dir, fname)

        if range_specs:
            lo = min(s for s, _ in range_specs)
            hi = max(e for _, e in range_specs)
            cps = [cp for cp in range_glyphs.get(rs, []) if lo <= cp <= hi]
        else:
            cps = sorted(range_glyphs.get(rs, []))

        glyphs_data = []
        for cp in cps:
            bmp, rw, rh, lf, tp, adv = render_glyph_sdf(pil_hi, font24, cp)
            glyphs_data.append((cp, bmp, rw, rh, lf, tp, adv))

        pbf = _build_range_pbf(fontstack_name, range_str, glyphs_data)
        with open(out_path, 'wb') as f:
            f.write(pbf)

        if idx == 1 or idx == total_ranges or idx % 40 == 0 or total_ranges <= 8:
            print(f"  [{idx}/{total_ranges}] {fname}: {len(glyphs_data)} glyphs, {len(pbf):,}B")

    print(f"  [{total_ranges}/{total_ranges}] done.")

    total_bytes = sum(os.path.getsize(os.path.join(output_dir, f))
                      for f in os.listdir(output_dir) if f.endswith('.pbf'))
    total_files = sum(1 for f in os.listdir(output_dir) if f.endswith('.pbf'))
    print(f"\nDone! {total_files} PBF files, {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <ttf_path> <output_dir> [range ...]")
        print(f"Example: python3 {sys.argv[0]} NotoSansSC.ttf fonts/data")
        print(f"Example: python3 {sys.argv[0]} NotoSansSC.ttf fonts/data 0-255")
        sys.exit(1)
    specs = _parse_range_specs(sys.argv[3:]) if len(sys.argv) > 3 else None
    generate_all_pbf(sys.argv[1], sys.argv[2], specs)
