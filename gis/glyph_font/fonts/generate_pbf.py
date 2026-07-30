#!/usr/bin/env python3
"""
Generate MapLibre Glyph PBF files from a TTF font using pure Python.

Uses TinySDF-style approach: render at 4x with anti-aliasing, 
subsample to target size. Fast and produces good SDF-like results.

Usage:
    python3 fonts/generate_pbf.py NotoSansSC.ttf "fonts/Klokantech Noto Sans Regular"
"""

import os
import sys
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np

FONT_SIZE = 24
SDF_BORDER = 3
SCALE = 4                     # TinySDF upscale factor
RENDER_SIZE = (FONT_SIZE + SDF_BORDER * 2) * SCALE  # 120px


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
    buf += b'\x18' + _varint(w)             # 3: rendered width
    buf += b'\x20' + _varint(h)             # 4: rendered height
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
# TinySDF – render hi-res anti-aliased, subsample to target
# ---------------------------------------------------------------------------

def _make_sdf_24(hi_res_arr, hi_res_w, hi_res_h, scale=SCALE):
    """Convert hi-res anti-aliased rendering to SDF via subsampling.
    
    Adapted from MapLibre TinySDF approach:
    - The hi-res image is rendered at (target_size + 2*border) * scale
    - Each output pixel takes the center pixel from the corresponding scale×scale block
    - Resulting SDF values are 0–255 where ~128 = edge
    """
    # The hi_res_arr covers the glyph with border at hi-res
    # Subsample: pick every `scale`-th pixel
    # Align to center of blocks
    h_aligned = (hi_res_h // scale) * scale
    w_aligned = (hi_res_w // scale) * scale
    if h_aligned == 0 or w_aligned == 0:
        return None
    
    # Take center pixel of each block
    # Center offset within each block: scale // 2
    off = scale // 2
    sampled = hi_res_arr[off:h_aligned:scale, off:w_aligned:scale].astype(np.uint8)
    
    # Map anti-aliased 0-255 to SDF-like 0-255
    # 128 originally = edge of glyph in anti-aliased rendering
    # We keep it as-is; the TinySDF paper uses the raw coverage as SDF
    return sampled


def render_glyph_tinysdf(pil_font, font24, codepoint):
    """Render glyph using TinySDF approach.
    
    Returns (sdf_24bitmap, rendered_w, rendered_h, left, top, advance)
    All metrics at FONT_SIZE (24px) resolution.
    """
    char = chr(codepoint)
    
    # Get metrics at target size
    bbox24 = ImageDraw.Draw(Image.new('L', (1,1))).textbbox((0, 0), char, font=font24)
    advance24 = round(font24.getlength(char))
    ascent24 = font24.getmetrics()[0]
    
    bw = bbox24[2] - bbox24[0]
    bh = bbox24[3] - bbox24[1]
    left24 = bbox24[0]
    
    if bw <= 0 or bh <= 0:
        return None, 0, 0, 0, -FONT_SIZE, max(1, advance24)
    
    # Render at hi-res for TinySDF
    # Need to render a region covering the glyph + SDF_BORDER at SCALE resolution
    hr_w = (bw + SDF_BORDER * 2) * SCALE
    hr_h = (bh + SDF_BORDER * 2) * SCALE
    
    canvas = Image.new('L', (hr_w, hr_h), 0)
    draw = ImageDraw.Draw(canvas)
    
    # Position: the ascender point should be at (border*scale, 0) in hi-res coords
    # In Pillow: glyph at (0,0) has ascender at (0,0)
    # We want: ascender at (SDF_BORDER * SCALE, 0) on canvas
    # The bbox is relative to (0,0)
    # So: text position = (SDF_BORDER * SCALE - bbox24[0] * SCALE, -bbox24[1] * SCALE)
    tx = (SDF_BORDER * SCALE) - (bbox24[0] * SCALE)
    ty = (SDF_BORDER * SCALE) - (bbox24[1] * SCALE)
    
    draw.text((tx, ty), char, font=pil_font, fill=255)
    
    arr = np.array(canvas, dtype=np.uint8)
    
    # Subsample to get SDF
    sampled = _make_sdf_24(arr, hr_w, hr_h)
    if sampled is None:
        return None, 0, 0, 0, -FONT_SIZE, max(1, advance24)
    
    h_sdf, w_sdf = sampled.shape
    
    # The expected dimensions: (bh+6) x (bw+6) at 24px
    expected_h = bh + SDF_BORDER * 2
    expected_w = bw + SDF_BORDER * 2
    
    # Resize if needed (should be close)
    if h_sdf != expected_h or w_sdf != expected_w:
        # Pad or trim to match
        import warnings
        warnings.warn(f"Size mismatch: got {w_sdf}x{h_sdf}, expected {expected_w}x{expected_h} for U+{codepoint:04X}")
        # Create target-sized array
        result = np.zeros((expected_h, expected_w), dtype=np.uint8)
        rh = min(h_sdf, expected_h)
        rw = min(w_sdf, expected_w)
        result[:rh, :rw] = sampled[:rh, :rw]
        sampled = result
    
    bitmap_data = sampled.tobytes()
    
    # Metrics: rendered size WITHOUT border
    rendered_w = bw
    rendered_h = bh
    
    # left bearing at 24px
    left = left24
    
    # top: in Pillow y-down coords, bbox[1] is top of glyph from ascender
    # In FreeType y-up: top_of_glyph = ascent24 - bbox24[1]
    # fontnik top = top_of_glyph - ascent24 = -bbox24[1]
    top = -bbox24[1]
    
    return bitmap_data, rendered_w, rendered_h, left, top, max(1, advance24)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_pbf(ttf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    all_codepoints = sorted(set(cmap.keys()))
    fontstack_name = get_font_name(ttf_path)
    font.close()
    
    pil_font_120 = ImageFont.truetype(ttf_path, RENDER_SIZE)
    pil_font_120.path = ttf_path
    pil_font_24 = ImageFont.truetype(ttf_path, FONT_SIZE)
    
    max_cp = max(all_codepoints) if all_codepoints else 0
    max_range = (max_cp // 256) * 256
    
    # Build a lookup: which codepoints belong to which range
    range_glyphs = {}
    for cp in all_codepoints:
        r = (cp // 256) * 256
        range_glyphs.setdefault(r, []).append(cp)
    
    total_ranges = max_range // 256 + 1
    print(f"Font: {fontstack_name}")
    print(f"Total codepoints: {len(all_codepoints)}")
    print(f"Range: 0 to {max_range} ({total_ranges} ranges)")
    print()
    
    for i in range(0, max_range + 1, 256):
        rs = i
        re = min(i + 255, max_cp)
        range_str = f"{rs}-{re}"
        fname = f"{rs}-{re}.pbf"
        out_path = os.path.join(output_dir, fname)
        
        cps = sorted(range_glyphs.get(rs, []))
        
        glyphs_data = []
        for cp in cps:
            result = render_glyph_tinysdf(pil_font_120, pil_font_24, cp)
            if result:
                bmp, rw, rh, lf, tp, adv = result
                glyphs_data.append((cp, bmp, rw, rh, lf, tp, adv))
        
        pbf = _build_range_pbf(fontstack_name, range_str, glyphs_data)
        with open(out_path, 'wb') as f:
            f.write(pbf)
        
        if (i // 256 + 1) % 40 == 0 or i == 0:
            n = len(glyphs_data)
            print(f"  [{i//256+1}/{total_ranges}] {fname}: {n} glyphs, {len(pbf):,}B")
    
    print(f"  [{total_ranges}/{total_ranges}] done.")
    
    total_bytes = sum(os.path.getsize(os.path.join(output_dir, f))
                      for f in os.listdir(output_dir) if f.endswith('.pbf'))
    total_files = sum(1 for f in os.listdir(output_dir) if f.endswith('.pbf'))
    print(f"\nDone! {total_files} PBF files, {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <ttf_path> <output_dir>")
        print(f"Example: python3 {sys.argv[0]} NotoSansSC.ttf 'fonts/Klokantech Noto Sans Regular'")
        sys.exit(1)
    generate_all_pbf(sys.argv[1], sys.argv[2])