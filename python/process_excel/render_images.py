# -*- coding: utf-8 -*-
"""
将词汇表渲染为多张适合 A4 打印的 JPEG 图片。
方案 A：横向 A4 + 3 块布局（序号/英文/中文 ×3），按序号横排填充。
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from extract_serial import extract_entries, SRC_XLSX, BASE_DIR

# ---- A4 / 渲染参数 ----
DPI = 300
MM2PX = DPI / 25.4
PAGE_W_MM, PAGE_H_MM = 297, 210          # 横向 A4
MARGIN_MM = 10
NUM_BLOCKS = 3
COL_NAMES = ("序号", "英文", "中文")

FONT_REG = "/mnt/c/Windows/Fonts/msyh.ttc"       # 微软雅黑
FONT_BOLD = "/mnt/c/Windows/Fonts/msyhbd.ttc"    # 微软雅黑 Bold
FONT_FALLBACKS = (FONT_REG, "/mnt/c/Windows/Fonts/simhei.ttf",
                  "/mnt/c/Windows/Fonts/simsun.ttc")

OUT_DIR = BASE_DIR / "images_a4"
QUALITY = 90


def load_font(path: str, size_px: int) -> ImageFont.FreeTypeFont:
    """加载字体，失败则依次尝试备选。"""
    try:
        return ImageFont.truetype(path, size_px)
    except Exception:
        for p in FONT_FALLBACKS:
            try:
                return ImageFont.truetype(p, size_px)
            except Exception:
                continue
        raise RuntimeError("无法加载任何中文字体")


def pt_to_px(pt: float) -> int:
    return int(round(pt * DPI / 72))


def measure_col_widths(font, entries):
    nums = [str(n) for n, _, _ in entries]
    ens = [e for _, e, _ in entries]
    zhs = [z for _, _, z in entries]
    return [
        max(font.getlength(s) for s in [COL_NAMES[0]] + nums),
        max((font.getlength(s) for s in [COL_NAMES[1]] + ens), default=0),
        max((font.getlength(s) for s in [COL_NAMES[2]] + zhs), default=0),
    ]


def fit_font(entries, usable_w_px: int) -> dict:
    """从大到小尝试字号，返回第一个使 3 块总宽 <= 可用宽 的布局。"""
    for half in range(18, 11, -1):        # 9.0 -> 6.0 pt
        pt = half / 2
        size_px = pt_to_px(pt)
        font = load_font(FONT_REG, size_px)
        base = measure_col_widths(font, entries)
        pad = size_px * 0.5
        col_w = [bw + 2 * pad for bw in base]
        block_w = sum(col_w)
        gap = pad
        total_w = block_w * NUM_BLOCKS + gap * (NUM_BLOCKS - 1)
        if total_w <= usable_w_px:
            return dict(pt=pt, font=font, font_bold=load_font(FONT_BOLD, size_px),
                        size_px=size_px, pad=pad, col_w=col_w, block_w=block_w,
                        gap=gap, total_w=total_w)
    raise RuntimeError("字号已降到 6pt 仍放不下，请减小边距或改用 2 块布局")


def render(entries, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_w = int(round(PAGE_W_MM * MM2PX))
    page_h = int(round(PAGE_H_MM * MM2PX))
    margin = int(round(MARGIN_MM * MM2PX))
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    lay = fit_font(entries, usable_w)
    font, font_bold = lay["font"], lay["font_bold"]
    col_w, block_w, gap, pad = lay["col_w"], lay["block_w"], lay["gap"], lay["pad"]
    row_h = int(round(lay["size_px"] * 1.5))
    header_h = row_h

    # 各块起始 x
    block_xs = []
    x = margin
    for b in range(NUM_BLOCKS):
        block_xs.append(x)
        x += block_w
        if b < NUM_BLOCKS - 1:
            x += gap
    table_right = block_xs[-1] + block_w
    col_off = [0, col_w[0], col_w[0] + col_w[1]]   # 块内 3 列的相对 x

    # 横排分组：每 NUM_BLOCKS 个条目组成一行（3 个块）
    grid = [entries[i:i + NUM_BLOCKS] for i in range(0, len(entries), NUM_BLOCKS)]

    rows_per_page = max(1, int((usable_h - header_h) // row_h))
    pages = [grid[i:i + rows_per_page] for i in range(0, len(grid), rows_per_page)]
    total = len(pages)

    light, header_bg = "#BFBFBF", "#D9E1F2"
    for pi, page_rows in enumerate(pages):
        img = Image.new("RGB", (page_w, page_h), "white")
        d = ImageDraw.Draw(img)
        y = margin

        # 表头
        for b in range(NUM_BLOCKS):
            bx = block_xs[b]
            for ci, name in enumerate(COL_NAMES):
                cx = bx + col_off[ci]
                d.rectangle([cx, y, cx + col_w[ci], y + header_h], fill=header_bg)
                d.text((cx + col_w[ci] / 2, y + header_h / 2), name,
                       font=font_bold, fill="black", anchor="mm")

        # 数据行
        yy = y + header_h
        for row_entries in page_rows:
            for b in range(NUM_BLOCKS):
                bx = block_xs[b]
                if b < len(row_entries):
                    num, en, zh = row_entries[b]
                    d.text((bx + col_w[0] / 2, yy + row_h / 2), str(num),
                           font=font, fill="black", anchor="mm")
                    d.text((bx + col_off[1] + pad, yy + row_h / 2), en,
                           font=font, fill="black", anchor="lm")
                    d.text((bx + col_off[2] + pad, yy + row_h / 2), zh,
                           font=font, fill="black", anchor="lm")
            yy += row_h

        # 网格线
        d.line([(margin, y), (table_right, y)], fill=light)                 # 顶
        d.line([(margin, y + header_h), (table_right, y + header_h)], fill=light)
        hy = y + header_h
        for _ in page_rows:
            hy += row_h
            d.line([(margin, hy), (table_right, hy)], fill=light)
        for b in range(NUM_BLOCKS):
            bx = block_xs[b]
            for xv in (bx, bx + col_w[0], bx + col_w[0] + col_w[1], bx + block_w):
                d.line([(xv, y), (xv, yy)], fill=light)
        d.rectangle([margin, y, table_right, yy], outline="black")          # 外框

        # 页码
        d.text((page_w / 2, page_h - margin / 2), f"第 {pi + 1} / {total} 页",
               font=font, fill="gray", anchor="mm")

        img.save(out_dir / f"p{pi + 1:02d}.jpg", "JPEG", quality=QUALITY)

    return dict(font_pt=lay["pt"], rows_per_page=rows_per_page, total_pages=total,
                total_rows=len(grid), out_dir=out_dir)


def main() -> None:
    if not SRC_XLSX.exists():
        raise FileNotFoundError(f"未找到源文件: {SRC_XLSX}")
    entries = extract_entries(SRC_XLSX)
    info = render(entries, OUT_DIR)
    print(f"渲染完成 ✅")
    print(f"  条目数 : {len(entries)}")
    print(f"  字号   : {info['font_pt']}pt")
    print(f"  每页行 : {info['rows_per_page']}")
    print(f"  总页数 : {info['total_pages']}")
    print(f"  输出   : {info['out_dir']}/p01.jpg ~ p{info['total_pages']:02d}.jpg")


if __name__ == "__main__":
    main()
