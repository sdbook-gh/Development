#!/usr/bin/env python3
"""
依据 style_localmap.json 生成 style_localmap.html，用于测试 MapLibre 渲染效果。
瓦片源、字体设置参考 index.html。
"""

import json
import os

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_JSON = os.path.join(SCRIPT_DIR, "style_localmap.json")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "style_localmap.html")

# ── 从 index.html 提取的配置 ──────────────────────────
TILES_URL = "http://127.0.0.1:3000/tiles/{z}/{x}/{y}"
GLYPHS_URL = "http://localhost:8080/fonts/{fontstack}/{range}.pbf"
SPRITE_URL = "http://127.0.0.1:3000/sprites/ofm_f384/ofm"
# 必须与 glyph_font/fonts/font_config.json 的 stacks key 一致
FONT_OLD = "Noto Sans Regular"
FONT_NEW = "Noto Sans Regular"
LOCAL_FONTS = "['Noto Sans SC', 'Microsoft YaHei', 'sans-serif']"
CENTER_LNG = 116.4074
CENTER_LAT = 39.9042
ZOOM = 10
MAPLIBRE_VERSION = "4.7.1"


def patch_style(style: dict) -> dict:
    """修补样式 JSON：瓦片 URL、glyphs URL、字体名称。"""
    # 1. 瓦片源
    for src in style.get("sources", {}).values():
        if src.get("type") == "vector" and src.get("tiles"):
            src["tiles"] = [TILES_URL]

    # 2. glyphs / sprite
    style["glyphs"] = GLYPHS_URL
    style["sprite"] = SPRITE_URL

    # 3. 字体名称替换
    for layer in style.get("layers", []):
        layout = layer.get("layout")
        if layout and "text-font" in layout:
            layout["text-font"] = [
                FONT_NEW if f == FONT_OLD else f for f in layout["text-font"]
            ]

    return style


def build_html(style_json_str: str) -> str:
    """生成完整的 HTML 字符串。"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LocalMap Style 渲染测试</title>
    <script src="https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
        body {{ margin: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}

        /* Zoom 指示器 */
        #zoom {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(255,255,255,0.9);
            padding: 6px 8px; border-radius: 4px;
            font-family: Arial, sans-serif; font-size: 13px;
            z-index: 2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        }}

        /* 鼠标实时坐标 */
        #mouse-coords {{
            position: absolute; bottom: 10px; right: 10px;
            background: rgba(255,255,255,0.9);
            padding: 4px 8px; border-radius: 4px;
            font-family: monospace; font-size: 11px;
            z-index: 2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            pointer-events: none;
        }}

        /* 点击坐标面板 */
        #coords-panel {{
            position: absolute; bottom: 30px; left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px; border-radius: 4px;
            font-family: monospace; font-size: 12px;
            z-index: 2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            min-width: 200px; display: none;
        }}
        #coords-panel h4 {{
            margin: 0 0 8px 0; font-size: 13px; color: #333;
            border-bottom: 1px solid #ddd; padding-bottom: 4px;
        }}
        #coords-panel .coord-row {{
            margin: 4px 0; display: flex; justify-content: space-between;
        }}
        #coords-panel .coord-label {{ color: #666; font-weight: bold; }}
        #coords-panel .coord-value {{ color: #0078ff; }}
        #coords-panel .close-btn {{
            position: absolute; top: 5px; right: 8px;
            cursor: pointer; color: #999; font-size: 14px;
        }}
        #coords-panel .close-btn:hover {{ color: #333; }}

        /* 图层列表面板 */
        #layers-panel {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px; border-radius: 4px;
            font-family: Arial, sans-serif; font-size: 12px;
            z-index: 2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            max-height: 80vh; overflow-y: auto; min-width: 180px;
        }}
        #layers-panel h4 {{
            margin: 0 0 6px 0; font-size: 13px; color: #333;
        }}
        .layer-item {{
            display: flex; align-items: center; gap: 4px;
            margin: 2px 0; cursor: pointer;
        }}
        .layer-item input {{ margin: 0; }}
        .layer-item label {{ cursor: pointer; }}
        .layer-toolbar {{
            display: flex; gap: 6px; margin-bottom: 6px;
            padding-bottom: 6px; border-bottom: 1px solid #ddd;
        }}
        .layer-toolbar button {{
            padding: 3px 10px; border: 1px solid #ccc;
            border-radius: 3px; background: #fff; cursor: pointer;
            font-size: 12px; flex: 1;
        }}
        .layer-toolbar button:hover {{ background: #f0f0f0; }}
    </style>
</head>

<body>
    <div id="map"></div>

    <div id="zoom">Zoom: {ZOOM:.2f}</div>
    <div id="mouse-coords">Lng: -, Lat: -</div>

    <div id="coords-panel">
        <span class="close-btn" onclick="document.getElementById('coords-panel').style.display='none'">&times;</span>
        <h4>📍 坐标信息</h4>
        <div class="coord-row"><span class="coord-label">经度:</span><span class="coord-value" id="coord-lng">-</span></div>
        <div class="coord-row"><span class="coord-label">纬度:</span><span class="coord-value" id="coord-lat">-</span></div>
        <div class="coord-row"><span class="coord-label">缩放级别:</span><span class="coord-value" id="coord-zoom">-</span></div>
        <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
        <div class="coord-row"><span class="coord-label">瓦片 X:</span><span class="coord-value" id="coord-tile-x">-</span></div>
        <div class="coord-row"><span class="coord-label">瓦片 Y:</span><span class="coord-value" id="coord-tile-y">-</span></div>
        <div class="coord-row"><span class="coord-label">瓦片 Z:</span><span class="coord-value" id="coord-tile-z">-</span></div>
    </div>

    <div id="layers-panel">
        <h4>图层控制</h4>
        <div class="layer-toolbar">
            <button id="btn-select-all">全选</button>
            <button id="btn-deselect-all">全不选</button>
        </div>
        <div id="layers-list"></div>
    </div>

    <script>
        // ── 内嵌样式（由 Python 脚本从 style_localmap.json 修补生成）──
        const styleSpec = {style_json_str};

        const map = new maplibregl.Map({{
            container: 'map',
            localIdeographFontFamily: {LOCAL_FONTS},
            style: styleSpec,
            center: [{CENTER_LNG}, {CENTER_LAT}],
            zoom: {ZOOM}
        }});

        // ── 控件 ──
        map.addControl(new maplibregl.NavigationControl(), 'top-left');
        map.addControl(new maplibregl.ScaleControl({{ maxWidth: 200, unit: 'metric' }}), 'bottom-right');

        // ── Zoom 指示器 ──
        const zoomEl = document.getElementById('zoom');
        map.on('move', () => {{
            zoomEl.textContent = 'Zoom: ' + map.getZoom().toFixed(2);
        }});

        // ── 鼠标实时坐标 ──
        const mouseCoordsEl = document.getElementById('mouse-coords');
        function lonLatToTile(lon, lat, zoom) {{
            const n = Math.pow(2, zoom);
            const x = Math.floor((lon + 180) / 360 * n);
            const y = Math.floor((1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n);
            return {{ x, y }};
        }}
        map.on('mousemove', (e) => {{
            const z = Math.floor(map.getZoom());
            const tile = lonLatToTile(e.lngLat.lng, e.lngLat.lat, z);
            mouseCoordsEl.textContent = `Lng: ${{e.lngLat.lng.toFixed(4)}}, Lat: ${{e.lngLat.lat.toFixed(4)}} | Tile: ${{tile.x}}/${{tile.y}}/${{z}}`;
        }});

        // ── 点击坐标面板 ──
        let clickMarker = null;
        map.on('click', (e) => {{
            const z = Math.floor(map.getZoom());
            const tile = lonLatToTile(e.lngLat.lng, e.lngLat.lat, z);
            document.getElementById('coord-lng').textContent = e.lngLat.lng.toFixed(6);
            document.getElementById('coord-lat').textContent = e.lngLat.lat.toFixed(6);
            document.getElementById('coord-zoom').textContent = map.getZoom().toFixed(2);
            document.getElementById('coord-tile-x').textContent = tile.x;
            document.getElementById('coord-tile-y').textContent = tile.y;
            document.getElementById('coord-tile-z').textContent = z;
            document.getElementById('coords-panel').style.display = 'block';

            if (clickMarker) clickMarker.remove();
            clickMarker = new maplibregl.Marker({{ color: '#ff0000' }})
                .setLngLat(e.lngLat)
                .addTo(map);
        }});

        // ── 图层控制面板 ──
        function setLayerVisibility(layerId, visible) {{
            map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
        }}

        map.on('load', () => {{
            const layersList = document.getElementById('layers-list');
            const layerIds = [];
            styleSpec.layers.forEach(layer => {{
                layerIds.push(layer.id);
                const div = document.createElement('div');
                div.className = 'layer-item';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.id = 'cb-' + layer.id;
                cb.dataset.layerId = layer.id;
                const lbl = document.createElement('label');
                lbl.htmlFor = 'cb-' + layer.id;
                lbl.textContent = layer.id;
                cb.addEventListener('change', () => {{
                    setLayerVisibility(layer.id, cb.checked);
                }});
                div.appendChild(cb);
                div.appendChild(lbl);
                layersList.appendChild(div);
            }});

            // 全选 / 全不选
            document.getElementById('btn-select-all').addEventListener('click', () => {{
                layerIds.forEach(id => {{
                    setLayerVisibility(id, true);
                    const cb = document.getElementById('cb-' + id);
                    if (cb) cb.checked = true;
                }});
            }});
            document.getElementById('btn-deselect-all').addEventListener('click', () => {{
                layerIds.forEach(id => {{
                    setLayerVisibility(id, false);
                    const cb = document.getElementById('cb-' + id);
                    if (cb) cb.checked = false;
                }});
            }});
        }});

        // ── 调试事件 ──
        map.on('load',  () => console.log('[map] load success'));
        map.on('error', (e) => console.error('[map] error:', e));
        map.on('data',  (e) => console.log('[map] data:', e.dataType, e.sourceId || ''));
    </script>
</body>

</html>"""


def main():
    # 1. 读取样式 JSON
    with open(STYLE_JSON, "r", encoding="utf-8") as f:
        style = json.load(f)

    # 2. 修补
    style = patch_style(style)

    # 3. 序列化为紧凑 JSON（嵌入 HTML）
    style_json_str = json.dumps(style, ensure_ascii=False, separators=(",", ":"))

    # 4. 生成 HTML
    html = build_html(style_json_str)

    # 5. 写出
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] 已生成: {OUTPUT_HTML}")
    print(f"     瓦片源: {TILES_URL}")
    print(f"     字体源: {GLYPHS_URL}")
    print(f"     图标源: {SPRITE_URL}")
    print(f"     字体名: {FONT_OLD} -> {FONT_NEW}")
    print(f"     图层数: {len(style['layers'])}")


if __name__ == "__main__":
    main()
