# MapLibre 离线地图调试（map_style）

本目录用于调试「MapLibre 离线地图」渲染流水线。整个系统由**三个独立服务/资产**组成，页面 `style_localmap.html` 只负责组装。

## 服务与端口总览

| 组件 | 端口 | 作用 | 来源/启动 |
|---|---|---|---|
| `tile_server.py` | **3000** | 提供矢量瓦片 `/tiles/{z}/{x}/{y}`、精灵图 `/sprites/ofm_f384/ofm.*` | 本项目 |
| serve_glyphs.py（Python） | **8080** | 提供字体 glyphs `/fonts/{fontstack}/{range}.pbf` | 见下「字体」 |
| `fontnik`/预生成 pbf | — | glyph 静态切片 | 见下「字体」 |
| `sprites/ofm_f384/` | — | 图标精灵图（png+json） | 复制自 Android assets |

## 一、启动瓦片服务器 tile_server.py

**必须把工作目录(cwd)设为数据目录**，因为代码里 `INDEX_DB_FILE="./tile_index.db"` 是相对 cwd 的，且 `sources` 表里的路径也相对它：

```bash
DATA=/mnt/e/deve/map/mbtiles_output/tile_index
mkdir -p /mnt/extdisk/map/app/map-offline-navigation/map_style/.pi/tmp
cd "$DATA_DIR" && \
nohup python3 /mnt/extdisk/map/app/map-offline-navigation/map_style/tile_server.py \
  > /mnt/extdisk/map/app/map-offline-navigation/map_style/.pi/tmp/tile_server.log 2>&1 &
```

- 健康检查：`curl localhost:3000/health` → 应看到 `"tiles_indexed": 294530`、`"sources": 3`。
- 瓦片抽查：`curl -o /tmp/t.pbf -w "%{http_code}\n" localhost:3000/tiles/10/844/388` → **200**（pbf，响应带 `Content-Encoding: gzip`）。
- **数据源**：真实索引 `/mnt/e/dev/map/mbtiles_output/tile_index/tile_index.db`（294530 瓦片，覆盖 z0–15），叠合 `../mbtiles/hebei-latest-free.mbtiles`(476MB) + 本目录 `overview.mbtiles`。
- ⚠️ **不要在空目录里启动**，否则会生成 0 字节 `tile_index.db`，导致所有瓦片 404。
- 停止：`pkill -f "python3 .*tile_server.py"`（注意 `pkill -f "tile_server.py"` 会误杀当前 shell）。

**历史修复（tile_server.py）**：`SQLiteConnectionPool._create_connection` 曾在只读连接上执行 `PRAGMA journal_mode=WAL`，导致每个瓦片查询抛 `attempt to write a readonly database`(500)。已删除该行，改用 `query_only=ON`（try/except 包裹）。

## 二、字体服务（8080）

字体服务是独立 Python 脚本，**需单独启动**：

```bash
cd /mnt/d/personal/github/Development/gis/glyph_font && python3 fonts/serve_glyphs.py
```

- 配置在 `fonts/font_config.json`；字体栈 `Klokantech Noto Sans Regular`，`pbf_dir=fonts/Klokantech Noto Sans Regular`（预生成静态 pbf，正常请求够用）。
- 页面样式里 `glyphs: http://localhost:8080/fonts/{fontstack}/{range}.pbf`。

## 三、精灵图

`sprites/ofm_f384/`

## 四、style_localmap.html 说明
- 内嵌整份 style 的 JSON（`styleSpec`），MapLibre 4.7.1 由 `unpkg` 加载。
- center `[116.4074, 39.9042]`（北京），zoom 10；已加一行 `window.map = map` 以便 `map_debug_test` 做 zoomRange 扫缩。
