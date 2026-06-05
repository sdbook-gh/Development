# map data
https://download.geofabrik.de/asia/china/beijing-260531-free.shp.zip
# convert shp to geojson
  #! /bin/bash
  mkdir -p geojson
  for f in *.shp; do
    ogr2ogr -f GeoJSON \
      -t_srs EPSG:4326 \
      geojson/${f%.shp}.geojson \
      $f
  done

# build mbtiles from geojson
GEO=/mnt/e/deve/terminal/map/beijing-260531-free.shp/geojson
TIP=/mnt/e/deve/terminal/map/tippecanoe/tippecanoe
OUT=beijing-260531.mbtiles

# build mbtiles
"$TIP" -f \
  -o "$OUT" \
  -z14 -Z6 \
  -n beijing \
  -N "beijing" \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  -L'{"file":"'"$GEO"'/gis_osm_adminareas_a_free_1.geojson","layer":"admin"}' \
  -L'{"file":"'"$GEO"'/gis_osm_natural_a_free_1.geojson","layer":"natural_a"}' \
  -L'{"file":"'"$GEO"'/gis_osm_protected_areas_a_free_1.geojson","layer":"protected"}' \
  -L'{"file":"'"$GEO"'/gis_osm_landuse_a_free_1.geojson","layer":"landuse"}' \
  -L'{"file":"'"$GEO"'/gis_osm_water_a_free_1.geojson","layer":"water"}' \
  -L'{"file":"'"$GEO"'/gis_osm_waterways_free_1.geojson","layer":"waterways"}' \
  -L'{"file":"'"$GEO"'/gis_osm_roads_free_1.geojson","layer":"roads"}' \
  -L'{"file":"'"$GEO"'/gis_osm_railways_free_1.geojson","layer":"railways"}' \
  -L'{"file":"'"$GEO"'/gis_osm_buildings_a_free_1.geojson","layer":"buildings"}' \
  -L'{"file":"'"$GEO"'/gis_osm_transport_a_free_1.geojson","layer":"transport_a"}' \
  -L'{"file":"'"$GEO"'/gis_osm_transport_free_1.geojson","layer":"transport"}' \
  -L'{"file":"'"$GEO"'/gis_osm_pois_a_free_1.geojson","layer":"pois_a"}' \
  -L'{"file":"'"$GEO"'/gis_osm_pois_free_1.geojson","layer":"pois"}' \
  -L'{"file":"'"$GEO"'/gis_osm_places_free_1.geojson","layer":"places"}' \
  -L'{"file":"'"$GEO"'/gis_osm_places_a_free_1.geojson","layer":"places_a"}' \
  -L'{"file":"'"$GEO"'/gis_osm_traffic_a_free_1.geojson","layer":"traffic_a"}' \
  -L'{"file":"'"$GEO"'/gis_osm_traffic_free_1.geojson","layer":"traffic"}'
参数说明
基本参数
参数	含义
-f / --force
若输出文件已存在则覆盖，否则报错退出
-o "$OUT"
输出 MBTiles 文件路径（SQLite 格式的矢量瓦片包）
-z14
最大缩放级别 14（建筑、小路等细节在 z14 可见）
-Z6
最小缩放级别 6（z0–5 不生成瓦片，减小体积；城市地图通常从 z6–8 起即可）
-n beijing
写入 MBTiles metadata 表的 name 字段
-N "..."
写入 metadata 表的 description 字段
抽稀 / 体积控制
参数	含义
--drop-densest-as-needed
单个瓦片超过约 500KB 限制时，优先丢弃最密集区域的要素，避免生成失败
--extend-zooms-if-still-dropping
若在最大 zoom 仍装不下，自动提高最大 zoom 再试，减少高 zoom 丢数据
北京建筑、道路数据量大，这两个参数建议保留。

# start web server
python3 -m http.server 8088
# start tiles server
python3 serve_mbtiles.py beijing-260531.mbtiles 3000
# start maptiler/tileserver-gl server
docker compose up -d --force-recreate
docker compose ps
