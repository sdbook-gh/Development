import sqlite3
import gzip
import math
import sys
import mapbox_vector_tile

def search_mbtiles(db_path, keyword):
    print(f"正在读取文件: {db_path}")
    print(f"正在全量解析要素并搜索关键词: '{keyword}'...")
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # 1. 获取最大缩放级别
        cursor.execute("SELECT MAX(zoom_level) FROM tiles")
        max_zoom = cursor.fetchone()[0]
        print(f"搜索最高级别: {max_zoom}")

        # 2. 扫描瓦片数据
        cursor.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?", 
            (max_zoom,)
        )

        found_results = []
        keyword_lower = keyword.lower()
        
        tile_count = 0
        for z, x, y_tms, data in cursor:
            tile_count += 1
            if tile_count % 100 == 0:
                print(f"已处理 {tile_count} 个瓦片...", end='\r')

            try:
                if data.startswith(b'\x1f\x8b'):
                    data = gzip.decompress(data)
                
                # 解码瓦片，设置 y_coord_down=True 以匹配 MVT 规范
                # 遵循新版 API 签名，使用 default_options 传递参数以消除警告
                decoded_tile = mapbox_vector_tile.decode(data, default_options={'y_coord_down': True})
                
                # 转换坐标所需参数
                # MBTiles 使用 TMS (y起于南)，MapLibre 使用 XYZ (y起于北)
                y_xyz = (1 << z) - 1 - y_tms
                n = 2.0 ** z
                
                for layer_name, layer_data in decoded_tile.items():
                    extent = layer_data.get('extent', 4096)
                    
                    for feature in layer_data['features']:
                        properties = feature.get('properties', {})
                        name = properties.get('name')
                        
                        if name and keyword_lower in name.lower():
                            # --- 直接从地图元素的 geometry 获取坐标 ---
                            geom = feature.get('geometry', {})
                            coords = geom.get('coordinates', [])
                            
                            # 提取一个具有代表性的点
                            if geom['type'] == 'Point':
                                px, py = coords
                            elif geom['type'] in ['LineString', 'MultiPoint']:
                                px, py = coords[0]
                            elif geom['type'] in ['Polygon', 'MultiLineString']:
                                px, py = coords[0][0]
                            elif geom['type'] == 'MultiPolygon':
                                px, py = coords[0][0][0]
                            else:
                                continue

                            # 将瓦片本地坐标 (px, py) 转换为全球经纬度
                            # 1. 经度计算
                            lon = (x + px / extent) / n * 360.0 - 180.0
                            # 2. 纬度计算 (Web Mercator 反投影)
                            y_frac = (y_xyz + py / extent) / n
                            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_frac)))
                            lat = math.degrees(lat_rad)

                            result = {
                                "name": name,
                                "layer": layer_name,
                                "lat": lat,
                                "lon": lon
                            }
                            if result not in found_results:
                                found_results.append(result)
                                
            except Exception:
                continue

        print(f"\n扫描完成，共处理 {tile_count} 个瓦片。")
        conn.close()

        # 3. 输出结果
        if not found_results:
            print(f"在本地 MBTiles 的要素属性中未找到包含 '{keyword}' 的元素。")
        else:
            print(f"\n找到 {len(found_results)} 个匹配元素：")
            print("-" * 80)
            print(f"{'地图元素名称':<35} | {'图层':<15} | {'精确坐标 (纬, 经)'}")
            print("-" * 80)
            found_results.sort(key=lambda x: x['name'])
            for res in found_results:
                print(f"{res['name']:<35} | {res['layer']:<15} | {res['lat']:>12.9f}, {res['lon']:>12.9f}")
            print("-" * 80)

    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 search_mbtiles.py <mbtiles文件> <关键词>")
    else:
        search_mbtiles(sys.argv[1], sys.argv[2])
