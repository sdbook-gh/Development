import sqlite3
import gzip
import math
import sys
import mapbox_vector_tile

def xyz_to_latlon(z, x, y_tms):
    """将 MBTiles 的 Z/X/Y (TMS) 转换为经纬度坐标"""
    y_xyz = (1 << z) - 1 - y_tms
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_xyz / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

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
        # 我们主要搜索 max_zoom 级别，因为信息最全
        cursor.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?", 
            (max_zoom,)
        )

        found_results = []
        keyword_lower = keyword.lower()
        
        tile_count = 0
        for z, x, y, data in cursor:
            tile_count += 1
            if tile_count % 100 == 0:
                print(f"已处理 {tile_count} 个瓦片...", end='\r')

            try:
                # 处理 PBF 瓦片的解压
                if data.startswith(b'\x1f\x8b'):
                    data = gzip.decompress(data)
                
                # --- 正确的矢量瓦片解析逻辑 ---
                # 使用 mapbox_vector_tile 库将二进制 PBF 解码为 Python 字典
                decoded_tile = mapbox_vector_tile.decode(data)
                
                # 遍历图层 (如 roads, buildings, water 等)
                for layer_name, layer_data in decoded_tile.items():
                    # 遍历图层中的每一个地图元素 (Feature)
                    for feature in layer_data['features']:
                        properties = feature.get('properties', {})
                        name = properties.get('name')
                        
                        # 使用 name 属性进行匹配
                        if name and keyword_lower in name.lower():
                            lat, lon = xyz_to_latlon(z, x, y)
                            result = {
                                "name": name,
                                "layer": layer_name,
                                "lat": lat,
                                "lon": lon
                            }
                            # 去重
                            if result not in found_results:
                                found_results.append(result)
                                
            except Exception as e:
                # 忽略损坏的瓦片
                continue

        print(f"\n扫描完成，共处理 {tile_count} 个瓦片。")
        conn.close()

        # 3. 输出结果
        if not found_results:
            print(f"在本地 MBTiles 的要素属性中未找到包含 '{keyword}' 的元素。")
        else:
            print(f"\n找到 {len(found_results)} 个匹配元素：")
            print("-" * 80)
            print(f"{'地图元素名称':<35} | {'图层':<15} | {'坐标 (纬, 经)'}")
            print("-" * 80)
            # 按名称排序
            found_results.sort(key=lambda x: x['name'])
            for res in found_results:
                print(f"{res['name']:<35} | {res['layer']:<15} | {res['lat']:>9.6f}, {res['lon']:>10.6f}")
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
