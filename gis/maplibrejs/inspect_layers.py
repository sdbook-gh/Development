import sqlite3
import json
import gzip
import mapbox_vector_tile
import sys

def check_label_layers(MBTILES_FILE):
    conn = sqlite3.connect(MBTILES_FILE)
    cursor = conn.cursor()

    # 1. 从元数据中查找
    cursor.execute("SELECT value FROM metadata WHERE name='json'")
    row = cursor.fetchone()
    if row:
        metadata_json = json.loads(row[0])
        print("元数据中定义的包含 'name' 属性的图层：")
        for layer in metadata_json.get('vector_layers', []):
            layer_id = layer.get('id')
            # 检查 attributes
            fields = layer.get('fields', {})
            if 'name' in fields:
                print(f"- {layer_id} (属性: {', '.join(fields.keys())})")
    
    print("\n--- 深度扫描瓦片要素 ---")
    
    # 2. 采样一些瓦片来确认
    cursor.execute("SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = 14 LIMIT 20")
    
    layer_names_with_labels = set()
    
    for z, x, y, data in cursor:
        try:
            if data.startswith(b'\x1f\x8b'):
                data = gzip.decompress(data)
            decoded = mapbox_vector_tile.decode(data)
            for layer_name, layer_data in decoded.items():
                for feature in layer_data['features']:
                    if 'name' in feature.get('properties', {}):
                        layer_names_with_labels.add(layer_name)
                        break
        except:
            continue
            
    print("实际瓦片中发现包含 'name' 属性的图层：")
    for name in sorted(layer_names_with_labels):
        print(f"- {name}")

    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 inspect_layers.py <mbtiles文件>")
    check_label_layers(sys.argv[1])
