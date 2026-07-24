# -*- coding: utf-8 -*-
import sqlite3
import json
import sys
def show_mbtiles_info(mbtiles_path):
  """显示 MBTiles 文件的矢量图层信息"""
  # 连接到mbtiles文件
  conn = sqlite3.connect(mbtiles_path)
  cursor = conn.cursor()
  # 获取json数据
  cursor.execute("SELECT value FROM metadata WHERE name = 'json';")
  row = cursor.fetchone()
  if row is None:
      print(f"错误: 在 {mbtiles_path} 中未找到 metadata json 数据")
      conn.close()
      return
  json_data = row[0]
  data = json.loads(json_data)
  # 打印标题
  print("=" * 100)
  print(f"{'矢量图层信息':^100}")
  print("=" * 100)
  print(f"{'图层ID':<25} {'最小缩放':<10} {'最大缩放':<10} {'字段数量':<10} {'字段列表':<45}")
  print("-" * 100)
  # 遍历所有图层
  for layer in data.get('vector_layers', []):
      layer_id = layer.get('id', '')
      minzoom = layer.get('minzoom', 0)
      maxzoom = layer.get('maxzoom', 0)
      fields = layer.get('fields', {})
      field_names = list(fields.keys())
      field_count = len(field_names)
      fields_display = ', '.join(field_names)
      print(f"{layer_id:<25} {minzoom:<10} {maxzoom:<10} {field_count:<10} {fields_display:<45}")
  print("-" * 100)
  print(f"\n总图层数: {len(data.get('vector_layers', []))}")
  conn.close()
def main():
  if len(sys.argv) < 2:
      print("用法: python3 mbtiles_info.py <mbtiles文件路径>")
      print("示例: python3 mbtiles_info.py /mnt/extdisk/map/china.mbtiles")
      sys.exit(1)
  mbtiles_path = sys.argv[1]
  show_mbtiles_info(mbtiles_path)
if __name__ == '__main__':
  main()
