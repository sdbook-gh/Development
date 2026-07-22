#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

def str_display_width(s):
    """计算字符串的显示宽度，中文字符算2个宽度"""
    width = 0
    for char in s:
        if ord(char) > 127:  # 简单判断非ASCII为中文字符（适用中文、日韩等）
            width += 2
        else:
            width += 1
    return width

def ljust_with_width(s, width):
    """左对齐字符串，考虑中文字符宽度"""
    current_width = str_display_width(s)
    if current_width >= width:
        return s
    return s + ' ' * (width - current_width)

def read_input():
    """从stdin读取输入，直到EOF（ctrl+d）"""
    lines = []
    try:
        for line in sys.stdin:
            lines.append(line.rstrip('\n'))
    except KeyboardInterrupt:
        pass
    return lines

def parse_data(lines):
    """
    解析输入数据，支持 TSV（制表符分隔）格式。
    第一行为表头，后续每行为一条记录。
    """
    if not lines:
        return [], []
    
    # 去除完全空行，但保留可能含空格的行
    lines = [line for line in lines if line.strip()]
    if not lines:
        return [], []
    
    first_line = lines[0]
    
    # 检查第一行是否包含制表符 → TSV 格式
    if '\t' in first_line:
        # 表头
        headers = [h.strip() for h in first_line.split('\t') if h.strip()]
        num_cols = len(headers)
        
        records = []
        for line in lines[1:]:
            if not line.strip():
                continue
            # 按制表符拆分，保留空字段（连续制表符会生成空字符串）
            parts = line.split('\t')
            # 取前 num_cols 个字段，不足补空，多余截断
            cells = [p.strip().strip('`') for p in parts[:num_cols]]
            while len(cells) < num_cols:
                cells.append('')
            records.append(cells)
        
        return headers, records
    
    else:
        # 非 TSV 格式时，尝试回退逻辑（简单处理，不保证通用）
        # 此处保留原自动推断逻辑（但原逻辑有缺陷，仅作兜底）
        # 由于输入明确是 TSV，实际不会走到这里
        # 为保持完整性，返回空
        return [], []

def create_table(headers, records):
    """创建对齐的文本表格"""
    # 计算每列的最大显示宽度
    col_widths = [str_display_width(header) for header in headers]
    
    for record in records:
        for i, cell in enumerate(record):
            cell_width = str_display_width(cell)
            if cell_width > col_widths[i]:
                col_widths[i] = cell_width
    
    # 添加额外的列间距（可调整）
    col_widths = [w + 4 for w in col_widths]  # 原为+6，稍微减小
    
    # 构建表格
    table = []
    
    # 表头
    header_line = ''.join(ljust_with_width(header, col_widths[i]) for i, header in enumerate(headers))
    table.append(header_line)
    
    # 分隔线
    separator_line = ''.join('-' * col_widths[i] for i in range(len(headers)))
    table.append(separator_line)
    
    # 数据行
    for record in records:
        data_line = ''.join(ljust_with_width(cell, col_widths[i]) for i, cell in enumerate(record))
        table.append(data_line)
    
    return table

def main():
    lines = read_input()
    if not lines:
        print("没有输入数据", file=sys.stderr)
        return
    
    headers, records = parse_data(lines)
    if not headers or not records:
        print("无法解析表头或数据，请确保输入为制表符分隔的TSV格式", file=sys.stderr)
        return
    
    table = create_table(headers, records)
    
    # 输出到屏幕
    for line in table:
        print(line)
    
    # 同时输出到文件
    with open('output.txt', 'w', encoding='utf-8') as f:
        for line in table:
            f.write(line + '\n')
    print("\n表格已保存到 output.txt")

if __name__ == "__main__":
    main()
