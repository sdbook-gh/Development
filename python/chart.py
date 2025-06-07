import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import pandas as pd
import numpy as np
from io import StringIO

# 设置中文字体支持 - 使用文泉驿微米黑
rcParams['font.family'] = 'WenQuanYi Micro Hei' # https://www.webfontfree.com/
rcParams['axes.unicode_minus'] = False  # 正常显示负号

csv_data = """
"2024年02月01日",10:09:35
"2024年02月02日",10:11:43
"2024年02月03日",10:13:52
"2024年02月04日",10:16:03
"""

# 读取数据
df = pd.read_csv(StringIO(csv_data), header=None, names=['日期', '昼长'])

# 将时间字符串转换为小时数
def time_to_hours(time_str):
    parts = time_str.split(':')
    return int(parts[0]) + int(parts[1])/60 + int(parts[2])/3600

df['小时数'] = df['昼长'].apply(time_to_hours)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(25, 10))

# 创建归一化对象
norm = mpl.colors.Normalize(vmin=df['小时数'].min(), vmax=df['小时数'].max())
cmap = plt.cm.viridis  # 使用viridis渐变色

# 创建柱状图
bars = ax.bar(df['日期'], df['小时数'], 
             color=cmap(norm(df['小时数'])),  # 应用渐变色
             edgecolor='black', 
             linewidth=1.0,
             alpha=0.9)

# 设置坐标轴标签和标题
ax.set_xlabel('日期', fontsize=14, labelpad=10)
ax.set_ylabel('昼长(小时)', fontsize=14, labelpad=10)
ax.set_title('日期昼长分布', fontsize=18, pad=20, fontweight='bold')

# 在柱子顶部添加时间标签
for bar, duration in zip(bars, df['昼长']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, 
            height + 0.03, 
            duration, 
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold')

# 旋转X轴标签
ax.set_xticks(range(len(df['日期'])))
ax.set_xticklabels(df['日期'], rotation=75, fontsize=10, ha='right')
ax.tick_params(axis='y', labelsize=10)

# 设置Y轴范围
ax.set_ylim(8.0, 12.0)

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 正确创建颜色条 - 修复错误的关键
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 必须设置数组
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('昼长强度', fontsize=12)
cbar.ax.tick_params(labelsize=9)

# 设置颜色条刻度标签字体
for label in cbar.ax.get_yticklabels():
    label.set_fontname('WenQuanYi Micro Hei')

fig.text(0.38, -0.01, 
         '注：数据来自网站 https://richurimo.bmcx.com/', 
         ha='left', fontsize=10, color='#555555')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# 显示图表
plt.savefig('day_time.png', dpi=100, bbox_inches='tight')
