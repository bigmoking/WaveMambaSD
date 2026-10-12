import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# ===================== 1. 学术风格配置（论文专用） =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['grid.alpha'] = 0.3

# ===================== 2. 精准数据 + 学术配色 =====================
labels = ['Original', 'LL\n(Low-freq)', 'LH\n(Horizontal)', 'HL\n(Vertical)', 'HH\n(Diagonal)']
# 正确数据
enl_values = [67.63, 97.64, 2.22, 2.17, 2.08]
x = np.arange(len(labels))
bar_width = 0.55

# IEEE低饱和学术配色
colors = ['#555555', '#4A6FA5', '#7A96B8', '#9AB1C8', '#B9CBD8']

# ===================== 3. 创建画布 =====================
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

# ===================== 4. 绘制主图 =====================
bars = ax.bar(x, enl_values, bar_width, color=colors, edgecolor='black', linewidth=0.4)
ax.set_title('Fig.1.b  ENL Statistics of Wavelet Sub-bands (SARDet-100K)', fontsize=13, pad=12)
ax.set_ylabel('Equivalent Number of Looks (ENL)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 主图数值标注
for bar, val in zip(bars, enl_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ===================== 5. 局部放大图（已移至【右侧】） =====================
# 位置参数：[左偏移, 下偏移, 宽度, 高度] → 调整左偏移到右侧
ax_inset = ax.inset_axes([0.55, 0.18, 0.38, 0.35])
ax_inset.bar(x, enl_values, bar_width, color=colors, edgecolor='black', linewidth=0.4)
ax_inset.margins(y=0)
ax_inset.set_ylim(0, 3)
ax_inset.set_yticks([0,1,2,3])
ax_inset.set_xticks([])
ax_inset.grid(axis='y', linestyle='--', alpha=0.3)

# 放大图数值标注
for bar, val in zip(ax_inset.patches, enl_values):
    height = bar.get_height()
    if height < 3:
        ax_inset.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                      f'{val}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# ===================== 6. 适配右侧放大图的连接线 =====================
con1 = ConnectionPatch(xyA=(2, 0), xyB=(2, 3), coordsA="data", coordsB="data",
                      axesA=ax_inset, axesB=ax, color='black', linewidth=0.5)
con2 = ConnectionPatch(xyA=(4, 0), xyB=(4, 3), coordsA="data", coordsB="data",
                      axesA=ax_inset, axesB=ax, color='black', linewidth=0.5)
ax.add_artist(con1)
ax.add_artist(con2)

# ===================== 7. 导出/展示 =====================
plt.tight_layout()
# 论文投稿高清PDF
plt.savefig('Fig1_b_ENL.jpg', bbox_inches='tight', dpi=300, format='jpg')
plt.show()