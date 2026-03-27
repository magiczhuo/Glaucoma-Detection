import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 指标名称
labels = np.array(['AP', 'AUC', 'Acc', 'F1', 'Sen', 'Spe'])
# 数据个数
num_vars = len(labels)
"""
data = {
    'Patch5Model':   [0.942, 0.976, 0.916, 0.875, 0.839, 0.839],
    'Branch2CBAM':   [0.946, 0.980, 0.927, 0.866, 0.956, 0.918],
    'Branch3CBAM':   [0.953, 0.982, 0.941, 0.882, 0.895, 0.956],
    'Branch3KECBAM': [0.965, 0.985, 0.946, 0.888, 0.880, 0.967]
}"""
data = {
    'Branch3CBAM':   [0.713 , 0.935 , 0.917 , 0.640 , 0.630, 0.909],
    'Branch3RCBAM': [ 0.774 , 0.959 , 0.929 , 0.751 , 0.730 , 0.936]
}

# 学术风格配色 (使用了主要学术期刊常用的颜色)
"""colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # 蓝, 橙, 红, 青
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']"""

colors = ['#E15759', '#76B7B2']  # 红, 青
line_styles = [ '-.', ':']
markers = ['^', 'D']

# 设置字体，确保显示清晰
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

# 将圆周平分为 num_vars 份，并闭合（首尾相连）
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合圆圈

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 调整旋转角度，使第一个轴（AP）在正上方
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 绘制每个模型
for idx, (model_name, values) in enumerate(data.items()):
    # 数据闭合
    values_copy = values + values[:1]
    
    # 绘制线条
    ax.plot(angles, values_copy, 
            color=colors[idx], 
            linewidth=2, 
            linestyle=line_styles[idx], 
            marker=markers[idx],
            markersize=6,
            label=model_name)
    
    # 填充颜色 (透明度设低一点以免遮挡)
    ax.fill(angles, values_copy, color=colors[idx], alpha=0.1)

# 设置 x 轴标签 (每个角的指标名)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# 设置 y 轴刻度 (根据数据范围调整，这里主要集中在 0.8-1.0)
# 为了让差异更明显，可以设置起始范围不为0，或者保持0-1
# 这里为了展示完整雷达图通常保留 0.5 - 1.0 的区间比较好看，或者根据实际数据自适应
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], ["0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"], color="grey", size=10)
plt.ylim(0.33, 1.0)  # 设置下限为0.8，这样差异看起来更明显

# 移除极坐标的圈外框线，让图看起来更像多边形
# ax.spines['polar'].set_visible(False) # 可选：是否隐藏最外面的圆圈

# 添加图例，放置在图表外侧
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, frameon=True, shadow=True)

# 添加标题
plt.title('Model Performance Comparison', size=16, color='black', y=1.1, fontweight='bold')

# 调整布局以适应标签
plt.tight_layout()

# 保存图像
output_path = 'model_radar_chart_3cls.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"雷达图已保存至: {output_path}")
plt.show()
