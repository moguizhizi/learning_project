import numpy as np

# 示例数据
P = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])

# 计算直方图，bins=2048
hist, bin_edges = np.histogram(P, bins=2048)

# 输出结果
print("计数 (hist):", hist[:10])  # 只显示前10个计数
print("区间边界 (bin_edges):", bin_edges[:10])  # 只显示前10个边界

# 可视化（可选）
import matplotlib.pyplot as plt
plt.hist(P, bins=2048)
plt.savefig('/data1/project/learning_project/histogram.png', dpi=300, bbox_inches='tight')