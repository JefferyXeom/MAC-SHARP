import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from numba import jit

# 读取原始坐标数据
x = []
y = []

with open('./test_data/corr_ind.txt', 'r') as f:
    for line in f:
        a, b = map(int, line.split())
        x.append(a)
        y.append(b)

# 读取掩码数据（0 或 1）
mask = []

with open('./test_data/label.txt', 'r') as f:
    for line in f:
        mask.append(int(line.strip()))  # 读取每行是 0 或 1

# 转换为 NumPy 数组
x = np.array(x)
y = np.array(y)
mask = np.array(mask)

# 找到数据的最大值，用来设置图像的大小
max_x = np.max(x)
max_y = np.max(y)

# 创建一个空的二维网格
grid = np.zeros((max_y + 1, max_x + 1))  # 这里 +1 是为了避免从 0 到 max_x 索引时出错

# 对每一对坐标点，将对应位置标记为 1
for i in range(len(x)):
    grid[y[i], x[i]] = 1

# 预定义一个膨胀结构（3x3 或 5x5）
dilated_grid = np.zeros_like(grid)

# 使用 `scipy.ndimage` 膨胀操作
dilated_grid = binary_dilation(grid, structure=np.ones((3, 3))).astype(int)

# 使用更强的膨胀操作（对于 mask 为 1 的行）
for i in range(len(x)):
    if mask[i] == 1:
        dilated_grid[y[i], x[i]] = 1
        dilated_grid = binary_dilation(dilated_grid, structure=np.ones((5, 5))).astype(int)

# 绘制膨胀后的热图
plt.figure(figsize=(10, 8))
plt.imshow(dilated_grid, cmap='Blues', interpolation='nearest')
plt.colorbar(label="Point Density")
plt.title("Optimized Dilated Grid Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
