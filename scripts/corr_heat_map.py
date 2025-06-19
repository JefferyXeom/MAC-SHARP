# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler

# # 读取 CSV 文件
# matrix = pd.read_csv('./test_data/demo_graph_(score_th=60).csv', header=None)

# # 降采样，将矩阵分块，按2x2块求平均，减少显示的数据量
# downsampled_matrix = matrix.values[::2, ::2]  # 每隔一个取一个元素

# # 归一化矩阵数据（0 到 1）
# scaler = MinMaxScaler()
# normalized_matrix = scaler.fit_transform(matrix)

# # 创建热图，使用合适的颜色映射
# plt.figure(figsize=(10, 8))

# sns.heatmap(downsampled_matrix, cmap='viridis', annot=False, cbar=True)

# # 设置标题
# plt.title("Downsampled Heatmap of Eigen Matrix")

# # 显示热图
# plt.show()


import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader import transfer_functions as tf
import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import binary_dilation
import numpy as np

# 读取数据
matrix = pd.read_csv('./test_data/demo_graph_(score_th=60).csv', header=None)

# 使用 3x3 卷积核膨胀图像
# dilated_matrix = binary_dilation(matrix, structure=np.ones((3, 3))).astype(int)

# 将矩阵转换为 xarray DataArray
# matrix_xarray = xr.DataArray(dilated_matrix)
matrix_xarray = xr.DataArray(matrix)

# 创建 datashader canvas
canvas = ds.Canvas(plot_width=800, plot_height=800)

# 将数据绘制到画布上
heatmap = canvas.raster(matrix_xarray)

# 使用 matplotlib 获取 'viridis' cmap
cmap = plt.cm.inferno  # 或者换成 'inferno', 'cividis' 等其他可用颜色映射

# 使用 datashader 渲染图像
img = tf.shade(heatmap, cmap=cmap, how='log')
img = img[::-1]  # 翻转图像
img.to_pil().show()
