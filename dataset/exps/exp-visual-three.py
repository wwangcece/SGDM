import numpy as np
import matplotlib.pyplot as plt

# 读取三个 npy 文件
data1 = np.load('model/Flows/results/style_2d.npy')
data2 = np.load('model/Flows/results/style_2d_lr.npy')
data3 = np.load('model/Flows/results/style_2d_sampled.npy')

# 创建一个新的图形
plt.figure(figsize=(8, 6))

# 可视化第一个 npy 文件的数据（使用红色）
plt.scatter(data1[:, 0], data1[:, 1], color='red', label='HR', s=5)

# 可视化第二个 npy 文件的数据（使用绿色）
plt.scatter(data2[:, 0], data2[:, 1], color='green', label='LR', s=5)

# 可视化第三个 npy 文件的数据（使用蓝色）
plt.scatter(data3[:, 0], data3[:, 1], color='blue', label='Sampled', s=5)

# 添加图例
plt.legend(fontsize=14)

# 显示图形
plt.savefig('model/Flows/results/style_compare.png')

print("Saved to: model/Flows/results/style_compare.png")