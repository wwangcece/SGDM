import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

sys.path.append("/mnt/massive/wangce/SGDM/DiffBIR")
from model.Flows.mu_sigama_estimate_normflows_single import CreateFlow

# 1. 读取 npz 文件，降维，可视化
# data1 = np.load('model/Flows/results/style_std_lr.npy')  # 替换为你的文件路径
# data2 = np.load('model/Flows/results/style_mean_lr.npy')
# data = np.concatenate((data1, data2), axis=1)
# # data = data2

# num_point = 40000
# tensor = data[np.random.choice(data.shape[0], num_point, replace=False)]

# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data = tsne.fit_transform(tensor)

# np.save('model/Flows/results/style_2d_lr.npy', embedded_data)

# # 可视化结果
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
# plt.title('t-SNE Visualization')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')

# # 保存图像为 PNG
# plt.savefig('model/Flows/results/tsne_style_lr_mean.png')  # 替换为你想要保存的文件路径

# 2. Flow采样，降维，可视化
# mean_ckpt_path = "model/Flows/checkpoints/flow_tanh_mini_mean"
# std_ckpt_path = "model/Flows/checkpoints/flow_tanh_mini_std"
# device = "cuda:1"

# num_point = 40000

# # flow_mean_std = CreateFlow(dim=512, num_layers=8, hidden_layers=[256, 512])
# flow_mean = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
# flow_std = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])

# flow_mean.load_state_dict(torch.load(mean_ckpt_path))
# flow_mean.to(device).eval()
# flow_std.load_state_dict(torch.load(mean_ckpt_path))
# flow_std.to(device).eval()

# mean_sample_data, _ = flow_mean.sample(num_point)
# mean_sample_data = mean_sample_data.cpu().detach().numpy()
# std_sample_data, _ = flow_std.sample(num_point)
# std_sample_data = std_sample_data.cpu().detach().numpy()

# print("Sampling done!!!")

# nan_mask = np.isnan(mean_sample_data)
# mean_sample_data[nan_mask] = 0
# inf_mask = np.isinf(mean_sample_data)
# mean_sample_data[inf_mask] = 0

# nan_mask = np.isnan(std_sample_data)
# std_sample_data[nan_mask] = 0
# inf_mask = np.isinf(std_sample_data)
# std_sample_data[inf_mask] = 0

# # sample_data = np.concatenate((mean_sample_data, std_sample_data), axis=1)
# # np.save('model/Flows/results/style_mean_sampled.npy', mean_sample_data)
# # np.save('model/Flows/results/style_std_sampled.npy', std_sample_data)
# sample_data = mean_sample_data
# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data = tsne.fit_transform(sample_data)
# np.save('model/Flows/results/style_2d_mean_sampled.npy', embedded_data)

# # 可视化结果
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
# plt.title('t-SNE Visualization for sample data')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')

# # 保存图像为 PNG
# plt.savefig('model/Flows/results/tsne_style_mean_sampled.png')  # 替换为你想要保存的文件路径


# 3. HR LR Sampling 联合TSNE
# hr_style_mean_path = "model/Flows/results/style_mean.npy"
# hr_style_std_path = "model/Flows/results/style_std.npy"
# lr_style_mean_path = "model/Flows/results/style_mean_lr.npy"
# lr_style_std_path = "model/Flows/results/style_std_lr.npy"
# sample_style_mean_path = "model/Flows/results/style_mean_sampled.npy"
# sample_style_std_path = "model/Flows/results/style_std_sampled.npy"
# point_num = 40000

# hr_style_mean = np.load(hr_style_mean_path)
# hr_style_std = np.load(hr_style_std_path)
# # hr_style = np.concatenate((hr_style_mean, hr_style_std), axis=1)[:point_num]
# hr_style = hr_style_mean[:point_num]

# lr_style_mean = np.load(lr_style_mean_path)
# lr_style_std = np.load(lr_style_std_path)
# # lr_style = np.concatenate((lr_style_mean, lr_style_std), axis=1)[:point_num]
# lr_style = lr_style_mean[:point_num]

# sample_style_mean = np.load(sample_style_mean_path)
# sample_style_std = np.load(sample_style_std_path)
# # sample_style = np.concatenate((sample_style_mean, sample_style_std), axis=1)[:point_num]
# sample_style = sample_style_mean[:point_num]

# total_style = np.concatenate((hr_style, lr_style, sample_style), axis=0)
# print("Data prepared!!")

# print("TSNE......")
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data = tsne.fit_transform(total_style)
# np.save("model/Flows/results/style_2d_mean_total.npy", embedded_data)
# print("Saved!!")

from matplotlib import rc

# 使用新罗马字体
rc('font', family='Times New Roman')

embedded_data = np.load('model/Flows/results/style_2d_mean_total.npy')
hr_style, lr_style, sample_style = np.split(embedded_data, 3, axis=0)

num_points = 10000

# 创建一个新的图形
plt.figure(figsize=(8, 6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 可视化第一个 npy 文件的数据（使用红色）
plt.scatter(hr_style[:num_points, 0], hr_style[:num_points, 1], label="HR", c="red", s=5, alpha=0.5)

# 可视化第二个 npy 文件的数据（使用绿色）
plt.scatter(lr_style[:num_points, 0], lr_style[:num_points, 1], label="LR", c="green", s=5, alpha=0.5)

# 可视化第三个 npy 文件的数据（使用蓝色）
plt.scatter(sample_style[:num_points, 0], sample_style[:num_points, 1], label="Sampled", c='blue', s=5, alpha=0.5)

# 添加图例
plt.legend(fontsize=12, loc="upper left")

# 显示图形
plt.savefig("model/Flows/results/style_compare.png")
print("See: model/Flows/results/style_compare.png")
