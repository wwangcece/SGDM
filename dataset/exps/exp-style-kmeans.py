import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
sys.path.append("/mnt/massive/wangce/backup/DiffBIR")
from model.Flows.mu_sigama_estimate_normflows import CreateFlow

# 1. 读取 npz 文件，降维，可视化
data1 = np.load('model/Flows/results/tanh_mini_std.npy')  # 替换为你的文件路径
data2 = np.load('model/Flows/results/tanh_mini_mean.npy')
data = np.concatenate((data1, data2), axis=1)

# 2. 进行聚类并计算各个聚类中心
num_clusters = 5  # 假设您希望将数据分成5个聚类
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)
cluster_centers = kmeans.cluster_centers_

# 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data = tsne.fit_transform(data)

# 3. 将聚类的可视化结果保存为png图片
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100, label='Cluster Centers')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('model/Flows/results/clustering_result.png')
plt.show()

# 4. 保存聚类中心为npy文件
np.save('model/Flows/results/cluster_centers.npy', cluster_centers)