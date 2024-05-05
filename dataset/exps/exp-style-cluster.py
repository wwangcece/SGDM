from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# 加载VGG16模型
vgg16 = models.vgg16(pretrained=True)
# 提取relu4_1层的输出
vgg16_features = vgg16.features[:19]  # 只取relu4_1之前的层

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 图片文件夹路径
# image_folder_path = "/mnt/massive/wangce/dataset/Winter2Summer/trainA"
# image_folder_path = "/mnt/massive/wangce/backup/DiffBIR/dataset/S2-NAIP-512/train/hr_512"
image_folder_path = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256"
# 读取并处理图像，提取特征
feature_matrix = []
if len(os.listdir(image_folder_path)) > 10000:
    num = 10000
else:
    num = len(os.listdir(image_folder_path))
img_list = os.listdir(image_folder_path)[:num]
with tqdm(total=len(img_list), desc=f"Processing...") as pbar:
    for filename in img_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
            image_tensor = Variable(image_tensor.unsqueeze(0))
            vgg16_features.eval()
            with torch.no_grad():
                features = vgg16_features(image_tensor)
                # features = torch.std(features, dim=(2, 3))
                mean = torch.mean(features, dim=(2, 3))
                var = torch.var(features, dim=(2, 3))
                features = torch.cat([mean, var], dim=-1)
            features_np = features.squeeze().cpu().numpy().reshape(1, -1)
            feature_matrix.append(features_np)
            pbar.update(1)

# 将特征矩阵转换为N×HW形状
feature_matrix = np.concatenate(feature_matrix, axis=0)

# 使用PCA进行降维
tsne = TSNE(n_components=2, random_state=42)
print("PCA...")
reduced_features = tsne.fit_transform(feature_matrix)

# 使用K-means进行聚类
n_clusters = 5  # 你可以设置为你想要的簇的数量
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(feature_matrix)

# 可视化降维后的特征并根据聚类结果进行着色
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
plt.title('PCA Visualization with Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_visualization_with_clustering(HrTrain-Mean-Std).png')
