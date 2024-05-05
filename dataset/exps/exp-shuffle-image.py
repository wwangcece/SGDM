import os
import numpy as np
from PIL import Image
import random
import shutil

# 创建保存图像的文件夹
output_folder = "/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/val/shuffle_ref_256"
source_floder = "/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/val/ref_256"
source_list = sorted(os.listdir(source_floder))
os.makedirs(output_folder, exist_ok=True)

origin_index = list(range(len(source_list)))
random.shuffle(origin_index)
shuffled_index = origin_index

# 设置图像参数
image_size = (256, 256)
num_images = len(source_list)

# 生成并保存噪声图
for i in range(num_images):
    source_path = os.path.join(source_floder, source_list[i])
    target_path = os.path.join(output_folder, source_list[shuffled_index[i]])
    shutil.copy(source_path, target_path)

print(f"{num_images} images have been shuffled and saved to {output_folder}.")