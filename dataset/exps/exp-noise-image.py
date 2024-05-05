import os
import numpy as np
from PIL import Image

# 创建保存图像的文件夹
output_folder = "/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/val/noise_ref_256"
name_source_floder = "/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/val/ref_256"
name_list = sorted(os.listdir(name_source_floder))
os.makedirs(output_folder, exist_ok=True)

# 设置图像参数
image_size = (256, 256)
num_images = len(name_list)

# 生成并保存噪声图
for i in range(num_images):
    # 生成随机噪声数据
    noise_data = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)

    # 创建PIL Image对象
    noise_image = Image.fromarray(noise_data, 'RGB')

    # 保存图像
    image_filename = os.path.join(output_folder, name_list[i])
    noise_image.save(image_filename)

print(f"{num_images} noise images have been generated and saved to {output_folder}.")