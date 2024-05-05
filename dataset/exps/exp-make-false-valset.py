import os
import shutil
from PIL import Image
import numpy as np
import cv2

# source_image_path = (
#     "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/random_hr_val/val-water/source-water.png"
# )
# target_dir_path = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/random_hr_val/val-water/hr_256"
# target_image_names = [
#     "18_41518_92090",
#     "18_41560_92201",
#     "18_41722_93692",
#     "18_41862_93625",
#     "18_41876_90356",
# ]
# suffix = ".png"

# for name in target_image_names:
#     source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
#     source_image = cv2.resize(source_image, (256, 256), cv2.INTER_CUBIC)
#     target_path = os.path.join(target_dir_path, name + suffix)
#     Image.fromarray(source_image.astype(np.uint8)).save(target_path)

# import os
# import shutil
# import random

# def copy_images(folder_A_path, folder_B_path, output_path):
#     # 获取文件夹A中的所有图片文件名
#     image_names_A = [file for file in os.listdir(folder_A_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    
#     # 获取文件夹B中的所有图片文件名
#     image_names_B = [file for file in os.listdir(folder_B_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    
#     # 随机选择N张图片
#     selected_images_B = random.sample(image_names_B, len(image_names_A))
    
#     # 创建输出文件夹
#     os.makedirs(output_path, exist_ok=True)
    
#     # 复制选定的图片到输出文件夹，并按照文件夹A中的名称命名
#     for name_A, name_B in zip(image_names_A, selected_images_B):
#         src_path = os.path.join(folder_B_path, name_B)
#         dest_path = os.path.join(output_path, name_A)
#         shutil.copy(src_path, dest_path)

# # 替换为实际的文件夹路径
# folder_A_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/sr_16_256'
# folder_B_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256'
# output_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/shuffle_hr_val/hr_256'

# # 执行复制操作
# copy_images(folder_A_path, folder_B_path, output_path)


import os
import shutil
import random

def copy_images(folder_A_path, image_B_path, output_path):
    # 获取文件夹A中的所有图片文件名
    image_names_A = [file for file in os.listdir(folder_A_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 创建输出文件夹
    os.makedirs(output_path, exist_ok=True)
    
    # 复制选定的图片到输出文件夹，并按照文件夹A中的名称命名
    for name_A in (image_names_A):
        src_path = image_B_path
        dest_path = os.path.join(output_path, name_A)
        shutil.copy(src_path, dest_path)

# 替换为实际的文件夹路径
folder_A_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/sr_16_256'
image_B_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/forest_hr_val/source-forest.png'
output_path = '/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/forest_hr_val/hr_256'

# 执行复制操作
copy_images(folder_A_path, image_B_path, output_path)