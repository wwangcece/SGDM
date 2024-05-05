import os
import random
import shutil
from PIL import Image
import numpy as np

def move_and_delete_files(src_paths, dest_paths, num_to_keep, num_to_move):
    ratio = 0.5

    max_num_building = int(num_to_move * ratio)
    num_building = 0
    # 获取所有文件
    files = list(os.listdir(src_paths[0]))
    
    # 随机打乱文件顺序
    random.shuffle(files)

    # 保留前45300个文件
    files_to_keep = files[:num_to_keep]
    
    # 移动文件到目标文件夹
    moved_img_num = 0
    for i in range(num_to_keep):
        if moved_img_num == num_to_move:
            break
        file = files_to_keep[i]
        img = np.asarray(Image.open(os.path.join(src_paths[1], file)).convert("L"))
        if(np.var(img) < 20):
            moved_img_num += 1
            for src_path, dest_path in zip(src_paths, dest_paths):
                src_file = os.path.join(src_path, file)
                dest_file = os.path.join(dest_path, file)
                # shutil.move(src_file, dest_file)
                shutil.copy(src_file, dest_file)
        elif(num_building < max_num_building):
            moved_img_num += 1
            num_building += 1
            for src_path, dest_path in zip(src_paths, dest_paths):
                src_file = os.path.join(src_path, file)
                dest_file = os.path.join(dest_path, file)
                # shutil.move(src_file, dest_file)
                shutil.copy(src_file, dest_file)
        else:    
            continue
        
    # 删除源文件夹中除了保留文件外的其他文件
    # for src_path in src_paths:
    #     for file in os.listdir(src_path):
    #         if file not in files_to_keep:
    #             file_path = os.path.join(src_path, file)
    #             os.remove(file_path)

# 处理训练集
# 保留各文件夹前45300个文件
num_to_keep = 50460
num_val = 300
# 输入文件夹
in_path1 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256"
in_path2 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/ref_256"
in_path3 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/sr_16_256"

# 输出文件夹
out_path1 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v6/hr_256"
out_path2 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v6/ref_256"
out_path3 = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v6/sr_16_256"

os.makedirs(out_path1, exist_ok=True)
os.makedirs(out_path2, exist_ok=True)
os.makedirs(out_path3, exist_ok=True)

move_and_delete_files([in_path1, in_path2, in_path3], [out_path1, out_path2, out_path3], num_to_keep, num_val)