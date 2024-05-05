from PIL import Image
import os
import math
import re
# 提取mu{i}-sigma{j}模式，并按照(i, j)排序
def custom_sort_key(s):
    a = re.findall(r'(?<=mu)\d+\.?\d*',s)
    b = re.findall(r'(?<=sigama)\d+\.?\d*',s)
    return (int(a[-1]), int(b[-1]))

def image_cat(base_dir, size = (256, 256)):
    output_folder = base_dir  # 保存合并后大图的文件夹路径

    # 获取子文件夹列表
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    subfolders = sorted(subfolders, key=custom_sort_key)
    cols = int(math.sqrt(len(subfolders)))
    rows = int(math.sqrt(len(subfolders)))

    # 确定大图的大小
    big_image_size = (cols * size[0], rows * size[1])

    image_names = [name for name in os.listdir(subfolders[0]) if "samples" in name]
    num_image = len(image_names)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 合并图片
    for k in range(num_image):
        big_image = Image.new("RGB", big_image_size)
        for i, folder in enumerate(subfolders):
            image_path = os.path.join(folder, image_names[k])
            curr_img = Image.open(image_path)

            curr_row = int(i / cols)
            curr_col = i - curr_row * cols
            big_image.paste(curr_img, (curr_col * size[0], curr_row * size[1]))

        # 保存合并后的大图
        output_path = os.path.join(output_folder, f"merged_image_{k+1}.jpg")
        big_image.save(output_path)

base_dir = "/mnt/massive/wangce/backup/DiffBIR/experiment-s2-arcgis-new/results-x16/test-mu-sigama"
image_cat(base_dir)