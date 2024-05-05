import os
import shutil

# 定义输入文件夹路径
input_folder = r'/mnt/massive/wangce/SGDM/DiffBIR-exp/exp-refsr-o-sim-real/validation/step--37116'

# 定义输出文件夹路径
output_folder_hq = input_folder + '/hr'
output_folder_samples = input_folder + '/sr'

# 确保输出文件夹存在，如果不存在就创建
os.makedirs(output_folder_hq, exist_ok=True)
os.makedirs(output_folder_samples, exist_ok=True)

# 遍历输入文件夹中的文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # 检查文件是否为PNG格式
    if filename.lower().endswith('.png') and os.path.isfile(file_path):
        # 检查文件名是否以hq结尾
        if '_hq' in filename:
            filename = filename.replace('_hq', '')
            # 如果是，将文件复制到hq文件夹，并在目标路径中包含文件名
            shutil.copy(file_path, os.path.join(output_folder_hq, filename))
        # 检查文件名是否以samples结尾
        elif '_samples' in filename:
            filename = filename.replace('_samples', '')
            # 如果是，将文件复制到samples文件夹，并在目标路径中包含文件名
            shutil.copy(file_path, os.path.join(output_folder_samples, filename))

print("PNG文件已复制完成。")