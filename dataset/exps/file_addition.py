import os
import shutil

def folder_addition(folder1, folder2):
    # 获取两个文件夹中的文件列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 计算差集并将文件夹2中的文件移动到文件夹1中
    files_to_add = files2.difference(files1)
    for file_name in files_to_add:
        src_path = os.path.join(folder2, file_name)
        dst_path = os.path.join(folder1, file_name)
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} to {dst_path}")

if __name__ == "__main__":
    folder1_paths = ["/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/ref_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/sr_16_256"]  # 替换为第一个文件夹的路径
    folder2_paths = ["/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/hr_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/ref_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/sr_16_256"]  # 替换为第二个文件夹的路径
    for folder1_path, folder2_path in zip(folder1_paths, folder2_paths):
        folder_addition(folder1_path, folder2_path)