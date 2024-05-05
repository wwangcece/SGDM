import os
import shutil

def folder_subtraction(folder1, folder2):
    # 获取两个文件夹中的文件列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 计算差集并删除文件夹1中的文件
    files_to_delete = files1.intersection(files2)
    for file_name in files_to_delete:
        file_path = os.path.join(folder1, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

if __name__ == "__main__":
    folder1_paths = ["/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/ref_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/sr_16_256"]  # 替换为第一个文件夹的路径
    folder2_paths = ["/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v4/hr_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v4/ref_256", "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val-v4/sr_16_256"]  # 替换为第二个文件夹的路径

    for folder1_path, folder2_path in zip(folder1_paths, folder2_paths):
        folder_subtraction(folder1_path, folder2_path)