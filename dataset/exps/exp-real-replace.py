import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_image(subfolder, output_folder):
    # 获取子文件夹的名称
    folder_name = os.path.basename(subfolder)

    # 获取子文件夹中的所有图片文件
    image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg')) and 'tci' in f.name]

    for image_file in image_files:
        # 构建新的文件名，形如：子文件夹名_原文件名
        new_filename = f"{folder_name}.png"
        
        # 构建输出文件的完整路径
        output_path = os.path.join(output_folder, new_filename)

        # 复制并重命名图片文件到输出文件夹
        shutil.copy(image_file, output_path)

def process_images_threaded(input_folder, output_folder, max_num):
    # 获取父文件夹中所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
    subfolders = subfolders[:max_num]

    # 使用 tqdm 展示进度
    with tqdm(total=max_num, desc="Processing") as pbar:
        def update_pbar(*_):
            pbar.update()

        with ThreadPoolExecutor(os.cpu_count()) as executor:
            # 使用多线程处理每个子文件夹
            futures = [executor.submit(process_image, subfolder, output_folder) for subfolder in subfolders]

            # 添加回调函数，每次任务完成时更新 tqdm 进度条
            for future in futures:
                future.add_done_callback(update_pbar)

            # 等待所有线程完成
            for future in futures:
                future.result()

if __name__ == "__main__":
    # 指定输入和输出文件夹的路径
    input_folder = r"/mnt/massive/wangce/ssr_data/train_urban_set/sentinel2"
    output_folder = r"/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel2"
    max_num = 200000
    os.makedirs(output_folder, exist_ok=True)

    # 调用多线程函数处理图片
    process_images_threaded(input_folder, output_folder, max_num)