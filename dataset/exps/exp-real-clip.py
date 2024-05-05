from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
from tqdm import tqdm  # 导入tqdm

def crop_save(input_path, output_folder):
    image = Image.open(input_path)
    # 获取图片尺寸
    width, height = image.size

    # 计算裁剪的坐标
    left = 0
    top = 0
    right = width // 2
    bottom = width // 2

    num_row = int(height / width)

    base_name = os.path.basename(input_path)
    x = 2*int(base_name[:5])
    y = 2*int(base_name[6:11])

    x_curr = 2*x
    y_curr = 2*y

    # 裁剪并保存四个区域
    for i in range(4):
        cropped_image = image.crop((left, top, right, bottom))
        # 更新裁剪坐标
        if i == 0:
            left += width // 2
            right += width // 2
            x_curr = x + 1
            y_curr = y
        elif i==1:
            top += width // 2
            bottom += width // 2
            x_curr = x + 1
            y_curr = y + 1
        elif i == 2:
            left -= width // 2
            right -= width // 2
            x_curr = x
            y_curr = y + 1

        output_filename = f"18_{x_curr}{y_curr}"
        output_path = os.path.join(output_folder, output_filename)
        cropped_image.save(output_path)

        
    # 关闭原始图片
    image.close()

def main(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        with tqdm(total=len(os.listdir(input_folder)), desc=f"Processing...") as pbar:
            # 遍历输入文件夹中的图片
            for filename in os.listdir(input_folder):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    # 读取图片
                    input_path = os.path.join(input_folder, filename)
                    executor.submit(crop_save, input_path, output_folder)
                    pbar.update(1)


if __name__ == "__main__":
    input_folders = ["/mnt/massive/wangce/backup/DiffBIR/dataset/S2-Arcgis-512/lr_32"]
    output_folders = ["/mnt/massive/wangce/backup/DiffBIR/dataset/S2-Arcgis-512/lr_16"]
    for input_folder, output_folder in zip(input_folders, output_folders):
        main(input_folder, output_folder)
