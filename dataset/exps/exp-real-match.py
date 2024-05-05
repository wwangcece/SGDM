from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm  # 导入tqdm

def match(lr_img, hr_img):
    num_rows = int(lr_img.shape[0] / lr_img.shape[1])
    patch_size = lr_img.shape[1]
    hr_size = hr_img.shape[0]
    min_mse = float("inf")
    min_index = 0
    for i in range(num_rows):
        lr_patch = lr_img[i * patch_size : (i + 1) * patch_size, :, :]
        sr_patch = cv2.resize(
            lr_patch, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC
        )
        diff = np.mean(np.abs(sr_patch - hr_img)*np.abs(sr_patch - hr_img))
        if diff < min_mse:
            min_mse = diff
            min_index = i
    best_match = lr_img[min_index * patch_size : (min_index + 1) * patch_size, :, :].astype(np.uint8)
    best_match = cv2.cvtColor(best_match, cv2.COLOR_BGR2RGB)

    return best_match

def main(hr_folder, lr_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    lr_imgs = os.listdir(lr_folder)
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        with tqdm(total=len(lr_imgs), desc=f"Processing...") as pbar:
            for img in lr_imgs:
                lr_img = cv2.imread(os.path.join(lr_folder, img), cv2.IMREAD_COLOR)
                hr_img = cv2.imread(os.path.join(hr_folder, img), cv2.IMREAD_COLOR)
                result = executor.submit(match, lr_img, hr_img)
                Image.fromarray(result.result()).save(os.path.join(output_folder, img))
                pbar.update(1)


hr_folder = "/mnt/massive/wangce/map-sat/dataset/N2-NAIP/hr_512"
lr_folder = "/mnt/massive/wangce/map-sat/dataset/N2-NAIP/lr_32"
output_folder = "/mnt/massive/wangce/map-sat/dataset/N2-NAIP/lr_32_"

main(hr_folder, lr_folder, output_folder)
