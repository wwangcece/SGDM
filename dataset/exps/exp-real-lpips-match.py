from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing
import os
import sys

from tqdm import tqdm
current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_script_directory, '../..'))
sys.path.append(project_root)
print(sys.path)

from PIL import Image
import numpy as np
import cv2
import torch
import lpips
import shutil
import pyiqa
from tqdm.contrib.concurrent import process_map

def calculate_lpips(img1, img2, lpips:lpips.LPIPS):
    tensor1 = img1.copy()
    tensor2 = img2.copy()
    tensor1 = (torch.from_numpy(tensor1.transpose(2, 0, 1)).float() / 255.).cuda()
    tensor2 = (torch.from_numpy(tensor2.transpose(2, 0, 1)).float() / 255.).cuda()
    tensor1 = tensor1 * 2 -1
    tensor2 = tensor2 * 2 -1
    value = lpips(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
    return float(value)

def calculate_niqe(img, device, niqe_metric):
    tensor = img.copy()
    tensor = (torch.from_numpy(tensor.transpose(2, 0, 1)).float() / 255.).cuda(device).unsqueeze(0)
    try:
        niqe_score = niqe_metric(tensor) #评估
        return float(niqe_score)
    except Exception as e:
        return 1e5

def process(s2_des_img, s2_source_dir, arcgis_dir, out_dir, lpips_metric):
    # s2_source_dir, arcgis_dir, out_dir, lpips_metric, s2_des_img = args
    # 读取source中S2多时相影像
    x_18, y_18 = int(s2_des_img[3:8]), int(s2_des_img[9:14])
    x_17, y_17 = x_18 // 2, y_18 // 2
    delta_x, delta_y = x_18 - x_17 * 2, y_18 - y_17 * 2
    source_name = "{}_{}.png".format(x_17, y_17)

    source_img = np.asarray(
        Image.open(os.path.join(s2_source_dir, source_name)).convert("RGB")
    )
    gt_img = np.asarray(Image.open(os.path.join(arcgis_dir, s2_des_img)).convert("RGB"))
    height, lr_size, _ = source_img.shape
    hr_size, hr_size, _ = gt_img.shape
    # 开始逐行搜索，计算LPIPS
    left = 0
    top = 0
    # curr_out_dir = os.path.join(out_dir, s2_des_img)
    # os.makedirs(curr_out_dir, exist_ok=True)

    min_val = float("inf")
    min_s2_img = None
    for i in range(height // lr_size):
        top = i * lr_size
        left = 0
        if delta_x > 0:
            left += lr_size // 2
        if delta_y > 0:
            top += lr_size // 2
        clip_lr_img = source_img[top:top+lr_size // 2, left:left+lr_size // 2, :]
        clip_sr_img = cv2.resize(clip_lr_img, (hr_size, hr_size), cv2.INTER_CUBIC)
        val = calculate_lpips(clip_sr_img, gt_img, lpips_metric)
        # val += 2 * calculate_niqe(clip_sr_img, device, niqe_metric) / 20.
        val += np.mean((clip_sr_img / 255. - gt_img / 255.)**2) * 100
        if val < min_val:
            min_val = val
            min_s2_img = clip_sr_img

        # 保存结果
        # Image.fromarray(clip_sr_img.astype(np.uint8)).save(os.path.join(curr_out_dir, "{}.png".format(round(val, 3))))
    # shutil.copy(os.path.join(s2_des_dir, s2_des_img), os.path.join(curr_out_dir, "origin_lr.png"))
    file_name = os.path.splitext(s2_des_img)[0]
    shutil.copy(os.path.join(arcgis_dir, s2_des_img), os.path.join(out_dir, "{}_hr.png".format(file_name)))
    Image.fromarray(min_s2_img.astype(np.uint8)).save(os.path.join(out_dir, "{}_lr.png".format(file_name)))
    print("done")


# if __name__ == "__main__":
#     s2_source_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel2"
#     s2_des_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/sr_16_256"
#     arcgis_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/hr_256"
#     out_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/match-lpips-mse-best-multi"
#     device = "cuda"

#     os.makedirs(out_dir, exist_ok=True)
#     lpips_metric = lpips.LPIPS(net='alex').cuda()

#     args_list = [(s2_source_dir, arcgis_dir, out_dir, lpips_metric, s2_des_img) for s2_des_img in os.listdir(s2_des_dir)]
#     for args in args_list:
#         process(args)

# if __name__ == "__main__":
#     # Set the start method to 'spawn' for CUDA compatibility
#     multiprocessing.set_start_method('spawn', force=True)
#     # ... (unchanged)
#     s2_source_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel2"
#     s2_des_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/sr_16_256"
#     arcgis_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256"
#     out_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/val/match-lpips-mse-best-multi"
#     os.makedirs(out_dir, exist_ok=True)
#     lpips_metric = lpips.LPIPS(net='alex').cuda()

#     # Create a pool of worker processes
#     num_processes = 8  # You can adjust this value
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # Use partial to create a function with fixed arguments (s2_source_dir, arcgis_dir, out_dir, lpips_metric)
#         partial_process = partial(process, s2_source_dir=s2_source_dir, arcgis_dir=arcgis_dir, out_dir=out_dir, lpips_metric=lpips_metric)

#         # Map the function to the arguments list using the pool
#         pool.map(partial_process, os.listdir(s2_des_dir))

#     print("All processes completed.")

# multithreading
if __name__ == "__main__":
    # Set the number of threads you want to run
    num_threads = 40
    s2_source_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel2"
    s2_des_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/sr_16_256"
    arcgis_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/hr_256"
    out_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/train/match-lpips-mse-best"

    os.makedirs(out_dir, exist_ok=True)
    lpips_metric = lpips.LPIPS(net='alex').cuda()

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use partial to create a function with fixed arguments (s2_source_dir, arcgis_dir, out_dir)
        partial_process = partial(process, s2_source_dir=s2_source_dir, arcgis_dir=arcgis_dir, out_dir=out_dir, lpips_metric=lpips_metric)

        # Map the function to the arguments list using the executor
        list(tqdm(executor.map(partial_process, os.listdir(s2_des_dir)), total=len(os.listdir(s2_des_dir))))

    print("All threads completed.")