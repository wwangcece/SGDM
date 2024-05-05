from matlab_resize import imresize
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
import cv2

def process_image(hr_image_path, sr_image_path):
    hr_image = np.asarray(Image.open(hr_image_path).convert("RGB"))
    # sr_image = imresize(hr_image, 1.0/2.0)
    sr_image = cv2.resize(hr_image, (256, 256), cv2.INTER_CUBIC)
    Image.fromarray(sr_image).save(sr_image_path)

def main():
    hr_path = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel-2"
    sr_path = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel-2-256"
    hr_image_names = os.listdir(hr_path)
    os.makedirs(sr_path, exist_ok=True)

    # 使用 thread_map 替代 ThreadPoolExecutor
    with tqdm(total=len(hr_image_names), desc=f"Processing...") as pbar:
        thread_map(lambda hr_image_name: process_image(
            os.path.join(hr_path, hr_image_name),
            os.path.join(sr_path, hr_image_name)
        ), hr_image_names, max_workers=os.cpu_count())
        pbar.update(len(hr_image_names))

if __name__ == "__main__":
    main()

# def process_image(lr_image_path, sr_image_path):
#     lr_image = np.asarray(Image.open(lr_image_path).convert("RGB"))
#     sr_image = imresize(lr_image, 16.0)
#     Image.fromarray(sr_image).save(sr_image_path)

# def main():
#     lr_path = "./dataset/S2-NAIP/val/lr_32"
#     sr_path = "./dataset/S2-NAIP/val/sr_32_512"
#     lr_image_names = os.listdir(lr_path)
#     os.makedirs(sr_path, exist_ok=True)

#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = []
#         for lr_image_name in lr_image_names:
#             lr_image_path = os.path.join(lr_path, lr_image_name)
#             sr_image_path = os.path.join(sr_path, lr_image_name)

#             # Submit the image processing task to the thread pool
#             future = executor.submit(process_image, lr_image_path, sr_image_path)
#             futures.append(future)
            
#         for future in futures:
#             future.result()

# if __name__ == "__main__":
#     main()