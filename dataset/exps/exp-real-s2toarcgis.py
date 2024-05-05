import os
import shutil
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_image(img, s2_dir, arc_dir, out_dir):
    x_18 = int(img[3:8])
    y_18 = int(img[9:14])
    x_17 = x_18 // 2
    y_17 = y_18 // 2

    s2_img = np.asarray(
        Image.open(os.path.join(s2_dir, "{}_{}.png".format(x_17, y_17))).convert("RGB")
    )
    arcgis_img = np.asarray(Image.open(os.path.join(arc_dir, img)).convert("RGB"))

    height, size, _ = s2_img.shape
    hr_size, hr_size, _ = arcgis_img.shape
    num_rows = int(height / size)

    delta_x = x_18 - x_17 * 2
    delta_y = y_18 - y_17 * 2

    min_mse = float("inf")
    min_index = 0
    for i in range(num_rows):
        top = i * size
        left = 0
        if delta_x > 0:
            left += size // 2
        if delta_y > 0:
            top += size // 2
        clipped_patch = s2_img[top : top + size // 2, left : left + size // 2, :]
        sr_patch = cv2.resize(
            clipped_patch, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC
        )
        diff = np.mean(np.abs(sr_patch - arcgis_img) * np.abs(sr_patch - arcgis_img))
        if diff < min_mse:
            min_mse = diff
            min_index = i

    left = 0
    top = min_index * size
    if delta_x > 0:
        left += size // 2
    if delta_y > 0:
        top += size // 2
    best_match = s2_img[
        top : top + size//2, left : left + size // 2, :
    ].astype(np.uint8)

    Image.fromarray(best_match).save(os.path.join(out_dir, img))

def process_images_threaded(s2_dir, arc_dir, out_dir):
    arcgis_images = os.listdir(arc_dir)

    with ThreadPoolExecutor() as executor, tqdm(total=len(arcgis_images), desc="Processing") as pbar:
        def update_pbar(*_):
            pbar.update()

        futures = []
        for img in arcgis_images:
            future = executor.submit(process_image, img, s2_dir, arc_dir, out_dir)
            future.add_done_callback(update_pbar)
            futures.append(future)

        for future in futures:
            future.result()

if __name__ == "__main__":
    s2_dir = "/mnt/massive/wangce/backup/DiffBIR/dataset/S2-Arcgis-512/lr_32"
    arc_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/worldimagery-clarity"
    out_dir = "/mnt/massive/wangce/map-sat/dataset/S2-Arcgis/Sentinel-2"

    os.makedirs(out_dir, exist_ok=True)

    process_images_threaded(s2_dir, arc_dir, out_dir)
