import cv2
import os
import shutil

img_path = "/mnt/massive/wangce/map-sat/dataset/Synthetic-v18-45k/exp-val/ref_256/resized_2x.png"
save_dir = "/mnt/massive/wangce/map-sat/dataset/Synthetic-v18-45k/exp-val-ref"

# 读取图片
image = cv2.imread(img_path)

# 上采样并保存
for i in range(1, 6):
    scale_factor = 2 ** i
    image = cv2.imread(img_path)
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor))
    sr_path = os.path.join(save_dir, "sr_0_256")
    os.makedirs(sr_path, exist_ok=True)
    save_path = os.path.join(sr_path, f'resized_{scale_factor}x.png')
    cv2.imwrite(save_path, image)
    hr_path = os.path.join(save_dir, "hr_256")
    os.makedirs(hr_path, exist_ok=True)
    # shutil.copy(img_path, os.path.join(hr_path, f'resized_{scale_factor}x.png'))
    cv2.imwrite(os.path.join(hr_path, f'resized_{scale_factor}x.png'), image)

    
