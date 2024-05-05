import os
from PIL import Image
import numpy as np
import math

# img_name = "17_63855_40941"
# img1_path = f"results/exp-swinir/swinir/{img_name}.png.png"
# img2_path = f"results/exp-swinir/hr/{img_name}.png"

img_sr_path = "dataset/dataset/val/ref_256/18_127602_81702.png"
img_hr_path = "results-x8/exp/exp/sample.png"
img1 = np.asarray(Image.open(img_sr_path).convert("RGB"))
img2 = np.asarray(Image.open(img_hr_path).convert("RGB"))
psnr = 20 * math.log10(255.0 / math.sqrt(np.mean((img1 - img2) ** 2)))


# img_sr_list = os.listdir(img_sr_path)
# img_hr_list = os.listdir(img_hr_path)

# psnr = 0
# for i in range(len(img_sr_list)):
#     img_sr = os.path.join(img_sr_path, img_sr_list[i])
#     img_hr = os.path.join(img_hr_path, img_hr_list[i])
#     img1 = np.asarray(Image.open(img_sr).convert("RGB"))
#     img2 = np.asarray(Image.open(img_hr).convert("RGB"))
#     psnr += 20 * math.log10(255.0 / math.sqrt(np.mean((img1 - img2) ** 2)))
# psnr /= len(img_sr_list)

print(psnr)
