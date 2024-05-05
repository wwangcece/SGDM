import os

dir = "dataset/S2-NAIP/train/ref_512"
for img in os.listdir(dir):
    origin_path = os.path.join(dir, img)
    target_path = os.path.join(dir, img[3:])
    os.rename(origin_path, target_path)