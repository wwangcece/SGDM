import os
from PIL import Image

# 删除dir_path2中dir_path1不包含的图片
def cleanUp(dir_path1, dir_path2, dir_path3):
    target_set = set(os.listdir(dir_path1))
    for img_name in os.listdir(dir_path2):
        if(img_name in target_set):
            continue
        else:
            os.remove(os.path.join(dir_path2, img_name))
            os.remove(os.path.join(dir_path3, img_name))


dir_path1 = "./dataset/dataset/lr"
dir_path2 = "./dataset/dataset/hr"
dir_path3 = "./dataset/dataset/ref"

cleanUp(dir_path1, dir_path2, dir_path3)