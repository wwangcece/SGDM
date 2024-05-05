import os
import torch
import torchvision
import random
import numpy as np
from torchvision.transforms import functional as F
from einops import rearrange

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            # 水平上逆序切片（调转列的方向）
            img = img[:, ::-1, :]
        if vflip:
            # 垂直上逆序切片(调转行的方向)
            img = img[::-1, :, :]
        if rot90:
            # 行列调换,表示旋转90°
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    # 范围是[0,1]
    img = img.astype(np.float32) / 255.
    # 如果图像是2维的,则将其扩展为三维
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    # 如果第三维度的尺寸大于3,则只取前三个维度
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()


def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        HF = torch.rand(1) < 0.5
        VF = torch.rand(1) < 0.5
        RR = torch.rand(1) < 0.75
        angles = [90, 180, 270]
        # RandomHorizontalFlip
        if HF:
            imgs = [F.hflip(img) for img in imgs]
        # RandomVerticalFlip
        if VF:
            imgs = [F.vflip(img) for img in imgs]
        # RandomRotation
        if RR:
            random_angle = random.choice(angles)
            imgs = [F.rotate(img, random_angle) for img in imgs]

    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    ret_img = [rearrange(img, 'c h w -> h w c') for img in imgs]
    return ret_img
