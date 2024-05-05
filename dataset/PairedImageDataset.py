import os
from typing import Dict
import time
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from dataset.exps.util import IMG_EXTENSIONS
from utils.image import augment, random_crop_arr, center_crop_arr

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

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

class PairedImageDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        hq_dataroot: str,
        lq_dataroot: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
    ) -> "PairedImageDataset":
        super(PairedImageDataset, self).__init__()
        self.hq_paths = get_paths_from_images(hq_dataroot)
        self.lq_paths = get_paths_from_images(lq_dataroot)
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.out_size = out_size

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images and lq images-------------------------------- #
        hq_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        for _ in range(3):
            try:
                hq_img = Image.open(hq_path).convert("RGB")
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"
        assert success, f"failed to load image {lq_path}"
        
        if self.crop_type == "random":
            hq_img = random_crop_arr(hq_img, self.out_size)
            lq_img = random_crop_arr(lq_img, self.out_size)
        elif self.crop_type == "center":
            hq_img = center_crop_arr(hq_img, self.out_size)
            lq_img = random_crop_arr(lq_img, self.out_size)
        # self.crop_type is "none"
        else:
            hq_img = np.array(hq_img)
            lq_img = np.array(lq_img)
            assert hq_img.shape[:2] == (self.out_size, self.out_size)
            assert lq_img.shape[:2] == (self.out_size, self.out_size)
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (hq_img[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (lq_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        img_lq = augment(img_lq, self.use_hflip, self.use_rot)

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        img_lq = torch.from_numpy(
            img_lq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()

        return {
            "hq": img_hq, "lq": img_lq, "txt": ""
        }

    def __len__(self) -> int:
        return len(self.hq_paths)
