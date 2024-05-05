from PIL import Image
import cv2
from torch.utils.data import Dataset
import dataset.exps.util as Util
import numpy as np
import os

class LRHRRefDataset(Dataset):
    def __init__(
        self,
        dataroot,
        l_resolution=16,
        r_resolution=256,
        split="train",
        data_len=-1,
        use_txt=True,
        style_path=None
    ):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split
        self.use_txt = use_txt
        self.style_path = style_path

        self.sr_path = Util.get_paths_from_images(
            "{}/sr_{}_{}".format(dataroot, l_resolution, r_resolution)
        )
        self.hr_path = Util.get_paths_from_images(
            "{}/hr_{}".format(dataroot, r_resolution)
        )
        self.ref_path = Util.get_paths_from_images(
            "{}/ref_{}".format(dataroot, r_resolution)
        )
        if self.style_path:
            # Style image directory
            if os.path.isdir(self.style_path):
                self.style_path = Util.get_paths_from_images(self.style_path)
            # Single image for style guidance
            else:
                self.style_path = [self.style_path]

        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_Ref = None
        img_SR = None
        img_style = None
        img_size = 1472

        img_HR = (
            np.asarray(Image.open(self.hr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_Ref = (
            np.asarray(Image.open(self.ref_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_SR = (
            np.asarray(Image.open(self.sr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)

        if self.style_path:
            if len(self.style_path) > 1:
                img_style = (
                    np.asarray(Image.open(self.style_path[index]).convert("RGB")) / 255.0
                ).astype(np.float32)
            else:
                img_style = (
                    np.asarray(Image.open(self.style_path[0]).convert("RGB")) / 255.0
                ).astype(np.float32)

        if self.use_txt:
            txt_book = np.asarray(Image.open(self.ref_path[index]).convert("RGB"))
            building_vec = np.array([226, 224, 210])
            road_vec1 = np.array([254, 254, 254])
            road_vec2 = np.array([255, 204, 133])
            woods_vec1 = np.array([217, 233, 198])
            woods_vec2 = np.array([228, 236, 213])
            woods_vec3 = np.array([208, 230, 190])
            water_vec = np.array([133, 203, 249])

            building_exist = np.all((txt_book == building_vec), axis=-1).any()
            road_exist = (
                np.all((txt_book == road_vec1), axis=-1).any()
                or np.all((txt_book == road_vec2), axis=-1).any()
            )
            woods_exist = (
                np.all((txt_book == woods_vec1), axis=-1).any()
                or np.all((txt_book == woods_vec2), axis=-1).any()
                or np.all((txt_book == woods_vec3), axis=-1).any()
            )
            water_exist = np.all((txt_book == water_vec), axis=-1).any()

            txt = "land, "
            if building_exist:
                txt += "building of regular shape, "
            if road_exist:
                txt += "road, "
            if woods_exist:
                txt += "woods, "
            if water_exist:
                txt += "water area, "
        else:
            txt = ""
        
        if self.style_path:
            [img_style, img_SR, img_Ref, img_HR] = Util.transform_augment(
                [img_style, img_SR, img_Ref, img_HR], split=self.split
            )
            img_HR = img_HR * 2 - 1
            return {
                "HR": img_HR,
                "SR": img_SR,
                "Ref": img_Ref,
                "Style": img_style,
                "txt": txt,
                "path": self.hr_path[index],
            }
            
        else:
            [img_SR, img_Ref, img_HR] = Util.transform_augment(
                [img_SR, img_Ref, img_HR], split=self.split
            )
            img_HR = img_HR * 2 - 1
            return {
                "HR": img_HR,
                "SR": img_SR,
                "Ref": img_Ref,
                "txt": txt,
                "path": self.hr_path[index],
            }