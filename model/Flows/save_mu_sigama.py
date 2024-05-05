import os
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import einops
from omegaconf import OmegaConf
import sys
sys.path.append("/mnt/massive/wangce/SGDM/DiffBIR")

from ldm.xformers_state import disable_xformers
from utils.common import instantiate_from_config, load_state_dict
from torch.utils.data import DataLoader
from model.vgg16 import HR_Style_Encoder
from tqdm import tqdm

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # TODO: add help info for these options
    parser.add_argument(
        "--ckpt",
        default="/mnt/massive/wangce/SGDM/DiffBIR-exp/exp-refsr-3.2.1-Adapter-AdaIN-mid/lightning_logs/version_1/checkpoints/step=119999-val_lpips=0.515.ckpt",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--config",
        default="configs/model/refsr_real.yaml",
        type=str,
        help="model config path",
    )
    parser.add_argument(
        "--train_config", type=str, default="configs/dataset/reference_sr_train_real.yaml"
    )
    parser.add_argument(
        "--savepath", type=str, default="model/Flows/results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", choices=["cpu", "cuda", "mps"]
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    train_dataset = instantiate_from_config(OmegaConf.load(args.train_config)["dataset"])
    train_dataloader = DataLoader(
        dataset=train_dataset, **(OmegaConf.load(args.train_config)["data_loader"])
    )
    print("dataset loaded!!!!!")

    model = HR_Style_Encoder()
    
    static_dic = torch.load(args.ckpt, map_location="cpu")

    model_dict = {}
    for key, value in static_dic["state_dict"].items():
        if "HR_Style_Encoder" in key:
            model_key = key.replace("adapter.merge_encoder.HR_Style_Encoder.", '')
            model_dict[model_key] = value

    load_state_dict(model, model_dict, strict=True)

    model.eval()
    model.to(args.device)

    style_means = []
    style_stds = []
    for idx, train_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        hr = train_data["SR"]
        hr = einops.rearrange(hr, "b h w c -> b c h w").to(memory_format=torch.contiguous_format).float().to(args.device)
        hr_cond = model(hr)
        hr_cond = hr_cond.detach().cpu()
        style_mean = hr_cond.mean(dim=(2, 3))
        style_std = hr_cond.std(dim=(2, 3)) + 1e-6
        style_means.append(style_mean)
        style_stds.append(style_std)

    style_mean = torch.cat(style_means, dim=0)
    style_std = torch.cat(style_stds, dim=0)
    np.save(os.path.join(args.savepath, "style_mean_lr.npy"), style_mean.numpy())
    np.save(os.path.join(args.savepath, "style_std_lr.npy"), style_std.numpy())

if __name__ == "__main__":
    main()
