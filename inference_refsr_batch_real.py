import shutil
import os
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ldm.xformers_state import disable_xformers
from utils.common import instantiate_from_config, load_state_dict
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from model.Flows.mu_sigama_estimate_normflows import CreateFlow

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
        "--style_scale",
        default = 1,
        type=int,
    )
    parser.add_argument(
        "--sample_style",
        default=False,
        # default = None,
        type=str,
        help="Whether to perform style sampling from the pretrained normalizing flow model. If true, 'ckpt_flow_mean' and 'ckpt_flow_std' must not be 'None'",
    )
    parser.add_argument(
        "--ckpt_flow_mean",
        default="model/Flows/checkpoints/flow_tanh_mini_mean",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--ckpt_flow_std",
        default="model/Flows/checkpoints/flow_tanh_mini_std",
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
        "--val_config", type=str, default="configs/dataset/reference_sr_val_real.yaml"
    )
    # FlowSampler-sampleFromTrain
    parser.add_argument(
        "--output", type=str, default="/mnt/massive/wangce/SGDM/DiffBIR-exp/test-randn-hr-guide-style"
    )
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda:1", choices=["cpu", "cuda", "mps"]
    )

    return parser.parse_args()


def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device

def split_result(input_folder):
    # 定义输出文件夹路径
    output_folder_hq = input_folder + '/hr'
    output_folder_samples = input_folder + '/sr'

    # 确保输出文件夹存在，如果不存在就创建
    os.makedirs(output_folder_hq, exist_ok=True)
    os.makedirs(output_folder_samples, exist_ok=True)

    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为PNG格式
        if filename.lower().endswith('.png') and os.path.isfile(file_path):
            # 检查文件名是否以hq结尾
            if 'hq' in filename:
                filename = filename.replace("_hq", "")
                # 如果是，将文件复制到hq文件夹，并在目标路径中包含文件名
                shutil.copy(file_path, os.path.join(output_folder_hq, filename))
            # 检查文件名是否以samples结尾
            elif 'samples' in filename:
                filename = filename.replace("_samples", "")
                # 如果是，将文件复制到samples文件夹，并在目标路径中包含文件名
                shutil.copy(file_path, os.path.join(output_folder_samples, filename))

    print("PNG文件已复制完成。")

def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    val_dataset = instantiate_from_config(OmegaConf.load(args.val_config)["dataset"])
    val_dataloader = DataLoader(
        dataset=val_dataset, **(OmegaConf.load(args.val_config)["data_loader"])
    )
    print("dataset loaded!!!!!")

    model = instantiate_from_config(OmegaConf.load(args.config))

    static_dic = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, static_dic, strict=False)

    if args.ckpt_flow_mean and args.ckpt_flow_std:
        flow_mean = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
        static_dic_flow_mean = torch.load(args.ckpt_flow_mean, map_location="cpu")
        load_state_dict(flow_mean, static_dic_flow_mean, strict=True)
        model.flow_mean = flow_mean

        flow_std = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
        static_dic_flow_std = torch.load(args.ckpt_flow_std, map_location="cpu")
        load_state_dict(flow_std, static_dic_flow_std, strict=True)
        model.flow_std = flow_std

    model.freeze()
    model.to(args.device)

    val_dataset = instantiate_from_config(OmegaConf.load(args.val_config)["dataset"])
    val_dataloader = DataLoader(
        dataset=val_dataset, **(OmegaConf.load(args.val_config)["data_loader"])
    )

    psnr = 0
    lpips = 0

    for idx, val_data in enumerate(val_dataloader):
        model.eval()
        this_psnr, this_lpips = model.validation_inference(val_data, idx, args.output, args.sample_style, args.style_scale)
        # this_psnr, this_lpips = model.validation_inference(val_data, idx, args.output)
        psnr += this_psnr
        lpips += this_lpips
    
    psnr /= len(val_dataloader)
    lpips /= len(val_dataloader)
    split_result(args.output)

    os.makedirs(
        os.path.join(args.output, f"psnr-{round(psnr, 3)}-lpips-{round(lpips, 3)}"),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
