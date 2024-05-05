import itertools
from typing import Mapping, Any, Tuple
import copy
from collections import OrderedDict
import os
from PIL import Image
import einops
import torch
import torch as th
import torch.nn as nn
import torchvision
import numpy as np
import math
from torch.nn import functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    UNetModel,
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler
from utils.metrics import calculate_psnr_pt, LPIPS
from .adapters_simu import My_Adapter


# Do forward process for UNetModel with prepared "control" tensors
class ControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        # "control" is output of "ControlNet" model
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if (i - 1) % 3 == 0 and ((i - 1) / 3 < len(control)):
                h = h + control[int((i - 1) / 3)]
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i % 3 == 0 and ((3 - i / 3) < len(control)):
                h = h + control[int(3 - i / 3)]

        h = h.type(x.dtype)
        return self.out(h)

class ControlLDM(LatentDiffusion):
    def __init__(
        self,
        sr_key: str,
        ref_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        disable_preprocess=False,
        frozen_diff=True,
        *args,
        **kwargs,
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.adapter = My_Adapter()
        self.sr_key = sr_key
        self.ref_key = ref_key
        self.disable_preprocess = disable_preprocess
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.frozen_diff = frozen_diff

        # instantiate preprocess module (SwinIR)
        # self.preprocess_model = instantiate_from_config(preprocess_config)
        # frozen_module(self.preprocess_model)

        self.lpips_metric = LPIPS(net="alex")

        if self.frozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if "attn" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def apply_cond_ref_encoder(self, control_sr, control_ref):
        cond_latent = self.adapter(control_sr *2 -1, control_ref*2-1)
        cond_latent = [cond * self.scale_factor for cond in cond_latent]
        return cond_latent

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, self.hr_key, *args, **kwargs)
            # x: HR encoded
            # c: conditional text
            sr_cond = batch[self.sr_key]
            ref_cond = batch[self.ref_key]

            if bs is not None:
                sr_cond = sr_cond[:bs]
                ref_cond = ref_cond[:bs]

            sr_cond = sr_cond.to(self.device)
            ref_cond = ref_cond.to(self.device)

            sr_cond = einops.rearrange(sr_cond, "b h w c -> b c h w")
            ref_cond = einops.rearrange(ref_cond, "b h w c -> b c h w")

            sr_cond = sr_cond.to(memory_format=torch.contiguous_format).float()
            ref_cond = ref_cond.to(memory_format=torch.contiguous_format).float()

            lq = sr_cond

        # apply condition encoder
        cond_latent = self.apply_cond_ref_encoder(sr_cond, ref_cond)

        return x, dict(
            c_crossattn=[c],
            cond_latent=[cond_latent],
            lq=[lq],
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        cond_control = cond["cond_latent"][0]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=cond_control,
            only_mid_control=self.only_mid_control,
        )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.hr_key)

        # [-1 ,1]
        log["hq"] = einops.rearrange(batch[self.hr_key], "b h w c -> b c h w")
        # [0, 1]
        log["lq"] = einops.rearrange(batch[self.sr_key], "b h w c -> b c h w")
        # [0, 1]
        log["ref"] = einops.rearrange(batch[self.ref_key], "b h w c -> b c h w")

        samples = self.sample_log(
            cond=c,
            steps=sample_steps,
        )
        # [0, 1]
        log["samples"] = samples

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["cond_latent"][0][0].shape
        shape = (b, self.channels, h, w)
        samples = sampler.sample(steps, shape, cond)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.adapter.parameters())
            + list(self.model.parameters())
        )
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_validation_epoch_start(self):
        # PSNR、LPIPS metrics are set zero
        self.val_psnr = 0
        self.val_lpips = 0

    def validation_step(self, batch, batch_idx):
        val_results = self.log_images(batch)

        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)
        # calculate psnr
        # bchw;[0, 1];tensor
        hr_batch_tensor = val_results["hq"].detach().cpu()
        hr_batch_tensor = (hr_batch_tensor + 1) / 2.0
        sr_batch_tensor = val_results["samples"].detach().cpu()
        this_psnr = 0
        for i in range(len(hr_batch_tensor)):
            curr_hr = (
                hr_batch_tensor[i]
                .transpose(0, 1)
                .transpose(1, 2)
                .numpy()
                .astype(np.float64)
            )
            curr_sr = (
                sr_batch_tensor[i]
                .transpose(0, 1)
                .transpose(1, 2)
                .numpy()
                .astype(np.float64)
            )
            curr_psnr = 20 * math.log10(
                1.0 / math.sqrt(np.mean((curr_hr - curr_sr) ** 2))
            )
            self.val_psnr += curr_psnr
            this_psnr += curr_psnr
        this_psnr /= len(hr_batch_tensor)

        # calculate lpips
        this_lpips = 0
        hq = ((val_results["hq"] + 1) / 2).clamp_(0, 1).detach().cpu()
        pred = val_results["samples"].detach().cpu()
        curr_lpips = self.lpips_metric(hq, pred).sum().item()
        self.val_lpips += curr_lpips
        this_lpips = curr_lpips / len(hr_batch_tensor)

        # log metrics out
        self.log("val_psnr", this_psnr)
        self.log("val_lpips", this_lpips)

        # save images
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            if image_key == "hq":
                image = (image + 1.0) / 2.0
            N = len(image)

            for i in range(N):
                img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
                curr_img = image[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, image_key)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

    def on_validation_epoch_end(self):
        # calculate average metrics
        self.val_psnr /= self.trainer.datamodule.val_config.dataset.params.data_len / 1
        self.val_lpips /= self.trainer.datamodule.val_config.dataset.params.data_len / 1
        # make saving dir
        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        save_dir = os.path.join(
            save_dir, f"psnr-{round(self.val_psnr, 2)}-lpips-{round(self.val_lpips, 2)}"
        )
        os.makedirs(save_dir, exist_ok=True)

    def validation_inference(self, batch, batch_idx, save_dir):
        val_results = self.log_images(batch)
        os.makedirs(save_dir, exist_ok=True)
        # calculate psnr
        # bchw;[0, 1];tensor
        hr_batch_tensor = val_results["hq"].detach().cpu()
        hr_batch_tensor = (hr_batch_tensor + 1) / 2.0
        sr_batch_tensor = val_results["samples"].detach().cpu()
        this_psnr = 0
        for i in range(len(hr_batch_tensor)):
            curr_hr = (
                hr_batch_tensor[i]
                .transpose(0, 1)
                .transpose(1, 2)
                .numpy()
                .astype(np.float64)
            )
            curr_sr = (
                sr_batch_tensor[i]
                .transpose(0, 1)
                .transpose(1, 2)
                .numpy()
                .astype(np.float64)
            )
            curr_psnr = 20 * math.log10(
                1.0 / math.sqrt(np.mean((curr_hr - curr_sr) ** 2))
            )
            this_psnr += curr_psnr
        this_psnr /= len(hr_batch_tensor)

        # calculate lpips
        hq = ((val_results["hq"] + 1) / 2).clamp_(0, 1).detach().cpu()
        pred = val_results["samples"].detach().cpu()
        this_lpips = self.lpips_metric(hq, pred).sum().item()
        this_lpips /= len(hr_batch_tensor)

        # save images
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            if image_key == "hq":
                image = (image + 1.0) / 2.0
            N = len(image)

            for i in range(N):
                img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
                curr_img = image[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, image_key)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

        return this_psnr, this_lpips
