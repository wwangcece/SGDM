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
from .cond_ref_encoder import cond_ref_encoder
from .vgg16 import SR_Encoder, Ref_Encoder, SR_EncoderUp, SR_Ref_Encoder_Spade
from .swinir import SwinIR
from utils.metrics import calculate_psnr_pt, LPIPS
from inspect import isfunction


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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

        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, hint_channels * 3, model_channels, 3, padding=1)
                )
            ]
        )
        """
            self.input_blocks: 0:(32)[conv]
            1~3: (32, 32, 320)[res attn] [res attn] [res down] 
            4~6: (16, 16, 640)[res attn] [res attn] [res down]
            7~9: (8, 8, 1280)[res attn] [res attn] [res down] 
            10~11: (4, 4, 1280)[res res]

            self.middle_blocks: (4, 4, 1280)[res, attn, res]

            self.output_blocks: ······
        """
        # True
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, cond_sr_ref, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # TODO: use conv to erase the domain gap between vectormap and image
        outs = []

        x = torch.cat((x, cond_sr_ref), dim=1)

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    def __init__(
        self,
        sr_key: str,
        ref_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        # preprocess_config=None,
        control_stage_config=None,
        # structcond_stage_config=None,
        disable_preprocess=False,
        frozen_diff=True,
        *args,
        **kwargs,
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.sr_key = sr_key
        self.ref_key = ref_key
        self.disable_preprocess = disable_preprocess
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.frozen_diff = frozen_diff

        # instantiate preprocess module (SwinIR)
        # self.preprocess_model = instantiate_from_config(preprocess_config)
        # frozen_module(self.preprocess_model)

        self.lpips_metric = LPIPS(net="alex")

        self.sr_ref_encoder = SR_Ref_Encoder_Spade()
        if self.frozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if "attn" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def apply_cond_ref_encoder(self, control_sr, control_ref):
        return (
            self.sr_ref_encoder(control_sr * 2 - 1, control_ref * 2 - 1)
            * self.scale_factor
        )

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, self.hr_key, *args, **kwargs)
            # x: HR encoded
            # c: conditional text
            sr_cond = batch[self.sr_key]
            ref_cond = batch[self.ref_key]
            hr_cond = batch[self.hr_key]

            if bs is not None:
                sr_cond = sr_cond[:bs]
                ref_cond = ref_cond[:bs]
                hr_cond = hr_cond[:bs]

            sr_cond = sr_cond.to(self.device)
            ref_cond = ref_cond.to(self.device)
            hr_cond = hr_cond.to(self.device)

            sr_cond = einops.rearrange(sr_cond, "b h w c -> b c h w")
            ref_cond = einops.rearrange(ref_cond, "b h w c -> b c h w")
            hr_cond = einops.rearrange(hr_cond, "b h w c -> b c h w")

            sr_cond = sr_cond.to(memory_format=torch.contiguous_format).float()
            ref_cond = ref_cond.to(memory_format=torch.contiguous_format).float()
            hr_cond = hr_cond.to(memory_format=torch.contiguous_format).float()

            lq = sr_cond
            hq = hr_cond

        # apply condition encoder
        sr_ref_cond_latent = self.apply_cond_ref_encoder(sr_cond, ref_cond)

        return x, dict(
            c_crossattn=[c],
            sr_ref_cond_latent=[sr_ref_cond_latent],
            lq=[lq],
            hq=[hq]
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        # ref_cond_latent as context for diffusion u-net
        # semantic = einops.rearrange(cond["ref_cond_latent"][0], 'b c h w -> b (h w) c').contiguous()

        control = self.control_model(
            x=x_noisy,
            cond_sr_ref=cond["sr_ref_cond_latent"][0],
            timesteps=t,
            context=cond_txt,
        )

        control = [c * scale for c, scale in zip(control, self.control_scales)]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=control,
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
        b, c, h, w = cond["sr_ref_cond_latent"][0].shape
        shape = (b, self.channels, h, w)
        samples = sampler.sample(steps, shape, cond)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.control_model.parameters())
            + list(self.sr_ref_encoder.parameters())
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
        self.val_psnr /= self.trainer.datamodule.val_config.dataset.params.data_len / 2
        self.val_lpips /= self.trainer.datamodule.val_config.dataset.params.data_len / 2
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
