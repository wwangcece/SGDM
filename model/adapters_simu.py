from ast import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from .vgg16 import SR_Ref_Encoder_Spade, SR_Encoder, SR_Ref_Encoder
from ldm.modules.diffusionmodules.util import (
    linear,
    zero_module,
    timestep_embedding,
)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(
            module: torch.nn.Module,
        ) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class My_Adapter(nn.Module):
    def __init__(
        self,
        channels=[320, 640, 1280],
        nums_rb=2,
        cin=3*64,
        ksize=1,
        sk=True,
        use_conv=True,
    ):
        super(My_Adapter, self).__init__()
        self.channels = channels
        self.nums_rb = nums_rb
        self.merge_encoder = SR_Ref_Encoder_Spade(out_channel=cin*2)
        # w/o-map branch
        # self.merge_encoder = SR_Encoder(out_channel=cin*2)
        # w/o-sgm
        # self.merge_encoder = SR_Ref_Encoder(out_channel=cin*2)
        self.conv_in = nn.Conv2d(cin*2, channels[0], 3, 1, 1)
        self.body = []
        for i in range(len(channels) - 1):
            for j in range(nums_rb):
                if j == 0:
                    self.body.append(
                        ResnetBlock(
                            channels[i],
                            channels[i + 1],
                            down=True,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock(
                            channels[i + 1],
                            channels[i + 1],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
        self.body = nn.ModuleList(self.body)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, sr, ref):
        # unshuffle
        x = self.merge_encoder(sr, ref)
        # x = self.merge_encoder(sr)
        # extract features
        features = []
        x = self.conv_in(x)
        features.append(x)
        for i in range(len(self.channels) - 1):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features

def count_parameters(model):
    """
    计算神经网络的参数量
    Args:
    - model: 继承自nn.Module的神经网络

    Returns:
    - total_params: 参数总量
    - trainable_params: 可训练参数总量
    """
    total_params = 0

    for parameter in model.parameters():
        total_params += parameter.numel()

    return total_params

# model = Adapter_XL(ksize=3, use_conv=True, sk=False)
# param_num = count_parameters(model)
# print(f"{param_num/1e6}M")

# device = "cuda"
# sr = torch.randn(16, 3, 256, 256).to(device)
# ref = torch.randn(16, 3, 256, 256).to(device)

# sr_path = "dataset/exps/data/sr/17_64069_40915.png"
# ref_path = "dataset/exps/data/ref/17_64326_40904.png"
# lr_path = "dataset/dataset/val/lr_8/18_127602_81702.png"
# sr = np.asarray(Image.open(sr_path).convert("RGB"))
# ref = np.asarray(Image.open(ref_path).convert("RGB"))
# lr = np.asarray(Image.open(lr_path).convert("RGB"))

# sr = torch.from_numpy(sr).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
# ref = torch.from_numpy(ref).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
# lr = torch.from_numpy(lr).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)

# encoder = Adapter_XL().to(device)
# latent = encoder(sr)
