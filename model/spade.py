import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc=3):
        super().__init__()
        # self.param_free_norm = SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm2d(norm_nc, affine=True))
        # self.param_free_norm = nn.GroupNorm(32, norm_nc)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=True)
        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=True)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        # segmap = segmap[str(x.size(-1))]
        normalized = self.param_free_norm(x)
        # normalized = x

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SpadeResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spade1 = SPADE(norm_nc=dim)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        self.spade2 = SPADE(norm_nc=dim)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, input):
        x, segmap = input[0], input[1]
        out = self.spade1(x, segmap)
        out = self.silu1(out)
        out = self.conv1(out)

        out = self.spade2(out, segmap)
        out = self.silu2(out)
        out = self.conv2(out)
        return out + x
