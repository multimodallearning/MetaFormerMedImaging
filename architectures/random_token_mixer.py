# heavily inspired by https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L299
from typing import List

from torch import nn
import torch
import math


class RandomMixer(nn.Module):
    def __init__(self, patch_size=List[int]):
        super().__init__()
        self.patch_size = patch_size
        num_tokens = math.prod(patch_size)
        self.register_buffer('random_matrix', torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1))

    def forward_original(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x

    def forward(self, x: torch.Tensor):
        x_ = x.flatten(start_dim=2)
        y_ = torch.einsum('mn, bcn -> bcm', self.random_matrix, x_)
        y = y_.unflatten(-1, self.patch_size)
        return y


if __name__ == '__main__':
    # 2D
    x = torch.randn(32, 64, 16, 16)
    m = RandomMixer([16, 16])
    y_hwc = m.forward_original(x.permute(0, 2, 3, 1))
    y_chw = m(x)
    print(y_hwc.shape, y_chw.shape)
    print(torch.equal(y_hwc.permute(0, -1, 1, 2), y_chw))

    # 3D
    x = torch.randn(32, 64, 16, 16, 16)
    m = RandomMixer([16, 16, 16])
    y_chw = m(x)
    print(y_chw.shape)
