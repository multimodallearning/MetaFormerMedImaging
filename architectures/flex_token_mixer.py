import math
from typing import List, Any

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from architectures import poolformer as pf

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


class FlexTokenMixer(nn.Module):
    def __init__(self, num_channel: int, num_heads: int, block_mask=None):
        super().__init__()
        self.num_heads = num_heads

        # flex_attention currently only supports embed_dim of power of 2, finding the closest power of 2
        lower = 2 ** math.floor(math.log2(num_channel))
        upper = 2 ** math.ceil(math.log2(num_channel))
        embed_dim = lower if abs(num_channel - lower) <= abs(num_channel - upper) else upper

        self.in_proj = nn.Linear(num_channel, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, num_channel)
        self.block_mask = block_mask
        self.kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32,
                               "BLOCK_N2": 16, }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        # (B, C, H, W) -> (B, N, C)
        x = x.flatten(start_dim=2).transpose(1, 2)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q_ = self.view4heads(q, self.num_heads)
        k_ = self.view4heads(k, self.num_heads)
        v_ = self.view4heads(v, self.num_heads)

        y_ = flex_attention_compiled(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask)
        y = y_.transpose(2, 1).flatten(2, -1)
        y = self.out_proj(y)

        # (B, N, C) -> (B, C, H, W)
        y = y.transpose(1, 2).unflatten(2, (H, W))
        return y

    @staticmethod
    def view4heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        return x.unflatten(-1, (num_heads, -1)).transpose(2, 1)


class FlexFormer(nn.Module):
    def __init__(self, n_classes: int, patch_size: List[int], num_heads: int, model_name: str = "poolformer_s12",
                 pretrained: bool = False,
                 device: str = "cuda"):
        super().__init__()
        assert not pretrained, "Not implemented yet"
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

        patch_size = torch.tensor(patch_size)
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
            block_mask = self.generate_block_mask(num_heads, 3, patch_size / (4 * 2 ** i), device)
            for l in range(len(blocks)):
                num_channel = blocks[l].norm1.num_channels
                blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def generate_block_mask(num_heads: int, kernel: int, patch_size: torch.Tensor, device: str) -> Any:
        kernel = torch.tensor([kernel, kernel], device=device)
        patch_size = patch_size.int().to(device)

        def compute_mask(b, h, q_idx, kv_idx):
            # unravel index
            q_x = q_idx % patch_size[1]
            q_y = q_idx // patch_size[1]
            kv_x = kv_idx % patch_size[1]
            kv_y = kv_idx // patch_size[1]

            # compute mask
            is_valid_x = (q_x - kv_x).abs() <= kernel[0] // 2
            is_valid_y = (q_y - kv_y).abs() <= kernel[1] // 2
            is_valid = is_valid_x & is_valid_y
            return is_valid

        S = patch_size.prod().item()
        block_mask = create_block_mask(compute_mask, None, num_heads, S, S, device, _compile=True)
        return block_mask


if __name__ == '__main__':
    f = FlexFormer(2, [224, 224], 4, device='cuda').cuda()
    print(f)
    print(f(torch.randn(128, 3, 224, 224).cuda()).shape)
