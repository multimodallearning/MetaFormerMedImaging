from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class Flextension(nn.Module):
    def __init__(self, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, block_mask, attn_heads):
        super().__init__()
        # weight and bias for linear layers
        self.in_proj_weight = in_proj_weight
        self.in_proj_bias = in_proj_bias
        self.out_proj_weight = out_proj_weight
        self.out_proj_bias = out_proj_bias
        # attributes
        self.attn_heads = attn_heads
        self.batch_first = True
        self._qkv_same_embed_dim = True

        self.flex_attention = torch.compile(flex_attention, dynamic=False)

        # attributes for flex_attention
        self.kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32,
                              "BLOCK_N2": 16, }
        self.block_mask = block_mask

    def forward(self, x, x1, x2, attn_mask=None, key_padding_mask=None, need_weights=False, is_causal=False):
        q, k, v = F.linear(x, self.in_proj_weight, bias=self.in_proj_bias).chunk(3, -1)
        q_ = q.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)

        k_ = k.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)
        v_ = v.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)
        y_ = self.flex_attention(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask)
        y = y_.transpose(2, 1).flatten(2, -1)
        y = F.linear(y, self.out_proj_weight, bias=self.out_proj_bias)
        return y


class FlexFormer(nn.TransformerEncoderLayer):
    def __init__(self, patch_size: torch.Tensor, kernel: int, in_channel: int, nhead: int, device: str,
                 dim_ff_scale: int = 4, dropout: float = 0.1, activation=F.leaky_relu, pos_emb=None,
                 pe_learnable=False):
        super().__init__(d_model=in_channel, nhead=nhead, dim_feedforward=in_channel * dim_ff_scale, dropout=dropout,
                         activation=activation, layer_norm_eps=1e-5, batch_first=True, norm_first=True)

        print(patch_size.tolist(), in_channel)
        assert (patch_size % 2 == 0).all(), "Patch size must be even"
        self.kernel = torch.tensor([kernel, kernel], device=device)
        self.patch_size = patch_size.to(device)
        S = patch_size.prod().item()
        self.block_mask = create_block_mask(self.compute_mask, None, nhead, S, S, device, _compile=True)
        self.self_attn = Flextension(self.self_attn.in_proj_weight, self.self_attn.in_proj_bias,
                                     self.self_attn.out_proj.weight, self.self_attn.out_proj.bias, self.block_mask,
                                     attn_heads=nhead)

        # positional embedding
        if pos_emb is None:
            pos_prior = torch.zeros(1, S, len(patch_size), device=device)
        elif pos_emb == 'rnd':
            pos_prior = torch.randn(1, S, len(patch_size), device=device)
        elif pos_emb == 'coord':
            pos_prior = torch.meshgrid(*[torch.linspace(-1, 1, d.item()) for d in patch_size])
            pos_prior = torch.stack(pos_prior, -1).flatten(0, -2).unsqueeze(0)
        else:
            raise ValueError(f"Unknown positional prior: {pos_emb}")
        self.pe_proj = nn.Linear(len(patch_size), in_channel)
        if pe_learnable:
            self.pos_emb = nn.Parameter(pos_prior)
        else:
            self.register_buffer('pos_emb', pos_prior)

    def forward(self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,):
        # add positional embedding
        x = src + self.pe_proj(self.pos_emb)
        return super().forward(x, src_mask, src_key_padding_mask, is_causal)

    def compute_mask(self, b, h, q_idx, kv_idx):
        # unravel index
        q_x = q_idx % self.patch_size[1]
        q_y = q_idx // self.patch_size[1]
        kv_x = kv_idx % self.patch_size[1]
        kv_y = kv_idx // self.patch_size[1]

        # compute mask
        is_valid_x = (q_x - kv_x).abs() <= self.kernel[0] // 2
        is_valid_y = (q_y - kv_y).abs() <= self.kernel[1] // 2
        is_valid = is_valid_x & is_valid_y
        return is_valid


if __name__ == '__main__':
    #from torchinfo import summary

    patch_size = torch.tensor([16, 16])
    f = FlexFormer(patch_size, 3, 64, 4, 'cuda', pos_prior=None).cuda()
    print(f)
    f = torch.compile(f, dynamic=True)
    x = torch.rand(1, patch_size.prod().item(), 64).cuda()
    y = f(x)
    print(x.shape, y.shape)
    #summary(f, input_size=(1, patch_size.prod().item(), 64))
