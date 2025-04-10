import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


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

        # attributes for flex_attention
        self.kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32,
                               "BLOCK_N2": 16, }
        self.block_mask = block_mask

    def forward(self, x, x1, x2, attn_mask=None, key_padding_mask=None, need_weights=False, is_causal=False):
        q, k, v = F.linear(x, self.in_proj_weight, bias=self.in_proj_bias).chunk(3, -1)
        q_ = q.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)

        k_ = k.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)
        v_ = v.unflatten(-1, (self.attn_heads, -1)).transpose(2, 1)
        y_ = flex_attention_compiled(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask)
        y = y_.transpose(2, 1).flatten(2, -1)
        y = F.linear(y, self.out_proj_weight, bias=self.out_proj_bias)
        return y


class FlexFormer(nn.TransformerEncoderLayer):
    def __init__(self, in_channel: int, nhead: int, dim_ff_scale: int = 4, dropout: float = 0.1,
                 activation=F.leaky_relu, block_mask=None):
        super().__init__(d_model=in_channel, nhead=nhead, dim_feedforward=in_channel * dim_ff_scale, dropout=dropout,
                         activation=activation, layer_norm_eps=1e-5, batch_first=True, norm_first=True)
        self.self_attn = Flextension(self.self_attn.in_proj_weight, self.self_attn.in_proj_bias,
                                     self.self_attn.out_proj.weight, self.self_attn.out_proj.bias, block_mask,
                                     attn_heads=nhead)


class FlexBlock(nn.Module):
    def __init__(self, patch_size: torch.Tensor, kernel: int, in_channel: int, nhead: int, n_layers: int,
                 dim_ff_scale: int, device: str):
        super().__init__()
        print(n_layers, 'x', in_channel, patch_size.tolist())
        self.kernel = torch.tensor([kernel, kernel], device=device)
        self.patch_size = patch_size.to(device)
        S = patch_size.prod().item()
        block_mask = create_block_mask(self.compute_mask, None, nhead, S, S, device, _compile=True)

        self.layers = nn.ModuleList(
            [FlexFormer(in_channel, nhead, dim_ff_scale, block_mask=block_mask) for _ in range(n_layers)])
        self.use_id_mapping = n_layers > 1

    def forward(self, x):
        identity = x
        # todo pos_emb
        for layer in self.layers:
            x = layer(x)
        if self.use_id_mapping:
            x += identity
        return x

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
    f = FlexFormer(128, 4)
    print(f)
