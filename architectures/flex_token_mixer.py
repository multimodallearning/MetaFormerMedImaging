from typing import List, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from architectures import poolformer as pf
from timm.models import adapt_input_conv

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def noscore(score, b, h, q_idx, kv_idx):
    return score * 0 + 1


class FlexTokenMixer(nn.Module):
    def __init__(self, num_channel: int, num_heads: int, block_mask=None, learn_pos_emb: bool = True,
                 eps: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.block_mask = block_mask
        self.learn_pos_emb = learn_pos_emb and (block_mask is not None)
        if self.learn_pos_emb:
            _, H, L, _ = block_mask.shape
            assert H == num_heads
            self.pos_emb = nn.Parameter(torch.zeros(L))
        self.kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16,
                               'num_stages': 2}  # todo would be nice to have this optimzed

        self.in_proj_qk_weights = nn.Parameter(torch.randn(2 * num_channel, num_channel) * eps)
        self.in_proj_qk_bias = nn.Parameter(torch.zeros(2 * num_channel))
        self.in_proj_v_weights = nn.Parameter(torch.eye(num_channel))
        self.in_proj_v_bias = nn.Parameter(torch.zeros(num_channel))
        self.out_prof_weights = nn.Parameter(torch.eye(num_channel))
        self.out_proj_bias = nn.Parameter(torch.zeros(num_channel))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_ = x.flatten(start_dim=2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        q, k = F.linear(x_, self.in_proj_qk_weights, self.in_proj_qk_bias).chunk(2, dim=-1)
        v = F.linear(x_, self.in_proj_v_weights, self.in_proj_v_bias)

        q_ = self.view4heads(q, self.num_heads)
        k_ = self.view4heads(k, self.num_heads)
        v_ = self.view4heads(v, self.num_heads)

        y_ = flex_attention_compiled(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask,
                                     score_mod=self.add_pos_embed if self.learn_pos_emb else None)  # (B, M, H*W, C/M)
        y_ = y_.transpose(1, 2).flatten(2)  # (B, M, H*W, C/M) -> (B, H*W, C)
        y_ = F.linear(y_, self.out_prof_weights, self.out_proj_bias)
        y = y_.transpose(1, 2).unflatten(2, (H, W))  # (B, N, C) -> (B, C, H, W)
        return y - x  # Subtract residual connection to mimic avgPool during initialization (see. MetaFormer paper Alg.1)

    @staticmethod
    def view4heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        return x.unflatten(-1, (num_heads, -1)).transpose(2, 1)

    def add_pos_embed(self, score, b, h, q_idx, kv_idx):
        return score + self.pos_emb[q_idx]


class FlexFormer(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int, patch_size: List[int], num_heads: int,
                 model_name: str = "poolformer_s12", pretrained: bool = True, learn_pe: bool = True,
                 drop_path: float = 0.1, device: str = "cuda"):
        super().__init__()
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        if self.model.head.out_features != n_classes:
            print('Replacing classifier head for new numbers of classes.')
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)
            if n_input_channel != 3:
                print('Reusing first conv weights and adapt to new number of input channel')
                self.model.patch_embed.proj.weight = nn.Parameter(
                    adapt_input_conv(n_input_channel, self.model.patch_embed.proj.weight))
                self.model.patch_embed.proj.in_channels = n_input_channel

        patch_size = torch.tensor(patch_size)
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            stage_patch_size = patch_size / (4 * 2 ** i)
            if stage_patch_size.prod() > 64:  # apply local self attention only when it is worth it
                block_mask = self.generate_block_mask(num_heads, 3, stage_patch_size, device)

            else:
                # print(f'Skipping stage {i}')
                # continue
                block_mask = None
            for l in range(len(blocks)):
                num_channel = blocks[l].norm1.num_channels
                blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, learn_pe)

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


# is not recommended to use this class, since the block mask is calculated for each block (in FlexFormer it is done once per stage)
class FlexTokenBlock(pf.PoolFormerBlock):
    def __init__(self, n_heads: int, patch_size: List[int], device: str, dim: int, pool_size: int = 3,
                 mlp_ratio: int = 4.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = pf.GroupNorm,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5):
        super().__init__(dim, pool_size, mlp_ratio, act_layer, norm_layer, drop, drop_path, use_layer_scale,
                         layer_scale_init_value)
        block_mask = FlexFormer.generate_block_mask(n_heads, pool_size, torch.tensor(patch_size), device)
        self.token_mixer = FlexTokenMixer(dim, n_heads, block_mask)


if __name__ == '__main__':
    f = FlexFormer(10, 3, [224, 224], 4).cuda()
    print(f)
    print(f(torch.randn(128, 3, 224, 224).cuda()).shape)
    # mask = FlexFormer.generate_block_mask(4, 3, torch.tensor([64, 64]), 'cpu')
    # pass
