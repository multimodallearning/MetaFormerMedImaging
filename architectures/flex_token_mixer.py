from typing import List, Any
import math

import torch
from torch import nn
from torch.nn import functional as F, init
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from architectures import poolformer as pf
from architectures import score_mode_functions
from timm.models import adapt_input_conv

from models.classifier_base import ClassifierBase

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)

class FlexTokenMixer(nn.Module):
    def __init__(self, num_channel: int, num_heads: int, block_mask=None, learn_pos_emb: bool = False,
                 use_slopes: bool = False, eps: float = 0.02):
        """
        FlexTokenMixer with local self-attention to replace AvgPool in PoolFormer. It is initialized to mimic AvgPool.
        :param num_channel: input channel and output channel
        :param num_heads: number of heads used in attention
        :param block_mask: precomputed block mask for local attention
        :param learn_pos_emb: if True, a two-layer MLP on normalized coordinates as learnable position embedding is used
        :param eps: std of the normal distribution used to initialize weights
        """
        super().__init__()
        self.num_heads = num_heads
        self.block_mask = block_mask
        if block_mask is not None:
            L = block_mask.shape[-1]
            patch_size = int(L ** 0.5)
        if use_slopes and block_mask is not None:
            self.score_mod_fn = getattr(score_mode_functions, f'wrapper_s4_{patch_size}')
        else:
            self.score_mod_fn = None
        self.learn_pos_emb = learn_pos_emb and (block_mask is not None)
        if self.learn_pos_emb:
            pos_emb = torch.meshgrid([torch.linspace(-1, 1, patch_size)] * 2, indexing='ij')
            pos_emb = torch.stack(pos_emb, -1).view(1, -1, 2)  # (1, L, 2)
            self.register_buffer('pos_emb', pos_emb)
            self.pos_emb_proj = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(), nn.Linear(16, num_channel))
            self.pos_emb_proj.apply(lambda m: self.near_zero_init(m, eps))
        self.kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16,
                               'num_stages': 2}  # todo would be nice to have this optimzed

        self.in_proj_qk_weights = nn.Parameter(torch.randn(2 * num_channel, num_channel) * eps)
        self.in_proj_qk_bias = nn.Parameter(torch.zeros(2 * num_channel))
        self.in_proj_v_weights = nn.Parameter(torch.eye(num_channel))
        self.in_proj_v_bias = nn.Parameter(torch.zeros(num_channel))
        self.out_proj_weights = nn.Parameter(torch.eye(num_channel))
        self.out_proj_bias = nn.Parameter(torch.zeros(num_channel))

    @staticmethod
    def near_zero_init(m: nn.Module, var: float):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, var)
            nn.init.constant_(m.bias, 0)

    # adapted from nn.Linear.reset_parameters()
    def random_init(self) -> None:
        for layer_name in ['in_proj_qk', 'in_proj_v', 'out_proj']:
            weights = getattr(self, f'{layer_name}_weights')
            init.kaiming_uniform_(weights, a=math.sqrt(5))

            bias = getattr(self, f'{layer_name}_bias')
            fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_ = x.flatten(start_dim=2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        if self.learn_pos_emb:
            pos_emb_projected = self.pos_emb_proj(self.pos_emb)
            x_ = x_ + pos_emb_projected
        q, k = F.linear(x_, self.in_proj_qk_weights, self.in_proj_qk_bias).chunk(2, dim=-1)
        v = F.linear(x_, self.in_proj_v_weights, self.in_proj_v_bias)

        q_ = self.view4heads(q, self.num_heads)
        k_ = self.view4heads(k, self.num_heads)
        v_ = self.view4heads(v, self.num_heads)

        y_ = flex_attention_compiled(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask,)
                                     #score_mod=self.score_mod_fn)  # (B, M, H*W, C/M)
        y_ = y_.transpose(1, 2).flatten(2)  # (B, M, H*W, C/M) -> (B, H*W, C)
        y_ = F.linear(y_, self.out_proj_weights, self.out_proj_bias)
        y = y_.transpose(1, 2).unflatten(2, (H, W))  # (B, N, C) -> (B, C, H, W)
        return y - x  # Subtract residual connection to mimic avgPool during initialization (see. MetaFormer paper Alg.1)

    @staticmethod
    def view4heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        return x.unflatten(-1, (num_heads, -1)).transpose(2, 1)


class FlexFormer(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int, patch_size: List[int], kernel_size: int, head_dim: int,
                 model_name: str = "poolformer_s12", pretrained: bool = True, learn_pe: bool = False, use_slopes: bool = False,
                 drop_path: float = 0.1, rw_percentage: float = 0.4, device: str = "cuda"):
        """
        Replace AvgPool in PoolFormer with local self-attention.
        :param n_classes: number of classes for classification head
        :param n_input_channel: number of image's input channels
        :param patch_size: size of the image
        :param kernel_size: size of the local attention kernel
        :param head_dim: number of dimensions per head
        :param model_name: poolformer model name to use
        :param pretrained: rather to use pretrained weights of poolformer or not
        :param learn_pe: use learnable position embedding in all attention blocks
        :param use_slopes: use slopes in directed local attention
        :param drop_path: stochastic depth rate
        :param rw_percentage: percentage of weights to reset
        :param device: device to use for computation. Has to be given due to block mask generation and compilation
        """
        super().__init__()
        if use_slopes and head_dim != 16:
            raise ValueError(f"head_dim has to be 16 for slopes, but is {head_dim}")
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

        # resetting pretrained weights
        if pretrained:
            ClassifierBase.reset_pretrained_weights(self.model, rw_percentage)

        patch_size = torch.tensor(patch_size)
        embed_dim = self.model.patch_embed.proj.out_channels
        for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
            if drop_path > 0.:
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

            stage_patch_size = patch_size / (4 * 2 ** i)
            stage_embed_dim = embed_dim * 2 ** i if embed_dim * 2 ** i != 256 else 320  # handle outlier of stage 3
            num_heads = stage_embed_dim // head_dim
            if stage_patch_size.prod() > 64:  # apply local self attention only when it is worth it
                block_mask = self.generate_block_mask(num_heads, kernel_size, stage_patch_size, device)

            else:
                # print(f'Skipping stage {i}')
                # continue
                block_mask = None  # apply global self attention
            for l in range(len(blocks)):
                num_channel = blocks[l].norm1.num_channels
                enable_pe = (l == 0) and learn_pe
                blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, learn_pos_emb=enable_pe,
                                                       use_slopes=use_slopes)
                if not pretrained:
                    blocks[l].token_mixer.random_init()

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
    f = FlexFormer(10, 3, [224, 224], 7, 32).cuda()
    print(f)
    print(f(torch.randn(128, 3, 224, 224).cuda()).shape)
    # mask = FlexFormer.generate_block_mask(4, 3, torch.tensor([64, 64]), 'cpu')
    # pass
