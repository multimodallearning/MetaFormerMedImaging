from typing import List, Any

import math
import torch
from timm.models import adapt_input_conv
from torch import nn
from torch.nn import functional as F, init
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from architectures import poolformer as pf
from architectures import score_mode_functions
from models.classifier_base import ClassifierBase

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


class FlexTokenMixer(nn.Module):
    def __init__(self, num_channel: int, num_heads: int, block_mask=None, learn_pos_emb: bool = False,
                 use_slopes: bool = False, eps: float = 0.02, init_as_pooling: bool = False):
        """
        FlexTokenMixer with local self-attention to replace AvgPool in PoolFormer. It is initialized to mimic AvgPool.
        :param num_channel: input channel and output channel
        :param num_heads: number of heads used in attention
        :param block_mask: precomputed block mask for local attention
        :param learn_pos_emb: if True, a two-layer MLP on normalized coordinates as learnable position embedding is used
        :param use_slopes: whether to use slopes as relative positional encoding in local attention
        :param eps: std of the normal distribution used to initialize weights
        """
        super().__init__()
        self.num_heads = num_heads
        self.block_mask = block_mask
        if block_mask is not None:
            L = block_mask.shape[-1]
            patch_size = int(L ** 0.5)
        use_slopes = use_slopes and (block_mask is not None)
        self.score_mod_fn = getattr(score_mode_functions, f'wrapper_s4_{patch_size}') if use_slopes else None
        self.learn_pos_emb = learn_pos_emb and (block_mask is not None)
        if self.learn_pos_emb:
            pos_emb = torch.meshgrid([torch.linspace(-1, 1, patch_size)] * 2, indexing='ij')
            pos_emb = torch.stack(pos_emb, -1).view(1, -1, 2)  # (1, L, 2)
            self.register_buffer('pos_emb', pos_emb)
            self.pos_emb_proj = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(), nn.Linear(16, num_channel))
            self.pos_emb_proj.apply(lambda m: self.near_zero_init(m, eps))
        self.kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16}  # todo would be nice to have this optimzed

        self.init_as_pooling = init_as_pooling
        self.out_proj_weights = nn.Parameter(torch.eye(num_channel))
        self.out_proj_bias = nn.Parameter(torch.zeros(num_channel))
        if init_as_pooling:
            self.in_proj_qk_weights = nn.Parameter(torch.randn(2 * num_channel, num_channel) * eps)
            self.in_proj_qk_bias = nn.Parameter(torch.zeros(2 * num_channel))
            self.in_proj_v_weights = nn.Parameter(torch.eye(num_channel))
            self.in_proj_v_bias = nn.Parameter(torch.zeros(num_channel))
        else:
            self.in_proj_qkv_weights = nn.Parameter(torch.randn(3 * num_channel, num_channel) * eps)
            self.in_proj_qkv_bias = nn.Parameter(torch.zeros(3 * num_channel))
            self.random_init()

    @staticmethod
    def near_zero_init(m: nn.Module, var: float):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, var)
            nn.init.constant_(m.bias, 0)

    # adapted from nn.Linear.reset_parameters()
    def random_init(self) -> None:
        layers2init = ['in_proj_qk', 'in_proj_v', 'out_proj'] if self.init_as_pooling else ['in_proj_qkv', 'out_proj']
        for layer_name in layers2init:
            weights = getattr(self, f'{layer_name}_weights')
            init.kaiming_uniform_(weights, a=math.sqrt(5))

            bias = getattr(self, f'{layer_name}_bias')
            fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor, cls: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ = x.flatten(start_dim=2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        if self.learn_pos_emb:
            pos_emb_projected = self.pos_emb_proj(self.pos_emb)
            x_ = x_ + pos_emb_projected
        if cls is not None:  # concat class token
            x_ = torch.cat([cls.unsqueeze(1), x_], dim=1)  # (B, N+1, C)

        if self.init_as_pooling:
            q, k = F.linear(x_, self.in_proj_qk_weights, self.in_proj_qk_bias).chunk(2, dim=-1)
            v = F.linear(x_, self.in_proj_v_weights, self.in_proj_v_bias)
        else:
            q, k, v = F.linear(x_, self.in_proj_qkv_weights, self.in_proj_qkv_bias).chunk(3, dim=-1)

        q_ = self.view4heads(q, self.num_heads)
        k_ = self.view4heads(k, self.num_heads)
        v_ = self.view4heads(v, self.num_heads)

        y_ = flex_attention_compiled(q_, k_, v_, kernel_options=self.kernel_options, block_mask=self.block_mask, #)
        score_mod=self.score_mod_fn)  # (B, M, H*W, C/M)
        y_ = y_.transpose(1, 2).flatten(2)  # (B, M, H*W, C/M) -> (B, H*W, C)
        y_ = F.linear(y_, self.out_proj_weights, self.out_proj_bias)
        if cls is not None:
            cls = y_[:, 0, :]
            y_ = y_[:, 1:, :]  # remove class token
        y = y_.transpose(1, 2).unflatten(2, (H, W))  # (B, N, C) -> (B, C, H, W)
        y -= x  # Subtract residual connection to mimic avgPool during initialization (see. MetaFormer paper Alg.1)
        return y if cls is None else (y, cls)

    @staticmethod
    def view4heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        return x.unflatten(-1, (num_heads, -1)).transpose(2, 1)


class FlexFormer(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int, patch_size: List[int], kernel_size: int, head_dim: int,
                 model_name: str = "poolformer_s12", pretrained: bool = True, use_cls_toke: bool = False,
                 learn_pe: bool = False, use_slopes: bool = False, rpl_patch_emb: bool = False, drop_path: float = 0.1,
                 rw_percentage: float = None, device: str = "cuda"):
        """
        Replace AvgPool in PoolFormer with local self-attention.
        :param n_classes: number of classes for classification head
        :param n_input_channel: number of image's input channels
        :param patch_size: size of the image
        :param kernel_size: size of the local attention kernel
        :param head_dim: number of dimensions per head
        :param model_name: poolformer model name to use
        :param pretrained: rather to use pretrained weights of poolformer or not
        :param use_cls_toke: whether to use global class token
        :param learn_pe: use learnable position embedding in all attention blocks
        :param use_slopes: use slopes in directed local attention
        :param rpl_patch_emb: whether to replace conv with avg_pool patch embedding (i.e. no learnable projection)
        :param drop_path: stochastic depth rate
        :param rw_percentage: percentage of weights to reset (between 0 and 1). If None, no weights are reset.
        :param device: device to use for computation. Has to be given due to block mask generation and compilation
        """
        super().__init__()
        if use_slopes and head_dim != 16:
            raise ValueError(f"head_dim has to be 16 for slopes, but is {head_dim}")
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        # handle CLS token
        self.use_cls_toke = use_cls_toke
        if use_cls_toke:
            n_channels = self.model.patch_embed.proj.out_channels  # dim of first stage
            self.cls = nn.Parameter(torch.randn(n_channels))
            proj_to_stage = []
            for i, seq in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
                new_n_channels = seq[0].norm1.num_channels
                proj_to_stage.append(nn.Linear(n_channels, new_n_channels))
                n_channels = new_n_channels  # prepare for next stage

                # add batch norm to each block
                for block in seq:
                    block.cls_norm = nn.BatchNorm1d(new_n_channels)

            proj_to_stage.pop(0)  # remove first stage projection, since cls was initialized with first stage dim
            proj_to_stage.append(nn.Identity())  # helps for the for loop in forward ;)
            self.proj_to_stage = nn.ModuleList(proj_to_stage)

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
                block_mask = self.generate_block_mask(num_heads, kernel_size, stage_patch_size, device,
                                                      self.use_cls_toke)

            else:
                # print(f'Skipping stage {i}')
                # continue
                block_mask = None  # apply global self attention
            for l in range(len(blocks)):
                num_channel = blocks[l].norm1.num_channels
                enable_pe = (l == 0) and learn_pe
                blocks[l].token_mixer = FlexTokenMixer(num_channel, num_heads, block_mask, learn_pos_emb=enable_pe,
                                                       use_slopes=use_slopes, init_as_pooling=True)
                if not pretrained:
                    blocks[l].token_mixer.random_init()

        if rpl_patch_emb:
            upsample_kwargs = {'mode': 'bilinear', 'align_corners': False}

            old_conv = self.model.patch_embed.proj
            self.model.patch_embed.proj = nn.Sequential(
                nn.Upsample(scale_factor=0.25, **upsample_kwargs),
                nn.Conv2d(old_conv.in_channels, old_conv.out_channels, kernel_size=1, bias=False)
            )

            for m in filter(lambda m: isinstance(m, pf.PatchEmbed), self.model.network.modules()):
                m.proj = nn.Sequential(
                    nn.Upsample(scale_factor=0.5, **upsample_kwargs),
                    nn.Conv2d(m.proj.in_channels, m.proj.out_channels, kernel_size=1, bias=False)
                )
                m.norm = nn.Identity()

    @staticmethod
    def forward_poolformerblock_with_cls(b: pf.PoolFormerBlock, x: torch.Tensor, cls: torch.Tensor) -> (
    torch.Tensor, torch.Tensor):
        # adaption of PoolFormerBlock.forward() to use class token
        z, cls = b.token_mixer(b.norm1(x), cls)
        if b.use_layer_scale:
            x = x + b.drop_path(
                b.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * z)
            x = x + b.drop_path(
                b.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * b.mlp(b.norm2(x)))
        else:
            x = x + b.drop_path(z)
            x = x + b.drop_path(b.mlp(b.norm2(x)))
        # use channel MLP and individual batch norm for cls token. Linear CLS if not used
        cls = b.cls_norm(cls)
        cls = b.mlp(cls.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return x, cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cls_toke:  # add class token
            # input embedding
            x = self.model.forward_embeddings(x)
            cls = self.cls.unsqueeze(0).expand(x.shape[0], -1)  # (B, C)
            stage_idx = 0
            for m in self.model.network:
                if isinstance(m, pf.PatchEmbed):
                    x = m(x)
                elif isinstance(m, nn.Sequential):
                    for block in m:
                        x, cls = self.forward_poolformerblock_with_cls(block, x, cls)
                    cls = self.proj_to_stage[stage_idx](cls)  # project cls token to the next stage dimension
                    stage_idx += 1
                else:
                    raise AssertionError('Unknown module type in PoolFormer network: ' + str(type(m)))
            # use CLS token for classification
            #y_hat = self.model.head(cls)

            # default classification
            x = self.model.norm(x)
            y_hat = self.model.head(x.mean(dim=[-2, -1]))  # global average pooling

        else:  # default forward without class token
            y_hat = self.model(x)

        return y_hat

    @staticmethod
    def generate_block_mask(num_heads: int, kernel: int, patch_size: torch.Tensor, device: str,
                            has_cls: bool = False) -> Any:
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

        def compute_mask_with_cls(b, h, q_idx, kv_idx):
            # unravel index with correction for global cls token at index 0
            q_x = (q_idx - 1) % patch_size[1]
            q_y = (q_idx - 1) // patch_size[1]
            kv_x = (kv_idx - 1) % patch_size[1]
            kv_y = (kv_idx - 1) // patch_size[1]

            # compute mask
            is_valid_x = (q_x - kv_x).abs() <= kernel[0] // 2
            is_valid_y = (q_y - kv_y).abs() <= kernel[1] // 2
            is_valid = is_valid_x & is_valid_y

            # allow connection to global cls token
            is_valid = is_valid | (kv_idx == 0) | (q_idx == 0)
            return is_valid

        S = patch_size.prod().item()
        if has_cls:
            block_mask = create_block_mask(compute_mask_with_cls, None, num_heads, S + 1, S + 1, device, _compile=True)
        else:
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
    for p in [256, 320, 512, 768, 1024]:
        print('patch size', p)
        patch_size = [p]*2
        torch.cuda.reset_peak_memory_stats()
        f = FlexFormer(10, 3, patch_size, 5, 16).cuda()
        #print(f)
        print(f(torch.randn(8, 3, *patch_size).cuda()).shape)
        f.model.fork_feat = True
        f.model.out_indices = [0, 2, 4, 6]
        for i in f.model.out_indices:
            f.model.add_module(f'norm{i}', nn.Identity())
        seg_hat = f(torch.randn(8, 3, *patch_size).cuda())
        for i, feat in enumerate(seg_hat):
            print(i, feat.shape)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memory usage: {peak_memory:.2f} MB\n")
