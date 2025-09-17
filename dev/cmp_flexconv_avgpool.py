from torch import nn
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn import functional as F

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def generate_block_mask(num_heads: int, kernel: int, patch_size: torch.Tensor, device: str):
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


def flexpooling(x: torch.Tensor, pool_size: int, device: str, kernel_options: dict) -> torch.Tensor:
    B, C, H, W = x.shape
    block_mask = generate_block_mask(1, pool_size, torch.tensor([H, W], device=device), device)

    q = k = torch.zeros(B, 1, H * W, C, device=device)
    v = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)
    v = v.unsqueeze(1)

    y_ = flex_attention_compiled(q, k, v, kernel_options=kernel_options, block_mask=block_mask)  # (B, 1, H*W, C)
    y = y_.transpose(2, 3).unflatten(-1, (H, W)).squeeze(1)  # (B, 1, C, H*W) -> (B, C, H, W)

    return y

def view4heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return x.unflatten(-1, (num_heads, -1)).transpose(2, 1)

def flexconv_init_as_pool(x: torch.Tensor, n_heads:int, pool_size: int, device: str, kernel_options: dict) -> torch.Tensor:
    B, C, H, W = x.shape
    block_mask = generate_block_mask(n_heads, pool_size, torch.tensor([H, W], device=device), device)
    in_proj_kq_weights = torch.zeros(2 * C, C, device=device)
    in_proj_kq_bias = torch.zeros(2 * C, device=device)
    in_proj_v_weights = torch.eye(C, device=device)
    in_proj_v_bias = torch.zeros(C, device=device)
    out_prof_weights = torch.eye(C, device=device)
    out_proj_bias = torch.zeros(C, device=device)

    x_ = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)
    q, k = F.linear(x_, in_proj_kq_weights, in_proj_kq_bias).chunk(2, dim=-1)
    v = F.linear(x_, in_proj_v_weights, in_proj_v_bias)

    q = view4heads(q, n_heads)
    k = view4heads(k, n_heads)
    v = view4heads(v, n_heads)

    y_ = flex_attention_compiled(q, k, v, kernel_options=kernel_options, block_mask=block_mask)  # (B, M, H*W, C/M)
    y_ = y_.transpose(1, 2).flatten(2)  # (B, M, H*W, C/M) -> (B, H*W, C)
    y_ = F.linear(y_, out_prof_weights, out_proj_bias)
    y = y_.transpose(1, 2).unflatten(-1, (H, W))  # (B, C, H*W) -> (B, C, H, W)

    return y


pool_size = 3
device = 'cpu'
# 16 is the lowest allowed value for BLOCK_M and BLOCK_N
kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16, 'num_stages': 2}

avg_pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

# resolutions of different poolformer stages
img_224 = [(64, 56),
           (128, 28),
           (256, 14),  # not included in the original sizes
           (320, 14),
           (512, 7)]

img_128 = [(64, 32),
           (128, 16),
           (256, 8),  # not included in the original sizes
           # (320, 8),
           (512, 4)]

for num_channel, spatial_dim in img_224:
    x = torch.randn(128, num_channel, spatial_dim, spatial_dim, device=device)
    y_pool = avg_pool(x)
    # y_flex = flexpooling(x, pool_size, device, kernel_options)
    y_flex = flexconv_init_as_pool(x, 4, pool_size, device, kernel_options)

    print(x.shape, x.norm())
    print(y_pool.shape, y_pool.norm())
    print(y_flex.shape, y_flex.norm())
    print('Difference:', (y_pool - y_flex).norm())
    print('\n')
