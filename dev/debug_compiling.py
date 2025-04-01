# Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
import os

# os.environ['TORCH_LOGS'] = '+dynamo'
# os.environ['TORCHDYNAMO_VERBOSE'] = '1'

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
from functools import reduce
from models.flex_modules import FlexFormer

patch_size = [32, 32]
S = reduce(lambda x, y: x * y, patch_size)
kernel = [3, 3]
H = 4

def compute_mask(b, h, q_idx, kv_idx):
    # unravel index
    q_x = q_idx % 32
    q_y = q_idx // 32
    kv_x = kv_idx % 32
    kv_y = kv_idx // 32

    # compute mask
    is_valid_x = (q_x - kv_x).abs() <= 3 // 2
    is_valid_y = (q_y - kv_y).abs() <= 3 // 2
    is_valid = is_valid_x & is_valid_y
    return is_valid


# x = torch.randn(1, H, S, 32).cuda()
# mask = create_block_mask(compute_mask, None, 4, S, S, 'cuda', _compile=True)
# attn = torch.compile(flex_attention)
# y_ = attn(x, x, x, block_mask=mask)
# print(y_.shape)

x = torch.randn(1, S, 16*H).cuda()
m = FlexFormer(torch.tensor(patch_size), 3, 16*H, H, 'cuda').cuda()
m = torch.compile(m)
y = m(x)
print(y.shape)




