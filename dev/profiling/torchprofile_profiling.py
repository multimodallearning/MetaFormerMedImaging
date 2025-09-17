import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torchprofile import profile_macs
from torch import nn
from architectures.flex_token_mixer import FlexTokenBlock
from architectures.poolformer import PoolFormerBlock
import pandas as pd
from tqdm import tqdm

results_df = pd.DataFrame(columns=['operation', 'kernel', 'spatial', 'macs'])
done_4_full_atn = False
for kernel_size in tqdm([3, 5, 7, 9], desc='Kernel'):
    for config in [(64, 128), (128, 64), (256, 32), (512, 16), (1024, 8), (2048, 4)]:
        n_channel, spatial_dim = config
        kernel_label = f'{kernel_size}²'
        spatial_label = f'{n_channel}x{spatial_dim}²'
        x = torch.rand(1, n_channel, spatial_dim, spatial_dim, device='cuda')

        conv = nn.Conv2d(n_channel, n_channel, kernel_size, padding=kernel_size // 2).cuda()
        macs = profile_macs(conv, x)
        results_df.loc[len(results_df.index)] = ['conv', kernel_label, spatial_label, macs]

        pool = PoolFormerBlock(n_channel, kernel_size).cuda()
        macs = profile_macs(pool, x)
        results_df.loc[len(results_df.index)] = ['pool', kernel_label, spatial_label, macs]
        #
        flex = FlexTokenBlock(4, [spatial_dim, spatial_dim], 'cuda', n_channel, kernel_size).cuda()
        try:
            macs = profile_macs(flex, x)
            results_df.loc[len(results_df.index)] = ['flex', kernel_label, spatial_label, macs]
        except:
            results_df.loc[len(results_df.index)] = ['flex', kernel_label, spatial_label, 'N/A']

        if not done_4_full_atn:
            full_attn = nn.TransformerEncoderLayer(n_channel, 4, n_channel * 4, norm_first=True, batch_first=True,
                                                   activation=torch.nn.functional.leaky_relu).cuda()
            macs = profile_macs(full_attn, x.flatten(2).transpose(1, 2))
            results_df.loc[len(results_df.index)] = ['full_attn', '-', spatial_label, macs]
    done_4_full_atn = True

#results_df.to_csv('dev/results.csv')
print(results_df)
