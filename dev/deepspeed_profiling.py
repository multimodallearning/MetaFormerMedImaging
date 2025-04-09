import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from torch import nn
from architectures.flex_modules import FlexBlock
import pandas as pd
from tqdm import tqdm


def profile(model: torch.nn.Module, input_shape: tuple[int, ...]):
    flops, macs, params = get_model_profile(model, input_shape, warm_up=10, print_profile=False, as_string=False)

    return flops, macs, params


results_df = pd.DataFrame(columns=['operation', 'kernel', 'spatial', 'flops', 'macs', 'params'])
done_4_full_atn = False
for kernel_size in tqdm([3, 5, 7, 9], desc='Kernel'):
    for config in [(64, 128), (128, 64), (256, 32), (512, 16)]:
        n_channel, spatial_dim = config
        kernel_label = f'{kernel_size}²'
        spatial_label = f'{n_channel}x{spatial_dim}²'

        conv = nn.Conv2d(n_channel, n_channel, kernel_size, padding=kernel_size // 2).cuda()
        flops, macs, params = profile(conv, (1, n_channel, spatial_dim, spatial_dim))
        results_df.loc[len(results_df.index)] = ['conv', kernel_label, spatial_label, flops, macs, params]

        flex = FlexBlock(torch.tensor([spatial_dim, spatial_dim]), kernel_size, n_channel, 4, 1, 4, 'cuda').cuda()
        flops, macs, params = profile(flex, (1, spatial_dim ** 2, n_channel))
        results_df.loc[len(results_df.index)] = ['flex', kernel_label, spatial_label, flops, macs, params]

        if not done_4_full_atn:
            full_attn = nn.TransformerEncoderLayer(n_channel, 4, n_channel * 4, norm_first=True, batch_first=True, activation=torch.nn.functional.leaky_relu).cuda()
            flops, macs, params = profile(full_attn, (1, spatial_dim ** 2, n_channel))
            results_df.loc[len(results_df.index)] = ['full_attn', '-', spatial_label, flops, macs, params]
    done_4_full_atn = True

results_df.to_csv('dev/results.csv')
print(results_df)
