import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import benchmark
from tqdm import tqdm

from architectures.flex_modules import FlexFormer

torch._inductor.config.realize_opcount_threshold = 500
torch._dynamo.config.cache_size_limit = 256

num_threads = torch.get_num_threads()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

results = []
for kernel_size in tqdm([3, 5, 7, 9], desc='Kernel'):
    for config in [(64, 128), (128, 64), (256, 32), (512, 16)]:
        n_channel, spatial_dim = config
        label = f'Kernel: {kernel_size}²'
        sub_label = f'{n_channel}x{spatial_dim}²'

        x = torch.randn(1, n_channel, spatial_dim, spatial_dim).to(device)
        results.append(benchmark.Timer(
            globals={'input': x, 'm': torch.compile(nn.Conv2d(n_channel, n_channel, kernel_size, padding=kernel_size // 2).to(device))},
            description='Conv',
            stmt='m(input)',
            num_threads=num_threads,
            label=label,
            sub_label=sub_label).blocked_autorange()
                       )
        x_ = torch.randn(1, spatial_dim * spatial_dim, n_channel).to(device)
        results.append(benchmark.Timer(
            globals={'input': x_, 'm': torch.compile(FlexFormer(torch.tensor([spatial_dim, spatial_dim]), kernel_size, n_channel,
                                                  4, device, 2).to(device))},
            description='LocAttn',
            stmt='m(input)',
            num_threads=num_threads,
            label=label,
            sub_label=sub_label).blocked_autorange()
                       )

results_full_attn = []
for config in tqdm([(64, 128), (128, 64), (256, 32), (512, 16)], desc='Full Attention'):
    n_channel, spatial_dim = config
    sub_label = f'{n_channel}x{spatial_dim}²'

    x_ = torch.randn(1, spatial_dim * spatial_dim, n_channel).to(device)
    results_full_attn.append(benchmark.Timer(
        globals={'input': x_, 'm': torch.compile(nn.TransformerEncoderLayer(n_channel, 4, n_channel * 2, norm_first=True,
                                                              batch_first=True, activation=F.leaky_relu).to(device))},
        description='FullAttn',
        stmt='m(input)',
        num_threads=num_threads,
        sub_label=sub_label).blocked_autorange()
                             )

benchmark.Compare(results).print()
benchmark.Compare(results_full_attn).print()
