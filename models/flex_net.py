from typing import List

import torch
from clearml import Task
from torch import nn

from architectures.flex_modules import FlexBlock
from models.med_mnist_base import MedMNISTBase

torch._inductor.config.realize_opcount_threshold = 500
torch._dynamo.config.cache_size_limit = 128


class PoolWithChannelExpansion(nn.Module):
    def __init__(self, patch_size, in_channel, out_channel, kernel_size, stride, padding, pool_op='avg'):
        super().__init__()
        self.patch_size = patch_size
        self.unflatten = nn.Unflatten(2, patch_size.tolist())
        if pool_op == 'avg':
            self.pool = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride, padding),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0)
            )
        elif pool_op == 'max':
            self.pool = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride, padding),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0)
            )
        elif pool_op == 'conv_block':
            self.pool = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False, groups=in_channel),
                nn.InstanceNorm2d(out_channel, affine=True),
                nn.LeakyReLU()
            )
        elif pool_op == 'conv':
            self.pool = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False, groups=in_channel)
        else:
            raise ValueError(f"Unknown pooling operation: {pool_op}")

    def forward(self, x):
        # x (B, H*W, C)
        x = x.permute(0, 2, 1)  # (B, C, H*W)
        x = self.unflatten(x)  # (B, C, H, W)
        y = self.pool(x)
        y_ = y.flatten(2, 3)  # (B, C', H*W)
        y_ = y_.permute(0, 2, 1)  # (B, H*W, C')
        return y_


class FlexNetPoolingModel(nn.Module):
    def __init__(self, n_channel: int, n_classes: int, n_heads: int = 4, ff_dim_scale: int = 2,
                 group_repeats: List[int] = [2, 2, 2, 2], patch_size: List[int] = [224, 224], pool_op: str = 'conv',
                 device: str = 'cuda'):
        super().__init__()
        patch_size = torch.tensor(patch_size)
        flex_kwargs = dict(nhead=n_heads, device=device, dim_ff_scale=ff_dim_scale)

        n_out_channels = 64
        self.first_layer = nn.Sequential(
            PoolWithChannelExpansion(patch_size.clone(), n_channel, n_out_channels, 7, 2, 3),
            FlexBlock(patch_size.clone() // 2, 3, n_out_channels, n_layers=1, **flex_kwargs),
        )
        patch_size //= 2

        self.layers = nn.ModuleList()
        for i, repeats in enumerate(group_repeats):
            if i == 0:  # max pool of second group
                self.layers.append(
                    PoolWithChannelExpansion(patch_size.clone(), n_out_channels, n_out_channels, 3, 2, 1, 'max'))
            else:
                self.layers.append(
                    PoolWithChannelExpansion(patch_size.clone(), n_out_channels, n_out_channels * 2, 3, 2, 1, pool_op))
                n_out_channels *= 2
            patch_size //= 2
            print(f'Group {i + 1}: {patch_size.tolist()} with {n_out_channels} channels')
            self.layers.append(FlexBlock(patch_size.clone(), 3, n_out_channels, n_layers=2 * repeats, **flex_kwargs))
        self.classifier = nn.Linear(n_out_channels, n_classes)

    def forward(self, x):
        # view for transformer
        x_ = x.flatten(2).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        x_ = self.first_layer(x_)
        for i, layer in enumerate(self.layers):
            x_ = layer(x_)
        # avg pooling
        x_ = x_.mean(1)
        y_hat = self.classifier(x_)
        return y_hat


class FlexNetPooling(MedMNISTBase):
    def __init__(self, dataset_name: str, n_heads: int = 4, ff_dim_scale: int = 4, patch_size: List[int] = [224, 224],
                 group_repeats: List[int] = [2, 2, 2, 2], lr: float = 1e-4, pool_op: str = 'conv', device: str = 'cuda'):
        super().__init__(dataset_name, lr)
        self.model = FlexNetPoolingModel(self.n_channels, self.n_classes, n_heads, ff_dim_scale, group_repeats,
                                         patch_size, pool_op, device)

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        if Task.current_task() is not None:
            Task.current_task().set_name(f'FlexAvgPoolNet_{self.ds_name}')
            Task.current_task().set_tags([f'{self.ds_name}', 'FlexAvgPoolNet'])


if __name__ == '__main__':
    # from torchinfo import summary
    patch_size = [224, 224]
    f = FlexNetPooling('BreastMNIST', group_repeats=[2, 2, 2], device='cuda', patch_size=patch_size).cuda()
    x = torch.randn(2, 1, *patch_size).cuda()
    print(x.shape)
    y = f(x)
    print(y.shape)
    print(f(torch.randn(4, 1, *patch_size).cuda()).shape)
    # #summary(f, (1, 1, 224, 224))
