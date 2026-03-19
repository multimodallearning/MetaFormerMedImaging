from clearml import Task
from monai.networks.nets.unet import UNet
from architectures.poolformer import PatchEmbed
from torch import nn

from models.segmentator_base import SegmentatorBase


class UNetSegmentator(SegmentatorBase):
    def __init__(self, ds_name: str, size:str = 's', conv_kernel:int=3, spatial_dim:int=2):
        """
        Plain UNet for Segmentation. Stage channel numbers are inspired by corresponding MetaFormer sizes.
        But it adds a stage with channel number following the previous pattern.
        :param ds_name: dataset name has to be one of 'jsrt', 'wristbone', 'tiger'
        :param size: Size inspired by corresponding MetaFormer sizes. Either 'S' or 'M'.
        :param conv_kernel: kernel size for convolution and up-convolution layers.
        :param spatial_dim: the spatial dimensionality of the input
        """
        super().__init__(ds_name)
        size = size.upper()
        assert size in ['S', 'M']
        if size == 'S':
            channels = [64, 128, 320, 512, 1024]
        elif size == 'M':
            channels = [96, 192, 384, 768, 1536]
        else:
            raise NotImplementedError(f"Size {size} is not implemented. Use 'S' or 'M'.")

        self.model = UNet(
            spatial_dims=spatial_dim,
            kernel_size=conv_kernel,
            up_kernel_size=conv_kernel,
            in_channels=self.n_channels,
            out_channels=self.n_classes,
            channels=channels,
            strides=[2] * (len(channels)-1),
            act='leakyrelu',
            norm=('Instance', {'affine': True}),
            bias=False,
            num_res_units={3:2, 5:1, 7:1, 9:1}[conv_kernel]
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'unet{self.hparams.size}_{self.hparams.ds_name}')

class UNetSegmentator3D(UNetSegmentator):
    def __init__(self, ds_name: str, size:str = 's', conv_kernel:int=3):
        super().__init__(ds_name, size, conv_kernel, 3)

class UNetOnPatchEmbedding(SegmentatorBase):
    def __init__(self, ds_name: str, size:str = 's', conv_kernel:int=3, in_patch_size:int=7, in_stride:int=4, in_padding=2):
        """
        UNet applied at the same first patch embedding layer as in MetaFormer architectures to match its receptive field.
        This time the number if stages is the same as in MetaFormer architectures.
        :param ds_name: dataset name has to be one of 'jsrt', 'wristbone', 'tiger'
        :param size: Size inspired by corresponding MetaFormer sizes. Either 'S' or 'M'.
        :param conv_kernel: kernel size for convolution and up-convolution layers.
        :param in_patch_size: kernel size for the first patch embedding layer.
        :param in_stride: stride for the first patch embedding layer.
        :param in_padding: padding for the first patch embedding layer.
        """
        super().__init__(ds_name)
        size = size.upper()
        assert size in ['S', 'M']
        if size == 'S':
            channels = [64, 128, 320, 512]
        elif size == 'M':
            channels = [96, 192, 384, 768]
        else:
            raise NotImplementedError(f"Size {size} is not implemented. Use 'S' or 'M'.")

        self.patch_embedding = PatchEmbed(in_patch_size, in_stride, in_padding, self.n_channels, channels[0])
        self.model = UNet(
            spatial_dims=2,
            kernel_size=conv_kernel,
            up_kernel_size=conv_kernel,
            in_channels=channels[0],
            out_channels=self.n_classes,
            channels=channels,
            strides=[2] * (len(channels)-1),
            act='leakyrelu',
            norm=('Instance', {'affine': True}),
            bias=False,
            num_res_units={3:6, 5:4, 7:4, 9:3}[conv_kernel]
        )
        self.upsample = nn.Upsample(scale_factor=in_stride, mode='bilinear', align_corners=False)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.patch_embedding(x)
        y_hat = self.model(x)
        y_hat = self.upsample(y_hat)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'unetOnPatchEmb_{self.hparams.size}_{self.hparams.ds_name}')

if __name__ == '__main__':
    import torch
    m = UNetSegmentator('wristbone', 's', 9)
    print(m)
    x = torch.randn(2, 1, 384, 224)
    y_hat = m(x)
    print(y_hat.shape)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(n_params)