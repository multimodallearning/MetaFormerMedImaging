from clearml import Task
from monai.networks.nets.unet import UNet

from models.segmentator_base import SegmentatorBase


class UNetSegmentator(SegmentatorBase):
    def __init__(self, ds_name: str, size:str = 's'):
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
            spatial_dims=2,
            in_channels=self.n_channels,
            out_channels=self.n_classes,
            channels=channels,
            strides=[2] * (len(channels)-1),
            act='leakyrelu',
            norm=('Instance', {'affine': True}),
            bias=False,
            num_res_units=1
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'unet{self.hparams.size}_{self.hparams.ds_name}')
        super().on_fit_start()

if __name__ == '__main__':
    import torch
    m = UNetSegmentator('wristbone', 's')
    print(m)
    x = torch.randn(2, 1, 384, 224)
    y_hat = m(x)
    print(y_hat.shape)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(n_params)