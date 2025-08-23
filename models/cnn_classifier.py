import timm
from clearml import Task
from timm.models.resnet import ResNet

from models.classifier_base import ClassifierBase
from architectures import poolformer as pf
import torch
from torch import nn
from timm.models import adapt_input_conv


class CNNClassifier(ClassifierBase):
    def __init__(self, ds_name: str, model: str = 'resnet18', pretrained: bool = False, kernel_size: int = 3,
                 use_scheduler: bool = True):
        super().__init__(ds_name, use_scheduler=use_scheduler)
        self.model_name = model
        self.ds_name = ds_name
        if self.is_2d:
            self.model = timm.create_model(model, pretrained=pretrained, num_classes=self.n_classes,
                                           in_chans=self.n_channels)
        elif self.is_3d:
            raise NotImplementedError("3D models not implemented yet")

        if kernel_size != 3:
            assert isinstance(self.model, ResNet), "Only ResNet models support kernel size adaptation"
            for i in range(1, 5):
                seq = getattr(self.model, f'layer{i}')
                for j in range(len(seq)):
                    seq[j].conv1 = nn.Conv2d(seq[j].conv1.in_channels, seq[j].conv1.out_channels, kernel_size=kernel_size,
                                             stride=seq[j].conv1.stride, padding=kernel_size // 2, bias=False)
                    seq[j].conv2 = nn.Conv2d(seq[j].conv2.in_channels, seq[j].conv2.out_channels, kernel_size=kernel_size,
                                             stride=seq[j].conv2.stride, padding=kernel_size // 2, bias=False)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.model_name}_{self.ds_name} {self.hparams.kernel_size}²')
            if self.hparams.pretrained:
                Task.current_task().set_tags(['finetuned'])


class ConvFormerClassifier(ClassifierBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True,
                 kernel: int = 3, depthwise: bool = True, drop_path: float = 0.1, rw_percentage: float = 0.4):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        # adapt classifer
        self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)
        # adapt first conv to new number of input channel
        self.model.patch_embed.proj.weight = nn.Parameter(
            adapt_input_conv(self.n_channels, self.model.patch_embed.proj.weight))
        self.model.patch_embed.proj.in_channels = self.n_channels

        # reset pretrained weights before adaptation
        super().reset_pretrained_weights(self.model, rw_percentage)

        if drop_path > 0.0:
            for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)
                        num_channel = blocks[l].norm1.num_channels
                        blocks[l].token_mixer = nn.Conv2d(num_channel, num_channel, kernel, padding=kernel // 2,
                                                          groups=num_channel if depthwise else 1, bias=False)
                        if depthwise:
                            # init as avgpool
                            nn.init.constant_(blocks[l].token_mixer.weight, 1 / (kernel ** 2))

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'conv')
            Task.current_task().set_name(f'{model_name}_{self.hparams.dataset_name}')


if __name__ == '__main__':
    from torchinfo import summary

    m = CNNClassifier('imagewoof', kernel_size=7)
    print(m)
    x = torch.randn(8, 3, 224, 224)
    y_hat = m(x)
    summary(m, (8, 3, 224, 224))
    print(y_hat.shape)
