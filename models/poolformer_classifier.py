import torch
from clearml import Task
from timm.models import adapt_input_conv
from torch import nn

from architectures import poolformer as pf
from architectures import metaformer as mf
from architectures.flex_token_mixer import FlexFormer
from models.classifier_base import ClassifierBase


class PoolFormerClassifier(ClassifierBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True,
                 train_poolformer: bool = True, lr_poolformer: float = 0.0001, weight_decay: float = 0.05,
                 drop_path: float = 0.1):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        # adapt classifer
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # adapt first conv to new number of input channel
        self.model.patch_embed.proj.weight = nn.Parameter(
            adapt_input_conv(self.n_channels, self.model.patch_embed.proj.weight))
        self.model.patch_embed.proj.in_channels = self.n_channels

        if drop_path > 0.0:
            for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        self.model.requires_grad_(False)
        param_dicts = [{"params": self.model.head.parameters()}, {"params": self.model.norm.parameters()}]
        self.model.head.requires_grad_(True)
        self.model.norm.requires_grad_(True)
        if self.hparams.train_poolformer:
            param_dicts.append({"params": self.model.network.parameters(), "lr": self.hparams.lr_poolformer})
            self.model.network.requires_grad_(True)
            param_dicts.append({"params": self.model.patch_embed.parameters(), "lr": self.hparams.lr_poolformer})
            self.model.patch_embed.requires_grad_(True)

        # use optimizer and scheduler from parent class
        optimizers, schedulers = super().configure_optimizers()
        assert len(optimizers) == 1, "Only one optimizer is supported"
        optimizer = optimizers[0]
        optimizer.param_groups = []
        for param_dict in param_dicts:
            optimizer.add_param_group(param_dict)
        return [optimizer], schedulers

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.hparams.model_name}_{self.hparams.dataset_name}')


class MetaFormerClassifier(ClassifierBase):
    def __init__(self, dataset_name: str, model_name: str = 'metaformer_pppa_s12_224', pretrained: bool = True,
                 drop_path: float = 0.1):
        super().__init__(dataset_name)
        assert self.is_2d, "MetaFormer is only implemented for 2D datasets"
        assert model_name in mf.model_urls, f"Model {model_name} not found in {mf.model_urls.keys()}"
        self.model = getattr(mf, model_name)(pretrained=pretrained)
        # adapt classifer
        self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)
        # adapt first conv to new number of input channel
        self.model.patch_embed.proj.weight = nn.Parameter(
            adapt_input_conv(self.n_channels, self.model.patch_embed.proj.weight))
        self.model.patch_embed.proj.in_channels = self.n_channels

        if drop_path > 0.0:
            for i, blocks in enumerate(filter(lambda m: isinstance(m, nn.Sequential), self.model.network)):
                for l in range(len(blocks)):
                    if hasattr(blocks[l], 'drop_path'):
                        blocks[l].drop_path = pf.DropPath(drop_path)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.hparams.model_name}_{self.hparams.dataset_name}')


class FlexFormerClassifier(ClassifierBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True,
                 num_heads: int = 4, lr: float = 1e-4, patch_size: int = 224, drop_path: float = 0.1):
        super().__init__(dataset_name, lr=lr)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        self.model = FlexFormer(self.n_classes, self.n_channels, [patch_size, patch_size], num_heads, model_name,
                                pretrained, drop_path)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.interpolate(x, size=(self.hparams.patch_size,) * 2, mode='bilinear', align_corners=False)
        y_hat = self.model(x)
        return y_hat

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'flex')
            Task.current_task().set_name(f'{model_name}_{self.hparams.dataset_name}')

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Skipping updates in case of unstable gradients.
    #     Based on https://github.com/Lightning-AI/lightning/issues/4956
    #     """
    #     # create generator for all gradients
    #     grads = (p.grad for p in self.parameters() if p.grad is not None)
    #
    #     for g in grads:
    #         if torch.isnan(g).any() or torch.isinf(g).any():
    #             print("Detected NaN or Inf in gradients. Skipping optimizer step.")
    #             # reset gradients
    #             self.zero_grad(set_to_none=False)
    #             break
    #
    #     super().optimizer_step(*args, **kwargs)


if __name__ == '__main__':
    from datasets.imagewoof_dataset import ImageWoofDataModule
    from datasets.med_mnist_dataset import MedMNISTDataModule
    from tqdm import trange
    from torch.nn import functional as F
    from kornia.augmentation import auto

    poolformer = FlexFormerClassifier('imagewoof', 'poolformer_s12', patch_size=224).cuda()
    # poolformer = PoolFormerClassifier('OrganAMNIST').cuda()
    print(poolformer.model)

    dm = ImageWoofDataModule(batch_size=128, use_data_aug=True, spatial_size=224)
    # dm = MedMNISTDataModule('OrganAMNIST', batch_size=128, spatial_size=224, use_data_aug=True)
    dm.setup('fit')
    train_loader = iter(dm.train_dataloader())
    data_aug = auto.AutoAugment(transformation_matrix_mode='skip')

    optim = torch.optim.Adam(poolformer.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    run_acc = torch.zeros(3000)
    run_loss = torch.zeros(3000)
    for i in trange(3000):
        try:
            imgs, labels = next(train_loader)
        except StopIteration:
            print('Epoch finished, restarting')
            train_loader = iter(dm.train_dataloader())
            imgs, labels = next(train_loader)
        x = imgs.cuda()
        y = labels.squeeze(-1).cuda()
        # affine = F.affine_grid(torch.eye(2, 3).cuda().unsqueeze(0) + torch.randn(128, 2, 3).mul(0.06).cuda(),
        #                        (128, 3, 224, 224), align_corners=False)
        # x = F.grid_sample(x, affine, align_corners=False)  # .expand(-1, 3, -1, -1)
        # x = dm.data_aug(x)
        # x = dm.normalize(x)
        x, y = dm.on_after_batch_transfer((x, y), None)
        # x = data_aug(x)
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = poolformer(x)
            loss = loss_fn(output, y)
        # if loss.isnan():
        #     print('Debug')
        #     print('img', x.isnan().any())
        #     print('y_hat', output.isnan().any())
        #     print('y', label.isnan().any())
        #     raise RuntimeError("Loss is NaN")
        loss.backward()
        optim.step()
        run_acc[i] = (output.argmax(1) == y).float().mean()
        run_loss[i] = loss.item()
        if (i % 50 == 40):
            print(i, run_loss[i - 30:i - 1].mean().item(), run_acc[i - 30:i - 1].mean().item())
