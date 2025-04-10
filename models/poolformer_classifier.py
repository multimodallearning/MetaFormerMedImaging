import torch
from clearml import Task
from torch import nn

from architectures import poolformer as pf
from models.med_mnist_base import MedMNISTBase
from architectures.flex_token_mixer import FlexFormer


class PoolFormerClassifier(MedMNISTBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained: bool = True,
                 train_poolformer: bool = False, lr_poolformer: float = 0.0001, weight_decay: float = 0.05):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        assert model_name in pf.model_urls, f"Model {model_name} not found in {pf.model_urls.keys()}"
        self.model = getattr(pf, model_name)(pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, self.n_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
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

        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.hparams.model_name}_{self.hparams.dataset_name}')

class FlexFormerClassifier(MedMNISTBase):
    def __init__(self, dataset_name: str, model_name: str = 'poolformer_s12', pretrained:bool=False,
                 num_heads:int=4, weight_decay: float = 0.05, patch_size: int = 224):
        super().__init__(dataset_name)
        assert self.is_2d, "PoolFormer is only implemented for 2D datasets"
        self.model = FlexFormer(self.n_classes, [patch_size, patch_size], num_heads, model_name, pretrained)

        self.save_hyperparameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        y_hat = self.model(x)
        return y_hat


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            model_name = self.hparams.model_name
            model_name = model_name.replace('pool', 'flex')
            Task.current_task().set_name(f'{model_name}_{self.hparams.dataset_name}')