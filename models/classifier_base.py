import warnings
from abc import abstractmethod
from argparse import Namespace
from typing import Optional, Any

import medmnist
import torch
from clearml import Logger, Task
from medmnist import INFO
from medmnist.dataset import MedMNIST2D, MedMNIST3D
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.scheduler import CosineLRScheduler
from torch import nn
from torchmetrics import classification, MetricCollection, MeanMetric

from datasets.imagewoof_dataset import ImageWoofDataset
from datasets.med_mnist_statistics import LOSS_WEIGHTS

warnings.filterwarnings("ignore", category=UserWarning)


class ClassifierBase(LightningModule):
    def __init__(self, ds_name: str, lr: float = 0.001, wd: float = 0.01, ce_label_smoothing: float = 0.1,
                 warmup_epochs: int = 5, min_lr: float = 1e-5, loss_weight_max: float = 10, use_scheduler:bool = True):
        """
        Base class for classification models on MedMNIST and ImageWoof datasets.
        :param ds_name: name of the dataset. Has to be one of the MedMNIST datasets or 'imagewoof'
        :param lr: learn rate used for training
        :param wd: weight decay deployed in AdamW optimizer
        :param ce_label_smoothing: label smoothing in cross entropy loss
        :param warmup_epochs: epochs for linear learning rate ramp-up
        :param min_lr: minimum learning rate
        :param loss_weight_max: clamp maximum for loss class weights
        :param use_scheduler: whether to use cosine lr scheduler or no scheduler
        """
        super().__init__()
        # attributes
        self.is_2d = False
        self.is_3d = False
        if ds_name.lower().endswith("mnist"):
            self.n_channels = INFO[ds_name.lower()]['n_channels']
            self.label = list(INFO[ds_name.lower()]['label'].values())
            self.n_classes = len(self.label)
            ds_class = getattr(medmnist, INFO[ds_name.lower()]['python_class'])
            self.is_2d = issubclass(ds_class, MedMNIST2D)
            self.is_3d = issubclass(ds_class, MedMNIST3D)
            task = INFO[ds_name.lower()]['task']
            loss_weights = torch.tensor(LOSS_WEIGHTS[ds_name.lower()])
        elif ds_name.lower() == "imagewoof":
            self.n_channels = 3
            self.label = ImageWoofDataset.LABEL
            self.n_classes = len(self.label)
            self.is_2d = True
            task = "multi-class"
            loss_weights = torch.tensor(ImageWoofDataset.LOSS_WEIGHTS)
        else:
            raise NotImplementedError(f'Dataset {ds_name} is not implemented.')

        assert self.is_2d != self.is_3d, "Either 2D or 3D dataset must be selected"
        # criterion
        loss_weights = loss_weights.clamp(max=loss_weight_max)
        if task.split(',')[0] == 'multi-label':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
            self.cls_mtl_exclude = False
        elif task in ['multi-class', 'binary-class']:
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=ce_label_smoothing)
            self.cls_mtl_exclude = True
        else:
            raise NotImplementedError(f"Task {task} is not implemented.")
        print('Classes are mutual exclusive:', self.cls_mtl_exclude)

        # metrics
        metrics_kwargs = {"num_classes": self.n_classes, "num_labels": self.n_classes, "average": None,
                          "task": 'multiclass' if self.cls_mtl_exclude else 'multilabel'}
        self.train_metrics = MetricCollection({
            "acc": classification.Accuracy(**metrics_kwargs),
            "f1": classification.F1Score(**metrics_kwargs),
            # "prec": classification.Precision(**metrics_kwargs),
            # "rec": classification.Recall(**metrics_kwargs),
            # "auroc": classification.AUROC(**metrics_kwargs),
        }, postfix='/train')
        self.val_metrics = self.train_metrics.clone(postfix='/val')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.optim_hp = Namespace(lr=lr, wd=wd, warmup_epochs=warmup_epochs, min_lr=min_lr,
                                  loss_weight_clamp_max=loss_weight_max, use_scheduler=use_scheduler)
        if Task.current_task() is not None:
            Task.current_task().connect(vars(self.optim_hp), name='optimizer_hyperparameters')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_hp.lr, weight_decay=self.optim_hp.wd)
        if self.optim_hp.use_scheduler:
            scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs,
                                          warmup_t=self.optim_hp.warmup_epochs,
                                          warmup_lr_init=self.optim_hp.min_lr, lr_min=self.optim_hp.min_lr,
                                          warmup_prefix=True)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        scheduler.step(self.current_epoch)
        if Logger.current_logger() is not None:
            Logger.current_logger().report_scalar('learning rate', 'lr',
                                                  scheduler._get_lr(self.current_epoch)[0], self.current_epoch)

    @staticmethod
    @torch.no_grad()
    def reset_pretrained_weights(model: nn.Module, scale: float) -> None:
        """
        Reset pretrained weights by adding noise with standard deviation of clamped weights * scale.
        :param model: module to reset pretrained weights
        :param scale: percentage of noise to add
        :return:
        """
        if scale is None or scale == 0:
            return

        assert 0 <= scale <= 1, "Reset weights percentage must be between 0 and 1"
        print('Adding noise to pretrained weights...')
        params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])
        sigma = torch.clamp(params, params.quantile(0.005), params.quantile(0.995)).std()
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data + torch.randn_like(p) * sigma * scale

    @abstractmethod
    def forward(self, batch):
        pass

    def common_step(self, batch, mode):
        x, y = batch
        y_hat = self.forward(x)
        if self.cls_mtl_exclude:
            y = y.squeeze(-1)  # handle case of (B, 1) of binary classification (rewritten as multi-class)
        else:  # multi-label, where y holds probabilities
            y = y.float()
        loss = self.criterion(y_hat, y)

        with torch.no_grad():
            # logging
            getattr(self, f"{mode}_loss")(loss)
            metrics = getattr(self, f"{mode}_metrics")
            if self.cls_mtl_exclude:
                metrics(y_hat.argmax(1), y.long())
            else:
                metrics(y_hat.sigmoid(), y.long())

        return loss

    def eval_and_log(self, mode: str):
        if Logger.current_logger() is None:
            return
        loss_logger = getattr(self, f"{mode}_loss")
        metric_collection = getattr(self, f"{mode}_metrics")

        Logger.current_logger().report_scalar('loss', mode, loss_logger.compute().cpu(), self.current_epoch)

        epoch_values = getattr(self, f"{mode}_metrics").compute()
        for name, value in epoch_values.items():
            # metric logging for pytorch lightning to enable selection of best model
            self.log(name, value.mean())

            # logging in clearml
            name = name.split('/')[0]
            Logger.current_logger().report_histogram(name, mode, value.cpu().numpy(), self.current_epoch,
                                                     xaxis='class', yaxis=name, xlabels=self.label)
            Logger.current_logger().report_scalar(name, mode, value.mean().cpu(), self.current_epoch)

        loss_logger.reset()
        metric_collection.reset()

    def training_step(self, batch):
        return self.common_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        self.eval_and_log('train')

    def validation_step(self, batch):
        return self.common_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self.eval_and_log('val')
