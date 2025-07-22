import warnings
from abc import abstractmethod
from argparse import Namespace
from typing import Optional, Any

import torch
from clearml import Logger, Task
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.scheduler import CosineLRScheduler
from torch import nn
from torchmetrics import classification, MetricCollection, MeanMetric

from datasets.grazpedwri_dataset import SegGrazPedWriDataset

warnings.filterwarnings("ignore", category=UserWarning)


class SegmentatorBase(LightningModule):
    def __init__(self, ds_name: str, lr: float = 0.001, wd: float = 0.01,
                 warmup_epochs: int = 5, min_lr: float = 1e-5, loss_weight_max: float = 10):
        super().__init__()
        # attributes
        if ds_name.lower() == 'wristbone':
            self.n_channels = 1
            self.n_classes = SegGrazPedWriDataset.N_CLASSES
            self.label = SegGrazPedWriDataset.BONE_LABEL
            task = "multi-label"
            loss_weights = SegGrazPedWriDataset.POS_CLASS_WEIGHT
        else:
            raise NotImplementedError(f'Dataset {ds_name} is not implemented.')

        # criterion
        loss_weights = loss_weights.view(-1, 1, 1).clamp(max=loss_weight_max)
        if task.split(',')[0] == 'multi-label':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
            self.cls_mtl_exclude = False
        elif task in ['multi-class', 'binary-class']:
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights)
            self.cls_mtl_exclude = True
        else:
            raise NotImplementedError(f"Task {task} is not implemented.")
        print('Classes are mutual exclusive:', self.cls_mtl_exclude)

        # metrics
        metrics_kwargs = {"num_classes": self.n_classes, "num_labels": self.n_classes, "average": None,
                          "task": 'multiclass' if self.cls_mtl_exclude else 'multilabel'}
        self.train_metrics = MetricCollection({
            "dsc": classification.F1Score(**metrics_kwargs),
        }, postfix='/train')
        self.val_metrics = self.train_metrics.clone(postfix='/val')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.optim_hp = Namespace(lr=lr, wd=wd, warmup_epochs=warmup_epochs, min_lr=min_lr,
                                  loss_weight_clamp_max=loss_weight_max)
        if Task.current_task() is not None:
            Task.current_task().connect(vars(self.optim_hp), name='optimizer_hyperparameters')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_hp.lr, weight_decay=self.optim_hp.wd)
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs,
                                      warmup_t=self.optim_hp.warmup_epochs,
                                      warmup_lr_init=self.optim_hp.min_lr, lr_min=self.optim_hp.min_lr,
                                      warmup_prefix=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        scheduler.step(self.current_epoch)
        if Logger.current_logger() is not None:
            Logger.current_logger().report_scalar('learning rate', 'lr',
                                                  scheduler._get_lr(self.current_epoch)[0], self.current_epoch)


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
