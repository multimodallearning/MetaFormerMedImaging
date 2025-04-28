import warnings
from abc import abstractmethod

import medmnist
import torch
from clearml import Logger
from medmnist import INFO
from medmnist.dataset import MedMNIST2D, MedMNIST3D
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import classification, MetricCollection, MeanMetric

from datasets.med_mnist_statistics import LOSS_WEIGHTS

warnings.filterwarnings("ignore", category=UserWarning)


class MedMNISTBase(LightningModule):
    def __init__(self, ds_name: str, learn_rate: float = 0.001):
        super().__init__()
        # attributes
        self.lr = learn_rate
        self.ds_name = ds_name
        self.n_channels = INFO[ds_name.lower()]['n_channels']
        self.label = list(INFO[ds_name.lower()]['label'].values())
        self.n_classes = len(self.label)
        ds_class = getattr(medmnist, INFO[ds_name.lower()]['python_class'])
        self.is_2d = issubclass(ds_class, MedMNIST2D)
        self.is_3d = issubclass(ds_class, MedMNIST3D)
        assert self.is_2d != self.is_3d, "Either 2D or 3D dataset must be selected"

        # criterion
        task = INFO[ds_name.lower()]['task']
        loss_weights = torch.tensor(LOSS_WEIGHTS[ds_name.lower()])
        if task in ["multi-label", "binary-class"]:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
            self.cls_mtl_exclude = False
        elif task == "multi-class":
            print('TODO: label smoothing')
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights)
            self.cls_mtl_exclude = True
        else:
            raise NotImplementedError(f"Task {task} is not implemented.")
        print('Classes are mutual exclusive:', self.cls_mtl_exclude)

        # metrics
        metrics_kwargs = {"num_classes": self.n_classes, "num_labels": self.n_classes, "average": None,
                          "task": 'multiclass' if task == "multi-class" else 'multilabel'}
        self.train_metrics = MetricCollection({
            "acc": classification.Accuracy(**metrics_kwargs),
            "f1": classification.F1Score(**metrics_kwargs),
            "prec": classification.Precision(**metrics_kwargs),
            "rec": classification.Recall(**metrics_kwargs),
            "auroc": classification.AUROC(**metrics_kwargs),
        }, postfix='/train')
        self.val_metrics = self.train_metrics.clone(postfix='/val')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @abstractmethod
    def forward(self, batch):
        pass

    def common_step(self, batch, mode):
        x, y = batch
        y_hat = self.forward(x)
        y = y.squeeze(-1)
        if not self.cls_mtl_exclude: # convert indices to probabilities
            raise NotImplementedError('Check for correctness, before using it')
            p = torch.zeros(len(y), self.n_classes, device=y.device)
            p.scatter_(1, y.unsqueeze(1), 1)
            y = p
        loss = self.criterion(y_hat, y)

        with torch.no_grad():
            # logging
            getattr(self, f"{mode}_loss")(loss)
            metrics = getattr(self, f"{mode}_metrics")
            if self.cls_mtl_exclude:
                batch_values = metrics(y_hat.softmax(1), y.long())
            else:
                batch_values = metrics(y_hat.sigmoid(), y.long())

            # metric logging for pytorch lightning to enable selection of best model
            logg_kwargs = {"on_step": False, "on_epoch": True, "batch_size": len(y), 'logger': Logger.current_logger()}
            for name, value in batch_values.items():
                self.log(name, value.mean(), **logg_kwargs)

        return loss

    def report_histogram(self, mode: str):
        if Logger.current_logger() is None:
            return
        loss_logger = getattr(self, f"{mode}_loss")
        metric_collection = getattr(self, f"{mode}_metrics")

        #raise NotImplementedError("WHAT IS THIS???")
        Logger.current_logger().report_scalar('loss', mode, loss_logger.compute().cpu(), self.current_epoch)

        epoch_values = getattr(self, f"{mode}_metrics").compute()
        for name, value in epoch_values.items():
            name = name.split('/')[0]
            Logger.current_logger().report_histogram(name, mode, value.cpu().numpy(), self.current_epoch,
                                                     xaxis='class', yaxis=name, xlabels=self.label)
            Logger.current_logger().report_scalar(name, mode, value.mean().cpu(), self.current_epoch)

        loss_logger.reset()
        metric_collection.reset()

    def training_step(self, batch):
        return self.common_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        self.report_histogram('train')

    def validation_step(self, batch):
        return self.common_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self.report_histogram('val')
