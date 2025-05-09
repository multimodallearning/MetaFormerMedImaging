import medmnist
import torch
from kornia.augmentation.auto import RandAugment
from pytorch_lightning import LightningDataModule
from torchvision.transforms import ToTensor

from datasets import nnunet_data_aug
from datasets.med_mnist_statistics import IMG_MEAN_STD, ALLOW_FLIPPING


class MedMNISTDataModule(LightningDataModule):
    def __init__(self, dataset_name: str, batch_size: int = 128, spatial_size: int = 224, use_data_aug: bool = True):
        """
        :param dataset_name: name of the dataset. Has to be one of the MedMNIST datasets
        :param batch_size: batch size
        :param spatial_size: spatial size of the images
        :param affine_params_rot_trans_scale: tuple with parameters for RandomAffine. If None, no data augmentation is used
        :param use_data_aug: if True, data augmentation is used
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.DataClass = getattr(medmnist, medmnist.INFO[dataset_name.lower()]['python_class'])

        self.use_data_aug = use_data_aug
        allow_flipping = ALLOW_FLIPPING[dataset_name.lower()]
        if issubclass(self.DataClass, medmnist.dataset.MedMNIST2D):
            self.data_aug = nnunet_data_aug.nnUNetDataAugmentation2D(allow_flipping)
        elif issubclass(self.DataClass, medmnist.dataset.MedMNIST3D):
            self.data_aug = nnunet_data_aug.nnUNetDataAugmentation3D(allow_flipping)
        else:
            raise ValueError(f"Unknown MedMNIST dataset class: {self.DataClass}")

        img_stats = IMG_MEAN_STD[dataset_name.lower()]
        self.mean = torch.tensor(img_stats.mean).view(1, -1, 1, 1)
        self.std = torch.tensor(img_stats.std).view(1, -1, 1, 1)

    def setup(self, stage: str = None):
        ds_kwargs = {'root': './data', 'download': True, 'size': self.spatial_size, 'transform': ToTensor()}
        if stage == 'fit':
            self.train_dataset = self.DataClass('train', **ds_kwargs)
            self.val_dataset = self.DataClass('val', **ds_kwargs)
        elif stage == 'test':
            self.test_dataset = self.DataClass('test', **ds_kwargs)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.dl_kwargs, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, **self.dl_kwargs, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.use_data_aug and self.trainer.training:
            x = self.data_aug(x)
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        x = (x - self.mean) / self.std
        return x, y


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dm = MedMNISTDataModule('OrganAMNIST', batch_size=1, spatial_size=224)
    dm.setup('fit')
    print('Length of train dataset:', len(dm.train_dataset), 'Length of val dataset:', len(dm.val_dataset))
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(x.shape, y)
    plt.imshow(x.squeeze(), 'gray')
    aug = RandAugment(n=2, m=9)
    x = aug(x)
    plt.figure()
    plt.imshow(x.squeeze(), 'gray')
    plt.show()
