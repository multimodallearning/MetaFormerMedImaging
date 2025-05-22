import medmnist
import torch
from kornia.augmentation.auto import RandAugment
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torchvision.transforms import ToTensor

from datasets import nnunet_data_aug
from datasets.med_mnist_statistics import IMG_MEAN_STD, ALLOW_FLIPPING


class RndAffineAug(torch.nn.Module):
    def __init__(self, std: float = 0.06, dims: int = 2):
        super().__init__()
        self.std = std
        if dims == 2:
            self.forward = self.transform2d
        elif dims == 3:
            self.forward = self.transform3d
        else:
            raise NotImplementedError(f'Not implemented for {dims} dimensions')

    def transform2d(self, x):
        id = torch.eye(2, 3, device=x.device).unsqueeze(0)
        rnd_offset = torch.randn(x.shape[0], 2, 3, device=x.device).mul(self.std)
        grid = F.affine_grid(id + rnd_offset, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')

        return x

    def transform3d(self, x):
        id = torch.eye(3, 4, device=x.device).unsqueeze(0)
        rnd_offset = torch.randn(x.shape[0], 3, 4, device=x.device).mul(self.std)
        grid = F.affine_grid(id + rnd_offset, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, mode='trilinear')

        return x


class MedMNISTDataModule(LightningDataModule):
    def __init__(self, dataset_name: str, batch_size: int = 128, spatial_size: int = 224, data_aug: str = 'affine',
                 data_aug_std: float = 0.1):
        """
        :param dataset_name: name of the dataset. Has to be one of the MedMNIST datasets
        :param batch_size: batch size
        :param spatial_size: spatial size of the images
        :param data_aug: mode of data augmentation. [affine, nnUnet, None]
        :param data_aug_std: standard deviation of the random affine transformation
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.DataClass = getattr(medmnist, medmnist.INFO[dataset_name.lower()]['python_class'])

        assert data_aug in [None, 'affine', 'nnUnet'], 'Unknown data augmentation mode.'
        self.use_data_aug = data_aug is not None
        allow_flipping = ALLOW_FLIPPING[dataset_name.lower()]
        if issubclass(self.DataClass, medmnist.dataset.MedMNIST2D):
            if data_aug == 'nnUnet':
                self.data_aug = nnunet_data_aug.nnUNetDataAugmentation2D(allow_flipping)
            elif data_aug == 'affine':
                self.data_aug = RndAffineAug(std=data_aug_std, dims=2)
        elif issubclass(self.DataClass, medmnist.dataset.MedMNIST3D):
            if data_aug == 'nnUnet':
                self.data_aug = nnunet_data_aug.nnUNetDataAugmentation3D(allow_flipping)
            elif data_aug == 'affine':
                self.data_aug = RndAffineAug(std=data_aug_std, dims=3)
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

    dm = MedMNISTDataModule('PneumoniaMNIST', batch_size=1, spatial_size=224)
    dm.setup('fit')
    print('Length of train dataset:', len(dm.train_dataset), 'Length of val dataset:', len(dm.val_dataset))
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(x.shape, y.shape)
    plt.imshow(x.squeeze(), 'gray')
    x = dm.data_aug(x)
    plt.figure()
    plt.imshow(x.squeeze(), 'gray')
    plt.show()

    # x2d = torch.randn(2, 1, 224, 224)
    # y2d = RndAffineAug()(x2d)
    #
    # x3d = torch.randn(2, 1, 224, 224, 224)
    # y3d = RndAffineAug(dims=3)(x3d)
