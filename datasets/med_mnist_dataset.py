from pytorch_lightning import LightningDataModule
import medmnist
import torch
from kornia import augmentation
from torchvision.transforms import ToTensor


class MedMNISTDataModule(LightningDataModule):
    def __init__(self, dataset_name: str, batch_size: int = 128, spatial_size: int = 224,
                 affine_params_rot_trans_scale: tuple = (30, 0.1, 0.15), use_data_aug: bool = True):
        """
        :param dataset_name: name of the dataset. Has to be one of the MedMNIST datasets
        :param batch_size: batch size
        :param spatial_size: spatial size of the images
        :param affine_params_rot_trans_scale: tuple with parameters for RandomAffine. If None, no data augmentation is used
        :param use_data_aug: if True, data augmentation is used
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available(), 'drop_last': True}
        self.DataClass = getattr(medmnist, medmnist.INFO[dataset_name.lower()]['python_class'])

        print("TODO: Implement normalization and loss weighting!!!")
        self.use_data_aug = use_data_aug
        rotate, translate, scale = affine_params_rot_trans_scale
        if issubclass(self.DataClass, medmnist.dataset.MedMNIST2D):
            self.data_aug = augmentation.RandomAffine(degrees=rotate, translate=(translate,) * 2,
                                                      scale=(1 - scale, 1 + scale), p=1)
        elif issubclass(self.DataClass, medmnist.dataset.MedMNIST3D):
            self.data_aug = augmentation.RandomAffine3D(degrees=rotate, translate=(translate,) * 3,
                                                        scale=(1 - scale, 1 + scale), p=1)
        else:
            raise ValueError(f"Unknown MedMNIST dataset class: {self.DataClass}")

        self.normalize = augmentation.Normalize(mean=0.5, std=0.5)

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
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.use_data_aug and self.trainer.training:
            x = self.data_aug(x)
        x = self.normalize(x)
        return x, y


if __name__ == '__main__':
    dm = MedMNISTDataModule('OrganAMNIST', batch_size=1, spatial_size=224)
    dm.setup('fit')
    print('Length of train dataset:', len(dm.train_dataset), 'Length of val dataset:', len(dm.val_dataset))
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(x.shape, y)
