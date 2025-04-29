import torch
from kornia import augmentation as K
from kornia import enhance
from kornia.augmentation import random_generator as rg
from kornia.augmentation.utils import _range_bound


class nnUNetDataAugmentation3D:
    def __init__(self, allow_flip: bool = True):
        self.rot_da = K.RandomRotation3D(degrees=30, p=0.2)
        self.affine_da = K.RandomAffine3D(degrees=0, translate=(0,) * 3, scale=(0.7, 1.4), p=0.2)
        self.flipping = K.AugmentationSequential(
            K.RandomHorizontalFlip3D(p=0.5),
            K.RandomVerticalFlip3D(p=0.5),
            K.RandomDepthicalFlip3D(p=0.5),
        ) if allow_flip else None
        self.random_gamma = self._random_gamma3d

        # random generator for the data augmentation
        brightness = _range_bound((0.75, 1.25), "brightness", center=1.0, bounds=(0.0, 2.0))
        contrast = _range_bound((0.75, 1.25), "contrast", center=1.0)
        self.param_generator = rg.PlainUniformGenerator((brightness, "brightness_factor", None, None),
                                                        (contrast, "contrast_factor", None, None))

    @staticmethod
    def _random_gaussian_noise(x: torch.Tensor, mean: float = 0.0, std: float = 1.0, p: float = 0.5) -> torch.Tensor:
        # only apply noise with probability p
        if torch.rand(1) < p:
            return x + torch.randn_like(x) * std + mean
        else:
            return x

    @staticmethod
    def _random_brightness(x: torch.Tensor, brightness: float = 0.0, p: float = 0.5) -> torch.Tensor:
        # only apply brightness with probability p
        if torch.rand(1) < p:
            return enhance.adjust_brightness(x, brightness, clip_output=True)
        else:
            return x

    @staticmethod
    def _random_contrast(x: torch.Tensor, contrast: float = 1.0, p: float = 0.5) -> torch.Tensor:
        # only apply contrast with probability p
        if torch.rand(1) < p:
            return enhance.adjust_contrast(x, contrast, clip_output=True)
        else:
            return x

    @staticmethod
    def _random_gamma3d(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, retain_stats: bool = False,
                      p: float = 0.5):
        def restore_spatial_dim(x: torch.Tensor):
            return x.view(*x.shape, 1, 1, 1)

        if torch.rand(1) > p:
            return data_sample

        if invert_image:
            data_sample = -data_sample

        if retain_stats:
            mn = data_sample.flatten(start_dim=2).mean(-1)
            sd = data_sample.flatten(start_dim=2).std(-1)
        if torch.rand(1) < 0.5 and gamma_range[0] < 1:
            gamma = torch.FloatTensor(len(data_sample)).uniform_(gamma_range[0], 1)
        else:
            gamma = torch.FloatTensor(len(data_sample)).uniform_(max(gamma_range[0], 1), gamma_range[1])
        gamma = gamma.to(data_sample.device, non_blocking=True)
        minm = restore_spatial_dim(data_sample.flatten(start_dim=2).min(-1).values)
        rnge = restore_spatial_dim(data_sample.flatten(start_dim=2).max(-1).values) - minm
        data_sample = torch.pow(((data_sample - minm) / (rnge + epsilon)), gamma.view(-1, 1, 1, 1, 1)) * rnge + minm
        if retain_stats:
            data_sample = data_sample - restore_spatial_dim(data_sample.flatten(start_dim=2).mean(-1))
            data_sample = data_sample / (restore_spatial_dim(data_sample.flatten(start_dim=2).std(-1) + 1e-8))
            data_sample *= restore_spatial_dim(sd)
            data_sample += restore_spatial_dim(mn)

        if invert_image:
            data_sample = -data_sample
        return data_sample

    @torch.no_grad()
    def __call__(self, x):
        data = x
        assert data.min() >= 0 and data.max() <= 1, "Data should be in range [0, 1], but found: " \
                                                    f"min: {data.min()}, max: {data.max()}"

        # get random parameters
        sampled_params = self.param_generator(data.shape)
        brightness_factor = sampled_params["brightness_factor"].to(data, non_blocking=True)
        contrast_factor = sampled_params["contrast_factor"].to(data, non_blocking=True)

        # spatial augmentation
        data = self.rot_da(data)
        data = self.affine_da(data)
        if self.flipping:
            data = self.flipping(data)

        # intensity augmentation
        data = self._random_gaussian_noise(data, std=0.1, p=0.1)
        data = self._random_brightness(data, brightness=brightness_factor - 1, p=0.15)
        data = self._random_contrast(data, contrast=contrast_factor, p=0.15)

        # gamma augmentation ported from nnUNet (on initial value ranges)
        data = self.random_gamma(data, gamma_range=(0.75, 1.5), invert_image=True, retain_stats=True, p=0.1)
        data = self.random_gamma(data, gamma_range=(0.75, 1.5), invert_image=False, retain_stats=True, p=0.3)

        return data


class nnUNetDataAugmentation2D(nnUNetDataAugmentation3D):
    def __init__(self, allow_flip: bool = True):
        super().__init__(allow_flip)
        self.is_3d = False
        # overwrite spatial augmentation with 2D augmentations
        self.rot_da = K.RandomRotation(degrees=30, p=0.2)
        self.affine_da = K.RandomAffine(degrees=0, translate=(0,) * 2, scale=(0.7, 1.4), p=0.2)
        self.flipping = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        ) if allow_flip else None
        self.random_gamma = self._random_gamma2d

    @staticmethod
    def _random_gamma2d(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, retain_stats: bool = False,
                      p: float = 0.5):
        def restore_spatial_dim(x: torch.Tensor):
            return x.view(*x.shape, 1, 1)

        if torch.rand(1) > p:
            return data_sample

        if invert_image:
            data_sample = -data_sample

        if retain_stats:
            mn = data_sample.flatten(start_dim=2).mean(-1)
            sd = data_sample.flatten(start_dim=2).std(-1)
        if torch.rand(1) < 0.5 and gamma_range[0] < 1:
            gamma = torch.FloatTensor(len(data_sample)).uniform_(gamma_range[0], 1)
        else:
            gamma = torch.FloatTensor(len(data_sample)).uniform_(max(gamma_range[0], 1), gamma_range[1])
        gamma = gamma.to(data_sample.device, non_blocking=True)
        minm = restore_spatial_dim(data_sample.flatten(start_dim=2).min(-1).values)
        rnge = restore_spatial_dim(data_sample.flatten(start_dim=2).max(-1).values) - minm
        data_sample = torch.pow(((data_sample - minm) / (rnge + epsilon)), gamma.view(-1, 1, 1, 1)) * rnge + minm
        if retain_stats:
            data_sample = data_sample - restore_spatial_dim(data_sample.flatten(start_dim=2).mean(-1))
            data_sample = data_sample / (restore_spatial_dim(data_sample.flatten(start_dim=2).std(-1) + 1e-8))
            data_sample *= restore_spatial_dim(sd)
            data_sample += restore_spatial_dim(mn)

        if invert_image:
            data_sample = -data_sample
        return data_sample

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from datasets.med_mnist_dataset import MedMNISTDataModule

    dm = MedMNISTDataModule('OrganAMNIST', batch_size=32, spatial_size=224)
    dm.setup('fit')
    print('Length of train dataset:', len(dm.train_dataset), 'Length of val dataset:', len(dm.val_dataset))
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(x.shape, y)
    aug = nnUNetDataAugmentation2D(False)
    x_ = aug(x)
    for img, img_ in zip(x, x_):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img.squeeze(), 'gray')
        axs[1].imshow(img_.squeeze(), 'gray')
    plt.show()
