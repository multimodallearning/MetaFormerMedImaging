from pathlib import Path

import torch
from monai import transforms, data
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F


class AbdomenAtlasDataModule(LightningDataModule):
    CLASS_MAP = {
        1: 'aorta',
        2: 'gall_bladder',
        3: 'kidney_left',
        4: 'kidney_right',
        5: 'liver',
        6: 'pancreas',
        7: 'postcava',
        8: 'spleen',
        9: 'stomach',
    }

    def __init__(self, batch_size: int = 8, patch_size: int = 128, use_data_aug: bool = True,
                 data_aug_std: float = 0.06):
        super().__init__()
        self.patch_size = (patch_size,) * 3
        self.use_data_aug = use_data_aug
        self.sigma = data_aug_std
        self.dl_kwargs = {'num_workers': 6, 'pin_memory': torch.cuda.is_available(), 'batch_size': batch_size}

        self.base_transforms = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-200, a_max=300,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label"
            ),
            transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=self.patch_size
            )
        ]

    def setup(self, stage: str) -> None:
        base = Path('/data_rechenmagd01_1/keuth/AbdomenAtlas1.0Mini')
        cache = Path('/data_rechenmagd01_1/keuth/monai_cache/abdomen_atlas')
        train_split = list()
        test_split = list()
        test_ids = open('datasets/AbdomenAtlasTestSplit.csv').read().splitlines()
        for p in filter(lambda x: x.is_dir(), sorted(base.iterdir())):
            sample = {
                'image': str(p / "ct.nii.gz"),
                'label': str(p / "combined_labels.nii.gz"),
                'id': p.stem
            }
            if p.stem in test_ids:
                test_split.append(sample)
            else:
                train_split.append(sample)
        assert len(test_split) == len(test_ids) and len(list(base.iterdir())) == (len(test_split) + len(train_split))
        det_trans = transforms.Compose([*self.base_transforms])
        match stage:
            case 'fit':
                self.train_det_ds = data.PersistentDataset(train_split, det_trans, cache_dir=cache / 'train')
                self.train_ds = data.Dataset(self.train_det_ds,
                                             transform=transforms.RandSpatialCropD(keys=["image", "label"],
                                                                                   roi_size=self.patch_size))
                self.val_ds = data.PersistentDataset(test_split, transforms.Compose([
                    *self.base_transforms,
                    transforms.CenterSpatialCropD(keys=["image", "label"], roi_size=self.patch_size)
                ]), cache_dir=cache / 'test')
            case 'test':
                self.test_ds = data.Dataset(test_split, det_trans)

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, shuffle=True, drop_last=True, **self.dl_kwargs)

    def test_dataloader(self):
        dl_kwargs = {**self.dl_kwargs, 'batch_size': 1}  # override batch size
        return data.DataLoader(self.test_ds, drop_last=False, **dl_kwargs)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch['image'], batch['label']
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        if hasattr(y, "as_tensor"):
            y = y.as_tensor()
        return x, y

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        trainer = getattr(self, "trainer", None)
        if self.use_data_aug and trainer and trainer.training:
            # random affine transformation
            id = torch.eye(3, 4, device=x.device).unsqueeze(0)
            rnd_offset = torch.randn(x.shape[0], 3, 4, device=x.device).mul(self.sigma)
            # image
            grid = F.affine_grid(id + rnd_offset, list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')
            # mask
            grid = F.affine_grid(id + rnd_offset, list(y.shape), align_corners=False)
            y = F.grid_sample(y, grid, align_corners=False, mode='nearest')

        return x, y


if __name__ == '__main__':
    from tqdm import trange

    ds = AbdomenAtlasDataModule(use_data_aug=False)
    ds.setup('fit')
    print("Building cache...")
    for i in trange(len(ds.val_ds)):
        _ = ds.val_ds[i]
    for i in trange(len(ds.train_det_ds)):
        _ = ds.train_det_ds[i]
    print("Cache ready!")
