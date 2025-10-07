from pathlib import Path
from typing import Any

import torch
from monai import data
from monai import transforms
from pytorch_lightning.core.datamodule import LightningDataModule

LABELS = ['invasive_tumor', 'tumor_associated_stroma', 'in_situ_tumor', 'healthy_glands', 'necrosis_not_in_situ',
          'inflamed_stroma', 'rest']
N_CLASSES = len(LABELS) + 1  # because of background
LBL_CNT = torch.tensor(
    [216838887, 310752344, 316315696, 35467227, 7607082, 48601048, 92258250, 92876335])  # include background


class TIGERDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 8, spatial_size: int = 768):
        """
        Dataset from the TIGER challenge by grand-challenge.org
        :param batch_size: batch size to use
        :param spatial_size: patch size extracted from whole slide images
        """
        super().__init__()
        assert batch_size % 4 == 0, 'Batch size must be divisible by 4, because we extract four equal space patches for validation.'
        self.batch_size = batch_size
        self.dl_kwargs = {'num_workers': 6, 'pin_memory': torch.cuda.is_available()}
        self.spatial_size = spatial_size
        mean = torch.tensor([0.7158108353614807, 0.5380678772926331, 0.676334798336029])
        std = torch.tensor([0.19980138540267944, 0.23654648661613464, 0.18582996726036072])

        self.base_transform = [
            transforms.LoadImaged(['image', 'label']),
            transforms.ToTensord(['image', 'label'], dtype=torch.float32),
            transforms.EnsureChannelFirstd(['image', 'label']),
            transforms.ScaleIntensityRangeD('image', a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=False),
            transforms.NormalizeIntensityd('image', mean, std, channel_wise=True),
            transforms.SpatialPadD(['image', 'label'], self.spatial_size, mode='reflect')
        ]

    def setup(self, stage):
        cache_rate = 0 if self.trainer.fast_dev_run else 1.0
        base = Path('data/tiger')
        train_data = (base / 'imagesTr').glob('*.png')
        train_data = [
            {'image': str(img_path), 'label': str(base / 'labelsTr' / img_path.name)}
            for img_path in sorted(list(train_data))
        ]
        test_data = (base / 'imagesTs').glob('*.png')
        test_data = [
            {'image': str(img_path), 'label': str(base / 'labelsTs' / img_path.name)}
            for img_path in sorted(list(test_data))
        ]

        if stage == 'fit':
            # calculate class weights
            lbl_ratio = (1 / LBL_CNT) ** 0.5
            lbl_ratio[0] = 0  # set probability for background to zero

            self.train_ds = data.CacheDataset(train_data, cache_rate=cache_rate, num_workers=None,
                                              transform=transforms.Compose([
                                                  *self.base_transform,
                                                  transforms.RandAxisFlipd(['image', 'label'], 0.5),
                                                  transforms.RandCropByLabelClassesD(['image', 'label'], 'label',
                                                                                     self.spatial_size,
                                                                                     lbl_ratio, N_CLASSES, 1),
                                                  # transforms.RandSpatialCropSamplesD(['image', 'label'],
                                                  #                                    roi_size=self.spatial_size,
                                                  #                                    num_samples=1,
                                                  #                                    random_size=False,
                                                  #                                    random_center=True),
                                              ]))

            self.val_ds = data.CacheDataset(test_data, cache_rate=cache_rate, num_workers=None,
                                            transform=transforms.Compose([
                                                *self.base_transform,
                                                transforms.GridSplitD(['image', 'label'], (2, 2), self.spatial_size),
                                            ]))
        elif stage == 'test':
            # skip padding
            self.test_ds = data.Dataset(test_data, transform=transforms.Compose(self.base_transform[:-1]))
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, **self.dl_kwargs,
                               drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size // 4, **self.dl_kwargs, drop_last=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, batch_size=1, **self.dl_kwargs)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch['image'].as_tensor(), batch['label'].as_tensor()  # match structure of the other datasets
