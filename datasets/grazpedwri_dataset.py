from pathlib import Path
from random import randint

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.cvat_parser import CVATParser


class SegGrazPedWriDataset(Dataset):
    # calculated over training split
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    # bone label
    BONE_LABEL = sorted([
        'Radius',
        'Ulna',
        'Os scaphoideum',
        'Os lunatum',
        'Os triquetrum',
        'Os pisiforme',
        'Os trapezium',
        'Os trapezoideum',
        'Os capitatum',
        'Os hamatum',
        'Ossa metacarpalia I',
        'Ossa metacarpalia II',
        'Ossa metacarpalia III',
        'Ossa metacarpalia IV',
        'Ossa metacarpalia V',
        'Epiphyse Radius',
        'Epiphyse Ulna'])
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)
    # BCE weights not including background
    POS_CLASS_WEIGHT = torch.tensor([108.1348, 349.1551, 69.6342, 96.0886, 167.7897, 364.5914, 131.5362,
                                     176.2591, 240.9182, 169.5408, 60.1363, 46.6512, 51.6916, 58.6216,
                                     52.5956, 11.2623, 17.9409])

    def __init__(self, mode: str, rescale_HW: tuple = (384, 224)):
        """
        :param mode: data split mode [training, validation, testing]
        :param rescale_HW: rescale image and ground truth (should be close to ratio of 1.75 as possible). None will not rescale.
        """
        super().__init__()
        # init ground truth parser considering the data split
        if mode == 'train':
            xml_files = list(Path('data/cvat_annotation_xml').glob(f'annotations_train[1-9].xml'))
        elif mode == 'val':
            xml_files = [Path(f'data/cvat_annotation_xml/annotations_{mode}.xml') for mode in ['val', 'test']]
        else:
            raise ValueError(f'Unknown mode {mode}')
        self.gt_parser = CVATParser(xml_files, True, False, True)

        # load img into memory
        img_path = Path('data/img_only_front_all_left')
        self.available_file_names = set([f.stem for f in img_path.glob('*.png')])
        self.available_file_names &= set(self.gt_parser.available_file_names)
        self.available_file_names = sorted(list(self.available_file_names)) # __getitem__ likes to use indices
        self.data = dict()
        for file_name in tqdm(self.available_file_names, unit='img', desc=f'Loading data for {mode}'):
            data_dict = dict()
            data_dict['image'] = Image.open(str(img_path.joinpath(file_name).with_suffix('.png'))).convert('L')
            data_dict['image'] = data_dict['image'].resize(rescale_HW[::-1], Image.Resampling.BILINEAR)

            seg_masks = self.gt_parser.extract_masks(file_name)
            seg_masks = CVATParser.cvt_mask_list_2_dict(seg_masks)
            data_dict.update(seg_masks)

            need2flip = file_name.split('_')[3].split('-')[1][0] == 'R'  # flip to left hand

            # stack masks
            for lbl in self.BONE_LABEL:
                try:
                    data_dict[lbl] = F.interpolate(torch.from_numpy(data_dict[lbl])[None, None, :, :],
                                                   size=rescale_HW, mode='nearest-exact').squeeze()
                except KeyError:
                    data_dict[lbl] = torch.zeros(rescale_HW)  # add empty mask if not annotated
            y = torch.stack([data_dict[lbl] for lbl in self.BONE_LABEL], dim=0)
            if need2flip:  # image is already stored as flipped image
                y = torch.flip(y, dims=[-1])

            # numpy image to tensor and add channel dimension
            img = torch.from_numpy(np.array(data_dict['image'])).unsqueeze(0).div(255)

            self.data[file_name] = {'image': img, 'mask': y.bool()}

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, str):
        """
        get item by index
        :param index: index of item
        :return: image, ground truth, file name
        """
        file_name = self.available_file_names[index]
        data_dict = self.data[file_name]
        x, y = data_dict['image'], data_dict['mask']

        return x, y.float()


class SegGrazPedWriDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 8, use_data_aug: bool = True, data_aug_std: float = 0.06):
        super().__init__()
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

        self.use_data_aug = use_data_aug
        self.data_aug_std = data_aug_std

        self.mean = torch.tensor(SegGrazPedWriDataset.IMG_MEAN).view(1, 1, 1, 1)
        self.std = torch.tensor(SegGrazPedWriDataset.IMG_STD).view(1, 1, 1, 1)

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.train_dataset = SegGrazPedWriDataset('train')
            self.val_dataset = SegGrazPedWriDataset('val')
        elif stage == 'test':
            self.test_dataset = SegGrazPedWriDataset('val')
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
            id = torch.eye(2, 3, device=x.device).unsqueeze(0)
            rnd_offset = torch.randn(x.shape[0], 2, 3, device=x.device).mul(self.data_aug_std)
            grid = F.affine_grid(id + rnd_offset, list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            y = F.grid_sample(y, grid, align_corners=False, mode='nearest')

        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        x = (x - self.mean) / self.std

        return x, y


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = SegGrazPedWriDataset('val')
    print(f'Number of classes: {ds.N_CLASSES}')
    idx = randint(0, len(ds) - 1)
    x, y = ds[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(y.argmax(0), alpha=y.any(0).float() * 0.5)
    for lbl, mask in zip(ds.BONE_LABEL, y):
        plt.figure(lbl)
        plt.imshow(x.squeeze(0), cmap='gray')
        plt.imshow(mask, alpha=(mask == 1).float())
        plt.title(lbl)
    plt.show()
