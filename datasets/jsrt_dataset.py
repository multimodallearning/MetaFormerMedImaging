from collections import OrderedDict

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F


class JSRTDataset(Dataset):
    SPLIT_IDX = 160
    PIXEL_RESOLUTION_MM = 1.4
    NUM_LANDMARKS = {'right_lung': 44, 'left_lung': 50, 'heart': 26, 'right_clavicle': 23, 'left_clavicle': 23}
    N_CLASSES = len(NUM_LANDMARKS)
    LABELS = list(NUM_LANDMARKS.keys())

    def __init__(self, mode: str, return_dist_map: bool = False):
        super().__init__()
        self.return_dist_map = return_dist_map

        # load data
        data = torch.load('data/JSRT_img0_lms.pth', map_location='cpu')
        self.imgs = data['JSRT_img0'].float()  # already z-normalized
        self.lms = data['JSRT_lms'].float()
        self.dist_map = torch.load('data/jsrt_distmaps.pth').float()
        del data
        self.seg_masks = torch.load('data/jsrt_seg_masks.pth').float()

        # check for equal number of samples
        assert self.imgs.shape[0] == self.lms.shape[0] == self.dist_map.shape[0]

        # normalize landmarks to [-1, 1]
        assert self.imgs.shape[-1] == self.imgs.shape[-2] == 256, f'Expected image size 256, but got {self.imgs.shape}'
        assert self.lms.max() <= 256 and self.lms.min() >= 0
        self.lms = self.lms / 256 * 2 - 1

        # select images for training or testing
        if mode == 'train':
            self.imgs = self.imgs[:self.SPLIT_IDX]
            self.lms = self.lms[:self.SPLIT_IDX]
            self.dist_map = self.dist_map[:self.SPLIT_IDX]
            self.seg_masks = self.seg_masks[:self.SPLIT_IDX]
        elif mode == 'val':
            self.imgs = self.imgs[self.SPLIT_IDX:]
            self.lms = self.lms[self.SPLIT_IDX:]
            self.dist_map = self.dist_map[self.SPLIT_IDX:]
            self.seg_masks = self.seg_masks[self.SPLIT_IDX:]
        else:
            raise ValueError(f'Unknown mode {mode}')

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        img = self.imgs[idx].unsqueeze(0)  # add channel dimension
        seg_mask = self.seg_masks[idx]

        if self.return_dist_map:
            dist_map = self.dist_map[idx]
            return img, seg_mask, dist_map
        else:
            return img, seg_mask

    @classmethod
    def get_training_shapes(cls) -> torch.Tensor:
        return cls('train').lms

    @classmethod
    def get_anatomical_structure_index(cls) -> OrderedDict:
        name_idx_dict = OrderedDict()
        idx = 0
        for organ, num_lms in cls.NUM_LANDMARKS.items():
            name_idx_dict[organ] = (idx, idx + num_lms)
            idx += num_lms

        return name_idx_dict


class JSRTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 8, use_data_aug: bool = True, data_aug_std: float = 0.06):
        super().__init__()
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

        self.use_data_aug = use_data_aug
        self.data_aug_std = data_aug_std

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.train_dataset = JSRTDataset('train')
            self.val_dataset = JSRTDataset('val')
        elif stage == 'test':
            self.test_dataset = JSRTDataset('val')
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

        return x, y

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from random import randint

    ds = JSRTDataset('val')
    x, y = ds[0]
    print(f'Number of classes: {ds.N_CLASSES}')
    idx = randint(0, len(ds) - 1)
    x, y = ds[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(y.argmax(0), alpha=y.any(0).float() * 0.5)
    for lbl, mask in zip(ds.LABELS, y):
        plt.figure(lbl)
        plt.imshow(x.squeeze(0), cmap='gray')
        plt.imshow(mask, alpha=(mask == 1).float())
        plt.title(lbl)
    plt.show()
