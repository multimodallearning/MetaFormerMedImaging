from pathlib import Path

import torch
import torchvision.transforms.functional as F
from kornia import augmentation
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from torch.nn import functional as F


# taken from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
class SquarePad:
    def __call__(self, image):
        h, w = image.shape[-2:]
        max_wh = max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp)
        return F.pad(image, padding, value=0, mode='constant')


# get dataset from https://github.com/fastai/imagenette#imagewoof
class ImageWoofDataset(Dataset):
    IMGNET_LABEL = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601',
                    'n02105641', 'n02111889', 'n02115641']
    LABEL = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 'Border terrier', 'Australian terrier',
             'Golden retriever', 'Old English sheepdog', 'Samoyed', 'Dingo']
    LOSS_WEIGHTS = [1.0042, 1.0037, 1.0091, 1.2791, 1.0000, 1.0032, 1.0000, 1.0113, 1.0151, 1.0048]

    def __init__(self, mode: str, img_size: int = 224):
        super().__init__()
        assert img_size <= 320, "Used version of Imagewoof provides only resolutions up to 320x320"
        assert mode in ["train", "val"], "mode must be either 'train' or 'val'"
        dir = Path('data/imagewoof2-320') / mode
        available_files = sorted(list(dir.rglob('*.JPEG')))
        self.data = []
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: img.float().div(255.0))
        ])
        #available_files = available_files[:128*4]  # limit for testing
        for file in tqdm(available_files, desc=f'loading {mode} dataset', unit='img'):
            label = self.IMGNET_LABEL.index(file.parent.name)
            img = read_image(str(file), ImageReadMode.RGB)
            img = transform(img)

            self.data.append({
                'image': img,
                'label': label,
                'filename': file.stem
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl, filename = self.data[idx]['image'], self.data[idx]['label'], self.data[idx]['filename']

        return img, lbl  # , filename


class ImageWoofDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 128, spatial_size: int = 224, use_data_aug: bool = True,
                 data_aug_std: float = 0.06):
        super().__init__()
        self.spatial_size = spatial_size
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

        self.use_data_aug = use_data_aug
        self.data_aug_std = data_aug_std

        self.mean = torch.tensor([0.3677, 0.3448, 0.2981]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.3043, 0.2908, 0.2799]).view(1, 3, 1, 1)

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.train_dataset = ImageWoofDataset('train', img_size=self.spatial_size)
            self.val_dataset = ImageWoofDataset('val', img_size=self.spatial_size)
        elif stage == 'test':
            self.test_dataset = ImageWoofDataset('val', img_size=self.spatial_size)
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

        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        x = (x - self.mean) / self.std

        return x, y


if __name__ == '__main__':

    ds = ImageWoofDataset('train', img_size=224)
    imgs = torch.stack([img['image'] for img in ds.data])
    lbls = torch.tensor([lbl['label'] for lbl in ds.data])

    # calculate mean and std
    pixel_cnt = 0
    sum = torch.zeros(3)
    sum_squared = torch.zeros(3)
    for batch in tqdm(torch.tensor_split(imgs, len(ds) // 512)):
        batch = torch.flatten(batch.transpose(0, 1), 1)
        pixel_cnt += batch.shape[-1]
        sum += batch.sum(-1)
        sum_squared += batch.pow(2).sum(-1)
    mean = sum / pixel_cnt
    variance = (sum_squared / pixel_cnt) - (mean ** 2)
    print('mean', mean)
    print('std', torch.sqrt(variance))

    n_classes = len(ds.LABEL)
    lbl, lbl_cnt = torch.unique(lbls, return_counts=True)
    assert torch.equal(lbl, torch.sort(lbl).values)
    lbl_cnt = lbl_cnt.float()
    # root inverse frequency
    weight = torch.sqrt(lbl_cnt.sum() / lbl_cnt)
    print('label counts', lbl_cnt.tolist())
    print('loss weight', weight.tolist())
