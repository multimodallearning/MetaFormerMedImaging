from pathlib import Path

import torch
from monai.data import GridPatchDataset, DataLoader, Dataset, CacheDataset
from monai import transforms
from matplotlib import pyplot as plt

base = Path('data/tiger')

train_data = (base / 'imagesTr').glob('*.png')
train_data = [
    {'image': str(img_path), 'label': str(base / 'labelsTr' / img_path.name)}
    for img_path in train_data
]

x = {'image':torch.randn(3, 256, 320), 'label': torch.randn(1, 256, 320)}
print('before', x['image'].shape, x['label'].shape)
x_transformed = transforms.SpatialPadD(['image', 'label'], 300, mode='reflect')(x)
print('after', x_transformed['image'].shape, x_transformed['label'].shape)

ds = CacheDataset(train_data, cache_rate=0.,
                  transform=transforms.Compose([
                      transforms.LoadImaged(['image', 'label']),
                      transforms.ToTensord(['image', 'label'], dtype=torch.uint8),
                      transforms.EnsureChannelFirstd(['image', 'label']),
                      transforms.SpatialPadD(['image', 'label'], 1024, mode='reflect'),
                      transforms.RandCropByLabelClassesD(['image', 'label'], 'label', 1024, ratios=[0, 1, 0, 0, 0, 0, 0, 0], num_classes=8, num_samples=1, allow_smaller=True),
                      # transforms.RandSpatialCropSamplesD(['image', 'label'], roi_size=256, num_samples=4),
                  ]))
dl = DataLoader(ds, batch_size=2)
sample = next(iter(dl))
for i in ds:
    for sample in i:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(sample['image'].permute(1, 2, 0), cmap='gray')
        plt.title('Image Patch')
        plt.subplot(1, 2, 2)
        plt.imshow(sample['label'].squeeze(), cmap='jet')
        plt.title('Label Patch')
        plt.tight_layout()
    plt.show()

pass
