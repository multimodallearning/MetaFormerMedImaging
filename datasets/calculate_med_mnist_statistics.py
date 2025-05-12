from datasets.med_mnist_dataset import MedMNISTDataModule
from medmnist.info import INFO
import numpy as np
from tqdm import tqdm
import math

ds_name = 'OrganAMNIST'

ds = MedMNISTDataModule(ds_name, batch_size=1, spatial_size=224)
ds.setup('fit')
ds = ds.train_dataset

# calculate mean and std
pixel_cnt = 0
sum = 0
sum_squared = 0.
for batch in tqdm(np.array_split(ds.imgs, len(ds) // 512)):
    batch = batch.astype('float32')
    batch /= 255.0

    pixel_cnt += batch.size
    sum += batch.sum()
    sum_squared += np.pow(batch, 2).sum()
mean = sum / pixel_cnt
variance = (sum_squared / pixel_cnt) - (mean ** 2)
print('mean', mean)
print('std', math.sqrt(variance))

task_type = INFO[ds_name.lower()]['task']
n_classes = len(INFO[ds_name.lower()]['label'])
lbl, lbl_cnt = np.unique(ds.labels, return_counts=True)
assert all(lbl == np.sort(lbl))
lbl_cnt = lbl_cnt.astype('float32')
if task_type == 'multi-class':
    # root inverse frequency
    weight = np.sqrt(lbl_cnt.sum() / lbl_cnt)
elif task_type in ['binary-class', 'multi-label']:
    # positive example weighting
    pos = lbl_cnt
    neg = lbl_cnt.sum() - lbl_cnt
    weight = neg / pos
else:
    raise ValueError(f"Unknown task type: {task_type}")
print('label counts', lbl_cnt.tolist())
print('loss weight', weight.tolist())
