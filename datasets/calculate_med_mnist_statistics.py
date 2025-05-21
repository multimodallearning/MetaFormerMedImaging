import medmnist
from tqdm import tqdm
import torch

def list_print(x:torch.Tensor, dec:int=5):
    return [round(e, dec) for e in x.tolist()]

ds_name = 'ChestMNIST'

DataClass = getattr(medmnist, medmnist.INFO[ds_name.lower()]['python_class'])
ds_kwargs = {'root': './data', 'download': True, 'size': 224}
ds = DataClass('train', **ds_kwargs)
imgs = torch.from_numpy(ds.imgs)

# calculate mean and std
n_channels = ds.info['n_channels']
if n_channels == 1:
    imgs = imgs.unsqueeze(-1)
pixel_cnt = torch.zeros(n_channels)
sum = torch.zeros(n_channels)
sum_squared = torch.zeros(n_channels)
for batch in tqdm(torch.tensor_split(imgs, len(ds) // 512)):
    batch = batch.float().div(255).permute(3, 0, 1, 2).flatten(1)

    pixel_cnt += batch[0].numel()
    sum += batch.sum(-1)
    sum_squared += batch.pow(2).sum(-1)
mean = sum / pixel_cnt
variance = (sum_squared / pixel_cnt) - (mean ** 2)
print('dataset name:', ds_name.lower(), '\n')
print('mean', list_print(mean))
print('std', list_print(variance.sqrt()), '\n')

task_type = ds.info['task']
n_classes = len(ds.info['label'])
if task_type in ['multi-class', 'binary-class']:
    lbl, lbl_cnt = torch.unique(torch.from_numpy(ds.labels), return_counts=True)
    assert all(lbl == torch.sort(lbl).values)
    lbl_cnt = lbl_cnt.float()
    # root inverse frequency
    weight = torch.sqrt(lbl_cnt.sum() / lbl_cnt)
elif task_type.split(',')[0] == 'multi-label':
    # positive example weighting
    lbl_cnt = torch.from_numpy(ds.labels).sum(0)
    pos = lbl_cnt
    neg = lbl_cnt.sum() - lbl_cnt
    weight = neg / pos
else:
    raise ValueError(f"Unknown task type: {task_type}")
weight /= weight.min()
print('label counts', lbl_cnt.tolist())
print('loss weight', list_print(weight))
