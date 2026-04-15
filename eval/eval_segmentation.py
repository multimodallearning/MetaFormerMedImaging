import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from clearml import Task
from monai import transforms, inferers, metrics
from torch.nn.functional import one_hot
from tqdm import tqdm


def nanstd(o, dim, keepdim=False):
    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim
        )
    )

    if keepdim:
        result = result.unsqueeze(dim)

    return result


def to_one_hot(x: torch.Tensor, c: int):
    assert x.shape[1] == 1
    x = one_hot(x.long(), c)
    x = x.squeeze(1).movedim(-1, 1)
    return x


def get_class_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)  # Split module vs class
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


parser = argparse.ArgumentParser("Evaluate experiment")
parser.add_argument("task_id", type=str, help="ClearML task ID")

task_id = parser.parse_args().task_id
task = Task.get_task(task_id)
param = task.get_parameters(cast=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
try:
    model_class = param['Args/fit.model.class_path']
except KeyError:
    model_class = "models.poolformer_segmentor.MetaFormerSegmentator"
model = get_class_from_path(model_class).load_from_checkpoint(task.artifacts['best.ckpt'].get_local_copy()).eval()
model = model.to(device)

# load dataset
try:
    dataset_class = param['Args/fit.data.class_path']
except KeyError:  # work around since the first trainings were only implemented on Graz
    dataset_class = 'datasets.grazpedwri_dataset.SegGrazPedWriDataModule'
dataset_name = dataset_class.split('.')[-1]
mask_background = False
if dataset_name == 'JSRTDataModule':
    dataset = get_class_from_path(dataset_class)(32, False)
    transform = transforms.ToDevice(device)  # images already z-std
    inferer = inferers.SimpleInferer()
elif dataset_name == 'TIGERDataModule':
    dataset = get_class_from_path(dataset_class)(8, param['Args/fit.data.init_args.spatial_size'])
    transform = transforms.ToDevice(device)  # z-std already done within dataset
    inferer = inferers.SlidingWindowInferer(param['Args/fit.data.init_args.spatial_size'], 1,
                                            mode='gaussian', padding_mode='reflect', device='cpu')
    mask_background = True
elif dataset_name == 'SegGrazPedWriDataModule':
    dataset = get_class_from_path(dataset_class)(32, False)
    transform = transforms.Compose([transforms.NormalizeIntensity(dataset.mean, dataset.std),
                                    transforms.ToDevice(device), transforms.ToTensor(track_meta=False)])
    inferer = inferers.SimpleInferer()
elif dataset_name == 'AbdomenAtlasDataModule':
    dataset = get_class_from_path(dataset_class)(8, param['Args/fit.data.init_args.patch_size'])
    transform = transforms.ToDevice(device)  # preprocessing already done within dataset
    inferer = inferers.SlidingWindowInferer(param['Args/fit.data.init_args.patch_size'], 1,
                                            mode='gaussian', padding_mode='constant', device=device)
    mask_background = True
else:
    raise NotImplementedError(f'Datset class {dataset_class} not know.')
dataset.setup('test')

dsc_metric = metrics.DiceMetric(not model.has_background, "none", num_classes=model.n_classes)
hdd_metric = metrics.HausdorffDistanceMetric(not model.has_background, percentile=95, reduction="none")

with torch.inference_mode():
    for sample in tqdm(dataset.test_dataloader(), desc='Predicting'):
        if isinstance(sample, (tuple, list)):
            x, y = sample
        elif isinstance(sample, dict):
            x, y = sample['image'], sample['label']
        else:
            raise TypeError(f'Unknown sample type: {type(sample)}')

        x = transform(x)
        y_hat = inferer(x, model)
        y_hat = y_hat.argmax(1, keepdim=True) if model.cls_mtl_exclude else y_hat.greater_equal(0)
        y_hat = y_hat.cpu()

        if mask_background:  # Ground truth in TIGER does not always cover the whole image hence a lot of background
            assert model.has_background and model.cls_mtl_exclude
            y_hat[y == 0] = 0

        dsc_metric(y_hat, y)
        if model.cls_mtl_exclude:
            hdd_metric(to_one_hot(y_hat, model.n_classes), to_one_hot(y, model.n_classes))
        else:
            hdd_metric(y_hat, y)

dsc_values = dsc_metric.aggregate()
hdd_values = hdd_metric.aggregate()
stats = torch.stack([dsc_values.nanmean(0), hdd_values.nanmean(0)], 1)
df = pd.DataFrame(stats, columns=['Dice', 'Hausdorff 95%'])
df['label'] = model.label
df.set_index('label', inplace=True)
df.loc['global'] = [dsc_values.nanmean().item(), hdd_values.nanmean().item()]
print('\n', task.name)
print(df.to_string())
print(str(round(dsc_values.nanmean().item(), 4)) + ',', round(hdd_values.nanmean().item(), 4))

try:
    kernel_size = param['Args/fit.model.init_args.kernel_size']
except KeyError:
    kernel_size = param['Args/fit.model.init_args.conv_kernel']

print(task.name, kernel_size)

# save instance DSC scores for s12 variants
if param['Args/fit.model.init_args.model_name'] and param['Args/fit.model.init_args.model_name'] != 'poolformer_s12':
    print('Not saving results to .pth, because other than s12 variant.')
else:
    token_mixer = param['Args/fit.model.init_args.token_mixer']
    base_path = Path('./eval/dsc_scores') / dataset_name
    base_path.mkdir(parents=True, exist_ok=True)
    torch.save(dsc_values.nanmean(1), base_path / f'{token_mixer}_{kernel_size}.pth')
