from clearml import Task
import importlib
import torch
from torchmetrics.classification import F1Score
from tqdm import tqdm
import pandas as pd
from monai import transforms, inferers, metrics
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class FakeTrainer:
    fast_dev_run: bool = False


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


def get_class_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)  # Split module vs class
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


task_id = "80ca4198e4ea4210aee0d66007911330"
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
except KeyError: # work around since the first trainings were only implemented on Graz
    dataset_class = 'datasets.grazpedwri_dataset.SegGrazPedWriDataModule'
dataset_name = dataset_class.split('.')[-1]
mask_background = False
if dataset_name == 'JSRTDataModule':
    dataset = get_class_from_path(dataset_class)(32, False)
    transform = transforms.ToDevice(device)  # images already z-std
    inferer = inferers.SimpleInferer()
elif dataset_name == 'TIGERDataModule':
    dataset = get_class_from_path(dataset_class)(8, param['Args/fit.data.init_args.spatial_size'])
    dataset.trainer = FakeTrainer()
    transform = transforms.ToDevice(device)  # z-std already done within dataset
    inferer = inferers.SlidingWindowInferer(param['Args/fit.data.init_args.spatial_size'], 1,
                                           mode='gaussian' ,padding_mode='reflect', device='cpu')
    mask_background = True
elif dataset_name == 'SegGrazPedWriDataModule':
    dataset = get_class_from_path(dataset_class)(32, False)
    transform = transforms.Compose([transforms.NormalizeIntensity(dataset.mean, dataset.std),
                                    transforms.ToDevice(device), transforms.ToTensor(track_meta=False)])
    inferer = inferers.SimpleInferer()
else:
    raise NotImplementedError(f'Datset class {dataset_class} not know.')
dataset.setup('test')

dsc_metric = metrics.DiceMetric(not model.has_background, "none", num_classes=model.n_classes)

dsc_values = []
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

        if mask_background: # Ground truth in TIGER does not always cover the whole image hence a lot of background
            assert model.has_background and model.cls_mtl_exclude
            y_hat[y == 0] = 0

        dsc_metric(y_hat.cpu(), y)

dsc_values = dsc_metric.aggregate()
dsc_stats = torch.stack([dsc_values.nanmean(0), nanstd(dsc_values, 0)], 1)
df = pd.DataFrame(dsc_stats, columns=['mean', 'std'])
df['label'] = model.label
df.set_index('label', inplace=True)
df.loc['global'] = [dsc_values.nanmean().item(), dsc_values[~dsc_values.isnan()].std().item()]
print('\n', task.name)
print(df.to_string())
print(round(dsc_values.nanmean().item(), 4), '±', round(dsc_values[~dsc_values.isnan()].std().item(), 4))
print(task.name)
