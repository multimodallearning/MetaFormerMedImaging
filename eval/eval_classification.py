import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from clearml import Task
from torchmetrics import classification, MetricCollection
from tqdm import tqdm

from models.metaformer_classifier import AdaptiveMetaformerClassifier
from models.pretrained_metaformer_classifier import PretrainedMetaformer


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
model_class = param['Args/fit.model.class_path']
model = get_class_from_path(model_class).load_from_checkpoint(task.artifacts['best.ckpt'].get_local_copy()).eval()
model = model.to(device)

# load dataset
dataset_class = param['Args/fit.data.class_path']
dataset_name = dataset_class.split('.')[-1]

if dataset_name == 'ImageWoofDataModule':
    dataset = get_class_from_path(dataset_class)(param['Args/fit.data.init_args.batch_size'],
                                                 param['Args/fit.data.init_args.spatial_size'], False)
    mean = dataset.mean
    std = dataset.std
elif dataset_name == 'MedMNISTDataModule':
    dataset = get_class_from_path(dataset_class)(param['Args/fit.data.init_args.ds_name'],
                                                 param['Args/fit.data.init_args.batch_size'],
                                                 param['Args/fit.data.init_args.spatial_size'], None)
    mean = dataset.mean
    std = dataset.std
else:
    raise NotImplementedError(f'Datset class {dataset_class} not know.')
dataset.setup('test')
mean = mean.to(device, non_blocking=True)
std = std.to(device, non_blocking=True)

metrics_kwargs = dict(num_classes=model.n_classes, num_labels=model.n_classes, average=None,
                      task='multiclass' if model.cls_mtl_exclude else 'multilabel')
metrics = MetricCollection({
    "acc": classification.Accuracy(**metrics_kwargs),
    "f1": classification.F1Score(**metrics_kwargs),
    "prec": classification.Precision(**metrics_kwargs),
    "rec": classification.Recall(**metrics_kwargs),
    "auroc": classification.AUROC(**metrics_kwargs)
})

pred = []
gt = []
with torch.inference_mode():
    for x, y in tqdm(dataset.test_dataloader(), desc='Predicting'):
        x = (x.to(device) - mean) / std
        y_hat = model(x)
        if model.cls_mtl_exclude:
            y = y.squeeze(-1)
        else:
            y_hat = y_hat.sigmoid()
        metrics(y_hat.cpu(), y)
        pred.append(y_hat)
        gt.append(y)

metrics_dict = metrics.compute()
metrics_dict['label'] = model.label
df = pd.DataFrame.from_dict(metrics_dict)
df.set_index('label', inplace=True)
df = pd.concat([df, df.describe().loc[['mean', 'std']]])
print('\n', task.name)
print(df.to_string())
print(', '.join(map(lambda s: str(round(s, 4)), df.loc['mean', ['acc', 'auroc', 'f1']])))
print(task.name, param['Args/fit.model.init_args.kernel_size'])  # , param['Args/fit.model.init_args.pretrained'])

# save prediction and ground truth for ranking
pred = torch.cat(pred).cpu()
gt = torch.cat(gt).cpu()
assert len(pred) == len(gt), 'Number of predictions and ground truths do not align.'
if isinstance(model, PretrainedMetaformer):
    architecture_signature = "2P2T"
    assert param['Args/fit.model.init_args.pretrained'], 'Ranking only for pretrained 2P2T. Skipping saving predictions.'
elif isinstance(model, AdaptiveMetaformerClassifier):
    architecture_signature = "4T"
else:
    raise ValueError('Unknown model type')
token_mixer = param['Args/fit.model.init_args.tokenmixer']
kernel = str(param['Args/fit.model.init_args.kernel_size'])
file_name = [architecture_signature, token_mixer, kernel]
try:
    param_start = 'warm' if param['Args/fit.model.init_args.attention_weights_warm_start'] else 'cold'
    file_name.append(param_start)
except KeyError:
    pass
file_name = '_'.join(file_name)

base_path = Path('./eval/classification_predictions') / param['Args/fit.model.init_args.ds_name']
base_path.mkdir(parents=True, exist_ok=True)
# disable saving for architecture others than s12
if param['Args/fit.model.init_args.architecture'] and param['Args/fit.model.init_args.architecture'] != 's12':
    print('Not saving results to .pth, because other than s12 variant.')
else:
    torch.save(pred, base_path / (file_name + '.pth'))
    torch.save(gt, base_path / 'gt.pth')
