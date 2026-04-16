import importlib
from argparse import ArgumentParser
from pathlib import Path

import torch
from clearml import Task
from kornia.enhance import normalize_min_max
from matplotlib import pyplot as plt
from monai import transforms, inferers, metrics
from torch.nn.functional import one_hot


def get_class_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)  # Split module vs class
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def to_one_hot(x: torch.Tensor, c: int):
    assert x.shape[1] == 1
    x = one_hot(x.long(), c)
    x = x.squeeze(1).movedim(-1, 1)
    return x


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


parser = ArgumentParser()
parser.add_argument('dataset', type=str, choices=['jsrt', 'graz', 'tiger'])
ds_name = parser.parse_args().dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
dsc_path = Path('eval/dsc_scores')
match ds_name:
    case 'jsrt':
        dsc_path = dsc_path / 'JSRTDataModule'
        model_ids = dict(
            pool3='5817883ba1d8431d98f1185a5e962c0e',
            gconv7='6e6f45dccd9f4fe1b61daa7bb4aab828',
            rnd='58d6b7a3f97340ee975f26d451850f76')
    case 'graz':
        dsc_path = dsc_path / 'SegGrazPedWriDataModule'
        model_ids = dict(
            pool3='65934d23d6a442cf8df7fe3954bfb989',
            gconv7='2e9ea9fec927426fbe0ab56f92cfdd79',
            rnd='283859768c7c4e8b8375208160f2b24a')
    case 'tiger':
        dsc_path = dsc_path / 'TIGERDataModule'
        model_ids = dict(
            pool3='TBA',
            gconv7='TBA',
            rnd='TBA')
    case _:
        raise ValueError('Unknown dataset')

# determine median case
dsc = torch.stack([torch.load(str(tm), weights_only=True) for tm in dsc_path.glob('*.pth')])
dsc_avg = dsc.mean(0)
median_idx = torch.where(dsc_avg == dsc_avg.median())[0].item()

preds = []
for task_id in model_ids.values():
    task = Task.get_task(task_id)
    param = task.get_parameters(cast=True)

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
    sample = dataset.test_dataloader().dataset[median_idx]
    if isinstance(sample, (tuple, list)):
        x, y = sample
    elif isinstance(sample, dict):
        x, y = sample['image'], sample['label']
    else:
        raise TypeError(f'Unknown sample type: {type(sample)}')

    with torch.inference_mode():
        x = transform(x.unsqueeze(0))
        y_hat = inferer(x, model)
        y_hat = y_hat.argmax(1, keepdim=True) if model.cls_mtl_exclude else y_hat.greater_equal(0)
        y_hat = y_hat.cpu()
        y = y.unsqueeze(0).cpu()

        if mask_background:  # Ground truth in TIGER does not always cover the whole image hence a lot of background
            assert model.has_background and model.cls_mtl_exclude
            y_hat[y == 0] = 0

    dsc_metric = metrics.DiceMetric(not model.has_background, "none", num_classes=model.n_classes)
    hdd_metric = metrics.HausdorffDistanceMetric(not model.has_background, percentile=95, reduction="none")

    dsc_metric(y_hat, y)
    if model.cls_mtl_exclude:
        hdd_metric(to_one_hot(y_hat, model.n_classes), to_one_hot(y, model.n_classes))
    else:
        hdd_metric(y_hat, y)

    dsc_scores = dsc_metric.aggregate().squeeze()
    hdd_distances = hdd_metric.aggregate().squeeze()

    preds.append(dict(mask=y_hat.squeeze(), dsc=(dsc_scores.nanmean().item(), nanstd(dsc_scores, 0).item()),
                      hdd=(hdd_distances.nanmean().item(), nanstd(hdd_distances,0).item())))

fig, axs = plt.subplots(1, 4, figsize=(20, 10))
x_norm = normalize_min_max(x.unsqueeze(0) if x.ndim == 3 else x).movedim(1, -1).squeeze()
for a in axs.flatten():
    a.set_axis_off()
    a.imshow(x_norm, 'gray')

selected_colormap = 'tab10' if model.n_classes <= 10 else 'tab20'
# create legend
cmap = plt.get_cmap(selected_colormap)
norm = plt.Normalize(vmin=0, vmax=model.n_classes - 1)
for i in range(model.n_classes):
    axs[0].scatter([], [], color=cmap(norm(i)), label=dataset.test_dataloader().dataset.LABELS[::-1][i])
axs[0].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='xx-small')

for i, (tokenmixer, pred) in enumerate(zip(model_ids.keys(), preds), start=1):
    if not model.cls_mtl_exclude:
        mask = pred['mask'].flip(0).float().argmax(0)
        alpha_mask = pred['mask'].any(0).float()
    else:
        mask = pred['mask'].max() - pred['mask'] # flip on integer values
        alpha_mask = torch.ones_like(mask, dtype=torch.float)
    axs[i].imshow(mask, alpha=alpha_mask * .8, cmap=selected_colormap)
    axs[i].set_title(tokenmixer)
    print(tokenmixer, pred['dsc'], pred['hdd'])
fig.savefig(f'eval/{dataset_name}_qualitativ.pdf', dpi=450, bbox_inches='tight')
plt.show()
