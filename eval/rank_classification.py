import argparse
from pathlib import Path

import pandas as pd
import torch
from torchmetrics.functional.classification import multiclass_auroc
from tqdm import tqdm

from eval import rank_utils

n_bootstraps = 5000

parser = argparse.ArgumentParser("Evaluate experiment")
parser.add_argument("--signature", type=str, help="Architecture signature to evaluate [4T, 2P2T].", default='4T',
                    required=False)
signature = parser.parse_args().signature
print('Evaluating architecture signatur:', signature)

# construct metric matrix
y_hat_dir = Path('eval/classification_predictions')
ds_dirs = sorted(list(y_hat_dir.glob('*')))
tm_names = list()
df = dict()
df_normalize = dict()
for ds in ds_dirs:
    y = torch.load(str(ds / 'gt.pth'), weights_only=True)
    n_classes = y.max().item() + 1  # including 0
    n_samples = len(y)
    auc_values = list()
    temp_tm_names = list()
    bootstrap_idx = torch.randint(low=0, high=n_samples, size=(n_bootstraps, n_samples),
                                  generator=torch.Generator().manual_seed(42))
    for tm in tqdm(sorted(list(ds.glob('*.pth'))), desc=f'Bootstrapping {ds.name}', unit='TokenMixer'):
        if tm.stem == 'gt' or (  # skipping ground truth
                # experiment selection
                signature == '4T' and tm.stem.startswith('2P2T')) or (
                signature == '2P2T' and tm.stem.startswith('4T')):
            continue

        y_hat = torch.load(str(tm), weights_only=True).softmax(1)
        auc = torch.empty(n_bootstraps)
        for i, idx in enumerate(bootstrap_idx):
            auc[i] = multiclass_auroc(y_hat[idx], y[idx], average='macro', num_classes=n_classes)
        temp_tm_names.append(tm.stem)
        auc_values.append(auc)
    auc_values = torch.stack(auc_values)
    if len(tm_names) > 0:
        assert temp_tm_names == tm_names
    else:  # first iteration
        tm_names = temp_tm_names

    # ranking
    ranking_scores = rank_utils.scores_better(auc_values, 'CI-based')
    df[ds.name] = ranking_scores.tolist()
    ranking_scores_normalized = rank_utils.rankscore_avgtie(ranking_scores)
    df_normalize[ds.name] = ranking_scores_normalized.tolist()

df = pd.DataFrame.from_dict(df)
df.insert(0, 'TokenMixer', tm_names)
df.set_index('TokenMixer', inplace=True)
print(df.to_string())

df_normalize = pd.DataFrame.from_dict(df_normalize)
df_normalize.insert(0, 'TokenMixer', tm_names)
df_normalize.set_index('TokenMixer', inplace=True)
df_normalize['mean'] = df_normalize.mean(1)
print(df_normalize.to_string())
