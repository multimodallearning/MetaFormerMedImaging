from pathlib import Path

import pandas as pd
import torch
from scipy.stats import gmean

from eval import rank_utils

# construct metric matrix
dsc_dir = Path('eval/dsc_scores')
ds_dirs = sorted(list(dsc_dir.glob('*')))
df = dict()
df_normalize = dict()
for ds in ds_dirs:
    dsc_values = list()
    tm_names = list()
    for tm in sorted(list(ds.glob('*.pth'))):
        dsc_values.append(torch.load(str(tm), weights_only=True))
        tm_names.append(tm.stem)
    dsc_values = torch.stack(dsc_values)
    # ranking
    ranking_scores = rank_utils.scores_better(dsc_values, 'signed-rank')
    df[ds.name] = pd.Series(index=tm_names,  data=ranking_scores.tolist())
    ranking_scores_normalized = rank_utils.rankscore_avgtie(ranking_scores)
    df_normalize[ds.name] = pd.Series(index=tm_names,  data=ranking_scores_normalized.tolist())

df = pd.DataFrame(df)
print(df.to_string())

df_normalize = pd.DataFrame(df_normalize)
df_normalize['gmean'] = df_normalize[~df_normalize.index.str.endswith('_9')].apply(lambda row: gmean(row.dropna()), 1)
df_normalize['gmean'] = df_normalize['gmean'].round(3).astype(str)
#df_normalize.sort_values(by='gmean', ascending=False, inplace=True)
print(df_normalize.to_string())

# save
df.to_csv('eval/ranking_scores/seg_abs.csv', index_label='TokenMixer')
df_normalize.to_csv('eval/ranking_scores/seg_relativ.csv', index_label='TokenMixer')
