from pathlib import Path

import pandas as pd
import torch

from eval import rank_utils

# construct metric matrix
dsc_dir = Path('eval/dsc_scores')
ds_dirs = sorted(list(dsc_dir.glob('*')))
tm_names = list()
df = dict()
df_normalize = dict()
for ds in ds_dirs:
    dsc_values = list()
    temp_tm_names = list()
    for tm in sorted(list(ds.glob('*.pth'))):
        dsc_values.append(torch.load(str(tm), weights_only=True))
        temp_tm_names.append(tm.stem)
    dsc_values = torch.stack(dsc_values)
    if len(tm_names) > 0:
        assert temp_tm_names == tm_names
    else:  # first iteration
        tm_names = temp_tm_names

    # ranking
    ranking_scores = rank_utils.scores_better(dsc_values, 'ranksums')
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
