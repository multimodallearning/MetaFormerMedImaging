import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="Experiment to evaluate")
experiment = parser.parse_args().experiment

df_abs = pd.read_csv(f'eval/ranking_scores/{experiment}_abs.csv', index_col=0)
df_rel = pd.read_csv(f'eval/ranking_scores/{experiment}_relativ.csv', index_col=0)

df_merge = pd.merge(df_abs, df_rel, on='TokenMixer')

df_tuples = pd.DataFrame({
    col: list(zip(df_merge[col+'_x'], df_merge[col+'_y'].round(2))) for col in df_abs.columns
}, index=df_merge.index)
df_tuples['gmean'] = df_rel['gmean'].round(3).astype(str)
if experiment != 'seg': # reorder columns to match order of paper
    df_tuples = df_tuples.iloc[:, [1, 3, 0, 4, 2, -1]]

print(df_tuples.to_string())
print(df_tuples.to_latex())

if experiment == "2P2T":
    df_rel.reset_index(drop=False, inplace=True)
    df_rel['TokenMixer'] = df_rel['TokenMixer'].apply(lambda t: '_'.join([t.rsplit('_', 2)[0], t.rsplit('_', 2)[-1], t.rsplit('_', 2)[-2]]))
    df_rel.set_index('TokenMixer', inplace=True)

df_rel[['TokenMixer', 'Kernel']] = df_rel.index.to_series().str.rsplit('_', expand=True, n=1)
df_rel.reset_index(inplace=True, drop=True)
print(df_rel.groupby('TokenMixer')['gmean'].mean().sort_values(ascending=False))
print(df_rel.groupby('Kernel')['gmean'].mean().sort_values(ascending=False))

