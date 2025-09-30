import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="Experiment to evaluate")
experiment = parser.parse_args().experiment.lower()

abs_df = pd.read_csv(f'eval/ranking_scores/{experiment}_abs.csv', index_col=0)
rel_df = pd.read_csv(f'eval/ranking_scores/{experiment}_relativ.csv', index_col=0)

df_merge = pd.merge(abs_df, rel_df, on='TokenMixer')

df_tuples = pd.DataFrame({
    col: list(zip(df_merge[col+'_x'], df_merge[col+'_y'].round(2))) for col in abs_df.columns
}, index=df_merge.index)
df_tuples['gmean'] = rel_df['gmean'].round(3).astype(str)
if experiment != 'seg': # reorder columns to match order of paper
    df_tuples = df_tuples.iloc[:, [1, 3, 0, 4, 2, -1]]

print(df_tuples.to_string())
print(df_tuples.to_latex())


