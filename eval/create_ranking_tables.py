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
    df_tuples = df_tuples.iloc[:, [2, 0, 3, 1, 4]]

print(df_tuples.to_string())
print(df_tuples.to_latex())

def parse_index(idx):
    prefix = experiment + "_"

    # Remove prefix if present
    if idx.startswith(prefix):
        rest = idx[len(prefix):]
    else:
        rest = idx

    parts = rest.split("_")

    # Last part is temp (warm/cold) or kernel
    last = parts[-1]

    if last in ['warm', 'cold']:
        # Last is temperature, check for kernel before it
        if len(parts) > 1 and parts[-2].isdigit():
            token_mixer = "_".join(parts[:-2])
            kernel = parts[-2]
        else:
            token_mixer = "_".join(parts[:-1])
            kernel = ""
    elif last.isdigit():
        # Last is kernel
        token_mixer = "_".join(parts[:-1])
        kernel = last
    else:
        # Neither, so no kernel
        token_mixer = "_".join(parts)
        kernel = ""

    return token_mixer, kernel

df_rel['TokenMixer'], df_rel['Kernel'] = zip(*df_rel.index.map(parse_index))
df_rel.reset_index(inplace=True, drop=True)
print(df_rel.groupby('TokenMixer')['gmean'].mean().sort_values(ascending=False))
print(df_rel.groupby('Kernel')['gmean'].mean().sort_values(ascending=False))
