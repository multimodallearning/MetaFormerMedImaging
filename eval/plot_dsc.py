import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

base_dir = Path("eval/dsc_scores")

data = []

for dataset_dir in base_dir.iterdir():
    if dataset_dir.is_dir():
        for pred_file in dataset_dir.glob("*.pth"):
            parts = pred_file.stem.split('_')
            # Parse token_mixer and kernel
            if "attn" in parts:
                attn_idx = parts.index("attn")
                token_mixer = "_".join(parts[:attn_idx+1])
                kernel_str = parts[attn_idx+1]
            elif len(parts) >= 3 and parts[0] == "sep" and parts[1] == "conv":
                token_mixer = "sep_conv"
                kernel_str = parts[2]
            else:
                token_mixer = parts[0]
                kernel_str = parts[1]
            if token_mixer in ['full_attn', 'identity', 'random']:
                kernel = '~'
            else:
                kernel = int(kernel_str)
            dsc_tensor = torch.load(pred_file, weights_only=True)
            dsc_mean = dsc_tensor.mean().item()
            data.append({
                'tokenmixer': token_mixer,
                'kernel': kernel,
                'dataset': dataset_dir.name,
                'DSC': dsc_mean
            })

df = pd.DataFrame(data)
df_plot = df.copy()

# For plotting, convert all kernel to string so numbers are handled together
df_plot['kernel'] = df_plot['kernel'].astype(str)

# Set kernel order: 3, 5, 7, 9, ~
kernel_order = ['3', '5', '7', '9', '~']
df_plot['kernel'] = pd.Categorical(df_plot['kernel'], categories=kernel_order, ordered=True)

# Rename tokenmixers
rename_dict = {
    'sep_conv': 'grouped conv',
    'loc_attn': 'local attn',
    'full_attn': 'global attn',
}
df_plot['tokenmixer'] = df_plot['tokenmixer'].replace(rename_dict)

# Rename datasets
dataset_rename = {
    "JSRTDataModule": "JSRT",
    "SegGrazPedWriDataModule": "GRAZ",
    "TIGERDataModule": "TIGER",
}
df_plot['dataset'] = df_plot['dataset'].map(dataset_rename)
dataset_order = ["JSRT", "GRAZ", "TIGER"]

df_plot['dataset'] = pd.Categorical(
    df_plot['dataset'],
    categories=dataset_order,
    ordered=True
)

# Add dodge for x positions
order = ['pooling', 'conv', 'grouped conv', 'local attn', 'global attn', 'random', 'identity']
hue_order = [name for name in order if name in df_plot['tokenmixer'].unique()]
df_plot = df_plot[df_plot['tokenmixer'].isin(hue_order)]  # Filter to only include tokenmixers in the order
n_hue = len(hue_order)
dodge_factor = 0.8  # Tune this value to adjust dodging amount (smaller = less spread)
dodge = dodge_factor / n_hue if n_hue > 1 else 0
kernel_to_num = {k: i for i, k in enumerate(kernel_order)}
df_plot['kernel_num'] = df_plot['kernel'].map(kernel_to_num).astype(int)
df_plot['hue_index'] = df_plot['tokenmixer'].map({h: i for i, h in enumerate(hue_order)}).astype(int)
df_plot['x_pos'] = df_plot['kernel_num'] + dodge * (df_plot['hue_index'] - (n_hue - 1) / 2)

# Create line_df for monotonic groups
line_df = df_plot.copy()
grouped = line_df.groupby(['tokenmixer', 'dataset'])
to_keep = []
for (tokenmixer, dataset), group in grouped:
    group_sorted = group.sort_values('kernel')
    dsc_list = group_sorted['DSC'].tolist()
    if len(dsc_list) > 1:
        is_increasing = all(dsc_list[i] <= dsc_list[i + 1] for i in range(len(dsc_list) - 1))
        is_decreasing = all(dsc_list[i] >= dsc_list[i + 1] for i in range(len(dsc_list) - 1))
        if is_increasing or is_decreasing:
            to_keep.extend(group.index.tolist())
line_df = line_df.loc[to_keep]

marker_map = {
    'pooling': 'o',
    'conv': 's',
    'grouped conv': 'D',
    'local attn': '^',
    'global attn': 'v',
    'random': 'X',
    'identity': 'P',
}

# Plot
n_datasets = df_plot['dataset'].nunique()
g = sns.relplot(
    data=df_plot,
    x='x_pos',
    y='DSC',
    hue='tokenmixer',
    style='tokenmixer',
    markers=marker_map,
    kind='scatter',
    col='dataset',
    col_order=dataset_order,
    col_wrap=n_datasets,
    height=4,
    aspect=0.8,
    s=90,
    hue_order=hue_order
)
g.set_titles("{col_name}", size=16)
g.set_axis_labels("Kernel Size", "DSC")
g.fig.subplots_adjust(wspace=0.1)

# Set x ticks to kernel labels
for ax in g.axes.flatten():
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(kernel_order)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

# Remove the default legend
if hasattr(g, '_legend') and g._legend is not None:
    g._legend.remove()

# Get handles and labels from the first axis (flattened)
first_ax = g.axes.flatten()[0]
handles, labels = first_ax.get_legend_handles_labels()

# Place a single legend below all plots, centered
g.fig.legend(
    handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.06)
)

# Remove vertical grid lines to show only horizontal lines
for ax in g.axes.flatten():
    ax.grid(axis='x', visible=False)

# Get color mapping from legend
color_dict = {label: handle.get_color() for label, handle in zip(labels, handles)}

# Plot lines for monotonic groups
for ax in g.axes.flatten():
    dataset_name = ax.get_title()
    for tokenmixer in line_df['tokenmixer'].unique():
        subset = line_df[(line_df['dataset'] == dataset_name) & (line_df['tokenmixer'] == tokenmixer)].sort_values(
            'kernel')
        if not subset.empty:
            ax.plot(subset['x_pos'], subset['DSC'], color=color_dict[tokenmixer], linestyle='-')

plt.tight_layout()
g.fig.savefig("dsc_plot.pdf", bbox_inches='tight', pad_inches=0)
print("Plot saved to dsc_plot.pdf")
# plt.show()
