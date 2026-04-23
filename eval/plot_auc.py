import argparse
import torch
from pathlib import Path
from torchmetrics import classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

parser = argparse.ArgumentParser("Plot AUC for predictions")
parser.add_argument("experiment", choices=["2p2t", "4t"], help="Experiment type: 2p2t or 4t")
args = parser.parse_args()

prefix = args.experiment.upper()  # 2P2T or 4T

base_dir = Path("eval/classification_predictions")

data = []

for dataset_dir in base_dir.iterdir():
    if dataset_dir.is_dir():
        if dataset_dir.name == "imagewoof":
            continue
        gt_path = dataset_dir / "gt.pth"
        if not gt_path.exists():
            continue
        gt = torch.load(gt_path, weights_only=True)
        is_multiclass = len(gt.shape) == 1
        if is_multiclass:
            num_classes = int(gt.max()) + 1
            task = "multiclass"
        else:
            num_classes = gt.shape[1]
            task = "multilabel"

        auroc_metric = classification.AUROC(num_classes=num_classes, task=task, average=None)

        for pred_file in dataset_dir.glob(f"{prefix}_*.pth"):
            parts = pred_file.stem.split('_')
            match prefix:
                case '4T':
                    # Handle token mixer + kernel logic
                    if "attn" in parts:
                        # full_attn or loc_attn
                        token_mixer = "_".join(parts[1:3])
                        rest = parts[3:]
                    elif parts[1] == "sep":
                        # sep_conv
                        token_mixer = "_".join(parts[1:3])
                        rest = parts[3:]
                    else:
                        token_mixer = parts[1]
                        rest = parts[2:]

                    # Kernel (if exists and is numeric)
                    kernel = '~'
                    if rest:
                        if rest[0].isdigit():
                            kernel = int(rest[0])
                case '2P2T':
                    parts = parts[1:]  # remove prefix

                    # --- extract warm/cold ---
                    mode = None
                    if parts[-1] in ("warm", "cold"):
                        mode = parts[-1]
                        parts = parts[:-1]

                        # --- detect token mixer ---
                    if len(parts) >= 2 and parts[0] == "sep":
                        token_mixer = "sep_conv"
                        parts = parts[2:]
                    elif len(parts) >= 2 and parts[0] in ("loc", "full") and parts[1] == "attn":
                        token_mixer = f"{parts[0]}_attn" if mode == "cold" else f"{parts[0]}_attn_warm"
                        parts = parts[2:]
                    else:
                        token_mixer = parts[0]
                        parts = parts[1:]

                        # --- kernel ---
                    kernel = '~'
                    if parts:
                        if parts[0].isdigit():
                            kernel = int(parts[0])

            pred = torch.load(pred_file, weights_only=True)
            if is_multiclass:
                pred = pred.softmax(dim=1)
            auroc_metric.reset()
            auroc_metric.update(pred, gt)
            auc_scores = auroc_metric.compute()
            auc_mean = auc_scores.mean().item()
            if token_mixer == 'random':
                kernel = '~'
            data.append({
                'tokenmixer': token_mixer,
                'kernel': kernel,
                'dataset': dataset_dir.name,
                'AUC': auc_mean
            })

df = pd.DataFrame(data)
df_plot = df.copy()

# For plotting, convert all kernel to string so '~' and numbers are handled together
df_plot['kernel'] = df_plot['kernel'].astype(str)

# Set kernel order: 3, 5, 7, ~
kernel_order = ['3', '5', '7', '~']
df_plot['kernel'] = pd.Categorical(df_plot['kernel'], categories=kernel_order, ordered=True)

# Rename tokenmixers
rename_dict = {
    'sep_conv': 'grouped conv',
    'loc_attn': 'local attn',
    'loc_attn_warm': 'local attn (warm start)',
    'full_attn': 'global attn',
    'full_attn_warm': 'global attn (warm start)',
}
df_plot['tokenmixer'] = df_plot['tokenmixer'].replace(rename_dict)

# Rename datasets
dataset_rename = {
    "pathmnist": "Path",
    "dermamnist": "Derma",
    "pneumoniamnist": "Pneumonia",
    "organsmnist": "OrganS",
}
df_plot['dataset'] = df_plot['dataset'].map(dataset_rename)
dataset_order = ["Path", "Derma", "Pneumonia", "OrganS"]

df_plot['dataset'] = pd.Categorical(
    df_plot['dataset'],
    categories=dataset_order,
    ordered=True
)

# Add dodge for x positions
order = ['pooling', 'conv', 'grouped conv', 'local attn', 'global attn', 'random', 'identity',
         'local attn (warm start)', 'global attn (warm start)']
hue_order = [name for name in order if name in df_plot['tokenmixer'].unique()]
df_plot = df_plot[df_plot['tokenmixer'].isin(hue_order)]  # Filter to only include tokenmixers in the order
n_hue = len(hue_order)
dodge_factor = 0.4  # Tune this value to adjust dodging amount (smaller = less spread)
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
    auc_list = group_sorted['AUC'].tolist()
    if len(auc_list) > 1:
        is_increasing = all(auc_list[i] <= auc_list[i + 1] for i in range(len(auc_list) - 1))
        is_decreasing = all(auc_list[i] >= auc_list[i + 1] for i in range(len(auc_list) - 1))
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
    'local attn (warm start)': '^',  # same shape, same family
    'global attn (warm start)': 'v',
}

# Plot
n_datasets = df_plot['dataset'].nunique()
g = sns.relplot(
    data=df_plot,
    x='x_pos',
    y='AUC',
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
g.set_axis_labels("Kernel Size", "AUC")
g.fig.subplots_adjust(wspace=0.1)

# Set x ticks to kernel labels
for ax in g.axes.flatten():
    ax.set_xticks([0, 1, 2, 3])
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
            ax.plot(subset['x_pos'], subset['AUC'], color=color_dict[tokenmixer], linestyle='-')

plt.tight_layout()
g.fig.savefig(f"auc_plot_{args.experiment}.pdf", bbox_inches='tight', pad_inches=0)
print(f"Plot saved to auc_plot_{args.experiment}.png")
plt.show()
