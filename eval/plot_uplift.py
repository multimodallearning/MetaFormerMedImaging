import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load the CSV
df = pd.read_csv('eval/df_auc_scratch_pretrained_joined.csv')

datasets = ['PathMNIST', 'DermaMNIST', 'PneumoniaMNIST', 'OrganSMNIST']

data = []

for _, row in df.iterrows():
    tokenmixer = row['TokenMixer']
    kernel = str(row['Kernel Size'])
    if kernel in ['-', '1']:
        kernel = '~'

    for dataset in datasets:
        auc_scratch = row[f'{dataset}_auc_scratch']
        auc_pretrained = row[f'{dataset}_auc_pretrained']
        if auc_scratch > 0:
            uplift = (auc_pretrained - auc_scratch) / auc_scratch * 100
            data.append({
                'tokenmixer': tokenmixer,
                'kernel': kernel,
                'dataset': dataset,
                'uplift': uplift
            })

df_plot = pd.DataFrame(data)

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
    "PathMNIST": "Path",
    "DermaMNIST": "Derma",
    "PneumoniaMNIST": "Pneumonia",
    "OrganSMNIST": "OrganS",
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

# Create line_df for monotonic groups (though uplift may not be monotonic, keeping for consistency)
line_df = df_plot.copy()
grouped = line_df.groupby(['tokenmixer', 'dataset'])
to_keep = []
for (tokenmixer, dataset), group in grouped:
    group_sorted = group.sort_values('kernel')
    uplift_list = group_sorted['uplift'].tolist()
    if len(uplift_list) > 1:
        is_increasing = all(uplift_list[i] <= uplift_list[i + 1] for i in range(len(uplift_list) - 1))
        is_decreasing = all(uplift_list[i] >= uplift_list[i + 1] for i in range(len(uplift_list) - 1))
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
    y='uplift',
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
g.set_axis_labels("Kernel Size", "Improvement [%]")
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
            ax.plot(subset['x_pos'], subset['uplift'], color=color_dict[tokenmixer], linestyle='-')

plt.tight_layout()
g.fig.savefig("uplift.pdf", bbox_inches='tight', pad_inches=0)
print("Plot saved to uplift.pdf")
#plt.show()
