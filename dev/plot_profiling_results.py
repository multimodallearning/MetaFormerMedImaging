import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('dev/results.csv')
for spatial_dim, g_df in df.groupby('spatial'):
    row_full_atn = g_df[g_df['operation'] == 'full_attn']
    # g_df = g_df[g_df['operation'] != 'full_attn']
    g_df = g_df[g_df['operation'] == 'flex']
    full_attn_hline_kwargs = dict(xmin=0, xmax=len(g_df['kernel'].unique()) - 1, colors='r', label='full_attn',
                                  linestyle='--')

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(spatial_dim)

    ln_plot_kwargs = dict(x='kernel', hue='operation', data=g_df)

    sns.lineplot(y='flops', ax=axs[0], **ln_plot_kwargs)
    # axs[0].hlines(row_full_atn['flops'].values, **full_attn_hline_kwargs)
    # axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].set_title('FLOPS')

    sns.lineplot(y='macs', ax=axs[1], **ln_plot_kwargs)
    # axs[1].hlines(row_full_atn['macs'].values, **full_attn_hline_kwargs)
    # axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_title('MACs')

    sns.lineplot(y='params', ax=axs[2], **ln_plot_kwargs)
    # axs[2].hlines(row_full_atn['params'].values, **full_attn_hline_kwargs)
    # axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].set_title('#Params')

plt.show()
