import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from plot_train_results import adjust_axes


def correlation_plot(df, cols, class_names=['dev', 'hk'], save_path=None):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hexbin(df[cols[0]], df[cols[1]], bins='log')
    
    adjust_axes(ax)
    # axes[0].set_ylabel('Predicted fold change [log2]', fontsize=10)

    pcc = pearsonr(df[cols[0]], df[cols[1]])[0]
    fig.suptitle(f'Correlation between the groups\n(PCC = {pcc:.3f})', fontsize=14)

    # fig.suptitle(title, fontsize=14)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 

    if class_names == ['dev', 'hk']:
        ax.set_xlabel('Developmental', fontsize=10)
        ax.set_ylabel('Housekeeping', fontsize=10)
    else:
        ax.set_xlabel('Primary', fontsize=10)
        ax.set_ylabel('Organoid', fontsize=10)

    if save_path:
        plt.savefig(save_path)
    plt.show()


input_file = 'data/deep-starr/Sequences_activity_Train.txt'
df = pd.read_csv(input_file, sep='\t')
cols = ['Dev_log2_enrichment', 'Hk_log2_enrichment']
correlation_plot(df, cols, save_path='plots/dev_hk_correlation.png')

input_file = 'data/lenti-mpra/da_library/preprocessed/Sequences_activity_Train.txt'
df = pd.read_csv(input_file, sep='\t')
cols = ['Primary_log2_enrichment', 'Organoid_log2_enrichment']
correlation_plot(df, cols, class_names=['prim', 'org'], save_path='plots/prim_org_correlation.png')
