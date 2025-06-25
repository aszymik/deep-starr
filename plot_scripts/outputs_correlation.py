import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from plot_train_results import adjust_axes


def correlation_plot(df, cols, class_names=['Developmental', 'Housekeeping'], save_path=None):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hexbin(df[cols[0]], df[cols[1]], bins='log')
    
    adjust_axes(ax)

    pcc = pearsonr(df[cols[0]], df[cols[1]])[0]
    fig.suptitle(f'Correlation between the groups\n(PCC = {pcc:.3f})', fontsize=14)

    ax.set_xlabel(class_names[0], fontsize=10)
    ax.set_ylabel(class_names[1], fontsize=10)

    if save_path:
        plt.savefig(save_path)
    plt.show()


input_file = 'data/deep-starr/Sequences_activity_Train.txt'
df = pd.read_csv(input_file, sep='\t')
cols = ['Dev_log2_enrichment', 'Hk_log2_enrichment']
correlation_plot(df, cols, save_path='plots/dev_hk_correlation.png')

