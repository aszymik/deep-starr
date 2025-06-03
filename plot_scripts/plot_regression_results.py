import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from io import StringIO
from scipy.stats import pearsonr

from plot_train_results import adjust_axes


def line_count(file):
    def _count_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with open(file, 'rb') as fp:
        c_generator = _count_generator(fp.raw.read)
        count = sum(buffer.count(b'\n') for buffer in c_generator) # count each \n
    return count

def filter_predictions(prediction_file, ids_file):
    with open(ids_file, 'r') as file:
        sequences = [line.strip() for line in file]
    sequences_set = set(sequences)
    filtered_predictions = []

    with open(prediction_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            sequence_id = parts[0]
            
            if sequence_id in sequences_set:
                filtered_predictions.append(line)

    return filtered_predictions


def regression_error_plot(prediction_file, title, linear_model=False):
    df = pd.read_csv(prediction_file, sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(df['length'], df['difference'], bins='log')
    axes[0].set_xlabel('Sequence length', fontsize=11)
    axes[0].set_ylabel('Predicted – observed fold change', fontsize=11)
    adjust_axes(axes[0])

    axes[1].hexbin(df['observed'], df['predicted'], bins='log')
    axes[1].set_xlabel('Observed fold change', fontsize=11)
    axes[1].set_ylabel('Predicted fold change', fontsize=11)
    adjust_axes(axes[1])

    fig.suptitle(title, fontsize=14)
    plt.show() 


def predicted_vs_observed(true, predicted, title, class_names=['dev', 'hk'], save_path=None):
    df_true = pd.read_csv(true, sep='\t')
    df_pred = pd.read_csv(predicted, sep='\t')

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    axes[0].hexbin(df_true[f'{class_names[0].capitalize()}_log2_enrichment'], df_pred[f'Predictions_{class_names[0]}'], bins='log')
    axes[1].hexbin(df_true[f'{class_names[1].capitalize()}_log2_enrichment'], df_pred[f'Predictions_{class_names[0]}'], bins='log')

    adjust_axes(axes[0])
    adjust_axes(axes[1])
    fig.supxlabel('Observed fold change [log2]', fontsize=10)
    axes[0].set_ylabel('Predicted fold change [log2]', fontsize=10)

    pcc_dev = pearsonr(df_true[f'{class_names[0].capitalize()}_log2_enrichment'], df_pred[f'Predictions_{class_names[0]}'])[0]
    pcc_hk = pearsonr(df_true[f'{class_names[1].capitalize()}_log2_enrichment'], df_pred[f'Predictions_{class_names[1]}'])[0]

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 

    if class_names == ['dev', 'hk']:
        axes[0].set_title(f'Developmental (PCC = {pcc_dev:.3f})', fontsize=10)
        axes[1].set_title(f'Housekeeping (PCC = {pcc_hk:.3f})', fontsize=10)
    else:
        axes[0].set_title(f'Primary (PCC = {pcc_dev:.3f})', fontsize=10)
        axes[1].set_title(f'Organoid (PCC = {pcc_hk:.3f})', fontsize=10)

    if save_path:
        plt.savefig(save_path)
    plt.show()


set_name = 'Test'
set_to_title = {
    'Test': 'test',
    'Train': 'training',
    'Val': 'validation'
}

if __name__ == '__main__':
    # true = f'data/deep-starr/Sequences_activity_{set_name}.txt'
    # seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    # for seed in seeds:
    #     pred_filename = f'outputs/Pred_new_torch_{seed}_{set_name}.txt'
    #     plot_filename = f'plots/05.05_new_torch/model_{seed}.png'
    #     predicted_vs_observed(true, pred_filename, f'DeepSTARR PyTorch model predictions on the {set_to_title[set_name]} set', save_path=plot_filename)

    true = f'data/lenti-mpra/da_library/preprocessed/Sequences_activity_{set_name}.txt'
    seeds = [7898]#, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    for seed in seeds:
        pred_filename = f'outputs/lenti-mpra/Pred_new_torch_{seed}_{set_name}.txt'
        plot_filename = f'plots/1.06_lenti-mpra/model_{seed}.png'
        predicted_vs_observed(true, pred_filename, f'DeepSTARR PyTorch model predictions on the PsychENCODE data {set_to_title[set_name]} set', class_names=['primary', 'organoid'], save_path=plot_filename)

    true = f'data/lenti-mpra/da_library/split_as_in_paper/Sequences_activity_{set_name}.txt'
    # seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    seeds = [7530]
    for seed in seeds:
        pred_filename = f'outputs/lenti-mpra/split_as_in_paper/Pred_new_torch_{seed}_{set_name}.txt'
        plot_filename = f'plots/1.06_lenti-mpra/model_{seed}_split_from_paper.png'
        predicted_vs_observed(true, pred_filename, f'DeepSTARR PyTorch model predictions on the PsychENCODE {set_to_title[set_name]} set (chr 4)', class_names=['primary', 'organoid'], save_path=plot_filename)

