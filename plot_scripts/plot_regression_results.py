import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from io import StringIO

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


# PLOTS

# def regression_error_plot(prediction_file, title, linear_model=False):
#     if not linear_model:
#         df = df_from_predictions(prediction_file)
#     else:
#         df = df_from_linear_model_output(prediction_file)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].hexbin(df['length'], df['difference'], bins='log')
#     axes[0].set_xlabel('Sequence length', fontsize=11)
#     axes[0].set_ylabel('Predicted – observed fold change', fontsize=11)
#     adjust_axes(axes[0])

#     # axes[1].scatter(df['observed'], df['predicted'], marker='.', color='indigo')
#     axes[1].hexbin(df['observed'], df['predicted'], bins='log')
#     axes[1].set_xlabel('Observed fold change', fontsize=11)
#     axes[1].set_ylabel('Predicted fold change', fontsize=11)
#     adjust_axes(axes[1])

#     fig.suptitle(title, fontsize=14)
#     plt.show() 


# def length_vs_error_two_groups(active, silent, title, linear_model=False, list=False):
    
#     if not linear_model:
#         if list:
#             df_active = df_from_prediction_list(active)
#             df_silent = df_from_prediction_list(silent)
#         else:
#             df_active = df_from_predictions(active)
#             df_silent = df_from_predictions(silent)
#     else:
#         df_active = df_from_linear_model_output(active)
#         df_silent = df_from_linear_model_output(silent)

#     fig, axes = plt.subplots(1,2, figsize=(12, 5))
#     axes[0].hexbin(df_active['length'], df_active['difference'], bins='log')
#     axes[1].hexbin(df_silent['length'], df_silent['difference'], bins='log')


#     fig.supxlabel('Sequence length', fontsize=10)
#     axes[0].set_ylabel('Predicted – observed fold change', fontsize=10)
#     adjust_axes(axes[0])
#     adjust_axes(axes[1])

#     mse_act = metrics.mean_squared_error(df_active['observed'], df_active['predicted'])
#     mse_sil = metrics.mean_squared_error(df_silent['observed'], df_silent['predicted'])
    
#     axes[0].set_title(f'Active (MSE = {mse_act:.3f})', fontsize=10)
#     axes[1].set_title(f'Silent (MSE = {mse_sil:.3f})', fontsize=10)

#     fig.suptitle(title, fontsize=14)
#     plt.show() 


def predicted_vs_observed(true, predicted, title):

    df_true = pd.read_csv(true, sep='\t')
    df_pred = pd.read_csv(predicted, sep='\t')

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    axes[0].hexbin(df_true['Dev_log2_enrichment'], df_pred['Predictions_dev'], bins='log')
    axes[1].hexbin(df_true['Hk_log2_enrichment'], df_pred['Predictions_hk'], bins='log')

    adjust_axes(axes[0])
    adjust_axes(axes[1])
    fig.supxlabel('Observed fold change', fontsize=10)
    axes[0].set_ylabel('Predicted fold change', fontsize=10)

    pearson_dev = df_true['Dev_log2_enrichment'].corr(df_pred['Predictions_dev'])
    pearson_hk = df_true['Hk_log2_enrichment'].corr(df_pred['Predictions_hk'])

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 

    axes[0].set_title(f'Developmental (PCC = {pearson_dev:.3f})', fontsize=10)
    axes[1].set_title(f'Housekeeping (PCC = {pearson_hk:.3f})', fontsize=10)
    plt.show()    

data_set = 'Train'
true = f'data/Sequences_activity_{data_set}.txt'
pred = f'outputs/Pred_activity_{data_set}.txt'


if __name__ == '__main__':
    predicted_vs_observed(true, pred, f'DeepSTARR predictions on the training set')