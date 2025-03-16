import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from io import StringIO
from plot_data_distribution import adjust_axes


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


def df_from_predictions(prediction_file):
    num_seq = line_count(prediction_file)
    data = np.zeros((num_seq, 4))
    with open(prediction_file, 'r') as file:
        for i, line in enumerate(file):
            elts = re.split(r":|\(|_|\t|\[|\]", line)
            coordinates = elts[1].split('-')
            length = int(coordinates[1])-int(coordinates[0])
            observed = np.float64(elts[4])
            predicted = np.float64(elts[6])
            data[i] = [length, observed, predicted, observed - predicted]

    df = pd.DataFrame(data, columns=['length', 'observed', 'predicted', 'difference'])
    return df


def df_from_prediction_list(prediction_list):
    num_seq = len(prediction_list)
    data = np.zeros((num_seq, 4))
    for i, line in enumerate(prediction_list):
        elts = re.split(r":|\(|_|\t|\[|\]", line)
        coordinates = elts[1].split('-')
        length = int(coordinates[1])-int(coordinates[0])
        try:
            observed = np.float64(elts[4])
            predicted = np.float64(elts[6])
        except ValueError:
            observed = np.float64(elts[3])
            predicted = np.float64(elts[5])
        data[i] = [length, observed, predicted, observed - predicted]

    df = pd.DataFrame(data, columns=['length', 'observed', 'predicted', 'difference'])
    return df


def df_from_linear_model_output(prediction_file):
    df = pd.read_csv(prediction_file, sep='\t')
    df.columns = ['id', 'length', 'GC content', 'observed', 'predicted']
    df['difference'] = df['observed'] - df['predicted']
    return df


def df_from_linear_model_output_list(prediction_list):
    prediction_text = ''.join(prediction_list)
    prediction_file = StringIO(prediction_text)
    return df_from_linear_model_output(prediction_file)


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

def regression_error_plot(prediction_file, title, linear_model=False):
    if not linear_model:
        df = df_from_predictions(prediction_file)
    else:
        df = df_from_linear_model_output(prediction_file)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(df['length'], df['difference'], bins='log')
    axes[0].set_xlabel('Sequence length', fontsize=11)
    axes[0].set_ylabel('Predicted – observed fold change', fontsize=11)
    adjust_axes(axes[0])

    # axes[1].scatter(df['observed'], df['predicted'], marker='.', color='indigo')
    axes[1].hexbin(df['observed'], df['predicted'], bins='log')
    axes[1].set_xlabel('Observed fold change', fontsize=11)
    axes[1].set_ylabel('Predicted fold change', fontsize=11)
    adjust_axes(axes[1])

    fig.suptitle(title, fontsize=14)
    plt.show() 


def length_vs_error(prediction_file, title, linear_model=False):
    if not linear_model:
        df = df_from_predictions(prediction_file)
    else:
        df = df_from_linear_model_output(prediction_file)

    fig, axes = plt.subplots(figsize=(8, 5))
    axes.hexbin(df['length'], df['difference'], bins='log')
    axes.set_xlabel('Sequence length', fontsize=10)
    axes.set_ylabel('Predicted – observed fold change', fontsize=10)
    adjust_axes(axes)

    mse = metrics.mean_squared_error(df['observed'], df['predicted'])
    plt.title(f'MSE: {mse:.3f}', fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.show() 

def predicted_vs_observed_fold_change(prediction_file, title, linear_model=False):
    if not linear_model:
        df = df_from_predictions(prediction_file)
    else:
        df = df_from_linear_model_output(prediction_file)

    fig, axes = plt.subplots(figsize=(8, 5))
    axes.hexbin(df['observed'], df['predicted'], bins='log')
    # axes.scatter(df['observed'], df['predicted'], marker='.', color='indigo')
    axes.set_xlabel('Observed fold change', fontsize=10)
    axes.set_ylabel('Predicted fold change', fontsize=10)
    adjust_axes(axes)

    r2 = metrics.r2_score(df['observed'], df['predicted'])
    pearson = df['observed'].corr(df['predicted'])

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    plt.title(f'PCC: {pearson:.3f}, R2 score: {r2:.3f}', fontsize=10)
    plt.show() 


def length_vs_error_two_groups(active, silent, title, linear_model=False, list=False):
    if not linear_model:
        if list:
            df_active = df_from_prediction_list(active)
            df_silent = df_from_prediction_list(silent)
        else:
            df_active = df_from_predictions(active)
            df_silent = df_from_predictions(silent)
    else:
        df_active = df_from_linear_model_output(active)
        df_silent = df_from_linear_model_output(silent)

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    axes[0].hexbin(df_active['length'], df_active['difference'], bins='log')
    axes[1].hexbin(df_silent['length'], df_silent['difference'], bins='log')


    fig.supxlabel('Sequence length', fontsize=10)
    axes[0].set_ylabel('Predicted – observed fold change', fontsize=10)
    adjust_axes(axes[0])
    adjust_axes(axes[1])

    mse_act = metrics.mean_squared_error(df_active['observed'], df_active['predicted'])
    mse_sil = metrics.mean_squared_error(df_silent['observed'], df_silent['predicted'])
    
    axes[0].set_title(f'Active (MSE = {mse_act:.3f})', fontsize=10)
    axes[1].set_title(f'Silent (MSE = {mse_sil:.3f})', fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.show() 


def predicted_vs_observed_fold_change_two_groups(active, silent, title, linear_model=False, list=False, custom_group_names=None):
    if not linear_model:
        if list:
            df_active = df_from_prediction_list(active)
            df_silent = df_from_prediction_list(silent)
        else:
            df_active = df_from_predictions(active)
            df_silent = df_from_predictions(silent)
    else:
        df_active = df_from_linear_model_output(active)
        df_silent = df_from_linear_model_output(silent)

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    axes[0].hexbin(df_active['observed'], df_active['predicted'], bins='log')
    axes[1].hexbin(abs(df_silent['observed']), abs(df_silent['predicted']), bins='log')

    fig.supxlabel('Absolute value of observed fold change', fontsize=10)
    axes[0].set_ylabel('Absolute value of predicted fold change', fontsize=10)
    adjust_axes(axes[0])
    adjust_axes(axes[1])

    r2_act = metrics.r2_score(df_active['observed'], df_active['predicted'])
    pearson_act = df_active['observed'].corr(df_active['predicted'])

    r2_sil = metrics.r2_score(df_silent['observed'], df_silent['predicted'])
    pearson_sil = df_silent['observed'].corr(df_silent['predicted'])

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    if custom_group_names:
        axes[0].set_title(f'{custom_group_names[0]} (PCC = {pearson_act:.3f})', fontsize=10)
        axes[1].set_title(f'{custom_group_names[1]} (PCC = {pearson_sil:.3f})', fontsize=10)
    else:
        axes[0].set_title(f'Active (PCC = {pearson_act:.3f}, R2 score = {r2_act:.3f})', fontsize=10)
        axes[1].set_title(f'Silent (PCC = {pearson_sil:.3f}, R2 score = {r2_sil:.3f})', fontsize=10)
    plt.show()    


def compare_length_vs_error(active, silent, active_baseline, silent_baseline, title, predictions_as_list=False):
    fig = plt.figure(figsize=(12,7))
    
    ax0 = fig.add_subplot(2, 1, 1, frameon=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2, frameon=False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4 = fig.add_subplot(2, 2, 3)
    ax5 = fig.add_subplot(2, 2, 4)

    if predictions_as_list:
        df_active = df_from_prediction_list(active)
        df_silent = df_from_prediction_list(silent)
        df_active_bl = df_from_linear_model_output_list(active_baseline)
        df_silent_bl = df_from_linear_model_output_list(silent_baseline)
        
    else:
        df_active = df_from_predictions(active)
        df_silent = df_from_predictions(silent)
        df_active_bl = df_from_linear_model_output(active_baseline)
        df_silent_bl = df_from_linear_model_output(silent_baseline)


    ax1.hexbin(df_active['length'], df_active['difference'], bins='log')
    ax2.hexbin(df_silent['length'], df_silent['difference'], bins='log')
    ax4.hexbin(df_active_bl['length'], df_active_bl['difference'], bins='log')
    ax5.hexbin(df_silent_bl['length'], df_silent_bl['difference'], bins='log')

    fig.supxlabel('Sequence length', fontsize=10)
    fig.supylabel('Predicted – observed fold change', fontsize=10, x=0.06)

    adjust_axes(ax1)
    adjust_axes(ax2)
    adjust_axes(ax4)
    adjust_axes(ax5)

    r2_act = metrics.r2_score(df_active['observed'], df_active['predicted'])
    pearson_act = df_active['observed'].corr(df_active['predicted'])
    r2_sil = metrics.r2_score(df_silent['observed'], df_silent['predicted'])
    pearson_sil = df_silent['observed'].corr(df_silent['predicted'])

    r2_act_bl = metrics.r2_score(df_active_bl['observed'], df_active_bl['predicted'])
    pearson_act_bl = df_active_bl['observed'].corr(df_active_bl['predicted'])
    r2_sil_bl = metrics.r2_score(df_silent_bl['observed'], df_silent_bl['predicted'])
    pearson_sil_bl = df_silent_bl['observed'].corr(df_silent_bl['predicted'])
    
    ax0.set_title('CNN', y=1.05)
    ax1.set_title(f'Active (PCC = {pearson_act:.3f}, R2 score = {r2_act:.3f})', fontsize=10)
    ax2.set_title(f'Silent (PCC = {pearson_sil:.3f}, R2 score = {r2_sil:.3f})', fontsize=10)
    ax3.set_title('Linear regression model', y=1.08)
    ax4.set_title(f'Active (PCC = {pearson_act_bl:.3f}, R2 score = {r2_act_bl:.3f})', fontsize=10)
    ax5.set_title(f'Silent (PCC = {pearson_sil_bl:.3f}, R2 score = {r2_sil_bl:.3f})', fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=0.4) 
    plt.show() 

def compare_predicted_vs_observed(active, silent, active_baseline, silent_baseline, title, predictions_as_list=False):
    fig = plt.figure(figsize=(12,7))
    
    ax0 = fig.add_subplot(2, 1, 1, frameon=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2, frameon=False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4 = fig.add_subplot(2, 2, 3)
    ax5 = fig.add_subplot(2, 2, 4)

    if predictions_as_list:
        df_active = df_from_prediction_list(active)
        df_silent = df_from_prediction_list(silent)
        df_active_bl = df_from_linear_model_output_list(active_baseline)
        df_silent_bl = df_from_linear_model_output_list(silent_baseline)
        
    else:
        df_active = df_from_predictions(active)
        df_silent = df_from_predictions(silent)
        df_active_bl = df_from_linear_model_output(active_baseline)
        df_silent_bl = df_from_linear_model_output(silent_baseline)

    ax1.hexbin(df_active['observed'], df_active['predicted'], bins='log')
    ax2.hexbin(df_silent['observed'], df_silent['predicted'], bins='log')
    ax4.hexbin(df_active_bl['observed'], df_active_bl['predicted'], bins='log')
    ax5.hexbin(df_silent_bl['observed'], df_silent_bl['predicted'], bins='log')

    fig.supxlabel('Observed fold change', fontsize=10)
    fig.supylabel('Predicted fold change', fontsize=10, x=0.06)

    adjust_axes(ax1)
    adjust_axes(ax2)
    adjust_axes(ax4)
    adjust_axes(ax5)

    mse_act = metrics.mean_squared_error(df_active['observed'], df_active['predicted'])
    mse_sil = metrics.mean_squared_error(df_silent['observed'], df_silent['predicted'])
    mse_act_bl = metrics.mean_squared_error(df_active_bl['observed'], df_active_bl['predicted'])
    mse_sil_bl = metrics.mean_squared_error(df_silent_bl['observed'], df_silent_bl['predicted'])
    
    ax0.set_title('CNN', y=1.05)
    ax1.set_title(f'Active (MSE = {mse_act:.3f})', fontsize=10)
    ax2.set_title(f'Silent (MSE = {mse_sil:.3f})', fontsize=10)
    ax3.set_title('Linear regression model', y=1.08)
    ax4.set_title(f'Active (MSE = {mse_act_bl:.3f})', fontsize=10)
    ax5.set_title(f'Silent (MSE = {mse_sil_bl:.3f})', fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=0.4) 
    plt.show()  


# Dwa osobne modele dla reference i alternative

predicted_ref = 'data/regression_lentiMPRA/outputs/ref_2000_predictions'
predicted_alt = 'data/regression_lentiMPRA/outputs/alt_2000_predictions'

train_ref = 'data/regression_lentiMPRA/outputs/lenti_reference_train.txt'
valid_ref = 'data/regression_lentiMPRA/outputs/lenti_reference_valid.txt'
test_ref = 'data/regression_lentiMPRA/outputs/lenti_reference_test.txt'

train_alt = 'data/regression_lentiMPRA/outputs/lenti_alternative_train.txt'
valid_alt = 'data/regression_lentiMPRA/outputs/lenti_alternative_valid.txt'
test_alt = 'data/regression_lentiMPRA/outputs/lenti_alternative_test.txt'


if __name__ == '__main__':
    predicted_ref_train = filter_predictions(predicted_ref, train_ref)
    predicted_ref_valid = filter_predictions(predicted_ref, valid_ref)
    predicted_ref_test = filter_predictions(predicted_ref, test_ref)

    predicted_alt_train = filter_predictions(predicted_alt, train_alt)
    predicted_alt_valid = filter_predictions(predicted_alt, valid_alt)
    predicted_alt_test = filter_predictions(predicted_alt, test_alt)

    # Two separate models
    predicted_vs_observed_fold_change_two_groups(predicted_ref_train, predicted_alt_train, 'Predicted vs. observed activity for training set', list=True, custom_group_names=['Reference', 'Alternative'])
    predicted_vs_observed_fold_change_two_groups(predicted_ref_valid, predicted_alt_valid, 'Predicted vs. observed activity for validation set', list=True, custom_group_names=['Reference', 'Alternative'])
    predicted_vs_observed_fold_change_two_groups(predicted_ref_test, predicted_alt_test, 'Predicted vs. observed activity for test set', list=True, custom_group_names=['Reference', 'Alternative'])
