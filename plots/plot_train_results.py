from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def dataframes(path):
    df = pd.read_csv(path, header=0)

    df_train = df[['Epoch', 'Train Loss', 'Train MSE Dev', 'Train MSE Hk', 'Train PCC Dev', 'Train PCC Hk', 'Train SCC Dev', 'Train SCC Hk']]
    df_valid = df[['Epoch', 'Val Loss', 'Val MSE Dev', 'Val MSE Hk', 'Val PCC Dev', 'Val PCC Hk', 'Val SCC Dev', 'Val SCC Hk']]

    print(df_train, df_valid)

    return df_train, df_valid


def prediction_dataframes(path):
    # Returns only the first row (epoch 0 results) of each dataframe
    df_train, df_valid = dataframes(path)
    return df_train[df_train.index == 0], df_valid[df_valid.index == 1]


def loss_dataframes(path):
    # Data frame for regression results: only one column – loss
    df = pd.read_csv(path, sep='\t', header=0)
    return df[df['Stage'] == 'train'], df[df['Stage'] == 'valid']  


def adjust_axes(ax, y_bottom, y_top, measure=''):
    # Adjust plot parameters
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_facecolor('gainsboro')
    ax.grid(color='white', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(y_bottom, y_top)
    ax.set_ylabel(f'{measure}')


def plot_train_results(path, title, three_classes=False):
    df_train, df_valid = dataframes(path)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    # to do: (measure, ylim) list to adjust the y-axis
    measures = [('Loss', 0.97, 1.63), ('Sensitivity', -0.05, 1.05), ('Specificity', -0.05, 1.05), ('AUC', -0.05, 1.05)]
    dfs = df_train, df_valid

    for i in range(len(measures)):
        # Change dataframes to have important columns (instead of tuple columns)
        measure, y_bottom, y_top = measures[i]

        for j in range(len(dfs)):
            # Plot results for both datasets
            pa = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - PA'], color='blue', s=3, alpha=0.6)
            na = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - NA'], color='orange', s=3, alpha=0.6)
            pi = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - PI'], color='green', s=3, alpha=0.6)
            ni = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - NI'], color='red', s=3, alpha=0.6)

            # Adjust plot parameters
            adjust_axes(axes[j,i], y_bottom, y_top, measure)
            axes[(j,i)].set_xlabel('Epoch')
        
    axes[(0,2)].set_title('Training', x=-0.2, y=1.05)
    axes[(1,2)].set_title('Validation', x=-0.2, y=1.05)
    fig.suptitle(title, fontsize=16)

    if three_classes:
        plt.figlegend((pa, na, pi, ni), ('active regions', 'silent regions', 'inactive regions', 'none'), loc='lower center', ncol=4, fontsize=8)
    else:
        plt.figlegend((pa, na, pi, ni), ('promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive'), loc='lower center', ncol=4, fontsize=8)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    plt.show()


def plot_loss(path, title):

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(x = df['Epoch'], y = df[f'Train MSE Dev'], s=3, alpha=0.6)
    axes[0].scatter(x = df['Epoch'], y = df[f'Val MSE Dev'], s=3, alpha=0.6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('MSE Dev')
    adjust_axes(axes[1], -0.1, 5.1)

    axes[1].scatter(x = df['Epoch'], y = df[f'Train MSE Hk'], s=3, alpha=0.6, label='training')
    axes[1].scatter(x = df['Epoch'], y = df[f'Val MSE Hk'], s=3, alpha=0.6, label='validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('MSE Hk')
    adjust_axes(axes[2], -0.1, 5.1)
    axes[1].legend()

    fig.suptitle(title, fontsize=16)
    plt.show()

def plot_pcc_and_scc(path, title):

    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0,0].scatter(x = df['Epoch'], y = df[f'Train PCC Dev'], s=3, alpha=0.6, label='training')
    axes[0,0].scatter(x = df['Epoch'], y = df[f'Val PCC Dev'], s=3, alpha=0.6, label='validation')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_title('PCC Dev')
    adjust_axes(axes[0,0], -0.1, 1.1)

    axes[0,1].scatter(x = df['Epoch'], y = df[f'Train PCC Hk'], s=3, alpha=0.6)
    axes[0,1].scatter(x = df['Epoch'], y = df[f'Val PCC Hk'], s=3, alpha=0.6)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_title('PCC Hk')
    adjust_axes(axes[0,1], -0.1, 1.1)

    axes[1,0].scatter(x = df['Epoch'], y = df[f'Train SCC Dev'], s=3, alpha=0.6)
    axes[1,0].scatter(x = df['Epoch'], y = df[f'Val SCC Dev'], s=3, alpha=0.6)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_title('SCC Dev')
    adjust_axes(axes[1,0], -0.1, 1.1)

    axes[1,1].scatter(x = df['Epoch'], y = df[f'Train SCC Hk'], s=3, alpha=0.6)
    axes[1,1].scatter(x = df['Epoch'], y = df[f'Val SCC Hk'], s=3, alpha=0.6)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_title('SCC Hk')
    adjust_axes(axes[1,1], -0.1, 1.1)

    fig.suptitle(title, fontsize=16)
    plt.show()


    'Val PCC Dev', 'Val PCC Hk', 'Val SCC Dev', 'Val SCC Hk'


def plot_joint_loss(results1, results2, title, custom_names=None):

    df_train1, df_valid1 = loss_dataframes(results1)
    df_train2, df_valid2 = loss_dataframes(results2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x = df_train1['Epoch'], y = df_train1[f'Loss'], s=3, alpha=0.6, label='training')
    axes[0].scatter(x = df_valid1['Epoch'], y = df_valid1[f'Loss'], s=3, alpha=0.6, label='validation')
    axes[0].set_xlabel('Epoch') 
    if custom_names:
        axes[0].set_title(f'{custom_names[0]}')
    else:    
        axes[0].set_title('Active sequences')
    adjust_axes(axes[0], -0.2, 2.3, 'MSE')

    train = axes[1].scatter(x = df_train2['Epoch'], y = df_train2[f'Loss'], s=3, alpha=0.6, label='training')
    val = axes[1].scatter(x = df_valid2['Epoch'], y = df_valid2[f'Loss'], s=3, alpha=0.6, label='validation')
    axes[1].set_xlabel('Epoch') 
    if custom_names:
        axes[1].set_title(f'{custom_names[1]}')
    else:
        axes[1].set_title('Silent sequences')
    adjust_axes(axes[1], -0.2, 2.3, '')
    axes[1].legend()
    # plt.figlegend((train, val), ('training', 'validation'), ncol=1, fontsize=8)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    fig.suptitle(title, fontsize=16)
    plt.show()


RES = 'outputs/training_log.csv'

if __name__ == '__main__':
    plot_loss(RES, '')
    plot_pcc_and_scc(RES, '')