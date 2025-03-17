from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def dataframes(path):
    df = pd.read_csv(path, header=0)

    df_train = df[['Epoch', 'Train Loss', 'Train MSE Dev', 'Train MSE Hk', 'Train PCC Dev', 'Train PCC Hk', 'Train SCC Dev', 'Train SCC Hk']]
    df_valid = df[['Epoch', 'Val Loss', 'Val MSE Dev', 'Val MSE Hk', 'Val PCC Dev', 'Val PCC Hk', 'Val SCC Dev', 'Val SCC Hk']]

    print(df_train, df_valid)
    return df_train, df_valid


def adjust_axes(ax, y_bottom=None, y_top=None, measure=''):
    # Adjust plot parameters
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_facecolor('gainsboro')
    ax.grid(color='white', alpha=0.5)
    ax.set_axisbelow(True)
    if y_bottom:
        ax.set_ylim(y_bottom, y_top)
    ax.set_ylabel(f'{measure}')


def plot_loss(path, title):

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(x = df['Epoch'], y = df[f'Train MSE Dev'], s=3, alpha=0.6)
    axes[0].scatter(x = df['Epoch'], y = df[f'Val MSE Dev'], s=3, alpha=0.6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('MSE Developmental')
    adjust_axes(axes[0], -0.1, 5.1)

    axes[1].scatter(x = df['Epoch'], y = df[f'Train MSE Hk'], s=3, alpha=0.6, label='training')
    axes[1].scatter(x = df['Epoch'], y = df[f'Val MSE Hk'], s=3, alpha=0.6, label='validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('MSE Housekeeping')
    adjust_axes(axes[1], -0.1, 5.1)
    axes[1].legend()

    plt.subplots_adjust(top=0.85)
    fig.suptitle(title, fontsize=16)
    plt.show()

def plot_pcc_and_scc(path, title):

    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0,0].scatter(x = df['Epoch'], y = df[f'Train PCC Dev'], s=3, alpha=0.6, label='training')
    axes[0,0].scatter(x = df['Epoch'], y = df[f'Val PCC Dev'], s=3, alpha=0.6, label='validation')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_title('PCC Developmental')
    adjust_axes(axes[0,0], -0.1, 1.1)

    axes[0,1].scatter(x = df['Epoch'], y = df[f'Train PCC Hk'], s=3, alpha=0.6)
    axes[0,1].scatter(x = df['Epoch'], y = df[f'Val PCC Hk'], s=3, alpha=0.6)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_title('PCC Housekeeping')
    adjust_axes(axes[0,1], -0.1, 1.1)

    axes[1,0].scatter(x = df['Epoch'], y = df[f'Train SCC Dev'], s=3, alpha=0.6)
    axes[1,0].scatter(x = df['Epoch'], y = df[f'Val SCC Dev'], s=3, alpha=0.6)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_title('SCC Developmental')
    adjust_axes(axes[1,0], -0.1, 1.1)

    axes[1,1].scatter(x = df['Epoch'], y = df[f'Train SCC Hk'], s=3, alpha=0.6)
    axes[1,1].scatter(x = df['Epoch'], y = df[f'Val SCC Hk'], s=3, alpha=0.6)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_title('SCC Housekeeping')
    adjust_axes(axes[1,1], -0.1, 1.1)

    plt.subplots_adjust(hspace=0.348)
    fig.suptitle(title, fontsize=16)
    plt.show()


    'Val PCC Dev', 'Val PCC Hk', 'Val SCC Dev', 'Val SCC Hk'


if __name__ == '__main__':
    RES = 'outputs/training_log.csv'
    plot_loss(RES, 'Training DeepSTARR on Drosophila melanogaster S2 cells enhancers')
    plot_pcc_and_scc(RES, 'Training DeepSTARR on Drosophila melanogaster S2 cells enhancers')