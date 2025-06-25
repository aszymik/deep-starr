import torch
import pandas as pd
import numpy as np
import argparse

from models import *
from utils import *


def parse_args(argv):
    parser = argparse.ArgumentParser(description='DeepSTARR sequence prediction')
    parser.add_argument('-s', '--seq', required=True, help='FASTA file with sequences')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained PyTorch model')
    return parser.parse_args(argv)

def predict(model, set_name, batch_size=128, set_dir='data/deep-starr', activity_cols=['Dev_log2_enrichment', 'Hk_log2_enrichment']):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_loader = prepare_input(set_name, batch_size, set_dir, activity_cols, shuffle=False)
    pred_dev_list, pred_hk_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            pred_dev, pred_hk = model(batch)
            pred_dev_list.append(pred_dev.cpu().numpy())
            pred_hk_list.append(pred_hk.cpu().numpy())

    # Concatenate predictions from all batches
    pred_dev = np.concatenate(pred_dev_list)
    pred_hk = np.concatenate(pred_hk_list)

    return pred_dev.squeeze(), pred_hk.squeeze()


if __name__ == '__main__':

    print('Loading sequences...')
    set_name = 'Test'
    sequences = load_fasta_sequences(f'data/deep-starr/Sequences_{set_name}.fa')
    seed = 7898

    print('Loading model...')
    model = load_model(f'models/deep-starr/DeepSTARR_model.model', PARAMS)

    print('Predicting...')
    set_dir = 'data/deep-starr'
    pred_dev, pred_hk = predict(model, set_name, set_dir=set_dir)

    out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_developmental': pred_dev, 'Predictions_housekeeping': pred_hk})
    out_filename = f'outputs/deep-starr/Predictions_{set_name}.txt'
    out_prediction.to_csv(out_filename, sep='\t', index=False)
    print(f'\nPredictions saved to {out_filename}\n')

