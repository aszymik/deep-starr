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

def predict(model, set_name, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_loader = prepare_input(set_name, batch_size, shuffle=False)
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
    # args = parse_args(sys.argv[1:])

    print('Loading sequences...')
    set_name = 'Test'
    # sequences = load_fasta_sequences(args.seq)
    sequences = load_fasta_sequences(f'data/Sequences_{set_name}.fa')

    seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    for seed in seeds:
        print('Loading model...')
        model = load_model(f'models/DeepSTARR_{seed}.model', PARAMS)

        print('Predicting...')
        pred_dev, pred_hk = predict(model, set_name)  # ta funkcja do zmiany
        out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_dev': pred_dev, 'Predictions_hk': pred_hk})
        
        out_filename = f'outputs/Pred_new_torch_{seed}_{set_name}.txt'
        out_prediction.to_csv(out_filename, sep='\t', index=False)
        print(f'\nPredictions saved to {out_filename}\n')

