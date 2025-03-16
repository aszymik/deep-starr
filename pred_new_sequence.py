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

def load_model(model_path, params):
    model = DeepSTARR(params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def predict(model, set_name, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_loader = prepare_input(set_name, batch_size)
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
    # sequences = load_fasta_sequences(args.seq)
    sequences = load_fasta_sequences('data/Sequences_Test.fa')

    print('Loading model...')
    # model = load_model(args.model, PARAMS)
    model = load_model('outputs/DeepSTARR.model', PARAMS)

    print('Predicting...')
    pred_dev, pred_hk = predict(model, 'Test')  # ta funkcja do zmiany

    # Save predictions
    out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_dev': pred_dev, 'Predictions_hk': pred_hk})
    # out_filename = f'{args.seq}_predictions_{args.model}.txt'
    out_filename = 'outputs/Pred_activity_Test.txt'
    out_prediction.to_csv(out_filename, sep='\t', index=False)

    print(f'\nPredictions saved to {out_filename}\n')
