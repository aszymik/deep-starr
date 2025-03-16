import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Bio import SeqIO
from datasets import DNADataset
from torch.utils.data import DataLoader
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, sequences):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    seq_matrix = one_hot_encode_dna(sequences)
    X_tensor = torch.tensor(seq_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_dev, pred_hk = model(X_tensor)

    return pred_dev.cpu().numpy().squeeze(), pred_hk.cpu().numpy().squeeze()

if __name__ == '__main__':
    # args = parse_args(sys.argv[1:])

    print('Loading sequences...')
    # sequences = load_fasta_sequences(args.seq)
    sequences = load_fasta_sequences('data/Sequences_Test.fa')

    print('Loading model...')
    # model = load_model(args.model, PARAMS)
    model = load_model('outputs/DeepSTARR.model', PARAMS)

    print('Predicting...')
    pred_dev, pred_hk = predict(model, sequences)

    # Save predictions
    out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_dev': pred_dev, 'Predictions_hk': pred_hk})
    # out_filename = f'{args.seq}_predictions_{args.model}.txt'
    out_filename = 'outputs/Test_predictions.txt'
    out_prediction.to_csv(out_filename, sep='\t', index=False)

    print(f'\nPredictions saved to {out_filename}\n')
