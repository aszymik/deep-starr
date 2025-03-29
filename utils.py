import torch
from torch.utils.data import DataLoader

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

from datasets import DNADataset
from models import *

PARAMS = {
    'batch_size': 128,
    'epochs': 100,
    'early_stop': 10,
    'kernel_size1': 7,
    'kernel_size2': 3,
    'kernel_size3': 5,
    'kernel_size4': 3,
    'lr': 0.002,
    'num_filters': 256,
    'num_filters2': 60,
    'num_filters3': 60,
    'num_filters4': 120,
    'n_conv_layer': 4,
    'n_add_layer': 2,
    'dropout_prob': 0.4,
    'dense_neurons1': 256,
    'dense_neurons2': 256,
    'pad': 'same'
}

def one_hot_encode_dna(sequences):
    """Converts a list of DNA sequences to a one-hot encoded matrix using sklearn's OneHotEncoder."""

    categories = np.array(['A', 'C', 'G', 'T'])
    encoder = OneHotEncoder(sparse_output=False, categories=[categories], handle_unknown='ignore', dtype=np.float32)
    encoder.fit(categories.reshape(-1, 1))

    one_hot_encoded = np.array([encoder.transform(np.array(list(seq)).reshape(-1, 1)) for seq in sequences])
    return one_hot_encoded.reshape(len(sequences), 4, -1)

def load_fasta_sequences(file_path):
    """Reads sequences from a FASTA file and returns a list of sequences."""
    sequences = [str(record.seq).upper() for record in SeqIO.parse(file_path, "fasta")]
    return sequences

def prepare_input(set_name, batch_size, set_dir="data", shuffle=True):
    """Loads sequences and their enhancer activity, converting sequences to one-hot encoding."""
    
    # Load sequences from FASTA file
    file_seq = f"{set_dir}/Sequences_{set_name}.fa"
    sequences = load_fasta_sequences(file_seq)
    
    # Convert sequences to one-hot encoding
    seq_matrix = one_hot_encode_dna(sequences)
    print(f"{set_name} Sequence Matrix Shape: {seq_matrix.shape}")
    
    # Replace NaN values and reshape for model input
    X = np.nan_to_num(seq_matrix)
    
    # Load enhancer activity data
    activity_file = f"{set_dir}/Sequences_activity_{set_name}.txt"
    activity_data = pd.read_table(activity_file)
    
    Y_dev = activity_data.Dev_log2_enrichment.values
    Y_hk = activity_data.Hk_log2_enrichment.values
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_dev_tensor = torch.tensor(Y_dev, dtype=torch.float32)
    Y_hk_tensor = torch.tensor(Y_hk, dtype=torch.float32)
    
    print(f"Loaded {set_name} data.")
    
    dataset = DNADataset(X_tensor, Y_dev_tensor, Y_hk_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def load_model(model_path, params):
    model = DeepSTARR(params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


def load_keras_model(weights_path):
    keras_weights = {}
    with h5py.File(weights_path, 'r') as f:
        for layer in f.keys():
            keras_weights[layer] = {param: np.array(f[layer][param]) for param in f[layer].keys()}

    model = DeepSTARR(PARAMS)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor(keras_weights['conv1']['kernel']).permute(2, 1, 0))  # Keras: (H, W, C_out, C_in) → PyTorch: (C_out, C_in, H, W)
        model.conv1.bias.copy_(torch.tensor(keras_weights['conv1']['bias']))

        model.bn1.weight.copy_(torch.tensor(keras_weights['batch_normalization']['gamma']))
        model.bn1.bias.copy_(torch.tensor(keras_weights['batch_normalization']['beta']))
        model.bn1.running_mean.copy_(torch.tensor(keras_weights['batch_normalization']['moving_mean']))
        model.bn1.running_var.copy_(torch.tensor(keras_weights['batch_normalization']['moving_variance']))

        model.fc1.weight.copy_(torch.tensor(keras_weights['dense']['kernel']).T)
        model.fc1.bias.copy_(torch.tensor(keras_weights['dense']['bias']))

        model.fc2.weight.copy_(torch.tensor(keras_weights['dense_1']['kernel']).T)
        model.fc2.bias.copy_(torch.tensor(keras_weights['dense_1']['bias']))
    
    return model


