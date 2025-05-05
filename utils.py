import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

from datasets import DNADataset
from models import *

import h5py
import numpy as np


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
    try:
        encoder = OneHotEncoder(sparse_output=False, categories=[categories], handle_unknown='ignore', dtype=np.float32)
    except TypeError:
        encoder = OneHotEncoder(sparse=False, categories=[categories], handle_unknown='ignore', dtype=np.float32)
    encoder.fit(categories.reshape(-1, 1))

    one_hot_encoded = np.array([encoder.transform(np.array(list(seq)).reshape(-1, 1)) for seq in sequences])
    return one_hot_encoded.reshape(len(sequences), 4, -1)

def load_fasta_sequences(file_path):
    """Reads sequences from a FASTA file and returns a list of sequences."""
    sequences = [str(record.seq).upper() for record in SeqIO.parse(file_path, 'fasta')]
    return sequences

def prepare_input(set_name, batch_size, set_dir='data', shuffle=True):
    """Loads sequences and their enhancer activity, converting sequences to one-hot encoding."""
    
    # Load sequences from FASTA file
    file_seq = f'{set_dir}/Sequences_{set_name}.fa'
    sequences = load_fasta_sequences(file_seq)
    
    # Convert sequences to one-hot encoding
    seq_matrix = one_hot_encode_dna(sequences)
    print(f'{set_name} Sequence Matrix Shape: {seq_matrix.shape}')
    
    # Replace NaN values and reshape for model input
    X = np.nan_to_num(seq_matrix)
    
    # Load enhancer activity data
    activity_file = f'{set_dir}/Sequences_activity_{set_name}.txt'
    activity_data = pd.read_table(activity_file)
    
    Y_dev = activity_data.Dev_log2_enrichment.values
    Y_hk = activity_data.Hk_log2_enrichment.values
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_dev_tensor = torch.tensor(Y_dev, dtype=torch.float32)
    Y_hk_tensor = torch.tensor(Y_hk, dtype=torch.float32)
    
    print(f'Loaded {set_name} data.')
    
    dataset = DNADataset(X_tensor, Y_dev_tensor, Y_hk_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def load_model(model_path, params):
    model = DeepSTARR(params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def smart_load_keras_weights(h5_path):
    keras_weights = {}
    
    with h5py.File(h5_path, 'r') as model_weights:
        
        for layer_name in model_weights:
            layer_group = model_weights[layer_name]
            try:
                sub_group = layer_group[layer_name]  # Usually repeated name
            except KeyError:
                sub_group = layer_group
            
            weights = {}
            for weight_name in sub_group:
                data = sub_group[weight_name][()]
                # Remove the :0 suffix if you like
                clean_name = weight_name.split(':')[0]
                weights[clean_name] = data
            
            keras_weights[layer_name] = weights
    
    return keras_weights

def smart_load_keras_model(weights_path):
    keras_weights = smart_load_keras_weights(weights_path)

    model = DeepSTARR(PARAMS)
    state_dict = {}

    with torch.no_grad():
        # Conv layers
        for i in range(4):
            conv_name = 'Conv1D_1st' if i == 0 else f'Conv1D_{i+1}'
            bn_name = f'batch_normalization_{i+1}'

            state_dict[f'conv{i+1}.weight'] = torch.tensor(keras_weights[conv_name]['kernel']).permute(2, 1, 0)
            state_dict[f'conv{i+1}.bias'] = torch.tensor(keras_weights[conv_name]['bias'])

            state_dict[f'bn{i+1}.weight'] = torch.tensor(keras_weights[bn_name]['gamma'])
            state_dict[f'bn{i+1}.bias'] = torch.tensor(keras_weights[bn_name]['beta'])
            state_dict[f'bn{i+1}.running_mean'] = torch.tensor(keras_weights[bn_name]['moving_mean'])
            state_dict[f'bn{i+1}.running_var'] = torch.tensor(keras_weights[bn_name]['moving_variance'])

        # Fully connected layers
        for i in range(1, 3):
            bn_name = f'batch_normalization_{i+4}'

            state_dict[f'fc{i}.weight'] = torch.tensor(keras_weights[f'Dense_{i}']['kernel']).T
            state_dict[f'fc{i}.bias'] = torch.tensor(keras_weights[f'Dense_{i}']['bias'])

            state_dict[f'bn_fc{i}.weight'] = torch.tensor(keras_weights[bn_name]['gamma'])
            state_dict[f'bn_fc{i}.bias'] = torch.tensor(keras_weights[bn_name]['beta'])
            state_dict[f'bn_fc{i}.running_mean'] = torch.tensor(keras_weights[bn_name]['moving_mean'])
            state_dict[f'bn_fc{i}.running_var'] = torch.tensor(keras_weights[bn_name]['moving_variance'])

        # Heads
        state_dict['fc_dev.weight'] = torch.tensor(keras_weights['Dense_Dev']['kernel']).T
        state_dict['fc_dev.bias'] = torch.tensor(keras_weights['Dense_Dev']['bias'])

        state_dict['fc_hk.weight'] = torch.tensor(keras_weights['Dense_Hk']['kernel']).T
        state_dict['fc_hk.bias'] = torch.tensor(keras_weights['Dense_Hk']['bias'])

    model.load_state_dict(state_dict)
    return model

def load_keras_weights(weights_path):
    keras_weights = {}
    with h5py.File(weights_path, 'r') as model_weights:

        for layer_name in model_weights.keys():
            layer_group = model_weights[layer_name]

            weights = []
            for weight_name in layer_group.keys():
                item = layer_group[weight_name]

                for sub_weight_name in item.keys():
                    weights.append(item[sub_weight_name][()])
            keras_weights[layer_name] = weights

    return keras_weights

