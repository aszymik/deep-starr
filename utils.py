import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

import keras
import keras.models as models

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

def load_keras_model(model_config_path, weights_path):
    # Load Keras architecture
    with open(model_config_path) as file:
        data = file.read()
    keras_model = models.model_from_json(data, custom_objects={'Model': keras.Model})

    # Load Keras weights
    keras_model.load_weights(weights_path)

    # Extract weights
    keras_weights = {}
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if weights:
            keras_weights[layer.name] = weights

    model = DeepSTARR(PARAMS)
    state_dict = {}

    with torch.no_grad():
        # Conv layers
        for i in range(4):
            conv_name = 'Conv1D_1st' if i == 0 else f'Conv1D_{i+1}'
            bn_name = f'batch_normalization_6{i}'

            # Convert weights
            state_dict[f'conv{i+1}.weight'] = torch.tensor(keras_weights[conv_name][0]).permute(2, 1, 0)
            state_dict[f'conv{i+1}.bias'] = torch.tensor(keras_weights[conv_name][1])

            state_dict[f'bn{i+1}.weight'] = torch.tensor(keras_weights[bn_name][0])
            state_dict[f'bn{i+1}.bias'] = torch.tensor(keras_weights[bn_name][1])
            state_dict[f'bn{i+1}.running_mean'] = torch.tensor(keras_weights[bn_name][2])
            state_dict[f'bn{i+1}.running_var'] = torch.tensor(keras_weights[bn_name][3])

        # Fully Connected (Dense) layers
        for i in range(1, 3):
            bn_name = f'batch_normalization_6{i+3}'

            state_dict[f'fc{i}.weight'] = torch.tensor(keras_weights[f'Dense_{i}'][0]).T
            state_dict[f'fc{i}.bias'] = torch.tensor(keras_weights[f'Dense_{i}'][1])

            state_dict[f'bn_fc{i}.weight'] = torch.tensor(keras_weights[bn_name][0])
            state_dict[f'bn_fc{i}.bias'] = torch.tensor(keras_weights[bn_name][1])
            state_dict[f'bn_fc{i}.running_mean'] = torch.tensor(keras_weights[bn_name][2])
            state_dict[f'bn_fc{i}.running_var'] = torch.tensor(keras_weights[bn_name][3])

        # Heads
        state_dict['fc_dev.weight'] = torch.tensor(keras_weights['Dense_Dev'][0]).T
        state_dict['fc_dev.bias'] = torch.tensor(keras_weights['Dense_Dev'][1])

        state_dict['fc_hk.weight'] = torch.tensor(keras_weights['Dense_Hk'][0]).T
        state_dict['fc_hk.bias'] = torch.tensor(keras_weights['Dense_Hk'][1])

    model.load_state_dict(state_dict)
    return model

"""
def load_keras_model(model_config_path, weights_path):

    with open(model_config_path) as file:
        data = file.read()
    keras_model = models.model_from_json(data, custom_objects={'Model': keras.Model})
    keras_model.load_weights(weights_path)

    keras_weights = {}
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if weights:
            keras_weights[layer.name] = weights

    model = DeepSTARR(PARAMS)
    with torch.no_grad():
        # Conv layers
        for i in range(4):
            conv_name = 'Conv1D_1st' if i == 0 else f'Conv1D_{i+1}'
            bn_name = f'batch_normalization_6{i}'

            # Pytorch layers
            conv_layer = getattr(model, f'conv{i+1}')
            bn_layer = getattr(model, f'bn{i+1}')

            # Load conv and batchnorm parameters
            conv_layer.weight.copy_(torch.tensor(keras_weights[conv_name][0]).permute(2, 1, 0))  # Keras: (H, W, C_out, C_in), PyTorch: (C_out, C_in, H, W)
            conv_layer.bias.copy_(torch.tensor(keras_weights[conv_name][1]))

            bn_layer.weight.copy_(torch.tensor(keras_weights[bn_name][0]))
            bn_layer.bias.copy_(torch.tensor(keras_weights[bn_name][1]))
            bn_layer.running_mean.copy_(torch.tensor(keras_weights[bn_name][2]))
            bn_layer.running_var.copy_(torch.tensor(keras_weights[bn_name][3]))

        # Fc layers
        for i in range(1, 3):
            bn_name = f'batch_normalization_6{i+3}'
            
            # Pytorch layers
            fc_layer = getattr(model, f'fc{i}')
            bn_layer = getattr(model, f'bn_fc{i}')

            fc_layer.weight.copy_(torch.tensor(keras_weights[f'Dense_{i}'][0]).T)
            fc_layer.bias.copy_(torch.tensor(keras_weights[f'Dense_{i}'][1]))

            bn_layer.weight.copy_(torch.tensor(keras_weights[bn_name][0]))
            bn_layer.bias.copy_(torch.tensor(keras_weights[bn_name][1]))
            bn_layer.running_mean.copy_(torch.tensor(keras_weights[bn_name][2]))
            bn_layer.running_var.copy_(torch.tensor(keras_weights[bn_name][3]))

        # Heads
        model.fc_dev.weight.copy_(torch.tensor(keras_weights['Dense_Dev'][0]).T)
        model.fc_dev.bias.copy_(torch.tensor(keras_weights['Dense_Dev'][1]))

        model.fc_hk.weight.copy_(torch.tensor(keras_weights['Dense_Hk'][0]).T)
        model.fc_hk.bias.copy_(torch.tensor(keras_weights['Dense_Hk'][1]))
    
    return model
"""
load_keras_model('outputs/DeepSTARR.model.json', 'outputs/DeepSTARR.model.h5')