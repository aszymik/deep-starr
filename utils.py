import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

from datasets import DNADataset

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

def prepare_input(set_name, batch_size, set_dir="data"):
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def save(model, model_name):
    """
    Save model weights to file.
    """
    model_json = model.to_json()  # doesn't work
    with open('model_' + model_name + '.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model_' + model_name + '.h5')