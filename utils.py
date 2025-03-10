import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import csv
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import OneHotEncoder

from datasets import DNADataset


def one_hot_encode_dna(sequences):
    """Converts a list of DNA sequences to a one-hot encoded matrix using sklearn's OneHotEncoder."""

    encoder = OneHotEncoder(categories=[['A', 'C', 'G', 'T']], dtype=np.float32, handle_unknown='ignore')
    sequences_array = np.array([list(seq) for seq in sequences])  # convert to 2D array
    one_hot_encoded = encoder.fit_transform(sequences_array).toarray()
    one_hot_encoded = one_hot_encoded.reshape(len(sequences), -1, 4)

    return one_hot_encoded

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


def train(model, train_loader, val_loader, params, log_file="training_log.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train MSE Dev", "Train MSE Hk", "Train PCC Dev", "Train PCC Hk", "Train SCC Dev", "Train SCC Hk", "Val Loss", "Val MSE Dev", "Val MSE Hk", "Val PCC Dev", "Val PCC Hk", "Val SCC Dev", "Val SCC Hk"])
    
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        mse_dev_train, mse_hk_train, pcc_dev_train, pcc_hk_train, scc_dev_train, scc_hk_train = 0, 0, 0, 0, 0, 0
        
        for X_batch, Y_dev_batch, Y_hk_batch in train_loader:
            X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
            optimizer.zero_grad()
            pred_dev, pred_hk = model(X_batch)
            loss = criterion(pred_dev.squeeze(), Y_dev_batch) + criterion(pred_hk.squeeze(), Y_hk_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Train statistics
            mse_dev_b, mse_hk_b, pcc_dev_b, pcc_hk_b, scc_dev_b, scc_hk_b = evaluate(pred_dev, pred_hk, Y_dev_batch, Y_hk_batch)
            mse_dev_train += mse_dev_b
            mse_hk_train += mse_hk_b
            pcc_dev_train += pcc_dev_b
            pcc_hk_train += pcc_hk_b
            scc_dev_train += scc_dev_b
            scc_hk_train += scc_hk_b
            
        avg_train_loss = total_loss / len(train_loader)
        mse_dev_train /= len(train_loader)
        mse_hk_train /= len(train_loader)
        pcc_dev_train /= len(train_loader)
        pcc_hk_train /= len(train_loader)
        scc_dev_train /= len(train_loader)
        scc_hk_train /= len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        mse_dev_val, mse_hk_val, pcc_dev_val, pcc_hk_val, scc_dev_val, scc_hk_val = 0, 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for X_batch, Y_dev_batch, Y_hk_batch in val_loader:
                X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
                pred_dev_batch, pred_hk_batch = model(X_batch)
                val_loss = criterion(pred_dev_batch.squeeze(), Y_dev_batch) + criterion(pred_hk_batch.squeeze(), Y_hk_batch)
                total_val_loss += val_loss.item()
                
                mse_dev_b, mse_hk_b, pcc_dev_b, pcc_hk_b, scc_dev_b, scc_hk_b = evaluate(pred_dev_batch, pred_hk_batch, Y_dev_batch, Y_hk_batch)
                mse_dev_val += mse_dev_b
                mse_hk_val += mse_hk_b
                pcc_dev_val += pcc_dev_b
                pcc_hk_val += pcc_hk_b
                scc_dev_val += scc_dev_b
                scc_hk_val += scc_hk_b
                
        avg_val_loss = total_val_loss / len(val_loader)
        mse_dev_val /= len(val_loader)
        mse_hk_val /= len(val_loader)
        pcc_dev_val /= len(val_loader)
        pcc_hk_val /= len(val_loader)
        scc_dev_val /= len(val_loader)
        scc_hk_val /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{params['epochs']}")
        print(f"Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        print("Train stats:")
        print(f"MSE Dev: {mse_dev_train:.2f}, PCC Dev: {pcc_dev_train:.2f}, SCC Dev: {scc_dev_train:.2f}")
        print(f"MSE Hk: {mse_hk_train:.2f}, PCC Hk: {pcc_hk_train:.2f}, SCC Hk: {scc_hk_train:.2f}")
        print("Validation stats:")
        print(f"MSE Dev: {mse_dev_val:.2f}, PCC Dev: {pcc_dev_val:.2f}, SCC Dev: {scc_dev_val:.2f}")
        print(f"MSE Hk: {mse_hk_val:.2f}, PCC Hk: {pcc_hk_val:.2f}, SCC Hk: {scc_hk_val:.2f}")
        
        # Logging
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, mse_dev_train, mse_hk_train, pcc_dev_train, pcc_hk_train, scc_dev_train, scc_hk_train,
                             avg_val_loss, mse_dev_val, mse_hk_val, pcc_dev_val, pcc_hk_val, scc_dev_val, scc_hk_val])
    
    return model


def evaluate(pred_dev, pred_hk, Y_dev, Y_hk):
    
    mse_dev = F.mse_loss(pred_dev.squeeze(), Y_dev).item()
    mse_hk = F.mse_loss(pred_hk.squeeze(), Y_hk).item()
    pcc_dev = pearsonr(Y_dev.cpu().numpy(), pred_dev.cpu().numpy().squeeze())[0]
    pcc_hk = pearsonr(Y_hk.cpu().numpy(), pred_hk.cpu().numpy().squeeze())[0]
    scc_dev = spearmanr(Y_dev.cpu().numpy(), pred_dev.cpu().numpy().squeeze())[0]
    scc_hk = spearmanr(Y_hk.cpu().numpy(), pred_hk.cpu().numpy().squeeze())[0]

    return mse_dev, mse_hk, pcc_dev, pcc_hk, scc_dev, scc_hk


def save(model, model_name):
    """
    Save model weights to file.
    """
    model_json = model.to_json()
    with open('model_' + model_name + '.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model_' + model_name + '.h5')