import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import torch
import numpy as np

from scipy.stats import spearmanr, pearsonr

from models import *
from utils import *

def train(model, train_loader, val_loader, params, log_file='train_logs/training_log.csv', seed=1234):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-7)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train MSE Dev', 'Train MSE Hk', 'Train PCC Dev', 'Train PCC Hk', 'Train SCC Dev', 'Train SCC Hk', 'Val Loss', 'Val MSE Dev', 'Val MSE Hk', 'Val PCC Dev', 'Val PCC Hk', 'Val SCC Dev', 'Val SCC Hk'])
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
        print(f'Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')
        print('Train stats:')
        print(f'MSE Dev: {mse_dev_train:.2f}, PCC Dev: {pcc_dev_train:.2f}, SCC Dev: {scc_dev_train:.2f}')
        print(f'MSE Hk: {mse_hk_train:.2f}, PCC Hk: {pcc_hk_train:.2f}, SCC Hk: {scc_hk_train:.2f}')
        print('Validation stats:')
        print(f'MSE Dev: {mse_dev_val:.2f}, PCC Dev: {pcc_dev_val:.2f}, SCC Dev: {scc_dev_val:.2f}')
        print(f'MSE Hk: {mse_hk_val:.2f}, PCC Hk: {pcc_hk_val:.2f}, SCC Hk: {scc_hk_val:.2f}')
        
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, mse_dev_train, mse_hk_train, pcc_dev_train, pcc_hk_train, scc_dev_train, scc_hk_train,
                            avg_val_loss, mse_dev_val, mse_hk_val, pcc_dev_val, pcc_hk_val, scc_dev_val, scc_hk_val])
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params['early_stop']:
                print('Early stopping triggered.')
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate(pred_dev, pred_hk, Y_dev, Y_hk):
    
    mse_dev = F.mse_loss(pred_dev.squeeze(), Y_dev).item()
    mse_hk = F.mse_loss(pred_hk.squeeze(), Y_hk).item()
    pcc_dev = pearsonr(Y_dev.cpu().numpy(), pred_dev.cpu().detach().numpy().squeeze())[0]
    pcc_hk = pearsonr(Y_hk.cpu().numpy(), pred_hk.cpu().detach().numpy().squeeze())[0]
    scc_dev = spearmanr(Y_dev.cpu().numpy(), pred_dev.cpu().detach().numpy().squeeze())[0]
    scc_hk = spearmanr(Y_hk.cpu().numpy(), pred_hk.cpu().detach().numpy().squeeze())[0]

    return mse_dev, mse_hk, pcc_dev, pcc_hk, scc_dev, scc_hk


if __name__ == '__main__':
    # seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    seeds = [2137]
    # set_dir = 'data/lenti-mpra/da_library/preprocessed'
    set_dir = 'data/lenti-mpra/da_library/split_as_in_paper'
    activity_cols = ['Primary_log2_enrichment', 'Organoid_log2_enrichment']

    train_loader = prepare_input(set_name='Train', 
                                 batch_size=PARAMS['batch_size'],
                                 set_dir=set_dir,
                                 activity_cols=activity_cols
                                 )
    val_loader = prepare_input(set_name='Val', 
                               batch_size=PARAMS['batch_size'],
                               set_dir=set_dir,
                               activity_cols=activity_cols
                               )
    
    model = DeepSTARR(PARAMS)
    for seed in seeds:
        # log_file = f'train_logs/lenti-mpra/training_log_{seed}.csv'
        log_file = f'train_logs/lenti-mpra/split_as_in_paper/training_log_{seed}.csv'
        trained_model = train(model, train_loader, val_loader, PARAMS, log_file, seed)
        # torch.save(model.state_dict(), f'models/lenti-mpra/DeepSTARR_lenti-mpra_{seed}.model')
        torch.save(model.state_dict(), f'models/lenti-mpra/split_as_in_paper/DeepSTARR_lenti-mpra_{seed}.model')
        