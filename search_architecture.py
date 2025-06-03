import wandb
from train import *
from models import *
from utils import *

wandb.login()

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'valid_pcc_dev',
        'goal': 'maximize'
    }
}

parameters_dict = {
    'batch_size': {
        'values': [64, 128]
    },
    'epochs': {
        'values': [50, 100]
    },
    'lr': {
        'values': [1e-4, 2e-4, 1e-3, 2e-3]
    },
    'early_stop': {
        'values': [4, 6, 8, 10]
    },
    'n_conv_layer': {
        'values': [3, 4]
    },
    'kernel_size1': {
        'values': [5, 7, 9, 11]
    },
    'num_filters1': {
        'values': [128, 256]
    },
    'kernel_size2': {
        'values': [3, 5, 7]
    },
    'num_filters2': {
        'values': [30, 60, 90, 120]
    },
    'kernel_size3': {
        'values': [3, 5, 7]
    },
    'num_filters3': {
        'values': [30, 60, 90, 120]
    },
    'kernel_size4': {
        'values': [3, 5, 7]
    },
    'num_filters4': {
        'values': [30, 60, 90, 120]
    },
    'n_add_layer': {
        'values': [1, 2, 3]
    },
    'dense_neurons1': {
        'values': [64, 128, 256]
    },
    'dense_neurons2': {
        'values': [64, 128, 256]
    },
    'dense_neurons3': {
        'values': [64, 128, 256]
    },
    'dropout_prob': {
        'values': [0.4, 0.5, 0.6]
    },
    }

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='DeepSTARR')


def wandb_train(seed=1234):
    # Initialize W&B run
    with wandb.init() as run:
        config = wandb.config

        train_loader = prepare_input('Train', config.batch_size)
        val_loader = prepare_input('Val', config.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepSTARRFlex(config)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-7)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        for epoch in range(config.epochs):
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

            # Log metrics to W&B
            wandb.log({
                'train_loss': avg_train_loss,
                'valid_loss': avg_val_loss,
                'train_mse_dev': mse_dev_train,
                'train_pcc_dev': pcc_dev_train,
                'train_scc_dev': scc_dev_train,
                'train_mse_hk': mse_hk_train,
                'train_pcc_hk': pcc_hk_train,
                'train_scc_hk': scc_hk_train,
                'valid_mse_dev': mse_dev_val,
                'valid_pcc_dev': pcc_dev_val,
                'valid_scc_dev': scc_dev_val,
                'valid_mse_hk': mse_hk_val,
                'valid_pcc_hk': pcc_hk_val,
                'valid_scc_hk': scc_hk_val,
            })
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.early_stop:
                    print('Early stopping triggered.')
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)


if __name__ == '__main__':
    wandb.agent(sweep_id, function=wandb_train)