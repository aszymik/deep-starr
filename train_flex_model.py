from train import *
from models import *
from utils import *


PARAMS = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.002,
    'early_stop': 10,
    'n_conv_layer': 4,
    'kernel_size1': 7,
    'num_filters1': 128,
    'kernel_size2': 3,
    'num_filters2': 60,
    'kernel_size3': 3,
    'num_filters3': 60,
    'kernel_size4': 7,
    'num_filters4': 30,
    'n_add_layer': 1,
    'dense_neurons1': 256,
    'dropout_prob': 0.4
    }

if __name__ == '__main__':
    seed = 7898
    
    train_loader = prepare_input('Train', PARAMS['batch_size'])
    val_loader = prepare_input('Val', PARAMS['batch_size'])
    
    model = DeepSTARRFlex(PARAMS)
    for seed in seeds:
        log_file = f'train_logs/deep-starr-flex/training_log_{seed}.csv'
        trained_model = train(model, train_loader, val_loader, PARAMS, log_file, seed)
        torch.save(model.state_dict(), f'models/deep-starr-flex/DeepSTARRFlex_model.model')
    