from train import *
from models import *
from utils import *

PARAMS = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.002,
    'early_stop': 6,
    'n_conv_layer': 4,
    'kernel_size1': 5,
    'num_filters1': 256,
    'kernel_size2': 5,
    'num_filters2': 120,
    'kernel_size3': 3,
    'num_filters3': 30,
    'kernel_size4': 7,
    'num_filters4': 90,
    'n_add_layer': 1,
    'dense_neurons1': 256,
    'dropout_prob': 0.5
    }

if __name__ == '__main__':
    seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    
    train_loader = prepare_input('Train', PARAMS['batch_size'])
    val_loader = prepare_input('Val', PARAMS['batch_size'])
    
    model = DeepSTARRFlex(PARAMS)
    for seed in seeds:
        log_file = f'train_logs/deep-starr-flex/training_log_{seed}.csv'
        trained_model = train(model, train_loader, val_loader, PARAMS, log_file, seed)
        torch.save(model.state_dict(), f'models/deep-starr-flex/DeepSTARR_{seed}.model')
    

    # # set_dir = 'data/lenti-mpra/da_library/preprocessed'
    # set_dir = 'data/lenti-mpra/da_library/split_as_in_paper'
    # activity_cols = ['Primary_log2_enrichment', 'Organoid_log2_enrichment']

    # train_loader = prepare_input(set_name='Train', 
    #                              batch_size=PARAMS['batch_size'],
    #                              set_dir=set_dir,
    #                              activity_cols=activity_cols
    #                              )
    # val_loader = prepare_input(set_name='Val', 
    #                            batch_size=PARAMS['batch_size'],
    #                            set_dir=set_dir,
    #                            activity_cols=activity_cols
    #                            )
    
    # model = DeepSTARR(PARAMS)
    # for seed in seeds:
    #     # log_file = f'train_logs/lenti-mpra/training_log_{seed}.csv'
    #     log_file = f'train_logs/lenti-mpra/split_as_in_paper/training_log_{seed}.csv'
    #     trained_model = train(model, train_loader, val_loader, PARAMS, log_file, seed)
    #     # torch.save(model.state_dict(), f'models/lenti-mpra/DeepSTARR_lenti-mpra_{seed}.model')
    #     torch.save(model.state_dict(), f'models/lenti-mpra/split_as_in_paper/DeepSTARR_lenti-mpra_{seed}.model')