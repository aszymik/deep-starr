from models import *
from utils import *


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


train_loader = prepare_input("Train", PARAMS['batch_size'])
val_loader = prepare_input("Val", PARAMS['batch_size'])
test_loader = prepare_input("Test", PARAMS['batch_size'])

model = DeepSTARR(PARAMS)

trained_model = train(model, train_loader, val_loader, PARAMS)
save(trained_model, 'DeepSTARR')




