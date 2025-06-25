from pred_new_sequence import *
from models import *

# PARAMS = {
#     'batch_size': 128,
#     'epochs': 50,
#     'lr': 0.002,
#     'early_stop': 6,
#     'n_conv_layer': 4,
#     'kernel_size1': 5,
#     'num_filters1': 256,
#     'kernel_size2': 5,
#     'num_filters2': 120,
#     'kernel_size3': 3,
#     'num_filters3': 30,
#     'kernel_size4': 7,
#     'num_filters4': 90,
#     'n_add_layer': 1,
#     'dense_neurons1': 256,
#     'dropout_prob': 0.5
#     }

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

def load_flex_model(model_path, params):
    model = DeepSTARRFlex(params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

if __name__ == '__main__':

    print('Loading sequences...')
    set_name = 'Test'
    sequences = load_fasta_sequences(f'data/deep-starr/Sequences_{set_name}.fa')
    seeds = [2211, 7530, 9982, 7653, 4949, 3008, 1234]
    
    for seed in seeds:
        print('Loading model...')
        # model = load_flex_model(f'models/deep-starr-flex/DeepSTARR_{seed}.model', PARAMS)
        model = load_flex_model(f'models/deep-starr-flex/DeepSTARR_{seed}_second.model', PARAMS)

        print('Predicting...')
        pred_dev, pred_hk = predict(model, set_name)  # ta funkcja do zmiany
        out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_dev': pred_dev, 'Predictions_hk': pred_hk})
        
        out_filename = f'outputs/deep-starr-flex/Pred_second_model_{seed}_{set_name}.txt'
        out_prediction.to_csv(out_filename, sep='\t', index=False)
        print(f'\nPredictions saved to {out_filename}\n')