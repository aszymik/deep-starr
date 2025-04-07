# from keras.layers.convolutional import Conv1D, MaxPooling1D
# from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
# from keras.layers import BatchNormalization, InputLayer, Input
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, History

from helper import IOHelper, SequenceHelper

MODEL_ID = 'models/DeepSTARR.model'
SET = 'Test'
set_path = f'data/Sequences_{SET}.fa'
out_path = f'outputs/Keras_predictions_{SET}.txt'

# Load sequences
print("\nLoading sequences ...\n")
input_fasta = IOHelper.get_fastas_from_file(set_path, uppercase=True)
print(input_fasta.shape)

# length of first sequence
sequence_length = len(input_fasta.sequence.iloc[0])

# Convert sequence to one hot encoding matrix
seq_matrix = SequenceHelper.do_one_hot_encoding(input_fasta.sequence, sequence_length,
                                                SequenceHelper.parse_alpha_to_seq)

### load model
def load_model(model_path):
    from keras.models import model_from_json
    keras_model_weights = model_path + '.h5'
    keras_model_json = model_path + '.json'
    keras_model = model_from_json(open(keras_model_json).read())
    keras_model.load_weights(keras_model_weights)
    #keras_model.summary()
    return keras_model, keras_model_weights, keras_model_json

keras_model, keras_model_weights, keras_model_json = load_model(MODEL_ID)

### predict dev and hk activity
print("\nPredicting ...\n")
pred=keras_model.predict(seq_matrix)
out_prediction = input_fasta
out_prediction['Predictions_dev'] = pred[0]
out_prediction['Predictions_hk'] = pred[1]

### save file
print("\nSaving file ...\n")
out_prediction.to_csv(out_path, sep='\t', index=False)