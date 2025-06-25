import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

import keras
from keras import backend as K
from keras.models import model_from_json

# TF1.x session and inference mode
sess = tf.compat.v1.Session()
K.set_session(sess)
K.set_learning_phase(0)

MODEL_ID = 'models/Model_DeepSTARR'
with open(MODEL_ID + '.json', 'r') as f:
    keras_model = model_from_json(f.read())

# Freeze all layers
for layer in keras_model.layers:
    layer.trainable = False

keras_model.compile(optimizer='adam', loss='mse')
keras_model.load_weights(MODEL_ID + '.h5')

# Initialize any uninitialized vars
uninitialized_vars = [
    v for v in tf.compat.v1.global_variables()
    if not sess.run(tf.compat.v1.is_variable_initialized(v))
]
sess.run(tf.compat.v1.variables_initializer(uninitialized_vars))

# Create input
batch_size = 1
seq_len = 249
channels = 4

np.random.seed(42)
dummy_input = np.random.rand(batch_size, seq_len, channels).astype(np.float32)
dummy_input = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

# Forward through Keras
keras_layers = [layer for layer in keras_model.layers]
x_tf = dummy_input
outputs_tf = {}

for layer in keras_layers[:-2]:
    if not isinstance(layer, keras.layers.Dropout):
        x_tf = layer(x_tf)
        layer_name = layer.name
        outputs_tf[layer_name] = sess.run(x_tf)


dense_dev = keras_model.get_layer('Dense_Dev')
dev_out = dense_dev(x_tf)
outputs_tf['Dense_Dev'] = sess.run(dev_out)

dense_hk = keras_model.get_layer('Dense_Hk')
hk_out = dense_hk(x_tf)
outputs_tf['Dense_Hk'] = sess.run(hk_out)

# Compare weights
keras_weights_dict = {}
for layer in keras_model.layers:
    if layer.weights:
        for w in layer.weights:
            keras_weights_dict[f'{w.name[:-2]}'] = sess.run(w)

pt_weights = torch.load('compare_keras_and_pytorch/pytorch_weights.pth')
pt_weights_dict = {name: weights for name, weights in pt_weights.items() if not 'num_batches_tracked' in name}


print('\nWeights comparison:')
for layer_pt, layer_keras in zip(pt_weights_dict, keras_weights_dict):

    pt_weights = pt_weights_dict[layer_pt].detach().cpu().numpy()
    keras_weights = keras_weights_dict[layer_keras]

    # Transpose weights if needed
    if len(pt_weights.shape) > 1:
        try:
            pt_weights = np.transpose(pt_weights, (2, 1, 0))  # conv
        except ValueError:
            pt_weights = np.transpose(pt_weights)  # fc

    if pt_weights.shape != keras_weights.shape:
        print(f'Shape mismatch for {layer_keras}: PyTorch {pt_weights.shape}, Keras {keras_weights.shape}')
        continue
    diff = np.mean(np.abs(pt_weights - keras_weights))

    print(f'\n{layer_pt} / {layer_keras}')
    print(f'Mean absolute difference: {diff:.6e}')


# Compare outputs
outputs_pt = np.load('compare_keras_and_pytorch/pytorch_outputs.npz')
print('\nLayer-wise comparison:')

for i, (out_pt, out_tf) in enumerate(zip(outputs_pt, outputs_tf)):

    out_pt_np = outputs_pt[out_pt]
    out_tf_np = outputs_tf[out_tf]
    print(f'\nLayer {i} ({out_pt}/{out_tf}):')
    
    # Adjust shapes if necessary
    if out_pt_np.shape[1:] != out_tf_np.shape[1:]:
        out_pt_np = np.transpose(out_pt_np, (0, 2, 1))  # try to adjust shape

    if out_pt_np.shape[1:] != out_tf_np.shape[1:]:
        print(f'Shape mismatch! PyTorch {out_pt_np.shape} vs Keras {out_tf_np.shape}')
    else:
        diff = np.mean(np.abs(out_pt_np - out_tf_np))
        print(f'Mean absolute difference = {diff:.6f}')