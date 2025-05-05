import torch
import torch.nn.functional as F
import numpy as np

import sys
import json
from utils import *

model = load_keras_model('models/Model_DeepSTARR.h5')
print(model)
model.eval()

batch_size = 1
seq_len = 249
channels = 4

np.random.seed(42)
dummy_input = np.random.rand(batch_size, seq_len, channels).astype(np.float32)

# Pytorch expects (batch, channels, seq_len)
dummy_input = torch.tensor(dummy_input).permute(0, 2, 1)

outputs = {}

# Forward manually, saving each intermediate
with torch.no_grad():
    outputs['input'] = dummy_input

    # Conv 1
    x = model.conv1(dummy_input)
    outputs['conv1'] = x.cpu().numpy()
    x = model.bn1(x)
    outputs['bn1'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu1'] = x.cpu().numpy()
    x = model.pool(x)
    outputs['pool1'] = x.cpu().numpy()

    # Conv 2
    x = model.conv2(x)
    outputs['conv2'] = x.cpu().numpy()
    x = model.bn2(x)
    outputs['bn2'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu2'] = x.cpu().numpy()
    x = model.pool(x)
    outputs['pool2'] = x.cpu().numpy()

    # Conv 3
    x = model.conv3(x)
    outputs['conv3'] = x.cpu().numpy()
    x = model.bn3(x)
    outputs['bn3'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu3'] = x.cpu().numpy()
    x = model.pool(x)
    outputs['pool3'] = x.cpu().numpy()

    # Conv 4
    x = model.conv4(x)
    outputs['conv4'] = x.cpu().numpy()
    x = model.bn4(x)
    outputs['bn4'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu4'] = x.cpu().numpy()
    x = model.pool(x)
    outputs['pool4'] = x.cpu().numpy()

    # Flatten in the way Keras does
    print(x.shape)
    x = x.permute(0, 2, 1)
    print(x.shape)
    x_flat = x.reshape(x.shape[0], -1)
    print(x_flat.shape)
    outputs['flatten'] = x_flat.cpu().numpy()

    # Fc 1
    x = model.fc1(x_flat)
    outputs['fc1'] = x.cpu().numpy()
    x = model.bn_fc1(x)
    outputs['bn_fc1'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu_fc1'] = x.cpu().numpy()

    # Fc 2
    x = model.fc2(x)
    outputs['fc2'] = x.cpu().numpy()
    x = model.bn_fc2(x)
    outputs['bn_fc2'] = x.cpu().numpy()
    x = F.relu(x)
    outputs['relu_fc2'] = x.cpu().numpy()

    out_dev = model.fc_dev(x)
    out_hk = model.fc_hk(x)

    outputs['out_dev'] = out_dev.cpu().numpy()
    outputs['out_hk'] = out_hk.cpu().numpy()

# Save all outputs
np.savez('compare_keras_and_pytorch/pytorch_outputs.npz', **outputs)
torch.save(model.state_dict(), 'compare_keras_and_pytorch/pytorch_weights.pth')