import torch.nn as nn
import torch.nn.functional as F


class DeepSTARR(nn.Module):
    def __init__(self, params, permute_before_flatten=False):
        super(DeepSTARR, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=params['num_filters'],
                               kernel_size=params['kernel_size1'], padding=params['pad'])
        self.bn1 = nn.BatchNorm1d(params['num_filters'], eps=1e-3, momentum=0.01)
  
        self.conv2 = nn.Conv1d(in_channels=params['num_filters'], out_channels=params['num_filters2'],
                               kernel_size=params['kernel_size2'], padding=params['pad'])
        self.bn2 = nn.BatchNorm1d(params['num_filters2'], eps=1e-3, momentum=0.01)
        
        self.conv3 = nn.Conv1d(in_channels=params['num_filters2'], out_channels=params['num_filters3'],
                               kernel_size=params['kernel_size3'], padding=params['pad'])
        self.bn3 = nn.BatchNorm1d(params['num_filters3'], eps=1e-3, momentum=0.01)
        
        self.conv4 = nn.Conv1d(in_channels=params['num_filters3'], out_channels=params['num_filters4'],
                               kernel_size=params['kernel_size4'], padding=params['pad'])
        self.bn4 = nn.BatchNorm1d(params['num_filters4'], eps=1e-3, momentum=0.01)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(120 * (249 // (2**4)), params['dense_neurons1'])
        self.bn_fc1 = nn.BatchNorm1d(params['dense_neurons1'], eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(params['dense_neurons1'], params['dense_neurons2'])
        self.bn_fc2 = nn.BatchNorm1d(params['dense_neurons2'], eps=1e-3, momentum=0.01)
        
        # Heads per task (developmental and housekeeping enhancer activities)
        self.fc_dev = nn.Linear(params['dense_neurons2'], 1)
        self.fc_hk = nn.Linear(params['dense_neurons2'], 1)
        
        self.dropout = nn.Dropout(params['dropout_prob'])
        self.permute_before_flatten = permute_before_flatten
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)  # flatten the way Keras does
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk
    

class DeepSTARRFlex(nn.Module):
    def __init__(self, params, permute_before_flatten=False):
        super(DeepSTARRFlex, self).__init__()

        self.permute_before_flatten = permute_before_flatten
        self.dropout_prob = params['dropout_prob']
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = 4
        current_length = params['input_length']  # e.g., 249

        for i in range(params['n_conv_layer']):
            out_channels = params[f'num_filters{i+1}']
            kernel_size = params[f'kernel_size{i+1}']
            padding = params['pad']

            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            self.bn_layers.append(
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            )

            in_channels = out_channels
            current_length = current_length // 2  # due to MaxPool1d with kernel_size=2

        self.flattened_size = current_length * in_channels

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bns = nn.ModuleList()

        fc_in = self.flattened_size
        for i in range(params['n_add_layer']):
            fc_out = params[f'dense_neurons{i+1}']
            self.fc_layers.append(nn.Linear(fc_in, fc_out))
            self.fc_bns.append(nn.BatchNorm1d(fc_out, eps=1e-3, momentum=0.01))
            fc_in = fc_out

        # Output heads
        self.fc_dev = nn.Linear(fc_in, 1)
        self.fc_hk = nn.Linear(fc_in, 1)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(F.relu(bn(conv(x))))

        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)

        x = x.view(x.size(0), -1)

        for fc, bn in zip(self.fc_layers, self.fc_bns):
            x = self.dropout(F.relu(bn(fc(x))))

        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)

        return out_dev, out_hk
