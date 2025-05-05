import torch.nn as nn
import torch.nn.functional as F


class DeepSTARR(nn.Module):
    def __init__(self, params):
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
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the way Keras does
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk