import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import defaultdict
import optuna

# 🔹 DCNv2 Model Definition
class DCNv2(nn.Module):
    def __init__(self, num_numerical, num_categorical, num_one_hot, embedding_sizes, cross_layers, deep_layers):
        super(DCNv2, self).__init__()
        
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories + 1, emb_size) for num_categories, emb_size in embedding_sizes])
        self.cross_network = nn.ModuleList([nn.Linear(num_numerical + sum([emb_size for _, emb_size in embedding_sizes]) + num_one_hot, num_numerical) for _ in range(cross_layers)])
        
        deep_input_size = num_numerical + sum([emb_size for _, emb_size in embedding_sizes]) + num_one_hot
        deep_layers_list = []
        for units in deep_layers:
            deep_layers_list.append(nn.Linear(deep_input_size, units))
            deep_layers_list.append(nn.ReLU())
            deep_input_size = units
        self.deep_network = nn.Sequential(*deep_layers_list)
        
        self.output_layer = nn.Linear(deep_input_size, 1)

    def forward(self, num_features, cat_features, one_hot_features):
        embedded = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([num_features, embedded, one_hot_features], dim=1)
        
        for layer in self.cross_network:
            x = x + layer(x)
        deep_out = self.deep_network(x)
        output = self.output_layer(deep_out)
        return torch.sigmoid(output).squeeze()


