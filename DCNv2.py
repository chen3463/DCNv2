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
    """
    Deep & Cross Network v2 implementation with parallel deep and cross networks.
    The outputs of the deep network and cross network are combined before the output layer.
    """

    def __init__(self, num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size):
        super(DCNv2, self).__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(cat_card + 1, embedding_dim) for cat_card in cat_cardinalities])

        # Input dimension for both networks
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim + onehot_size

        # Cross network layers (to capture interactions between features)
        self.cross_net = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(cross_layers)])

        # Deep network layers (fully connected network for high-level abstractions)
        self.deep_net = nn.Sequential(
            *[nn.Linear(input_dim, deep_layers[0]), nn.ReLU()] +
             sum([[nn.Linear(deep_layers[i], deep_layers[i + 1]), nn.ReLU()] for i in range(len(deep_layers) - 1)], [])
        )

        # Output layer
        self.output_layer = nn.Linear(deep_layers[-1] + input_dim, 1)

    def forward(self, numerical, categorical_emb, categorical_onehot):
        # Embedding for categorical features
        cat_embeds = [emb(categorical_emb[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)

        # Concatenate all input features
        x = torch.cat([numerical, cat_embeds, categorical_onehot], dim=1)

        # Cross network processing
        cross_x = x
        for layer in self.cross_net:
            cross_x = cross_x + layer(cross_x)

        # Deep network processing
        deep_x = self.deep_net(x)

        # Concatenate the outputs from the deep and cross networks
        combined_x = torch.cat([cross_x, deep_x], dim=1)

        # Output layer
        output = torch.sigmoid(self.output_layer(combined_x)).squeeze(1)

        return output


