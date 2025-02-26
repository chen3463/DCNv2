import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank=10):
        """
        Low-rank decomposed cross layer in DCN v2.
        :param input_dim: Feature dimension d
        :param rank: Rank for decomposition (r << d)
        """
        super(LowRankCrossLayer, self).__init__()
        self.W_q = nn.Linear(input_dim, rank, bias=False)  # Projection to rank r
        self.W_r = nn.Linear(rank, input_dim, bias=False)  # Projection back to d
        self.bias = nn.Parameter(torch.zeros(input_dim))   # Bias term

    def forward(self, x):
        """
        Forward pass for low-rank cross layer.
        :param x: Input tensor of shape (batch_size, input_dim)
        """
        interaction = self.W_r(self.W_q(x))  # Low-rank transformation
        return x + interaction + self.bias   # Residual connection

class LowRankCrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2, rank=10):
        """
        Multi-layer cross network using low-rank decomposition.
        :param input_dim: Feature dimension
        :param num_layers: Number of cross layers
        :param rank: Rank for decomposition
        """
        super(LowRankCrossNetwork, self).__init__()
        self.cross_layers = nn.ModuleList([
            LowRankCrossLayer(input_dim, rank) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.cross_layers:
            x = layer(x)
        return x

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        """
        Fully connected MLP network.
        :param input_dim: Input dimension
        :param hidden_dims: List of hidden layer sizes
        :param dropout: Dropout rate
        """
        super(DeepNetwork, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim  # Update for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


import torch
import torch.nn as nn

class DCNv2(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, mlp_dims=[128, 64, 32]):
        super(DCNv2, self).__init__()

        # Low-Rank Cross Network
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_cross_layers)
        ])
        self.cross_ranks = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_cross_layers)
        ])

        # Deep Network (MLP) with BatchNorm
        mlp_layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))  # BatchNorm added
            mlp_layers.append(nn.ReLU())
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final Fully Connected Layer
        self.final_layer = nn.Linear(input_dim + mlp_dims[-1], 1)

    def forward(self, x):
        x_0 = x.clone()

        # Cross Network (Low-Rank)
        for W_q, W_r in zip(self.cross_layers, self.cross_ranks):
            x = x_0 + W_r(W_q(x))

        # Deep Network (MLP)
        deep_out = self.mlp(x_0)

        # Concatenate Cross and Deep outputs
        combined = torch.cat([x, deep_out], dim=1)

        # Final Prediction
        out = self.final_layer(combined)
        return torch.sigmoid(out)  # Assuming binary classification

