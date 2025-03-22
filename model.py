import torch
import torch.nn as nn

class CrossLayerV2(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        return x0 * self.V(self.U(xl)) + self.bias + xl


class DCNv2(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim, cross_layers, cross_rank, deep_layers, onehot_size):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim) for card in cat_cardinalities
        ])
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim + onehot_size
        self.cross_net = nn.ModuleList([CrossLayerV2(input_dim, cross_rank) for _ in range(cross_layers)])
        deep = []
        dims = [input_dim] + deep_layers
        for i in range(len(deep_layers)):
            deep += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.deep_net = nn.Sequential(*deep)
        self.output_layer = nn.Linear(input_dim + deep_layers[-1], 1)

    def forward(self, numerical, categorical_emb, categorical_onehot):
        cat_embeds = [emb(categorical_emb[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        x = torch.cat([numerical, cat_embeds, categorical_onehot], dim=1)
        x0, xl = x, x
        for layer in self.cross_net:
            xl = layer(x0, xl)
        deep_x = self.deep_net(x)
        combined = torch.cat([xl, deep_x], dim=1)
        return torch.sigmoid(self.output_layer(combined)).squeeze(1)
