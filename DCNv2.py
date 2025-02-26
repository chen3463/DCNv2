import torch.nn as nn

class DCNv2(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, emb_dim=16, num_cross_layers=3, rank=2, mlp_dims=[128, 64, 32]):
        super(DCNv2, self).__init__()

        # 🔹 Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories + 1, emb_dim) for num_categories in cat_cardinalities  # Handle unseen categories
        ])
        emb_output_dim = len(cat_cardinalities) * emb_dim

        # 🔹 Normalize Numerical Features
        self.num_layer_norm = nn.LayerNorm(num_numerical)  

        input_dim = num_numerical + emb_output_dim

        # 🔹 Low-rank Cross Network
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, rank, bias=False) for _ in range(num_cross_layers)
        ])
        self.cross_ranks = nn.ModuleList([
            nn.Linear(rank, input_dim, bias=False) for _ in range(num_cross_layers)
        ])

        # 🔹 Deep Network (MLP)
        mlp_layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))  
            mlp_layers.append(nn.ReLU())
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 🔹 Final Layer
        self.final_layer = nn.Linear(input_dim + mlp_dims[-1], 1)

    def forward(self, numerical_features, categorical_features):
        embedded_cats = [emb(categorical_features[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embedded_cats, dim=1) if embedded_cats else None
        num_out = self.num_layer_norm(numerical_features)  

        x_0 = torch.cat([num_out, cat_out], dim=1) if cat_out is not None else num_out

        x = x_0.clone()
        for W_q, W_r in zip(self.cross_layers, self.cross_ranks):
            x = x_0 + W_r(W_q(x))

        deep_out = self.mlp(x_0)
        combined = torch.cat([x, deep_out], dim=1)
        return torch.sigmoid(self.final_layer(combined))
