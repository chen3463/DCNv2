import torch.nn as nn

class DCNv2(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, emb_dim=16, num_cross_layers=3, mlp_dims=[128, 64, 32]):
        super(DCNv2, self).__init__()

        # 🔹 1. Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories in cat_cardinalities
        ])
        emb_output_dim = len(cat_cardinalities) * emb_dim  # Total embedding output size

        # 🔹 2. Normalize Numerical Features
        self.num_layer_norm = nn.LayerNorm(num_numerical)  

        input_dim = num_numerical + emb_output_dim  # Final input size after processing

        # 🔹 3. Cross Network (Low-Rank)
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_cross_layers)
        ])
        self.cross_ranks = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_cross_layers)
        ])

        # 🔹 4. Deep Network (MLP) with BatchNorm
        mlp_layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))  # BatchNorm
            mlp_layers.append(nn.ReLU())
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 🔹 5. Final Fully Connected Layer
        self.final_layer = nn.Linear(input_dim + mlp_dims[-1], 1)

    def forward(self, numerical_features, categorical_features):
        # 🔹 1. Process Categorical Data (Embedding)
        embedded_cats = [emb(categorical_features[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embedded_cats, dim=1) if embedded_cats else None

        # 🔹 2. Normalize Numerical Data
        num_out = self.num_layer_norm(numerical_features)  

        # 🔹 3. Combine Inputs
        if cat_out is not None:
            x_0 = torch.cat([num_out, cat_out], dim=1)
        else:
            x_0 = num_out

        # 🔹 4. Cross Network
        x = x_0.clone()
        for W_q, W_r in zip(self.cross_layers, self.cross_ranks):
            x = x_0 + W_r(W_q(x))

        # 🔹 5. Deep Network (MLP)
        deep_out = self.mlp(x_0)

        # 🔹 6. Concatenate Cross and Deep outputs
        combined = torch.cat([x, deep_out], dim=1)

        # 🔹 7. Final Prediction
        out = self.final_layer(combined)
        return torch.sigmoid(out)
