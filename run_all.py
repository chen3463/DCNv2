import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import average_precision_score
from collections import defaultdict
import optuna
import shap

# 🔹 Load and Prepare Data
class DataLoaderWrapper:
    def __init__(self, train_path, valid_path, test_path, numerical_columns, categorical_columns, one_hot_columns):
        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.test_df = pd.read_csv(test_path)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.one_hot_columns = one_hot_columns
        self.encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._fit_transformers()

    def _fit_transformers(self):
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col].astype(str))
            le_classes = list(le.classes_) + ['<UNK>']
            le.classes_ = np.array(le_classes)
            
            self.valid_df[col] = self.train_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else le.transform(['<UNK>'])[0])
            self.test_df[col] = self.test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else le.transform(['<UNK>'])[0])
            self.encoders[col] = le

        self.train_df[self.numerical_columns] = self.scaler.fit_transform(self.train_df[self.numerical_columns])
        self.valid_df[self.numerical_columns] = self.scaler.transform(self.valid_df[self.numerical_columns])
        self.test_df[self.numerical_columns] = self.scaler.transform(self.test_df[self.numerical_columns])

        self.one_hot_enc.fit(self.train_df[self.one_hot_columns])

    def get_dataloader(self, df, batch_size=32):
        num_features = torch.tensor(df[self.numerical_columns].values, dtype=torch.float32)
        cat_features = torch.tensor(df[self.categorical_columns].values, dtype=torch.long)
        one_hot_features = torch.tensor(self.one_hot_enc.transform(df[self.one_hot_columns]), dtype=torch.float32)
        labels = torch.tensor(df['target'].values, dtype=torch.float32)
        dataset = TensorDataset(num_features, cat_features, one_hot_features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 🔹 Low-Rank Cross Layer
class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank):
        super(LowRankCrossLayer, self).__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        interaction = self.V(self.U(x))
        return x0 * interaction + x + self.bias

# 🔹 DCNv2 Model Definition
class DCNv2(nn.Module):
    def __init__(self, num_numerical, num_categorical, num_one_hot, embedding_sizes, rank, cross_layers, deep_layers):
        super(DCNv2, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(num_categories + 1, emb_size) for num_categories, emb_size in embedding_sizes])
        self.cross_layers = nn.ModuleList([LowRankCrossLayer(num_numerical + sum([emb_size for _, emb_size in embedding_sizes]) + num_one_hot, rank) for _ in range(cross_layers)])

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

        x_cross = x.clone()
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)

        deep_out = self.deep_network(x_cross)
        output = self.output_layer(deep_out)
        return torch.sigmoid(output).squeeze()

# 🔹 Model Training and Evaluation
def train_and_evaluate(batch_size, rank, deep_layers, cross_layers):
    model = DCNv2(num_numerical=len(data_loader.numerical_columns),
                  num_categorical=len(data_loader.categorical_columns),
                  num_one_hot=len(data_loader.one_hot_columns),
                  embedding_sizes=[(len(data_loader.encoders[col].classes_), min(50, (len(data_loader.encoders[col].classes_) + 1) // 2)) for col in data_loader.categorical_columns],
                  rank=rank, cross_layers=cross_layers, deep_layers=deep_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    train_loader = data_loader.get_dataloader(data_loader.train_df, batch_size)
    valid_loader = data_loader.get_dataloader(data_loader.valid_df, batch_size)
    test_loader = data_loader.get_dataloader(data_loader.test_df, batch_size)

    best_val_aucpr = 0
    for epoch in range(10):
        model.train()
        for num_features, cat_features, one_hot_features, labels in train_loader:
            num_features, cat_features, one_hot_features, labels = num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(num_features, cat_features, one_hot_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for num_features, cat_features, one_hot_features, labels in valid_loader:
                num_features, cat_features, one_hot_features, labels = num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
                outputs = model(num_features, cat_features, one_hot_features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_aucpr = average_precision_score(val_labels, val_preds)
        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            best_model = model.state_dict()
    
    return best_model

def evaluate_on_test(best_params):
    model = DCNv2(num_numerical=len(data_loader.numerical_columns),
                  num_categorical=len(data_loader.categorical_columns),
                  num_one_hot=len(data_loader.one_hot_columns),
                  embedding_sizes=[(len(data_loader.encoders[col].classes_), min(50, (len(data_loader.encoders[col].classes_) + 1) // 2)) for col in data_loader.categorical_columns],
                  rank=best_params['rank'],
                  cross_layers=best_params['cross_layers'],
                  deep_layers=best_params['deep_layers']).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loader = data_loader.get_dataloader(data_loader.test_df, best_params['batch_size'])
    test_preds, test_labels = [], []
    with torch.no_grad():
        for num_features, cat_features, one_hot_features, labels in test_loader:
            num_features, cat_features, one_hot_features, labels = num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
            outputs = model(num_features, cat_features, one_hot_features)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_aucpr = average_precision_score(test_labels, test_preds)
    print(f"Test AUCPR: {test_aucpr:.4f}")

# 🔹 Run Feature Selection and Hyperparameter Tuning
best_params = hyperparameter_tuning_with_feature_selection()
evaluate_on_test(best_params)
