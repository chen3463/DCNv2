import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import average_precision_score
import optuna
import shap

# 🚀 Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoaderWrapper:
    def __init__(self, train_path, valid_path, test_path, numerical_columns, categorical_columns, one_hot_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.one_hot_columns = one_hot_columns

        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.test_df = pd.read_csv(test_path)

        self.encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')

        self._fit_transformers()

    def _fit_transformers(self):
        # Custom Label Encoding with Unseen Category Handling
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col].astype(str))
            
            # Store the learned classes and append "unknown"
            le_classes = np.append(le.classes_, "unknown")
            le.classes_ = le_classes
            self.encoders[col] = le

            # Apply encoding with unknown handling
            self.valid_df[col] = self._transform_with_unknown(le, self.valid_df[col])
            self.test_df[col] = self._transform_with_unknown(le, self.test_df[col])

        # Standard Scaling for Numerical Features
        self.train_df[self.numerical_columns] = self.scaler.fit_transform(self.train_df[self.numerical_columns])
        self.valid_df[self.numerical_columns] = self.scaler.transform(self.valid_df[self.numerical_columns])
        self.test_df[self.numerical_columns] = self.scaler.transform(self.test_df[self.numerical_columns])

        # One-Hot Encoding with Unknown Handling
        self.one_hot_enc.fit(self.train_df[self.one_hot_columns])

    def _transform_with_unknown(self, le, series):
        """Encodes categories, assigning unseen values to 'unknown'."""
        return series.apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["unknown"])[0])

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
        return x0 * self.V(self.U(x)) + x + self.bias

# 🔹 DCNv2 Model
class DCNv2(nn.Module):
    def __init__(self, num_numerical, num_categorical, num_one_hot, embedding_sizes, rank, cross_layers, deep_layers):
        super(DCNv2, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(num_categories + 1, emb_size) for num_categories, emb_size in embedding_sizes])
        input_size = num_numerical + sum([emb_size for _, emb_size in embedding_sizes]) + num_one_hot
        
        self.cross_layers = nn.ModuleList([LowRankCrossLayer(input_size, rank) for _ in range(cross_layers)])
        
        deep_list = []
        for units in deep_layers:
            deep_list.append(nn.Linear(input_size, units))
            deep_list.append(nn.ReLU())
            input_size = units
        self.deep_network = nn.Sequential(*deep_list)
        
        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, num_features, cat_features, one_hot_features):
        embedded = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([num_features, embedded, one_hot_features], dim=1)

        x_cross = x.clone()
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)

        deep_out = self.deep_network(x_cross)
        return torch.sigmoid(self.output_layer(deep_out)).squeeze()

# 🔹 Training Function
def train_and_evaluate(params, selected_features):
    train_loader = data_loader.get_dataloader(data_loader.train_df[selected_features])
    valid_loader = data_loader.get_dataloader(data_loader.valid_df[selected_features])

    model = DCNv2(len(data_loader.numerical_columns), len(data_loader.categorical_columns), len(data_loader.one_hot_columns),
                  [(len(le.classes_), 8) for le in data_loader.encoders.values()], params['rank'], params['cross_layers'], params['deep_layers']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

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
                num_features, cat_features, one_hot_features = num_features.to(device), cat_features.to(device), one_hot_features.to(device)
                outputs = model(num_features, cat_features, one_hot_features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_aucpr = average_precision_score(val_labels, val_preds)
        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            torch.save(model.state_dict(), "best_model.pth")

    return best_val_aucpr

# 🔹 Hyperparameter Tuning with Feature Selection
def hyperparameter_tuning_with_feature_selection():
    best_score = 0
    best_params = None
    selected_features = data_loader.numerical_columns + data_loader.categorical_columns + data_loader.one_hot_columns
    
    while True:
        def objective(trial):
            params = {
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'rank': trial.suggest_int('rank', 1, 10),
                'cross_layers': trial.suggest_int('cross_layers', 1, 3),
                'deep_layers': [trial.suggest_int(f'deep_layer_{i}', 32, 128) for i in range(3)]
            }
            return train_and_evaluate(params, selected_features)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        if study.best_value <= best_score:
            break

        best_score = study.best_value
        best_params = study.best_params

    return best_params

# 🔹 Final Evaluation on Test Data
best_params = hyperparameter_tuning_with_feature_selection()


# 🔹 Final Evaluation
def evaluate_on_test(best_params):
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

best_params = hyperparameter_tuning_with_feature_selection()
evaluate_on_test(best_params)
