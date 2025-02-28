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
            self.valid_df[col] = self.valid_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_))
            self.test_df[col] = self.test_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_))
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
        
def train_and_evaluate(params, selected_features):
    # Prepare Data Loaders
    train_loader = data_loader.get_dataloader(data_loader.train_df[selected_features + ['target']], batch_size=params['batch_size'])
    valid_loader = data_loader.get_dataloader(data_loader.valid_df[selected_features + ['target']], batch_size=params['batch_size'])
    
    # Model Initialization
    num_numerical = len([f for f in selected_features if f in data_loader.numerical_columns])
    num_categorical = len([f for f in selected_features if f in data_loader.categorical_columns])
    num_one_hot = len([f for f in selected_features if f in data_loader.one_hot_columns])
    
    embedding_sizes = [(len(data_loader.encoders[col].classes_), min(50, (len(data_loader.encoders[col].classes_) // 2) + 1)) for col in data_loader.categorical_columns if col in selected_features]
    
    # 🔹 Extract Deep Layers Correctly
    deep_layers = params.get("deep_layers", [64, 64, 64])  # Default to [64, 64, 64] if not found
    
    model = DCNv2(
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        num_one_hot=num_one_hot,
        embedding_sizes=embedding_sizes,
        rank=params['rank'],
        cross_layers=params['cross_layers'],
        deep_layers=deep_layers
    ).to(device)
    
    # Optimizer & Loss Function
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCELoss()  # Binary classification
    
    best_val_aucpr = 0
    best_model_path = "best_model.pth"
    
    # Training Loop
    for epoch in range(10):  # Adjust epoch count as needed
        model.train()
        train_preds, train_labels = [], []
        
        for num_features, cat_features, one_hot_features, labels in train_loader:
            num_features, cat_features, one_hot_features, labels = (
                num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model(num_features, cat_features, one_hot_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_aucpr = average_precision_score(train_labels, train_preds)

        # Validation Loop
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for num_features, cat_features, one_hot_features, labels in valid_loader:
                num_features, cat_features, one_hot_features, labels = (
                    num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
                )
                outputs = model(num_features, cat_features, one_hot_features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_aucpr = average_precision_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Train AUCPR = {train_aucpr:.4f}, Validation AUCPR = {val_aucpr:.4f}")

        # Save the best model
        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            torch.save(model.state_dict(), best_model_path)

    return best_val_aucpr

# 🔹 Feature Selection & Hyperparameter Tuning
def hyperparameter_tuning_with_feature_selection():
    best_score = 0
    best_params = None
    selected_features = data_loader.numerical_columns + data_loader.categorical_columns + data_loader.one_hot_columns
    
    while True:
        def objective(trial):
            params = {
                'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'rank': trial.suggest_int('rank', 1, 10),
                'cross_layers': trial.suggest_int('cross_layers', 1, 3),
                'deep_layers': [trial.suggest_int(f'deep_layer_{i}', 32, 128) for i in range(3)]
            }
            val_aucpr = train_and_evaluate(params, selected_features)
            return val_aucpr

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        
        if study.best_value <= best_score:
            break
        
        best_score = study.best_value
        best_params = study.best_params
        
        # Feature Importance Calculation
        explainer = shap.Explainer(model, torch.cat([num_features, cat_features, one_hot_features], dim=1))
        shap_values = explainer.shap_values(torch.cat([num_features, cat_features, one_hot_features], dim=1))
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_ranking = sorted(zip(selected_features, feature_importance), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in feature_ranking[:int(0.9 * len(feature_ranking))]]
        
        print("Selected Features in this iteration:", selected_features)

    return best_params

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
