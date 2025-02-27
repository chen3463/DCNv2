import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import defaultdict
import optuna

# 🔹 Train Model Function
def train_model(model, train_loader, valid_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for num_features, cat_features, one_hot_features, labels in train_loader:
            num_features, cat_features, one_hot_features, labels = num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(num_features, cat_features, one_hot_features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for num_features, cat_features, one_hot_features, labels in valid_loader:
                num_features, cat_features, one_hot_features, labels = num_features.to(device), cat_features.to(device), one_hot_features.to(device), labels.to(device)
                predictions = model(num_features, cat_features, one_hot_features)
                val_loss += criterion(predictions, labels).item()
        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    return best_model

# 🔹 Compute Feature Importance
def compute_feature_importance():
    # Placeholder for feature importance computation (replace with actual logic)
    return {feature: np.random.rand() for feature in numerical_columns + categorical_columns + one_hot_columns}

# 🔹 Define Objective Function for Optuna
def objective(trial, selected_features):
    model = DCNv2(len(numerical_columns), len(categorical_columns), len(one_hot_columns), [(len(encoders[col].classes_), 8) for col in categorical_columns], cross_layers=2, deep_layers=[64, 32])
    trained_model = train_model(model, train_loader, valid_loader, epochs=5, lr=trial.suggest_loguniform('lr', 1e-4, 1e-2))
    return np.random.rand()  # Replace with actual validation metric

        deep_out = self.mlp(x_0)
        combined = torch.cat([x, deep_out], dim=1)
        return torch.sigmoid(self.final_layer(combined))

