import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import DataLoader, Dataset
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

# ===============================
# Custom Dataset
# ===============================
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset to handle numerical, embedding categorical, and one-hot categorical features.
    """
    def __init__(self, numerical, categorical_emb, categorical_onehot, labels=None):
        self.numerical = torch.tensor(numerical, dtype=torch.float32)
        self.categorical_emb = torch.tensor(categorical_emb, dtype=torch.long)
        self.categorical_onehot = torch.tensor(categorical_onehot, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.numerical)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.numerical[idx], self.categorical_emb[idx], self.categorical_onehot[idx], self.labels[idx]
        return self.numerical[idx], self.categorical_emb[idx], self.categorical_onehot[idx]

# ===============================
# DCNv2 Model
# ===============================
class DCNv2(nn.Module):
    """
    Deep & Cross Network v2 implementation.
    Combines embedding layers, numerical inputs, one-hot inputs, cross layers, and deep layers.
    """
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size):
        super(DCNv2, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cat_card + 1, embedding_dim) for cat_card in cat_cardinalities])
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim + onehot_size
        self.cross_net = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(cross_layers)])
        self.deep_net = nn.Sequential(
            *[nn.Linear(input_dim, deep_layers[0]), nn.ReLU()] +
            sum([[nn.Linear(deep_layers[i], deep_layers[i+1]), nn.ReLU()] for i in range(len(deep_layers)-1)], [])
        )
        self.output_layer = nn.Linear(deep_layers[-1], 1)

    def forward(self, numerical, categorical_emb, categorical_onehot):
        cat_embeds = [emb(categorical_emb[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        x = torch.cat([numerical, cat_embeds, categorical_onehot], dim=1)
        for layer in self.cross_net:
            x = x + layer(x)
        x = self.deep_net(x)
        return torch.sigmoid(self.output_layer(x)).squeeze(1)

# ===============================
# Feature Importance with SHAP
# ===============================
# Feature importance function using SHAP
# def feature_importance(model, train_loader, feature_names):
#     """
#     Extract SHAP feature importance and return as a DataFrame.
#
#     Args:
#         model: Trained DCNv2 model.
#         train_loader: DataLoader for training data.
#         feature_names (list): List of feature names corresponding to model input.
#
#     Returns:
#         pd.DataFrame: DataFrame with columns ['feature', 'mean_abs_shap_value'] sorted by importance.
#     """
#     model.eval()
#     # Only sample one batch for SHAP to reduce computation
#     numerical_data, categorical_emb_data, categorical_onehot_data, _ = next(iter(train_loader))
#     input_data = torch.cat([numerical_data, categorical_emb_data.float(), categorical_onehot_data], dim=1)
#
#     explainer = shap.Explainer(model, input_data)
#     shap_values = explainer(input_data)
#
#     mean_shap = np.abs(shap_values.values).mean(axis=0)
#     df = pd.DataFrame({
#         'feature': feature_names,
#         'mean_abs_shap_value': mean_shap
#     }).sort_values(by='mean_abs_shap_value', ascending=False).reset_index(drop=True)
#
#     return df

# ===============================
# Evaluate Model on Test Data
# ===============================
def evaluate_model(model, test_loader):
    """
    Evaluate model on test data.
    Inputs:
        model: Trained DCNv2 model.
        test_loader: DataLoader for test data.
    Returns:
        Predictions (probabilities) on test data.
    """
    model.eval()
    test_preds = []
    with torch.no_grad():
        for numerical, categorical_emb, categorical_onehot in test_loader:
            outputs = model(numerical, categorical_emb, categorical_onehot)
            test_preds.extend(outputs.numpy())
    return np.array(test_preds)

# Function for training and evaluation during hyperparameter tuning
def train_evaluate_model(trial, train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size):
    embedding_dim = trial.suggest_int('embedding_dim', 4, 16)
    cross_layers = trial.suggest_int('cross_layers', 1, 3)
    deep_layers = [trial.suggest_int(f'deep_layer_{i}', 16, 64) for i in range(2)]

    model = DCNv2(num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))
    criterion = nn.BCELoss()

    best_auc = 0
    patience, patience_counter = 3, 0

    for epoch in range(20):
        model.train()
        for numerical, categorical_emb, categorical_onehot, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(numerical, categorical_emb, categorical_onehot).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for numerical, categorical_emb, categorical_onehot, labels in val_loader:
                outputs = model(numerical, categorical_emb, categorical_onehot).squeeze()
                val_preds.extend(outputs.numpy())
                val_labels.extend(labels.squeeze().numpy())

        auc = roc_auc_score(val_labels, val_preds)
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model, best_auc

# Optuna optimization loop
def optimize_dcn_hyperparameters(train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size, n_trials=50):
    def objective(trial):
        _, best_auc = train_evaluate_model(trial, train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size)
        return best_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study

## Feature importance function with DataFrame output
def feature_importance(model, data_loader, feature_names, emb_feature_slices, onehot_feature_slices):
    model.eval()

    numerical_data, categorical_emb_data, categorical_onehot_data, _ = next(iter(data_loader))

    input_data = torch.cat([numerical_data, categorical_emb_data.float(), categorical_onehot_data], dim=1)
    input_data_np = input_data.cpu().numpy()

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            num = x_tensor[:, :numerical_data.shape[1]]
            emb = x_tensor[:, numerical_data.shape[1]:numerical_data.shape[1] + categorical_emb_data.shape[1]].long()
            onehot = x_tensor[:, numerical_data.shape[1] + categorical_emb_data.shape[1]:]
            return model(num, emb, onehot).squeeze().cpu().numpy()

    masker = shap.maskers.Independent(input_data_np)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(input_data_np)

    num_features = numerical_data.shape[1]

    mean_shap = np.abs(shap_values.values).mean(axis=0)

    mean_shap_values = []

    # Numerical
    mean_shap_values.extend(mean_shap[:num_features])

    # Embeddings (aggregate per original variable)
    for start, end in emb_feature_slices:
        mean_emb = mean_shap[start:end].mean()
        mean_shap_values.append(mean_emb)

    # One-hot (aggregate per original variable)
    for start, end in onehot_feature_slices:
        mean_onehot = mean_shap[start:end].mean()
        mean_shap_values.append(mean_onehot)

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap_value': mean_shap_values
    }).sort_values(by='mean_abs_shap_value', ascending=False)

    return df

# Auto feature name builder with per-variable aggregation
def build_feature_names(num_numerical, emb_cat_cardinalities, onehot_cat_names):
    num_feature_names = [f"num_{i}" for i in range(num_numerical)]
    emb_feature_names = [f"emb_cat_{i}" for i in range(len(emb_cat_cardinalities))]
    onehot_feature_names = [f"onehot_cat_{name}" for name in onehot_cat_names]

    return num_feature_names + emb_feature_names + onehot_feature_names

# Helper to create feature slices for embeddings and one-hot encoding
def build_feature_slices(num_numerical, emb_cat_cardinalities, onehot_cardinalities):
    emb_feature_slices = []
    start = num_numerical
    for cardinality in emb_cat_cardinalities:
        emb_feature_slices.append((start, start + cardinality))
        start += cardinality

    onehot_feature_slices = []
    for cardinality in onehot_cardinalities:
        onehot_feature_slices.append((start, start + cardinality))
        start += cardinality

    return emb_feature_slices, onehot_feature_slices

# ===============================
# Sample Data for Testing Framework
# ===============================
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Dummy data preparation
num_numerical = 5
num_categorical_emb = 3
onehot_cardinalities = [4, 3]  # e.g., 2 one-hot encoded categorical features
onehot_size = sum(onehot_cardinalities)

# Simulate numerical, categorical_emb (int ids for embedding), and one-hot features
N = 1000  # number of samples

numerical_data = torch.randn(N, num_numerical)
categorical_emb_data = torch.randint(0, 10, (N, num_categorical_emb))  # assuming cardinality 10 for all embeddings
categorical_onehot_data = torch.randint(0, 2, (N, onehot_size))
labels = torch.randint(0, 2, (N, 1)).float()

# Train/val split
split_idx = int(N * 0.8)
train_dataset = TensorDataset(
    numerical_data[:split_idx],
    categorical_emb_data[:split_idx],
    categorical_onehot_data[:split_idx],
    labels[:split_idx]
)
val_dataset = TensorDataset(
    numerical_data[split_idx:],
    categorical_emb_data[split_idx:],
    categorical_onehot_data[split_idx:],
    labels[split_idx:]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Dummy cardinalities for embedding layer (e.g., if categorical variables have 10 unique values each)
cat_cardinalities = [10, 10, 10]  # for 3 categorical embedding features

# Run Optuna optimization (reduced n_trials for speed)
study = optimize_dcn_hyperparameters(train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size, n_trials=5)

# Assuming you've already run the Optuna optimization and have the best hyperparameters
best_params = study.best_trial.params

# Recreate and retrain the model with the best params
embedding_dim = best_params['embedding_dim']
cross_layers = best_params['cross_layers']
deep_layers = [best_params['deep_layer_0'], best_params['deep_layer_1']]
learning_rate = best_params['lr']

# Instantiate model
final_model = DCNv2(num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size)
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Train the final model on full train set
final_model.train()
for epoch in range(10):  # Fewer epochs for demonstration
    for numerical, categorical_emb, categorical_onehot, labels in train_loader:
        labels = labels.squeeze(1)
        optimizer.zero_grad()
        outputs = final_model(numerical, categorical_emb, categorical_onehot)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
