import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import optuna
import shap
import logging

# ===============================
# Logging Configuration
# ===============================
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ===============================
# Custom Dataset
# ===============================
class CustomDataset(Dataset):
    def __init__(self, numerical, categorical_emb, categorical_onehot, labels=None):
        self.numerical = torch.tensor(numerical, dtype=torch.float32)
        self.categorical_emb = torch.tensor(categorical_emb, dtype=torch.long)
        self.categorical_onehot = torch.tensor(categorical_onehot, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.numerical)

    def __getitem__(self, idx):
        data = (self.numerical[idx], self.categorical_emb[idx], self.categorical_onehot[idx])
        return data + (self.labels[idx],) if self.labels is not None else data

# ===============================
# Model Definition
# ===============================
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

# ===============================
# Training & Evaluation
# ===============================
def train_evaluate_model(train_loader, val_loader, model, optimizer, criterion, device, trial, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for numerical, categorical_emb, categorical_onehot, target in train_loader:
            numerical, categorical_emb, categorical_onehot, target = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device), target.view(-1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(numerical, categorical_emb, categorical_onehot), target)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for numerical, categorical_emb, categorical_onehot, target in val_loader:
                numerical, categorical_emb, categorical_onehot = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device)
                output = model(numerical, categorical_emb, categorical_onehot)
                all_preds.append(output.cpu())
                all_targets.append(target.view(-1).cpu())
        auc = roc_auc_score(torch.cat(all_targets), torch.cat(all_preds))
        trial.report(auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return auc

# ===============================
# Hyperparameter Optimization
# ===============================
def optimize_dcn_hyperparameters(train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size, device, n_trials=50):
    def objective(trial):
        params = {
            'embedding_dim': trial.suggest_categorical('embedding_dim', [8, 16, 32]),
            'cross_layers': trial.suggest_int('cross_layers', 2, 6),
            'cross_rank': trial.suggest_categorical('cross_rank', [4, 8, 16, 32]),
            'deep_layers': [trial.suggest_int('deep_layer_size', 64, 512, step=64) for _ in range(trial.suggest_int('deep_num_layers', 1, 3))],
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        }
        model = DCNv2(num_numerical, cat_cardinalities, params['embedding_dim'], params['cross_layers'], params['cross_rank'], params['deep_layers'], onehot_size)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.BCELoss()
        return train_evaluate_model(train_loader, val_loader, model, optimizer, criterion, device, trial)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=n_trials)
    return study

# ===============================
# SHAP Feature Importance
# ===============================
def feature_importance(model, data_loader, feature_names, emb_feature_slices, onehot_feature_slices):
    model.eval()
    numerical, categorical_emb, categorical_onehot, _ = next(iter(data_loader))
    input_data = torch.cat([numerical, categorical_emb.float(), categorical_onehot], dim=1)
    input_data_np = input_data.cpu().numpy()

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        num = x_tensor[:, :numerical.shape[1]]
        emb = x_tensor[:, numerical.shape[1]:numerical.shape[1] + categorical_emb.shape[1]].long()
        onehot = x_tensor[:, numerical.shape[1] + categorical_emb.shape[1]:]
        return model(num, emb, onehot).squeeze().cpu().detach().numpy()

    masker = shap.maskers.Independent(input_data_np)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(input_data_np)
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap_values = []

    mean_shap_values.extend(mean_shap[:numerical.shape[1]])  # numerical
    for start, end in emb_feature_slices:                   # embeddings
        mean_shap_values.append(mean_shap[start:end].mean())
    for start, end in onehot_feature_slices:                # one-hot
        mean_shap_values.append(mean_shap[start:end].mean())

    df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap_value': mean_shap_values})
    return df.sort_values(by='mean_abs_shap_value', ascending=False)

# ===============================
# Feature Name Helper
# ===============================
def build_feature_names(num_numerical, emb_cat_cardinalities, onehot_cat_names, onehot_cat_dims):
    num_feature_names = [f"num_{i}" for i in range(num_numerical)]
    emb_feature_names, emb_feature_slices = [], []
    start = num_numerical
    for i in range(len(emb_cat_cardinalities)):
        emb_feature_names.append(f"emb_cat_{i}")
        emb_feature_slices.append((start, start + 1))
        start += 1
    onehot_feature_names, onehot_feature_slices = [], []
    for i, name in enumerate(onehot_cat_names):
        dim = onehot_cat_dims[i]
        onehot_feature_names.append(name)
        onehot_feature_slices.append((start, start + dim))
        start += dim
    feature_names = num_feature_names + emb_feature_names + onehot_feature_names
    return feature_names, emb_feature_slices, onehot_feature_slices

# ===============================
# Save Artifacts
# ===============================
def save_results(df_importance, model, model_path="best_model.pth", shap_path="feature_importance.csv"):
    torch.save(model.state_dict(), model_path)
    df_importance.to_csv(shap_path, index=False)
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Feature importances saved to {shap_path}")

# ===============================
# Dummy Data for Demo
# ===============================
if __name__ == "__main__":
    num_numerical = 5
    num_categorical_emb = 3
    onehot_cardinalities = [4, 3]
    onehot_size = sum(onehot_cardinalities)
    N = 1000

    numerical_data = torch.randn(N, num_numerical)
    categorical_emb_data = torch.randint(0, 10, (N, num_categorical_emb))
    categorical_onehot_data = torch.randint(0, 2, (N, onehot_size))
    labels = torch.randint(0, 2, (N, 1)).float()

    split_idx = int(N * 0.8)
    train_loader = DataLoader(TensorDataset(
        numerical_data[:split_idx], categorical_emb_data[:split_idx], categorical_onehot_data[:split_idx], labels[:split_idx]
    ), batch_size=32, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        numerical_data[split_idx:], categorical_emb_data[split_idx:], categorical_onehot_data[split_idx:], labels[split_idx:]
    ), batch_size=32)

    cat_cardinalities = [10] * num_categorical_emb
    study = optimize_dcn_hyperparameters(train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size, device=device, n_trials=5)
    best_params = study.best_trial.params
    deep_layers = [best_params['deep_layer_size']] * best_params['deep_num_layers']

    final_model = DCNv2(num_numerical, cat_cardinalities, best_params['embedding_dim'], best_params['cross_layers'], best_params['cross_rank'], deep_layers, onehot_size)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.BCELoss()

    final_model.train()
    for epoch in range(10):
        for numerical, categorical_emb, categorical_onehot, labels in train_loader:
            labels = labels.squeeze(1)
            optimizer.zero_grad()
            loss = criterion(final_model(numerical, categorical_emb, categorical_onehot), labels)
            loss.backward()
            optimizer.step()

    onehot_cat_names = [f"cat{i+1}" for i in range(len(onehot_cardinalities))]
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(np.random.randint(0, 2, size=(N, len(onehot_cardinalities))))
    onehot_cat_dims = [len(cats) for cats in onehot_encoder.categories_]

    feature_names, emb_feature_slices, onehot_feature_slices = build_feature_names(
        num_numerical, cat_cardinalities, onehot_cat_names, onehot_cat_dims
    )
    df_importance = feature_importance(final_model, train_loader, feature_names, emb_feature_slices, onehot_feature_slices)
    print(df_importance)
