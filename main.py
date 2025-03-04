from data_load import DataLoaderWrapper
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset sizes
num_train = 10000
num_valid = 2000
num_test = 2000

# Define feature distributions
def generate_data(num_samples):
    # Numerical features (random normal distribution)
    num_features = np.random.randn(num_samples, 5) * 10

    # Categorical features (random categories)
    cat_1 = np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=num_samples)
    cat_2 = np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=num_samples)
    cat_3 = np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=num_samples)

    # One-hot encoded features (binary categories)
    one_hot_1 = np.random.choice([0, 1], size=num_samples)
    one_hot_2 = np.random.choice([0, 1], size=num_samples)

    # Binary target (1 if sum of num features + encoded categorical effect > threshold)
    target = (num_features[:, 0] + np.where(cat_1 == 'A', 5, -5) + np.where(cat_2 == 'X', 3, -3) > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame(num_features, columns=[f'num_{i + 1}' for i in range(5)])
    df['cat_1'], df['cat_2'], df['cat_3'] = cat_1, cat_2, cat_3
    df['one_hot_1'], df['one_hot_2'] = one_hot_1, one_hot_2
    df['target'] = target

    return df


# Generate datasets
train_df = generate_data(num_train)
print(train_df.describe())
valid_df = generate_data(num_valid)
print(valid_df.describe())
test_df = generate_data(num_test)
print(test_df.describe())

# Standardize numerical features
scaler = StandardScaler()
train_df.iloc[:, :5] = scaler.fit_transform(train_df.iloc[:, :5])
valid_df.iloc[:, :5] = scaler.transform(valid_df.iloc[:, :5])
test_df.iloc[:, :5] = scaler.transform(test_df.iloc[:, :5])

# Encode categorical features using LabelEncoder
for col in ['cat_1', 'cat_2', 'cat_3']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    valid_df[col] = le.transform(valid_df[col])
    test_df[col] = le.transform(test_df[col])

# Save CSV files
train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)
test_df.to_csv("test.csv", index=False)


numerical_columns = ['num_1', 'num_2', 'num_3', 'num_4', 'num_5']
categorical_columns = ['cat_1', 'cat_2', 'cat_3']
one_hot_columns = ['one_hot_1', 'one_hot_2']

data_loader = DataLoaderWrapper("train.csv", "valid.csv", "test.csv", numerical_columns, categorical_columns, one_hot_columns)

class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank):
        super(LowRankCrossLayer, self).__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        # Ensure U(x) outputs correct shape
        print(f"Shape of x before linear layer: {x.shape}")  # Should match self.U's expected input
        u_x = self.U(x)  # Shape: (batch_size, rank)
        v_u_x = self.V(u_x)  # Shape: (batch_size, input_dim)

        if v_u_x.shape != x0.shape:
            raise ValueError(f"Shape mismatch: v_u_x {v_u_x.shape} vs x0 {x0.shape}")

        return x0 * v_u_x + x + self.bias


class DCNModel(nn.Module):
    def __init__(self, num_num_features, cat_emb_cardinalities, cat_one_hot_cardinalities, embedding_dim=16,
                 unknown_idx=0):
        super(DCNModel, self).__init__()

        self.num_num_features = num_num_features
        self.unknown_idx = unknown_idx  # Index for unknown categories

        # Define embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories + 1, embedding_dim)  # Add 1 for unknown category
            for num_categories in cat_emb_cardinalities
        ])

        # Total one-hot encoding dimension
        self.cat_one_hot_cardinalities = cat_one_hot_cardinalities
        self.one_hot_dim = sum(cat_one_hot_cardinalities)

        # Compute total input dimension
        total_input_dim = num_num_features + len(cat_emb_cardinalities) * embedding_dim + self.one_hot_dim

        # Fully connected layer (example)
        self.fc = nn.Linear(total_input_dim, 1)

    def forward(self, num_features, cat_emb_features, cat_one_hot_features):
        # Handle unknown categories in embeddings by clamping values
        cat_emb_features = torch.clamp(cat_emb_features, min=0, max=len(self.embeddings) - 1)

        # Process categorical features with embeddings
        embedded_features = [emb(cat_emb_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_features = torch.cat(embedded_features, dim=1)  # Concatenate embeddings

        # One-hot encoding categorical features dynamically (with unknown category handling)
        # One-hot encoding categorical features dynamically (with unknown category handling)
        one_hot_encoded = []
        for i, card in enumerate(self.cat_one_hot_cardinalities):
            # Clamp values for unknown categories
            safe_cat_one_hot = torch.clamp(cat_one_hot_features[:, i], min=0,
                                           max=card - 1).long()  # Ensure it's LongTensor
            one_hot = F.one_hot(safe_cat_one_hot, num_classes=card)
            one_hot_encoded.append(one_hot)

        one_hot_encoded = torch.cat(one_hot_encoded, dim=1)  # Concatenate one-hot encodings

        # Concatenate numerical, embedded, and one-hot categorical features
        x = torch.cat([num_features, embedded_features, one_hot_encoded], dim=1)

        # Forward pass through FC layer
        output = self.fc(x)
        return output





# 🔹 Feature Selection with SHAP
class FeatureSelector:
    def __init__(self, data_loader, selected_features):
        self.data_loader = data_loader
        self.selected_features = selected_features

    def select_features(self):
        # Fetch a batch
        data_iter = iter(self.data_loader.get_dataloader(self.data_loader.train_df, self.selected_features))
        batch = next(data_iter)

        # Unpack batch correctly
        if len(batch) == 4:
            num_features, cat_features, one_hot_features, labels = batch
        elif len(batch) == 3:
            num_features, cat_features, one_hot_features = batch
            labels = None  # If labels are missing
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        print(f"Selected Features: {self.selected_features}")

        # Perform feature selection logic (Placeholder)
        return self.selected_features


# 🔹 Train and Evaluate the Model
def train_and_evaluate(params, selected_features, data_loader, device):
    """Train and evaluate DCNModel with selected features."""

    # Define model
    model = DCNModel(
        num_num_features=len(data_loader.numerical_columns),
        cat_emb_cardinalities=[len(data_loader.encoders[col].classes_) for col in data_loader.categorical_columns],
        cat_one_hot_cardinalities=[len(data_loader.one_hot_enc.categories_[i]) for i, col in
                                   enumerate(data_loader.one_hot_columns)],
        embedding_dim=16
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load data
    train_loader = data_loader.get_dataloader(data_loader.train_df, selected_features, batch_size=params['batch_size'])
    valid_loader = data_loader.get_dataloader(data_loader.valid_df, selected_features, batch_size=params['batch_size'])

    best_auc = 0.0
    best_model_state = None  # Store model state dict, not model itself

    for epoch in range(10):
        model.train()
        for num_features, cat_emb_features, cat_one_hot_features, labels in train_loader:
            num_features, cat_emb_features, cat_one_hot_features, labels = (
                num_features.to(device), cat_emb_features.to(device), cat_one_hot_features.to(device), labels.to(device)
            )

            optimizer.zero_grad()
            outputs = model(num_features, cat_emb_features, cat_one_hot_features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for num_features, cat_emb_features, cat_one_hot_features, labels in valid_loader:
                num_features, cat_emb_features, cat_one_hot_features, labels = (
                    num_features.to(device), cat_emb_features.to(device), cat_one_hot_features.to(device),
                    labels.to(device)
                )

                outputs = model(num_features, cat_emb_features, cat_one_hot_features).squeeze(1)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())  # Apply sigmoid for probabilities
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}: Validation AUC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()  # Save only state_dict

    # Ensure best model is properly returned
    best_model = DCNModel(
        num_num_features=len(data_loader.numerical_columns),
        cat_emb_cardinalities=[len(data_loader.encoders[col].classes_) for col in data_loader.categorical_columns],
        cat_one_hot_cardinalities=[len(data_loader.one_hot_enc.categories_[i]) for i, col in
                                   enumerate(data_loader.one_hot_columns)],
        embedding_dim=16
    ).to(device)

    if best_model_state:
        best_model.load_state_dict(best_model_state)

    return best_model, best_auc  # Return trained model



def evaluate_on_test_data(model, data_loader, selected_features, device):
    """Evaluate DCNModel on test data."""

    test_loader = data_loader.get_dataloader(data_loader.test_df, selected_features, batch_size=32)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for num_features, cat_emb_features, cat_one_hot_features, labels in test_loader:
            num_features, cat_emb_features, cat_one_hot_features, labels = (
                num_features.to(device), cat_emb_features.to(device), cat_one_hot_features.to(device), labels.to(device)
            )

            outputs = model(num_features, cat_emb_features, cat_one_hot_features).squeeze(1)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())  # Convert logits to probabilities
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    print(f"Test AUC: {auc:.4f}")


# 🔹 Hyperparameter Tuning with Feature Selection
def hyperparameter_tuning(data_loader, device):
    def objective(trial):
        params = {
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'rank': trial.suggest_int('rank', 1, 10),
            'cross_layers': trial.suggest_int('cross_layers', 1, 3),
            'deep_layers': [
                trial.suggest_int(f'deep_layer_{i}', 32, 128)
                for i in range(trial.suggest_int("num_deep_layers", 1, 5))
            ]
        }
        feature_selector = FeatureSelector(data_loader,
                                           selected_features=data_loader.numerical_columns + data_loader.categorical_columns + data_loader.one_hot_columns)
        selected_features = feature_selector.select_features()

        return train_and_evaluate(params, selected_features, data_loader, device)[1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


# 🚀 Device Configuration
# 🔹 Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Initialize DataLoaderWrapper (Ensure data_loader is created before this step)
feature_selector = FeatureSelector(
    data_loader,
    selected_features=data_loader.numerical_columns + data_loader.categorical_columns + data_loader.one_hot_columns
)

# 🔹 Run Hyperparameter Tuning
best_params = hyperparameter_tuning(data_loader, device)

# 🔹 Train the Model Using Best Params
selected_features = feature_selector.select_features()
best_model = train_and_evaluate(best_params, selected_features, data_loader, device)

# 🔹 Evaluate on Test Data
evaluate_on_test_data(best_model[0], data_loader, selected_features, device)



