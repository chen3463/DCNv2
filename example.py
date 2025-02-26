import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define Objective Function for Optuna
def objective(trial):
    """
    Optuna objective function to optimize DCN v2 hyperparameters.
    """
    # 🔹 Sample Hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Batch size
    mlp_layers = trial.suggest_int('mlp_layers', 2, 4)  # Number of MLP layers
    mlp_hidden_dim = trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256])  # Hidden layer dim
    
    # 🔹 Prepare Data Loaders with Selected Batch Size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 🔹 Define Model with Sampled MLP Architecture
    model = DCNv2(
        num_numerical=10,  # Replace with your actual numerical feature count
        cat_cardinalities=[5, 10, 8],  # Replace with your actual categorical cardinalities
        emb_dim=16,
        num_cross_layers=3,
        mlp_dims=[mlp_hidden_dim] * mlp_layers  # Dynamic MLP structure
    ).to(device)

    # 🔹 Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔹 Training Loop (Shortened for Efficiency)
    best_val_loss = float('inf')
    for epoch in range(20):  # Shorter training for tuning
        model.train()
        for num_features, cat_features, labels in train_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(num_features, cat_features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 🔹 Validate Model Performance
        val_loss, _ = evaluate_model(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Track best validation loss

    return best_val_loss  # Minimize validation loss

# 🔹 Run Optuna Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Run 20 trials

# 🔹 Print Best Hyperparameters
print("\n✅ Best Hyperparameters Found:")
print(study.best_params)
