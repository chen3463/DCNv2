import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    mlp_layers = trial.suggest_int('mlp_layers', 2, 4)
    mlp_hidden_dim = trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256])

    # Reload Data with New Batch Size
    train_loader = create_dataloader(train_num, train_cat, train_labels, batch_size)
    valid_loader = create_dataloader(valid_num, valid_cat, valid_labels, batch_size)

    # Initialize Model
    model = DCNv2(
        num_numerical=len(numerical_columns),
        cat_cardinalities=[train_df[col].nunique() for col in categorical_columns],
        emb_dim=16,
        num_cross_layers=3,
        mlp_dims=[mlp_hidden_dim] * mlp_layers
    ).to(device)

    # Train Model
    train_model(model, train_loader, valid_loader, epochs=20, lr=lr, device=device)

    # Evaluate Validation Loss
    val_loss = 0
    with torch.no_grad():
        for num_features, cat_features, labels in valid_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            outputs = model(num_features, cat_features).squeeze()
            val_loss += nn.BCELoss()(outputs, labels).item()
    
    return val_loss / len(valid_loader)

# 🔹 Run Optuna Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\n✅ Best Hyperparameters:")
print(study.best_params)
