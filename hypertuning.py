def objective(trial):
    # 🔹 Sample Hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    mlp_layers = trial.suggest_int('mlp_layers', 2, 4)
    mlp_hidden_dim = trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256])
    rank = trial.suggest_int('rank', 1, 4)  

    # 🔹 Define Model with Sampled Hyperparameters
    model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, 
                   num_cross_layers=3, rank=rank, mlp_dims=[mlp_hidden_dim] * mlp_layers).to(device)

    # 🔹 Train Model & Save Best One
    save_path = f"best_model_trial_{trial.number}.pth"
    trained_model = train_model(model, train_loader, valid_loader, epochs=20, lr=lr, patience=5, device=device, save_path=save_path)

    # 🔹 Evaluate Model on Validation Data
    trained_model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for num_features, cat_features, labels in valid_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            outputs = trained_model(num_features, cat_features).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    aucpr = average_precision_score(val_targets, val_preds)
    return -aucpr  # Optuna maximizes, so we minimize the negative value

# 🔹 Run Hyperparameter Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# 🔹 Load Best Model After Tuning
best_trial = study.best_trial
best_model_path = f"best_model_trial_{best_trial.number}.pth"
print(f"\n✅ Best Hyperparameters: {best_trial.params}")
print(f"🔥 Best model saved at: {best_model_path}")

# 🔹 Load the Best Model for Final Evaluation
best_model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, 
                   num_cross_layers=3, rank=best_trial.params['rank'], mlp_dims=[best_trial.params['mlp_hidden_dim']] * best_trial.params['mlp_layers']).to(device)

best_model.load_state_dict(torch.load(best_model_path))

# 🔹 Evaluate on Test Data
evaluate_model(best_model, test_loader, device)
