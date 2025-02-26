import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    mlp_layers = trial.suggest_int('mlp_layers', 2, 4)
    mlp_hidden_dim = trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256])
    rank = trial.suggest_int('rank', 1, 4)  

    model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, num_cross_layers=3, rank=rank, mlp_dims=[mlp_hidden_dim] * mlp_layers).to(device)
    
    train_model(model, train_loader, valid_loader, epochs=20, lr=lr, device=device)

    return -average_precision_score(val_targets, val_preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print("\n✅ Best Hyperparameters:", study.best_params)
