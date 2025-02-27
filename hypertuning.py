# 🔹 Feature Selection with Low-Rank Cross Network
def low_rank_feature_importance(model, numerical_columns):
    importance = torch.stack([torch.norm(layer.weight, p=2, dim=0) for layer in model.cross_layers])
    mean_importance = torch.mean(importance, dim=0).cpu().numpy()
    return dict(zip(numerical_columns, mean_importance))

# 🔹 Gradient-based Feature Selection
def gradient_feature_importance(model, data_loader, device):
    model.eval()
    importance_scores = None
    
    for num_features, cat_features, labels in data_loader:
        num_features = num_features.to(device).requires_grad_()
        outputs = model(num_features, cat_features).squeeze()
        loss = outputs.mean()
        loss.backward()
        
        if importance_scores is None:
            importance_scores = num_features.grad.abs().mean(dim=0).cpu().numpy()
        else:
            importance_scores += num_features.grad.abs().mean(dim=0).cpu().numpy()
    
    return importance_scores / len(data_loader)

# 🔹 Feature Selection Function
def feature_selection(model, train_loader, numerical_columns, device):
    gradient_importance = gradient_feature_importance(model, train_loader, device)
    
    # Rank features by importance
    sorted_importance = sorted(zip(numerical_columns, gradient_importance), key=lambda x: x[1], reverse=True)
    cumulative_importance = np.cumsum([imp[1] for imp in sorted_importance]) / sum(gradient_importance)
    selected_features = [imp[0] for i, imp in enumerate(sorted_importance) if cumulative_importance[i] <= 0.9]
    
    print("Feature Importance Ranking:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.6f}")
    print(f"Selected Features (90% importance): {selected_features}")
    
    return selected_features

# 🔹 Hyperparameter Optimization
def objective(trial, train_loader, valid_loader, numerical_columns, device):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    rank = trial.suggest_int('rank', 1, 4)
    mlp_hidden_dim = trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256])
    mlp_layers = trial.suggest_int('mlp_layers', 2, 4)
    
    model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, 
                   num_cross_layers=3, rank=rank, mlp_dims=[mlp_hidden_dim] * mlp_layers).to(device)
    
    selected_features = feature_selection(model, train_loader, numerical_columns, device)
    filtered_columns = [i for i, col in enumerate(numerical_columns) if col in selected_features]
    
    train_loader.filtered_features = filtered_columns
    valid_loader.filtered_features = filtered_columns
    
    save_path = f"best_model_trial_{trial.number}.pth"
    trained_model = train_model(model, train_loader, valid_loader, epochs=20, lr=lr, patience=5, device=device, save_path=save_path)
    
    trained_model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for num_features, cat_features, labels in valid_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            outputs = trained_model(num_features, cat_features).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    aucpr = average_precision_score(val_targets, val_preds)
    return aucpr

# 🔹 Run Hyperparameter Optimization
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_loader, valid_loader, numerical_columns, device), n_trials=20)

# 🔹 Load Best Model
best_trial = study.best_trial
best_model_path = f"best_model_trial_{best_trial.number}.pth"
best_model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, 
                   num_cross_layers=3, rank=best_trial.params['rank'], mlp_dims=[best_trial.params['mlp_hidden_dim']] * best_trial.params['mlp_layers']).to(device)
best_model.load_state_dict(torch.load(best_model_path))

# 🔹 Evaluate on Test Data
evaluate_model(best_model, test_loader, device)
