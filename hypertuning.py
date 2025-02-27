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

# 🔹 Hyperparameter Tuning with Feature Selection
def hyperparameter_tuning_with_feature_selection():
    best_performance = 0
    patience = 3
    patience_counter = 0
    selected_features = numerical_columns + categorical_columns + one_hot_columns
    
    while patience_counter < patience:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, selected_features), n_trials=10)
        
        best_trial = study.best_trial
        print(f"Best Hyperparameters: {best_trial.params}")
        
        # Feature Importance Analysis
        importance_scores = compute_feature_importance()  # Implement feature importance computation
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        cumulative_importance = 0
        selected_features = []
        for feature, importance in sorted_features:
            selected_features.append(feature)
            cumulative_importance += importance
            if cumulative_importance >= 0.9:
                break
        
        print(f"Selected Features for Next Iteration: {selected_features}")
        
        if best_trial.value > best_performance:
            best_performance = best_trial.value
            patience_counter = 0
        else:
            patience_counter += 1

# 🔹 Run Feature Selection and Hyperparameter Tuning
hyperparameter_tuning_with_feature_selection()
