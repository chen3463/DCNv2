def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for num_features, cat_features, labels in test_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            outputs = model(num_features, cat_features).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # Compute Test Metrics
    aucpr = average_precision_score(test_targets, test_preds)
    auc = roc_auc_score(test_targets, test_preds)

    print(f"\n📌 Test Results:")
    print(f"🔥 AUCPR: {aucpr:.4f}")
    print(f"🔥 AUC: {auc:.4f}")

    return aucpr, auc

# 🔹 Load Best Model from Hyperparameter Tuning
best_model = DCNv2(len(numerical_columns), [train_df[col].nunique() for col in categorical_columns], emb_dim=16, 
                   num_cross_layers=3, rank=study.best_params['rank'], mlp_dims=[study.best_params['mlp_hidden_dim']] * study.best_params['mlp_layers']).to(device)

best_model.load_state_dict(torch.load("best_model.pth"))  # Ensure to save the best model during training

# 🔹 Evaluate on Test Data
evaluate_model(best_model, test_loader, device)
