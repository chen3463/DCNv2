import logging
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from utils import setup_logging, get_device
from data_preprocessing import generate_dummy_data, build_feature_names
from hyperparameter_optimization import optimize_dcn_hyperparameters
from model import DCNv2
from shap_analysis import feature_importance
import pandas as pd

def main():
    # Setup logging and device
    setup_logging()
    device = get_device()

    # Data loading
    N = 1000
    num_numerical = 5
    num_categorical_emb = 3
    onehot_cardinalities = [4, 3]
    train_loader, val_loader, numerical_data, categorical_emb_data, categorical_onehot_data, labels = generate_dummy_data(
        N, num_numerical, num_categorical_emb, onehot_cardinalities)

    # Build feature names for SHAP analysis
    feature_names, emb_feature_slices, onehot_feature_slices = build_feature_names(
        num_numerical, [10]*num_categorical_emb, ["onehot_cat1", "onehot_cat2"], [2, 2])

    # Hyperparameter optimization
    study = optimize_dcn_hyperparameters(train_loader, val_loader, num_numerical, [10]*num_categorical_emb, sum(onehot_cardinalities), device, n_trials=5)
    best_params = study.best_trial.params
    deep_layers = [best_params['deep_layer_size']] * best_params['deep_num_layers']

    # Final model training
    final_model = DCNv2(num_numerical, [10]*num_categorical_emb, best_params['embedding_dim'], best_params['cross_layers'], best_params['cross_rank'], deep_layers, sum(onehot_cardinalities))
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = torch.nn.BCELoss()

    # Training the final model
    final_model.train()
    for epoch in range(10):
        for numerical, categorical_emb, categorical_onehot, target in train_loader:
            target = target.squeeze(1)
            optimizer.zero_grad()
            loss = criterion(final_model(numerical, categorical_emb, categorical_onehot), target)
            loss.backward()
            optimizer.step()

    # Model Evaluation: AUC on validation set
    final_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for numerical, categorical_emb, categorical_onehot, target in val_loader:
            output = final_model(numerical, categorical_emb, categorical_onehot)
            all_preds.append(output.cpu())
            all_targets.append(target.view(-1).cpu())
    auc = roc_auc_score(torch.cat(all_targets), torch.cat(all_preds))
    logging.info(f"Model AUC: {auc:.4f}")

    # Save the trained model
    model_save_path = 'dcnv2_model.pth'
    torch.save(final_model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # SHAP Analysis for feature importance
    shap_df = feature_importance(final_model, val_loader, feature_names, emb_feature_slices, onehot_feature_slices, device)
    logging.info(f"Top 10 important features:\n{shap_df.head(10)}")

    # Optionally save the SHAP results to CSV for further analysis
    shap_df.to_csv('shap_feature_importance.csv', index=False)

if __name__ == "__main__":
    main()
