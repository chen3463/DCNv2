import optuna
from train import train_evaluate_model
from model import DCNv2
import torch.optim as optim
import torch.nn as nn

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
