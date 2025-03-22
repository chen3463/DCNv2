import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import optuna

def train_evaluate_model(train_loader, val_loader, model, optimizer, criterion, device, trial, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for numerical, categorical_emb, categorical_onehot, target in train_loader:
            numerical, categorical_emb, categorical_onehot, target = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device), target.view(-1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(numerical, categorical_emb, categorical_onehot), target)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for numerical, categorical_emb, categorical_onehot, target in val_loader:
                numerical, categorical_emb, categorical_onehot = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device)
                output = model(numerical, categorical_emb, categorical_onehot)
                all_preds.append(output.cpu())
                all_targets.append(target.view(-1).cpu())
        auc = roc_auc_score(torch.cat(all_targets), torch.cat(all_preds))
        trial.report(auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return auc
