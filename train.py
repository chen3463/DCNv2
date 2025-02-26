from sklearn.metrics import average_precision_score, roc_auc_score
import torch

def train_model(model, train_loader, valid_loader, epochs, lr, patience, device, save_path="best_model.pth"):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_aucpr = 0
    best_model_path = save_path
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for num_features, cat_features, labels in train_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(num_features, cat_features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 🔹 Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for num_features, cat_features, labels in valid_loader:
                num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
                outputs = model(num_features, cat_features).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        aucpr = average_precision_score(val_targets, val_preds)
        auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch [{epoch+1}/{epochs}]  AUCPR: {aucpr:.4f}  AUC: {auc:.4f}")

        # 🔹 Save Best Model
        if aucpr > best_aucpr:
            best_aucpr = aucpr
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 🔹 Early Stopping
        if patience_counter >= patience:
            print(f"⏳ Early stopping at epoch {epoch+1}")
            break

    # 🔹 Load Best Model Before Returning
    model.load_state_dict(torch.load(best_model_path))
    return model

