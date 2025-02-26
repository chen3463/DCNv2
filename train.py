from sklearn.metrics import average_precision_score

def train_model(model, train_loader, valid_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for num_features, cat_features, labels in train_loader:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(num_features, cat_features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for num_features, cat_features, labels in valid_loader:
                num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
                outputs = model(num_features, cat_features).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        aucpr = average_precision_score(val_targets, val_preds)
        print(f"Epoch [{epoch+1}/{epochs}], AUCPR: {aucpr:.4f}")
