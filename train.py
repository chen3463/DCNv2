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
        val_loss = 0
        with torch.no_grad():
            for num_features, cat_features, labels in valid_loader:
                num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
                outputs = model(num_features, cat_features).squeeze()
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss/len(valid_loader):.4f}")
