# Sample training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNv2(input_dim=20, cross_layers=3, rank=5).to(device)  # Example with 20 features
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary classification loss

# Dummy data
X_train = torch.randn(100, 20).to(device)  # 100 samples, 20 features
y_train = torch.randint(0, 2, (100, 1)).float().to(device)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
