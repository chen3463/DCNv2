import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 🔹 Load CSV Files
def load_data(file_path):
    return pd.read_csv(file_path)

# 🔹 Preprocess Data
def preprocess_data(df, categorical_columns, numerical_columns, target_column):
    """
    Preprocess dataset:
    - Encodes categorical features
    - Normalizes numerical features
    - Converts data into PyTorch tensors
    """
    cat_encoders = {col: LabelEncoder() for col in categorical_columns}
    cat_tensors = [torch.tensor(cat_encoders[col].fit_transform(df[col]), dtype=torch.long) for col in categorical_columns]

    scaler = StandardScaler()
    num_tensor = torch.tensor(scaler.fit_transform(df[numerical_columns]), dtype=torch.float32)
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    cat_tensor = torch.stack(cat_tensors, dim=1) if cat_tensors else torch.empty((len(df), 0))
    return num_tensor, cat_tensor, target_tensor

# 🔹 Load Data
train_df = load_data("train.csv")
valid_df = load_data("valid.csv")
test_df = load_data("test.csv")

# 🔹 Define Columns
categorical_columns = ["cat1", "cat2", "cat3"]  # Update with actual column names
numerical_columns = ["num1", "num2", "num3", "num4"]  # Update with actual column names
target_column = "label"

# 🔹 Preprocess Data
train_num, train_cat, train_labels = preprocess_data(train_df, categorical_columns, numerical_columns, target_column)
valid_num, valid_cat, valid_labels = preprocess_data(valid_df, categorical_columns, numerical_columns, target_column)
test_num, test_cat, test_labels = preprocess_data(test_df, categorical_columns, numerical_columns, target_column)

# 🔹 Create DataLoaders
def create_dataloader(num_tensor, cat_tensor, labels, batch_size):
    dataset = TensorDataset(num_tensor, cat_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 64  # Can be tuned
train_loader = create_dataloader(train_num, train_cat, train_labels, batch_size)
valid_loader = create_dataloader(valid_num, valid_cat, valid_labels, batch_size)
test_loader = create_dataloader(test_num, test_cat, test_labels, batch_size)
