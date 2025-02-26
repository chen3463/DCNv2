import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 🔹 Load CSV Files
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 🔹 Identify Column Types
def preprocess_data(df, categorical_columns, numerical_columns, target_column):
    """
    Preprocesses the dataset:
    - Encodes categorical features using LabelEncoder.
    - Normalizes numerical features using StandardScaler.
    - Converts data into PyTorch tensors.

    Returns:
        tensors for numerical features, categorical features, and target labels.
    """
    # Encode categorical features
    cat_encoders = {col: LabelEncoder() for col in categorical_columns}
    cat_tensors = [torch.tensor(cat_encoders[col].fit_transform(df[col]), dtype=torch.long) for col in categorical_columns]

    # Normalize numerical features
    scaler = StandardScaler()
    num_tensor = torch.tensor(scaler.fit_transform(df[numerical_columns]), dtype=torch.float32)

    # Convert target column
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    # Stack categorical tensors
    cat_tensor = torch.stack(cat_tensors, dim=1) if cat_tensors else torch.empty((len(df), 0))

    return num_tensor, cat_tensor, target_tensor

# 🔹 Load Data
train_df = load_data("train.csv")
valid_df = load_data("valid.csv")
test_df = load_data("test.csv")

# 🔹 Define Feature Columns
categorical_columns = ["cat1", "cat2", "cat3"]  # Replace with actual categorical column names
numerical_columns = ["num1", "num2", "num3", "num4"]  # Replace with actual numerical column names
target_column = "label"  # Replace with your actual target column

# 🔹 Preprocess Data
train_num, train_cat, train_labels = preprocess_data(train_df, categorical_columns, numerical_columns, target_column)
valid_num, valid_cat, valid_labels = preprocess_data(valid_df, categorical_columns, numerical_columns, target_column)
test_num, test_cat, test_labels = preprocess_data(test_df, categorical_columns, numerical_columns, target_column)

# 🔹 Create Tensor Datasets
train_dataset = TensorDataset(train_num, train_cat, train_labels)
valid_dataset = TensorDataset(valid_num, valid_cat, valid_labels)
test_dataset = TensorDataset(test_num, test_cat, test_labels)

# 🔹 Create DataLoaders
batch_size = 64  # Can be tuned using Optuna
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
