import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# 🔹 Load CSV Files
def load_data(file_path):
    return pd.read_csv(file_path)

# 🔹 Fit Label Encoders on Training Data
def fit_label_encoders(df, categorical_columns):
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(df[col])
        encoders[col] = encoder
    return encoders

# 🔹 Transform Data & Handle Unseen Categories
def transform_categorical(df, categorical_columns, encoders):
    cat_tensors = []
    for col in categorical_columns:
        encoder = encoders[col]
        transformed = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_))
        cat_tensors.append(torch.tensor(transformed.values, dtype=torch.long))
    
    return torch.stack(cat_tensors, dim=1) if cat_tensors else torch.empty((len(df), 0))

# 🔹 Preprocess Data
def preprocess_data(df, categorical_columns, numerical_columns, target_column, encoders=None, scaler=None, train_mode=True):
    if train_mode:
        encoders = fit_label_encoders(df, categorical_columns)
        scaler = StandardScaler().fit(df[numerical_columns])

    cat_tensor = transform_categorical(df, categorical_columns, encoders)
    num_tensor = torch.tensor(scaler.transform(df[numerical_columns]), dtype=torch.float32)
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return num_tensor, cat_tensor, target_tensor, encoders, scaler

# 🔹 Load Data
train_df = load_data("train.csv")
valid_df = load_data("valid.csv")
test_df = load_data("test.csv")

# 🔹 Define Columns
categorical_columns = ["cat1", "cat2", "cat3"]  
numerical_columns = ["num1", "num2", "num3", "num4"]  
target_column = "label"

# 🔹 Preprocess Data
train_num, train_cat, train_labels, encoders, scaler = preprocess_data(train_df, categorical_columns, numerical_columns, target_column, train_mode=True)
valid_num, valid_cat, valid_labels, _, _ = preprocess_data(valid_df, categorical_columns, numerical_columns, target_column, encoders, scaler, train_mode=False)
test_num, test_cat, test_labels, _, _ = preprocess_data(test_df, categorical_columns, numerical_columns, target_column, encoders, scaler, train_mode=False)

# 🔹 Create DataLoaders
def create_dataloader(num_tensor, cat_tensor, labels, batch_size):
    dataset = TensorDataset(num_tensor, cat_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 64  
train_loader = create_dataloader(train_num, train_cat, train_labels, batch_size)
valid_loader = create_dataloader(valid_num, valid_cat, valid_labels, batch_size)
test_loader = create_dataloader(test_num, test_cat, test_labels, batch_size)
