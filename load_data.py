import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import defaultdict
import optuna

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

# 🔹 One-Hot Encode Data
def one_hot_encode(df, one_hot_columns):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[one_hot_columns])
    return torch.tensor(one_hot_encoded, dtype=torch.float32), encoder

# 🔹 Preprocess Data
def preprocess_data(df, categorical_columns, one_hot_columns, numerical_columns, target_column, encoders=None, scaler=None, one_hot_encoder=None, train_mode=True):
    if train_mode:
        encoders = fit_label_encoders(df, categorical_columns)
        scaler = StandardScaler().fit(df[numerical_columns])
        one_hot_tensor, one_hot_encoder = one_hot_encode(df, one_hot_columns)
    else:
        one_hot_tensor = torch.tensor(one_hot_encoder.transform(df[one_hot_columns]), dtype=torch.float32)

    cat_tensor = transform_categorical(df, categorical_columns, encoders)
    num_tensor = torch.tensor(scaler.transform(df[numerical_columns]), dtype=torch.float32)
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return num_tensor, cat_tensor, one_hot_tensor, target_tensor, encoders, scaler, one_hot_encoder

# 🔹 Define Columns
categorical_columns = ["cat1", "cat2", "cat3"]  
one_hot_columns = ["one_hot1", "one_hot2"]
numerical_columns = ["num1", "num2", "num3", "num4"]  
target_column = "label"

# 🔹 Preprocess Data
train_df = load_data("train.csv")
valid_df = load_data("valid.csv")
test_df = load_data("test.csv")
train_num, train_cat, train_one_hot, train_labels, encoders, scaler, one_hot_encoder = preprocess_data(train_df, categorical_columns, one_hot_columns, numerical_columns, target_column, train_mode=True)
valid_num, valid_cat, valid_one_hot, valid_labels, _, _, _ = preprocess_data(valid_df, categorical_columns, one_hot_columns, numerical_columns, target_column, encoders, scaler, one_hot_encoder, train_mode=False)
test_num, test_cat, test_one_hot, test_labels, _, _, _ = preprocess_data(test_df, categorical_columns, one_hot_columns, numerical_columns, target_column, encoders, scaler, one_hot_encoder, train_mode=False)

# 🔹 Create DataLoaders
def create_dataloader(num_tensor, cat_tensor, one_hot_tensor, labels, batch_size):
    dataset = TensorDataset(num_tensor, cat_tensor, one_hot_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 64  
train_loader = create_dataloader(train_num, train_cat, train_one_hot, train_labels, batch_size)
valid_loader = create_dataloader(valid_num, valid_cat, valid_one_hot, valid_labels, batch_size)
test_loader = create_dataloader(test_num, test_cat, test_one_hot, test_labels, batch_size)
