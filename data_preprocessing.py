import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder

def generate_dummy_data(N, num_numerical, num_categorical_emb, onehot_cardinalities):
    numerical_data = torch.randn(N, num_numerical)
    categorical_emb_data = torch.randint(0, 10, (N, num_categorical_emb))
    onehot_size = sum(onehot_cardinalities)
    categorical_onehot_data = torch.randint(0, 2, (N, onehot_size))
    labels = torch.randint(0, 2, (N, 1)).float()

    split_idx = int(N * 0.8)
    train_loader = DataLoader(TensorDataset(
        numerical_data[:split_idx], categorical_emb_data[:split_idx], categorical_onehot_data[:split_idx], labels[:split_idx]
    ), batch_size=32, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        numerical_data[split_idx:], categorical_emb_data[split_idx:], categorical_onehot_data[split_idx:], labels[split_idx:]
    ), batch_size=32)

    return train_loader, val_loader, numerical_data, categorical_emb_data, categorical_onehot_data, labels


def build_feature_names(num_numerical, emb_cat_cardinalities, onehot_cat_names, onehot_cat_dims):
    num_feature_names = [f"num_{i}" for i in range(num_numerical)]
    emb_feature_names, emb_feature_slices = [], []
    start = num_numerical
    for i in range(len(emb_cat_cardinalities)):
        emb_feature_names.append(f"emb_cat_{i}")
        emb_feature_slices.append((start, start + 1))
        start += 1
    onehot_feature_names, onehot_feature_slices = [], []
    for i, name in enumerate(onehot_cat_names):
        dim = onehot_cat_dims[i]
        onehot_feature_names.append(name)
        onehot_feature_slices.append((start, start + dim))
        start += dim
    feature_names = num_feature_names + emb_feature_names + onehot_feature_names
    return feature_names, emb_feature_slices, onehot_feature_slices
