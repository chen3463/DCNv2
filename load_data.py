import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# 🔹 Data Loader with Unknown Category Handling
class DataLoaderWrapper:
    def __init__(self, train_path, valid_path, test_path, numerical_columns, categorical_columns, one_hot_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.one_hot_columns = one_hot_columns

        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.test_df = pd.read_csv(test_path)

        self.encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # ✅ Fixed

        self._fit_transformers()

    def _fit_transformers(self):
        # 🔹 Label Encoding with Unseen Category Handling
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col].astype(str))

            # Append "unknown" as a new category
            le_classes = list(le.classes_)
            le_classes.append("unknown")
            le.classes_ = np.array(le_classes)

            # Transform validation & test data (map unknowns)
            self.valid_df[col] = self.valid_df[col].astype(str).apply(lambda x: x if x in le.classes_ else "unknown")
            self.valid_df[col] = le.transform(self.valid_df[col])

            self.test_df[col] = self.test_df[col].astype(str).apply(lambda x: x if x in le.classes_ else "unknown")
            self.test_df[col] = le.transform(self.test_df[col])

            self.encoders[col] = le

        # 🔹 Standard Scaling for Numerical Features
        if self.numerical_columns:
            self.train_df[self.numerical_columns] = self.scaler.fit_transform(self.train_df[self.numerical_columns])
            self.valid_df[self.numerical_columns] = self.scaler.transform(self.valid_df[self.numerical_columns])
            self.test_df[self.numerical_columns] = self.scaler.transform(self.test_df[self.numerical_columns])

        # 🔹 One-Hot Encoding (Fixed Unknown Handling)
        if self.one_hot_columns:
            self.one_hot_enc.fit(self.train_df[self.one_hot_columns])

    def get_dataloader(self, df, selected_features, batch_size=32):
        num_features = torch.tensor(df[[col for col in self.numerical_columns if col in selected_features]].values, dtype=torch.float32)
        cat_features = torch.tensor(df[[col for col in self.categorical_columns if col in selected_features]].values, dtype=torch.long)

        # 🔹 Fixed One-Hot Encoding for Unknown Categories
        if self.one_hot_columns:
            try:
                one_hot_features = self.one_hot_enc.transform(df[self.one_hot_columns])
            except:
                one_hot_features = pd.get_dummies(df[self.one_hot_columns]).values  # Fallback
            one_hot_features = torch.tensor(one_hot_features, dtype=torch.float32)
        else:
            one_hot_features = torch.empty((len(df), 0))  # No one-hot features

        labels = torch.tensor(df['target'].values, dtype=torch.float32)

        dataset = TensorDataset(num_features, cat_features, one_hot_features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
