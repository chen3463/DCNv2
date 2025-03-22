import shap
import torch
import numpy as np
import pandas as pd

def feature_importance(model, data_loader, feature_names, emb_feature_slices, onehot_feature_slices, device):
    model.eval()
    numerical, categorical_emb, categorical_onehot, _ = next(iter(data_loader))
    input_data = torch.cat([numerical, categorical_emb.float(), categorical_onehot], dim=1)
    input_data_np = input_data.cpu().numpy()

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        num = x_tensor[:, :numerical.shape[1]]
        emb = x_tensor[:, numerical.shape[1]:numerical.shape[1] + categorical_emb.shape[1]].long()
        onehot = x_tensor[:, numerical.shape[1] + categorical_emb.shape[1]:]
        return model(num, emb, onehot).squeeze().cpu().detach().numpy()

    masker = shap.maskers.Independent(input_data_np)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(input_data_np)
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap_values = []

    mean_shap_values.extend(mean_shap[:numerical.shape[1]])  # numerical
    for start, end in emb_feature_slices:                   # embeddings
        mean_shap_values.append(mean_shap[start:end].mean())
    for start, end in onehot_feature_slices:                # one-hot
        mean_shap_values.append(mean_shap[start:end].mean())

    df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap_value': mean_shap_values})
    return df.sort_values(by='mean_abs_shap_value', ascending=False)
