from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from vni.usage_vni import VNI


def yaml_to_dict(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load for security
    return data


def get_max_pdf_values(
    predicted_pdf: torch.Tensor, y_values: torch.Tensor
) -> torch.Tensor:
    """
    Extracts the values of y_values with the maximum predicted PDF for each feature in each batch.

    Args:
        predicted_pdf (torch.Tensor): Tensor of shape [batch_size, n_featY, n_samplesY]
                                      representing the predicted PDF values.
        y_values (torch.Tensor): Tensor of shape [batch_size, n_featY, n_samplesY]
                                 representing the possible y values.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, n_featY] containing the y_values
                      corresponding to the maximum predicted PDF for each feature in each batch.
    """
    # Find the index of the maximum PDF for each feature in each batch
    max_pdf_indices = predicted_pdf.argmax(dim=-1)  # Shape: [batch_size, n_featY]

    # Get batch indices for advanced indexing
    batch_indices = torch.arange(y_values.shape[0], device=y_values.device).unsqueeze(
        1
    )  # Shape: [batch_size, 1]

    # Get feature indices for advanced indexing
    feat_indices = torch.arange(y_values.shape[1], device=y_values.device).unsqueeze(
        0
    )  # Shape: [1, n_featY]

    # Use advanced indexing to gather the y_values
    max_y_values = y_values[
        batch_indices, feat_indices, max_pdf_indices
    ]  # Shape: [batch_size, n_featY]

    return max_y_values


def benchmarking_df(
    df: pd.DataFrame,
    target_features: List[str],
    intervention_features: List[str] = None,
    batch_size: int = 64,
    n_samples_y: int = 1024,
    show_res: bool = False,
    estimator_config: Dict = None,
):
    Y_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col in target_features
    ]
    X_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col not in target_features and col not in intervention_features
    ]
    intervention_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col in intervention_features
    ]

    XY_prior_tensor = torch.tensor(df.values, dtype=torch.float32, device="cuda")

    estimator_config = {"estimator": "mean_covariance"}

    vni = VNI(
        XY_prior_tensor.T, estimator_config, X_indices, Y_indices, intervention_indices
    )
    y_true = XY_prior_tensor[:, Y_indices]
    y_pred = np.zeros_like(y_true.cpu())  # [n_samples_data, n_features_y]

    for t in tqdm(range(batch_size, XY_prior_tensor.shape[0], batch_size)):
        true_values = y_true[t - batch_size : t]

        X_query = XY_prior_tensor[t - batch_size : t, X_indices]
        Y_query = None  # ( true_values.clone())  # TODO: if statement for computing density function or density value
        X_do = XY_prior_tensor[t - batch_size : t, intervention_indices]

        pdf, y_values = vni.query(
            X_query,
            Y_query=Y_query,
            X_do=X_do,
            n_samples=n_samples_y,
        )  # [batch_size, n_target_features, n_samples]

        if show_res:
            vni.plot_result(pdf, y_values, true_values)

        y_pred[t - batch_size : t, :] = get_max_pdf_values(pdf, y_values).cpu().numpy()

    return y_true.cpu(), y_pred
