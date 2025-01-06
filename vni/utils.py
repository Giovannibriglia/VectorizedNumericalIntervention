from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from tqdm import tqdm

from vni.vni import VNI


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
    estimator_config: Dict,
    target_features: List[str],
    intervention_features: List[str] = None,
    batch_size: int = 64,
    n_samples_x: int = 1024,
    n_samples_y: int = 1024,
    show_res: bool = False,
    density_value: bool = False,
    device: str = "cuda",
):
    vni, XY_prior_tensor, X_indices, Y_indices, intervention_indices = setup_vni(
        df, target_features, intervention_features, estimator_config, device
    )
    y_true = XY_prior_tensor[:, Y_indices]
    y_pred = np.zeros_like(y_true.cpu())  # [n_samples_data, n_features_y]

    for t in tqdm(range(batch_size, XY_prior_tensor.shape[0], batch_size)):
        true_values = y_true[t - batch_size : t]

        X_query = XY_prior_tensor[t - batch_size : t, X_indices]
        Y_query = true_values if density_value else None
        X_do = XY_prior_tensor[t - batch_size : t, intervention_indices]

        pdf, y_values = vni.query(
            X_query,
            Y_query=Y_query,
            X_do=X_do,
            n_samples_x=n_samples_x,
            n_samples_y=n_samples_y,
        )  # [batch_size, n_target_features, n_samples]

        if show_res:
            vni.plot_result(pdf, y_values, true_values)

        y_pred[t - batch_size : t, :] = get_max_pdf_values(pdf, y_values).cpu().numpy()

    return y_true.cpu(), y_pred


def setup_vni(
    df: pd.DataFrame,
    target_features: List[str],
    intervention_features: List[str] = None,
    estimator_config: Dict = None,
    device: str = "cuda",
):

    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

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

    XY_prior_tensor = torch.tensor(df.values, dtype=torch.float32, device=device)

    vni = VNI(XY_prior_tensor.T, estimator_config)

    vni.set_indices(X_indices, Y_indices, intervention_indices)

    return vni, XY_prior_tensor, X_indices, Y_indices, intervention_indices


def single_query(
    vni: VNI,
    X_query: torch.Tensor,
    Y_query: torch.Tensor = None,
    X_do: torch.Tensor = None,
    batch_size_max: int = 64,
    show_res: bool = False,
):
    n_samples = X_query.shape[0]
    if X_query.shape[0] > batch_size_max:
        y_pred = np.zeros((n_samples, 1))  # [n_samples_data, n_features_y]

        for t in range(batch_size_max, X_query.shape[0], batch_size_max):

            X_query_new = X_query[t - batch_size_max : t]
            Y_query_new = None if Y_query is None else Y_query[t - batch_size_max : t]
            X_do_new = None if X_do is None else X_do[t - batch_size_max : t]

            pdf, y_values = vni.query(
                X_query_new,
                Y_query=Y_query_new,
                X_do=X_do_new,
                n_samples_x=n_samples,
                n_samples_y=n_samples,
            )  # [batch_size, n_target_features, n_samples]

            y_pred[t - batch_size_max : t, :] = (
                get_max_pdf_values(pdf, y_values).cpu().numpy()
            )
    else:
        pdf, y_values = vni.query(
            X_query,
            Y_query=Y_query,
            X_do=X_do,
            n_samples_x=n_samples,
            n_samples_y=n_samples,
        )  # [X_query.shape[0], n_target_features, n_samples]

        y_pred = get_max_pdf_values(pdf, y_values).cpu().numpy()

    if show_res:
        vni.plot_result(pdf, y_values, Y_query)

    return y_pred


def print_metrics(y_pred, y_true):
    """
    Prints MAE, MSE, and R^2 metrics for the given predictions and ground truth.

    Args:
        y_pred (torch.Tensor or numpy.ndarray): Predicted values
        y_true (torch.Tensor or numpy.ndarray): Ground truth values
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    # Ensure both are 1D arrays for compatibility with metrics
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")

    # Handle binary classification: Apply threshold to y_pred
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)  # For multi-class
    else:
        y_pred = (y_pred >= 0.5).astype(int)  # For binary

    y_true = y_true.flatten().astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Classification Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix:\n{conf_matrix}")
