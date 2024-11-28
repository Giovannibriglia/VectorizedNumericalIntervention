from typing import List

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from vni.vni2 import VNI


def benchmark_dataframe(
    df: pd.DataFrame, target_column: str, intervention_columns: List, **kwargs
):
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if_plot = kwargs.get("if_plot", False)

    # Separate observed X data and Y target based on specified target column
    observed_X = torch.tensor(
        df.drop(columns=[target_column]).values,
        dtype=torch.float16,
        device="cpu",
    )

    observed_Y = torch.tensor(
        df[target_column].values, dtype=torch.float16, device="cpu"
    ).view(-1, 1)

    # Identify indices for intervention columns
    intervention_indices = [
        df.drop(columns=[target_column]).columns.get_loc(col)
        for col in intervention_columns
    ]

    # Initialize VNI object with the observed data
    vni = VNI(observed_X, observed_Y, intervention_indices)

    # Initialize lists to store predictions and true values
    predictions = []
    true_values = []

    # Evaluate predictor on each row in df
    for n, row in tqdm(df.iterrows(), total=len(df)):
        # Prepare X and Y values as tensors on the specified device
        x_value = torch.tensor(
            row.drop(target_column).values, dtype=torch.float16, device=device
        )
        y_true = torch.tensor(row[target_column], dtype=torch.float16, device=device)

        if intervention_indices:
            # Get intervention values as tensors
            intervention_values = torch.tensor(
                row.iloc[intervention_indices].values,
                dtype=torch.float16,
                device=device,
            )
        else:
            intervention_values = None

        # Compute conditional PDF P(Y | X = x_value) with intervention values
        conditional_pdf, y_values = vni.compute_conditional_pdf(
            x_value,
        )

        # Get the prediction from the maximum of the PDF
        y_pred = max(conditional_pdf.cpu().numpy())
        predictions.append(y_pred)
        true_values.append(y_true.item())

        # Plot PDF with intervention
        if if_plot:
            vni.plot_pdf(
                conditional_pdf,
                y_values,
                y_true,
                do=True if intervention_values is not None else False,
            )

    # Convert lists to numpy arrays for metric calculations
    predictions = torch.tensor(predictions, dtype=torch.float32)
    true_values = torch.tensor(true_values, dtype=torch.float32)

    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    metrics = {"MSE": float(mse), "MAE": float(mae), "R2": r2}

    return metrics
