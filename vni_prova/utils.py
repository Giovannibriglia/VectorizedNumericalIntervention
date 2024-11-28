from typing import Dict, Tuple

import pandas as pd
import torch


def df_to_batched_tensor_with_index(
    df: pd.DataFrame, batch_size: int, **kwargs
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Converts a pandas DataFrame to a 3D PyTorch tensor with shape (batch_size, n_features, n_samples),
    and returns a dictionary mapping feature names to their indices.

    Args:
        df (pd.DataFrame): DataFrame with features as columns.
        batch_size (int): Desired batch size.
        kwargs: Additional arguments for torch.tensor creation, such as device.

    Returns:
        Tuple[torch.Tensor, Dict[str, int]]: 3D tensor of shape (batch_size, n_features, n_samples),
        and a dictionary with feature name to index mapping.
    """
    # Convert each column of the DataFrame to a separate torch tensor with additional kwargs
    feature_tensors = [
        torch.tensor(df[col].values, dtype=torch.float32, **kwargs)
        for col in df.columns
    ]
    # Stack the tensors together along a new dimension (columns as features)
    stacked_tensor = torch.stack(feature_tensors, dim=0)
    batched_tensor = stacked_tensor.unsqueeze(0).repeat(batch_size, 1, 1)

    # Create the feature_name to index mapping
    feature_name_index = {col: idx for idx, col in enumerate(df.columns)}

    return batched_tensor, feature_name_index


def get_memory_usage(tensor: torch.Tensor):
    # Memory usage in bytes
    memory_bytes = tensor.element_size() * tensor.nelement()

    # Memory usage in megabytes (optional)
    memory_megabytes = memory_bytes / (1024**2)

    return memory_megabytes
