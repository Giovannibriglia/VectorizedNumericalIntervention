from typing import Dict, Tuple

import pandas as pd
import torch


def df_to_tensor_of_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
    # Convert each column of the DataFrame to a separate torch.tensor
    feature_tensors = [torch.tensor(df[col].values) for col in df.columns]

    # Stack the tensors together along a new dimension (rows as tensor of feature tensors)
    stacked_tensor = torch.stack(feature_tensors, dim=1)

    # Create the feature_name to index codification
    feature_name_index = {col: idx for idx, col in enumerate(df.columns)}

    return stacked_tensor, feature_name_index
