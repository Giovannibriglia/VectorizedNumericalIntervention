from typing import Tuple

import torch

from vni.base.estimator import BaseEstimator


class CovarianceEstimator(BaseEstimator):
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the covariance matrix of the data.

        Returns:
            torch.Tensor: Covariance matrix of the data.
        """
        if self.data is None:
            raise ValueError(
                "No data available. Please add data using `add_data` or `set_data`."
            )

        # Compute mean vector
        mean_vector = torch.mean(self.data, dim=0)

        # Center the data by subtracting the mean vector from each sample
        centered_data = self.data - mean_vector

        # Compute covariance matrix as (X^T * X) / (n - 1), where X is the centered data
        n_samples = self.data.shape[0]
        covariance_matrix = (centered_data.T @ centered_data) / (n_samples - 1)

        return mean_vector, covariance_matrix
