from typing import Tuple

import torch
from torch.distributions import MultivariateNormal

from vni.base.estimator import BaseParametricEstimator


class MeanCovarianceEstimator(BaseParametricEstimator):
    def __init__(self):
        super(MeanCovarianceEstimator, self).__init__()

    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X_query.shape[0]

        # Extract prior parameters
        (
            mu,
            sigma,
        ) = self.prior_parameters  # mu: [n_features], sigma: [n_features, n_features]

        # Validate dimensions
        assert mu.dim() == 1, f"Expected mu to have 1 dimension, got {mu.shape}"
        assert (
            sigma.dim() == 2
        ), f"Expected sigma to have 2 dimensions, got {sigma.shape}"
        assert (
            mu.shape[0] == sigma.shape[0] == sigma.shape[1]
        ), f"Mismatch in dimensions: mu.shape = {mu.shape}, sigma.shape = {sigma.shape}"

        # Expand mu and sigma for broadcasting
        mu_expanded = mu.unsqueeze(0)  # Shape: [1, n_features]
        sigma_expanded = sigma.unsqueeze(0)  # Shape: [1, n_features, n_features]

        # Validate indices
        assert all(
            0 <= idx < mu.shape[0] for idx in self.Y_indices
        ), "Y_indices out of bounds"
        assert all(
            0 <= idx < mu.shape[0] for idx in self.X_indices
        ), "X_indices out of bounds"

        # Partition the mean vector
        mu_target = mu_expanded[:, self.Y_indices]  # Shape: [1, n_target_features]
        mu_obs = mu_expanded[:, self.X_indices]  # Shape: [1, n_obs_features]

        # Partition the covariance matrix
        sigma_aa = sigma_expanded[:, self.Y_indices][
            :, :, self.Y_indices
        ]  # [1, n_target_features, n_target_features]
        sigma_bb = sigma_expanded[:, self.X_indices][
            :, :, self.X_indices
        ]  # [1, n_obs_features, n_obs_features]
        sigma_ab = sigma_expanded[:, self.Y_indices][
            :, :, self.X_indices
        ]  # [1, n_target_features, n_obs_features]

        # Add tolerance to sigma_bb diagonal to prevent singularity
        sigma_bb = sigma_bb + 1e-8 * torch.eye(
            sigma_bb.shape[-1], device=sigma_bb.device
        ).expand_as(sigma_bb)

        # Compute the inverse of sigma_bb
        inv_sigma_bb = torch.linalg.inv(
            sigma_bb
        )  # Shape: [1, n_obs_features, n_obs_features]

        # Calculate the deviation of observed values from their mean
        obs_diff = (X_query - mu_obs).unsqueeze(
            -1
        )  # Shape: [batch_size, n_obs_features, 1]

        # Compute the conditional mean of target features given observed values
        mu_target_given_obs = (
            mu_target.unsqueeze(0)
            + torch.matmul(sigma_ab, torch.matmul(inv_sigma_bb, obs_diff))
        ).squeeze(
            -1
        )  # Shape: [batch_size, n_target_features]

        # Compute the conditional covariance of target features given observed values
        Sigma_target_given_obs = sigma_aa - torch.matmul(
            sigma_ab, torch.matmul(inv_sigma_bb, sigma_ab.transpose(-1, -2))
        )  # Shape: [1, n_target_features, n_target_features]

        # Expand covariance to match the batch size
        Sigma_target_given_obs = Sigma_target_given_obs.expand(batch_size, -1, -1)

        # Use MultivariateNormal for PDF evaluation
        mvn = MultivariateNormal(
            loc=mu_target_given_obs, covariance_matrix=Sigma_target_given_obs
        )

        if Y_query is None:
            # Calculate min and max for each feature in Y
            Y_min = torch.min(
                self.XY_prior[self.Y_indices, :], dim=1
            ).values  # Shape: [n_target_features]
            Y_max = torch.max(
                self.XY_prior[self.Y_indices, :], dim=1
            ).values  # Shape: [n_target_features]

            # Create a linspace template
            linspace_template = torch.linspace(
                0, 1, n_samples, device=X_query.device
            ).unsqueeze(
                0
            )  # [1, n_samples]

            # Scale the linspace to each feature's range
            Y_values = Y_min.unsqueeze(1) + linspace_template * (
                Y_max - Y_min
            ).unsqueeze(
                1
            )  # [n_target_features, n_samples]

            # Expand Y_values to match batch size
            Y_values = Y_values.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, n_target_features, n_samples]

            # Preallocate log_pdf_Y
            log_pdf_Y = torch.zeros_like(
                Y_values, device=X_query.device
            )  # [batch_size, n_target_features, n_samples]

            for i in range(n_samples):
                # Extract the i-th sample for all batches
                Y_sample = Y_values[:, :, i]  # Shape: [batch_size, n_target_features]

                # Ensure proper shape for log_prob
                log_pdf_Y[:, :, i] = mvn.log_prob(Y_sample)[
                    :, None
                ]  # Shape: [batch_size, 1]

        else:
            # Use provided Y_query values
            Y_values = Y_query  # [batch_size, n_target_features]
            log_pdf_Y = (
                mvn.log_prob(Y_values).unsqueeze(-1).unsqueeze(-1)
            )  # [batch_size, n_target_features, 1]

        # Convert log PDF to actual PDF
        pdf_Y = torch.exp(
            log_pdf_Y
        )  # [batch_size, n_target_features, n_samples] if no Y_query; else [batch_size, n_target_features, 1]

        return pdf_Y, Y_values

    def _compute_prior_parameters(self, XY: torch.Tensor):
        """
        Compute the mean and covariance of the target (Y) given observed variables (X).

        Args:
            XY (torch.Tensor): Observed and Target variables of shape [n_features_X + n_features_Y, n_samples].
        """

        # Compute the mean vector along the sample dimension
        mean = XY.mean(dim=1)  # [n_features_X + n_features_Y]

        # Compute the covariance matrix
        centered_data = XY - mean.unsqueeze(
            1
        )  # Center the data along the sample dimension
        covariance = (centered_data @ centered_data.T) / (XY.shape[1] - 1)  # Covariance

        return mean, covariance
