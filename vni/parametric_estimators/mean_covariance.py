from typing import Tuple

import torch
from torch.distributions import MultivariateNormal

from vni.base.estimator import BaseParametricEstimator


class MeanCovarianceEstimator(BaseParametricEstimator):
    def __init__(
        self,
    ):
        super(MeanCovarianceEstimator, self).__init__()

    def _predict(
        self, X_values: torch.Tensor, Y_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param X_values: [batch_size, n_features_X+n_features_X_do, n_samples_x]
        :param Y_values: [batch_size, n_features_X+n_features_X_do, 1]
        :return: pdf and Y_values. [batch_size, n_target_features, n_samples_Y_values], [batch_size, n_target_features, n_samples_Y_values]
        """
        batch_size = X_values.shape[0]

        mu, sigma = self.prior_parameters

        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(
                batch_size, -1
            )  # Shape: [batch_size, n_features]
            sigma = sigma.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # Shape: [batch_size, n_features, n_features]

        (
            mu_target_given_obs,
            Sigma_target_given_obs,
        ) = self._compute_conditional_parameters(
            mu, sigma, X_values
        )  # [batch_size]

        # Use MultivariateNormal for PDF evaluation
        mvn = MultivariateNormal(
            loc=mu_target_given_obs, covariance_matrix=Sigma_target_given_obs
        )

        pdf, values = self._evaluate_Y(Y_values, mvn)

        return pdf, values

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

        covariance = self._ensure_covariance_matrix(covariance)

        return mean, covariance

    def _compute_conditional_parameters(self, mu, sigma, X_values):
        """
        Compute conditional mean and covariance of target features given observed values.

        Args:
            mu (torch.Tensor): Mean vector, shape [batch_size, n_features].
            sigma (torch.Tensor): Covariance matrix, shape [batch_size, n_features, n_features].
            X_values (torch.Tensor): Observed/intervened values for X,
                                     shape [batch_size, n_features_X + n_features_X_do, n_samples_X].

        Returns:
            mu_target_given_obs (torch.Tensor): Conditional mean, shape [batch_size, n_target_features].
            Sigma_target_given_obs (torch.Tensor): Conditional covariance, shape [batch_size, n_target_features, n_target_features].
        """
        # Validate indices
        assert all(
            0 <= idx < mu.shape[1] for idx in self.Y_indices
        ), "Y_indices out of bounds"
        assert all(
            0 <= idx < mu.shape[1] for idx in self.all_X_indices
        ), "X_indices out of bounds"

        # Partition the mean vector
        mu_target = mu[:, self.Y_indices]  # Shape: [batch_size, n_target_features]
        mu_obs = mu[:, self.all_X_indices]  # Shape: [batch_size, n_obs_features]

        # Partition the covariance matrix
        sigma_aa = sigma[:, self.Y_indices][
            :, :, self.Y_indices
        ]  # [batch_size, n_target_features, n_target_features]
        sigma_bb = sigma[:, self.all_X_indices][
            :, :, self.all_X_indices
        ]  # [batch_size, n_obs_features, n_obs_features]
        sigma_ab = sigma[:, self.Y_indices][
            :, :, self.all_X_indices
        ]  # [batch_size, n_target_features, n_obs_features]

        # Add tolerance to sigma_bb diagonal to prevent singularity
        sigma_bb = sigma_bb + 1e-8 * torch.eye(
            sigma_bb.shape[-1], device=sigma_bb.device
        ).expand_as(sigma_bb)

        # Compute the inverse of sigma_bb
        inv_sigma_bb = torch.linalg.inv(
            sigma_bb
        )  # Shape: [batch_size, n_obs_features, n_obs_features]

        # Aggregate X_values across samples (e.g., take mean across the last dimension)
        X_mean = X_values.mean(dim=-1)  # Shape: [batch_size, n_obs_features]

        # Compute the deviation of observed/intervened values from their mean
        obs_diff = X_mean - mu_obs  # Shape: [batch_size, n_obs_features]

        # Compute the conditional mean of target features given observed values
        mu_target_given_obs = mu_target + torch.matmul(
            sigma_ab, torch.matmul(inv_sigma_bb, obs_diff.unsqueeze(-1))
        ).squeeze(
            -1
        )  # Shape: [batch_size, n_target_features]

        # Compute the conditional covariance of target features given observed values
        Sigma_target_given_obs = sigma_aa - torch.matmul(
            sigma_ab, torch.matmul(inv_sigma_bb, sigma_ab.transpose(-1, -2))
        )  # Shape: [batch_size, n_target_features, n_target_features]

        # Ensure covariance matrix is valid (e.g., symmetric, positive semi-definite)
        Sigma_target_given_obs = self._ensure_covariance_matrix(Sigma_target_given_obs)

        return mu_target_given_obs, Sigma_target_given_obs

    def _evaluate_Y(self, Y_query, dist):
        """
        Evaluate the PDF of the target features given the query values.
        :param Y_query: [batch_size, n_target_features, n_samples_y]
        :param dist: event_shape: [n_target_features], batch_shape: [batch_size]
        :return:
        """
        # Rearrange Y_query for batch processing: [batch_size, n_target_features, n_samples_y] -> [batch_size, n_samples_y, n_target_features]
        Y_query_reshaped = Y_query.permute(
            0, 2, 1
        )  # Shape: [batch_size, n_samples_y, n_target_features]

        # Flatten the batch and sample dimensions: [batch_size * n_samples_y, n_target_features]
        Y_query_flat = Y_query_reshaped.reshape(-1, Y_query.shape[1])

        # Extract parameters from the original distribution
        loc = dist.loc  # Shape: [batch_size, n_target_features]
        covariance_matrix = (
            dist.covariance_matrix
        )  # Shape: [batch_size, n_target_features, n_target_features]

        # Repeat the distribution parameters for each sample
        loc_repeated = loc.repeat_interleave(
            Y_query.shape[2], dim=0
        )  # Shape: [batch_size * n_samples_y, n_target_features]
        covariance_matrix_repeated = covariance_matrix.repeat_interleave(
            Y_query.shape[2], dim=0
        )  # Shape: [batch_size * n_samples_y, n_target_features, n_target_features]

        # Create a new distribution with expanded parameters
        expanded_dist = torch.distributions.MultivariateNormal(
            loc=loc_repeated, covariance_matrix=covariance_matrix_repeated
        )

        # Compute log_prob for all samples at once: [batch_size * n_samples_y]
        log_prob_flat = expanded_dist.log_prob(Y_query_flat)

        # Reshape log_prob back to [batch_size, n_target_features, n_samples_y]
        log_pdf_Y = log_prob_flat.view(Y_query.shape[0], Y_query.shape[2], -1).permute(
            0, 2, 1
        )

        # Convert log PDF to actual PDF
        pdf_Y = torch.exp(
            log_pdf_Y
        )  # Shape: [batch_size, n_target_features, n_samples_y]

        return pdf_Y, Y_query

    def _ensure_covariance_matrix(
        self, covariance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Validate and ensure the covariance matrix is symmetric, positive definite,
        and well-conditioned for a batch of covariance matrices.

        Args:
            covariance_matrix (torch.Tensor): Covariance matrix of shape (batch_size, n, n).

        Returns:
            torch.Tensor: Validated and corrected covariance matrix with the same shape as the input.
        """
        # Save the initial shape to ensure it is preserved
        initial_shape = covariance_matrix.shape

        # Ensure symmetry
        if not torch.allclose(
            covariance_matrix, covariance_matrix.transpose(-1, -2), atol=1e-5
        ):
            covariance_matrix = 0.5 * (
                covariance_matrix + covariance_matrix.transpose(-1, -2)
            )

        # Add small value to the diagonal (regularization)
        epsilon = 1e-5
        eye = torch.eye(covariance_matrix.size(-1), device=covariance_matrix.device)
        covariance_matrix += epsilon * eye

        # Ensure positive diagonal values
        diag = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1)
        diag = torch.clamp(
            diag, min=1e-6
        )  # Replace negatives with small positive values
        covariance_matrix = covariance_matrix.clone()
        covariance_matrix.diagonal(dim1=-2, dim2=-1).copy_(diag)

        # Check for positive definiteness
        try:
            torch.linalg.cholesky(covariance_matrix)
        except RuntimeError:
            # Handle ill-conditioned matrices
            # print("Matrix is not positive definite. Applying fallback.")
            average_variance = torch.mean(diag)
            fallback = average_variance * torch.eye(
                covariance_matrix.size(-1), device=covariance_matrix.device
            )
            fallback = fallback.expand_as(covariance_matrix)  # Preserve original shape
            covariance_matrix = fallback

        # Ensure the output shape matches the input shape
        assert (
            covariance_matrix.shape == initial_shape
        ), f"covariance matrix has different shape: {covariance_matrix.shape}"

        return covariance_matrix
