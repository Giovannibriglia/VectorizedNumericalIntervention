from typing import List, Tuple

import torch
from torch.distributions import MultivariateNormal

from vni.base.estimator import BaseParametricEstimator


class MeanCovarianceEstimator(BaseParametricEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super(MeanCovarianceEstimator, self).__init__(
            X_indices, Y_indices, intervention_indices
        )

    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X_query.shape[0]

        if X_do is None:
            mu, sigma = (
                self.prior_parameters
                if self.intervention_indices is None
                else self.prior_parameters_after_interventions
            )
        else:
            pass
            " self._compute_prior_parameters(new_XY) "

        # Validate dimensions
        assert mu.dim() == 1, f"Expected mu to have 1 dimension, got {mu.shape}"
        assert (
            sigma.dim() == 2
        ), f"Expected sigma to have 2 dimensions, got {sigma.shape}"
        assert (
            mu.shape[0] == sigma.shape[0] == sigma.shape[1]
        ), f"Mismatch in dimensions: mu.shape = {mu.shape}, sigma.shape = {sigma.shape}"

        (
            mu_target_given_obs,
            Sigma_target_given_obs,
        ) = self._compute_conditional_parameters(mu, sigma, X_query, batch_size)

        # Use MultivariateNormal for PDF evaluation
        mvn = MultivariateNormal(
            loc=mu_target_given_obs, covariance_matrix=Sigma_target_given_obs
        )

        pdf, values = self._evaluate_Y(Y_query, mvn, n_samples, batch_size)

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

        return mean, covariance

    def _compute_conditional_parameters(self, mu, sigma, X_query, batch_size):
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

        return mu_target_given_obs, Sigma_target_given_obs

    def _evaluate_Y(self, Y_query, dist, n_samples, batch_size):

        Y_values = self._define_Y_values(Y_query, n_samples, batch_size)

        if Y_values.dim() == 3:
            # Preallocate log_pdf_Y
            log_pdf_Y = torch.zeros_like(
                Y_values, device=self.XY_prior.device
            )  # [batch_size, n_target_features, n_samples]

            for i in range(n_samples):
                # Extract the i-th sample for all batches
                Y_sample = Y_values[:, :, i]  # Shape: [batch_size, n_target_features]

                # Ensure proper shape for log_prob
                log_pdf_Y[:, :, i] = dist.log_prob(Y_sample)[
                    :, None
                ]  # Shape: [batch_size, 1]
        elif Y_values.dim() == 2:
            # Use provided Y_query values
            Y_values = Y_query.clone()  # [batch_size, n_target_features]
            log_pdf_Y = (
                dist.log_prob(Y_values).unsqueeze(-1).unsqueeze(-1)
            )  # [batch_size, n_target_features, 1]

        else:
            raise ValueError("Y_values is not well defined")

        # Convert log PDF to actual PDF
        pdf_Y = torch.exp(
            log_pdf_Y
        )  # [batch_size, n_target_features, n_samples] if no Y_query; else [batch_size, n_target_features, 1]

        return pdf_Y, Y_values
