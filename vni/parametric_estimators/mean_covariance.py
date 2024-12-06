from typing import List, Tuple

import torch

from torch.distributions import MultivariateNormal, Uniform

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

        if self.intervention_indices is None:
            if X_do is None:
                mu, sigma = self.prior_parameters
            else:
                raise ValueError(
                    "intervention indices are not initialized, make sure they are"
                )
        else:
            if X_do is None:
                mu, sigma = self.prior_parameters_after_interventions
            else:
                new_XY = self.XY_prior.clone()
                if batch_size == 1:
                    new_XY[:, self.intervention_indices] = X_do

                    mu, sigma = self._compute_prior_parameters(new_XY)

                else:
                    mu = torch.zeros(
                        (batch_size, self.XY_prior.shape[0]), device=X_query.device
                    )  # Shape: [batch_size, n_features]
                    sigma = torch.zeros(
                        (batch_size, self.XY_prior.shape[0], self.XY_prior.shape[0]),
                        device=X_query.device,
                    )  # Shape: [batch_size, n_features, n_features]

                    y_min = torch.min(
                        self.XY_prior[self.intervention_indices, :], dim=1
                    ).values  # [n_features_X_do]
                    y_max = torch.max(
                        self.XY_prior[self.intervention_indices, :], dim=1
                    ).values  # [n_features_X_do]

                    new_XY = new_XY.expand(
                        batch_size, -1, -1
                    )  # Shape: [batch_size, n_features, n_samples_data]

                    for i in range(batch_size):
                        new_samples = Uniform(y_min, y_max).sample((new_XY.shape[2],))
                        new_XY[i, self.intervention_indices, :] = new_samples.T

                        mu_batch, sigma_batch = self._compute_prior_parameters(
                            new_XY[i, :, :]
                        )
                        mu[i] = mu_batch
                        sigma[i] = sigma_batch

        if mu.dim() == 1:
            mu = mu.unsqueeze(0)  # Shape: [1, n_features]
            sigma = sigma.unsqueeze(0)  # Shape: [1, n_features, n_features]

        (
            mu_target_given_obs,
            Sigma_target_given_obs,
        ) = self._compute_conditional_parameters(
            mu, sigma, X_query
        )  # [batch_size]

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

        covariance = self._ensure_covariance_matrix(covariance)

        return mean, covariance

    def _compute_conditional_parameters(self, mu, sigma, X_query):

        # Validate indices
        assert all(
            0 <= idx < mu.shape[1] for idx in self.Y_indices
        ), "Y_indices out of bounds"
        assert all(
            0 <= idx < mu.shape[1] for idx in self.X_indices
        ), "X_indices out of bounds"

        # Partition the mean vector
        mu_target = mu[:, self.Y_indices]  # Shape: [batch_size, n_target_features]
        mu_obs = mu[:, self.X_indices]  # Shape: [batch_size, n_obs_features]

        # Partition the covariance matrix
        sigma_aa = sigma[:, self.Y_indices][
            :, :, self.Y_indices
        ]  # [batch_size, n_target_features, n_target_features]
        sigma_bb = sigma[:, self.X_indices][
            :, :, self.X_indices
        ]  # [batch_size, n_obs_features, n_obs_features]
        sigma_ab = sigma[:, self.Y_indices][
            :, :, self.X_indices
        ]  # [batch_size, n_target_features, n_obs_features]

        # Add tolerance to sigma_bb diagonal to prevent singularity
        sigma_bb = sigma_bb + 1e-8 * torch.eye(
            sigma_bb.shape[-1], device=sigma_bb.device
        ).expand_as(sigma_bb)

        # Compute the inverse of sigma_bb
        inv_sigma_bb = torch.linalg.inv(
            sigma_bb
        )  # Shape: [batch_size, n_obs_features, n_obs_features]

        # Calculate the deviation of observed values from their mean
        obs_diff = (X_query - mu_obs).unsqueeze(
            -1
        )  # Shape: [batch_size, n_obs_features, 1]

        # Compute the conditional mean of target features given observed values
        mu_target_given_obs = mu_target + torch.matmul(
            sigma_ab, torch.matmul(inv_sigma_bb, obs_diff)
        ).squeeze(-1)

        # Compute the conditional covariance of target features given observed values
        Sigma_target_given_obs = sigma_aa - torch.matmul(
            sigma_ab, torch.matmul(inv_sigma_bb, sigma_ab.transpose(-1, -2))
        )  # Shape: [batch_size, n_target_features, n_target_features]

        Sigma_target_given_obs = self._ensure_covariance_matrix(Sigma_target_given_obs)

        return mu_target_given_obs, Sigma_target_given_obs

    def _evaluate_Y(self, Y_query, dist, n_samples, batch_size):

        Y_values = self._define_Y_values(Y_query, n_samples, batch_size)

        if Y_values.dim() == 3:
            # Preallocate log_pdf_Y
            log_pdf_Y = torch.zeros_like(
                Y_values, device=self.XY_prior.device
            )  # [batch_size, n_target_features, n_samples]

            for i in range(Y_values.shape[2]):
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
