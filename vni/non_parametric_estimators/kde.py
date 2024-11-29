from abc import abstractmethod
from typing import List, Tuple

import torch

from vni.base.estimator import BaseNonParametricEstimator


class KernelDensityEstimator(BaseNonParametricEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super(KernelDensityEstimator, self).__init__(
            X_indices, Y_indices, intervention_indices
        )
        self.bandwidth = 0.5

    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict conditional density using KDE.

        Args:
            X_query (torch.Tensor): Query for X with shape [batch_size, n_features_X].
            Y_query (torch.Tensor, optional): Query for Y with shape [batch_size, n_features_Y]. Defaults to None.
            X_do (torch.Tensor, optional): Interventional values for X with shape [batch_size, n_features_X_do]. Defaults to None.
            n_samples (int, optional): Number of samples for marginalization. Defaults to 1000.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Conditional PDF and sampled Y_values.
        """
        batch_size = X_query.shape[0]

        # Extract prior samples for the observed features (X).
        X_prior = self.XY_prior[self.X_indices, :].T  # [n_samples_data, n_features_X]

        # Define or sample Y values based on the query.
        Y_values = self._define_Y_values(
            Y_query, n_samples, batch_size
        )  # Output: [batch_size, n_target_features, n_samples] if Y_query is None, else [batch_size, n_target_features, 1]

        Y_values = Y_values.squeeze(-1)  # [batch_size, n_target_features]

        joint_points = torch.zeros(
            (batch_size, X_query.shape[1] + Y_query.shape[1]),
            device=X_query.device,
        )

        joint_points[:, self.X_indices] = X_query
        joint_points[:, self.Y_indices] = Y_query

        marginal_points = X_query.clone()

        # Compute conditional PDF
        pdf = self._evaluate_Y(self.XY_prior.T, joint_points, X_prior, marginal_points)

        return pdf, Y_values

    def _compute_density(self, data: torch.Tensor, points: torch.Tensor):
        """Evaluate KDE for the provided points."""
        if data is None:
            raise ValueError("KDE must be fitted with data before evaluation.")

        diff = points[:, None, :] - data[None, :, :]
        dist_sq = torch.sum(diff.mul(diff), dim=-1)
        kernel_vals = self._kernel_function(dist_sq)
        return kernel_vals.mean(dim=1) / (
            self.bandwidth * (2 * torch.pi) ** (points.shape[1] / 2)
        )

    @abstractmethod
    def _kernel_function(self, dist_sq):
        raise NotImplementedError

    def _evaluate_Y(self, XY, joint_points, X, marginal_points):
        """
        Evaluate the conditional density P(Y | X = x) using KDE.

        Args:
            XY (torch.Tensor): Prior samples for joint density evaluation.
            joint_points (torch.Tensor): Query points for joint density P(X, Y).
            X (torch.Tensor): Prior samples for marginal density evaluation.
            marginal_points (torch.Tensor): Query points for marginal density P(X).

        Returns:
            torch.Tensor: Conditional PDF with shape [batch_size, n_samples_Y].
        """
        # Evaluate the joint density P(X, Y)
        joint_density = self._compute_density(
            XY, joint_points
        )  # Shape: [batch_size, n_target_features, n_samples_Y]

        # Evaluate the marginal density P(X = x)
        marginal_density = torch.mean(
            self._compute_density(X, marginal_points)
        )  # Shape: [batch_size]

        # Add a small epsilon for numerical stability
        epsilon = 1e-8
        # Compute the conditional density P(Y | X = x) = P(X, Y) / P(X)
        conditional_pdf = joint_density / (
            marginal_density + epsilon
        )  # Shape: [batch_size, n_samples_Y]
        conditional_pdf = conditional_pdf.unsqueeze(-1)

        return conditional_pdf


class GaussianKDE(KernelDensityEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super(GaussianKDE, self).__init__(X_indices, Y_indices, intervention_indices)
        self.bandwidth = 0.5

    def _kernel_function(self, dist_sq):
        """Default Gaussian kernel function."""
        return torch.exp(-0.5 * dist_sq / self.bandwidth**2)
