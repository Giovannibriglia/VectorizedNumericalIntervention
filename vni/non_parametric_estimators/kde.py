from abc import abstractmethod
from typing import Tuple

import torch

from vni.base.estimator import BaseNonParametricEstimator


class KernelDensityEstimator(BaseNonParametricEstimator):
    def __init__(
        self,
    ):
        super(KernelDensityEstimator, self).__init__()

    def _fit(self):
        pass

    def _predict(
        self, X_values: torch.Tensor, Y_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param X_values: [batch_size, n_features_X+n_features_X_do, n_samples_x]
        :param Y_values: [batch_size, n_features_X+n_features_X_do, 1]
        :return: pdf and Y_values. [batch_size, n_target_features, n_samples_Y_values], [batch_size, n_target_features, n_samples_Y_values]
        """
        batch_size = X_values.shape[0]

        Y_prior = self.XY_prior[self.Y_indices, :].expand(
            batch_size, -1, -1
        )  # Shape: [batch_size, n_target_features, n_samples_data]

        y_kernel = self._compute_kernel_density(
            Y_prior, Y_values
        )  # Shape: [batch_size, n_target_features, n_samples]

        X_prior = self.XY_prior[self.all_X_indices, :].expand(
            batch_size, -1, -1
        )  # Shape: [batch_size, n_X_features+n_X_do_features, n_samples_data]

        x_kernel = self._compute_kernel_density(
            X_prior, X_values
        )  # Shape: [batch_size, n_feat_X+n_feat_X_do, 1]

        joint_density = (x_kernel.sum(dim=1, keepdim=True) * y_kernel) / Y_values.shape[
            2
        ]  # Shape: [batch_size, n_target_features, n_samples_Y_values]

        # Compute marginal KDE
        marginal_density = (x_kernel * X_prior).sum(
            dim=2, keepdim=True
        )  # Shape: [batch_size, n_feat_X+n_feat_X_do, 1]

        marginal_mean = torch.mean(
            marginal_density, dim=1, keepdim=True
        )  # Shape: [batch_size, 1, 1]

        # Avoid division by zero (add a small epsilon)
        epsilon = 1e-8
        marginal_mean = marginal_mean + epsilon

        # Compute the PDF
        pdf = joint_density / marginal_mean

        return (
            pdf,
            Y_values,
        )  # [batch_size, n_target_features, n_samples_Y_values], [batch_size, n_target_features, n_samples_Y_values]

    def _compute_kernel_density(
        self, prior_data: torch.Tensor, query_points: torch.Tensor
    ):
        """

        :param prior_data: prior data. Shape: [batch_size, n_features, n_samples_data]
        :param query_points: query points to evaluate. Shape: [batch_size, n_features, n_samples_points]
        :return: kernel density. Shape: [batch_size, n_features, n_samples_data]
        """

        """Evaluate KDE for the provided points."""
        if prior_data is None:
            raise ValueError("KDE must be fitted with data before evaluation.")

        # Compute difference
        diff = self._compute_diff(
            prior_data, query_points
        )  # [batch_size, n_features, n_samples_data]

        # Compute the kernel values
        kernel_values = self._kernel_function(diff)

        return kernel_values

    @staticmethod
    def _compute_diff(prior_data: torch.Tensor, query_points: torch.Tensor):
        """
        Compute the difference between query points and prior data, reducing to the shape of query points.

        Args:
            query_points: torch.Tensor of shape [batch_size, n_features, n_samples_y], query points.
            prior_data: torch.Tensor of shape [batch_size, n_features, n_samples_data], data points.

        Returns:
            diff: torch.Tensor of shape [batch_size, n_features, n_samples_y], summarized differences.
        """

        # Compute absolute differences between each query point and all prior data points
        differences = torch.abs(
            query_points.unsqueeze(-2) - prior_data.unsqueeze(-1)
        )  # [batch_size, n_features, n_samples_data, n_samples_y]

        # Summarize along the n_samples_data dimension (e.g., take the mean)
        diff = differences.mean(dim=-2)  # [batch_size, n_features, n_samples_y]

        return diff

    @abstractmethod
    def _kernel_function(self, dist_sq):
        raise NotImplementedError


class MultivariateGaussianKDE(KernelDensityEstimator):
    def __init__(
        self,
    ):
        super(MultivariateGaussianKDE, self).__init__()
        self.bandwidth = 0.5
        print("MultivariateGaussianKDE")

    def _kernel_function(self, diff):
        """
        Multivariate Gaussian kernel function (per dimension).

        Args:
            diff: torch.Tensor of shape [batch_size, d, n_samples], differences.

        Returns:
            kernel: torch.Tensor of shape [batch_size, d, n_samples], unnormalized kernel values.
        """
        # Ensure bandwidth has the correct shape
        if isinstance(self.bandwidth, float):
            # Scalar bandwidth applied equally to all dimensions
            bandwidth = torch.full((diff.size(1),), self.bandwidth, device=diff.device)
        else:
            # Use provided bandwidth tensor
            bandwidth = self.bandwidth  # Shape: [d]

        # Normalize by bandwidth
        norm_diff = diff / bandwidth.view(
            1, -1, 1
        )  # Adjust dimensions for broadcasting
        kernel = torch.exp(
            -0.5 * norm_diff.pow(2)
        )  # Gaussian kernel (no product across dimensions)

        d = diff.size(1)  # Dimensionality
        norm_const = self._compute_normalization_constant(d, diff.device)

        # Return normalized kernel values
        return kernel / norm_const

    def _compute_normalization_constant(self, d, device):
        """
        Compute the normalization constant for the multivariate Gaussian kernel.

        Args:
            d: int, dimensionality of the data.
            device: torch.device, device for computation.

        Returns:
            norm_const: Tensor, normalization constant.
        """
        if isinstance(self.bandwidth, float):
            bandwidth = torch.full((d,), self.bandwidth, device=device)
        else:
            bandwidth = self.bandwidth

        # Compute log-scale normalization constant
        log_norm_const = d * torch.log(
            torch.tensor(2 * torch.pi, device=device)
        ) + torch.sum(torch.log(bandwidth))
        return torch.exp(0.5 * log_norm_const)
