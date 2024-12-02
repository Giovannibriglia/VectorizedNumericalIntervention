from abc import abstractmethod
from typing import List, Tuple

import torch

from torch.distributions import Normal

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

        batch_size = X_query.shape[0]

        Y_prior = self.XY_prior[self.Y_indices, :].expand(
            batch_size, -1, -1
        )  # Shape: [batch_size, n_target_features, n_samples_data]

        # Define or sample Y values based on the query.
        Y_values = self._define_Y_values(
            Y_query, n_samples, batch_size
        )  # Shape: [batch_size, n_target_features, n_samples] if Y_query is None, else [batch_size, n_target_features, 1]

        y_kernel = self._compute_kernel_density(
            Y_prior, Y_values
        )  # Shape: [batch_size, n_target_features, n_samples]

        new_X_prior, new_X_query = self._define_X(
            X_query, X_do
        )  # [batch_size, n_feat_X+n_feat_X_do, n_samples_data]. Shape: [batch_size, n_feat_X+n_feat_X_do, 1]

        x_kernel = self._compute_kernel_density(
            new_X_prior, new_X_query
        )  # Shape: [batch_size, n_feat_X+n_feat_X_do, 1]

        joint_density = (x_kernel.sum(dim=1, keepdim=True) * y_kernel) / Y_values.shape[
            2
        ]  # Shape: [batch_size, n_target_features, n_samples_Y_values]

        # Compute marginal KDE
        marginal_density = (x_kernel * new_X_prior).sum(
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

        self._check_output(pdf, Y_values, Y_query, batch_size, n_samples)

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

    def _define_X(self, X_query: torch.Tensor, X_do: torch.Tensor):
        """

        :param X_query: observation. Shape: [batch_size, n_features_X, 1]
        :param X_do: intervention. Shape: [batch_size, n_features_X_do, 1]
        :return: X_prior and X_points. Shape: [batch_size, n_features_X+n_features_X_do, n_samples_data]. Shape: [batch_size, n_features_X+n_features_X_do, 1]
        """

        batch_size = X_query.shape[0]

        if X_do is None:
            X_do_data = None
            X_do_query = None
        else:
            X_do_data, X_do_query = self._define_X_do(
                X_do
            )  # Shape: X_do_prior: [batch_size, n_features_X_do, n_samples_data] or None. Shape X_do_samples: [batch_size, n_features_X_do, 1] if X_do else None

        X_prior = self.XY_prior[self.X_indices, :].expand(
            batch_size, -1, -1
        )  # [batch_size, n_features_X, n_samples_data]

        if X_do_data is None or X_do_query is None:
            return X_prior, X_query.unsqueeze(-1)
        else:
            X_concat = torch.cat(
                (X_prior, X_do_data), dim=1
            )  # [batch_size, n_features_X+n_features_X_do, n_samples_data]
            X_query_concat = torch.cat(
                (X_query.unsqueeze(-1), X_do_query), dim=1
            )  # [batch_size, n_features_X+n_features_X_do, 1]

            return X_concat, X_query_concat

    def _define_X_do(self, X_do: torch.Tensor):
        """
        :param X_do: intervention. Shape: [batch_size, n_features, n_samples_data]
        :return: X_do_data, X_do_query. # Shape: [batch_size, n_features_X_do, n_samples_data] if X_do else None. Shape: [batch_size, n_features, 1] if X_do else None
        """
        batch_size = X_do.shape[0]

        if self.intervention_indices is None:
            if X_do is None:
                X_do_data = None
                X_do_query = None
            else:
                raise ValueError(
                    "intervention indices are not initialized, make sure they are"
                )
        else:
            X_do_query = X_do.clone().unsqueeze(-1)  # [batch_size, n_features_X_do, 1]

            if X_do is None:
                X_do_data = self.XY_prior_intervention[
                    self.intervention_indices, :
                ].expand(
                    batch_size, -1, -1
                )  # Shape: [batch_size, n_features_X_do, n_samples_data]

            else:
                n_samples_data = self.XY_prior.shape[1]
                tolerance = torch.full(X_do.shape, 1e-8, device=X_do.device)
                X_do_data = (
                    Normal(X_do, tolerance).sample((n_samples_data,)).permute(1, 2, 0)
                )  # Shape: [batch_size, n_features_X_do, n_samples_data]

        return X_do_data, X_do_query


class MultivariateGaussianKDE(KernelDensityEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super(MultivariateGaussianKDE, self).__init__(
            X_indices, Y_indices, intervention_indices
        )
        self.bandwidth = 0.5

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
