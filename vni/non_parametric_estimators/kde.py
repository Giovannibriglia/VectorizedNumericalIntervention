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

        """if X_do is None:
            if self.intervention_indices is None:
                X_prior = self.XY_prior[
                    self.X_indices, :
                ].T  # [n_samples_data, n_features_X]
                Y_prior = self.XY_prior[
                    self.Y_indices, :
                ].T  # [n_samples_data, n_features_Y]
            else:
                X_prior = self.XY_prior_intervention[
                    self.X_indices, :
                ].T  # [n_samples_data, n_features_X]
                Y_prior = self.XY_prior_intervention[
                    self.Y_indices, :
                ].T  # [n_samples_data, n_features_Y]
        else:
            if self.intervention_indices is None:
                raise ValueError(
                    "intervention indices are not initialized, make sure they are"
                )
            else:
                Y_prior = self.XY_prior_intervention[
                    self.Y_indices, :
                ].T  # [n_samples_data, n_features_Y]

                X_prior = (
                    self.XY_prior[self.X_indices, :]
                    .clone()
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                )  # [batch_size, n_features_X, n_samples_data]

                tolerance = torch.full_like(X_do, 1e-8, device=X_do.device)

                normal_dist = Normal(X_do, tolerance)
                new_samples = normal_dist.sample((self.XY_prior.shape[1],))
                print(new_samples.shape, X_prior.shape)
                X_prior[:, self.intervention_indices, :] = new_samples
                print(X_prior.shape)"""

        """ ******************************************************************************++ """
        """ # Extract prior samples for the observed features (X).
        X_prior = self.XY_prior[self.X_indices, :].T  # [n_samples_data, n_features_X]
        Y_prior = self.XY_prior[self.Y_indices, :].T  # [n_samples_data, n_features_Y]"""
        """ ******************************************************************************++ """

        Y_prior = self.XY_prior[self.Y_indices, :].T  # [n_samples_data, n_features_Y]

        # Define or sample Y values based on the query.
        Y_values = self._define_Y_values(
            Y_query, n_samples, batch_size
        )  # Output: [batch_size, n_target_features, n_samples] if Y_query is None, else [batch_size, n_target_features, 1]

        pdf = torch.zeros_like(Y_values, device=Y_values.device)

        """ SPOSTARE QUESTO CICLO FOR (BATCH) DENTRO, SOLO PER CALCOLARE LE X_DATA E X_POINTS"""
        for batch_index in range(Y_values.shape[0]):
            for feature_idx in range(Y_values.shape[1]):
                for value in range(Y_values.shape[2]):

                    X_prior = self.XY_prior[
                        self.X_indices, :
                    ].T  # [n_samples_data, n_features_X]
                    X_do_prior = None

                    if X_do is None:
                        if self.intervention_indices is not None:
                            X_do_prior = self.XY_prior_intervention[
                                self.intervention_indices, :
                            ].T  # [n_samples_data, n_features_X_do]
                    else:
                        if self.intervention_indices is None:
                            raise ValueError(
                                "intervention indices are not initialized, make sure they are"
                            )
                        else:

                            X_do_single_batch = X_do[batch_index, :]

                            tolerance = torch.full_like(
                                X_do_single_batch, 1e-8, device=X_do_single_batch.device
                            )  # [n_features_X_do]
                            normal_dist = Normal(X_do_single_batch, tolerance)

                            X_do_prior = normal_dist.sample(
                                (self.XY_prior.shape[1],)
                            )  # [n_samples_data, n_features_X_do]

                    if X_do_prior is None:
                        x_kernel = self._compute_density(
                            X_prior, X_query
                        )  # [batch_size, n_samples_data]

                        marginal_density = x_kernel.sum(dim=-1) / X_prior.size(
                            0
                        )  # [batch_size]
                    else:
                        X_query_single_batch = X_query[batch_index, :]  # [n_features_X]
                        X_do_single_batch = X_do[batch_index, :]  # [n_features_X_do]

                        X_prior_and_do_concat = torch.cat(
                            (X_prior, X_do_prior), dim=1
                        )  # Shape: [n_sample_data, n_features_X + n_features_X_do]
                        X_query_concat = torch.cat(
                            (X_query_single_batch, X_do_single_batch), dim=0
                        ).unsqueeze(
                            0
                        )  # Shape: [1, n_X + n_X_do]

                    x_kernel = self._compute_density(
                        X_prior_and_do_concat, X_query_concat
                    )  # [batch_size, n_samples_data]

                    marginal_density = x_kernel.sum(
                        dim=-1
                    ) / X_prior_and_do_concat.size(
                        0
                    )  # [batch_size]

                    y_query_feature = Y_values[:, feature_idx, value].unsqueeze(
                        -1
                    )  # [batch_size, 1]
                    y_samples_feature = Y_prior[:, feature_idx].unsqueeze(
                        -1
                    )  # [n_samples_data, 1]

                    y_kernel = self._compute_density(
                        y_samples_feature, y_query_feature
                    )  # [batch_size, n_samples_data]

                    # Compute joint KDE
                    joint_density = (x_kernel * y_kernel).sum(dim=-1) / Y_values[
                        :, feature_idx, :
                    ].size(
                        0
                    )  # [batch_size]

                    # Compute conditional P(X | Y = y)
                    cpd_feature = joint_density / (
                        marginal_density + 1e-8
                    )  # Avoid division by zero [batch_size]

                    pdf[:, feature_idx, value] = cpd_feature

        self._check_output(pdf, Y_values, Y_query, batch_size, n_samples)

        return (
            pdf,
            Y_values,
        )  # [batch_size, n_target_features, n_samples_Y_values], [batch_size, n_target_features, n_samples_Y_values]

    def _compute_density(self, data: torch.Tensor, points: torch.Tensor):
        """Evaluate KDE for the provided points."""
        if data is None:
            raise ValueError("KDE must be fitted with data before evaluation.")

        # Compute difference
        diff = self._compute_diff(points, data)

        # Compute the kernel values
        kernel_values = self._kernel_function(diff)

        return kernel_values

    @staticmethod
    def _compute_diff(x, y):
        """
        Compute the difference between two tensors in a universal way.

        Args:
            x: torch.Tensor of shape [n_queries, d], query points.
            y: torch.Tensor of shape [n_samples, d], data points.

        Returns:
            diff: torch.Tensor of shape [n_queries, n_samples, d], differences.
        """

        return x[:, None, :] - y[None, :, :]  # Shape: [n_queries, n_samples, d]

    def _apply_kernel(self, x, y):
        """
        General function to apply a kernel function between two tensors.

        Args:
            x: torch.Tensor of shape [n_queries, d], query points.
            y: torch.Tensor of shape [n_samples, d], data points.
            kernel_function: Callable, kernel function to compute values.
            **kwargs: Additional arguments to pass to the kernel function.

        Returns:
            kernel_values: torch.Tensor of shape [n_queries, n_samples], kernel values.
        """
        diff = self._compute_diff(x, y)
        return self._kernel_function(diff)

    @abstractmethod
    def _kernel_function(self, dist_sq):
        raise NotImplementedError


class GaussianKDE(KernelDensityEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super(GaussianKDE, self).__init__(X_indices, Y_indices, intervention_indices)
        self.bandwidth = 0.5

    def _kernel_function(self, diff):
        """ "
        Multivariate Gaussian kernel function.

        Args:
            diff: torch.Tensor of shape [n_queries, n_samples, d], differences.

        Returns:
            kernel: torch.Tensor of shape [n_queries, n_samples], unnormalized kernel values.
        """
        # Ensure bandwidth has the correct shape
        if isinstance(self.bandwidth, float):
            bandwidth = torch.full((diff.size(-1),), self.bandwidth, device=diff.device)
        else:
            bandwidth = self.bandwidth

        # Normalize by bandwidth
        norm_diff = diff / bandwidth  # Shape: [n_queries, n_samples, d]
        kernel = torch.exp(-0.5 * norm_diff.pow(2)).prod(
            dim=-1
        )  # Gaussian kernel and product across dimensions

        d = diff.shape[2]
        norm_const = self._compute_normalization_constant(d, diff.device)

        return kernel / norm_const

    def _compute_normalization_constant(self, d, device):
        """
        Compute the normalization constant for the multivariate Gaussian kernel.

        Args:
            d: int, dimensionality of the data.
            device: torch.device, device for computation.

        Returns:
            norm_const: float, normalization constant.
        """
        if isinstance(self.bandwidth, float):
            bandwidth = torch.full((d,), self.bandwidth, device=device)
        else:
            bandwidth = self.bandwidth
        return (
            torch.sqrt(torch.tensor((2 * torch.pi) ** d, device=device))
            * bandwidth.prod()
        )
