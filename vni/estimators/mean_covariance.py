from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
from torch.distributions import MultivariateNormal

from vni import DEFAULT_TOLERANCE
from vni.base.estimator import BaseEstimator


class MeanCovarianceEstimator(BaseEstimator):
    def update_parameters(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the mean and covariance matrix of the data.

        Args:
            data (torch.Tensor): Input data with shape [batch_size, n_features, n_samples].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean vector and covariance matrix of the data.
        """
        if data is None:
            raise ValueError(
                "No data available. Please add data using `add_data` or `set_data`."
            )

        batch_size, n_features, n_samples = data.shape

        # Compute mean for each batch
        mu = data.mean(dim=2)  # Shape: [batch_size, n_features]

        # Center the data by subtracting the mean from each sample
        centered_data = data - mu.unsqueeze(
            -1
        )  # Shape: [batch_size, n_features, n_samples]

        # Compute covariance using batch matrix multiplication
        # cov_matrix = (X @ X^T) / (n_samples - 1), done batch-wise
        sigma = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (
            n_samples - 1
        )

        # Resulting shapes:
        # mu: [batch_size, n_features]
        # sigma: [batch_size, n_features, n_features]

        self.parameters = (mu.to(self.device), sigma.to(self.device))

        return mu, sigma

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generates samples based on the stored parameters (mean and covariance) for all batches simultaneously.

        Args:
            n_samples (int): Number of samples to generate for each batch.

        Returns:
            torch.Tensor: Generated samples with shape [batch_size, n_features, n_samples].
        """
        # Retrieve the stored mean and covariance
        mu, sigma = self.parameters

        # Create a multivariate normal distribution that handles the entire batch
        distribution = torch.distributions.MultivariateNormal(mu, sigma)

        # Sample n_samples for each distribution in the batch
        samples = distribution.sample(
            (n_samples,)
        )  # Shape: [n_samples, batch_size, n_features]

        # Rearrange the dimensions to get [batch_size, n_features, n_samples]
        samples = samples.permute(1, 2, 0)

        return samples

    def cpds(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the conditional probability density (CPD) of the target features given observed features
        based on the stored mean and covariance parameters.

        Args:
            **kwargs: Arguments containing indices and values for conditional computation.
                - target_indexes: Indices of target features to condition.
                - obs_indexes: Indices of observed features for conditioning.
                - obs_values: Observed values for 'b' features.

        Returns:
            torch.Tensor: A sample from the aggregated MultivariateNormal distribution of shape [batch_size]
        """
        target_indexes = kwargs.pop("target_indexes")
        obs_indexes = kwargs.pop("obs_indexes")
        obs_values = kwargs.pop("obs_values")

        # Retrieve the stored mean and covariance
        mu, sigma = self.parameters

        # Partition the mean vector:
        mu_target = mu[:, target_indexes]  # Mean for target features 'a'
        mu_obs = mu[:, obs_indexes]  # Mean for observed features 'b'

        # Partition the covariance matrix:
        Sigma_aa = sigma[:, target_indexes][:, :, target_indexes]
        Sigma_bb = sigma[:, obs_indexes][:, :, obs_indexes]
        Sigma_ab = sigma[:, target_indexes][:, :, obs_indexes]

        # Add tolerance to Sigma_bb diagonal to prevent singularity
        Sigma_bb = Sigma_bb + DEFAULT_TOLERANCE * torch.eye(
            Sigma_bb.shape[-1], device=Sigma_bb.device
        ).expand_as(Sigma_bb)

        # Compute the inverse of Sigma_bb
        inv_Sigma_bb = torch.linalg.inv(Sigma_bb)

        # Calculate the deviation of observed values from their mean
        obs_diff = obs_values - mu_obs.unsqueeze(2)

        # Compute the conditional mean of target features given observed values
        mu_target_given_obs = (
            mu_target.unsqueeze(2)
            + torch.matmul(Sigma_ab, torch.matmul(inv_Sigma_bb, obs_diff))
        ).squeeze(2)

        # Compute the conditional covariance of target features given observed values
        Sigma_target_given_obs = Sigma_aa - torch.matmul(
            Sigma_ab, torch.matmul(inv_Sigma_bb, Sigma_ab.transpose(1, 2))
        )

        return self._get_average_cpds(mu_target_given_obs, Sigma_target_given_obs)

    @staticmethod
    def _get_average_cpds(
        mu: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute an aggregated MultivariateNormal distribution from batch-level CPDs by averaging the
        means and variances across evaluations and batches.

        Args:
            mu (torch.Tensor): Mean tensor of shape [n_evaluations, n_features_target, batch_size] or [n_evaluations, n_features_target].
            sigma (torch.Tensor): Covariance tensor of shape [n_evaluations, n_features_target, n_features_target].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A sample from the aggregated MultivariateNormal distribution and
            its covariance matrix of shapes [n_features_target] and [n_features_target, n_features_target].
        """

        # Check the dimensionality of mu and adjust mean calculation
        if mu.dim() == 3:
            mean_mu = mu.mean(dim=(0, 2))  # Shape: [n_features_target]
        elif mu.dim() == 2:
            mean_mu = mu.mean(dim=0)  # Shape: [n_features_target]
        else:
            raise ValueError(
                f"Unexpected shape for mu: {mu.shape}. Expected 2 or 3 dimensions."
            )

        # Check dimensionality of sigma and compute mean
        if sigma.dim() == 3:
            mean_sigma = sigma.mean(
                dim=0
            )  # Shape: [n_features_target, n_features_target]
        else:
            raise ValueError(
                f"Unexpected shape for sigma: {sigma.shape}. Expected 3 dimensions."
            )

        # Create the MultivariateNormal distribution with the aligned mean and covariance
        aggregated_cpd = MultivariateNormal(mean_mu, covariance_matrix=mean_sigma)

        # Generate a sample from the aggregated distribution
        sample = aggregated_cpd.sample()  # Sample shape: [n_features_target]

        return sample, mean_sigma

    @staticmethod
    def plot_conditional_pdfs(**kwargs) -> plt.Figure:
        """
        Plot individual PDF line plots for each target feature in separate subplots.

        Args:
        """
        # Ensure data is on CPU for plotting
        mean = kwargs.pop("mu").cpu()  # shape: [n_features_target]
        covariance = kwargs.pop(
            "sigma"
        ).cpu()  # shape: [n_features_target, n_features_target]
        feature_names = kwargs.pop("feature_names")  # shape: [n_features_target]
        min_values = kwargs.pop(
            "min_values"
        )  # Expected to be shape: [n_features_target]
        max_values = kwargs.pop(
            "max_values"
        )  # Expected to be shape: [n_features_target]

        n_features = mean.shape[0]
        fig, axes = plt.subplots(
            1, n_features, figsize=(5 * n_features, 5), squeeze=False
        )

        for i in range(n_features):
            # Extract mean and standard deviation for each feature
            mean_i = mean[i].item()
            std_i = covariance[i, i].sqrt().item()

            # Define x range based on min and max for each feature
            x_range = np.linspace(min_values[i], max_values[i], 1000)
            pdf_values = norm.pdf(x_range, mean_i, std_i)

            # Plot the PDF line for each feature in its subplot
            ax = axes[0, i]
            ax.plot(x_range, pdf_values, label=f"{feature_names[i]}")
            ax.set_title(f"{feature_names[i]} - Mean: {mean_i:.2f}, Std: {std_i:.2f}")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        plt.show()

        return fig
