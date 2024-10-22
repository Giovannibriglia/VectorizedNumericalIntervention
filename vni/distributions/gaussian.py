import torch
from torch.distributions import Normal

from vni.base.distribution import BaseDistribution


class GaussianDistribution(BaseDistribution):
    def generate_data_from_distribution(
        self, n_samples: int, start: float, stop: float, **kwargs
    ):
        """
        Generates a batch of tensors based on normal distributions, each with its own mean and variance
        calculated from the input data.

        Args:
            n_samples (int): Number of samples to generate.
            start (float): The minimum value to clip the samples.
            stop (float): The maximum value to clip the samples.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, n_samples) with sampled values.
        """

        if self.data is None:
            raise KeyError("The required key 'data' is missing.")

        assert isinstance(self.data, torch.Tensor), "data must be torch.Tensor"

        # Compute mean and variance for each batch (assuming shape [batch_size, ...])
        means = (
            self.data.mean(dim=-1)
            if self.central_points is None
            else self.central_points
        )  # Mean for each batch along the last dimension
        variances = (
            self.data.var(dim=-1, unbiased=False)
            if self.central_points is None
            else torch.full((self.data.shape[0],), self.tolerance)
        )  # Variance for each batch along the last dimension

        # Ensure standard deviation (sqrt of variance)
        std_devs = torch.sqrt(variances)

        # Create normal distributions for each batch with their respective mean and std_dev
        normal_dists = Normal(means, std_devs)

        # Sample n_samples for each distribution (batch_size, n_samples)
        samples = normal_dists.sample((n_samples,)).transpose(0, 1)

        # Clip the samples between start and stop to create the truncated normal behavior
        samples = torch.clamp(samples, min=start, max=stop)

        return samples
