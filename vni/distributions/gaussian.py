import torch

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

        assert (
            self.central_points.shape == self.standard_deviations.shape
        ), "Mean and std tensors must have the same shape."

        normal_dists = torch.distributions.Normal(
            self.central_points.unsqueeze(1), self.standard_deviations.unsqueeze(1)
        )

        # Sample from the distributions (shape will be (batch_size, n_samples))
        samples = normal_dists.sample((n_samples,)).transpose(
            0, 1
        )  # Transpose to get shape (batch_size, n_samples)

        # Clip the samples between start and stop
        clipped_samples = torch.clamp(samples, min=start, max=stop)

        # Remove the last dimension if it's 1 (e.g., from shape (batch_size, n_samples, 1) to (batch_size, n_samples))
        clipped_samples = clipped_samples.squeeze(-1)  # Squeeze the last dimension

        return clipped_samples
