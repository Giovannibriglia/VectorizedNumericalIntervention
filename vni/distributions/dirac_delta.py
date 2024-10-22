import torch
from torch.distributions import Normal

from vni.base.distribution import BaseDistribution


class DiracDeltaDistribution(BaseDistribution):
    def generate_data_from_distribution(
        self, n_samples: int, start: float, stop: float, **kwargs
    ):
        """
        Generates a batch of tensors approximating delta Dirac functions, each with its own center,
        defined by the input centers tensor, with a specified tolerance, using a truncated normal distribution.

        Args:
            n_samples (int): Number of samples to generate for each delta Dirac.
            start (float): The minimum value to clip the samples.
            stop (float): The maximum value to clip the samples.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, n_samples) with delta Dirac approximations.
        """
        if self.central_points is None:
            raise ValueError(
                "self.delta_support_points is not defined; use the specific method `set_delta_support_point` "
            )

        # Create normal distributions for each batch with their respective mean and std_dev
        normal_dists = Normal(self.central_points, self.tolerance)

        # Sample n_samples for each distribution (batch_size, n_samples)
        samples = normal_dists.sample((n_samples,)).transpose(0, 1)

        # Clip the samples between start and stop to create the truncated normal behavior
        samples = torch.clamp(samples, min=start, max=stop)

        return samples
