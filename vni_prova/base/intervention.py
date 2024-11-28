from typing import Tuple

import torch
from abc import ABC

from vni.base import DISTRIBUTIONS_MAP


class InterventionalDistribution(ABC):
    def __init__(self, dist_class: type, device: str = None, **dist_kwargs):
        """
        Initialize the Intervention with a distribution class and initial parameters.

        Args:
            dist_class (type): The class of the torch.distribution (e.g., torch.distributions.Normal).
            **dist_kwargs: Initial parameters for the distribution (e.g., loc, scale).
        """
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dist_class = dist_class
        self.dist_kwargs = dist_kwargs
        if dist_kwargs != {}:
            self.dist = dist_class(
                **dist_kwargs
            )  # Instantiate the distribution with initial parameters
        else:
            self.dist = None

    def set_dist_parameters(self, **new_params):
        """
        Update the distribution parameters dynamically.

        Args:
            **new_params: New parameters for the distribution (e.g., loc, scale).
        """
        # Update the parameters and re-instantiate the distribution
        self.dist_kwargs.update(new_params)
        self.dist = self.dist_class(**self.dist_kwargs)

    def get_dist_parameters(self):
        return self.dist_kwargs

    def sample(self, size: torch.Size | Tuple):
        """
        Sample from the current distribution with the updated parameters.

        Args:
            size (torch.Size|Tuple): The shape of the samples to generate. This size should match or be compatible with batch dimensions.

        Returns:
            torch.Tensor: Samples from the distribution.
        """
        if not isinstance(size, torch.Size):
            size = torch.Size(size)

        return self.dist.sample(size)


if __name__ == "__main__":
    # Example with Normal distribution
    intervention = InterventionalDistribution(
        DISTRIBUTIONS_MAP["Normal"], loc=torch.tensor(0.0), scale=torch.tensor(1.0)
    )

    # Update parameters
    intervention.set_dist_parameters(loc=torch.tensor(0.5), scale=torch.tensor(0.2))
    samples = intervention.sample(size=torch.Size([10]))
    print("Normal samples:", samples)

    # **************************************************************************************

    # Example with Bernoulli distribution
    intervention_bernoulli = InterventionalDistribution(
        DISTRIBUTIONS_MAP["Bernoulli"], probs=torch.tensor(0.7)
    )

    # Update parameters
    intervention_bernoulli.set_dist_parameters(probs=torch.tensor(0.9))
    samples = intervention_bernoulli.sample(size=torch.Size([10]))
    print("Bernoulli samples:", samples)

    # **************************************************************************************

    # Batch parameters for Normal distribution
    batch_loc = torch.tensor([0.0, 1.0, 2.0])  # Mean for each batch
    batch_scale = torch.tensor([1.0, 0.5, 0.2])  # Standard deviation for each batch

    # Initialize the Intervention with batch parameters
    intervention = InterventionalDistribution(
        DISTRIBUTIONS_MAP["Normal"], loc=batch_loc, scale=batch_scale
    )

    # Sample from the batch of distributions; the output will have a batch size matching `loc` and `scale`
    samples = intervention.sample(size=torch.Size([5]))  # 5 samples per batch
    print("Normal samples with batch parameters:", samples)
