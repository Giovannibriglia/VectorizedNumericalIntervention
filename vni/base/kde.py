from abc import ABC

import torch


class BaseKDE(ABC):
    def __init__(self, bandwidth=1.0, device="cpu"):
        self.bandwidth = bandwidth
        self.device = device
        self.data = None

    def evaluate(self, data: torch.Tensor, points: torch.Tensor):
        """Evaluate KDE for the provided points."""
        if data is None:
            raise ValueError("KDE must be fitted with data before evaluation.")
        diff = points[:, None, :] - data[None, :, :]
        dist_sq = torch.sum(diff.mul(diff), dim=-1)
        kernel_vals = self._kernel_function(dist_sq)
        return kernel_vals.mean(dim=1) / (
            self.bandwidth * (2 * torch.pi) ** (points.shape[1] / 2)
        )

    def _kernel_function(self, dist_sq):
        raise NotImplementedError

    def sample(self, data: torch.Tensor, n_samples: int):
        raise NotImplementedError
