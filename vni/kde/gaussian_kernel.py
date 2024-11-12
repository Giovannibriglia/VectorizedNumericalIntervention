import torch

from vni.base.kde import BaseKDE


class GaussianKDE(BaseKDE):
    def __init__(self, bandwidth=1.0, device="cpu"):
        super().__init__(bandwidth, device)

    def _kernel_function(self, dist_sq):
        """Default Gaussian kernel function."""
        return torch.exp(-0.5 * dist_sq / self.bandwidth**2)
