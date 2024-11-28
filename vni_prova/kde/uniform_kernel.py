from vni.base.kde import BaseKDE


class UniformKDE(BaseKDE):
    def __init__(self, bandwidth=1.0, device="cpu"):
        super().__init__(bandwidth, device)

    def _kernel_function(self, dist_sq):
        """Uniform kernel."""
        return (dist_sq < self.bandwidth**2).float()
