import torch
from matplotlib import pyplot as plt


class BaseEstimator(object):
    def __init__(self, **kwargs):
        self.data = None

        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.parameters = None

    def update_parameters(self, data: torch.Tensor):
        raise NotImplementedError

    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError

    def cpds(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def plot_conditional_pdfs(**kwargs) -> plt.Figure:
        """
        Plot contour plots for each feature based on its mean and standard deviation.
        """
        raise NotImplementedError
