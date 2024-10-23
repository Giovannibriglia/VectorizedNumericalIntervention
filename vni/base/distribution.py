import seaborn as sns
import torch
from matplotlib import pyplot as plt

from vni import DEFAULT_TOLERANCE


class BaseDistribution(object):
    def __init__(self, device):
        self.data = None
        self.central_points = None
        self.variances = None
        self.device = device

    def generate_data_from_distribution(
        self, n_samples: int, start: float, stop: float, **kwargs
    ):
        assert start < stop, "start must be less than stop"

        raise NotImplementedError

    def add_data(self, data: torch.Tensor):
        """
        Adds new data to the existing data. If no existing data is present, initializes the data attribute.

        Args:
            data (torch.Tensor): New data to add.
        """
        if not isinstance(data, torch.Tensor):
            raise ValueError("The input data must be a torch.Tensor")

        if self.data is None:
            # If no data is present, set self.data to the new data
            self.data = data
        else:
            # If data exists, concatenate the new data to the existing data
            self.data = torch.cat((self.data, data), dim=0)

    def set_data(self, data: torch.Tensor):
        """Sets the data, replacing any existing data."""
        self.data = data

    def set_parameters(
        self, central_points: torch.Tensor, variances: torch.Tensor, **kwargs
    ):
        """
        Sets the central points (means) and variances for the distribution.
        If any variances are zero, they will be set to a small positive tolerance (DEFAULT_TOLERANCE).

        Args:
            central_points (torch.Tensor): A tensor containing the central points (means).
            variances (torch.Tensor): A tensor containing the variances.
            kwargs (dict): Additional keyword arguments.

        Raises:
            AssertionError: If central_points and variances do not have the same shape.
        """
        # Assert that central_points and variances have the same shape
        assert central_points.shape == variances.shape, (
            f"central_points and variances must have the same shape, but got "
            f"central_points shape: {central_points.shape}, variances shape: {variances.shape}"
        )

        tolerance = kwargs.pop("tolerance", DEFAULT_TOLERANCE)

        # Replace zero variances with the default tolerance
        variances = torch.where(variances == 0, torch.tensor(tolerance), variances)

        # Set the parameters
        self.central_points = central_points
        self.variances = variances

    @staticmethod
    def plot_distribution(distributions: torch.Tensor, n: int = 0):
        """
        Plots the probability density function (PDF) for the n-th batch of distributions.

        Args:
            distributions (torch.Tensor): A tensor of distributions.
            n (int): Index of the batch to plot. Default is 0.
        """
        # Extract the n-th batch
        first_batch = distributions[n].cpu().numpy()  # Convert to numpy if on GPU

        # Plot the PDF using seaborn
        plt.figure(figsize=(8, 6), dpi=500)
        sns.kdeplot(first_batch, fill=True, color="g", label=f"PDF of Batch {n}")

        # Set x-axis limits from 0 to 1
        plt.xlim(0, 1)

        # Adding labels and title
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        # Show the plot
        plt.show()
