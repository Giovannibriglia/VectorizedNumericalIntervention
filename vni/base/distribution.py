import torch

from vni import DEFAULT_TOLERANCE


class BaseDistribution(object):
    def __init__(self):
        self.data = None
        self.central_points = None
        self.tolerance = DEFAULT_TOLERANCE

    def get_distribution_parameters(self):
        raise NotImplementedError

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

    def set_central_points(self, delta_support_points: torch.Tensor):
        self.central_points = delta_support_points

    def set_tolerance(self, tolerance: float):
        self.tolerance = tolerance
