import torch


class BaseEstimator(object):
    def __init__(self):
        self.data = None

    def get_mean_vector(self):
        raise NotImplementedError

    def get_estimation_matrix(self):
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
