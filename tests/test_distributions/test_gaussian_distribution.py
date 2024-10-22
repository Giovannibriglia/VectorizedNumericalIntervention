import pytest
import torch

from vni.distributions.gaussian import GaussianDistribution


class TestGaussianDistribution:
    def setup_method(self):
        """Setup method to initialize the GaussianDistribution and test data."""
        self.gaussian_dist = GaussianDistribution()
        self.n_samples = 10
        self.start = 0.0
        self.stop = 10.0
        self.data = torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])  # Example data

    def test_add_data(self):
        """Test if the data is added correctly."""
        self.gaussian_dist.add_data(self.data)
        assert (
            self.gaussian_dist.data is not None
        ), "Data should not be None after adding data."
        assert (
            self.gaussian_dist.data.shape == self.data.shape
        ), f"Expected data shape {self.data.shape}, but got {self.gaussian_dist.data.shape}"

    def test_output_size(self):
        """Test if the output size is correct."""
        self.gaussian_dist.add_data(self.data)
        samples = self.gaussian_dist.generate_data_from_distribution(
            self.n_samples, self.start, self.stop, data=self.data
        )
        assert samples.shape == (
            self.data.shape[0],
            self.n_samples,
        ), f"Expected shape ({self.data.shape[0]}, {self.n_samples}), but got {samples.shape}"

    def test_values_within_bounds(self):
        """Test if the generated samples are within the specified bounds (start, stop)."""
        self.gaussian_dist.add_data(self.data)
        samples = self.gaussian_dist.generate_data_from_distribution(
            self.n_samples, self.start, self.stop, data=self.data
        )
        assert torch.all(samples >= self.start) and torch.all(
            samples <= self.stop
        ), "Some samples are out of bounds"


if __name__ == "__main__":
    pytest.main()
