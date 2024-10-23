import matplotlib.pyplot as plt
import pytest
import torch

from vni import DEFAULT_TOLERANCE
from vni.base.distribution import BaseDistribution


class TestBaseDistribution:
    @pytest.fixture
    def base_distribution(self):
        """Fixture for initializing the BaseDistribution class."""
        return BaseDistribution(device="cpu")

    def test_set_parameters(self, base_distribution):
        """Test if set_parameters sets the central_points and variances correctly, and replaces zero variances with DEFAULT_TOLERANCE."""
        central_points = torch.tensor([0.5, 0.4, 0.3])
        variances = torch.tensor([0.02, 0.0, 0.01])  # One zero variance

        base_distribution.set_parameters(
            central_points=central_points, standard_deviations=variances
        )

        # Expected variances should replace the zero variance with DEFAULT_TOLERANCE
        expected_variances = torch.tensor([0.02, DEFAULT_TOLERANCE, 0.01])

        assert torch.equal(
            base_distribution.central_points, central_points
        ), "Central points were not set correctly."
        assert torch.equal(
            base_distribution.standard_deviations, expected_variances
        ), "Variances were not set correctly."

    def test_set_parameters_shape_mismatch(self, base_distribution):
        """Test that an AssertionError is raised when central_points and variances have different shapes."""
        central_points = torch.tensor([0.5, 0.4])
        variances = torch.tensor([0.02, 0.03, 0.01])

        with pytest.raises(AssertionError):
            base_distribution.set_parameters(
                central_points=central_points, standard_deviations=variances
            )

    def test_add_data(self, base_distribution):
        """Test the addition of new data to the distribution."""
        data1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        data2 = torch.tensor([[0.5, 0.6], [0.7, 0.8]])

        base_distribution.add_data(data1)
        assert torch.equal(
            base_distribution.data, data1
        ), "Initial data was not set correctly."

        base_distribution.add_data(data2)
        expected_data = torch.cat((data1, data2), dim=0)
        assert torch.equal(
            base_distribution.data, expected_data
        ), "Data was not concatenated correctly."

    def test_add_data_invalid_type(self, base_distribution):
        """Test that a ValueError is raised when non-tensor data is passed to add_data."""
        with pytest.raises(ValueError):
            base_distribution.add_data([0.1, 0.2])

    def test_set_data(self, base_distribution):
        """Test setting the data directly with set_data."""
        data = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
        base_distribution.set_data(data)
        assert torch.equal(base_distribution.data, data), "Data was not set correctly."

    def test_plot_distribution(self, base_distribution):
        """Test the plot_distribution function."""
        # Generate sample data
        # Broadcast the mean and std to have the same shape (N, n_samples)
        mean = torch.tensor([0.5, 0.3], device="cpu")
        std = torch.tensor([0.05, 0.02], device="cpu")
        n_samples = 1000

        mean_expanded = mean.unsqueeze(1).expand(-1, n_samples).to("cpu")
        std_expanded = std.unsqueeze(1).expand(-1, n_samples).to("cpu")

        # Generate normal distributions for each mean and std
        sample_data = torch.normal(mean=mean_expanded, std=std_expanded).to("cpu")
        # Plot for batch 0
        BaseDistribution.plot_distribution(sample_data, n=0)
        plt.close()  # Close the plot after the test to avoid display issues
