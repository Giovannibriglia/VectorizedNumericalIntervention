import pytest
import torch

from vni import DEFAULT_TOLERANCE
from vni.distributions.gaussian import GaussianDistribution


class TestGaussianDistribution:
    @pytest.fixture
    def gaussian_distribution(self):
        """Fixture for initializing the GaussianDistribution class."""
        return GaussianDistribution(device="cpu")

    def test_generate_data_from_distribution(self, gaussian_distribution):
        """Test generating data from the Gaussian distribution with clipping."""
        # Set parameters: means (central points) and standard_deviations
        central_points = torch.tensor([0.4, 0.6, 0.5])
        standard_deviations = torch.tensor([0.01, 0.02, 0.03])

        gaussian_distribution.set_parameters(
            central_points=central_points, standard_deviations=standard_deviations
        )

        # Generate data
        n_samples = 5
        start, stop = 0.0, 1.0
        generated_data = gaussian_distribution.generate_data_from_distribution(
            n_samples=n_samples, start=start, stop=stop
        )

        # Verify shape
        assert generated_data.shape == (len(central_points), n_samples), (
            f"Expected shape: ({len(central_points)}, {n_samples}), "
            f"but got {generated_data.shape}"
        )

        # Verify that all values are within the clipping range
        assert torch.all(generated_data >= start) and torch.all(
            generated_data <= stop
        ), f"Generated data contains values outside the range [{start}, {stop}]"

    def test_generate_data_with_zero_std(self, gaussian_distribution):
        """Test that generating data with zero std replaces the standard_deviations with DEFAULT_TOLERANCE."""
        # Set parameters: means (central points) and zero stds
        central_points = torch.tensor([0.4, 0.6, 0.5])
        standard_deviations = torch.tensor([0.0, 0.0, 0.0])  # Zero standard_deviations
        gaussian_distribution.set_parameters(
            central_points=central_points, standard_deviations=standard_deviations
        )

        # Expected stds should replace zero stds with DEFAULT_TOLERANCE
        expected_stds = torch.tensor(
            [DEFAULT_TOLERANCE, DEFAULT_TOLERANCE, DEFAULT_TOLERANCE]
        )
        assert torch.equal(
            gaussian_distribution.standard_deviations, expected_stds
        ), "Standard deviations were not replaced with DEFAULT_TOLERANCE where they were zero."

        # Generate data
        n_samples = 5
        start, stop = 0.0, 1.0
        generated_data = gaussian_distribution.generate_data_from_distribution(
            n_samples=n_samples, start=start, stop=stop
        )

        # Verify that all values are within the clipping range
        assert torch.all(generated_data >= start) and torch.all(
            generated_data <= stop
        ), f"Generated data contains values outside the range [{start}, {stop}]"

    def test_clipping_of_generated_data(self, gaussian_distribution):
        """Test that generated data is properly clipped between the specified range."""
        # Set parameters: means and standard_deviations
        central_points = torch.tensor(
            [0.9, -0.5, 0.3]
        )  # Values that should trigger clipping
        standard_deviations = torch.tensor([0.01, 0.02, 0.03])
        gaussian_distribution.set_parameters(
            central_points=central_points, standard_deviations=standard_deviations
        )

        # Generate data
        n_samples = 5
        start, stop = 0.0, 1.0
        generated_data = gaussian_distribution.generate_data_from_distribution(
            n_samples=n_samples, start=start, stop=stop
        )

        # Verify that all values are within the clipping range
        assert torch.all(generated_data >= start) and torch.all(
            generated_data <= stop
        ), f"Generated data contains values outside the range [{start}, {stop}]"
