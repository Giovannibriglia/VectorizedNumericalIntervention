import pytest
import torch

from vni.distributions.dirac_delta import DiracDeltaDistribution


class TestDiracDeltaDistribution:
    def setup_method(self):
        """Setup method to initialize the DiracDeltaDistribution and test data."""
        self.dirac_dist = DiracDeltaDistribution()
        self.n_samples = 10
        self.start = 0.0
        self.stop = 10.0
        self.delta_support_points = torch.tensor(
            [1.0, 5.0, 8.0]
        )  # Example delta support points

    def test_add_data(self):
        """Test if the delta support points are added correctly."""
        self.dirac_dist.set_central_points(self.delta_support_points)
        assert (
            self.dirac_dist.central_points is not None
        ), "Delta support points should not be None after adding."
        assert (
            self.dirac_dist.central_points.shape == self.delta_support_points.shape
        ), f"Expected shape {self.delta_support_points.shape}, but got {self.dirac_dist.central_points.shape}"

    def test_output_size(self):
        """Test if the output size is correct when delta support points are set."""
        self.dirac_dist.set_central_points(self.delta_support_points)
        samples = self.dirac_dist.generate_data_from_distribution(
            self.n_samples, self.start, self.stop
        )
        assert samples.shape == (
            self.delta_support_points.shape[0],
            self.n_samples,
        ), f"Expected shape ({self.delta_support_points.shape[0]}, {self.n_samples}), but got {samples.shape}"

    def test_values_within_bounds(self):
        """Test if the generated samples are within the specified bounds (start, stop)."""
        self.dirac_dist.set_central_points(self.delta_support_points)
        samples = self.dirac_dist.generate_data_from_distribution(
            self.n_samples, self.start, self.stop
        )
        assert torch.all(samples >= self.start) and torch.all(
            samples <= self.stop
        ), "Some samples are out of bounds"

    def test_error_when_support_points_not_set(self):
        """Test if an error is raised when delta support points are not set."""
        with pytest.raises(
            ValueError, match="self.delta_support_points is not defined"
        ):
            self.dirac_dist.generate_data_from_distribution(
                self.n_samples, self.start, self.stop
            )

    def test_tolerance_set_correctly(self):
        """Test if tolerance is set correctly."""
        new_tolerance = 0.01
        self.dirac_dist.set_tolerance(new_tolerance)
        assert (
            self.dirac_dist.tolerance == new_tolerance
        ), f"Expected tolerance {new_tolerance}, but got {self.dirac_dist.tolerance}"


if __name__ == "__main__":
    pytest.main()
