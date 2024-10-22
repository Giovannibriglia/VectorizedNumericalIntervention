import pytest
import torch

from vni.estimators.covariance import (
    CovarianceEstimator,
)  # Adjust the import path to your file structure


class TestCovarianceEstimator:
    def setup_method(self):
        """Setup method to initialize the CovarianceEstimator and test data."""
        self.cov_estimator = CovarianceEstimator()

        # Example batch data: 3 samples with 3 features each
        self.batch_data = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

    def test_add_data(self):
        """Test if the data is added correctly."""
        self.cov_estimator.add_data(self.batch_data)
        assert (
            self.cov_estimator.data is not None
        ), "Data should not be None after adding data."
        assert (
            self.cov_estimator.data.shape == self.batch_data.shape
        ), f"Expected data shape {self.batch_data.shape}, but got {self.cov_estimator.data.shape}"

    def test_mean_vector(self):
        """Test if the mean vector is calculated correctly."""
        self.cov_estimator.add_data(self.batch_data)
        mean_vector = self.cov_estimator.get_mean_vector()
        expected_mean = torch.tensor([4.0, 5.0, 6.0])  # Expected mean for each feature
        assert torch.allclose(
            mean_vector, expected_mean
        ), f"Expected mean vector {expected_mean}, but got {mean_vector}"

    def test_covariance_matrix(self):
        """Test if the covariance matrix is calculated correctly."""
        self.cov_estimator.add_data(self.batch_data)
        covariance_matrix = self.cov_estimator.get_estimation_matrix()

        # Manual calculation of covariance matrix
        expected_covariance = torch.tensor(
            [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]
        )  # Covariance matrix for this specific data
        assert torch.allclose(
            covariance_matrix, expected_covariance
        ), f"Expected covariance matrix {expected_covariance}, but got {covariance_matrix}"

    def test_error_when_no_data(self):
        """Test if an error is raised when trying to calculate mean or covariance with no data."""
        with pytest.raises(ValueError, match="No data available"):
            self.cov_estimator.get_mean_vector()

        with pytest.raises(ValueError, match="No data available"):
            self.cov_estimator.get_estimation_matrix()


if __name__ == "__main__":
    pytest.main()
