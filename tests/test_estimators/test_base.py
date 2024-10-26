import pytest
import torch

from vni.base.estimator import BaseEstimator


class MockEstimator(BaseEstimator):
    def update_parameters(self, data: torch.Tensor):
        self.parameters = data.mean(dim=0)  # Simplified example

    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.randn(n_samples, *self.parameters.shape, device=self.device)

    def cpds(self, **kwargs) -> torch.Tensor:
        return (
            torch.ones_like(self.parameters, device=self.device) * 0.5
        )  # Dummy output


class TestBaseEstimator:
    @pytest.fixture
    def estimator(self):
        return MockEstimator()

    def test_device_initialization(self, estimator):
        assert estimator.device in ["cuda", "cpu"], "Device initialization failed."

    def test_update_parameters(self, estimator):
        data = torch.randn(10, 3, device=estimator.device)
        estimator.update_parameters(data)
        assert estimator.parameters is not None, "Parameters not updated."
        assert estimator.parameters.shape == (3,), "Parameter shape mismatch."

    def test_sample_output_shape(self, estimator):
        data = torch.randn(10, 3, device=estimator.device)
        estimator.update_parameters(data)
        n_samples = 5
        samples = estimator.sample(n_samples)
        assert samples.shape == (n_samples, 3), "Sample output shape mismatch."

    def test_cpds_output_shape(self, estimator):
        data = torch.randn(10, 3, device=estimator.device)
        estimator.update_parameters(data)
        cpds_output = estimator.cpds()
        assert (
            cpds_output.shape == estimator.parameters.shape
        ), "CPDs output shape mismatch."
