import pytest
import torch
from matplotlib import pyplot as plt

from vni.estimators.mean_covariance import MeanCovarianceEstimator


class TestMeanCovarianceEstimator:
    @pytest.fixture
    def estimator(self):
        return MeanCovarianceEstimator()

    def test_update_parameters(self, estimator):
        data = torch.randn(
            10, 3, 50, device=estimator.device
        )  # [batch_size, n_features, n_samples]
        mu, sigma = estimator.update_parameters(data)
        assert mu.shape == (10, 3), "Mean shape mismatch."
        assert sigma.shape == (10, 3, 3), "Covariance shape mismatch."
        assert estimator.parameters is not None, "Parameters not updated correctly."

    def test_sample(self, estimator):
        data = torch.randn(10, 3, 50, device=estimator.device)
        estimator.update_parameters(data)
        samples = estimator.sample(5)  # Generate 5 samples
        assert samples.shape == (10, 3, 5), "Sample shape mismatch."

    def test_cpds(self, estimator):
        data = torch.randn(
            10, 5, 50, device=estimator.device
        )  # [batch_size, n_features, n_samples]
        estimator.update_parameters(data)

        # Define conditional inputs
        target_indexes = [0, 1]
        obs_indexes = [2, 3]
        obs_values = torch.randn(10, len(obs_indexes), 1, device=estimator.device)

        cpd_mean, cpd_cov = estimator.cpds(
            target_indexes=target_indexes,
            obs_indexes=obs_indexes,
            obs_values=obs_values,
        )

        assert cpd_mean.shape == torch.Size(
            [len(target_indexes)]
        ), "Conditional mean shape mismatch."
        assert cpd_cov.shape == torch.Size(
            [len(target_indexes), len(target_indexes)]
        ), "Conditional covariance shape mismatch."

    def test_plot_conditional_pdfs(self, estimator):
        mean = torch.tensor([0.5, 1.0, -0.5], device=estimator.device)
        covariance = torch.diag(torch.tensor([0.2, 0.3, 0.1], device=estimator.device))
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
        min_values = [-1.0, 0.0, -2.0]
        max_values = [2.0, 3.0, 1.0]

        fig = estimator.plot_conditional_pdfs(
            mu=mean,
            sigma=covariance,
            feature_names=feature_names,
            min_values=min_values,
            max_values=max_values,
        )

        assert isinstance(fig, plt.Figure), "Plot output is not a matplotlib Figure."
