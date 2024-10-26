import torch
from matplotlib import pyplot as plt

from vni.estimators import MEAN_AND_COVARIANCE
from vni.estimators.mean_covariance import MeanCovarianceEstimator


class Pipeline:
    def __init__(self, **kwargs):
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.data = None

        # Initialize the estimator
        self.estimator = None
        self.estimator_parameters = None
        self.estimator_name = kwargs.pop("estimator_name", MEAN_AND_COVARIANCE)
        self._set_estimator(**kwargs)

        """# Data distribution for modeling the data
        self.obs_distribution = None
        self.obs_distribution_name = kwargs.pop("distribution_name", GAUSSIAN)
        self._set_obs_distribution(**kwargs)

        # Distribution used for interventions
        self.do_distribution = None
        self.do_distribution_name = kwargs.pop("do_distribution_name", GAUSSIAN)
        self._set_do_distribution(**kwargs)"""

        self.features_info = None

    def _set_estimator(self, **kwargs):
        """Set the appropriate estimator (covariance by default)."""
        if self.estimator_name == MEAN_AND_COVARIANCE:
            self.estimator = MeanCovarianceEstimator(**kwargs)
        else:
            raise NotImplementedError

    def get_estimator(self):
        return self.estimator

    """def _set_obs_distribution(self, **kwargs):
        if self.obs_distribution_name == GAUSSIAN:
            self.obs_distribution = MultivariateNormalDistribution()
        else:
            raise NotImplementedError

    def get_obs_distribution(self):
        return self.obs_distribution

    def _set_do_distribution(self, **kwargs):
        if self.do_distribution_name == GAUSSIAN:
            self.do_distribution = MultivariateNormalDistribution()
        else:
            raise NotImplementedError

    def get_do_distribution(self):
        return self.do_distribution"""

    def set_data(self, new_data: torch.Tensor):
        """Replace the current data with new data."""
        if self.data is None:
            self.data = new_data
        else:
            assert new_data.size(1) == self.data.size(
                1
            ), f"Data must have the same number of columns as the current tensor: {self.data.size(1)}"
            self.data = new_data

        self.estimator.update_parameters(self.data)

    def add_data(self, new_data: torch.Tensor):
        """Add new data to the existing data."""
        if self.data is None:
            self.data = new_data
        else:
            assert new_data.size(1) == self.data.size(
                1
            ), f"Data must have the same number of columns as the current tensor: {self.data.size(1)}"
            self.data = torch.cat((self.data, new_data), dim=0)

        self.estimator.update_parameters(self.data)

    def get_cpds(self, **kwargs):
        return self.estimator.cpds(**kwargs)

    def get_samples(self, n_samples: int):
        return self.estimator.sample(n_samples)

    def estimator_plot_contours(self, **kwargs) -> plt.Figure:
        self.estimator.plot_conditional_pdfs(**kwargs)
