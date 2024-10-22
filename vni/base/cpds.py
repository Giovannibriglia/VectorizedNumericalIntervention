from typing import Dict

import torch

from vni.distributions import DELTA_DIRAC, GAUSSIAN
from vni.distributions.dirac_delta import DiracDeltaDistribution
from vni.distributions.gaussian import GaussianDistribution
from vni.estimators.covariance import CovarianceEstimator


class CPDs:
    def __init__(self, batch_size: int, **kwargs):

        self.device = kwargs.pop(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size

        self.data = None
        self.features_indexes_corr = None

        self.estimator = None
        self.estimator_name = kwargs.pop("estimator_name", "covariance")
        self.set_estimator(**kwargs)

        self.data_distribution = None
        self.data_distribution_name = kwargs.pop("distribution_name", GAUSSIAN)
        self.set_data_distribution(**kwargs)

        self.do_distribution = None
        self.do_distribution_name = kwargs.pop("do_distribution_name", DELTA_DIRAC)
        self.set_do_distribution(**kwargs)

    def set_estimator(self, **kwargs):
        if self.estimator_name == "covariance":
            self.estimator = CovarianceEstimator()
        else:
            raise NotImplementedError

    def set_data_distribution(self, **kwargs):
        if self.data_distribution_name == GAUSSIAN:
            self.data_distribution = GaussianDistribution()
        else:
            raise NotImplementedError

    def set_do_distribution(self, **kwargs):
        if self.do_distribution_name == DELTA_DIRAC:
            self.do_distribution = DiracDeltaDistribution()
        else:
            raise NotImplementedError

    def set_data(self, new_data: torch.Tensor):
        """
        Replace the current data with the new data.
        If no existing data is present, initializes the data attribute.
        """
        if self.data is None:
            # Initialize the data if it's not already set
            self.data = new_data
        else:
            assert new_data.size(1) == self.data.size(
                1
            ), f"Data must have the same number of columns as the current tensor: {self.data.size(1)}"
            self.data = new_data

    def add_data(self, new_data: torch.Tensor):
        """
        Add new data to the existing stacked tensor.
        The data must have the same number of features (columns) as the existing tensor.
        If no existing data is present, initializes the data attribute.
        """
        if self.data is None:
            # Initialize the data if it's not already set
            self.data = new_data
        else:
            assert new_data.size(1) == self.data.size(
                1
            ), f"Data must have the same number of columns as the current tensor: {self.data.size(1)}"
            # Concatenate the new data along the first dimension (rows)
            self.data = torch.cat((self.data, new_data), dim=0)

    def set_features_indexes_correspondence(self, correspondence: Dict):
        self.features_indexes_corr = correspondence

    """def query(self, specification: TensorDict, n_samples: int, **kwargs):
        start, stop = 0.0, 1.0

        do_values = specification["do"]["do_values"]
        self.do_distribution.set_central_points(do_values)
        do_distributions = self.do_distribution.generate_data_from_distribution(
            n_samples, start, stop
        )

        obs_indexes = specification["obs"]["obs_indexes"]
        obs_values = specification["obs"]["obs_values"]
        self.data_distribution.set_data(self.data[obs_indexes])
        obs_distributions = self.do_distribution.generate_data_from_distribution(
            n_samples, start, stop
        )

        target_indexes = specification["target"]["target_indexes"]
        gathered_data = self.data[target_indexes]
        data_for_estimators = torch.cat(
            (do_distributions, obs_distributions, gathered_data), dim=1
        )

        final_target_indexes = obs_values.shape[1] + do_values.shape[1]

        # Prepare the input TensorDict with observed and 'do' values
        input_values = torch.cat([obs_values, do_values], dim=1)

        # Set combined distribution to the estimator
        self.estimator.set_data(data_for_estimators)

        # Get joint mean and covariance matrix from the estimator
        mu_joint = self.estimator.get_mean_vector()
        estimator_matrix = self.estimator.get_estimation_matrix()

        # Compute conditional probability distributions (CPDs)
        x_values, conditional_pdf, mu_conditional, std_conditional = self.get_cpds(
            mu_joint, estimator_matrix, n_samples
        )

        # Print the computed conditional mean and standard deviation
        print(f"Conditional Mean: {mu_conditional}")
        print(f"Conditional Std Dev: {std_conditional}")
    """

    """def get_cpds(
        self,
        mu_joint: torch.Tensor,
        estimator_matrix: torch.Tensor,
        input_values: torch.Tensor,
        target_indexes: torch.Tensor,
        n_samples: int,
    ):

        # Partition the estimator matrix and means into components
        Sigma_11 = estimator_matrix[
            :, target_indexes, target_indexes
        ]  # Shape: (batch_size,)
        Sigma_12 = estimator_matrix[
            :, target_indexes, : target_indexes[0]
        ]  # Shape: (batch_size, len(input_indexes))
        Sigma_22 = estimator_matrix[:, : target_indexes[0], :][
            :, :, : target_indexes[0]
        ]  # Shape: (batch_size, len(input_indexes), len(input_indexes))

        # Compute the inverse of Sigma_22 (with batch support)
        Sigma_22_inv = torch.linalg.inv(
            Sigma_22
        )  # Shape: (batch_size, len(input_indexes), len(input_indexes))

        # Compute the conditional mean and covariance of the target variable given the observed values
        observed_values_diff = (
            input_values - mu_joint[:, : target_indexes[0]]
        )  # Shape: (batch_size, len(input_indexes))
        mu_conditional = mu_joint[:, target_indexes] + torch.bmm(
            Sigma_12,
            torch.bmm(Sigma_22_inv, observed_values_diff.unsqueeze(-1)).squeeze(-1),
        )  # Shape: (batch_size,)
        Sigma_conditional = Sigma_11 - torch.bmm(
            Sigma_12, torch.bmm(Sigma_22_inv, Sigma_12.transpose(1, 2))
        ).squeeze(
            -1
        )  # Shape: (batch_size,)
        std_conditional = torch.sqrt(Sigma_conditional)  # Shape: (batch_size,)

        # Generate values for the conditional Gaussian distribution
        x_values = torch.linspace(
            mu_conditional - 3 * std_conditional,
            mu_conditional + 3 * std_conditional,
            n_samples,
            device=estimator_matrix.device,
        )  # Shape: (batch_size, n_samples)

        # Define a batched Normal distribution with mu_conditional and std_conditional
        normal_dist = Normal(
            mu_conditional.unsqueeze(1), std_conditional.unsqueeze(1)
        )  # Shape: (batch_size, 1)

        # Compute the conditional PDF for each batch using the batched Normal distribution
        conditional_pdf = normal_dist.log_prob(
            x_values
        ).exp()  # Shape: (batch_size, n_samples)

        return x_values, conditional_pdf, mu_conditional, std_conditional"""
