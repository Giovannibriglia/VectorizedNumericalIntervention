from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch

from vni.base.estimator import BaseEstimator
from vni.non_parametric_estimators.kde import MultivariateGaussianKDE
from vni.parametric_estimators.mean_covariance import MeanCovarianceEstimator


class VNI:
    def __init__(
        self,
        tensor_prior_data: torch.Tensor,
        estimator_config: Dict,
    ):
        self.estimator = self._create_estimator(estimator_config)
        self.estimator.fit(tensor_prior_data)

        self.X_indices = None
        self.X_do_indices = None
        self.Y_indices = None

    def set_indices(
        self,
        obs_indices: List[int],
        target_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        # Ensure Y_indices and intervention_indices are not None
        target_indices = target_indices if target_indices is not None else []
        intervention_indices = (
            intervention_indices if intervention_indices is not None else []
        )

        # Check that X_indices and Y_indices are not overlapping
        if set(obs_indices) & set(target_indices):
            raise ValueError("X_indices and Y_indices must not overlap.")

        # Check that no intervention_indices are in Y_indices
        if set(intervention_indices) & set(target_indices):
            raise ValueError("intervention_indices must not overlap with Y_indices.")

        self.X_indices = obs_indices
        self.X_do_indices = intervention_indices
        self.Y_indices = target_indices

        self.estimator.set_indices(self.X_indices, self.Y_indices, self.X_do_indices)

    def query(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples_x: int = 1024,
        n_samples_y: int = 1024,
    ):
        X_query, Y_query, X_do = self._check_input(X_query, Y_query, X_do)

        return self.estimator.predict(X_query, Y_query, X_do, n_samples_x, n_samples_y)

    @staticmethod
    def _check_input(
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if len(X_query.shape) == 1:
            X_query = X_query.unsqueeze(0)

        if Y_query is not None and len(Y_query.shape) == 1:
            Y_query = Y_query.unsqueeze(0)

        if X_do is not None and len(X_do.shape) == 1:
            X_do = X_do.unsqueeze(0)

        if Y_query is not None:
            assert (
                X_query.shape[0] == Y_query.shape[0]
            ), "X_query and Y_query must have the same batch size"

        if X_do is not None:
            assert (
                X_query.shape[0] == X_do.shape[0]
            ), "X_query and X_do must have the same batch size"

        return X_query, Y_query, X_do

    @staticmethod
    def plot_result(
        pdf: torch.Tensor, y_values: torch.Tensor, true_values: torch.Tensor = None
    ):
        """
        :param pdf: probability density function over Y-values. Shape [batch_size, n_target_features, n_samples]
        :param y_values: evaluated Y-values. Shape [batch_size, n_target_features, n_samples]
        :param true_values: true Y-values. Shape [batch_size, n_target_features]
        :return: None
        """
        pdf = pdf.cpu().numpy()
        y_values = y_values.cpu().numpy()
        true_values = true_values.cpu().numpy() if true_values is not None else None

        batch_index = 0
        target_feature_index = 0

        plt.figure(figsize=(8, 5))
        if pdf.shape[2] > 1:
            pdf1 = pdf[batch_index][target_feature_index]
            y_values1 = y_values[batch_index][target_feature_index]

            if len(np.unique(y_values1)) > 10:  # continuous
                plt.plot(y_values1, pdf1, label="predicted pdf")
            else:  # discrete
                unique_values, indices = np.unique(y_values1, return_inverse=True)
                pdf_aggregated = np.zeros_like(unique_values, dtype=np.float64)

                # Aggregate pdf1 for each unique value
                for i, value in enumerate(unique_values):
                    pdf_aggregated[i] = pdf1[
                        indices == i
                    ].mean()  # Example: average probabilities

                # Plot the bar plot for discrete data
                plt.bar(
                    unique_values,
                    pdf_aggregated,
                    width=0.01,
                    label="predicted pdf",
                )

                plt.xticks(unique_values)

            if true_values is not None:
                true_values1 = true_values[batch_index][target_feature_index]
                plt.scatter(
                    true_values1,
                    np.max(pdf1),
                    c="red",
                    label="ground truth",
                )

            plt.xlim(
                (
                    (
                        np.min(y_values1)
                        if np.max(y_values1) > np.min(y_values1)
                        else np.min(y_values1) - 0.01
                    ),
                    (
                        np.max(y_values1)
                        if np.max(y_values1) > np.min(y_values1)
                        else np.max(y_values1) + 0.01
                    ),
                )
            )
        else:
            plt.scatter(y_values, pdf, label="predicted density values")
            plt.xlim(
                (
                    (
                        np.min(y_values)
                        if np.max(y_values) > np.min(y_values)
                        else np.min(y_values) - 0.01
                    ),
                    (
                        np.max(y_values)
                        if np.max(y_values) > np.min(y_values)
                        else np.max(y_values) + 0.01
                    ),
                )
            )

        plt.xlabel("target feature values")
        # print(np.all(pdf == 0))
        # plt.ylim((-(np.min(pdf) + np.min(pdf) / 10), np.max(pdf) + np.max(pdf) / 10))
        plt.ylabel("PDF")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    @staticmethod
    def _create_estimator(
        config_params: Dict,
    ) -> BaseEstimator:
        # TODO: set estimator config
        if config_params["estimator"] == "multivariate_gaussian_kde":
            return (
                MultivariateGaussianKDE()
            )  # MultivariateGaussianKDE(X_indices, Y_indices, intervention_indices)
        elif config_params["estimator"] == "mean_covariance":
            return (
                MeanCovarianceEstimator()
            )  # MeanCovarianceEstimator(X_indices, Y_indices, intervention_indices)
        else:
            raise ValueError("problem")
