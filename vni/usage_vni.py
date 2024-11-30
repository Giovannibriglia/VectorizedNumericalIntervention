from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vni.utils import create_estimator


class VNI:
    def __init__(
        self,
        XY_prior_tensor: torch.Tensor,
        estimator_config: Dict,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        # Ensure Y_indices and intervention_indices are not None
        Y_indices = Y_indices if Y_indices is not None else []
        intervention_indices = (
            intervention_indices if intervention_indices is not None else []
        )

        # 1) Check that intervention_indices are in X_indices
        if not set(intervention_indices).issubset(X_indices):
            raise ValueError("All intervention_indices must be a subset of X_indices.")

        # 2) Check that X_indices and Y_indices are not overlapping
        if set(X_indices) & set(Y_indices):
            raise ValueError("X_indices and Y_indices must not overlap.")

        # 3) Check that no intervention_indices are in Y_indices
        if set(intervention_indices) & set(Y_indices):
            raise ValueError("intervention_indices must not overlap with Y_indices.")

        # Set attributes and initialize the estimator
        self.intervention_indices = intervention_indices
        self.estimator = create_estimator(
            estimator_config, X_indices, Y_indices, intervention_indices
        )
        self.estimator.fit(XY_prior_tensor)

    def query(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = None,
    ):
        X_query, Y_query, X_do = self._check_input(X_query, Y_query, X_do)

        return self.estimator.predict(X_query, Y_query, X_do, n_samples)

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
        pdf: torch.Tensor, y_values: torch.Tensor, true_values: torch.Tensor
    ):
        """
        :param pdf: probability density function over Y-values. Shape [batch_size, n_target_features, n_samples]
        :param y_values: evaluated Y-values. Shape [batch_size, n_target_features, n_samples]
        :param true_values: true Y-values. Shape [batch_size, n_target_features]
        :return: None
        """
        pdf = pdf.cpu().numpy()
        y_values = y_values.cpu().numpy()
        true_values = true_values.cpu().numpy()

        batch_index = 0
        target_feature_index = 0

        plt.figure(figsize=(8, 5))
        if pdf.shape[2] > 1:
            pdf1 = pdf[batch_index][target_feature_index]
            y_values1 = y_values[batch_index][target_feature_index]
            true_values1 = true_values[batch_index][target_feature_index]

            plt.plot(y_values1, pdf1, label="predicted density function")
            plt.scatter(
                true_values1,
                np.max(pdf1),
                c="red",
                label="ground truth",
            )
        else:
            plt.scatter(y_values, pdf, label="predicted density value")

        plt.xlabel("target feature values")
        plt.ylabel("PDF")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    target_features = ["agent_0_action_0"]
    intervention_features = ["agent_0_reward"]

    df = pd.read_pickle("../data/df_navigation_pomdp_discrete_actions_0.pkl")
    agent0_columns = [col for col in df.columns if "agent_0" in col]
    df = df.loc[:, agent0_columns]

    Y_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col in target_features
    ]
    X_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col not in target_features
    ]
    intervention_indices = [
        df.columns.get_loc(col)
        for col in df.columns.to_list()
        if col in intervention_features
    ]

    obs_features = [s for s in agent0_columns if s not in target_features]
    X = df.loc[:, agent0_columns]
    Y = df.loc[:, target_features]

    XY_prior_tensor = torch.tensor(df.values, dtype=torch.float32, device="cuda")

    estimator_config = {}

    vni = VNI(
        XY_prior_tensor.T, estimator_config, X_indices, Y_indices, intervention_indices
    )

    batch_size = 1
    n_samples = 1024
    for t in tqdm(range(batch_size, XY_prior_tensor.shape[0], batch_size)):

        true_values = XY_prior_tensor[t - batch_size : t, Y_indices]

        X_query = XY_prior_tensor[t - batch_size : t, X_indices]
        Y_query = None  # true_values.clone()
        X_do = XY_prior_tensor[t - batch_size : t, intervention_indices]

        pdf, y_values = vni.query(
            X_query,
            Y_query=Y_query,
            X_do=X_do,
            n_samples=n_samples,
        )  # [batch_size, n_target_features, n_samples]

        vni.plot_result(pdf, y_values, true_values)
