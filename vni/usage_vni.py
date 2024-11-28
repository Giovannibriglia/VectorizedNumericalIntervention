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
        Y_indices: List = None,
        intervention_indices: List = None,
    ):
        self.estimator = create_estimator(estimator_config)

        self.intervention_indices = intervention_indices

        # Create a mask to exclude Y_indices from all columns
        all_indices = list(range(XY_prior_tensor.shape[0]))
        X_indices = [i for i in all_indices if i not in Y_indices]

        self.estimator.fit(XY_prior_tensor, X_indices, Y_indices)

        del XY_prior_tensor

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
    def plot_result(pdf: torch.Tensor, y_values: torch.Tensor):
        """
        :param pdf: probability density function over Y-values. Shape [batch_size, n_target_features, n_samples]
        :param y_values: evaluated Y-values. Shape [batch_size, n_target_features, n_samples]
        :return: None
        """
        pdf = pdf.cpu().numpy()
        y_values = y_values.cpu().numpy()

        plt.figure(figsize=(8, 5))
        if pdf.shape[2] > 1:
            pdf1 = pdf[0][0]
            y_values1 = y_values[0][0]

            plt.plot(y_values1, pdf1, label="prediction")
            plt.scatter(
                Y_query[0].cpu().numpy(), np.max(pdf1), c="red", label="ground truth"
            )
        else:
            plt.scatter(y_values, pdf, label="log probability")

        plt.xlabel("target feature values")
        plt.ylabel("PDF")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    target_features = ["agent_0_reward"]

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

    intervention_indices = Y_indices.copy()

    obs_features = [s for s in agent0_columns if s not in target_features]
    X = df.loc[:, agent0_columns]
    Y = df.loc[:, target_features]

    XY_prior_tensor = torch.tensor(df.values, dtype=torch.float32, device="cuda")

    estimator_config = {}

    vni = VNI(XY_prior_tensor.T, estimator_config, Y_indices, intervention_indices)

    batch_size = 9196
    for t in tqdm(range(batch_size, XY_prior_tensor.shape[0], batch_size)):
        # single prediction
        X_query = XY_prior_tensor[t - batch_size : t, X_indices]
        Y_query = XY_prior_tensor[t - batch_size : t, Y_indices]
        pdf, y_values = vni.query(
            X_query, Y_query, n_samples=9196
        )  # [batch_size, n_target_features, n_samples]

        vni.plot_result(pdf, y_values)
